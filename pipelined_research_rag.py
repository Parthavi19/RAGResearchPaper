import os
import google.generativeai as genai
import numpy as np
import PyPDF2
import re
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import time
import google.api_core.exceptions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import sys
import asyncio
import concurrent.futures
from queue import Queue, Empty
from threading import Thread, Event
import multiprocessing as mp
from functools import partial

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
_embedding_model = None
_tokenizer = None
_model_loading_error = None

def get_embedding_model():
    """Lazy load embedding model to avoid startup delays."""
    global _embedding_model, _tokenizer, _model_loading_error
    
    if _model_loading_error:
        logger.error(f"Embedding model loading previously failed: {_model_loading_error}")
        raise _model_loading_error
    
    if _embedding_model is None:
        try:
            logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
            start_time = time.time()
            _tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            _embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            _embedding_model.eval()
            logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"Failed to load all-MiniLM-L6-v2: {e}. Falling back to bert-base-uncased.")
            try:
                start_time = time.time()
                _tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                _embedding_model = AutoModel.from_pretrained('bert-base-uncased')
                _embedding_model.eval()
                logger.info(f"Fallback embedding model loaded in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to load any embedding model: {e}")
                _model_loading_error = Exception(f"Could not load embedding model: {e}")
                raise _model_loading_error
    
    return _embedding_model, _tokenizer

@dataclass
class ResearchChunk:
    text: str
    page_number: int
    section: str
    chunk_id: str
    embedding: np.ndarray = None

@dataclass
class PipelineData:
    """Data structure for pipeline stages"""
    pdf_path: str = None
    pages_text: List[Dict] = None
    chunks: List[ResearchChunk] = None
    embeddings: np.ndarray = None
    metadata: Dict = None
    stage: str = "init"
    error: Exception = None

class PipelineStage:
    """Base class for pipeline stages"""
    def __init__(self, name: str, input_queue: Queue, output_queue: Queue):
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = Event()
        self.thread = None
        
    def start(self):
        """Start the pipeline stage thread"""
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the pipeline stage"""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            
    def _run(self):
        """Main execution loop for the stage"""
        while not self.stop_event.is_set():
            try:
                # Get data from input queue with timeout
                data = self.input_queue.get(timeout=1.0)
                if data is None:  # Poison pill
                    break
                    
                # Process the data
                processed_data = self.process(data)
                
                # Send to output queue
                self.output_queue.put(processed_data)
                self.input_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}", exc_info=True)
                # Send error data downstream
                error_data = PipelineData(error=e, stage=self.name)
                self.output_queue.put(error_data)
                
    def process(self, data: PipelineData) -> PipelineData:
        """Override this method in subclasses"""
        raise NotImplementedError

class PDFExtractionStage(PipelineStage):
    """Stage 1: Extract text from PDF"""
    
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__("PDF_Extraction", input_queue, output_queue)
        
    def process(self, data: PipelineData) -> PipelineData:
        try:
            logger.info(f"Extracting text from PDF: {data.pdf_path}")
            pages_text = []
            
            with open(data.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        cleaned_text = self.clean_text(text)
                        section = self._infer_section(cleaned_text, page_num)
                        pages_text.append({
                            'text': cleaned_text,
                            'page_number': page_num + 1,
                            'section': section
                        })
            
            data.pages_text = pages_text
            data.stage = "pdf_extracted"
            logger.info(f"Extracted {len(pages_text)} pages")
            return data
            
        except Exception as e:
            logger.error(f"Error in PDF extraction: {e}", exc_info=True)
            data.error = e
            data.stage = "pdf_extraction_error"
            return data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        return text.strip()

    def _infer_section(self, text: str, page_num: int) -> str:
        """Infer section type based on text content."""
        text_lower = text.lower()
        if page_num <= 2 and 'abstract' in text_lower:
            return 'Abstract'
        elif 'introduction' in text_lower[:200]:
            return 'Introduction'
        elif any(keyword in text_lower[:200] for keyword in ['method', 'experiment', 'approach']):
            return 'Methodology'
        elif any(keyword in text_lower[:200] for keyword in ['result', 'finding', 'evaluation']):
            return 'Results'
        elif any(keyword in text_lower[:200] for keyword in ['conclusion', 'discussion']):
            return 'Conclusion'
        return 'Other'

class ChunkingStage(PipelineStage):
    """Stage 2: Create text chunks"""
    
    def __init__(self, input_queue: Queue, output_queue: Queue, chunk_size: int = 600, overlap: int = 50):
        super().__init__("Chunking", input_queue, output_queue)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def process(self, data: PipelineData) -> PipelineData:
        try:
            if data.error:
                return data
                
            logger.info("Creating text chunks...")
            chunks = []
            
            for page_data in data.pages_text:
                text = page_data['text']
                page_num = page_data['page_number']
                section = page_data['section']
                
                if not text.strip():
                    continue
                
                # Use regex sentence tokenization
                sentences = self.regex_sentence_tokenize(text)
                
                current_chunk = ""
                
                for sentence in sentences:
                    potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if len(potential_chunk) <= self.chunk_size:
                        current_chunk = potential_chunk
                    else:
                        if current_chunk.strip():
                            chunk_id = str(uuid.uuid4())
                            chunks.append(ResearchChunk(
                                text=current_chunk.strip(),
                                page_number=page_num,
                                section=section,
                                chunk_id=chunk_id
                            ))
                        
                        current_chunk = sentence[-self.overlap:] + " " + sentence if len(sentence) > self.overlap else sentence
                
                if current_chunk.strip():
                    chunk_id = str(uuid.uuid4())
                    chunks.append(ResearchChunk(
                        text=current_chunk.strip(),
                        page_number=page_num,
                        section=section,
                        chunk_id=chunk_id
                    ))
            
            data.chunks = chunks
            data.stage = "chunked"
            logger.info(f"Created {len(chunks)} chunks")
            return data
            
        except Exception as e:
            logger.error(f"Error in chunking: {e}", exc_info=True)
            data.error = e
            data.stage = "chunking_error"
            return data
    
    def regex_sentence_tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences using regex patterns."""
        abbreviations = {
            'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'corp', 'co',
            'vs', 'etc', 'fig', 'e.g', 'i.e', 'al', 'et', 'cf', 'p',
            'pp', 'vol', 'no', 'ed', 'eds', 'min', 'max', 'approx',
            'dept', 'univ', 'assoc', 'dev', 'eng', 'tech', 'sci',
            'int', 'nat', 'comp', 'gen', 'spec', 'std', 'ref'
        }
        
        protected_text = text
        for abbr in abbreviations:
            pattern = rf'\b{re.escape(abbr)}\.'
            replacement = f'{abbr}<PERIOD>'
            protected_text = re.sub(pattern, replacement, protected_text, flags=re.IGNORECASE)
        
        protected_text = re.sub(r'(\d+)\.(\d+)', r'\1<PERIOD>\2', protected_text)
        protected_text = re.sub(r'\bet al\. (\d{4})\)', r'et al<PERIOD> \1)', protected_text)
        
        sentence_pattern = r'[.!?]+(?:\s+(?=[A-Z])|$)'
        sentences = re.split(sentence_pattern, protected_text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.replace('<PERIOD>', '.').strip()
            if sentence:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

class EmbeddingStage(PipelineStage):
    """Stage 3: Create embeddings with batching"""
    
    def __init__(self, input_queue: Queue, output_queue: Queue, batch_size: int = 8):
        super().__init__("Embedding", input_queue, output_queue)
        self.batch_size = batch_size
        self.embedding_model = None
        self.tokenizer = None
        
    def process(self, data: PipelineData) -> PipelineData:
        try:
            if data.error:
                return data
                
            # Lazy load embedding model
            if self.embedding_model is None:
                self.embedding_model, self.tokenizer = get_embedding_model()
            
            logger.info(f"Creating embeddings for {len(data.chunks)} chunks...")
            chunks = data.chunks
            texts = [f"{chunk.section}: {chunk.text}" for chunk in chunks]
            embeddings = []

            # Process in batches for better memory management
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._create_batch_embeddings(batch_texts)
                embeddings.append(batch_embeddings)

            embeddings = np.vstack(embeddings)
            
            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
            
            data.embeddings = embeddings
            data.stage = "embedded"
            logger.info("Embeddings created successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error in embedding: {e}", exc_info=True)
            data.error = e
            data.stage = "embedding_error"
            return data
    
    def _create_batch_embeddings(self, batch_texts: List[str]) -> np.ndarray:
        """Create embeddings for a batch of texts"""
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        return batch_embeddings.cpu().numpy()

class VectorStoreStage(PipelineStage):
    """Stage 4: Store vectors in Qdrant"""
    
    def __init__(self, input_queue: Queue, output_queue: Queue, qdrant_client: QdrantClient):
        super().__init__("VectorStore", input_queue, output_queue)
        self.qdrant_client = qdrant_client
        
    def process(self, data: PipelineData) -> PipelineData:
        try:
            if data.error:
                return data
                
            logger.info("Storing vectors in Qdrant...")
            
            # Generate unique collection name
            collection_name = f"research_papers_{uuid.uuid4().hex[:8]}"
            
            # Create collection
            embedding_dim = data.chunks[0].embedding.shape[0]
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            # Prepare points for batch insertion
            points = []
            for chunk in data.chunks:
                points.append(PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding.tolist(),
                    payload={
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "section": chunk.section
                    }
                ))
            
            # Batch insert points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
            
            # Store collection name in data
            if not data.metadata:
                data.metadata = {}
            data.metadata['collection_name'] = collection_name
            
            data.stage = "stored"
            logger.info(f"Stored {len(points)} vectors in collection: {collection_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error in vector storage: {e}", exc_info=True)
            data.error = e
            data.stage = "storage_error"
            return data

class MetadataExtractionStage(PipelineStage):
    """Stage 5: Extract paper metadata (runs in parallel with other stages)"""
    
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__("MetadataExtraction", input_queue, output_queue)
        
    def process(self, data: PipelineData) -> PipelineData:
        try:
            if data.error or not data.pages_text:
                return data
                
            logger.info("Extracting metadata...")
            
            first_page = data.pages_text[0]['text'] if data.pages_text else ""
            abstract = ""
            title = ""
            authors = ""
            
            for page in data.pages_text[:2]:
                text_lower = page['text'].lower()
                if 'abstract' in text_lower:
                    abstract_start = text_lower.find('abstract')
                    abstract = page['text'][abstract_start:abstract_start + 300].strip()
                
                if page['page_number'] == 1:
                    lines = page['text'].split('. ')
                    title = lines[0].strip() if lines else ""
                    authors = lines[1].strip() if len(lines) > 1 else ""
            
            metadata = {
                'title': title,
                'authors': authors,
                'abstract': abstract
            }
            
            if data.metadata:
                data.metadata.update(metadata)
            else:
                data.metadata = metadata
            
            data.stage = "metadata_extracted"
            logger.info("Metadata extracted successfully")
            return data

class PipelinedResearchPaperRAG:
    """Main class for pipelined research paper RAG processing"""
    
    def __init__(self, api_key: str, qdrant_url: str, qdrant_api_key: str, 
                 chunk_size: int = 600, overlap: int = 50, max_workers: int = 4):
        logger.info("Initializing PipelinedResearchPaperRAG...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=30,
            )
        )
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers
        
        # Preload embedding model
        try:
            get_embedding_model()
            logger.info("Embedding model preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload embedding model: {str(e)}", exc_info=True)
            raise
        
        # Initialize pipeline queues and stages
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        
        self.pdf_extraction = PDFExtractionStage(self.input_queue, self.output_queue)
        self.chunking = ChunkingStage(self.output_queue, self.output_queue)
        self.embedding = EmbeddingStage(self.output_queue, self.output_queue)
        self.vector_store = VectorStoreStage(self.output_queue, self.output_queue, self.qdrant_client)
        self.metadata_extraction = MetadataExtractionStage(self.input_queue, self.output_queue)
        
        # Start pipeline stages
        for stage in [self.pdf_extraction, self.chunking, self.embedding, self.vector_store, self.metadata_extraction]:
            stage.start()
        
        self.current_data = PipelineData()
        self.paper_metadata = {}
        
    def load_research_paper(self, pdf_path: str) -> None:
        """Load and process a research paper through the pipeline"""
        logger.info(f"Loading research paper from: {pdf_path}")
        
        self.current_data = PipelineData(pdf_path=pdf_path)
        self.input_queue.put(self.current_data)
        
        # Wait for pipeline completion
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.output_queue.get(timeout=1.0)
                if result.error:
                    logger.error(f"Pipeline failed at stage {result.stage}: {result.error}")
                    raise result.error
                if result.stage == "stored" and result.metadata:
                    self.current_data = result
                    self.paper_metadata = result.metadata
                    logger.info("Pipeline completed successfully")
                    break
            except Empty:
                continue
                
        if self.current_data.stage != "stored":
            logger.error("Pipeline timed out or failed")
            raise TimeoutError("Pipeline processing timed out or failed")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up PipelinedResearchPaperRAG resources...")
        for stage in [self.pdf_extraction, self.chunking, self.embedding, self.vector_store, self.metadata_extraction]:
            stage.stop()
        if self.current_data and self.current_data.metadata and 'collection_name' in self.current_data.metadata:
            try:
                self.qdrant_client.delete_collection(self.current_data.metadata['collection_name'])
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the stored vectors and generate a response"""
        if not self.current_data or not self.current_data.metadata or 'collection_name' not in self.current_data.metadata:
            logger.error("No data loaded for querying")
            raise ValueError("No data loaded. Please load a research paper first.")
        
        collection_name = self.current_data.metadata['collection_name']
        
        # Generate query embedding
        embedding_model, tokenizer = get_embedding_model()
        inputs = tokenizer(
            question,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :]
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        query_embedding = query_embedding.cpu().numpy().tolist()[0]
        
        # Search in Qdrant
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Prepare context
        context = "\n".join([f"Page {hit.payload['page_number']}, Section: {hit.payload['section']}:\n{hit.payload['text']}" for hit in search_result])
        
        # Generate response using Gemini
        try:
            response = self.model.generate_content(f"Based on the following context:\n{context}\n\nQuestion: {question}")
            return {
                'answer': response.text,
                'context': context,
                'top_k_results': [(hit.payload['text'], hit.score) for hit in search_result]
            }
        except google.api_core.exceptions.GoogleAPIError as e:
            logger.error(f"API error during query: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during query generation: {e}")
            raise
