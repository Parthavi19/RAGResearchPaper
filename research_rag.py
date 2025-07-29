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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
_embedding_model = None
_tokenizer = None
_model_loading_error = None

def get_embedding_model():
    """Lazy load embedding model to avoid startup delays."""
    global _embedding_model, _tokenizer, _model_loading_error
    
    if _model_loading_error:
        raise _model_loading_error
    
    if _embedding_model is None:
        try:
            logger.info("Loading embedding model...")
            _tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            _embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            _embedding_model.eval()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load all-MiniLM-L6-v2: {e}. Falling back to bert-base-uncased.")
            try:
                _tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                _embedding_model = AutoModel.from_pretrained('bert-base-uncased')
                _embedding_model.eval()
                logger.info("Fallback embedding model loaded successfully")
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
                logger.error(f"Error in {self.name}: {e}")
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
            
        except Exception as e:
            data.error = e
            data.stage = "metadata_error"
            return data

class PipelinedResearchPaperRAG:
    """Main RAG class with pipelined processing"""
    
    def __init__(self, api_key: str, qdrant_url: str, qdrant_api_key: str, 
                 chunk_size: int = 600, overlap: int = 50, max_workers: int = 4):
        logger.info("Initializing PipelinedResearchPaperRAG...")
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=30,
            )
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers
        
        # Pipeline queues
        self.pdf_queue = Queue()
        self.chunking_queue = Queue()
        self.embedding_queue = Queue()
        self.storage_queue = Queue()
        self.metadata_queue = Queue()
        self.result_queue = Queue()
        
        # Pipeline stages
        self.stages = []
        
        # Processing state
        self.current_data = None
        self.collection_name = None
        self.paper_metadata = {}
        
        # Thread pool for parallel operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def _setup_pipeline(self):
        """Setup and start pipeline stages"""
        # Create pipeline stages
        pdf_stage = PDFExtractionStage(self.pdf_queue, self.chunking_queue)
        chunking_stage = ChunkingStage(self.chunking_queue, self.embedding_queue, 
                                     self.chunk_size, self.overlap)
        embedding_stage = EmbeddingStage(self.embedding_queue, self.storage_queue)
        storage_stage = VectorStoreStage(self.storage_queue, self.result_queue, self.qdrant_client)
        
        # Parallel metadata extraction
        metadata_stage = MetadataExtractionStage(self.metadata_queue, Queue())
        
        self.stages = [pdf_stage, chunking_stage, embedding_stage, storage_stage, metadata_stage]
        
        # Start all stages
        for stage in self.stages:
            stage.start()
    
    def _cleanup_pipeline(self):
        """Stop and cleanup pipeline stages"""
        for stage in self.stages:
            stage.stop()
        self.stages = []
    
    def load_research_paper(self, pdf_path: str):
        """Load and process research paper using pipeline"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Loading research paper with pipeline: {pdf_path}")
        
        # Setup pipeline
        self._setup_pipeline()
        
        try:
            # Initialize pipeline data
            pipeline_data = PipelineData(pdf_path=pdf_path)
            
            # Start processing
            self.pdf_queue.put(pipeline_data)
            
            # Also send to metadata queue for parallel processing
            self.metadata_queue.put(pipeline_data)
            
            # Wait for completion
            result_data = self.result_queue.get(timeout=300)  # 5 minute timeout
            
            if result_data.error:
                raise result_data.error
            
            # Store results
            self.current_data = result_data
            self.collection_name = result_data.metadata.get('collection_name')
            self.paper_metadata = {
                k: v for k, v in result_data.metadata.items() 
                if k != 'collection_name'
            }
            
            logger.info(f"Successfully loaded paper with {len(result_data.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            raise
        finally:
            # Cleanup pipeline
            self._cleanup_pipeline()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[ResearchChunk, float]]:
        """Retrieve relevant chunks using pipelined query processing"""
        if not self.collection_name:
            logger.error("No document loaded")
            raise ValueError("No document loaded")
        
        # Create query embedding using parallel processing
        future = self.executor.submit(self._create_query_embedding, query)
        query_embedding = future.result()
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2
        )
        
        results = []
        for result in search_results:
            payload = result.payload
            if "publication date" not in payload["text"].lower():
                chunk = ResearchChunk(
                    text=payload["text"],
                    page_number=payload["page_number"],
                    section=payload["section"],
                    chunk_id=result.id
                )
                results.append((chunk, result.score))
            
            if len(results) >= top_k:
                break
        
        logger.debug(f"Retrieved {len(results)} chunks for query: {query}")
        return results
    
    def _create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query text"""
        embedding_model, tokenizer = get_embedding_model()
        
        query_inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            query_outputs = embedding_model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :]
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        return query_embedding.cpu().numpy()[0]
    
    def generate_response(self, query: str, context_chunks: List[ResearchChunk]) -> str:
        """Generate response using Gemini with parallel processing"""
        context = "\n\n".join([f"[Section: {chunk.section}, Page {chunk.page_number}]\n{chunk.text}"
                               for chunk in context_chunks])
        
        prompt = f"""You are a research assistant specializing in academic papers. Answer the question based only on the provided context and metadata, using technical terms as they appear in the paper.

PAPER METADATA:
Title: {self.paper_metadata.get('title', 'Unknown')}
Authors: {self.paper_metadata.get('authors', 'Unknown')}
Abstract: {self.paper_metadata.get('abstract', 'Not extracted')}

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide a complete and detailed answer, including all relevant information from the context.
2. Use bullet points for key points or lists to ensure clarity.
3. Cite sections and pages from the context to support your answer.
4. If information is missing, state: "Information not found in the provided context."
5. Avoid speculation or external knowledge.
6. Preserve technical terms and definitions from the paper.
7. Do not truncate the response; provide the full answer regardless of length.

ANSWER:"""
        
        # Use thread pool for parallel response generation with retries
        future = self.executor.submit(self._generate_with_retries, prompt)
        return future.result()
    
    def _generate_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Generate response with retry logic"""
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except google.api_core.exceptions.ResourceExhausted as e:
                logger.warning(f"API quota exceeded on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(60)
                else:
                    logger.error(f"Failed to generate response after {retries} attempts: {e}")
                    return f"Error generating response: API quota exceeded."
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return f"Error generating response: {str(e)}"
    
    def calculate_simple_confidence(self, similarities: List[float]) -> float:
        """Calculate a simple confidence score based on similarity scores."""
        if not similarities:
            return 0.0
        
        weights = [1.0 / (i + 1) for i in range(len(similarities))]
        weighted_sum = sum(sim * weight for sim, weight in zip(similarities, weights))
        total_weight = sum(weights)
        
        return min(100.0, (weighted_sum / total_weight) * 100)
    
    def query(self, question: str, top_k: int = 10) -> Dict:
        """Process query with pipelined operations"""
        if not question.strip():
            logger.warning("Empty question provided")
            return {
                'answer': "Please provide a valid question.",
                'sources': [],
                'confidence': 0.0,
                'metadata': {}
            }
        
        # Use parallel processing for retrieval and response generation
        retrieval_future = self.executor.submit(self.retrieve, question, top_k)
        relevant_chunks = retrieval_future.result()
        
        if not relevant_chunks:
            logger.info("No relevant chunks found")
            return {
                'answer': "No relevant information found in the paper.",
                'sources': [],
                'confidence': 0.0,
                'metadata': self.paper_metadata
            }
        
        # Parallel confidence calculation and response generation
        similarities = [sim for _, sim in relevant_chunks]
        confidence_future = self.executor.submit(self.calculate_simple_confidence, similarities)
        response_future = self.executor.submit(
            self.generate_response, question, [chunk for chunk, _ in relevant_chunks]
        )
        
        # Get results
        confidence = confidence_future.result()
        answer = response_future.result()
        
        sources = [{
            'rank': i + 1,
            'section': chunk.section,
            'page': chunk.page_number,
            'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            'relevance_score': float(round(sim, 3))
        } for i, (chunk, sim) in enumerate(relevant_chunks)]
        
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': float(round(confidence, 1)),
            'metadata': self.paper_metadata
        }
        
        logger.info(f"Query processed successfully with confidence: {confidence}%")
        return result
    
    def cleanup(self):
        """Clean up resources"""
        if self.collection_name:
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Cleanup pipeline if still running
        self._cleanup_pipeline()

# Backward compatibility wrapper
def ResearchPaperRAG(api_key: str, qdrant_url: str = None, qdrant_api_key: str = None, **kwargs):
    """Backward compatibility wrapper"""
    if qdrant_url and qdrant_api_key:
        return PipelinedResearchPaperRAG(api_key, qdrant_url, qdrant_api_key, **kwargs)
    else:
        # Fallback to original implementation if Qdrant not configured
        from specterragchain1 import ResearchPaperRAG as OriginalRAG
        return OriginalRAG(api_key, **kwargs)
