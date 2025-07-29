import os
import google.generativeai as genai
import numpy as np
import PyPDF2
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import time
import google.api_core.exceptions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data structure to hold information between stages
@dataclass
class PipelineData:
    paper_id: str
    paper_path: str
    text_chunks: List[str] = None
    chunk_embeddings: List[List[float]] = None
    metadata: Dict = None

class PipelinedResearchPaperRAG:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        # Configure Google API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=api_key)

        # Initialize Qdrant client (local file-based)
        try:
            self.qdrant_client = QdrantClient(path="./local_qdrant")
            logger.info("Initialized local Qdrant client")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

        # Initialize embedding model with error handling
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize Gemini model
        try:
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks with error handling"""
        try:
            # Handle empty input
            if not texts:
                return []
            
            # Batch processing for efficiency
            max_length = 512  # Limit token length
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                model_output = self.model(**inputs)
            
            # Mean pooling
            embeddings = model_output.last_hidden_state.mean(dim=1)
            return embeddings.numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            raise

    def _extract_text_from_pdf(self, paper_path: str) -> str:
        """Extract text from PDF with better error handling"""
        try:
            if not os.path.exists(paper_path):
                raise FileNotFoundError(f"PDF file not found: {paper_path}")
            
            text_parts = []
            with open(paper_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                if len(reader.pages) == 0:
                    raise ValueError("PDF file appears to be empty")
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue
            
            if not text_parts:
                raise ValueError("No text could be extracted from the PDF")
            
            full_text = " ".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(text_parts)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text with overlap for better context preservation"""
        try:
            # Clean text
            text = re.sub(r'\s+', ' ', text.strip())
            
            if len(text) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                
                # Try to break at word boundaries
                if end < len(text):
                    # Look for the last space within the chunk
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = end - overlap
                if start >= len(text):
                    break
            
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in text chunking: {e}")
            raise

    def _store_chunks(self, paper_id: str, chunks: List[str], embeddings: List[List[float]]):
        """Store chunks in Qdrant with error handling"""
        try:
            if not chunks or not embeddings:
                raise ValueError("No chunks or embeddings to store")
            
            if len(chunks) != len(embeddings):
                raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
            
            # Create collection if it doesn't exist
            try:
                collections = self.qdrant_client.get_collections()
                collection_exists = any(c.name == self.collection_name for c in collections.collections)
            except:
                collection_exists = False
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")

            # Create points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=embedding, 
                    payload={
                        "chunk": chunk, 
                        "paper_id": paper_id,
                        "chunk_index": i
                    }
                )
                points.append(point)
            
            # Store in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.qdrant_client.upsert(collection_name=self.collection_name, points=batch)
            
            logger.info(f"Stored {len(points)} chunks in Qdrant")
            
        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            raise

    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata using Gemini with better error handling"""
        try:
            # Limit text length for API call
            text_preview = text[:4000] if len(text) > 4000 else text
            
            prompt = (
                "Extract metadata from this research paper text. "
                "Provide a structured response with title, authors, and a brief summary. "
                "If information is not available, indicate 'Not found'.\n\n"
                f"Text: {text_preview}"
            )
            
            response = self.gemini_model.generate_content(prompt)
            
            if not response or not response.text:
                return {"extracted_metadata": "Metadata extraction failed"}
            
            return {"extracted_metadata": response.text.strip()}
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"extracted_metadata": f"Metadata extraction failed: {str(e)}"}

    def load_research_paper(self, paper_path: str) -> Dict:
        """Load and process research paper - simplified sequential version"""
        try:
            paper_id = str(uuid.uuid4())
            logger.info(f"Processing paper: {paper_path} with ID: {paper_id}")

            # Step 1: Extract text
            text = self._extract_text_from_pdf(paper_path)

            # Step 2: Chunk text
            chunks = self._chunk_text(text)

            # Step 3: Generate embeddings
            embeddings = self._embed(chunks)

            # Step 4: Store in vector database
            self._store_chunks(paper_id, chunks, embeddings)

            # Step 5: Extract metadata
            metadata = self._extract_metadata(text)

            logger.info(f"Successfully processed paper {paper_id}")
            return metadata

        except Exception as e:
            logger.error(f"Error in load_research_paper: {e}")
            raise

    def query(self, query_text: str, top_k: int = 5) -> str:
        """Query the loaded documents"""
        try:
            logger.info(f"Querying: {query_text}")
            
            # Generate query embedding
            query_embedding = self._embed([query_text])[0]

            # Search in Qdrant
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k
                )
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")
                return f"Search failed: {str(e)}"

            if not results:
                return "No relevant information found in the uploaded document."

            # Build context from search results
            context_parts = []
            for i, hit in enumerate(results):
                chunk_text = hit.payload.get("chunk", "")
                score = hit.score
                context_parts.append(f"Context {i+1} (relevance: {score:.3f}):\n{chunk_text}")

            context = "\n\n".join(context_parts)

            # Generate response using Gemini
            prompt = (
                f"Based on the following context from a research paper, "
                f"please answer this question: {query_text}\n\n"
                f"Context:\n{context}\n\n"
                f"Please provide a comprehensive answer based on the context. "
                f"If the context doesn't contain enough information to answer the question, "
                f"please indicate that clearly."
            )

            try:
                response = self.gemini_model.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                else:
                    return "Failed to generate response from the model."
                    
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                return f"Response generation failed: {str(e)}"

        except Exception as e:
            logger.error(f"Error in query: {e}")
            return f"Query processing failed: {str(e)}"

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'qdrant_client') and hasattr(self, 'collection_name'):
                try:
                    self.qdrant_client.delete_collection(self.collection_name)
                    logger.info(f"Deleted collection: {self.collection_name}")
                except:
                    pass  # Ignore cleanup errors
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Example usage (commented out for production)
# if __name__ == '__main__':
#     rag = PipelinedResearchPaperRAG("test_collection")
#     metadata = rag.load_research_paper("path/to/paper.pdf")
#     print(metadata)
#     print(rag.query("What is the main contribution?"))
