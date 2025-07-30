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
import tempfile
import shutil

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
        
        try:
            genai.configure(api_key=api_key)
            logger.info("Google API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Google API: {e}")
            raise

        # Initialize Qdrant client (cloud or local)
        try:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if qdrant_url and qdrant_api_key:
                # Use cloud Qdrant
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=60,  # Increase timeout for cloud operations
                )
                logger.info(f"Initialized cloud Qdrant client: {qdrant_url[:50]}...")
            else:
                # Fallback to local Qdrant with proper path handling
                local_path = os.path.join(tempfile.gettempdir(), "local_qdrant")
                os.makedirs(local_path, exist_ok=True)
                self.qdrant_client = QdrantClient(path=local_path)
                logger.info(f"Initialized local Qdrant client at: {local_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

        # Initialize embedding model with error handling and retry logic
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model: {model_name}")
            
            # Try to load with retries for cloud environments
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=None,  # Use default cache
                        local_files_only=False
                    )
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=None,  # Use default cache
                        local_files_only=False
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Model loading attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2)
            
            # Set model to evaluation mode
            self.model.eval()
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize Gemini model with error handling
        try:
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks with comprehensive error handling"""
        try:
            # Handle empty input
            if not texts:
                logger.warning("Empty text list provided for embedding")
                return []
            
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts found after filtering")
                return []
            
            logger.info(f"Generating embeddings for {len(valid_texts)} text chunks")
            
            # Process in batches to handle memory constraints
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                try:
                    # Tokenize with proper truncation
                    max_length = 512
                    inputs = self.tokenizer(
                        batch_texts, 
                        padding=True, 
                        truncation=True, 
                        max_length=max_length,
                        return_tensors="pt"
                    )
                    
                    # Generate embeddings
                    with torch.no_grad():
                        model_output = self.model(**inputs)
                    
                    # Mean pooling
                    embeddings = model_output.last_hidden_state.mean(dim=1)
                    batch_embeddings = embeddings.numpy().tolist()
                    all_embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(valid_texts)-1)//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Create zero embeddings for failed batch
                    embedding_dim = 384  # all-MiniLM-L6-v2 dimension
                    zero_embeddings = [[0.0] * embedding_dim for _ in batch_texts]
                    all_embeddings.extend(zero_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Critical error in embedding generation: {e}")
            raise

    def _extract_text_from_pdf(self, paper_path: str) -> str:
        """Extract text from PDF with comprehensive error handling"""
        try:
            if not os.path.exists(paper_path):
                raise FileNotFoundError(f"PDF file not found: {paper_path}")
            
            file_size = os.path.getsize(paper_path)
            if file_size == 0:
                raise ValueError("PDF file is empty")
            
            logger.info(f"Extracting text from PDF: {paper_path} ({file_size} bytes)")
            
            text_parts = []
            total_pages = 0
            
            try:
                with open(paper_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    total_pages = len(reader.pages)
                    
                    if total_pages == 0:
                        raise ValueError("PDF file appears to have no pages")
                    
                    logger.info(f"PDF has {total_pages} pages")
                    
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                # Clean up the text
                                cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                                text_parts.append(cleaned_text)
                                logger.debug(f"Extracted {len(cleaned_text)} chars from page {page_num + 1}")
                            else:
                                logger.warning(f"No text found on page {page_num + 1}")
                        except Exception as e:
                            logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                            continue
            
            except Exception as e:
                logger.error(f"Error reading PDF file: {e}")
                raise ValueError(f"Failed to read PDF: {e}")
            
            if not text_parts:
                raise ValueError("No text could be extracted from any page of the PDF")
            
            full_text = " ".join(text_parts)
            logger.info(f"Successfully extracted {len(full_text)} characters from {len(text_parts)}/{total_pages} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunk text with overlap for better context preservation"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for chunking")
                return []
            
            # Clean text thoroughly
            text = re.sub(r'\s+', ' ', text.strip())
            
            if len(text) <= chunk_size:
                logger.info("Text is shorter than chunk size, returning single chunk")
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
                if chunk and len(chunk) > 10:  # Minimum chunk size
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = end - overlap
                if start >= len(text):
                    break
            
            # Remove duplicate chunks
            unique_chunks = []
            seen = set()
            for chunk in chunks:
                chunk_hash = hash(chunk)
                if chunk_hash not in seen:
                    unique_chunks.append(chunk)
                    seen.add(chunk_hash)
            
            logger.info(f"Created {len(unique_chunks)} unique text chunks (removed {len(chunks) - len(unique_chunks)} duplicates)")
            return unique_chunks
            
        except Exception as e:
            logger.error(f"Error in text chunking: {e}")
            raise

    def _store_chunks(self, paper_id: str, chunks: List[str], embeddings: List[List[float]]):
        """Store chunks in Qdrant with comprehensive error handling"""
        try:
            if not chunks or not embeddings:
                raise ValueError("No chunks or embeddings to store")
            
            if len(chunks) != len(embeddings):
                raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
            
            logger.info(f"Storing {len(chunks)} chunks in Qdrant collection: {self.collection_name}")
            
            # Create collection if it doesn't exist
            try:
                collections = self.qdrant_client.get_collections()
                collection_exists = any(c.name == self.collection_name for c in collections.collections)
            except Exception as e:
                logger.warning(f"Could not check collections: {e}")
                collection_exists = False
            
            if not collection_exists:
                try:
                    # Determine vector size from first embedding
                    vector_size = len(embeddings[0])
                    logger.info(f"Creating collection with vector size: {vector_size}")
                    
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                    logger.info(f"Created collection: {self.collection_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Collection {self.collection_name} already exists")
                    else:
                        logger.error(f"Failed to create collection: {e}")
                        raise

            # Create points with better error handling
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                try:
                    # Validate embedding
                    if not embedding or not isinstance(embedding, list):
                        logger.warning(f"Invalid embedding for chunk {i}, skipping")
                        continue
                    
                    point = PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=embedding, 
                        payload={
                            "chunk": chunk[:1000],  # Limit chunk size in payload
                            "paper_id": paper_id,
                            "chunk_index": i,
                            "chunk_length": len(chunk)
                        }
                    )
                    points.append(point)
                except Exception as e:
                    logger.warning(f"Failed to create point for chunk {i}: {e}")
                    continue
            
            if not points:
                raise ValueError("No valid points created for storage")
            
            # Store in batches to handle large documents
            batch_size = 100
            successful_batches = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(points)-1)//batch_size + 1
                
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name, 
                        points=batch,
                        wait=True  # Wait for operation to complete
                    )
                    successful_batches += 1
                    logger.info(f"Successfully stored batch {batch_num}/{total_batches}")
                    
                except Exception as e:
                    logger.error(f"Failed to store batch {batch_num}: {e}")
                    # Continue with other batches instead of failing completely
                    continue
            
            if successful_batches == 0:
                raise Exception("Failed to store any batches in Qdrant")
            
            logger.info(f"Successfully stored {successful_batches}/{total_batches} batches ({len(points)} total points)")
            
        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            raise

    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata using Gemini with comprehensive error handling"""
        try:
            # Limit text length for API call
            text_preview = text[:4000] if len(text) > 4000 else text
            
            prompt = (
                "Extract metadata from this research paper text. "
                "Provide a structured response with title, authors, abstract, and key topics. "
                "Format your response as a clear summary. "
                "If specific information is not available, indicate 'Not found'.\n\n"
                f"Text: {text_preview}"
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=1000
                        )
                    )
                    
                    if response and response.text:
                        metadata_text = response.text.strip()
                        logger.info("Successfully extracted metadata using Gemini")
                        return {
                            "extracted_metadata": metadata_text,
                            "extraction_method": "gemini",
                            "text_length": len(text),
                            "preview_length": len(text_preview)
                        }
                    else:
                        logger.warning(f"Empty response from Gemini on attempt {attempt + 1}")
                        
                except Exception as e:
                    logger.warning(f"Gemini API error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        break
                    time.sleep(1)  # Brief delay before retry
            
            # Fallback: simple text-based extraction
            logger.info("Falling back to simple metadata extraction")
            return self._extract_simple_metadata(text_preview)
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "extracted_metadata": f"Metadata extraction failed: {str(e)}",
                "extraction_method": "failed",
                "error": str(e)
            }

    def _extract_simple_metadata(self, text: str) -> Dict:
        """Simple fallback metadata extraction"""
        try:
            lines = text.split('\n')[:20]  # First 20 lines
            
            # Try to find title (usually in first few lines)
            title = "Title not found"
            for line in lines[:5]:
                line = line.strip()
                if len(line) > 10 and not line.isupper():
                    title = line
                    break
            
            # Try to find abstract
            abstract = "Abstract not found"
            text_lower = text.lower()
            abstract_start = text_lower.find('abstract')
            if abstract_start != -1:
                abstract_text = text[abstract_start:abstract_start + 500]
                abstract = abstract_text[:abstract_text.find('\n\n')] if '\n\n' in abstract_text else abstract_text
            
            return {
                "extracted_metadata": f"Title: {title}\n\nAbstract: {abstract}",
                "extraction_method": "simple",
                "text_length": len(text)
            }
            
        except Exception as e:
            return {
                "extracted_metadata": f"Simple extraction failed: {str(e)}",
                "extraction_method": "failed",
                "error": str(e)
            }

    def load_research_paper(self, paper_path: str) -> Dict:
        """Load and process research paper with comprehensive error handling"""
        try:
            paper_id = str(uuid.uuid4())
            logger.info(f"Processing paper: {paper_path} with ID: {paper_id}")

            # Step 1: Extract text
            logger.info("Step 1: Extracting text from PDF...")
            text = self._extract_text_from_pdf(paper_path)

            # Step 2: Chunk text
            logger.info("Step 2: Chunking text...")
            chunks = self._chunk_text(text)
            
            if not chunks:
                raise ValueError("No text chunks created from PDF")

            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings...")
            embeddings = self._embed(chunks)
            
            if len(embeddings) != len(chunks):
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")

            # Step 4: Store in vector database
            logger.info("Step 4: Storing in vector database...")
            self._store_chunks(paper_id, chunks, embeddings)

            # Step 5: Extract metadata
            logger.info("Step 5: Extracting metadata...")
            metadata = self._extract_metadata(text)

            # Add processing statistics
            metadata.update({
                "paper_id": paper_id,
                "processing_stats": {
                    "total_chunks": len(chunks),
                    "total_embeddings": len(embeddings),
                    "text_length": len(text),
                    "collection_name": self.collection_name
                }
            })

            logger.info(f"Successfully processed paper {paper_id} with {len(chunks)} chunks")
            return metadata

        except Exception as e:
            logger.error(f"Error in load_research_paper: {e}")
            logger.error(traceback.format_exc())
            raise

    def query(self, query_text: str, top_k: int = 5) -> str:
        """Query the loaded documents with comprehensive error handling"""
        try:
            if not query_text or not query_text.strip():
                return "Please provide a valid question."
            
            query_text = query_text.strip()
            logger.info(f"Processing query: {query_text[:100]}...")
            
            # Generate query embedding
            try:
                query_embedding = self._embed([query_text])
                if not query_embedding:
                    return "Failed to generate query embedding."
                query_vector = query_embedding[0]
            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
                return f"Failed to process query: {str(e)}"

            # Search in Qdrant
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=0.3  # Minimum similarity threshold
                )
                logger.info(f"Found {len(results)} search results")
            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")
                return f"Search failed: {str(e)}"

            if not results:
                return "No relevant information found in the uploaded document. Try rephrasing your question or ask about different aspects of the paper."

            # Build context from search results
            context_parts = []
            for i, hit in enumerate(results):
                chunk_text = hit.payload.get("chunk", "")
                score = hit.score
                if chunk_text and score > 0.3:  # Filter low-quality matches
                    context_parts.append(f"Context {i+1} (relevance: {score:.3f}):\n{chunk_text}")

            if not context_parts:
                return "No sufficiently relevant information found. Please try a different question."

            context = "\n\n".join(context_parts)
            logger.info(f"Built context from {len(context_parts)} chunks")

            # Generate response using Gemini
            prompt = (
                f"Based on the following context from a research paper, "
                f"please answer this question: {query_text}\n\n"
                f"Context:\n{context}\n\n"
                f"Please provide a comprehensive and accurate answer based on the context. "
                f"If the context doesn't contain enough information to fully answer the question, "
                f"please indicate what information is available and what is missing. "
                f"Be specific and cite relevant parts of the context when possible."
            )

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=2000
                        )
                    )
                    
                    if response and response.text:
                        answer = response.text.strip()
                        logger.info("Successfully generated response")
                        return answer
                    else:
                        logger.warning(f"Empty response from Gemini on attempt {attempt + 1}")
                        
                except Exception as e:
                    logger.warning(f"Gemini generation error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        break
                    time.sleep(1)
            
            # Fallback response
            return f"Based on the available context, I found {len(context_parts)} relevant sections, but couldn't generate a proper response due to API issues. The most relevant context was: {context_parts[0][:500]}..." if context_parts else "Failed to generate response."
                    
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return f"Query processing failed: {str(e)}"

    def cleanup(self):
        """Clean up resources with comprehensive error handling"""
        try:
            logger.info(f"Cleaning up RAG instance for collection: {self.collection_name}")
            
            if hasattr(self, 'qdrant_client') and hasattr(self, 'collection_name'):
                try:
                    # For cloud Qdrant, we might not want to delete collections
                    # as they could be shared or persistent
                    qdrant_url = os.getenv("QDRANT_URL")
                    if qdrant_url:
                        logger.info(f"Skipping collection deletion for cloud Qdrant: {self.collection_name}")
                    else:
                        # Only delete for local Qdrant
                        try:
                            self.qdrant_client.delete_collection(self.collection_name)
                            logger.info(f"Deleted local collection: {self.collection_name}")
                        except Exception as e:
                            if "not found" in str(e).lower():
                                logger.info(f"Collection {self.collection_name} already deleted or not found")
                            else:
                                raise
                except Exception as e:
                    logger.warning(f"Collection cleanup warning: {e}")
                    
            # Clear model references to free memory
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Example usage (commented out for production)
# if __name__ == '__main__':
#     try:
#         rag = PipelinedResearchPaperRAG("test_collection")
#         metadata = rag.load_research_paper("path/to/paper.pdf")
#         print("Metadata:", metadata)
#         
#         answer = rag.query("What is the main contribution?")
#         print("Answer:", answer)
#         
#         rag.cleanup()
#     except Exception as e:
#         print(f"Error: {e}")
