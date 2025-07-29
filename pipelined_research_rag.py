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
import uuid
import sys
import queue
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up GenAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(path="./local_qdrant")

        # Set up embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Set up Gemini model
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Tokenize and embed
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.numpy().tolist()

    def _extract_text_from_pdf(self, paper_path: str) -> str:
        with open(paper_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        text = re.sub(r'\s+', ' ', text)
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _store_chunks(self, paper_id: str, chunks: List[str], embeddings: List[List[float]]):
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"chunk": chunk, "paper_id": paper_id})
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

    def _extract_metadata(self, text: str) -> Dict:
        prompt = (
            "Given the text of a research paper, extract metadata including title, authors, abstract, and keywords. "
            f"Here is the text: \n{text[:4000]}"
        )
        response = self.gemini_model.generate_content(prompt)
        return {"extracted_metadata": response.text.strip()}

    def load_research_paper(self, paper_path: str) -> Dict:
        paper_id = str(uuid.uuid4())
        data = PipelineData(paper_id=paper_id, paper_path=paper_path)

        # Define thread-safe queues for this instance
        text_queue = queue.Queue()
        chunk_queue = queue.Queue()
        embedding_queue = queue.Queue()

        # Stage 1: PDF Text Extraction
        def extract_pdf():
            logger.info(f"Extracting text from {data.paper_path}")
            text = self._extract_text_from_pdf(data.paper_path)
            text_queue.put(text)

        # Stage 2: Chunking
        def chunk_text():
            text = text_queue.get()
            logger.info("Chunking text")
            chunks = self._chunk_text(text)
            data.text_chunks = chunks
            chunk_queue.put(chunks)

        # Stage 3: Embedding
        def embed_chunks():
            chunks = chunk_queue.get()
            logger.info("Embedding chunks")
            embeddings = self._embed(chunks)
            data.chunk_embeddings = embeddings
            embedding_queue.put((chunks, embeddings))

        # Stage 4: Store in Vector DB
        def store_in_qdrant():
            chunks, embeddings = embedding_queue.get()
            logger.info("Storing in Qdrant")
            self._store_chunks(data.paper_id, chunks, embeddings)

        # Stage 5: Metadata Extraction
        def extract_metadata():
            text = text_queue.queue[0]  # Reuse the same text
            logger.info("Extracting metadata")
            data.metadata = self._extract_metadata(text)

        # Start pipeline threads
        threads = [
            threading.Thread(target=extract_pdf),
            threading.Thread(target=chunk_text),
            threading.Thread(target=embed_chunks),
            threading.Thread(target=store_in_qdrant),
            threading.Thread(target=extract_metadata),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return data.metadata

    def query(self, query_text: str, top_k: int = 5) -> str:
        logger.info(f"Querying: {query_text}")
        query_embedding = self._embed([query_text])[0]

        try:
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
        except google.api_core.exceptions.NotFound:
            logger.error(f"Collection '{self.collection_name}' not found.")
            return "No results found."

        context = "\n".join([hit.payload["chunk"] for hit in results])
        prompt = f"Using the following context, answer the query: {query_text}\n\nContext:\n{context}"
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()

# Usage Example:
# if __name__ == '__main__':
#     rag1 = PipelinedResearchPaperRAG("my_rag_collection")
#     metadata1 = rag1.load_research_paper("path/to/first.pdf")
#     print(metadata1)
#
#     rag2 = PipelinedResearchPaperRAG("my_rag_collection")
#     metadata2 = rag2.load_research_paper("path/to/second.pdf")
#     print(metadata2)
#
#     print(rag1.query("What is the main contribution?"))

