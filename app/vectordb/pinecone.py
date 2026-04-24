import os
from openai import OpenAI
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from app.vectordb.basevectorstore import BaseVectorStore
import time
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_dense_embedding(text: str, dimension:int) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=dimension
    ).data[0].embedding

class PineconeStore(BaseVectorStore):
    def __init__(self):
        self.pc = Pinecone()
        self.bm25 = BM25Encoder()
        # Graceful fallback if the pre-fitted model isn't on disk yet
        if os.path.exists("bm25_model.json"):
            self.bm25.load("bm25_model.json")
        else:
            print("Warning: bm25_model.json not found. Loading default BM25 weights.")
            self.bm25.default()

    def set_index(self, index_name="fastapi-chat-clean"):
        self.index = self.pc.Index(index_name)
        print(f"Pinecone Index : {self.index}")

    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        existing_indexes = self.pc.list_indexes().names()
        
        if index_name in existing_indexes:
            if recreate:
                self.pc.delete_index(index_name)
            else:
                return f"Index '{index_name}' already exists."

        # From 1_Pinecone.ipynb: Dotproduct is required for hybrid search
        self.pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait for readiness
        while not self.pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
            
        return f"Pinecone index '{index_name}' created with dimension {dimension}."

    def ingest(self, corpus: List[str], dimension:int):
        records = []
        for i, text in enumerate(corpus):
            print(f"\n\nPinecone : {i} -- {text} ")
            dense_vec = get_dense_embedding(text, dimension)
            sparse_vec = self.bm25.encode_documents(text)
            records.append({
                "id": f"doc_{i}",
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": {"text": text}
            })
        self.index.upsert(vectors=records)

    def hybrid_search(self, query: str, dimension:int, top_k: int = 10):
        dense_vec = get_dense_embedding(query, dimension)
        sparse_vec = self.bm25.encode_queries(query)
        results = self.index.query(
            vector=dense_vec, sparse_vector=sparse_vec, top_k=top_k, include_metadata=True
        )
        return [{"text": match["metadata"]["text"], "score": match["score"]} for match in results["matches"]]