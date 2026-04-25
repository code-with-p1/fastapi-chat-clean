import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from app.vectordb.common.basevectorstore import BaseVectorStore

load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_dense_embedding(text: str, dimension: int) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=dimension
    ).data[0].embedding


class PineconeStore(BaseVectorStore):
    def __init__(self):
        self.pc = Pinecone()
        self.index = None
        self.bm25 = BM25Encoder() # Initialize an empty encoder

    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        existing_indexes = self.pc.list_indexes().names()
        
        if index_name in existing_indexes:
            if recreate:
                self.pc.delete_index(index_name)
                print(f"Dropped existing Pinecone index '{index_name}'.")
            else:
                return f"Index '{index_name}' already exists."

        # Dotproduct is required for Pinecone hybrid search
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

    def set_index(self, index_name="fastapi-chat-clean"):
        self.index = self.pc.Index(index_name)
        print(f"Pinecone Index set to : {index_name}")

    def ingest(self, corpus: List[str], dimension: int):
        if not self.index:
            raise ValueError("Index not set. Call set_index() before ingesting.")
            
        print("Fitting BM25 model on corpus...")
        self.bm25.fit(corpus)
            
        records = []
        for i, text in enumerate(corpus):
            print(f"Pinecone preparing : {i} -- {text[:55]}...")
            dense_vec = get_dense_embedding(text, dimension)
            
            # Encode sparse vectors using the newly fitted model
            sparse_vec = self.bm25.encode_documents(text)
            
            records.append({
                "id": f"doc_{i}_{int(time.time())}", # Ensure unique IDs for tests
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": {"text": text}
            })
            
        # Insert all records
        t0 = time.time()
        self.index.upsert(vectors=records)
        print(f"\nUpserted {len(records)} documents in {round(time.time()-t0, 4)}s")


    def hybrid_search(self, query: str, dimension: int, top_k: int = 10):
        if not self.index:
            raise ValueError("Index not set. Call set_index() before searching.")
            
        dense_vec = get_dense_embedding(query, dimension)
        sparse_vec = self.bm25.encode_queries(query)
        
        results = self.index.query(
            vector=dense_vec, 
            sparse_vector=sparse_vec, 
            top_k=top_k, 
            include_metadata=True
        )
        
        return [{"text": match["metadata"]["text"], "score": match["score"]} for match in results["matches"]]


# =====================================================================
# Testing Block
# =====================================================================
if __name__ == "__main__":
    print("--- Starting Pinecone End-to-End Test ---")
    
    # 1. Initialize parameters
    TEST_DIMENSION = 1536
    TEST_INDEX = "test-hybrid-collection"
    test_corpus = [
        "A fluffy cat is sleeping on the rug near the fireplace.",
        "The dog is chasing a ball in the backyard.",
        "Python is a versatile programming language for data science.",
        "Milvus supports hybrid search using dense and sparse vectors."
    ]
    test_query = "feline resting indoors"

    # 2. Instantiate Store
    store = PineconeStore()

    # 3. Create Index (with recreate=True to ensure clean state)
    print(f"\n[Test Setup] Creating index '{TEST_INDEX}'...")
    creation_msg = store.create_index(index_name=TEST_INDEX, dimension=TEST_DIMENSION, recreate=True)
    print(creation_msg)

    # 4. Set Index
    store.set_index(TEST_INDEX)

    # 5. Ingest Corpus (This now automatically fits BM25!)
    print("\n[Test] Running Ingestion...")
    store.ingest(corpus=test_corpus, dimension=TEST_DIMENSION)
    
    # 6. Hybrid Search
    print(f"\n[Test] Running Hybrid Search for query: '{test_query}'")
    search_results = store.hybrid_search(query=test_query, dimension=TEST_DIMENSION, top_k=2)
    
    for i, res in enumerate(search_results, 1):
        print(f"  Rank {i} | Score: {res['score']:.4f} | Text: {res['text']}")
        
    print("\n--- Test Complete ---")