import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from openai import OpenAI
from pymilvus import (
    MilvusClient,
    DataType as MilvusDataType,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

from app.vectordb.common.basevectorstore import BaseVectorStore

load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_dense_embedding(text: str, dimension: int) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=dimension
    ).data[0].embedding

class MilvusStore(BaseVectorStore):
    def __init__(self):
        self.client = MilvusClient(uri=os.environ.get("MILVUS_URL"), token=os.environ.get("MILVUS_API_KEY"))
        self.collection_name = None
        # Initialize BM25 Embedding Function for Milvus
        analyzer = build_default_analyzer(language="en")
        self.bm25 = BM25EmbeddingFunction(analyzer)

    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        if self.client.has_collection(index_name):
            if recreate:
                self.client.drop_collection(index_name)
                print(f"Dropped existing collection '{index_name}'.")
            else:
                return f"Collection '{index_name}' already exists."

        # Schema construction tailored for standard text ingestion
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("doc_id", MilvusDataType.INT64, is_primary=True, description="document id")
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=1024, description="raw text")
        schema.add_field("dense_vector", MilvusDataType.FLOAT_VECTOR, dim=dimension, description="dense embedding")
        schema.add_field("sparse_vector", MilvusDataType.SPARSE_FLOAT_VECTOR, description="BM25 sparse embedding")

        # Index parameters
        index_params = self.client.prepare_index_params()
        
        # Dense index
        index_params.add_index(
            field_name="dense_vector", 
            index_type="IVF_FLAT", 
            metric_type="COSINE", 
            params={"nlist": 128}
        )
        
        # Sparse index
        index_params.add_index(
            field_name="sparse_vector", 
            index_type="SPARSE_INVERTED_INDEX", 
            metric_type="IP", 
            params={"drop_ratio_build": 0.2}
        )

        self.client.create_collection(
            collection_name=index_name,
            schema=schema,
            index_params=index_params,
        )
        return f"Milvus collection '{index_name}' created with schema and IVF_FLAT/SPARSE indexes."

    def set_index(self, index_name="fastapi-chat-clean"):
        self.collection_name = index_name
        print(f"Milvus Index set to : {self.collection_name}")

    def ingest(self, corpus: List[str], dimension: int):
        if not self.collection_name:
            raise ValueError("Index not set. Call set_index() before ingesting.")

        print("Generating sparse embeddings via BM25...")
        sparse_vecs = self.bm25.encode_documents(corpus)
        
        rows = []
        for i, text in enumerate(corpus):
            print(f"Milvus preparing : {i} -- {text[:55]}...")
            dense_vec = get_dense_embedding(text, dimension)
            
            # Convert sparse row -> dict format required by Milvus (using .col per the notebook)
            sparse_row = sparse_vecs[i]
            sparse_dict = {
                int(col): float(val)
                for col, val in zip(sparse_row.col, sparse_row.data)
            }
            
            rows.append({
                "doc_id": i + int(time.time()),  # Pseudo-unique ID for repeated test runs
                "text": text,
                "dense_vector": dense_vec,
                "sparse_vector": sparse_dict
            })
            
        # Insert and Flush
        t0 = time.time()
        result = self.client.insert(collection_name=self.collection_name, data=rows)
        print(f"\nInserted {result['insert_count']} documents in {round(time.time()-t0, 4)}s")
        
        print("Flushing data to Milvus...")
        self.client.flush(self.collection_name)
        print("Flush completed.")

    def hybrid_search(self, query: str, dimension: int, top_k: int = 10):
        if not self.collection_name:
            raise ValueError("Index not set. Call set_index() before searching.")

        # 1. Encode query with both models
        query_dense = get_dense_embedding(query, dimension)
        
        # IMPORTANT: queries use encode_queries(), NOT encode_documents()
        query_sparse_row = self.bm25.encode_queries([query])[0]
        query_sparse = {
            int(idx): float(val)
            for idx, val in zip(query_sparse_row.col, query_sparse_row.data)
        }

        # 2. Build per-field ANN search requests
        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=20,  # Candidate pool before fusion
        )
        
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=20,
        )

        # 3. Fuse with RRF and return top_k
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),  # k=60 is standard RRF constant
            limit=top_k,
            output_fields=["text"]
        )
        
        # 4. Format output
        formatted_results = []
        if results and len(results) > 0:
            for hit in results[0]:  # results[0] contains hits for the first (and only) query
                formatted_results.append({
                    "text": hit["entity"]["text"],
                    "score": hit["distance"]
                })
                
        return formatted_results


# =====================================================================
# Testing Block
# =====================================================================
if __name__ == "__main__":
    print("--- Starting MilvusStore End-to-End Test ---")
    
    # 1. Initialize parameters
    TEST_DIMENSION = 1536
    TEST_INDEX = "test_hybrid_collection"
    test_corpus = [
        "A fluffy cat is sleeping on the rug near the fireplace.",
        "The dog is chasing a ball in the backyard.",
        "Python is a versatile programming language for data science.",
        "Milvus supports hybrid search using dense and sparse vectors."
    ]
    test_query = "feline resting indoors"

    # 2. Instantiate Store
    store = MilvusStore()

    # 3. Create Index (with recreate=True to ensure clean state)
    print(f"\n[Test Setup] Creating index '{TEST_INDEX}'...")
    creation_msg = store.create_index(index_name=TEST_INDEX, dimension=TEST_DIMENSION, recreate=True)
    print(creation_msg)

    # 4. Set Index
    store.set_index(TEST_INDEX)

    # 5. Ingest Corpus
    print("\n[Test] Running Ingestion...")
    store.ingest(corpus=test_corpus, dimension=TEST_DIMENSION)
    
    # 6. Hybrid Search
    print(f"\n[Test] Running Hybrid Search for query: '{test_query}'")
    search_results = store.hybrid_search(query=test_query, dimension=TEST_DIMENSION, top_k=2)
    
    for i, res in enumerate(search_results, 1):
        print(f"  Rank {i} | Score: {res['score']:.4f} | Text: {res['text']}")
        
    print("\n--- Test Complete ---")