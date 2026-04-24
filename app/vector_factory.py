from typing import List, Dict, Any
from app.vectordb.pinecone import PineconeStore
from app.vectordb.weaviate import WeaviateStore
from app.vectordb.qdrant import QdrantStore
from app.vectordb.milvus import MilvusStore

class BaseVectorStore:
    def set_index(self, index_name: str):
        raise NotImplementedError
    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        raise NotImplementedError
    def ingest(self, corpus: List[str], dimension: int):
        raise NotImplementedError
    def hybrid_search(self, query: str, dimension:int, top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

def get_vector_store(db_type: str) -> BaseVectorStore:
    stores = {
        "pinecone": PineconeStore,
        "weaviate": WeaviateStore,
        "qdrant": QdrantStore,
        "milvus": MilvusStore
    }
    if db_type.lower() not in stores:
        raise ValueError(f"Unsupported DB provider: {db_type}")
    return stores[db_type.lower()]()