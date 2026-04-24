import os
from openai import OpenAI
from typing import List, Dict, Any
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus import MilvusClient, DataType as MilvusDataType

from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512 # Or 1536 depending on your existing setup

def get_dense_embedding(text: str) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM
    ).data[0].embedding


class BaseVectorStore:
    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        raise NotImplementedError
    def ingest(self, corpus: List[str]):
        raise NotImplementedError
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError


class MilvusStore(BaseVectorStore):
    def __init__(self):
        self.client = MilvusClient(uri=os.environ.get("MILVUS_URL"), token=os.environ.get("MILVUS_API_KEY"))

    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        if self.client.has_collection(index_name):
            if recreate:
                self.client.drop_collection(index_name)
            else:
                return f"Collection '{index_name}' already exists."

        # From 4_Milvus.ipynb: Detailed schema construction
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("doc_id", MilvusDataType.INT64, is_primary=True)
        schema.add_field("text", MilvusDataType.VARCHAR, max_length=1024)
        schema.add_field("dense_vector", MilvusDataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("sparse_vector", MilvusDataType.SPARSE_FLOAT_VECTOR)

        # Index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128}
        )
        index_params.add_index(
            field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2}
        )

        self.client.create_collection(
            collection_name=index_name,
            schema=schema,
            index_params=index_params,
        )
        return f"Milvus collection '{index_name}' created with schema and IVF_FLAT/SPARSE indexes."

    def ingest(self, corpus: List[str]):
        pass
    
    def hybrid_search(self, query: str, top_k: int = 10):
        pass