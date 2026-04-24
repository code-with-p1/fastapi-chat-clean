import os
from openai import OpenAI
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models as qdrant_models
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512 # Or 1536 depending on your existing setup

def get_dense_embedding(text: str) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL
    ).data[0].embedding

class BaseVectorStore:
    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        raise NotImplementedError
    def ingest(self, corpus: List[str]):
        raise NotImplementedError
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

class QdrantStore(BaseVectorStore):
    def __init__(self, collection_name="fastapi_chat_clean"):
        self.client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        self.collection_name = collection_name
        self.sparse_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")

    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        existing = [c.name for c in self.client.get_collections().collections]
        
        if index_name in existing:
            if recreate:
                self.client.delete_collection(index_name)
            else:
                return f"Collection '{index_name}' already exists."

        # From 3_Qdrant.ipynb: Configure dense and sparse vector spaces
        self.client.create_collection(
            collection_name=index_name,
            vectors_config={
                "dense_idx": qdrant_models.VectorParams(
                    size=dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse_idx": qdrant_models.SparseVectorParams(
                    index=qdrant_models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        
        # Optional: Create payload index for metadata filtering
        self.client.create_payload_index(
            collection_name=index_name,
            field_name="category",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        
        return f"Qdrant collection '{index_name}' created with hybrid config."

    def get_sparse_embedding(self, text: str):
        res = list(self.sparse_model.embed([text]))[0]
        return qdrant_models.SparseVector(indices=res.indices.tolist(), values=res.values.tolist())

    def ingest(self, corpus: List[str]):
        points = []
        for i, text in enumerate(corpus):
            print(f"\n\nQDRANT : {i} -- {text} ")
            points.append(qdrant_models.PointStruct(
                id=i,
                vector={
                    "dense_idx": get_dense_embedding(text),
                    "sparse_idx": self.get_sparse_embedding(text)
                },
                payload={"text": text}
            ))
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def hybrid_search(self, query: str, top_k: int = 10):
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                qdrant_models.Prefetch(query=get_dense_embedding(query), using="dense_idx", limit=20),
                qdrant_models.Prefetch(query=self.get_sparse_embedding(query), using="sparse_idx", limit=20),
            ],
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )
        return [{"text": hit.payload["text"], "score": hit.score} for hit in results.points]