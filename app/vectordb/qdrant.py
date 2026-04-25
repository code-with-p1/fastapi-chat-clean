import os
from openai import OpenAI
from typing import List
from qdrant_client import QdrantClient, models as qdrant_models
from fastembed import SparseTextEmbedding
from app.vectordb.common.basevectorstore import BaseVectorStore
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_dense_embedding(text: str, dimension:int) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=dimension
    ).data[0].embedding

class QdrantStore(BaseVectorStore):
    def __init__(self):
        self.client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
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

    def set_index(self, index_name="fastapi-chat-clean"):
        self.collection_name = index_name
        print(f"Qdrant Index : {self.collection_name}")

    def get_sparse_embedding(self, text: str):
        res = list(self.sparse_model.embed([text]))[0]
        return qdrant_models.SparseVector(indices=res.indices.tolist(), values=res.values.tolist())

    def ingest(self, corpus: List[str], dimension:int):
        points = []
        for i, text in enumerate(corpus):
            print(f"\n\nQdrant : {i} -- {text} ")
            points.append(qdrant_models.PointStruct(
                id=i,
                vector={
                    "dense_idx": get_dense_embedding(text, dimension),
                    "sparse_idx": self.get_sparse_embedding(text)
                },
                payload={"text": text}
            ))
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def hybrid_search(self, query: str, dimension:int, top_k: int = 10):
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                qdrant_models.Prefetch(query=get_dense_embedding(query, dimension), using="dense_idx", limit=20),
                qdrant_models.Prefetch(query=self.get_sparse_embedding(query), using="sparse_idx", limit=20),
            ],
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )
        return [{"text": hit.payload["text"], "score": hit.score} for hit in results.points]


# =====================================================================
# Testing Block
# =====================================================================
if __name__ == "__main__":
    print("--- Starting Qdrant End-to-End Test ---")
    
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
    store = QdrantStore()

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