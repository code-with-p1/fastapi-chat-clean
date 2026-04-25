import os
from openai import OpenAI
from typing import List
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType as WvDataType
from app.vectordb.common.basevectorstore import BaseVectorStore
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

def get_dense_embedding(text: str, dimension:int) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model=EMBEDDING_MODEL, dimensions=dimension
    ).data[0].embedding

class WeaviateStore(BaseVectorStore):
    def __init__(self):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"])
        )

    def create_index(self, index_name: str, dimension: int, recreate: bool = False):
        # Weaviate refers to these as Collections
        if self.client.collections.exists(index_name):
            if recreate:
                self.client.collections.delete(index_name)
            else:
                return f"Collection '{index_name}' already exists."

        # From 2_Weaviate.ipynb: Define properties explicitly
        self.client.collections.create(
            name=index_name,
            vectorizer_config=Configure.Vectorizer.none(), # We bring our own vectors
            properties=[
                Property(name="text", data_type=WvDataType.TEXT),
                Property(name="doc_id", data_type=WvDataType.TEXT),
            ],
        )
        return f"Weaviate collection '{index_name}' created."

    def set_index(self, index_name="fastapi-chat-clean"):
        self.collection = self.client.collections.get(index_name)
        print(f"Weaviate Index : {self.collection}")

    def ingest(self, corpus: List[str], dimension:int):
        with self.collection.batch.dynamic() as batch:
            for i, text in enumerate(corpus):
                print(f"\n\nWeaviate : {i} -- {text} ")
                batch.add_object(
                    properties={"text": text, "doc_id": f"doc_{i}"},
                    vector=get_dense_embedding(text, dimension)
                )

    def hybrid_search(self, query: str, dimension:int, top_k: int = 10):
        results = self.collection.query.hybrid(
            query=query,
            vector=get_dense_embedding(query, dimension),
            alpha=0.5,
            limit=top_k,
            return_metadata=MetadataQuery(score=True)
        )
        return [{"text": obj.properties["text"], "score": obj.metadata.score} for obj in results.objects]


# =====================================================================
# Testing Block
# =====================================================================
if __name__ == "__main__":
    print("--- Starting Weaviate End-to-End Test ---")
    
    # 1. Initialize parameters
    TEST_DIMENSION = 1536
    TEST_INDEX = "Test_hybrid_collection"
    test_corpus = [
        "A fluffy cat is sleeping on the rug near the fireplace.",
        "The dog is chasing a ball in the backyard.",
        "Python is a versatile programming language for data science.",
        "Milvus supports hybrid search using dense and sparse vectors."
    ]
    test_query = "feline resting indoors"

    # 2. Instantiate Store
    store = WeaviateStore()

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