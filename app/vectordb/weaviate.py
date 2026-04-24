import os
from openai import OpenAI
from typing import List, Dict, Any
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType as WvDataType
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

class WeaviateStore(BaseVectorStore):
    def __init__(self):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ["WEAVIATE_URL"],
            auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"])
        )

    def set_index(self, index_name="fastapi-chat-clean"):
        self.collection = self.client.collections.get(index_name)
        print(f"Weaviate Index : {self.collection}")

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

