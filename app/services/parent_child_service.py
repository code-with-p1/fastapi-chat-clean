import uuid
import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vector_factory import get_vector_store 
from app.services import redis_service

class ParentChildProcessor:
    def __init__(self, db_provider: str, index_name: str, dimension: int = 1536):
        self.db_provider = db_provider
        self.index_name = index_name
        self.dimension = dimension
        self.store = get_vector_store(db_provider)
        self.store.set_index(index_name)
        
        # Parent chunker (~1000 tokens)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        # Child chunker (~200 tokens)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    async def ingest_document(self, text: str):
        """Splits into parents and children, stores parents in Redis, children in Vector DB."""
        parent_chunks = self.parent_splitter.split_text(text)
        
        child_records_to_embed = []
        
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            # 1. Save Parent to Redis
            await redis_service.redis_client.set(f"parent_doc:{parent_id}", parent_text)
            
            # 2. Split into Children
            child_chunks = self.child_splitter.split_text(parent_text)
            for child_text in child_chunks:
                # We modify the text slightly to inject the parent_id so we can extract it later
                # Since your current BaseVectorStore expects a list of strings for 'corpus'
                encoded_child = f"[PARENT_ID:{parent_id}]\n{child_text}"
                child_records_to_embed.append(encoded_child)
                
        # 3. Ingest Children into Vector DB
        if child_records_to_embed:
            self.store.ingest(corpus=child_records_to_embed, dimension=self.dimension)
            return len(parent_chunks), len(child_records_to_embed)
        return 0, 0

    async def retrieve_parents(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieves children, extracts parent IDs, and fetches full parents from Redis."""
        # 1. Search for closest children
        child_results = self.store.hybrid_search(query=query, dimension=self.dimension, top_k=top_k * 2)
        
        parent_ids_seen = set()
        parent_docs = []
        
        # 2. Extract Parent IDs and fetch from Redis
        for hit in child_results:
            text = hit["text"]
            if "[PARENT_ID:" in text:
                try:
                    # Parse out the parent ID
                    header, actual_text = text.split("]\n", 1)
                    parent_id = header.replace("[PARENT_ID:", "")
                    
                    if parent_id not in parent_ids_seen:
                        parent_ids_seen.add(parent_id)
                        # Fetch full parent context
                        parent_data = await redis_service.redis_client.get(f"parent_doc:{parent_id}")
                        if parent_data:
                            parent_docs.append(parent_data.decode("utf-8"))
                            
                        if len(parent_docs) >= top_k:
                            break
                except Exception:
                    continue
                    
        return parent_docs