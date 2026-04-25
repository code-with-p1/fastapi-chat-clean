import os
import glob
import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.vector_factory import get_vector_store
from app.services import redis_service

class ParentChildProcessor:
    def __init__(self, db_provider: str, index_name: str, dimension: int = 1536):
        self.db_provider = db_provider
        self.index_name = index_name
        self.dimension = dimension
        self.store = get_vector_store(db_provider)
        self.store.set_index(index_name)
        
        # Parent chunker (~2000 characters)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        # Child chunker (~400 characters)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    async def ingest_document(self, text: str):
        """Splits into parents and children, stores parents in Redis, children in Vector DB."""
        parent_chunks = self.parent_splitter.split_text(text)
        child_records_to_embed = []
        
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            await redis_service.set_value(f"parent_doc:{parent_id}", parent_text)
            
            # 2. Split into Children
            child_chunks = self.child_splitter.split_text(parent_text)
            for child_text in child_chunks:
                # Inject parent_id so we can extract it during retrieval
                encoded_child = f"[PARENT_ID:{parent_id}]\n{child_text}"
                child_records_to_embed.append(encoded_child)
                
        # 3. Ingest Children into Vector DB
        if child_records_to_embed:
            self.store.ingest(corpus=child_records_to_embed, dimension=self.dimension)
            return len(parent_chunks), len(child_records_to_embed)
        return 0, 0

    async def ingest_directory(self, directory_path: str):
        """Reads all PDFs in a directory and applies parent-child chunking."""
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)
        total_parents = 0
        total_children = 0

        for file_path in pdf_files:
            try:
                print(f"Processing PDF for Parent-Child: {file_path}")
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                full_text = "\n".join([page.page_content for page in pages])
                
                if full_text.strip():
                    p, c = await self.ingest_document(full_text)
                    total_parents += p
                    total_children += c
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        return total_parents, total_children

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
                        parent_data = await redis_service.get_value(f"parent_doc:{parent_id}")
                        if parent_data:
                            parent_docs.append(parent_data)
                            
                        if len(parent_docs) >= top_k:
                            break
                except Exception:
                    continue
                    
        return parent_docs