import os
import glob
import uuid
from typing import List
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from app.vector_factory import get_vector_store

# Initialize OpenAI for generating embeddings directly in this service
openai_client = OpenAI()

def get_dense_embedding(text: str, dimension: int) -> List[float]:
    return openai_client.embeddings.create(
        input=text, model="text-embedding-3-small", dimensions=dimension
    ).data[0].embedding


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
        """Splits into parents and children, storing the Parent text directly in the Child's metadata."""
        parent_chunks = self.parent_splitter.split_text(text)
        records_to_upsert = []
        
        for parent_text in parent_chunks:
            # Split Parent into Children
            child_chunks = self.child_splitter.split_text(parent_text)
            
            for child_text in child_chunks:
                # 1. Generate Embeddings for the CHILD only
                dense_vec = get_dense_embedding(child_text, self.dimension)
                sparse_vec = self.store.bm25.encode_documents(child_text)
                
                # 2. Store BOTH texts in the metadata payload
                records_to_upsert.append({
                    "id": f"doc_{uuid.uuid4()}",
                    "values": dense_vec,
                    "sparse_values": sparse_vec,
                    "metadata": {
                        "text": child_text,              # Keeps standard hybrid_search from breaking
                        "full_parent_text": parent_text  # The fail-safe persistent parent payload
                    }
                })
                
        # 3. Upsert directly to Pinecone IN BATCHES
        if records_to_upsert:
            batch_size = 100  # Safe batch size to stay well under the 2MB limit
            for i in range(0, len(records_to_upsert), batch_size):
                batch = records_to_upsert[i : i + batch_size]
                self.store.index.upsert(vectors=batch)
                print(f"Upserted batch {i//batch_size + 1} ({len(batch)} records)...")
                
            return len(parent_chunks), len(records_to_upsert)
            
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
        """Queries Pinecone for best children and extracts the embedded parent text."""
        
        # 1. Generate Embeddings for the User Query
        dense_vec = get_dense_embedding(query, self.dimension)
        sparse_vec = self.store.bm25.encode_queries(query)
        
        # 2. Query Pinecone directly to gain access to the full metadata payload
        results = self.store.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k * 2, # Fetch extra in case multiple children belong to the same parent
            include_metadata=True
        )
        
        parent_texts_seen = set()
        parent_docs = []
        
        # 3. Extract the persistent 'full_parent_text' from the results
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            parent_text = metadata.get("full_parent_text")
            
            if parent_text and parent_text not in parent_texts_seen:
                parent_texts_seen.add(parent_text)
                parent_docs.append(parent_text)
                
                if len(parent_docs) >= top_k:
                    break
                    
        return parent_docs