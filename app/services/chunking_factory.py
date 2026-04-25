import os
import glob
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class BaseChunker:
    """Base strategy for document chunking."""
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError

class RecursiveChunker(BaseChunker):
    """Splits text by characters, recursively trying to keep paragraphs/sentences together."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

class TokenChunker(BaseChunker):
    """Splits text strictly by LLM tokens (using tiktoken/OpenAI encodings)."""
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int) -> BaseChunker:
    """Factory to retrieve the desired chunking strategy."""
    strategies = {
        "recursive": RecursiveChunker,
        "token": TokenChunker
    }
    
    if strategy.lower() not in strategies:
        raise ValueError(f"Unsupported chunking strategy: {strategy}. Use 'recursive' or 'token'.")
        
    return strategies[strategy.lower()](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def extract_and_chunk_directory(directory_path: str, chunker: BaseChunker) -> List[str]:
    """Reads all PDFs in a directory recursively and applies the chunker."""
    pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)
    all_chunks = []

    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            # Combine pages into a single document string before chunking
            full_text = "\n".join([page.page_content for page in pages])
            
            if full_text.strip():
                chunks = chunker.chunk(full_text)
                all_chunks.extend(chunks)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    return all_chunks