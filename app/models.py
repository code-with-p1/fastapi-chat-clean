from __future__ import annotations
import re
import uuid
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, model_validator, Field
from typing import List

class MessageRole(str, Enum):
    system    = "system"
    user      = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role:    MessageRole = MessageRole.user
    content: str


class ChatRequest(BaseModel):
    messages:      list[ChatMessage] = Field(..., min_length=1)
    session_id:    Optional[str]     = None
    system_prompt: Optional[str]     = None
    temperature:   Optional[float]   = Field(None, ge=0.0, le=2.0)
    max_tokens:    Optional[int]     = Field(None, ge=1, le=8192)
    use_rag: bool = Field(default=False, description="Enable RAG context injection")
    db_provider: Optional[str] = Field(default="pinecone", description="pinecone, weaviate, qdrant, milvus")
    index_name: Optional[str] = Field(default=None, description="Index Name")
    dimension : Optional[int] = Field(default=1536, description="Dimension")
    top_k: int = Field(default=10, description="Initial hybrid search depth")
    rerank_top_n: int = Field(default=5, description="Final documents to keep after Cohere re-ranking")


class SyncChatRequest(BaseModel):
    messages:      list[ChatMessage] = Field(..., min_length=1)
    session_id:    Optional[str]     = None
    system_prompt: Optional[str]     = None
    temperature:   Optional[float]   = Field(None, ge=0.0, le=2.0)
    max_tokens:    Optional[int]     = Field(None, ge=1, le=8192)
    use_rag: bool = Field(default=False, description="Enable RAG context injection")
    db_provider: Optional[str] = Field(default="pinecone", description="pinecone, weaviate, qdrant, milvus")
    index_name: Optional[str] = Field(default=None, description="Index Name")
    dimension : Optional[int] = Field(default=1536, description="Dimension")
    top_k: int = Field(default=10, description="Initial hybrid search depth")
    rerank_top_n: int = Field(default=5, description="Final documents to keep after Cohere re-ranking")


class SyncChatResponse(BaseModel):
    request_id:    str
    session_id:    str
    message:       ChatMessage
    finish_reason: str   = "stop"
    usage:         dict  = {}
    latency_ms:    float = 0.0


class HealthResponse(BaseModel):
    status:  str
    version: str
    checks:  dict[str, str] = {}


class SetupIndexRequest(BaseModel):
    db_provider: str         # "pinecone", "weaviate", "qdrant", "milvus"
    index_name: str          # Name of the collection/index
    dimension: int = 1536    # Default to OpenAI text-embedding-3-small dimension
    recreate: bool = False   # Set to True to drop existing collections with the same name

    @model_validator(mode='after')
    def validate_index_name(self) -> 'SetupIndexRequest':
        provider = self.db_provider.lower()
        name = self.index_name

        if provider == "pinecone":
            # Pinecone: Lowercase alphanumeric and hyphens. Must start/end with alphanumeric. Max 45 chars.
            if not re.match(r"^[a-z0-9][a-z0-9-]{0,43}[a-z0-9]$|^[a-z0-9]$", name):
                raise ValueError(
                    "Pinecone index names must be 1-45 characters long, contain only lowercase letters, "
                    "numbers, and hyphens, and must start and end with an alphanumeric character."
                )

        elif provider == "weaviate":
            # Weaviate: Class/Collection names must start with a capital letter, followed by alphanumeric/underscores.
            if not re.match(r"^[A-Z][a-zA-Z0-9_]*$", name):
                raise ValueError(
                    "Weaviate collection names must start with a capital letter and contain only "
                    "alphanumeric characters and underscores."
                )

        elif provider == "qdrant":
            # Qdrant: Alphanumeric, hyphens, and underscores.
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValueError(
                    "Qdrant collection names must contain only alphanumeric characters, hyphens, and underscores."
                )

        elif provider == "milvus":
            # Milvus: Must start with a letter or underscore, followed by alphanumeric/underscores. Max 255 chars.
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]{0,254}$", name):
                raise ValueError(
                    "Milvus collection names must be under 255 characters, start with a letter or underscore, "
                    "and contain only alphanumeric characters and underscores."
                )
                
        else:
            raise ValueError(f"Unsupported db_provider: '{provider}'. Supported providers are: pinecone, weaviate, qdrant, milvus.")

        return self


class IngestRequest(BaseModel):
    db_provider: str  # "pinecone", "weaviate", "qdrant"
    index_name: str          # Name of the collection/index
    dimension: int = 1536    # Default to OpenAI text-embedding-3-small dimension
    documents: List[str]


class QueryRequest(BaseModel):
    db_provider: str
    query: str
    index_name: str          # Name of the collection/index
    dimension: int = 1536    # Default to OpenAI text-embedding-3-small dimension
    top_k: int = 10
    rerank_top_n: int = 5


class DirectoryIngestRequest(BaseModel):
    db_provider: str
    index_name: str
    dimension: int = 1536
    directory_path: str = Field(..., description="Absolute or relative path to the local directory containing PDFs")
    chunking_strategy: str = Field(default="recursive", description="'recursive', 'token', or 'semantic'")
    chunk_size: int = Field(default=1000, description="Size of each chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between consecutive chunks")
    semantic_threshold_type: str = Field(
        default="percentile", 
        description="Used only if strategy is 'semantic'. Options: 'percentile', 'standard_deviation', 'interquartile'"
    )

class ParentChildIngestRequest(BaseModel):
    db_provider: str
    index_name: str
    dimension: int = 1536
    text: str

class ParentChildDirectoryIngestRequest(BaseModel):
    db_provider: str
    index_name: str
    dimension: int = 1536
    directory_path: str = Field(..., description="Absolute or relative path to local PDF directory")