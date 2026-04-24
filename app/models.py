from __future__ import annotations
import uuid
from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


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
