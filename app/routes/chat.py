from __future__ import annotations
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.models import ChatMessage, ChatRequest, MessageRole, SyncChatRequest, SyncChatResponse
from app.services import redis_service
from app.services.llm_service import stream_chat, sync_chat
from app.services.rag_service import fetch_rag_context # <-- New Import

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


async def _prepare_rag_system_prompt(body: ChatRequest | SyncChatRequest, current_system_prompt: str | None) -> str | None:
    """Helper to fetch RAG context and format the final system prompt."""
    if not getattr(body, "use_rag", False) or not body.db_provider:
        return current_system_prompt

    # Extract the actual query (the last message from the user)
    latest_query = next((m.content for m in reversed(body.messages) if m.role == MessageRole.user), None)
    
    if not latest_query:
        return current_system_prompt

    logger.info(f"Triggering RAG pipeline using {body.db_provider} for query: '{latest_query}'")
    context = await fetch_rag_context(
        query=latest_query,
        db_provider=body.db_provider,
        index_name=body.index_name,
        dimension=body.dimension,
        top_k=getattr(body, "top_k", 10),
        rerank_top_n=getattr(body, "rerank_top_n", 5)
    )

    if context:
        rag_instruction = (
            "\n\n--- RELEVANT CONTEXT ---\n"
            f"{context}\n\n"
            "Use the context above to inform your answer. If the context does not contain "
            "the answer, state that you don't have enough information."
        )
        return (current_system_prompt or "") + rag_instruction
    
    return current_system_prompt


# ── POST /chat/stream ─────────────────────────────────────────────────────────

@router.post("/stream", summary="Stream chat via SSE")
async def chat_stream(
    body:     ChatRequest,
    request:  Request,
    settings: Settings = Depends(get_settings),
) -> EventSourceResponse:

    request_id = str(uuid.uuid4())
    session_id = body.session_id or str(uuid.uuid4())

    async def _generator() -> AsyncGenerator[dict, None]:
        # Load session history
        history = await redis_service.load_session(session_id)
        history_msgs = [ChatMessage(**m) for m in history]
        full_messages = history_msgs + body.messages
        
        # Inject RAG context dynamically
        final_system_prompt = await _prepare_rag_system_prompt(body, body.system_prompt)

        accumulated = []

        try:
            async for token in stream_chat(
                messages=full_messages,
                system_prompt=final_system_prompt, # Pass the RAG-enhanced prompt
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                accumulated.append(token)
                yield {
                    "event": "token",
                    "data":  json.dumps({"request_id": request_id, "session_id": session_id, "data": token}),
                }

            # Save updated session
            updated = [m.model_dump() for m in full_messages]
            updated.append({"role": MessageRole.assistant, "content": "".join(accumulated)})
            await redis_service.save_session(session_id, updated)

            yield {
                "event": "done",
                "data":  json.dumps({"request_id": request_id, "session_id": session_id}),
            }

        except Exception as exc:
            logger.error("Stream error request_id=%s: %s", request_id, exc)
            yield {
                "event": "error",
                "data":  json.dumps({"request_id": request_id, "error": str(exc)}),
            }

    return EventSourceResponse(
        _generator(),
        headers={
            "X-Request-ID": request_id,
            "X-Session-ID": session_id,
            "Cache-Control": "no-cache",
        },
    )


# ── POST /chat/sync ───────────────────────────────────────────────────────────

@router.post("/sync", response_model=SyncChatResponse, summary="Sync deterministic chat")
async def chat_sync(
    body:     SyncChatRequest,
    request:  Request,
    settings: Settings = Depends(get_settings),
) -> SyncChatResponse:

    request_id = str(uuid.uuid4())
    session_id = body.session_id or str(uuid.uuid4())

    history      = await redis_service.load_session(session_id)
    history_msgs = [ChatMessage(**m) for m in history]
    full_messages = history_msgs + body.messages
    
    # Inject RAG context dynamically
    final_system_prompt = await _prepare_rag_system_prompt(body, body.system_prompt)

    t0 = time.perf_counter()
    text, meta = await sync_chat(
        messages=full_messages,
        system_prompt=final_system_prompt, # Pass the RAG-enhanced prompt
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Save session
    updated = [m.model_dump() for m in full_messages]
    updated.append({"role": MessageRole.assistant, "content": text})
    await redis_service.save_session(session_id, updated)

    return SyncChatResponse(
        request_id=request_id,
        session_id=session_id,
        message=ChatMessage(role=MessageRole.assistant, content=text),
        usage=meta.get("usage", {}),
        latency_ms=round(latency_ms, 2),
    )