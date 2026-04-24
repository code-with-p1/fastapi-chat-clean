from __future__ import annotations
import logging
import time
from typing import AsyncGenerator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_classic.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from app.config import get_settings
from app.models import ChatMessage, MessageRole

logger   = logging.getLogger(__name__)
settings = get_settings()

DEFAULT_SYSTEM = (
    "You are a helpful, concise, and reliable AI assistant. "
    "Always respond in clear, well-structured language."
)


def _to_lc(messages: list[ChatMessage], system_prompt: str | None) -> list:
    out = [SystemMessage(content=system_prompt or DEFAULT_SYSTEM)]
    for m in messages:
        if m.role == MessageRole.user:
            out.append(HumanMessage(content=m.content))
        elif m.role == MessageRole.assistant:
            out.append(AIMessage(content=m.content))
        elif m.role == MessageRole.system:
            out.append(SystemMessage(content=m.content))
    print(f"\n\n Final : {out}")
    return out


# ── Streaming ─────────────────────────────────────────────────────────────────

async def stream_chat(
    messages:      list[ChatMessage],
    system_prompt: str | None  = None,
    temperature:   float | None = None,
    max_tokens:    int | None   = None,
) -> AsyncGenerator[str, None]:
    import asyncio

    callback = AsyncIteratorCallbackHandler()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        streaming=True,
        callbacks=[callback],
        timeout=settings.request_timeout_seconds,
    )

    async def _run():
        try:
            print(f"\n\n system_prompt : {system_prompt}")
            await llm.agenerate([_to_lc(messages, system_prompt)])
        except Exception as exc:
            logger.error("LLM stream error: %s", exc)
        finally:
            callback.done.set()

    task = asyncio.create_task(_run())
    try:
        async for token in callback.aiter():
            yield token
    finally:
        await task


# ── Sync / deterministic ──────────────────────────────────────────────────────

async def sync_chat(
    messages:      list[ChatMessage],
    system_prompt: str | None  = None,
    temperature:   float | None = None,
    max_tokens:    int | None   = None,
) -> tuple[str, dict]:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        streaming=False,
        timeout=settings.request_timeout_seconds,
    )

    t0     = time.perf_counter()
    result = await llm.agenerate([_to_lc(messages, system_prompt)])
    latency_ms = (time.perf_counter() - t0) * 1000

    generation = result.generations[0][0]
    text = (
        generation.message.content
        if hasattr(generation, "message")
        else generation.text
    )

    usage: dict = {}
    if result.llm_output and "token_usage" in result.llm_output:
        raw = result.llm_output["token_usage"]
        usage = {
            k: v for k, v in raw.items()
            if isinstance(v, (int, float))
        }

    logger.info("sync_chat done model=%s latency=%.1fms", settings.openai_model, latency_ms)
    return text, {"usage": usage, "latency_ms": round(latency_ms, 2)}
