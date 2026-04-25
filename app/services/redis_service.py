from __future__ import annotations
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

_client: Optional[aioredis.Redis] = None


async def init_redis() -> None:
    global _client
    pool = aioredis.ConnectionPool.from_url(
        settings.redis_url, max_connections=50, decode_responses=True
    )
    _client = aioredis.Redis(connection_pool=pool)
    await _client.ping()
    logger.info("✅ Redis connected at %s:%s", settings.redis_host, settings.redis_port)


async def close_redis() -> None:
    if _client:
        await _client.aclose()


def get_redis() -> aioredis.Redis:
    if _client is None:
        raise RuntimeError("Redis not initialised")
    return _client


# ── Session history ───────────────────────────────────────────────────────────

async def save_session(session_id: str, messages: list[dict]) -> None:
    await get_redis().setex(f"chat:session:{session_id}", 3600, json.dumps(messages))


async def load_session(session_id: str) -> list[dict]:
    raw = await get_redis().get(f"chat:session:{session_id}")
    return json.loads(raw) if raw else []

async def set_value(key: str, value: str) -> None:
    """Stores a generic string value in Redis."""
    await get_redis().set(key, value)

async def get_value(key: str) -> Optional[str]:
    """Retrieves a generic string value from Redis."""
    return await get_redis().get(key)