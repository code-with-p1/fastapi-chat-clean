from __future__ import annotations
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import get_settings
from app.models import HealthResponse
from app.services import redis_service

router   = APIRouter(tags=["health"])
logger   = logging.getLogger(__name__)
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)


@router.get("/health/ready")
async def health_ready() -> JSONResponse:
    checks:  dict[str, str] = {}
    overall = "ok"

    try:
        await redis_service.get_redis().ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"
        overall = "degraded"

    return JSONResponse(
        {"status": overall, "version": settings.app_version, "checks": checks},
        status_code=200 if overall == "ok" else 503,
    )
