from __future__ import annotations
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


class TracingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id   = request.headers.get("X-Trace-ID",   str(uuid.uuid4()))
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.trace_id   = trace_id
        request.state.request_id = request_id

        t0          = time.perf_counter()
        response    = None
        status_code = 500

        try:
            response    = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            logger.error("trace_id=%s error: %s", trace_id, exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            if response is not None:
                response.headers["X-Trace-ID"]   = trace_id
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"

        return response
