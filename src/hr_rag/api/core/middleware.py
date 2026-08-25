"""ASGI middleware: per-request correlation id + HTTP metrics.

``RequestContextMiddleware`` gives every request a request_id that the JSON
logger propagates via a contextvar, so logs across the whole call stack can
be correlated. It also counts every response by method/path/status for the
Prometheus ``/metrics`` endpoint.
"""
from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from hr_rag.api.core.logging import logger, new_request_id
from hr_rag.api.core.metrics import HTTP_REQUESTS


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = new_request_id()
        start = time.perf_counter()
        status_code = 0
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            HTTP_REQUESTS.labels(
                method=request.method,
                path=request.url.path,
                status=str(status_code),
            ).inc()
            logger.info(
                "http",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": round(elapsed, 1),
                    }
                },
            )