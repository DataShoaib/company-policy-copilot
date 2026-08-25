"""Structured JSON logging with per-request correlation ids.

Replaces the ad-hoc ``print()`` calls used at startup so the API emits
machine-parseable logs: every line is one JSON object carrying a request_id
that ties together the middleware, auth, retrieval and LLM steps of a single
request. Feed these straight into a log aggregator / SIEM.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger("hr_rag")

# Per-request context. Python 3.7+ contextvars are Task/thread-safe, which
# matters because FastAPI runs sync endpoints on a threadpool.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def new_request_id() -> str:
    """Assign a fresh id for the current context and return it."""
    rid = uuid4().hex[:12]
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_var.get()


class JsonFormatter(logging.Formatter):
    """Format a LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": _request_id_var.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install the JSON handler on the root logger exactly once."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
           for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers[:] = [handler]
    root.setLevel(level)