"""Prometheus metrics for the API, exposed at ``/metrics``.

Tracked signals (the ones that actually matter for a RAG service):
- query volume / latency histogram (p95 visible in Prometheus/Grafana)
- cache hit ratio (cache that never hits is a sign the scope keys are too
  narrow or TTL too short)
- auth outcomes (login success vs failure, signup)
- RBAC denials, so a policy mis-configuration shows up as a spike instead of
  silently leaking.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

# --- core RAG ------------------------------------------------------------
QUERY_REQUESTS = Counter("hrrag_query_requests_total", "Total /query requests", ["outcome"])
QUERY_LATENCY = Histogram(
    "hrrag_query_latency_seconds",
    "End-to-end query latency (includes cache, retrieval and LLM)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
CACHE_LOOKUPS = Counter("hrrag_cache_lookups_total", "Cache lookups", ["hit"])
CACHE_HIT_RATIO = Gauge("hrrag_cache_hit_ratio", "1s-smoothed cache hit ratio")

# ------------------------------------------------------------------------- auth
AUTH_TOTAL = Counter("hrrag_auth_total", "Auth operations", ["kind", "outcome"])
RBAC_DENIALS = Counter("hrrag_rbac_denials_total", "Requests refused by RBAC")

# ------------------------------------------------------------------------- LLM
LLM_CALLS = Counter("hrrag_llm_calls_total", "LLM inference calls", ["provider", "result"])
LLM_FALLBACKS = Counter("hrrag_llm_fallbacks_total", "Primary LLM failed and fell back")

# ------------------------------------------------------------------------- http
HTTP_REQUESTS = Counter("hrrag_http_requests_total", "All HTTP requests", ["method", "path", "status"])


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)