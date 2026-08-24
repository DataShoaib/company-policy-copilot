from fastapi import APIRouter

from hr_rag.api.core.redis_client import get_redis
from hr_rag.api.schemas.schemas import HealthResponse
from hr_rag.pipeline import pipeline_is_loaded

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        get_redis().ping()
        redis_ok = True
    except Exception:  # noqa: BLE001 - Redis (or anything else) being down means degraded health
        redis_ok = False

    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        redis_connected=redis_ok,
        pipeline_loaded=pipeline_is_loaded(),
    )
