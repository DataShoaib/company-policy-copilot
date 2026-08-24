from fastapi import APIRouter, Depends, HTTPException, status

from hr_rag.api.core.deps import get_current_user
from hr_rag.api.schemas.schemas import QueryRequest, QueryResponse, SourceChunk
from hr_rag.api.services.rag_service import answer_question
from hr_rag.api.services.rate_limit import (
    RateLimitExceeded,
    RateLimitServiceUnavailable,
    check_rate_limit,
)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask_policy_question(body: QueryRequest, user: dict = Depends(get_current_user)):  # noqa: B008
    try:
        check_rate_limit(user_id=user["username"])
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {e.retry_after_seconds} seconds.",
            headers={"Retry-After": str(e.retry_after_seconds)},
        )
    except RateLimitServiceUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Query service is temporarily unavailable because rate limiting is offline. Start Redis and try again.") from exc

    try:
        answer, sources, cached, latency_ms = answer_question(
            question=body.question,
            role=user["role"],
            category=body.category,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The answer provider is temporarily unavailable. Check the configured GROQ_MODEL and try again.",
        ) from exc

    return QueryResponse(
        answer=answer,
        sources=[SourceChunk(**s) for s in sources],
        cached=cached,
        latency_ms=latency_ms,
    )
