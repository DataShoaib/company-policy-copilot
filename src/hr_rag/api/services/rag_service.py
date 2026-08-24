import time

from hr_rag.api.core.rbac import allowed_categories_for_role
from hr_rag.api.services.cache import (
    cached_answer_is_allowed,
    get_cached_answer,
    set_cached_answer,
)
from hr_rag.pipeline import get_pipeline
from hr_rag.retrievers.router import route_question

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


def _answer_with_retry(question: str, category: str | None, allowed_categories: list[str]):
    pipeline = get_pipeline()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return pipeline.answer(question, category=category, allowed_categories=allowed_categories)
        except Exception as e:  # noqa: BLE001 - retry transient provider and retrieval failures
            last_error = e
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(f"pipeline failed after {MAX_RETRIES + 1} attempts: {last_error}") from last_error


def answer_question(question: str, role: str, category: str | None = None) -> tuple[str, list[dict], bool, int]:
    start = time.time()

    allowed_categories = allowed_categories_for_role(role)
    cached = get_cached_answer(question, category)
    if cached is not None and cached_answer_is_allowed(cached, allowed_categories, category):
        return cached["answer"], cached["sources"], True, int((time.time() - start) * 1000)

    routed_categories = route_question(question, allowed_categories, category)
    if not routed_categories:
        return "I don't have access to that policy category for your role. Please check with HR directly.", [], False, int((time.time() - start) * 1000)
    answer, docs = _answer_with_retry(question, routed_categories[0] if len(routed_categories) == 1 else None, routed_categories)

    sources = [
        {
            "category": d.metadata.get("category", "unknown"),
            "policy_doc_id": d.metadata.get("policy_doc_id", "unknown"),
            "title": d.metadata.get("title", "unknown"),
            "snippet": d.page_content[:200].strip() + ("..." if len(d.page_content) > 200 else ""),
        }
        for d in docs
    ]

    # don't cache empty-source fallbacks: they're role/scope-specific dead ends,
    # and a user with broader access asking the same question deserves a real lookup
    if sources:
        set_cached_answer(question, answer, sources, category)
    return answer, sources, False, int((time.time() - start) * 1000)
