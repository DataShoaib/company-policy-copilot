import time

from hr_rag.api.core.metrics import (
    CACHE_HIT_RATIO,
    CACHE_LOOKUPS,
    QUERY_LATENCY,
    QUERY_REQUESTS,
    RBAC_DENIALS,
)
from hr_rag.api.services.guardrails import check_input_guardrails, check_output_guardrails

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

# small in-proc window so we can publish a cache hit-ratio gauge without a
# separate aggregator. Counters(Counter) keep cumulative totals; this ratio is
# just the last N lookups, which is what an ops dashboard cares about.
_RATIO_WINDOW: list[bool] = []
_RATIO_WINDOW_SIZE = 100


def _record_lookup(hit: bool) -> None:
    CACHE_LOOKUPS.labels(hit="hit" if hit else "miss").inc()
    _RATIO_WINDOW.append(hit)
    if len(_RATIO_WINDOW) > _RATIO_WINDOW_SIZE:
        del _RATIO_WINDOW[:- _RATIO_WINDOW_SIZE]
    if _RATIO_WINDOW:
        CACHE_HIT_RATIO.set(sum(_RATIO_WINDOW) / len(_RATIO_WINDOW))


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

    # ----- INPUT GUARDRAILS (before any LLM/retrieval cost) -----
    blocked = check_input_guardrails(question)
    if blocked:
        QUERY_REQUESTS.labels(outcome="input_blocked").inc()
        _record_lookup(hit=False)
        QUERY_LATENCY.observe(time.time() - start)
        return blocked["detail"], [], False, int((time.time() - start) * 1000)

    cached = get_cached_answer(question, category)
    if cached is not None and cached_answer_is_allowed(cached, allowed_categories, category):
        QUERY_REQUESTS.labels(outcome="cached").inc()
        _record_lookup(hit=True)
        QUERY_LATENCY.observe(time.time() - start)
        return cached["answer"], cached["sources"], True, int((time.time() - start) * 1000)
    _record_lookup(hit=False)

    routed_categories = route_question(question, allowed_categories, category)
    if not routed_categories:
        # RBAC refused the scope — count it so a mis-configuration shows as a
        # denial spike rather than silently returning the same canned message.
        RBAC_DENIALS.inc()
        QUERY_REQUESTS.labels(outcome="denied").inc()
        QUERY_LATENCY.observe(time.time() - start)
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

    # ----- OUTPUT GUARDRAILS (before the answer reaches the user) -----
    context_chunks = [d.page_content for d in docs]
    output_blocked = check_output_guardrails(answer, context_chunks)
    if output_blocked:
        QUERY_REQUESTS.labels(outcome="output_blocked").inc()
        _record_lookup(hit=False)
        QUERY_LATENCY.observe(time.time() - start)
        # do NOT cache blocked output — force a re-evaluation next time
        return output_blocked["detail"], [], False, int((time.time() - start) * 1000)

    # don't cache empty-source fallbacks: they're role/scope-specific dead ends,
    # and a user with broader access asking the same question deserves a real lookup
    if sources:
        set_cached_answer(question, answer, sources, category)

    QUERY_REQUESTS.labels(outcome="answered").inc()
    QUERY_LATENCY.observe(time.time() - start)
    return answer, sources, False, int((time.time() - start) * 1000)
