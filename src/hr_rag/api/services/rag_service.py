"""Request orchestration for the policy Q&A endpoint.

Every request runs inside a single traced LangSmith span (``hr-rag-request``),
so the guardrail chains and the answer chain are recorded as children of one
trace instead of as unrelated root traces.
"""

import time
from functools import partial

from langchain_core.runnables import RunnableLambda

from hr_rag.api.core.rbac import allowed_categories_for_role
from hr_rag.api.services.cache import (
    cached_answer_is_allowed,
    get_cached_answer,
    scope_fingerprint,
    set_cached_answer,
)
from hr_rag.api.services.guardrails import (
    check_input_guardrails,
    check_input_guardrails_llm,
    check_output_guardrails,
    check_output_guardrails_llm,
)
from hr_rag.pipeline import get_pipeline
from hr_rag.retrievers.router import route_question

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

NO_ACCESS_MESSAGE = "I don't have access to that policy category for your role. Please check with HR directly."


def _elapsed_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def _run_summary(outcome: str, sources: list[dict], cached: bool, latency_ms: int, reason: str | None = None) -> dict:
    """Trace output for one request.

    Retrieved documents are not JSON serialisable, so only the ids of the
    documents that backed the answer are reported.
    """
    summary = {
        "outcome": outcome,
        "cached": cached,
        "latency_ms": latency_ms,
        "source_count": len(sources),
        "source_documents": [source.get("policy_doc_id", "unknown") for source in sources],
        "source_categories": sorted({source.get("category", "unknown") for source in sources}),
    }
    if reason:
        summary["reason"] = reason
    return summary


def _retrieve_span(holder: dict, payload: dict) -> dict:
    """Traced retrieval step.

    The Document objects are parked on ``holder`` instead of being returned: the
    tracer serialises a run's payload and documents are not JSON safe.
    """
    docs = get_pipeline().retrieve(
        payload["question"],
        category=payload["category"],
        allowed_categories=payload["allowed_categories"],
    )
    holder["docs"] = docs
    return {
        "document_count": len(docs),
        "documents": [doc.metadata.get("policy_doc_id", "unknown") for doc in docs],
        "categories": sorted({doc.metadata.get("category", "unknown") for doc in docs}),
    }


def _generate_span(holder: dict, payload: dict) -> dict:
    """Traced generation step over the retrieved context."""
    answer, docs = get_pipeline().answer_from_documents(payload["question"], holder["docs"])
    holder["answer"] = answer
    holder["docs"] = docs
    return {"answer_chars": len(answer), "grounded_documents": len(docs)}


def _answer_with_retry(question: str, category: str | None, allowed_categories: list[str]):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            outcome: dict = {}
            RunnableLambda(partial(_retrieve_span, outcome), name="policy-retrieval").invoke(
                {"question": question, "category": category, "allowed_categories": allowed_categories}
            )
            RunnableLambda(partial(_generate_span, outcome), name="answer-generation").invoke(
                {"question": question, "document_count": len(outcome["docs"])}
            )
            return outcome["answer"], outcome["docs"]
        except Exception as e:  # noqa: BLE001 - retry any transient provider failure
            last_error = e
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(f"pipeline failed after {MAX_RETRIES + 1} attempts: {last_error}") from last_error


def _answer_request(payload: dict) -> dict:
    question = payload["question"]
    role = payload["role"]
    category = payload["category"]
    start = time.time()

    allowed_categories = allowed_categories_for_role(role)
    scope = scope_fingerprint(allowed_categories)

    blocked = check_input_guardrails(question)
    if blocked:
        latency_ms = _elapsed_ms(start)
        payload["result"] = (blocked["detail"], [], False, latency_ms)
        return _run_summary("input_blocked", [], False, latency_ms, blocked["reason"])

    cached = get_cached_answer(question, scope, category)
    if cached is not None and cached_answer_is_allowed(cached, allowed_categories, category):
        latency_ms = _elapsed_ms(start)
        payload["result"] = (cached["answer"], cached["sources"], True, latency_ms)
        return _run_summary("cache_hit", cached["sources"], True, latency_ms)

    llm_blocked = check_input_guardrails_llm(question)
    if llm_blocked:
        latency_ms = _elapsed_ms(start)
        payload["result"] = (llm_blocked["detail"], [], False, latency_ms)
        return _run_summary("input_blocked_llm", [], False, latency_ms, llm_blocked["reason"])

    routed_categories = route_question(question, allowed_categories, category)
    if not routed_categories:
        latency_ms = _elapsed_ms(start)
        payload["result"] = (NO_ACCESS_MESSAGE, [], False, latency_ms)
        return _run_summary("category_not_allowed", [], False, latency_ms, "rbac")

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

    context_chunks = [d.page_content for d in docs]
    output_blocked = check_output_guardrails(answer, context_chunks)
    if output_blocked:
        latency_ms = _elapsed_ms(start)
        payload["result"] = (output_blocked["detail"], [], False, latency_ms)
        return _run_summary("output_blocked", [], False, latency_ms, output_blocked["reason"])

    output_blocked_llm = check_output_guardrails_llm(answer, context_chunks)
    if output_blocked_llm:
        latency_ms = _elapsed_ms(start)
        payload["result"] = (output_blocked_llm["detail"], [], False, latency_ms)
        return _run_summary("output_blocked_llm", [], False, latency_ms, output_blocked_llm["reason"])

    if sources:
        set_cached_answer(question, answer, sources, scope, category)

    latency_ms = _elapsed_ms(start)
    payload["result"] = (answer, sources, False, latency_ms)
    return _run_summary("answered", sources, False, latency_ms)


_TRACED_REQUEST = RunnableLambda(_answer_request, name="hr-rag-request")


def answer_question(question: str, role: str, category: str | None = None) -> tuple[str, list[dict], bool, int]:
    payload = {"question": question, "role": role, "category": category, "result": None}
    allowed_categories = allowed_categories_for_role(role)

    _TRACED_REQUEST.invoke(
        payload,
        config={
            "run_name": "hr-rag-request",
            "tags": ["hr-rag", "api", f"role:{role}"],
            "metadata": {
                "question": question[:500],
                "role": role,
                "category": category or "(auto-routed)",
                "allowed_categories": ", ".join(allowed_categories),
            },
        },
    )
    return payload["result"]
