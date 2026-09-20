import time

from langchain_core.documents import Document
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
from hr_rag.formatting import format_docs
from hr_rag.pipeline import _clean_answer, get_pipeline
from hr_rag.retrievers.router import route_question

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5


# Module-level retrieval step — creates a visible LangChain span named "policy-retrieval"
_retrieval_step = RunnableLambda(
    lambda ctx: get_pipeline().retrieve(
        ctx["question"],
        category=ctx.get("category"),
        allowed_categories=ctx.get("allowed_categories"),
    ),
    name="policy-retrieval",
)


def _retrieve(question: str, category: str | None, allowed_categories: list[str]) -> list[Document]:
    """Retrieve docs with a child trace span named 'policy-retrieval'."""
    return _retrieval_step.invoke({
        "question": question,
        "category": category,
        "allowed_categories": allowed_categories,
    })


def _generate_with_retry(context: str, question: str) -> str:
    """Generate answer with retry (retrieval is not retried — local Qdrant call)."""
    pipeline = get_pipeline()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            raw = pipeline._answer_chain.invoke({"context": context, "question": question})
            return _clean_answer(raw)
        except Exception as e:  # noqa: BLE001 - retry any transient provider failure
            last_error = e
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise RuntimeError(f"generation failed after {MAX_RETRIES + 1} attempts: {last_error}") from last_error


def answer_question(question: str, role: str, category: str | None = None) -> tuple[str, list[dict], bool, int]:
    return _TRACED_QUESTION.invoke(
        {"question": question, "role": role, "category": category},
        config={
            "run_name": "hr-rag-request",
            "tags": ["hr-rag", "api", f"role:{role}"],
            "metadata": {
                "question": question[:500],
                "role": role,
                "category": category or "(auto-routed)",
            },
        },
    )


def _answer_question(payload: dict) -> tuple[str, list[dict], bool, int]:
    question = payload["question"]
    role = payload["role"]
    category = payload["category"]
    start = time.time()

    allowed_categories = allowed_categories_for_role(role)
    scope = scope_fingerprint(allowed_categories)

    blocked = check_input_guardrails(question)
    if blocked:
        return blocked["detail"], [], False, int((time.time() - start) * 1000)

    cached = get_cached_answer(question, scope, category)
    if cached is not None and cached_answer_is_allowed(cached, allowed_categories, category):
        return cached["answer"], cached["sources"], True, int((time.time() - start) * 1000)

    llm_blocked = check_input_guardrails_llm(question)
    if llm_blocked:
        return llm_blocked["detail"], [], False, int((time.time() - start) * 1000)

    routed_categories = route_question(question, allowed_categories, category)
    if not routed_categories:
        return "I don't have access to that policy category for your role. Please check with HR directly.", [], False, int((time.time() - start) * 1000)

    docs = _retrieve(question, routed_categories[0] if len(routed_categories) == 1 else None, routed_categories)
    if not docs:
        return "I don't have information on that in the policy documents I can access for your role. Please check with HR directly.", [], False, int((time.time() - start) * 1000)

    context = format_docs(docs)
    answer = _generate_with_retry(context, question)

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
        return output_blocked["detail"], [], False, int((time.time() - start) * 1000)

    refusal_markers = ("This is not covered in the policy documents.", "not have information on that")
    is_plain_refusal = any(m in answer for m in refusal_markers)
    output_blocked_llm = None if is_plain_refusal else check_output_guardrails_llm(answer, context_chunks)
    if output_blocked_llm:
        return output_blocked_llm["detail"], [], False, int((time.time() - start) * 1000)

    if sources:
        set_cached_answer(question, answer, sources, scope, category)

    return answer, sources, False, int((time.time() - start) * 1000)


_TRACED_QUESTION = RunnableLambda(_answer_question, name="hr-rag-request")
