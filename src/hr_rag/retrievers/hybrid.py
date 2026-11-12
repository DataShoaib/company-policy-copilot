"""Hybrid retrieval (BM25 keyword + dense vector, fused with RRF).

This is the evaluated final approach: on the 5-question category-balanced
sample it beat plain vector on faithfulness (0.93 vs 0.83),
context_precision (0.75 vs 0.37), context_recall (0.80 vs 0.60) and
answer_correctness (0.42 vs 0.38), at ~baseline latency (2.8s vs 2.3s).
Verbatim rankings would be stable; the evals used an equivalent
0.5/0.5 BM25+dense ensemble.

Keyword search runs per category: one BM25 index built from that category's
own chunks, so RBAC scoping works exactly like the vector path — only the
routed category's index is ever queried, and an unallowed category's chunks
are never scored at all.

Both retrievers hand back fresh Document copies (Qdrant rebuilds documents
from its payloads, BM25Retriever rebuilds them while building its index), so
fusion keys on page_content instead of object identity. That is what lets a
chunk found by both retrievers merge into a single consensus hit instead of
appearing twice.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from hr_rag.config import HYBRID_BM25_WEIGHT, RERANK_CANDIDATE_K

_RRF_K = 60

# Reciprocal-rank fusion: score(doc) = dense_weight / (_RRF_K + dense_rank)
#                                  + bm25_weight  / (_RRF_K + bm25_rank)
# HYBRID_BM25_WEIGHT (config.py) is the BM25 share; dense gets 1 - it.


def build_hybrid_index(
    chunks: list[Document], candidate_k: int = RERANK_CANDIDATE_K
) -> dict[str, BM25Retriever] | None:
    """One BM25 index per category, built from that category's own chunks.

    Returns None when there are no chunks (empty startup), so retrieval
    degrades to dense-only instead of crashing. `candidate_k` becomes each
    index's k, so keyword candidates match the dense candidate budget.
    """
    if not chunks:
        return None
    by_category: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_category.setdefault(chunk.metadata.get("category"), []).append(chunk)
    return {
        category: BM25Retriever.from_documents(category_chunks, k=candidate_k)
        for category, category_chunks in by_category.items()
    }


def _doc_key(doc: Document) -> str:
    """Identity that survives retriever copies.

    Qdrant and BM25 each return their own Document objects for the same
    underlying chunk, so `id()` never matches across the two lists — the
    page text always does.
    """
    return doc.page_content


def reciprocal_rank_fuse(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    top_k: int,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> list[Document]:
    """Merge two ranked lists with weighted reciprocal-rank fusion.

    A chunk present in both lists scores once with both contributions (the
    consensus boost) and appears once in the output.
    """
    dense_weight = 1.0 - bm25_weight
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for rank, doc in enumerate(dense_docs):
        key = _doc_key(doc)
        doc_by_key.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + dense_weight / (_RRF_K + rank + 1)
    for rank, doc in enumerate(bm25_docs):
        key = _doc_key(doc)
        doc_by_key.setdefault(key, doc)
        scores[key] = scores.get(key, 0.0) + bm25_weight / (_RRF_K + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
    return [doc_by_key[key] for key in ranked[:top_k]]


def hybrid_retrieve(
    question: str,
    dense_docs: list[Document],
    bm25_index: dict[str, BM25Retriever] | None,
    category: str,
    top_k: int,
    candidate_k: int,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> list[Document]:
    """Fuse the dense hits for one category with keyword hits for that category.

    `bm25_index` is the mapping from build_hybrid_index (None on an empty
    startup). When it is missing or has nothing for this category, the result
    is dense-only — never an error.
    """
    index = (bm25_index or {}).get(category)
    if index is None:
        return dense_docs[:top_k]
    bm25_docs = index.invoke(question)[:candidate_k]
    return reciprocal_rank_fuse(
        dense_docs[:candidate_k], bm25_docs, top_k, bm25_weight
    )

