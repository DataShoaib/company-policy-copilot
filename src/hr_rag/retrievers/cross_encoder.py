
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from hr_rag.config import DEFAULT_TOP_K, RERANK_CANDIDATE_K

_cross_encoder = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def cross_encoder_retrieve(db: FAISS, question: str, candidate_k: int = RERANK_CANDIDATE_K, top_k: int = DEFAULT_TOP_K) -> list[Document]:
    candidates = db.as_retriever(search_kwargs={"k": candidate_k}).invoke(question)
    pairs = [[question, d.page_content] for d in candidates]
    scores = _get_cross_encoder().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
