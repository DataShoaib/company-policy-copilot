
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from hr_rag.config import DEFAULT_TOP_K


def build_hybrid_ensemble(chunks: list[Document], db: FAISS, k: int = DEFAULT_TOP_K, bm25_weight: float = 0.5) -> EnsembleRetriever:
    """BM25 + dense over a given (chunks, db) pair. Shared by the single-index
    experiment in the notebook and the per-category collections in the
    production pipeline -- same fusion logic either way, just scoped to a
    smaller chunk set when it's one category's collection."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    dense = db.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[bm25_weight, 1 - bm25_weight])


def get_hybrid_retriever(db: FAISS, chunks: list[Document], k: int = DEFAULT_TOP_K, bm25_weight: float = 0.5):
    return build_hybrid_ensemble(chunks, db, k=k, bm25_weight=bm25_weight)
