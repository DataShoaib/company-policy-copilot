from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from hr_rag.config import DEFAULT_TOP_K


def build_hybrid_ensemble(chunks: list[Document], db: VectorStore, k: int = DEFAULT_TOP_K, bm25_weight: float = 0.5) -> EnsembleRetriever:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    dense = db.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[bm25_weight, 1 - bm25_weight])


def get_hybrid_retriever(db: VectorStore, chunks: list[Document], k: int = DEFAULT_TOP_K, bm25_weight: float = 0.5):
    return build_hybrid_ensemble(chunks, db, k=k, bm25_weight=bm25_weight)
