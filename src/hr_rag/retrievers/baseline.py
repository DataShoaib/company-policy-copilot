from langchain_core.vectorstores import VectorStore

from hr_rag.config import DEFAULT_TOP_K


def get_baseline_retriever(db: VectorStore, k: int = DEFAULT_TOP_K):
    return db.as_retriever(search_kwargs={"k": k})
