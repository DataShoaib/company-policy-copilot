from langchain_community.vectorstores import FAISS

from hr_rag.config import DEFAULT_TOP_K


def get_baseline_retriever(db: FAISS, k: int = DEFAULT_TOP_K):
    return db.as_retriever(search_kwargs={"k": k})
