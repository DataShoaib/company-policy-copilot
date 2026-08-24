
from langchain_community.vectorstores import FAISS

from hr_rag.config import DEFAULT_TOP_K


def get_metadata_filtered_retriever(db: FAISS, category: str | None = None, k: int = DEFAULT_TOP_K):
    search_kwargs = {"k": k}
    if category:
        search_kwargs["filter"] = {"category": category}
    return db.as_retriever(search_kwargs=search_kwargs)
