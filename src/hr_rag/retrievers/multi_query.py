from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

from hr_rag.config import DEFAULT_TOP_K
from hr_rag.llm import get_llm


def get_multi_query_retriever(db: FAISS, k: int = DEFAULT_TOP_K):
    base = db.as_retriever(search_kwargs={"k": k})
    return MultiQueryRetriever.from_llm(retriever=base, llm=get_llm())
