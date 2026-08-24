from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from hr_rag.config import DEFAULT_TOP_K
from hr_rag.llm import get_llm
from hr_rag.prompts import QUERY_REWRITE_PROMPT


def rewrite_and_retrieve(db: FAISS, question: str, k: int = DEFAULT_TOP_K):
    rewrite_chain = QUERY_REWRITE_PROMPT | get_llm() | StrOutputParser()
    rewritten = rewrite_chain.invoke({"question": question})
    return db.as_retriever(search_kwargs={"k": k}).invoke(rewritten)
