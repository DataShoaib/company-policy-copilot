
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from hr_rag.config import DEFAULT_TOP_K
from hr_rag.llm import get_llm
from hr_rag.prompts import HYDE_PROMPT


def hyde_retrieve(db: FAISS, question: str, k: int = DEFAULT_TOP_K) -> list[Document]:
    hyde_chain = HYDE_PROMPT | get_llm() | StrOutputParser()
    hypothetical = hyde_chain.invoke({"question": question})
    return db.as_retriever(search_kwargs={"k": k}).invoke(hypothetical)
