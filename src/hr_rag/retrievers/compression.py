from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.vectorstores import VectorStore

from hr_rag.config import DEFAULT_TOP_K
from hr_rag.llm import get_llm


def get_compression_retriever(db: VectorStore, k: int = DEFAULT_TOP_K):
    base = db.as_retriever(search_kwargs={"k": k})
    compressor = LLMChainExtractor.from_llm(get_llm())
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
