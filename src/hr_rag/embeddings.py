from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from hr_rag.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
