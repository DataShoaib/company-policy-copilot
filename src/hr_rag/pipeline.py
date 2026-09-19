import re
from collections import Counter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from hr_rag.config import CATEGORIES, DEFAULT_TOP_K, RERANK_CANDIDATE_K
from hr_rag.formatting import format_docs
from hr_rag.llm import get_llm
from hr_rag.prompts import RAG_ANSWER_PROMPT
from hr_rag.qdrant_store import QdrantCategoryStore
from hr_rag.retrievers.router import route_question

_pipeline_singleton = None


class HRPolicyRAGPipeline:
    def __init__(self, category_stores: dict[str, QdrantCategoryStore],
                 top_k: int = DEFAULT_TOP_K, candidate_k: int = RERANK_CANDIDATE_K):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.collections = category_stores

        self._answer_chain = RAG_ANSWER_PROMPT | get_llm() | StrOutputParser()

    def retrieve(self, question: str, category: str | None = None, allowed_categories: list[str] | None = None, metadata_filter: dict | None = None) -> list[Document]:
        if category:
            if allowed_categories is not None and category not in allowed_categories:
                return []
            search_categories = [category]
        elif allowed_categories is not None:
            if not allowed_categories:
                return []
            search_categories = route_question(question, allowed_categories)
        else:
            search_categories = list(self.collections.keys())

        search_categories = [c for c in search_categories if c in self.collections]
        if not search_categories:
            return []

        if len(search_categories) == 1:
            return self.collections[search_categories[0]].invoke(question, self.candidate_k, metadata_filter)[: self.top_k]

        per_collection = [self.collections[c].invoke(question, self.candidate_k, metadata_filter)[: self.top_k] for c in search_categories]
        merged, i = [], 0
        while len(merged) < self.top_k and any(i < len(docs) for docs in per_collection):
            for docs in per_collection:
                if i < len(docs) and len(merged) < self.top_k:
                    merged.append(docs[i])
            i += 1
        return merged

    def answer(self, question: str, category: str | None = None, allowed_categories: list[str] | None = None) -> tuple[str, list[Document]]:
        docs = self.retrieve(question, category=category, allowed_categories=allowed_categories)
        return self.answer_from_documents(question, docs)

    def answer_from_documents(self, question: str, docs: list[Document]) -> tuple[str, list[Document]]:
        if not docs:
            return (
                "I don't have information on that in the policy documents I can access for your role. Please check with HR directly.",
                [],
            )
        answer = self._answer_chain.invoke({"context": format_docs(docs), "question": question})
        return _clean_answer(answer), docs

def _clean_answer(answer: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    answer = re.sub(r"<reasoning>.*?</reasoning>", "", answer, flags=re.DOTALL | re.IGNORECASE)
    return answer.strip()


def pipeline_is_loaded() -> bool:
    return _pipeline_singleton is not None


def get_pipeline() -> HRPolicyRAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        from hr_rag.chunking import chunk_documents
        from hr_rag.data_loading import load_policy_documents
        docs = load_policy_documents()
        chunks = chunk_documents(docs)

        from hr_rag.qdrant_store import (
            build_category_collections,
            load_category_collections,
        )

        category_stores = load_category_collections()
        expected = Counter(chunk.metadata.get("category") for chunk in chunks)
        if set(category_stores) != set(CATEGORIES) or any(
            category_stores[category].count() != expected[category] for category in CATEGORIES
        ):
            category_stores = build_category_collections(chunks, force=True)
        _pipeline_singleton = HRPolicyRAGPipeline(category_stores)
    return _pipeline_singleton
