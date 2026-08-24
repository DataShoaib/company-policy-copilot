import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from hr_rag.config import CATEGORIES, DEFAULT_TOP_K, RERANK_CANDIDATE_K
from hr_rag.formatting import format_docs
from hr_rag.llm import get_llm
from hr_rag.prompts import RAG_ANSWER_PROMPT
from hr_rag.qdrant_store import QdrantCategoryStore
from hr_rag.retrievers.router import route_question

_pipeline_singleton = None

# each policy category gets its own small Qdrant collection instead of
# one big index over all documents. a query never searches a wider space than
# it needs to -- explicit category -> just that collection, RBAC-scoped role
# -> only the collections it's allowed to see, cross-category question ->
# pooled across allowed collections. this is also what makes RBAC cheap: an
# "employee" role process never even touches the compensation collection.


class HRPolicyRAGPipeline:
    def __init__(self, category_stores: dict[str, QdrantCategoryStore], chunks_by_category: dict[str, list[Document]],
                 top_k: int = DEFAULT_TOP_K, candidate_k: int = RERANK_CANDIDATE_K):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.collections = category_stores

        self._llm = get_llm()
        self._answer_chain = RAG_ANSWER_PROMPT | self._llm | StrOutputParser()

    def retrieve(self, question: str, category: str | None = None, allowed_categories: list[str] | None = None, metadata_filter: dict | None = None) -> list[Document]:
        if category:
            if allowed_categories is not None and category not in allowed_categories:
                return []
            search_categories = [category]
        elif allowed_categories is not None:
            # an explicitly empty allow-list means "nothing is permitted" --
            # never fall back to all collections, that would bypass RBAC
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

        # cross-category question -- pull a few from each allowed collection
        # and interleave, so no single category can crowd out the others
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
    """Public accessor so the /health route doesn't need to poke at the private singleton."""
    return _pipeline_singleton is not None


def get_pipeline() -> HRPolicyRAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        from hr_rag.chunking import chunk_documents, group_by_category
        from hr_rag.data_loading import load_policy_documents
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        chunks_by_category = group_by_category(chunks)

        from hr_rag.qdrant_store import (
            build_category_collections,
            load_category_collections,
        )

        category_stores = load_category_collections()
        if len(category_stores) != len(CATEGORIES):
            category_stores = build_category_collections(chunks)
        _pipeline_singleton = HRPolicyRAGPipeline(category_stores, chunks_by_category)
    return _pipeline_singleton
