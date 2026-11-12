import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hr_rag.pipeline import HRPolicyRAGPipeline
from hr_rag.retrievers.hybrid import (
    build_hybrid_index,
    hybrid_retrieve,
    reciprocal_rank_fuse,
)


class FakeDoc:
    def __init__(self, category, text):
        self.metadata = {"category": category}
        self.page_content = text


class FakeCollection:
    """Stands in for the real QdrantCategoryStore (the dense side of hybrid)
    so the category-routing logic in HRPolicyRAGPipeline.retrieve() can be
    tested without downloading an embedding model."""

    def __init__(self, docs):
        self._docs = docs

    def invoke(self, question, limit=3, metadata_filter=None):
        return self._docs


def make_pipeline(collections: dict):
    pipeline = HRPolicyRAGPipeline.__new__(HRPolicyRAGPipeline)
    pipeline.top_k = 3
    pipeline.candidate_k = 10
    pipeline.collections = collections
    pipeline.bm25_index = None
    return pipeline


class TestCategoryRouting:
    def test_explicit_category_only_searches_that_collection(self):
        leave_docs = [FakeDoc("leave", f"leave chunk {i}") for i in range(5)]
        comp_docs = [FakeDoc("compensation", f"comp chunk {i}") for i in range(5)]
        pipeline = make_pipeline({
            "leave": FakeCollection(leave_docs),
            "compensation": FakeCollection(comp_docs),
        })

        results = pipeline.retrieve("what is the leave policy", category="leave")
        assert all(d.metadata["category"] == "leave" for d in results)

    def test_rbac_scoping_excludes_disallowed_categories(self):
        leave_docs = [FakeDoc("leave", f"leave chunk {i}") for i in range(5)]
        comp_docs = [FakeDoc("compensation", f"comp chunk {i}") for i in range(5)]
        pipeline = make_pipeline({
            "leave": FakeCollection(leave_docs),
            "compensation": FakeCollection(comp_docs),
        })

        # an "employee" role scoped to leave only, asking a compensation question
        results = pipeline.retrieve("what is the PF contribution", allowed_categories=["leave"])
        assert all(d.metadata["category"] == "leave" for d in results)
        assert not any(d.metadata["category"] == "compensation" for d in results)

    def test_category_outside_allowed_returns_nothing(self):
        comp_docs = [FakeDoc("compensation", "comp chunk")]
        pipeline = make_pipeline({"compensation": FakeCollection(comp_docs)})

        # requested category isn't in the collections at all
        results = pipeline.retrieve("question", category="nonexistent")
        assert results == []

    def test_cross_category_question_pools_from_all_allowed(self):
        leave_docs = [FakeDoc("leave", f"leave {i}") for i in range(3)]
        conduct_docs = [FakeDoc("conduct", f"conduct {i}") for i in range(3)]
        pipeline = make_pipeline({
            "leave": FakeCollection(leave_docs),
            "conduct": FakeCollection(conduct_docs),
        })

        results = pipeline.retrieve("cross policy question", allowed_categories=["leave", "conduct"])
        categories_seen = {d.metadata["category"] for d in results}
        # both allowed categories should be represented, not just one crowding out the other
        assert categories_seen == {"leave", "conduct"}

    def test_result_never_exceeds_top_k(self):
        docs = [FakeDoc("leave", f"chunk {i}") for i in range(10)]
        pipeline = make_pipeline({"leave": FakeCollection(docs)})

        results = pipeline.retrieve("question", category="leave")
        assert len(results) <= pipeline.top_k

    def test_no_collections_match_returns_empty(self):
        pipeline = make_pipeline({})
        assert pipeline.retrieve("anything") == []

    def test_pf_keyword_routes_only_to_compensation(self):
        comp_docs = [FakeDoc("compensation", "PF contribution rate")]
        finance_docs = [FakeDoc("finance", "travel policy")]
        it_docs = [FakeDoc("it", "laptop policy")]
        pipeline = make_pipeline(
            {
                "compensation": FakeCollection(comp_docs),
                "finance": FakeCollection(finance_docs),
                "it": FakeCollection(it_docs),
            }
        )
        results = pipeline.retrieve("what is the PF contribution rate", allowed_categories=["compensation", "finance", "it"])
        assert all(d.metadata["category"] == "compensation" for d in results)

    def test_finance_keyword_routes_only_to_finance(self):
        finance_docs = [FakeDoc("finance", "expense policy")]
        it_docs = [FakeDoc("it", "laptop policy")]
        pipeline = make_pipeline({"finance": FakeCollection(finance_docs), "it": FakeCollection(it_docs)})
        results = pipeline.retrieve("what is the travel reimbursement limit", allowed_categories=["finance", "it"])
        assert all(d.metadata["category"] == "finance" for d in results)


def _lc_doc(category: str, text: str) -> Document:
    return Document(page_content=text, metadata={"category": category})


class TestReciprocalRankFuse:
    def test_doc_in_both_lists_outranks_single_list_docs(self):
        shared = _lc_doc("leave", "casual leave entitlement 12 days")
        dense_only = _lc_doc("leave", "unrelated policy sentence")
        bm25_only = _lc_doc("leave", "another unrelated policy sentence")
        fused = reciprocal_rank_fuse([shared, dense_only], [bm25_only, shared], top_k=3)
        assert fused[0] is shared

    def test_top_k_caps_output(self):
        docs = [_lc_doc("leave", f"chunk {i}") for i in range(6)]
        fused = reciprocal_rank_fuse(docs[:5], docs[1:], top_k=3)
        assert len(fused) == 3

    def test_full_bm25_weight_prefers_keyword_order(self):
        first = _lc_doc("leave", "casual leave 12 days per year")
        second = _lc_doc("leave", "leave application process portal")
        fused = reciprocal_rank_fuse([second, first], [first, second], top_k=2, bm25_weight=1.0)
        assert fused[0] is first


class TestHybridRetrieve:
    def _category(self, name: str, texts: list[str]) -> list[Document]:
        return [_lc_doc(name, text) for text in texts]

    def test_none_index_falls_back_to_dense(self):
        dense = self._category("leave", ["a", "b", "c", "d"])
        out = hybrid_retrieve("q", dense, None, "leave", top_k=3, candidate_k=5)
        assert out == dense[:3]
        # index exists but has nothing for this category -> dense-only too
        only_finance = build_hybrid_index(self._category("finance", ["x"]))
        out = hybrid_retrieve("q", dense, only_finance, "leave", top_k=3, candidate_k=5)
        assert out == dense[:3]

    def test_other_category_chunks_never_leak_in(self):
        leave_chunks = self._category("leave", ["casual leave 12 days per year"])
        finance_chunks = self._category("finance", ["travel reimbursement 50000 rupees"])
        index = build_hybrid_index(leave_chunks + finance_chunks)
        dense = self._category("leave", ["casual leave application portal"])
        out = hybrid_retrieve("travel reimbursement limits", dense,
                              index, "leave", top_k=3, candidate_k=5)
        # the leave index contributes its lexical chunk even though dense
        # never returned it...
        assert any(d.page_content == leave_chunks[0].page_content for d in out)
        # ...while finance's index — which holds the best lexical match for
        # this very query — is never consulted for a leave question.
        assert all(d.metadata["category"] == "leave" for d in out)
        assert not any(d.page_content == finance_chunks[0].page_content for d in out)

    def test_keyword_hit_dense_missed_is_added(self):
        # Dense top-2 misses the exact-match chunk entirely; fusion must
        # surface it from the BM25 side (the recall gain hybrid is for).
        good = _lc_doc("leave", "casual leave entitlement is 12 days per calendar year")
        bad1 = _lc_doc("leave", "leave application process uses the hrms portal")
        bad2 = _lc_doc("leave", "public holidays are declared each january")
        index = build_hybrid_index([good, bad1, bad2])
        dense = [bad1, bad2]  # ranked list from the vector store, no `good`
        out = hybrid_retrieve("casual leave entitlement 12 days", dense,
                              index, "leave", top_k=3, candidate_k=5, bm25_weight=0.5)
        assert any(d.page_content == good.page_content for d in out)
        # consensus (both lists) outranks a single-list hit
        assert out[0] is bad1

    def test_retriever_copies_of_same_chunk_merge_into_one_hit(self):
        # Production reality: Qdrant and BM25 each hand back their own
        # Document copies of a chunk. Fusion must count them as ONE hit —
        # keying on object identity (id()) split them in two, and behind
        # the old id()-based category filter it dropped every keyword hit
        # entirely, silently degrading hybrid to dense-only.
        chunk = _lc_doc("leave", "casual leave 12 days per year")
        index = build_hybrid_index([chunk])
        dense_copy = Document(page_content=chunk.page_content,
                              metadata=dict(chunk.metadata))
        out = hybrid_retrieve("casual leave", [dense_copy], index, "leave",
                              top_k=3, candidate_k=5)
        assert [d.page_content for d in out] == [chunk.page_content]


class TestPipelineHybridWiring:
    def _pipeline_with_index(self, collections: dict, chunks_by_cat: dict):
        pipeline = make_pipeline(collections)
        flat = [doc for docs in chunks_by_cat.values() for doc in docs]
        pipeline.bm25_index = build_hybrid_index(flat)
        return pipeline

    def test_pipeline_retrieve_stays_rbac_scoped_with_hybrid(self):
        leave_docs = [FakeDoc("leave", "casual leave 12 days")]
        finance_docs = [FakeDoc("finance", "travel budget")]
        pipeline = self._pipeline_with_index(
            {"leave": FakeCollection(leave_docs), "finance": FakeCollection(finance_docs)},
            {"leave": [_lc_doc("leave", "casual leave 12 days")],
             "finance": [_lc_doc("finance", "travel budget")]},
        )
        results = pipeline.retrieve("travel budget", allowed_categories=["leave"])
        assert results  # router falls back to the allowed list when no keyword hits
        assert all(d.metadata["category"] == "leave" for d in results)
        # the finance index holds an exact match for "travel budget" but is
        # out of RBAC scope — it must never surface
        assert not any(d.page_content == "travel budget" for d in results)

    def test_pipeline_retrieve_without_index_behaves_as_before(self):
        leave_docs = [FakeDoc("leave", f"leave {i}") for i in range(5)]
        pipeline = make_pipeline({"leave": FakeCollection(leave_docs)})
        results = pipeline.retrieve("question", category="leave")
        assert len(results) == 3

