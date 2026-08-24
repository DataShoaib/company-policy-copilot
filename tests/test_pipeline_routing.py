import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hr_rag.pipeline import HRPolicyRAGPipeline


class FakeDoc:
    def __init__(self, category, text):
        self.metadata = {"category": category}
        self.page_content = text


class FakeCollection:
    """Stands in for the real EnsembleRetriever(BM25 + dense) so the
    category-routing logic in HRPolicyRAGPipeline.retrieve() can be tested
    without downloading an embedding model."""

    def __init__(self, docs):
        self._docs = docs

    def invoke(self, question, limit=3, metadata_filter=None):
        return self._docs


def make_pipeline(collections: dict):
    pipeline = HRPolicyRAGPipeline.__new__(HRPolicyRAGPipeline)
    pipeline.top_k = 3
    pipeline.candidate_k = 10
    pipeline.collections = collections
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
