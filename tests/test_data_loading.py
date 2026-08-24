import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hr_rag.chunking import chunk_documents, group_by_category
from hr_rag.data_loading import load_policy_documents


class TestDataLoading:
    def test_loads_all_policy_docs(self):
        assert len(load_policy_documents()) == 9

    def test_every_doc_has_category(self):
        docs = load_policy_documents()
        categories = {d.metadata["category"] for d in docs}
        assert categories == {"leave", "compensation", "conduct", "performance", "recruitment", "finance", "it", "legal", "operations"}

    def test_no_doc_has_unknown_category(self):
        docs = load_policy_documents()
        assert all(d.metadata["category"] != "unknown" for d in docs)

    def test_policy_doc_ids_extracted(self):
        docs = load_policy_documents()
        doc_ids = {d.metadata["policy_doc_id"] for d in docs}
        assert doc_ids == {"HRP-001", "HRP-002", "HRP-003", "HRP-004", "HRP-005", "FIN-001", "IT-001", "LEG-001", "OPS-001"}


class TestChunking:
    def test_chunking_produces_multiple_chunks_per_doc(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        assert len(chunks) > len(docs)

    def test_chunks_inherit_source_metadata(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        assert all("category" in c.metadata for c in chunks)
        assert all("policy_doc_id" in c.metadata for c in chunks)

    def test_chunks_have_sequential_chunk_id(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))

    def test_no_chunk_wildly_exceeds_configured_size(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        assert all(len(c.page_content) < 1500 for c in chunks)


class TestGroupByCategory:
    def test_every_chunk_lands_in_exactly_one_category(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        grouped = group_by_category(chunks)
        assert sum(len(v) for v in grouped.values()) == len(chunks)

    def test_nine_category_slices_present(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        grouped = group_by_category(chunks)
        assert set(grouped.keys()) == {"leave", "compensation", "conduct", "performance", "recruitment", "finance", "it", "legal", "operations"}

    def test_no_category_is_empty(self):
        docs = load_policy_documents()
        chunks = chunk_documents(docs)
        grouped = group_by_category(chunks)
        assert all(len(v) > 0 for v in grouped.values())
