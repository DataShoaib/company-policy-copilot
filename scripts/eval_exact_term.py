"""Retrieval-only A/B experiment: why BM25 / hybrid beats dense-only.

Runs three retrievers over the SAME corpus and eval items -- no LLM, no API keys,
purely retrieval scoring:

  1. dense   : Qdrant (in-memory) + all-MiniLM-L6-v2 (semantic similarity only)
  2. bm25    : sparse lexical scoring (exact-token matching)
  3. hybrid  : 50/50 EnsembleRetriever (same builder the app/notebook uses)

Usage:  python scripts/eval_exact_term.py
"""

# ruff: noqa: E402 -- the sys.path bootstrap below must run before the hr_rag imports
import os
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
sys.path.insert(0, "data/eval")

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qa_dataset import get_subset
from qdrant_client import QdrantClient, models

from hr_rag.chunking import chunk_documents
from hr_rag.config import DEFAULT_TOP_K
from hr_rag.data_loading import load_policy_documents
from hr_rag.embeddings import get_embeddings
from hr_rag.retrievers.hybrid import build_hybrid_ensemble

TOP_K = DEFAULT_TOP_K


class QdrantDenseRetriever:
    def __init__(self, client: QdrantClient, collection: str, embeddings, k: int):
        self.client = client
        self.collection = collection
        self.embeddings = embeddings
        self.k = k

    def invoke(self, question: str) -> list[Document]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.embeddings.embed_query(question),
            limit=self.k,
            with_payload=True,
        )
        return [
            Document(page_content=point.payload["text"], metadata=point.payload.get("metadata", {}))
            for point in response.points
        ]

    def as_retriever(self, search_kwargs: dict | None = None):
        k = (search_kwargs or {}).get("k", self.k)
        return QdrantDenseRetriever(self.client, self.collection, self.embeddings, k)


def build_dense_store(chunks: list[Document], embeddings) -> QdrantDenseRetriever:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="eval",
        vectors_config=models.VectorParams(
            size=len(embeddings.embed_query("dimension check")),
            distance=models.Distance.COSINE,
        ),
    )
    client.upload_collection(
        collection_name="eval",
        vectors=embeddings.embed_documents([chunk.page_content for chunk in chunks]),
        payload=[{"text": chunk.page_content, "metadata": dict(chunk.metadata)} for chunk in chunks],
        ids=list(range(len(chunks))),
    )
    return QdrantDenseRetriever(client, "eval", embeddings, TOP_K)


def _norm(text: str) -> str:
    return text.lower()


def _chunk_fulfills(chunk, keywords: list[str]) -> bool:
    text = _norm(chunk.page_content)
    return all(_norm(kw) in text for kw in keywords)


def _keywords_in_topk(docs, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = " ".join(_norm(d.page_content) for d in docs)
    return sum(1 for kw in keywords if _norm(kw) in text) / len(keywords)


def score_retriever(name: str, retriever, items) -> dict:
    full_hits, rr_sum, coverage_sum = 0, 0.0, 0.0
    for item in items:
        docs = retriever.invoke(item.question)[:TOP_K]
        kws = item.expected_keywords or []

        coverage_sum += _keywords_in_topk(docs, kws)
        if not docs:
            continue

        # full_hit: all keywords present SOMEWHERE in top-k union
        union_text = " ".join(_norm(d.page_content) for d in docs)
        if all(_norm(kw) in union_text for kw in kws):
            full_hits += 1

        # MRR: first chunk containing ALL keywords
        for rank, d in enumerate(docs, start=1):
            if _chunk_fulfills(d, kws):
                rr_sum += 1.0 / rank
                break

    n = len(items)
    return {
        "retriever": name,
        f"full_hit@{TOP_K}": round(full_hits / n, 3),
        f"avg_kw_coverage@{TOP_K}": round(coverage_sum / n, 3),
        f"mrr@{TOP_K}": round(rr_sum / n, 3),
    }


def main():
    print("Loading + chunking policies ...")
    docs = load_policy_documents()
    chunks = chunk_documents(docs)
    print(f"  {len(docs)} documents -> {len(chunks)} chunks\n")

    print("Building retrievers (dense | bm25 | hybrid) ...")
    db = build_dense_store(chunks, get_embeddings())
    dense = db.as_retriever(search_kwargs={"k": TOP_K})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    hybrid = build_hybrid_ensemble(chunks, db, k=TOP_K, bm25_weight=0.5)

    retrievers = {
        "dense (vector only)": dense,
        "bm25 (sparse only)": bm25,
        "hybrid (0.5/0.5)": hybrid,
    }

    slices = {
        "EXACT_TERM (code lookups)": get_subset(question_type="exact_keyword"),
        "CONTROL: paraphrase slice": get_subset(question_type="paraphrase"),
    }

    results: dict[str, list[dict]] = {}
    for slice_name, items in slices.items():
        print(f"\n=== {slice_name}  ({len(items)} items, k={TOP_K}) ===")
        rows = []
        for name, r in retrievers.items():
            row = score_retriever(name, r, items)
            rows.append(row)
            print(f"  {name:<20} {row}")
        results[slice_name] = rows

    # ---- verdict --------------------------------------------------------
    print("\n=== VERDICT ===")
    exact = {r["retriever"]: r for r in results["EXACT_TERM (code lookups)"]}
    para = {r["retriever"]: r for r in results["CONTROL: paraphrase slice"]}
    for name in retrievers:
        e_mrr = exact[name][f"mrr@{TOP_K}"]
        p_mrr = para[name][f"mrr@{TOP_K}"]
        print(
            f"  {name:<20} exact-MRR={e_mrr:.3f}  paraphrase-MRR={p_mrr:.3f}"
        )
    dense_e = exact["dense (vector only)"][f"mrr@{TOP_K}"]
    hyb_e = exact["hybrid (0.5/0.5)"][f"mrr@{TOP_K}"]
    bm_p = para["bm25 (sparse only)"][f"mrr@{TOP_K}"]
    if hyb_e >= dense_e:
        print(
            "\n  Hybrid recovers what dense-only loses on exact-term queries "
            f"(MRR {dense_e:.3f} -> {hyb_e:.3f}) while staying competitive on "
            f"paraphrases where pure BM25 sits at {bm_p:.3f}."
        )
        print("  => Justifies the BM25+dense hybrid retriever choice.")


if __name__ == "__main__":
    main()
