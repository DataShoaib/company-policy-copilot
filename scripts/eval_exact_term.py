"""Retrieval-only A/B experiment: why BM25 / hybrid beats dense-only.

Runs three retrievers over the SAME corpus and eval items -- no LLM, no API keys,
purely retrieval scoring:

  1. dense   : FAISS + all-MiniLM-L6-v2 (semantic similarity only)
  2. bm25    : sparse lexical scoring (exact-token matching)
  3. hybrid  : 50/50 EnsembleRetriever (same builder the app/notebook uses)

Eval slices:
  - exact_keyword : questions whose answers hinge on rare lexical tokens
                    (codes/IDs/form numbers, e.g. "EXP-ENT-05", "HRP-001").
                    Dense embeddings blur these into generic prose; BM25 nails them.
  - paraphrase    : control slice -- semantically reworded questions. This is
                    where dense holds its own, which is exactly WHY hybrid (not
                    pure BM25) is the right choice.

Metrics (per retriever):
  - full_hit@k : fraction of items where ALL expected_keywords appear across top-k chunks
  - mrr        : mean reciprocal rank of the first chunk containing ALL expected keywords

Usage:  python scripts/eval_exact_term.py
"""
import os
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
sys.path.insert(0, "data/eval")

from hr_rag.config import DEFAULT_TOP_K  # noqa: E402
from hr_rag.data_loading import load_policy_documents  # noqa: E402
from hr_rag.chunking import chunk_documents  # noqa: E402
from hr_rag.embeddings import get_embeddings  # noqa: E402
from hr_rag.retrievers.hybrid import build_hybrid_ensemble  # noqa: E402
from qa_dataset import QA_ITEMS, get_subset  # noqa: E402

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402

TOP_K = DEFAULT_TOP_K


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
    db = FAISS.from_documents(chunks, get_embeddings())
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
