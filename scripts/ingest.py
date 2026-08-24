import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hr_rag.chunking import chunk_documents, group_by_category
from hr_rag.data_loading import load_policy_documents
from hr_rag.qdrant_store import build_category_collections


def main():
    parser = argparse.ArgumentParser(description="Build and persist per-category Qdrant collections.")
    parser.add_argument("--force", action="store_true", help="rebuild even if collections already exist")
    args = parser.parse_args()

    docs = load_policy_documents()
    print(f"loaded {len(docs)} documents")
    for d in docs:
        print(f"  {d.metadata['category']:15s} {d.metadata['policy_doc_id']:10s} {d.metadata['title']}")

    chunks = chunk_documents(docs)
    by_category = group_by_category(chunks)
    print(f"chunked into {len(chunks)} pieces across {len(by_category)} categories")
    for cat, cat_chunks in by_category.items():
        print(f"  {cat:15s} {len(cat_chunks)} chunks")

    stores = build_category_collections(chunks, force=args.force)
    print(f"saved {len(stores)} per-category Qdrant collections")


if __name__ == "__main__":
    main()
