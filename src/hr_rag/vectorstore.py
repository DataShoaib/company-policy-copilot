from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from hr_rag.chunking import group_by_category
from hr_rag.config import CATEGORIES, VECTORSTORE_DIR
from hr_rag.embeddings import get_embeddings


def build_vectorstore(chunks: list[Document]) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())


def save_vectorstore(db: FAISS, path: Path = VECTORSTORE_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(path))


def load_vectorstore(path: Path = VECTORSTORE_DIR) -> FAISS | None:
    if not (path / "index.faiss").exists():
        return None
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)


def build_or_load_vectorstore(chunks: list[Document], path: Path = VECTORSTORE_DIR) -> FAISS:
    db = load_vectorstore(path)
    if db is not None:
        return db
    db = build_vectorstore(chunks)
    save_vectorstore(db, path)
    return db


# --- per-category collections ---
# one small FAISS index per policy category instead of one big index over
# everything. a query scoped to "leave" only ever searches the leave
# collection, not all 5 categories worth of vectors -- smaller search space,
# and it's also what makes RBAC cheap: a role that can't see "compensation"
# never even touches that collection.

def build_category_vectorstores(chunks: list[Document]) -> dict[str, FAISS]:
    by_category = group_by_category(chunks)
    return {cat: FAISS.from_documents(docs, get_embeddings()) for cat, docs in by_category.items()}


def save_category_vectorstores(stores: dict[str, FAISS], base_path: Path = VECTORSTORE_DIR) -> None:
    for cat, db in stores.items():
        path = base_path / cat
        path.mkdir(parents=True, exist_ok=True)
        db.save_local(str(path))


def load_category_vectorstores(categories: list[str] = CATEGORIES, base_path: Path = VECTORSTORE_DIR) -> dict[str, FAISS]:
    stores = {}
    for cat in categories:
        path = base_path / cat
        if (path / "index.faiss").exists():
            stores[cat] = FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)
    return stores


def build_or_load_category_vectorstores(chunks: list[Document], categories: list[str] = CATEGORIES, base_path: Path = VECTORSTORE_DIR) -> dict[str, FAISS]:
    stores = load_category_vectorstores(categories, base_path)
    if len(stores) == len(categories):
        return stores
    stores = build_category_vectorstores(chunks)
    save_category_vectorstores(stores, base_path)
    return stores
