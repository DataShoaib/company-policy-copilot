from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hr_rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(docs: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def group_by_category(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c.metadata.get("category", "unknown")].append(c)
    return dict(grouped)
