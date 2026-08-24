
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from hr_rag.config import CATEGORIES
from hr_rag.embeddings import get_embeddings


def get_qdrant_client() -> QdrantClient:
    from hr_rag.api.core.settings import settings

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return QdrantClient(path=str(settings.qdrant_path))


def collection_name(category: str) -> str:
    from hr_rag.api.core.settings import settings

    return f"{settings.qdrant_collection_prefix}{category}"


def _document_payload(document: Document) -> dict:
    return {
        "text": document.page_content,
        "metadata": dict(document.metadata),
    }


def build_category_collections(chunks: list[Document], client: QdrantClient | None = None, force: bool = False) -> dict[str, "QdrantCategoryStore"]:
    client = client or get_qdrant_client()
    embeddings = get_embeddings()
    grouped: dict[str, list[Document]] = {category: [] for category in CATEGORIES}
    for chunk in chunks:
        category = chunk.metadata.get("category")
        if category in grouped:
            grouped[category].append(chunk)

    stores = {}
    for category, category_chunks in grouped.items():
        name = collection_name(category)
        if client.collection_exists(name) and not force:
            stores[category] = QdrantCategoryStore(client, name, embeddings)
            continue
        if force and client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=len(embeddings.embed_query("dimension check")),
                distance=models.Distance.COSINE,
            ),
        )
        vectors = embeddings.embed_documents([chunk.page_content for chunk in category_chunks])
        client.upload_collection(
            collection_name=name,
            vectors=vectors,
            payload=[_document_payload(chunk) for chunk in category_chunks],
            ids=list(range(len(category_chunks))),
        )
        stores[category] = QdrantCategoryStore(client, name, embeddings)
    return stores


def load_category_collections(client: QdrantClient | None = None) -> dict[str, "QdrantCategoryStore"]:
    client = client or get_qdrant_client()
    embeddings = get_embeddings()
    stores = {}
    for category in CATEGORIES:
        name = collection_name(category)
        if client.collection_exists(name):
            stores[category] = QdrantCategoryStore(client, name, embeddings)
    return stores


class QdrantCategoryStore:
    def __init__(self, client: QdrantClient, name: str, embeddings) -> None:
        self.client = client
        self.name = name
        self.embeddings = embeddings

    def invoke(self, question: str, limit: int = 3, metadata_filter: dict | None = None) -> list[Document]:
        query_filter = None
        if metadata_filter:
            query_filter = models.Filter(
                must=[models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value)) for key, value in metadata_filter.items()]
            )
        response = self.client.query_points(
            collection_name=self.name,
            query=self.embeddings.embed_query(question),
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            Document(page_content=point.payload["text"], metadata=point.payload.get("metadata", {}))
            for point in response.points
        ]