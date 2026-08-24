from qdrant_client import QdrantClient

c = QdrantClient(path="data/qdrant")
cols = c.get_collections().collections
print("collections:", [x.name for x in cols])
for col in cols:
    info = c.get_collection(col.name)
    print(col.name, "points:", info.points_count)