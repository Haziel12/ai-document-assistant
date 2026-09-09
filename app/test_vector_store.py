from src.vector_store import VectorStore


embeddings = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.9, 0.1, 0.0],
]


store = VectorStore(dimension=3)

store.add(embeddings)

store.save("data/vector_store/test.index")

loaded_store = VectorStore.load(
    "data/vector_store/test.index"
)

scores, indices = loaded_store.search(
    [1.0, 0.0, 0.0],
    top_k=2,
)

print("Scores:", scores)
print("Indices:", indices)