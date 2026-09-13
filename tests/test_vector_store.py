from app.src.vector_store import VectorStore
from app.src.metadata import save_chunks, load_chunks


embeddings = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.9, 0.1, 0.0],
]

chunks = [
    {
        "chunk_id": 0,
        "text": "This is the first document chunk.",
        "source": "test.pdf",
        "page": 1,
    },
    {
        "chunk_id": 1,
        "text": "This is the second document chunk.",
        "source": "test.pdf",
        "page": 2,
    },
    {
        "chunk_id": 2,
        "text": "This is the third document chunk.",
        "source": "test.pdf",
        "page": 3,
    },
]


store = VectorStore(dimension=3)

store.add(embeddings)

store.save("data/vector_store/test_index.faiss")

save_chunks(
    chunks,
    "data/vector_store/test_chunks.json",
)


loaded_store = VectorStore.load(
    "data/vector_store/test_index.faiss"
)

loaded_chunks = load_chunks(
    "data/vector_store/test_chunks.json"
)


scores, indices = loaded_store.search(
    [1.0, 0.0, 0.0],
    top_k=2,
)


print("Scores:", scores)
print("Indices:", indices)

for index in indices:
    print(loaded_chunks[index])