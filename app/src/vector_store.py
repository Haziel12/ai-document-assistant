import faiss
import numpy as np


class VectorStore:
    """Simple FAISS vector store."""

    def __init__(self, dimension: int):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: list[list[float]]) -> None:
        """Add embeddings to the index."""

        vectors = np.array(
            embeddings,
            dtype="float32",
        )

        faiss.normalize_L2(vectors)

        self.index.add(vectors)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> tuple[list[float], list[int]]:
        """Search for the most similar embeddings."""

        query = np.array(
            [query_embedding],
            dtype="float32",
        )

        faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query,
            top_k,
        )

        return scores[0].tolist(), indices[0].tolist()

    def save(self, path: str) -> None:
        """Save the FAISS index to disk."""

        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a FAISS index from disk."""

        index = faiss.read_index(path)

        store = cls(index.d)

        store.index = index

        return store