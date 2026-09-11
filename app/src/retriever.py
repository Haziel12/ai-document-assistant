from app.src.embeddings import create_embeddings
from app.src.vector_store import VectorStore
from app.src.metadata import load_chunks


class Retriever:
    """Retrieve relevant document chunks using semantic search."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
    ):
        self.vector_store = VectorStore.load(index_path)
        self.chunks = load_chunks(metadata_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""

        query_embedding = create_embeddings([query])[0]

        scores, indices = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        results = []

        for score, index in zip(scores, indices):
            chunk = self.chunks[index]

            results.append(
                {
                    **chunk,
                    "score": score,
                }
            )

        return results