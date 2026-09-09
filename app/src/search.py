import numpy as np


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(
    query_embedding: list[float],
    chunks: list[dict],
    chunk_embeddings: list[list[float]],
    top_k: int = 3,
) -> list[dict]:
    """Return the most semantically similar chunks."""

    results = []

    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)

        results.append(
            {
                **chunk,
                "score": score,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]