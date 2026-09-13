from app.src.retriever import Retriever


def test_retrieve():
    retriever = Retriever(
        index_path="data/vector_store/index.faiss",
        metadata_path="data/vector_store/chunks.json",
    )

    results = retriever.retrieve(
        "What is my experience with machine learning?",
        top_k=3,
    )

    # We requested the top 3 results
    assert len(results) == 3

    # Every result should contain the expected fields
    for result in results:
        assert "text" in result
        assert "score" in result
        assert "page" in result

        # Basic sanity checks
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

        assert isinstance(result["score"], (float, int))
        assert isinstance(result["page"], int)

    # Results should be ordered from most to least similar
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)