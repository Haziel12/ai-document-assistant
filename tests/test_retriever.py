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

    assert len(results) == 3

    for result in results:
        assert "text" in result
        assert "score" in result
        assert "page" in result

        print("\n---")
        print("Score:", result["score"])
        print("Page:", result["page"])
        print("Text:", result["text"])