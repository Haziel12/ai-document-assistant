from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Document Assistant API is running!"
    }


def test_ask():
    fake_answer = "This is a test answer."

    with patch(
        "app.main.ask_gemini",
        return_value=fake_answer,
    ):
        response = client.post(
            "/ask",
            json={"question": "What is my experience with machine learning?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is my experience with machine learning?"
    assert data["answer"] == fake_answer

    assert "sources" in data
    assert isinstance(data["sources"], list)

    for source in data["sources"]:
        assert "source" in source
        assert "page" in source
        assert "score" in source


def test_ask_invalid_request():
    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422

def test_ask_out_of_context():
    with patch(
        "app.main.ask_gemini"
    ) as mock_gemini:
        response = client.post(
            "/ask",
            json={
                "question": "What is my experience with quantum computing?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "The information is not available in the provided document."
    )

    assert data["sources"] == []

    mock_gemini.assert_not_called()