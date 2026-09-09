from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Document Assistant API is running!"
    }


def test_ask():
    response = client.post(
        "/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "question" in data
    assert "answer" in data
    assert data["question"] == "What is RAG?"
    assert isinstance(data["answer"], str)


def test_ask_invalid_request():
    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422