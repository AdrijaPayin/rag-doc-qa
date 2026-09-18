from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_returns_503_when_index_not_loaded():
    with patch("app.main.answer_question", side_effect=RuntimeError("Vector store not loaded.")):
        response = client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 503


def test_ask_returns_answer_when_pipeline_succeeds():
    fake_result = {
        "question": "What is RAG?",
        "answer": "RAG grounds answers in retrieved documents.",
        "sources": ["rag.txt"],
        "retrieved_chunks": [
            {"text": "RAG combines retrieval and generation.", "source": "rag.txt", "score": 0.95}
        ],
        "retrieval_latency_ms": 12.3,
        "total_latency_ms": 450.6,
    }
    with patch("app.main.answer_question", return_value=fake_result):
        response = client.post("/ask", json={"question": "What is RAG?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "RAG grounds answers in retrieved documents."
    assert body["sources"] == ["rag.txt"]
