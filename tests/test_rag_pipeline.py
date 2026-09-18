import numpy as np
import pytest
from unittest.mock import patch

from app import rag_pipeline
from app.vector_store import VectorStore


def _fake_store():
    chunks = [
        {"text": "RAG combines retrieval and generation.", "source": "rag.txt"},
        {"text": "FastAPI is a Python web framework.", "source": "fastapi.txt"},
    ]
    rng = np.random.default_rng(1)
    embeddings = rng.random((2, 8)).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    store = VectorStore()
    store.build(embeddings, chunks)
    return store, embeddings


@patch("app.rag_pipeline._vector_store")
def test_load_index_calls_store_load(mock_store):
    rag_pipeline.load_index()
    mock_store.load.assert_called_once()


def test_get_vector_store_returns_module_level_store():
    store, _ = _fake_store()
    rag_pipeline._vector_store = store
    assert rag_pipeline.get_vector_store() is store


def test_answer_question_raises_if_store_not_ready():
    rag_pipeline._vector_store = VectorStore()
    with pytest.raises(RuntimeError):
        rag_pipeline.answer_question("What is RAG?")


@patch("app.rag_pipeline.generate_answer")
@patch("app.rag_pipeline.embed_query")
def test_answer_question_returns_expected_shape(mock_embed_query, mock_generate_answer):
    store, embeddings = _fake_store()
    rag_pipeline._vector_store = store

    mock_embed_query.return_value = embeddings[0:1]
    mock_generate_answer.return_value = "RAG grounds answers in documents."

    result = rag_pipeline.answer_question("What is RAG?", top_k=2)

    assert result["question"] == "What is RAG?"
    assert result["answer"] == "RAG grounds answers in documents."
    assert set(result["sources"]) == {"rag.txt", "fastapi.txt"}
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieval_latency_ms"] >= 0
    assert result["total_latency_ms"] >= result["retrieval_latency_ms"]


@patch("app.rag_pipeline.generate_answer")
@patch("app.rag_pipeline.embed_query")
def test_answer_question_respects_top_k(mock_embed_query, mock_generate_answer):
    store, embeddings = _fake_store()
    rag_pipeline._vector_store = store

    mock_embed_query.return_value = embeddings[0:1]
    mock_generate_answer.return_value = "some answer"

    result = rag_pipeline.answer_question("Question?", top_k=1)
    assert len(result["retrieved_chunks"]) == 1
