import pytest
from unittest.mock import MagicMock, patch

import app.generator as generator_module
from app.generator import build_prompt, generate_answer


def _reset_cached_client():
    generator_module._client = None


def test_build_prompt_includes_question_and_context():
    chunks = [
        ({"text": "RAG combines retrieval and generation.", "source": "rag.txt"}, 0.9),
        ({"text": "FAISS is a vector search library.", "source": "faiss.txt"}, 0.8),
    ]
    prompt = build_prompt("What is RAG?", chunks)

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval and generation." in prompt
    assert "rag.txt" in prompt


def test_build_prompt_instructs_model_to_say_when_unknown():
    prompt = build_prompt("Anything?", [({"text": "x", "source": "a.txt"}, 0.5)])
    assert "I don't have enough information" in prompt


def test_generate_answer_returns_fallback_when_no_chunks_retrieved():
    answer = generate_answer("What is RAG?", [])
    assert "I don't have enough information" in answer


@patch("app.generator.GEMINI_API_KEY", "fake-key-for-testing")
@patch("app.generator.genai.Client")
def test_generate_answer_calls_gemini_and_returns_text(mock_client_class):
    _reset_cached_client()
    mock_response = MagicMock()
    mock_response.text = "  RAG grounds answers in retrieved documents.  "

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    chunks = [({"text": "RAG grounds answers.", "source": "rag.txt"}, 0.9)]
    answer = generate_answer("What is RAG?", chunks)

    assert answer == "RAG grounds answers in retrieved documents."
    _reset_cached_client()


@patch("app.generator.GEMINI_API_KEY", "")
def test_generate_answer_raises_clear_error_when_api_key_missing():
    _reset_cached_client()
    chunks = [({"text": "some context", "source": "a.txt"}, 0.5)]

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        generate_answer("What is RAG?", chunks)

    _reset_cached_client()
