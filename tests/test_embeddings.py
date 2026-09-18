import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import app.embeddings as embeddings_module
from app.embeddings import embed_texts, embed_query


def _reset_cached_client():
    embeddings_module._client = None


def _fake_embed_response(vectors):
    """Build a fake response object shaped like the Gemini embed_content result."""
    response = MagicMock()
    response.embeddings = [MagicMock(values=v) for v in vectors]
    return response


@patch("app.embeddings.GEMINI_API_KEY", "fake-key-for-testing")
@patch("app.embeddings.genai.Client")
def test_embed_texts_returns_normalized_float32_array(mock_client_class):
    _reset_cached_client()
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embed_response(
        [[3.0, 4.0], [1.0, 0.0]]  # first vector has norm 5, second norm 1
    )
    mock_client_class.return_value = mock_client

    result = embed_texts(["hello", "world"])

    assert result.dtype == np.float32
    assert result.shape == (2, 2)
    # normalized: [3,4]/5 = [0.6, 0.8]
    assert result[0][0] == pytest.approx(0.6, abs=1e-4)
    assert result[0][1] == pytest.approx(0.8, abs=1e-4)
    _reset_cached_client()


@patch("app.embeddings.GEMINI_API_KEY", "fake-key-for-testing")
@patch("app.embeddings.genai.Client")
def test_embed_query_returns_single_row(mock_client_class):
    _reset_cached_client()
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embed_response([[1.0, 0.0]])
    mock_client_class.return_value = mock_client

    result = embed_query("a single question")

    assert result.shape == (1, 2)
    _reset_cached_client()


@patch("app.embeddings.GEMINI_API_KEY", "")
def test_embed_texts_raises_clear_error_when_api_key_missing():
    _reset_cached_client()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        embed_texts(["hello"])
    _reset_cached_client()
