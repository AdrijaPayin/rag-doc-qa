"""
Generates embeddings using Gemini's hosted embedding API instead of a
local Sentence Transformer model. This keeps the deployed app small
and memory-light (no torch, no model download at startup), which is
what makes it deployable on Render's free tier without hitting
out-of-memory errors.
"""

from typing import List
import numpy as np
from google import genai

from app.config import GEMINI_API_KEY, EMBEDDING_MODEL_NAME

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2D numpy array of normalized
    embeddings, using Gemini's embedding API.
    """
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL_NAME,
        contents=texts,
    )
    vectors = [np.array(e.values, dtype="float32") for e in result.embeddings]
    embeddings = np.vstack(vectors)

    # Normalize to unit length so inner product == cosine similarity,
    # matching how the FAISS index is built.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, embedding_dimension)."""
    return embed_texts([query])
