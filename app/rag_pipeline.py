"""
The core RAG pipeline: embed question -> retrieve chunks -> generate answer.
"""

import time

from app.config import TOP_K
from app.embeddings import embed_query
from app.vector_store import VectorStore
from app.generator import generate_answer

_vector_store = VectorStore()


def load_index() -> None:
    _vector_store.load()


def get_vector_store() -> VectorStore:
    return _vector_store


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    if not _vector_store.is_ready():
        raise RuntimeError("Vector store not loaded. Call load_index() first.")

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    query_embedding = embed_query(question)
    retrieved = _vector_store.search(query_embedding, top_k)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    answer_text = generate_answer(question, retrieved)

    total_latency_ms = (time.perf_counter() - total_start) * 1000

    sources = sorted({chunk["source"] for chunk, _score in retrieved})

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": [
            {"text": chunk["text"], "source": chunk["source"], "score": score}
            for chunk, score in retrieved
        ],
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
    }
