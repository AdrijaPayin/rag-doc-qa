import numpy as np
import pytest

from app.vector_store import VectorStore


def _make_normalized_embeddings(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_build_creates_ready_index():
    chunks = [{"text": f"chunk {i}", "source": "doc.txt"} for i in range(5)]
    embeddings = _make_normalized_embeddings(5)

    store = VectorStore()
    store.build(embeddings, chunks)

    assert store.is_ready()
    assert store.index.ntotal == 5


def test_search_returns_most_similar_chunk_first():
    chunks = [{"text": f"chunk {i}", "source": "doc.txt"} for i in range(5)]
    embeddings = _make_normalized_embeddings(5)

    store = VectorStore()
    store.build(embeddings, chunks)

    query = embeddings[2:3]
    results = store.search(query, top_k=3)

    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert top_chunk["text"] == "chunk 2"
    assert top_score == pytest.approx(1.0, abs=1e-4)


def test_search_respects_top_k():
    chunks = [{"text": f"chunk {i}", "source": "doc.txt"} for i in range(10)]
    embeddings = _make_normalized_embeddings(10)

    store = VectorStore()
    store.build(embeddings, chunks)

    results = store.search(embeddings[0:1], top_k=4)
    assert len(results) == 4


def test_search_before_build_raises():
    store = VectorStore()
    with pytest.raises(RuntimeError):
        store.search(_make_normalized_embeddings(1), top_k=3)


def test_save_and_load_roundtrip(tmp_path):
    chunks = [{"text": f"chunk {i}", "source": "doc.txt"} for i in range(5)]
    embeddings = _make_normalized_embeddings(5)

    index_path = str(tmp_path / "index.bin")
    chunks_path = str(tmp_path / "chunks.json")

    store = VectorStore()
    store.build(embeddings, chunks)
    store.save(index_path=index_path, chunks_path=chunks_path)

    loaded_store = VectorStore()
    loaded_store.load(index_path=index_path, chunks_path=chunks_path)

    assert loaded_store.is_ready()
    assert len(loaded_store.chunks) == 5


def test_load_missing_files_raises(tmp_path):
    store = VectorStore()
    with pytest.raises(FileNotFoundError):
        store.load(
            index_path=str(tmp_path / "nope.bin"),
            chunks_path=str(tmp_path / "nope.json"),
        )


def test_is_ready_false_when_no_chunks():
    store = VectorStore()
    assert store.is_ready() is False
