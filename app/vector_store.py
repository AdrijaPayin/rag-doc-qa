"""
A thin wrapper around FAISS for building, searching, saving, and
loading a vector index of document chunks.
"""

import json
import os
from typing import List, Tuple

import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH, CHUNKS_STORE_PATH


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[dict] = []

    def build(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        if self.index is None:
            raise RuntimeError("Index is empty. Call build() or load() first.")

        scores, positions = self.index.search(query_embedding, top_k)

        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:
                continue
            results.append((self.chunks[pos], float(score)))
        return results

    def save(self, index_path: str = FAISS_INDEX_PATH, chunks_path: str = CHUNKS_STORE_PATH) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

    def load(self, index_path: str = FAISS_INDEX_PATH, chunks_path: str = CHUNKS_STORE_PATH) -> None:
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                "No saved index found. Run scripts/ingest.py first."
            )
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0
