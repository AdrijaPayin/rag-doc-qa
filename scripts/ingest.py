"""
Run this once (and whenever documents change) to build the FAISS index.

Usage:
    python scripts/ingest.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import SAMPLE_DOCS_DIR
from app.document_loader import load_and_chunk_documents
from app.embeddings import embed_texts
from app.vector_store import VectorStore


def main():
    print(f"Loading and chunking documents from '{SAMPLE_DOCS_DIR}'...")
    chunks = load_and_chunk_documents(SAMPLE_DOCS_DIR)

    if not chunks:
        print("No documents found. Add .txt or .pdf files to data/sample_docs/ and rerun.")
        return

    print(f"Created {len(chunks)} chunks. Generating embeddings via Gemini API...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print("Building FAISS index...")
    store = VectorStore()
    store.build(embeddings, chunks)
    store.save()

    print(f"Done. Indexed {len(chunks)} chunks from documents in '{SAMPLE_DOCS_DIR}'.")


if __name__ == "__main__":
    main()
