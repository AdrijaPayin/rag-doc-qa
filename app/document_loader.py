"""
Handles reading documents (.txt and .pdf) and splitting them
into small overlapping chunks.
"""

import os
from typing import List
from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return read_txt(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def load_and_chunk_documents(folder_path: str) -> List[dict]:
    all_chunks = []

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".txt", ".pdf"):
            continue

        raw_text = load_document(file_path)
        chunks = chunk_text(raw_text)

        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})

    return all_chunks
