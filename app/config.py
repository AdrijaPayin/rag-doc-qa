"""
Central place for all the settings the app needs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Embedding model (Gemini's hosted embedding API -- no local model,
# no torch, no heavy downloads. This keeps the deployed app small and
# avoids the memory issues that come with loading a local transformer.) ---
EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_DIMENSION = 768

# --- Text chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 5

# --- Storage paths ---
DATA_DIR = "data"
SAMPLE_DOCS_DIR = os.path.join(DATA_DIR, "sample_docs")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_STORE_PATH = os.path.join(DATA_DIR, "chunks.json")

# --- Generation model ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"
