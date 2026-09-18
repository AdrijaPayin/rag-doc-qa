"""
A small hand-labeled test set used by scripts/evaluate.py.
"""

EVAL_SET = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["retrieval", "generation", "document"],
    },
    {
        "question": "What does FAISS stand for?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["facebook", "similarity", "search"],
    },
    {
        "question": "What is Recall@K used to measure?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["recall", "retriever", "top"],
    },
    {
        "question": "What does FastAPI use for data validation?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["pydantic"],
    },
    {
        "question": "Where can I find FastAPI's interactive documentation?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["docs", "redoc"],
    },
    {
        "question": "What command is commonly used to run a FastAPI app during development?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["uvicorn"],
    },
    {
        "question": "What tool is commonly used to test FastAPI applications?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["pytest", "testclient"],
    },
]
