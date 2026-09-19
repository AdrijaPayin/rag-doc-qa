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
    {
        "question": "What are the three stages of a typical RAG pipeline?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["chunk", "embed", "retrieve"],
    },
    {
        "question": "Why does RAG reduce hallucination?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["context", "document", "hallucin"],
    },
    {
        "question": "What happens when a document changes in a RAG system?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["re-index", "reindex", "update"],
    },
    {
        "question": "What is the simplest type of FAISS index?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["flat"],
    },
    {
        "question": "What does a flat FAISS index do?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["exact", "nearest", "compar"],
    },
    {
        "question": "What are Sentence Transformers used for?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["embedding", "sentence", "paragraph"],
    },
    {
        "question": "What embedding dimension does all-MiniLM-L6-v2 produce?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["384"],
    },
    {
        "question": "Name some examples of vector databases mentioned in the documents.",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["faiss", "pinecone", "chroma"],
    },
    {
        "question": "What does a high Recall@5 score mean?",
        "expected_source": "rag_overview.txt",
        "expected_keywords": ["top five", "top 5", "present"],
    },
    {
        "question": "What is FastAPI built on top of?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["starlette", "pydantic"],
    },
    {
        "question": "What happens if a client sends invalid data to a FastAPI endpoint?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["error", "valid"],
    },
    {
        "question": "What UI does the /docs endpoint use in FastAPI?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["swagger"],
    },
    {
        "question": "What UI does the /redoc endpoint use in FastAPI?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["redoc"],
    },
    {
        "question": "How do you define an endpoint in FastAPI?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["decorate", "@app", "function"],
    },
    {
        "question": "What do type hints do in a FastAPI endpoint function?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["expect", "return", "data"],
    },
    {
        "question": "What ASGI server is commonly used to run FastAPI apps?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["uvicorn"],
    },
    {
        "question": "What does the --reload flag do when starting a FastAPI app?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["restart", "change"],
    },
    {
        "question": "What does FastAPI's TestClient let you do?",
        "expected_source": "fastapi_basics.txt",
        "expected_keywords": ["fake request", "request", "server"],
    },
]