"""
FastAPI app exposing the RAG /ask endpoint.

Run with: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_pipeline import load_index, answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_index()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Document QA with RAG",
    description="Ask questions about your documents, answered using Retrieval-Augmented Generation.",
    version="2.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[SourceChunk]
    retrieval_latency_ms: float
    total_latency_ms: float


@app.get("/")
def root():
    return {"message": "RAG Document QA API is running. POST to /ask with a question."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(request.question, top_k=request.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result
