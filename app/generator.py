"""
Takes retrieved chunks + the user's question and asks Gemini to
answer using ONLY that context.
"""

from typing import List, Tuple
from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

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


def build_prompt(question: str, retrieved_chunks: List[Tuple[dict, float]]) -> str:
    context_blocks = []
    for i, (chunk, _score) in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Chunk {i} - source: {chunk['source']}]\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below.

Context:
{context_text}

Question: {question}

Instructions:
- Answer using only the information in the context above.
- If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer that."
- Keep the answer clear and concise.

Answer:"""
    return prompt


def generate_answer(question: str, retrieved_chunks: List[Tuple[dict, float]]) -> str:
    if not retrieved_chunks:
        return "I don't have enough information in the provided documents to answer that."

    client = _get_client()
    prompt = build_prompt(question, retrieved_chunks)

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
    return response.text.strip()
