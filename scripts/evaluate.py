"""
Runs the labeled question set through the real RAG pipeline and
reports answer accuracy, Retrieval Recall@5, and average latency.

Usage:
    python scripts/evaluate.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_pipeline import load_index, answer_question
from scripts.eval_dataset import EVAL_SET

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

# Gemini's free tier allows 5 requests per minute. Each question makes
# 2 calls (1 embedding + 1 generation), so we pace ourselves at roughly
# 2 questions per minute to stay safely under that limit and avoid
# triggering 503 "high demand" / rate-limit errors in the first place.
SECONDS_BETWEEN_QUESTIONS = 25


def keyword_match(answer: str, expected_keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(keyword.lower() in answer_lower for keyword in expected_keywords)


def answer_question_with_retry(question: str, top_k: int = 5) -> dict:
    """
    Wraps answer_question with a small retry loop. Gemini occasionally
    returns a transient 503 "high demand" error -- this is temporary
    on Google's end, not a bug in the pipeline, so a short retry
    usually clears it without losing progress on the rest of the eval.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return answer_question(question, top_k=top_k)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"    (attempt {attempt} failed: {e}. Retrying in {RETRY_DELAY_SECONDS}s...)")
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


def main():
    print("Loading FAISS index...")
    load_index()

    total = len(EVAL_SET)
    correct_answers = 0
    correct_retrievals = 0
    retrieval_latencies = []
    total_latencies = []
    skipped = 0

    print(f"\nRunning evaluation on {total} questions...")
    print(f"(pacing ~{SECONDS_BETWEEN_QUESTIONS}s between questions to respect Gemini's free-tier rate limit)\n")

    for i, item in enumerate(EVAL_SET, start=1):
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_keywords"]

        print(f"[{i}/{total}] Q: {question}")

        try:
            result = answer_question_with_retry(question, top_k=5)
        except Exception as e:
            print(f"    SKIPPED -- failed after {MAX_RETRIES} attempts: {e}\n")
            skipped += 1
            # still pace even after a skip, so we don't hammer the API
            if i < total:
                time.sleep(SECONDS_BETWEEN_QUESTIONS)
            continue

        retrieved_sources = {c["source"] for c in result["retrieved_chunks"]}
        retrieval_hit = expected_source in retrieved_sources
        if retrieval_hit:
            correct_retrievals += 1

        answer_hit = keyword_match(result["answer"], expected_keywords)
        if answer_hit:
            correct_answers += 1

        retrieval_latencies.append(result["retrieval_latency_ms"])
        total_latencies.append(result["total_latency_ms"])

        status_retrieval = "PASS" if retrieval_hit else "FAIL"
        status_answer = "PASS" if answer_hit else "FAIL"
        print(f"    Retrieval: {status_retrieval}  |  Answer: {status_answer}")
        print(f"    Answer given: {result['answer'][:120]}")
        print()

        if i < total:
            time.sleep(SECONDS_BETWEEN_QUESTIONS)

    answered = total - skipped
    if answered == 0:
        print("No questions completed successfully -- nothing to summarize.")
        return

    accuracy = (correct_answers / answered) * 100
    recall_at_5 = (correct_retrievals / answered) * 100
    avg_retrieval_latency = sum(retrieval_latencies) / answered
    avg_total_latency = sum(total_latencies) / answered

    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    if skipped:
        print(f"Skipped (API errors):   {skipped}/{total}")
    print(f"Answer accuracy:        {accuracy:.1f}%  ({correct_answers}/{answered})")
    print(f"Retrieval Recall@5:     {recall_at_5:.1f}%  ({correct_retrievals}/{answered})")
    print(f"Avg retrieval latency:  {avg_retrieval_latency:.2f} ms")
    print(f"Avg total latency:      {avg_total_latency:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()