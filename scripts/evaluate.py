"""
Runs the labeled question set through the real RAG pipeline and
reports answer accuracy, Retrieval Recall@5, and average latency.

Usage:
    python scripts/evaluate.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag_pipeline import load_index, answer_question
from scripts.eval_dataset import EVAL_SET


def keyword_match(answer: str, expected_keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(keyword.lower() in answer_lower for keyword in expected_keywords)


def main():
    print("Loading FAISS index...")
    load_index()

    total = len(EVAL_SET)
    correct_answers = 0
    correct_retrievals = 0
    retrieval_latencies = []
    total_latencies = []

    print(f"\nRunning evaluation on {total} questions...\n")

    for i, item in enumerate(EVAL_SET, start=1):
        question = item["question"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_keywords"]

        result = answer_question(question, top_k=5)

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
        print(f"[{i}/{total}] Q: {question}")
        print(f"    Retrieval: {status_retrieval}  |  Answer: {status_answer}")
        print(f"    Answer given: {result['answer'][:120]}")
        print()

    accuracy = (correct_answers / total) * 100
    recall_at_5 = (correct_retrievals / total) * 100
    avg_retrieval_latency = sum(retrieval_latencies) / total
    avg_total_latency = sum(total_latencies) / total

    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Answer accuracy:        {accuracy:.1f}%  ({correct_answers}/{total})")
    print(f"Retrieval Recall@5:     {recall_at_5:.1f}%  ({correct_retrievals}/{total})")
    print(f"Avg retrieval latency:  {avg_retrieval_latency:.2f} ms")
    print(f"Avg total latency:      {avg_total_latency:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
