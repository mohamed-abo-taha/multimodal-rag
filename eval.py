"""Tiny retrieval-quality eval: run known Q&A pairs and check the answer contains the fact.

A lightweight, deterministic signal that the RAG pipeline returns correct, grounded answers.
Run:  .venv/Scripts/python.exe eval.py
"""
import os
import sys

from rag.pipeline import RAGPipeline

try:  # ensure non-ASCII model output prints on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

# (question, [acceptable substrings] — any match counts the answer as correct)
CASES = [
    ("How many days per week can employees work remotely?", ["three", "3"]),
    ("What is the home-office equipment stipend?", ["$500", "500"]),
    ("Who approves a fully remote arrangement?", ["director"]),
    ("What are the core hours employees must be available?", ["10", "3"]),
    ("Can employees use public Wi-Fi for remote work?", ["vpn"]),
]


def main():
    pipe = RAGPipeline()
    pipe.ingest_file(os.path.join(HERE, "sample_docs", "sample.txt"))

    passed = 0
    for question, expected in CASES:
        answer = (pipe.answer(question)["answer"] or "").lower()
        ok = any(e.lower() in answer for e in expected)
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {question}")
        print(f"        -> {answer[:90]}")
    pct = round(100 * passed / len(CASES))
    print(f"\nScore: {passed}/{len(CASES)} ({pct}%)")
    return passed == len(CASES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
