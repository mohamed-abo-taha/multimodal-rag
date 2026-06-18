"""Quick end-to-end check (no UI): index the sample doc and ask two questions.

Run:  .venv/Scripts/python.exe smoke_test.py
Proves the whole path works: chunk -> local embed -> Chroma retrieve -> Groq answer.
"""
import os
import sys

from rag.pipeline import RAGPipeline

try:  # ensure non-ASCII model output prints on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pipe = RAGPipeline()

    sample = os.path.join(HERE, "sample_docs", "sample.txt")
    info = pipe.ingest_file(sample)
    print(f"Indexed {info['source']}: {info['chunks']} chunks, {info['chars']} chars")
    print(f"Collection now holds {pipe.store.count()} chunks\n")

    questions = [
        "How many days per week can employees work remotely?",
        "What is the home-office equipment stipend and what can it be used for?",
        "Who approves a fully remote arrangement?",
    ]
    for q in questions:
        res = pipe.answer(q)
        print(f"Q: {q}")
        print(f"A: {res['answer']}")
        print(f"   sources: {', '.join(res['sources'])}\n")


if __name__ == "__main__":
    main()
