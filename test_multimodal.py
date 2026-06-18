"""Multi-modal end-to-end test: index a text-PDF and an image, then ask across both.

Exercises pypdf text extraction (invoice.pdf) and Groq vision OCR (poster.png),
followed by retrieval + grounded generation.

Run:  .venv/Scripts/python.exe make_samples.py   # first, to create the files
      .venv/Scripts/python.exe test_multimodal.py
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

    for fname in ["invoice.pdf", "poster.png"]:
        path = os.path.join(HERE, "sample_docs", fname)
        info = pipe.ingest_file(path)
        print(f"Indexed {info['source']}: {info['chunks']} chunks, {info['chars']} chars")
    print(f"\nCollection holds {pipe.store.count()} chunks total\n")

    questions = [
        "What is the total amount due on the invoice and what are the payment terms?",
        "Who is the invoice billed to, and what is the invoice number?",
        "When and where is the Accessibility Summit being held?",
        "What is the keynote topic at the summit?",
    ]
    for q in questions:
        res = pipe.answer(q)
        print(f"Q: {q}")
        print(f"A: {res['answer']}")
        print(f"   sources: {', '.join(res['sources'])}\n")


if __name__ == "__main__":
    main()
