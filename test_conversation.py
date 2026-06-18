"""Multi-turn test: follow-up questions that depend on conversation history.

Prints the rewritten standalone "search query" for each turn so you can see the
history-aware condensation working (e.g. "what about the stipend?" -> a full query).

Run:  .venv/Scripts/python.exe test_conversation.py
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

    # Make sure the sample corpus is present (text + PDF + image, if generated).
    for fname in ["sample.txt", "invoice.pdf", "poster.png"]:
        path = os.path.join(HERE, "sample_docs", fname)
        if os.path.exists(path):
            pipe.ingest_file(path)

    turns = [
        "How many days per week can employees work remotely?",
        "What about the equipment stipend?",        # 'what about' — needs history
        "Who has to approve a fully remote setup?",
        "Switching topics — what is the total on the invoice?",
        "And who is it billed to?",                 # 'it' = the invoice
    ]

    history = []
    for q in turns:
        res = pipe.answer(q, history=history)
        print(f"USER: {q}")
        print(f"   -> search query: {res['search_query']}")
        print(f"ASSISTANT: {res['answer']}")
        print(f"   sources: {', '.join(res['sources'])}\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": res["answer"]})


if __name__ == "__main__":
    main()
