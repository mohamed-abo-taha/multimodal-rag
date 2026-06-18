"""High-level RAG orchestration: ingest -> (history-aware) retrieve -> grounded answer."""
import os

from .vectorstore import VectorStore
from .llm import GroqClient
from .ingest import file_to_chunks
from . import config

SYSTEM_PROMPT = (
    "You are a precise document-analysis assistant. Answer the user's question using "
    "ONLY the provided context. If the answer is not contained in the context, say so "
    "plainly — do not invent facts. When you use information, mention the source "
    "filename it came from."
)

CONDENSE_PROMPT = (
    "Given the conversation so far and a follow-up question, rewrite the follow-up as a "
    "standalone search query that fully captures the user's intent — resolve pronouns and "
    "references to earlier turns. Output ONLY the rewritten query, nothing else.\n\n"
    "Conversation:\n{history}\n\nFollow-up: {question}\n\nStandalone query:"
)


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        self.llm = GroqClient()

    # --- ingestion ----------------------------------------------------------
    def ingest_file(self, path, source=None):
        """Read a file, chunk it, and add it to the vector store."""
        source = source or os.path.basename(str(path))
        full_text, chunks = file_to_chunks(path, llm=self.llm)
        n = self.store.add_chunks(chunks, source=source)
        return {"source": source, "chunks": n, "chars": len(full_text)}

    # --- retrieval helpers --------------------------------------------------
    @staticmethod
    def _build_context(hits):
        return "\n\n---\n\n".join(
            f"[Source: {h['metadata'].get('source', 'unknown')}]\n{h['text']}" for h in hits
        )

    def _condense(self, question, history):
        """Rewrite a follow-up into a standalone search query using recent history."""
        if not history:
            return question
        convo = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-config.HISTORY_TURNS:]
        )
        rewritten = self.llm.chat(
            [{"role": "user", "content": CONDENSE_PROMPT.format(history=convo, question=question)}],
            temperature=0.0, max_tokens=128,
        )
        rewritten = (rewritten or "").strip().strip('"').strip()
        return rewritten or question

    @staticmethod
    def cited_sources(answer, hits):
        """Sources the answer actually references (fallback: everything retrieved)."""
        retrieved = sorted({h["metadata"].get("source", "unknown") for h in hits})
        low = (answer or "").lower()
        cited = [s for s in retrieved if s.lower() in low]
        return cited or retrieved

    def prepare(self, question, history=None, top_k=None):
        """Build everything needed to answer: standalone query, hits, and LLM messages."""
        history = history or []
        search_query = self._condense(question, history)
        hits = self.store.query(search_query, top_k=top_k)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]}
                     for m in history[-config.HISTORY_TURNS:]]
        if hits:
            messages.append({"role": "user",
                             "content": f"Context:\n{self._build_context(hits)}\n\nQuestion: {question}"})
        else:
            messages.append({"role": "user", "content": question})
        return {"search_query": search_query, "hits": hits, "messages": messages}

    # --- answering ----------------------------------------------------------
    def answer(self, question, history=None, top_k=None):
        """Non-streaming answer (used by tests/CLI). Returns answer, sources, hits."""
        prep = self.prepare(question, history, top_k)
        if not prep["hits"]:
            return {
                "answer": "No documents are indexed yet — upload and index some files first.",
                "sources": [], "hits": [], "search_query": prep["search_query"],
            }
        answer = self.llm.chat(prep["messages"])
        return {
            "answer": answer,
            "sources": self.cited_sources(answer, prep["hits"]),
            "hits": prep["hits"],
            "search_query": prep["search_query"],
        }
