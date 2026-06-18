"""High-level RAG orchestration: ingest -> (history-aware) retrieve -> (re-rank) -> answer.

Every LLM-using method accepts an optional ``llm`` override so a per-request key
(e.g. a visitor's own Groq key) can be used without touching the shared default.
"""
import os
import re

from .vectorstore import VectorStore
from .llm import GroqClient
from .ingest import file_to_chunks
from . import config

SYSTEM_PROMPT = (
    "You are a precise document-analysis assistant. Answer using ONLY the provided context. "
    "The context passages are numbered like [1], [2]. Cite the passages you use inline with "
    "their numbers (e.g. [1]) and mention the source filename. If the answer is not in the "
    "context, say so plainly — do not invent facts."
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
        # The LLM needs a Groq key. Allow running key-less so the UI can start and let
        # a visitor supply their own key (self.llm stays None until then).
        self.llm = GroqClient() if config.GROQ_API_KEYS else None

    # --- ingestion ----------------------------------------------------------
    def ingest_file(self, path, source=None, llm=None):
        """Read a file, chunk it, and add it to the vector store."""
        source = source or os.path.basename(str(path))
        full_text, chunks = file_to_chunks(path, llm=llm or self.llm)
        n = self.store.add_chunks(chunks, source=source)
        return {"source": source, "chunks": n, "chars": len(full_text)}

    # --- retrieval helpers --------------------------------------------------
    @staticmethod
    def _build_context(hits):
        blocks = []
        for i, h in enumerate(hits, 1):
            src = h["metadata"].get("source", "unknown")
            blocks.append(f"[{i}] (source: {src})\n{h['text']}")
        return "\n\n---\n\n".join(blocks)

    def _condense(self, question, history, llm=None):
        """Rewrite a follow-up into a standalone search query using recent history."""
        llm = llm or self.llm
        if not history or llm is None:
            return question
        convo = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-config.HISTORY_TURNS:]
        )
        rewritten = llm.chat(
            [{"role": "user", "content": CONDENSE_PROMPT.format(history=convo, question=question)}],
            temperature=0.0, max_tokens=128,
        )
        rewritten = (rewritten or "").strip().strip('"').strip()
        return rewritten or question

    def _rerank(self, question, hits, top_k, llm):
        """Ask the LLM to order candidate passages by relevance; trim to top_k."""
        if llm is None or len(hits) <= top_k:
            return hits[:top_k]
        listing = "\n".join(f"[{i}] {h['text'][:300]}" for i, h in enumerate(hits))
        prompt = (
            f"Question: {question}\n\nCandidate passages:\n{listing}\n\n"
            f"List the passage numbers most relevant to the question, best first, "
            f"comma-separated, at most {top_k}. Numbers only."
        )
        try:
            resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
            order, seen = [], set()
            for tok in re.findall(r"\d+", resp or ""):
                idx = int(tok)
                if 0 <= idx < len(hits) and idx not in seen:
                    seen.add(idx)
                    order.append(idx)
            ranked = [hits[i] for i in order]
            ranked += [h for i, h in enumerate(hits) if i not in seen]  # keep the rest
            return ranked[:top_k]
        except Exception:  # noqa: BLE001 — re-ranking is best-effort
            return hits[:top_k]

    @staticmethod
    def cited_sources(answer, hits):
        """Sources the answer actually references (fallback: everything retrieved)."""
        retrieved = sorted({h["metadata"].get("source", "unknown") for h in hits})
        low = (answer or "").lower()
        cited = [s for s in retrieved if s.lower() in low]
        return cited or retrieved

    def prepare(self, question, history=None, top_k=None, llm=None, rerank=False):
        """Build everything needed to answer: standalone query, hits, and LLM messages."""
        history = history or []
        top_k = top_k or config.TOP_K
        search_query = self._condense(question, history, llm=llm)
        fetch_k = max(top_k, config.FETCH_K) if rerank else top_k
        hits = self.store.query(search_query, top_k=fetch_k)
        if rerank and hits:
            hits = self._rerank(question, hits, top_k, llm or self.llm)
        else:
            hits = hits[:top_k]
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
    def answer(self, question, history=None, top_k=None, llm=None, rerank=False, model=None):
        """Non-streaming answer (used by tests/CLI). Returns answer, sources, hits."""
        llm = llm or self.llm
        prep = self.prepare(question, history, top_k, llm=llm, rerank=rerank)
        if not prep["hits"]:
            return {"answer": "No documents are indexed yet — upload and index some files first.",
                    "sources": [], "hits": [], "search_query": prep["search_query"]}
        if llm is None:
            return {"answer": "No Groq API key configured — add one to get answers.",
                    "sources": [], "hits": prep["hits"], "search_query": prep["search_query"]}
        answer = llm.chat(prep["messages"], model=model)
        return {"answer": answer, "sources": self.cited_sources(answer, prep["hits"]),
                "hits": prep["hits"], "search_query": prep["search_query"]}

    def summarize(self, source, llm=None, model=None):
        """Summarize a single indexed document into a few bullet points."""
        llm = llm or self.llm
        if llm is None:
            return "No Groq API key configured — add one to summarize."
        text = self.store.get_source_text(source)
        if not text.strip():
            return "No indexed content found for that document."
        prompt = ("Summarize the following document in 5–8 concise bullet points, capturing "
                  "the key facts:\n\n" + text[:12000])
        return llm.chat([{"role": "user", "content": prompt}], model=model, max_tokens=600)
