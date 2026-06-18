"""Streamlit demo UI — history-aware, streaming, multi-modal RAG."""
import os
import sys
import tempfile

# Make the local `rag` package importable no matter where Streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# On Streamlit Community Cloud, secrets are set in the dashboard (st.secrets). Mirror
# them into the environment so rag/config.py (which reads os.getenv) picks them up.
# Harmless locally, where the .env file is used instead.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from rag.pipeline import RAGPipeline
from rag.ingest import file_to_chunks
from rag import config

st.set_page_config(page_title="Multi-Modal RAG", page_icon="📄", layout="wide")

UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff",
                "txt", "md", "csv", "json", "log", "html"]


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


def index_files(pipe, files):
    progress = st.progress(0.0)
    for idx, f in enumerate(files, 1):
        with st.spinner(f"Processing {f.name}…"):
            suffix = os.path.splitext(f.name)[1] or ".bin"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getbuffer())
                    tmp_path = tmp.name
                _, chunks = file_to_chunks(tmp_path, llm=pipe.llm)
                pipe.store.add_chunks(chunks, source=f.name)
                st.success(f"{f.name}: indexed {len(chunks)} chunks")
            except Exception as e:  # noqa: BLE001 — surface any failure per-file
                st.error(f"{f.name}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        progress.progress(idx / len(files))


def render_sources(prep, sources):
    if sources:
        st.caption("📎 Sources: " + ", ".join(sources))
    with st.expander("🔍 Retrieved context"):
        st.caption(f"Search query used: _{prep['search_query']}_")
        for i, h in enumerate(prep["hits"], 1):
            src = h["metadata"].get("source", "?")
            dist = h.get("distance")
            dist_s = f" · distance {dist:.3f}" if isinstance(dist, (int, float)) else ""
            st.markdown(f"**{i}. {src}**{dist_s}")
            snippet = h["text"][:800] + ("…" if len(h["text"]) > 800 else "")
            st.text(snippet)


def main():
    st.title("📄 Multi-Modal RAG — Document Intelligence")
    st.caption(
        "Ingest PDFs, images, and text, then chat about them. Answers are grounded in your "
        "documents and cite their source. LLM: Groq · Embeddings: local MiniLM · DB: Chroma."
    )

    if not config.GROQ_API_KEYS:
        st.error("No Groq API keys found. Add GROQ_API_KEYS to your .env file and restart.")
        st.stop()

    pipe = get_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("📥 Documents")
        files = st.file_uploader("Upload files", type=UPLOAD_TYPES, accept_multiple_files=True)
        if st.button("Index uploaded files", type="primary", disabled=not files):
            index_files(pipe, files)

        st.divider()
        st.subheader("Indexed sources")
        sources = pipe.store.list_sources()
        if sources:
            for s, c in sources.items():
                st.write(f"• **{s}** — {c} chunks")
            if st.button("🗑️ Clear index"):
                pipe.store.reset()
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("Nothing indexed yet.")

        if st.session_state.messages and st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about your documents…")
    if question:
        history = list(st.session_state.messages)  # turns prior to this question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                prep = pipe.prepare(question, history=history)
            if not prep["hits"]:
                answer = "No documents are indexed yet — upload and index some files first."
                st.markdown(answer)
            else:
                answer = st.write_stream(pipe.llm.chat_stream(prep["messages"]))
                render_sources(prep, pipe.cited_sources(answer, prep["hits"]))
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
