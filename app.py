"""Streamlit demo UI — history-aware, streaming, multi-modal RAG with optional BYO key."""
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
from rag.llm import GroqClient
from rag.ingest import file_to_chunks
from rag import config

st.set_page_config(page_title="Multi-Modal RAG", page_icon="📄", layout="wide")

GROQ_KEYS_URL = "https://console.groq.com/keys"
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff",
                "txt", "md", "csv", "json", "log", "html"]


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


def resolve_llm(pipe, user_key):
    """Use the visitor's own key if they entered one; else the app's configured key."""
    user_key = (user_key or "").strip()
    if user_key:
        try:
            return GroqClient(api_keys=[user_key])
        except Exception:  # noqa: BLE001
            return None
    return pipe.llm


def index_files(pipe, files, llm):
    progress = st.progress(0.0)
    for idx, f in enumerate(files, 1):
        with st.spinner(f"Processing {f.name}…"):
            suffix = os.path.splitext(f.name)[1] or ".bin"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getbuffer())
                    tmp_path = tmp.name
                _, chunks = file_to_chunks(tmp_path, llm=llm)
                pipe.store.add_chunks(chunks, source=f.name)
                st.success(f"{f.name}: indexed {len(chunks)} chunks")
            except Exception as e:  # noqa: BLE001
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

    pipe = get_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("🔑 Groq API key")
        st.markdown(
            f"Optional — bring your own so you're not sharing the demo's quota. "
            f"[Get a free key →]({GROQ_KEYS_URL})"
        )
        user_key = st.text_input(
            "Groq API key", type="password", placeholder="gsk_…",
            label_visibility="collapsed",
        )
        llm = resolve_llm(pipe, user_key)
        if (user_key or "").strip():
            if llm is None:
                st.error("That key doesn't look valid.")
            else:
                st.success("Using your key ✓")
        elif pipe.llm is not None:
            st.caption("No key entered — using the demo's key.")
        else:
            st.warning("Add a key above to start chatting.")

        st.divider()
        st.header("📥 Documents")
        files = st.file_uploader("Upload files", type=UPLOAD_TYPES, accept_multiple_files=True)
        if st.button("Index uploaded files", type="primary", disabled=not files):
            index_files(pipe, files, llm)

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
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if llm is None:
                answer = (f"Add a Groq API key in the sidebar to get answers — "
                          f"[grab a free one]({GROQ_KEYS_URL}).")
                st.markdown(answer)
            else:
                with st.spinner("Searching documents…"):
                    prep = pipe.prepare(question, history=history, llm=llm)
                if not prep["hits"]:
                    answer = "No documents are indexed yet — upload and index some files first."
                    st.markdown(answer)
                else:
                    try:
                        answer = st.write_stream(llm.chat_stream(prep["messages"]))
                        render_sources(prep, pipe.cited_sources(answer, prep["hits"]))
                    except Exception as e:  # noqa: BLE001
                        answer = f"⚠️ Couldn't reach Groq — check the API key. ({e})"
                        st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
