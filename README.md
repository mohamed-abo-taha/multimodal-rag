# Multi-Modal RAG — Document Intelligence

Ingest **PDFs, images, and text**, then **chat** about them — answers come **only** from your
documents (Retrieval-Augmented Generation), and follow-up questions remember the conversation.
Built to run **locally**:

| Layer        | Choice                                          |
|--------------|-------------------------------------------------|
| LLM          | **Groq** `llama-3.3-70b-versatile`              |
| Image OCR    | **Groq** `llama-4-scout` vision (no Tesseract)  |
| Embeddings   | **all-MiniLM-L6-v2** via Chroma (local, on CPU) |
| Vector DB    | **Chroma** (local, persistent)                  |
| UI           | **Streamlit** (streaming answers)               |

## Features

- **Multi-modal ingest** — PDFs (pypdf), images (Groq vision OCR), and text/markdown.
- **Conversational** — follow-ups like *"what about the stipend?"* are rewritten into a
  standalone search query using chat history, so retrieval stays accurate across turns.
- **Streaming answers** in the web UI.
- **Grounded + cited** — answers use only retrieved context and name the source file(s) used.
- **Local embeddings** — run on your CPU; only Groq generation hits the network.
- **Bring your own key** — visitors can optionally paste their own Groq key in the sidebar, so a public demo doesn't share its quota.
- **Resilient** — multiple Groq keys with automatic failover on rate limits.

## How it works

```
            ┌─────────── ingest ───────────┐
 PDF ─┐     │ pypdf                          │
 IMG ─┼──▶  │ Groq vision (OCR + describe)   │──▶ chunk ──▶ embed (MiniLM) ──▶ Chroma
 TXT ─┘     │ utf-8 read                     │
            └───────────────────────────────┘

 follow-up + history ─▶ condense to a standalone query ─▶ retrieve top-k from Chroma ─┐
                                                                                      ▼
            streamed, grounded answer + cited sources ◀── Groq LLM ◀── context + history + question
```

Code map:
- `rag/ingest.py` — file → text (PDF / image / text)
- `rag/splitter.py` — dependency-free chunker (unit-tested in `tests/`)
- `rag/vectorstore.py` — Chroma + **local** MiniLM embeddings
- `rag/llm.py` — Groq client: chat, streaming, vision, multi-key failover
- `rag/pipeline.py` — condense → retrieve → grounded answer
- `app.py` — Streamlit chat UI

## Setup

```powershell
# from C:\Users\Venom\Desktop\rag-system
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env   # then paste your Groq key(s) into .env
```

Get a Groq key at <https://console.groq.com/keys>.

## Run

```powershell
# Web app (streaming chat UI):
.venv\Scripts\python.exe -m streamlit run app.py

# --- checks / demos (no UI) ---
.venv\Scripts\python.exe smoke_test.py          # text Q&A
.venv\Scripts\python.exe make_samples.py        # create a sample PDF + image
.venv\Scripts\python.exe test_multimodal.py     # PDF extraction + image OCR
.venv\Scripts\python.exe test_conversation.py   # multi-turn follow-up questions

# unit tests:
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
```

Then upload files in the sidebar, click **Index**, and chat.

## Cost analysis (the economics)

The expensive parts of most RAG systems — embeddings and the LLM — are inexpensive here:

- **Embeddings run locally** on CPU (MiniLM). Indexing 1,000 pages costs **$0** and makes no
  network calls. A commercial embedding API (e.g. OpenAI `text-embedding-3-small` at
  ~$0.02 / 1M tokens) would cost ~**$0.01–0.02** for the same 1,000 pages.
- **Generation runs on Groq.** A typical answer sends ~2–4k context tokens and returns ~300.
  `llama-3.3-70b` is ~$0.59/$0.79 per 1M in/out tokens, i.e. **~$0.002 per question**. Each
  follow-up also spends one short (~150-token) call to condense the query. Per-key rate limits
  are the real constraint, which is why this ships with **multi-key failover**.
- **Scaling up:** move embeddings to a batch job, add a self-hosted model, and put Chroma behind
  a server or swap in a managed vector DB (Pinecone/Weaviate). Cost stays dominated by generation
  tokens, so the lever is retrieval quality — fewer, better chunks.

## Deployment

**Docker:**
```bash
docker build -t rag-system .
docker run -e GROQ_API_KEYS=your_key_here -p 8501:8501 -v rag_storage:/app/storage rag-system
# open http://localhost:8501
```
The image bakes in the local embedding model, so the first query is fast. The `rag_storage`
volume persists the index across restarts.

**Streamlit Community Cloud:** push to GitHub, add `GROQ_API_KEYS` as a secret.

Keep secrets in environment variables — never commit `.env` (it is git-ignored).

## Security

- API keys live only in `.env` (git-ignored). Rotate them at <https://console.groq.com/keys>
  if they're ever exposed.
- Indexed content is stored unencrypted under `storage/` on your machine.
