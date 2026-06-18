"""Central configuration. Values come from the environment / a local .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root regardless of the current working directory.
load_dotenv(PROJECT_ROOT / ".env")

# --- Groq -------------------------------------------------------------------
# Comma-separated API keys. Supplying more than one enables automatic failover
# when a key hits Groq's (generous but finite) free-tier rate limit.
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

# Models — both available on the Groq free tier.
TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# --- Chunking / retrieval ---------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
# How many prior chat turns to use when condensing a follow-up and prompting the answer.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))

# --- Local vector store -----------------------------------------------------
STORAGE_DIR = os.getenv("STORAGE_DIR") or str(PROJECT_ROOT / "storage")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
