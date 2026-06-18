"""Chroma-backed vector store. Embeddings are computed locally (free, no API key).

Chroma's DefaultEmbeddingFunction runs all-MiniLM-L6-v2 via ONNX on the CPU — it
downloads a small (~80 MB) model on first use and needs no network afterwards.
"""
import chromadb
from chromadb.utils import embedding_functions

from . import config


class VectorStore:
    def __init__(self, storage_dir=None, collection_name=None):
        storage_dir = storage_dir or config.STORAGE_DIR
        collection_name = collection_name or config.COLLECTION_NAME
        self.client = chromadb.PersistentClient(path=storage_dir)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks, source, extra_metadata=None):
        """Upsert chunks for one source document. Re-ingesting replaces old chunks."""
        if not chunks:
            return 0
        base = extra_metadata or {}
        ids = [f"{source}::chunk-{i}" for i in range(len(chunks))]
        metas = [{"source": source, "chunk": i, **base} for i in range(len(chunks))]
        self.collection.upsert(ids=ids, documents=list(chunks), metadatas=metas)
        return len(chunks)

    def query(self, question, top_k=None):
        top_k = top_k or config.TOP_K
        res = self.collection.query(query_texts=[question], n_results=top_k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            {"text": d, "metadata": m or {}, "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

    def list_sources(self):
        """Map of source filename -> number of indexed chunks."""
        got = self.collection.get(include=["metadatas"])
        counts = {}
        for m in got.get("metadatas") or []:
            s = (m or {}).get("source", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def count(self):
        return self.collection.count()

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
