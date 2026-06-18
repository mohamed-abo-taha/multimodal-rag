# Multi-modal RAG — container image for the Streamlit app.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rag/ ./rag/
COPY app.py .

# Bake the local embedding model into the image so the first query isn't slow.
# Call the function on a string so the model actually downloads at build time
# (instantiating alone is lazy and won't fetch it).
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.DefaultEmbeddingFunction()(['warmup'])"

EXPOSE 8501

# Provide keys at runtime:
#   docker run -e GROQ_API_KEYS=gsk_xxx -p 8501:8501 -v rag_storage:/app/storage rag-system
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
