# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Don't write .pyc files; don't buffer stdout, so logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first: sentence-transformers would otherwise pull the CUDA
# build (~2 GB of wheels this container can't use).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies in their own layer, so editing code doesn't re-install them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the reranker into the image: no Hugging Face download on first query,
# and at runtime the container needs network access only to OpenAI.
# Keep the model name in sync with RERANKER_MODEL in rag.py.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY api.py app.py ingest.py rag.py evaluate.py ./

# Run as a non-root user. /app/chroma_db is the volume mount point and must
# exist with the right owner before the volume is attached.
RUN useradd --create-home appuser \
    && mkdir -p /app/chroma_db /app/data \
    && chown -R appuser:appuser /app /opt/hf
USER appuser

EXPOSE 8000 8501

# Default command runs the API; compose overrides this for the UI and ingest.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
