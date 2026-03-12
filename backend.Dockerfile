# =============================================================================
# Omni-Help — Backend (FastAPI + LangGraph)
# =============================================================================
# Multi-stage build keeps the final image lean by separating dependency
# installation from the runtime layer.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: dependency builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some LangChain/ChromaDB native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder — keeps final image free of build tools
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/

# Copy pre-seeded data directory:
#   data/policies/ — source markdown files for ChromaDB ingestion
#   data/db/       — pre-seeded SQLite orders database (run init_db.py locally first)
#   data/vectors/  — pre-ingested ChromaDB embeddings (run ingest.py locally first)
# If these don't exist yet, run:
#   python src/utils/init_db.py
#   python src/utils/ingest.py
# then rebuild the image.
COPY data/ ./data/

# Expose FastAPI port
EXPOSE 8000

# Uvicorn serves the FastAPI app.
# --app-dir src tells uvicorn to add src/ to the Python path so
# relative imports (from graph.graph import ...) resolve correctly.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
