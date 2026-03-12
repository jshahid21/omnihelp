# =============================================================================
# Omni-Help — Frontend (Streamlit)
# =============================================================================
# The frontend image is intentionally kept separate from the backend so each
# service can be scaled and deployed independently on EC2 or ECS.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: dependency builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local

# Only the frontend source is needed — no data/ directory required
COPY src/ ./src/

# API_URL is injected at runtime by docker-compose (set to http://backend:8000).
# The default below is the local dev fallback when running without Docker.
ENV API_URL="http://localhost:8000"

# Expose Streamlit port
EXPOSE 8501

# --server.address=0.0.0.0 makes Streamlit reachable outside the container.
# --server.headless=true suppresses the "Do you want to receive email?" prompt.
CMD ["streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
