# ── Build stage ────────────────────────────────────────────────────────────────
# Using a multi-stage build to keep the final image small:
# Stage 1 installs deps, Stage 2 copies only what's needed.

FROM python:3.11-slim AS builder

# System dependencies for FAISS, scipy, and umap-learn compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies
# Copying requirements before source enables Docker layer caching:
# deps only reinstall when requirements.txt changes
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libs (OpenBLAS needed by FAISS/numpy at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/        ./src/
COPY config/     ./config/
COPY scripts/    ./scripts/
COPY artifacts/  ./artifacts/

# Copy env template (users mount .env at runtime; we use defaults if absent)
COPY .env.example .env.example

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

# Health check: ping the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# NOTE: Artifacts (FAISS index, GMM model) must be present inside the image
# or mounted as a volume: docker run -v $(pwd)/artifacts:/app/artifacts ...
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
