"""
src/api/app.py
──────────────
FastAPI application factory with lifespan management.

Design decisions:
- Using FastAPI's async lifespan context manager (replaces deprecated on_event)
  for startup/shutdown hooks. This correctly handles resource cleanup on SIGTERM.
- The SearchEngine singleton is loaded once at startup and injected via Depends().
- CORS middleware is permissive (allow all origins) for dev. In prod, restrict to
  your frontend domain.
- Structured logging via loguru's intercept_handler so uvicorn logs are unified
  with application logs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings
from src.api.routes import router
from src.search.engine import get_engine


# ── Lifespan: startup + shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI async lifespan context manager.
    - On startup: load FAISS index, GMM model, doc metadata.
    - On shutdown: log graceful cleanup (FAISS/GMM hold no external connections).
    """
    logger.info("=== Semantic Search Service Starting ===")
    engine = get_engine()

    try:
        engine.load_artifacts()
        logger.success(
            f"Service ready. Docs: {engine._store.n_docs}, "  # type: ignore[union-attr]
            f"Clusters: {engine._gmm.n_components}"  # type: ignore[union-attr]
        )
    except FileNotFoundError as e:
        logger.error(
            f"Startup failed: {e}\n"
            "→ Run `python scripts/ingest.py` to build the index first."
        )
        # Still yield so the server starts (health endpoint returns degraded)

    yield  # ← server runs here

    logger.info("=== Semantic Search Service Shutting Down ===")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    Separate factory function makes the app testable (no side effects at import).
    """
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
## Semantic Search over 20 Newsgroups

A production-grade semantic search service combining:

- 🔍 **Dense Vector Retrieval**: SentenceTransformers + FAISS IVFFlat
- 🎭 **Fuzzy Clustering**: Gaussian Mixture Models with soft memberships
- ⚡ **Semantic Cache**: Cluster-accelerated O(N/K) cosine similarity cache

### Key Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Search with semantic cache |
| `GET`  | `/cache/stats` | Cache hit rate, cluster distribution |
| `DELETE` | `/cache` | Flush the semantic cache |
| `GET`  | `/health` | Liveness probe |

### Cache Behaviour
Queries with cosine similarity ≥ **0.85** to a previously-seen query will be
served from cache — including paraphrases. Cache grows in O(N/K) lookup time
via GMM cluster routing.
        """,
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc UI
        openapi_url="/openapi.json",
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # tighten this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────────────
    app.include_router(router)

    return app


# ── Module-level app instance (for uvicorn) ────────────────────────────────────
app = create_app()


# ── Development entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
