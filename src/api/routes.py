"""
src/api/routes.py
─────────────────
FastAPI route handlers for the semantic search service.

Endpoints:
  POST /query           → run search (with cache)
  GET  /cache/stats     → cache performance metrics
  DELETE /cache         → flush the semantic cache
  GET  /health          → liveness check
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.models import (
    CacheClearResponse,
    CacheStatsResponse,
    DocumentHit,
    HealthResponse,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
)
from src.search.engine import SearchEngine, get_engine

router = APIRouter()


# ── POST /query ────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic Search",
    description=(
        "Run a semantic search over the 20 Newsgroups corpus. "
        "Results are served from the semantic cache if a similar query was "
        "previously seen (cosine similarity ≥ threshold), otherwise a FAISS "
        "nearest-neighbour search is executed and the result is cached."
    ),
    tags=["Search"],
)
async def query(
    request: QueryRequest,
    engine: SearchEngine = Depends(get_engine),
) -> QueryResponse:
    """
    Semantic search with cluster-accelerated caching.

    - **query**: Natural language search string
    - **top_k**: Number of results (1–20)
    - **bypass_cache**: Set true to force FAISS search (useful for benchmarking)
    """
    try:
        logger.info(f"POST /query | query='{request.query[:60]}' top_k={request.top_k}")

        if request.bypass_cache:
            # Temporarily monkey-patch cache threshold to 0 to force miss
            # Real production alternative: pass bypass flag down to engine
            original_threshold = engine._cache._threshold
            engine._cache._threshold = 2.0  # > 1.0 → impossible to hit
            try:
                response = engine.search(request.query, top_k=request.top_k)
            finally:
                engine._cache._threshold = original_threshold
        else:
            response = engine.search(request.query, top_k=request.top_k)

        hits = [DocumentHit.from_document_result(r) for r in response.results]

        return QueryResponse(
            query=response.query,
            results=hits,
            total_results=len(hits),
            cache_hit=response.cache_hit,
            cache_similarity=round(response.cache_similarity, 6),
            latency=LatencyBreakdown(
                total_ms=round(response.total_latency_ms, 3),
                faiss_ms=round(response.faiss_latency_ms, 3),
                cache_lookup_ms=round(response.cache_lookup_latency_ms, 3),
            ),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in /query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal search error. Check server logs.",
        )


# ── GET /cache/stats ───────────────────────────────────────────────────────────

@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache Statistics",
    description=(
        "Returns real-time statistics about the semantic cache: "
        "hit rate, entry count, per-cluster distribution, "
        "average lookup latency, and the current similarity threshold."
    ),
    tags=["Cache"],
)
async def cache_stats(
    engine: SearchEngine = Depends(get_engine),
) -> CacheStatsResponse:
    """Get semantic cache performance metrics."""
    try:
        stats_dict = engine.cache_stats()
        return CacheStatsResponse.from_dict(stats_dict)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


# ── DELETE /cache ──────────────────────────────────────────────────────────────

@router.delete(
    "/cache",
    response_model=CacheClearResponse,
    summary="Clear Cache",
    description="Flush all entries from the semantic cache. The cache will be empty after this call.",
    tags=["Cache"],
)
async def clear_cache(
    engine: SearchEngine = Depends(get_engine),
) -> CacheClearResponse:
    """Flush the semantic cache (useful for A/B testing or benchmarking)."""
    try:
        result = engine.clear_cache()
        return CacheClearResponse(
            message=f"Cache cleared successfully.",
            cleared_entries=result.get("cleared", 0),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


# ── GET /health ────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Liveness probe. Returns system status and basic stats.",
    tags=["System"],
)
async def health(
    engine: SearchEngine = Depends(get_engine),
) -> HealthResponse:
    """Kubernetes-style liveness probe."""
    try:
        stats = engine.cache_stats()
        n_docs = engine._store.n_docs if engine._store else 0
        n_clusters = engine._gmm.n_components if engine._gmm else 0

        return HealthResponse(
            status="healthy",
            n_docs=n_docs,
            n_clusters=n_clusters,
            cache_entries=stats.get("total_entries", 0),
            version="1.0.0",
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            n_docs=0,
            n_clusters=0,
            cache_entries=0,
            version="1.0.0",
        )
