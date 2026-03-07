"""
src/search/engine.py
────────────────────
SearchEngine: the central orchestrator connecting all subsystems.

Request flow:
  1. Encode query → L2-normalised embedding
  2. Check SemanticCache → if HIT, return cached result instantly
  3. If MISS → search FAISS index for top-k similar documents
  4. Append cluster metadata to each result (soft memberships from GMM)
  5. Insert (query, result) into cache for future hits
  6. Return structured SearchResponse

Design notes:
- The engine is stateless per-request; all state lives in sub-components.
- Startup loading (FAISS index, GMM, metadata) is handled once via load_artifacts().
- All sub-components accept dependency injection for easy testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from config.settings import settings
from src.cache.semantic_cache import SemanticCache
from src.clustering.fuzzy_gmm import FuzzyGMM
from src.embeddings.encoder import Encoder
from src.vectordb.store import FAISSVectorStore


# ── Response types ─────────────────────────────────────────────────────────────

@dataclass
class DocumentResult:
    """A single retrieved document with metadata and soft cluster memberships."""
    doc_id: int
    rank: int
    score: float                        # cosine similarity to query
    category: str
    subject: str
    text_snippet: str                   # first 300 chars for display
    cluster_memberships: dict[str, float]  # {cluster_id: probability}


@dataclass
class SearchResponse:
    """Full response from the search engine."""
    query: str
    results: list[DocumentResult]
    cache_hit: bool
    cache_similarity: float             # cosine sim of cached query (0 if miss)
    total_latency_ms: float
    faiss_latency_ms: float
    cache_lookup_latency_ms: float


# ── Engine ─────────────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Orchestrates encoding → cache → FAISS retrieval → cluster enrichment.

    Usage:
        engine = SearchEngine()
        engine.load_artifacts()             # load FAISS index + GMM + metadata
        response = engine.search("Linux graphics drivers", top_k=5)
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        vector_store: FAISSVectorStore | None = None,
        gmm: FuzzyGMM | None = None,
        cache: SemanticCache | None = None,
        doc_metadata: list[dict] | None = None,
    ) -> None:
        # Allow dependency injection for testing; otherwise lazy-loaded
        self._encoder = encoder or Encoder()
        self._store: FAISSVectorStore | None = vector_store
        self._gmm: FuzzyGMM | None = gmm
        self._cache: SemanticCache | None = cache
        self._doc_metadata: list[dict] = doc_metadata or []
        self._ready = False

    # ── Startup ────────────────────────────────────────────────────────────

    def load_artifacts(self) -> None:
        """
        Load FAISS index, GMM model, and doc metadata from disk.
        Called once at FastAPI startup via lifespan.
        """
        logger.info("Loading SearchEngine artifacts...")

        # FAISS vector store
        faiss_path = settings.faiss_index_path
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. "
                "Run `python scripts/ingest.py` first."
            )
        self._store = FAISSVectorStore.load(faiss_path)

        # Fuzzy GMM
        gmm_path = settings.gmm_model_path
        pca_path = settings.pca_model_path
        if not gmm_path.exists() or not pca_path.exists():
            raise FileNotFoundError(
                "GMM or PCA model not found. Run `python scripts/ingest.py` first."
            )
        self._gmm = FuzzyGMM.load(gmm_path, pca_path)

        # Document metadata (list of dicts with keys: doc_id, category, subject, text)
        import pickle
        meta_path = settings.doc_metadata_path
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Doc metadata not found at {meta_path}. "
                "Run `python scripts/ingest.py` first."
            )
        with open(meta_path, "rb") as f:
            self._doc_metadata = pickle.load(f)

        # Semantic cache (fresh on each server start)
        self._cache = SemanticCache(gmm=self._gmm)

        self._ready = True
        logger.success(
            f"SearchEngine ready: {self._store.n_docs} docs, "
            f"{self._gmm.n_components} clusters, cache threshold={settings.cache_similarity_threshold}"
        )

    # ── Main search ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = settings.default_top_k) -> SearchResponse:
        """
        Execute a semantic search query with cache-aware retrieval.

        Args:
            query:  Natural language query string.
            top_k:  Number of results to return.

        Returns:
            SearchResponse with results, cache status, and latency breakdown.
        """
        self._check_ready()
        t_total = time.perf_counter()

        # Step 1: Encode query
        q_emb = self._encoder.encode_single(query)[0]  # shape (dim,)

        # Step 2: Cache lookup
        t_cache = time.perf_counter()
        cache_result = self._cache.lookup(q_emb)  # type: ignore[union-attr]
        cache_latency_ms = _ms(t_cache)

        if cache_result.hit:
            # Return cached search response, updating latency metadata
            cached_response: SearchResponse = cache_result.entry.result  # type: ignore
            total_ms = _ms(t_total)
            logger.info(
                f"→ CACHE HIT [{query[:50]}] "
                f"sim={cache_result.similarity:.4f} latency={total_ms:.1f}ms"
            )
            return SearchResponse(
                query=query,
                results=cached_response.results,
                cache_hit=True,
                cache_similarity=cache_result.similarity,
                total_latency_ms=total_ms,
                faiss_latency_ms=0.0,
                cache_lookup_latency_ms=cache_latency_ms,
            )

        # Step 3: FAISS retrieval
        t_faiss = time.perf_counter()
        faiss_results = self._store.search(q_emb, top_k=top_k)  # type: ignore[union-attr]
        faiss_latency_ms = _ms(t_faiss)

        # Step 4: Enrich results with cluster memberships
        doc_results = self._build_results(faiss_results, q_emb)

        response = SearchResponse(
            query=query,
            results=doc_results,
            cache_hit=False,
            cache_similarity=cache_result.similarity,
            total_latency_ms=_ms(t_total),
            faiss_latency_ms=faiss_latency_ms,
            cache_lookup_latency_ms=cache_latency_ms,
        )

        # Step 5: Insert into cache
        self._cache.insert(query, q_emb, response)  # type: ignore[union-attr]

        logger.info(
            f"→ CACHE MISS [{query[:50]}] "
            f"faiss={faiss_latency_ms:.1f}ms total={response.total_latency_ms:.1f}ms"
        )
        return response

    def cache_stats(self) -> dict:
        """Return current cache statistics."""
        self._check_ready()
        return self._cache.stats()  # type: ignore[union-attr]

    def clear_cache(self) -> dict:
        """Clear the semantic cache and return number of evicted entries."""
        self._check_ready()
        return self._cache.clear()  # type: ignore[union-attr]

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_results(self, faiss_results, q_emb: np.ndarray) -> list[DocumentResult]:
        """
        Enrich FAISS search results with document metadata and GMM memberships.
        """
        output = []
        for r in faiss_results:
            if r.doc_id >= len(self._doc_metadata):
                continue
            meta = self._doc_metadata[r.doc_id]

            # Get soft cluster memberships for this document
            # (Pre-computed during ingestion and stored in metadata)
            cluster_probs = meta.get("cluster_probs", {})

            output.append(DocumentResult(
                doc_id=r.doc_id,
                rank=r.rank,
                score=round(r.score, 6),
                category=meta.get("category", "unknown"),
                subject=meta.get("subject", ""),
                text_snippet=meta.get("text", "")[:300],
                cluster_memberships=cluster_probs,
            ))
        return output

    def _check_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "SearchEngine not initialised. Call load_artifacts() first."
            )


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# Module-level singleton used by the FastAPI app
_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    """FastAPI dependency: returns the global engine singleton."""
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine
