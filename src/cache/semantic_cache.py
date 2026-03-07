"""
src/cache/semantic_cache.py
───────────────────────────
From-scratch semantic cache with cluster-accelerated lookup.

Architecture:
─────────────────────────────────────────────────────────────────────────────
The Semantic Cache solves the "paraphrase hit" problem:
  - "How do I install Linux?" ≡ "What are the steps to set up Linux?"
  - These should return the same cached result.
  - Exact-string matching (like Redis) fails here.

Design: Cluster-Partitioned FAISS Mini-Index
─────────────────────────────────────────────────────────────────────────────

                         ┌─────────────────────────────────┐
  incoming query ─→ GMM ─│ top-3 most probable clusters    │
                         └───────────────┬─────────────────┘
                                         │
                         ┌───────────────▼─────────────────┐
                         │ cluster_index: {cluster_id →    │
                         │   [cache_entry_ids]}            │
                         └───────────────┬─────────────────┘
                                         │  candidate set (≪ N)
                         ┌───────────────▼─────────────────┐
                         │ FAISS mini-index: cosine sim    │
                         │ over candidate embeddings only  │
                         └───────────────┬─────────────────┘
                                         │ max_score
                         ┌───────────────▼─────────────────────────────────┐
                         │ if max_score ≥ threshold: return cached result  │
                         │ else: run full FAISS search + add to cache       │
                         └──────────────────────────────────────────────────┘

Complexity Analysis:
- Naive cosine scan: O(N) per query
- Cluster-routed:   O(N/K) where K = number of clusters
  For K=20 clusters and N=10,000 cache entries: 500 comparisons vs 10,000

Additional features:
- LRU-style max size eviction (oldest entries removed)
- Thread-safe with RLock for concurrent FastAPI requests
- Hit/miss/latency statistics for monitoring
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
from loguru import logger

from config.settings import settings
from src.clustering.fuzzy_gmm import FuzzyGMM


# ── Data contract ──────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A single cached query → result pair."""
    entry_id: int
    query: str                      # original query text
    query_embedding: np.ndarray     # shape (dim,), L2-normalised
    result: Any                     # the stored search results
    cluster_id: int                 # dominant GMM cluster
    cluster_probs: np.ndarray       # full probability distribution (K,)
    inserted_at: float = field(default_factory=time.time)
    hit_count: int = 0              # how many times this entry was returned


@dataclass
class CacheLookupResult:
    """Result of a cache lookup operation."""
    hit: bool
    entry: CacheEntry | None
    similarity: float
    lookup_latency_ms: float
    candidates_checked: int


# ── Main cache class ───────────────────────────────────────────────────────────

class SemanticCache:
    """
    Cluster-accelerated semantic cache using GMM routing + FAISS mini-index.

    Thread-safe for concurrent FastAPI request handling.
    """

    def __init__(
        self,
        gmm: FuzzyGMM,
        dim: int = settings.embedding_dim,
        similarity_threshold: float = settings.cache_similarity_threshold,
        max_size: int = settings.cache_max_size,
        top_k_clusters: int = settings.cache_top_k_clusters,
    ) -> None:
        self._gmm = gmm
        self._dim = dim
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._top_k_clusters = top_k_clusters

        # ── Storage ────────────────────────────────────────────────────────
        self._entries: list[CacheEntry] = []
        self._entry_counter = 0

        # cluster_id → list of entry_ids in that cluster
        self._cluster_index: dict[int, list[int]] = defaultdict(list)

        # Insertion order for eviction (oldest first)
        self._insertion_order: deque[int] = deque()

        # FAISS flat index over ALL cached query embeddings (for global
        # fallback if cluster routing yields no candidates)
        self._global_index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)

        # Mapping: FAISS internal row → entry_id
        self._faiss_row_to_entry: list[int] = []

        # ── Statistics ─────────────────────────────────────────────────────
        self._hits = 0
        self._misses = 0
        self._total_lookup_ms = 0.0

        # ── Threading ──────────────────────────────────────────────────────
        self._lock = threading.RLock()

    # ── Public interface ───────────────────────────────────────────────────

    def lookup(self, query_embedding: np.ndarray) -> CacheLookupResult:
        """
        Search the cache for a semantically similar query.

        Algorithm:
        1. Get top-K cluster IDs from GMM for the query embedding.
        2. Collect candidate entry IDs from those clusters.
        3. If candidates exist: compute cosine sim via FAISS sub-index.
        4. If no candidates: fall back to global FAISS index.
        5. Return hit if best similarity ≥ threshold.

        Args:
            query_embedding: shape (dim,) or (1, dim), L2-normalised.

        Returns:
            CacheLookupResult with hit status and matched entry (if hit).
        """
        t0 = time.perf_counter()

        with self._lock:
            if len(self._entries) == 0:
                self._misses += 1
                return CacheLookupResult(
                    hit=False, entry=None, similarity=0.0,
                    lookup_latency_ms=_ms(t0), candidates_checked=0
                )

            q = np.atleast_1d(query_embedding).astype(np.float32).reshape(1, -1)

            # Step 1: get top-K cluster IDs via GMM
            top_clusters = self._gmm.top_k_clusters(q, k=self._top_k_clusters)
            candidate_ids = self._get_candidates(top_clusters)

            candidates_checked = len(candidate_ids)

            # Step 2: if cluster routing found candidates, score them
            if candidate_ids:
                best_entry, best_score = self._score_candidates(q, candidate_ids)
            else:
                # Fallback: global FAISS scan (rare: happens when cluster
                # routing returns empty clusters because cache just started)
                best_entry, best_score, candidates_checked = self._global_fallback(q)

            if best_score >= self._threshold and best_entry is not None:
                best_entry.hit_count += 1
                self._hits += 1
                latency = _ms(t0)
                self._total_lookup_ms += latency
                logger.debug(
                    f"Cache HIT: similarity={best_score:.4f}, "
                    f"candidates={candidates_checked}, latency={latency:.1f}ms"
                )
                return CacheLookupResult(
                    hit=True, entry=best_entry, similarity=best_score,
                    lookup_latency_ms=latency, candidates_checked=candidates_checked
                )

            self._misses += 1
            latency = _ms(t0)
            self._total_lookup_ms += latency
            logger.debug(
                f"Cache MISS: best_sim={best_score:.4f}, "
                f"candidates={candidates_checked}, latency={latency:.1f}ms"
            )
            return CacheLookupResult(
                hit=False, entry=None, similarity=best_score,
                lookup_latency_ms=latency, candidates_checked=candidates_checked
            )

    def insert(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any,
    ) -> CacheEntry:
        """
        Add a new query/result pair to the cache.

        Automatically evicts oldest entry if max_size is exceeded.

        Args:
            query:           Original query text.
            query_embedding: L2-normalised embedding, shape (dim,) or (1, dim).
            result:          Arbitrary result to cache (search response dict).

        Returns:
            The newly created CacheEntry.
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._entries) >= self._max_size:
                self._evict_oldest()

            vec = np.atleast_1d(query_embedding).astype(np.float32).flatten()

            # Get GMM cluster assignment
            top_clusters = self._gmm.top_k_clusters(vec, k=1)
            dominant_cluster = top_clusters[0][0]
            cluster_probs = self._gmm.predict_proba(vec.reshape(1, -1))[0]

            entry_id = self._entry_counter
            self._entry_counter += 1

            entry = CacheEntry(
                entry_id=entry_id,
                query=query,
                query_embedding=vec,
                result=result,
                cluster_id=dominant_cluster,
                cluster_probs=cluster_probs,
            )

            self._entries.append(entry)
            self._cluster_index[dominant_cluster].append(entry_id)
            self._insertion_order.append(entry_id)

            # Add to global FAISS index
            self._global_index.add(vec.reshape(1, -1))
            self._faiss_row_to_entry.append(entry_id)

            logger.debug(
                f"Cache INSERT: entry_id={entry_id}, cluster={dominant_cluster}, "
                f"cache_size={len(self._entries)}"
            )
            return entry

    def clear(self) -> dict[str, int]:
        """Flush all cache entries and reset statistics."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._cluster_index.clear()
            self._insertion_order.clear()
            self._faiss_row_to_entry.clear()
            self._global_index = faiss.IndexFlatIP(self._dim)
            self._hits = 0
            self._misses = 0
            self._total_lookup_ms = 0.0
            self._entry_counter = 0
            logger.info(f"Cache cleared: removed {n} entries")
            return {"cleared": n}

    def stats(self) -> dict:
        """Return cache performance metrics for the monitoring endpoint."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / max(total, 1)
            avg_latency = self._total_lookup_ms / max(total, 1)

            # Per-cluster entry counts
            cluster_dist = {
                f"cluster_{k}": len(v)
                for k, v in sorted(self._cluster_index.items())
            }

            return {
                "total_entries": len(self._entries),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "avg_lookup_latency_ms": round(avg_latency, 3),
                "similarity_threshold": self._threshold,
                "cluster_distribution": cluster_dist,
                "top_k_clusters_searched": self._top_k_clusters,
            }

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_candidates(self, top_clusters: list[tuple[int, float]]) -> list[int]:
        """
        Collect entry IDs from the top-K most probable clusters.

        Only includes entries from clusters with meaningful probability mass
        (prob > 0.05) to avoid spurious matches in distant clusters.
        """
        MIN_PROB = 0.05
        candidate_ids: set[int] = set()
        for cluster_id, prob in top_clusters:
            if prob > MIN_PROB:
                candidate_ids.update(self._cluster_index.get(cluster_id, []))
        return list(candidate_ids)

    def _score_candidates(
        self,
        q: np.ndarray,
        candidate_ids: list[int],
    ) -> tuple[CacheEntry | None, float]:
        """
        Score only the candidate entries using cosine similarity via dot product.
        (Vectors are L2-normalised → dot product = cosine similarity)
        """
        # Build a tiny on-the-fly matrix of candidate embeddings
        entry_map = {e.entry_id: e for e in self._entries}
        candidates = [entry_map[eid] for eid in candidate_ids if eid in entry_map]

        if not candidates:
            return None, 0.0

        # Stack embeddings into a matrix and score with a single dot product
        cand_matrix = np.stack([c.query_embedding for c in candidates])  # (M, dim)
        scores = (cand_matrix @ q.T).flatten()  # cosine similarities

        best_idx = int(np.argmax(scores))
        return candidates[best_idx], float(scores[best_idx])

    def _global_fallback(
        self, q: np.ndarray
    ) -> tuple[CacheEntry | None, float, int]:
        """Fall back to global FAISS index search when cluster routing fails."""
        if self._global_index.ntotal == 0:
            return None, 0.0, 0

        scores, indices = self._global_index.search(q, 1)
        faiss_row = indices[0][0]
        if faiss_row == -1:
            return None, 0.0, 0

        entry_id = self._faiss_row_to_entry[faiss_row]
        entry_map = {e.entry_id: e for e in self._entries}
        entry = entry_map.get(entry_id)
        return entry, float(scores[0][0]), self._global_index.ntotal

    def _evict_oldest(self) -> None:
        """Remove the oldest cache entry (FIFO eviction policy)."""
        if not self._insertion_order:
            return

        oldest_id = self._insertion_order.popleft()
        # Remove from entries list
        self._entries = [e for e in self._entries if e.entry_id != oldest_id]
        # Remove from cluster index
        for cluster_list in self._cluster_index.values():
            if oldest_id in cluster_list:
                cluster_list.remove(oldest_id)

        logger.debug(f"Cache eviction: removed entry_id={oldest_id}")


def _ms(t0: float) -> float:
    """Return elapsed milliseconds since t0 (perf_counter)."""
    return (time.perf_counter() - t0) * 1000
