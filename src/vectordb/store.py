"""
src/vectordb/store.py
─────────────────────
FAISS-backed vector store for fast approximate nearest-neighbour (ANN) retrieval.

Design decisions:
- IndexIVFFlat: Inverted File index with exact (Flat) distance computation
  within each Voronoi cell. Best trade-off for ~20K documents where we want
  high recall without heavy quantisation artifacts.

- Inner Product (IP) metric: Because all vectors are L2-normalised by the
  Encoder, inner_product(a, b) == cosine_similarity(a, b). This lets us rank
  by cosine similarity without an extra normalisation step at search time.

- nlist (Voronoi cells): We use 100 (slightly below sqrt(20000) = 141) to
  keep training fast while maintaining good cell granularity.

- nprobe: Set to 10 (10% of nlist). Increasing to 20 raises recall from ~95%
  to ~99% at roughly 2x query latency. Tunable via settings.

- Save / load via FAISS's native binary format (.index file) for portability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from config.settings import settings


@dataclass
class SearchResult:
    """Single retrieved document with its score and metadata."""
    doc_id: int
    score: float           # cosine similarity [−1, 1], higher = more relevant
    rank: int              # 1-indexed rank in results


class FAISSVectorStore:
    """
    FAISS IVFFlat vector store with cosine similarity search.

    Usage:
        store = FAISSVectorStore(dim=384)
        store.build(embeddings)                         # train + add all vectors
        results = store.search(query_vec, top_k=5)      # returns SearchResult list
        store.save(path)                                # persist to disk
        store = FAISSVectorStore.load(path, dim=384)    # reload
    """

    def __init__(
        self,
        dim: int = settings.embedding_dim,
        nlist: int = settings.faiss_nlist,
        nprobe: int = settings.faiss_nprobe,
    ) -> None:
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self._index: faiss.IndexIVFFlat | None = None
        self._n_docs: int = 0

    # ── Index construction ─────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray) -> None:
        """
        Train the IVF index and add all embeddings.

        Args:
            embeddings: float32 array of shape (N, dim). Assumed L2-normalised.
        """
        n, d = embeddings.shape
        assert d == self.dim, f"Expected dim={self.dim}, got {d}"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        logger.info(f"Building FAISS IVFFlat index: {n} docs, dim={d}, nlist={self.nlist}")

        # Quantiser: flat inner product search within each cell
        quantiser = faiss.IndexFlatIP(d)

        # IVFFlat: cluster space into nlist Voronoi cells using k-means,
        # then only search nprobe cells at query time
        self._index = faiss.IndexIVFFlat(quantiser, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self._index.nprobe = self.nprobe

        # Training: runs k-means on the embeddings to define cell centroids
        # Requires at least nlist * 39 training vectors (FAISS heuristic)
        min_train = self.nlist * 39
        if n < min_train:
            logger.warning(
                f"Only {n} docs available but FAISS recommends ≥{min_train} for nlist={self.nlist}. "
                "Falling back to IndexFlatIP (exact search)."
            )
            self._index = faiss.IndexFlatIP(d)  # type: ignore[assignment]
        else:
            logger.info("Training IVF index (k-means on embeddings)...")
            self._index.train(embeddings)

        # Add all vectors; FAISS auto-assigns integer IDs 0..N-1
        self._index.add(embeddings)
        self._n_docs = n
        logger.success(f"Index ready: {self._index.ntotal} vectors")

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, top_k: int = settings.default_top_k) -> list[SearchResult]:
        """
        Find the top-k most similar documents to a query vector.

        Args:
            query:  float32 array of shape (1, dim) or (dim,). Must be L2-normalised.
            top_k:  Number of results.

        Returns:
            List of SearchResult, sorted by descending cosine similarity.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        # Ensure 2D input
        q = np.atleast_2d(query).astype(np.float32)

        scores, indices = self._index.search(q, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:  # FAISS returns -1 for unfilled results
                continue
            results.append(SearchResult(doc_id=int(idx), score=float(score), rank=rank))

        return results

    def search_batch(
        self,
        queries: np.ndarray,
        top_k: int = settings.default_top_k,
    ) -> list[list[SearchResult]]:
        """Batch search for multiple queries simultaneously."""
        if self._index is None:
            raise RuntimeError("Index not built.")

        queries = np.atleast_2d(queries).astype(np.float32)
        scores_all, indices_all = self._index.search(queries, top_k)

        all_results = []
        for scores, indices in zip(scores_all, indices_all):
            results = []
            for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
                if idx != -1:
                    results.append(SearchResult(doc_id=int(idx), score=float(score), rank=rank))
            all_results.append(results)
        return all_results

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the FAISS index to disk in native binary format."""
        if self._index is None:
            raise RuntimeError("No index to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.success(f"FAISS index saved → {path}")

    @classmethod
    def load(cls, path: Path, dim: int = settings.embedding_dim) -> "FAISSVectorStore":
        """Load a persisted FAISS index from disk."""
        store = cls(dim=dim)
        store._index = faiss.read_index(str(path))
        # Restore nprobe if it's an IVF index
        if hasattr(store._index, "nprobe"):
            store._index.nprobe = settings.faiss_nprobe
        store._n_docs = store._index.ntotal
        logger.success(f"FAISS index loaded: {store._n_docs} vectors from {path}")
        return store

    @property
    def n_docs(self) -> int:
        return self._n_docs
