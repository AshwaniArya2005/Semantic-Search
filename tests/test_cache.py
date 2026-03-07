"""
tests/test_cache.py
────────────────────
Unit tests for the SemanticCache module.

Tests verify:
1. Cache miss on first query (cold start)
2. Cache hit on identical query (exact)
3. Cache hit on paraphrased query (semantic similarity)
4. Cache miss on unrelated query (different topic)
5. Cache eviction when max_size is exceeded
6. Stats reporting
7. Cache clear

Uses a mock GMM that assigns fixed cluster IDs to avoid dependency on real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache


# ── Mock GMM ───────────────────────────────────────────────────────────────────

class MockGMM:
    """Deterministic GMM mock for unit testing."""
    n_components = 3

    def top_k_clusters(self, embedding, k=3):
        # Always returns cluster 0 with 100% probability
        return [(0, 1.0), (1, 0.0), (2, 0.0)][:k]

    def predict_proba(self, embeddings):
        n = np.atleast_2d(embeddings).shape[0]
        probs = np.zeros((n, 3), dtype=np.float32)
        probs[:, 0] = 1.0
        return probs


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cache():
    return SemanticCache(
        gmm=MockGMM(),
        dim=8,
        similarity_threshold=0.85,
        max_size=5,
        top_k_clusters=1,
    )


def _vec(seed: float, dim: int = 8) -> np.ndarray:
    """Create a deterministic L2-normalised vector from a seed."""
    np.random.seed(int(seed * 1000))
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_cold_cache_is_miss(cache):
    """Empty cache must always return a miss."""
    q = _vec(1.0)
    result = cache.lookup(q)
    assert result.hit is False
    assert result.entry is None


def test_exact_query_hit(cache):
    """Identical query embedding should always produce a cache hit."""
    q = _vec(1.0)
    cache.insert("How do graphics cards work?", q, {"results": ["doc1"]})

    result = cache.lookup(q)
    assert result.hit is True
    assert result.similarity >= 0.99  # exact match → near-1.0 cosine


def test_paraphrase_hit(cache):
    """Very similar vectors (paraphrases) should hit above threshold."""
    q1 = _vec(1.0)
    cache.insert("How do graphics cards work?", q1, {"results": ["doc1"]})

    # Slightly perturb q1 to simulate a paraphrase (high similarity ~0.98)
    q2 = q1 + np.random.default_rng(42).normal(0, 0.05, q1.shape).astype(np.float32)
    q2 = (q2 / np.linalg.norm(q2)).astype(np.float32)

    similarity = float(np.dot(q1, q2))
    if similarity >= 0.85:
        result = cache.lookup(q2)
        assert result.hit is True
    else:
        pytest.skip("Random perturbation produced low similarity — skip")


def test_unrelated_query_miss(cache):
    """Orthogonal vectors (completely different topics) should be cache misses."""
    q1 = np.zeros(8, dtype=np.float32)
    q1[0] = 1.0  # unit vector along dimension 0

    q2 = np.zeros(8, dtype=np.float32)
    q2[1] = 1.0  # unit vector along dimension 1 → cosine = 0

    cache.insert("Linux installation guide", q1, {"results": ["doc_linux"]})
    result = cache.lookup(q2)
    assert result.hit is False


def test_max_size_eviction(cache):
    """Oldest entry should be evicted when cache exceeds max_size (5)."""
    # Insert 5 distinct entries
    vecs = [_vec(float(i)) for i in range(6)]  # 6 entries
    for i, v in enumerate(vecs):
        cache.insert(f"query {i}", v, {"results": [f"doc_{i}"]})

    # After 6 inserts into a max_size=5 cache, first entry is gone
    assert cache.stats()["total_entries"] == 5


def test_cache_stats_hit_rate(cache):
    """Hit rate should update correctly after hits and misses."""
    q = _vec(1.0)
    cache.insert("test query", q, {"result": "docs"})

    cache.lookup(q)     # hit
    cache.lookup(_vec(99.0))  # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_cache_clear(cache):
    """After clear(), cache should be empty with zeroed stats."""
    q = _vec(1.0)
    cache.insert("test", q, {})
    cache.lookup(q)

    result = cache.clear()
    assert result["cleared"] == 1

    stats = cache.stats()
    assert stats["total_entries"] == 0
    assert stats["hits"] == 0


def test_cluster_distribution_in_stats(cache):
    """Stats should show per-cluster entry distribution."""
    for i in range(3):
        cache.insert(f"q{i}", _vec(float(i)), {})

    stats = cache.stats()
    # MockGMM assigns everything to cluster_0
    assert "cluster_0" in stats["cluster_distribution"]
    assert stats["cluster_distribution"]["cluster_0"] == 3
