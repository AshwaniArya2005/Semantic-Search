"""
tests/test_search.py
────────────────────
Integration tests for the SearchEngine using a tiny mock corpus.

These tests verify the full search → cache → return pipeline without
needing real FAISS index or GMM models loaded from disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.search.engine import DocumentResult, SearchEngine, SearchResponse


# ── Mocks ──────────────────────────────────────────────────────────────────────

class MockEncoder:
    dim = 8

    def encode_single(self, text: str) -> np.ndarray:
        # Hash-based deterministic embedding for testing
        seed = sum(ord(c) for c in text) % 256
        np.random.seed(seed)
        v = np.random.randn(1, self.dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        return v


class MockFAISSStore:
    n_docs = 3

    def search(self, query, top_k=5):
        from src.vectordb.store import SearchResult
        return [
            SearchResult(doc_id=0, score=0.92, rank=1),
            SearchResult(doc_id=1, score=0.85, rank=2),
            SearchResult(doc_id=2, score=0.76, rank=3),
        ][:top_k]


class MockGMM:
    n_components = 3
    _fitted = True

    def top_k_clusters(self, embedding, k=3):
        return [(0, 0.7), (1, 0.2), (2, 0.1)][:k]

    def predict_proba(self, embeddings):
        n = np.atleast_2d(embeddings).shape[0]
        p = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        return np.repeat(p, n, axis=0)


MOCK_METADATA = [
    {"doc_id": 0, "category": "comp.graphics", "subject": "OpenGL tips", "text": "OpenGL is a cross-platform API for rendering 2D/3D graphics. " * 5, "cluster_probs": {"0": 0.7, "1": 0.2}},
    {"doc_id": 1, "category": "sci.space", "subject": "Mars mission", "text": "NASA plans a crewed Mars mission by 2040 using SLS rockets. " * 5, "cluster_probs": {"0": 0.1, "2": 0.9}},
    {"doc_id": 2, "category": "rec.sport.baseball", "subject": "World Series", "text": "The Yankees won the World Series last night in a dramatic finale. " * 5, "cluster_probs": {"1": 0.8, "2": 0.2}},
]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    from src.cache.semantic_cache import SemanticCache

    e = SearchEngine(
        encoder=MockEncoder(),
        vector_store=MockFAISSStore(),
        gmm=MockGMM(),
        doc_metadata=MOCK_METADATA,
    )
    e._cache = SemanticCache(gmm=MockGMM(), dim=8, similarity_threshold=0.85)
    e._ready = True
    return e


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_basic_search_returns_results(engine):
    """A search should return 3 results (matching our MockFAISSStore)."""
    response = engine.search("How does OpenGL work?", top_k=3)
    assert isinstance(response, SearchResponse)
    assert len(response.results) == 3
    assert response.results[0].rank == 1
    assert response.results[0].score > 0.9


def test_first_query_is_cache_miss(engine):
    """First query always misses an empty cache."""
    response = engine.search("Linux kernel GPU drivers")
    assert response.cache_hit is False


def test_second_identical_query_is_cache_hit(engine):
    """Identical second query should hit the cache."""
    engine.search("GPU benchmarks for gaming")        # cold
    response = engine.search("GPU benchmarks for gaming")  # warm
    assert response.cache_hit is True
    assert response.cache_similarity >= 0.99


def test_cache_stats_returned(engine):
    """cache_stats() should return valid dict with expected keys."""
    engine.search("some query")
    stats = engine.cache_stats()
    assert "total_entries" in stats
    assert "hit_rate" in stats
    assert "cluster_distribution" in stats


def test_clear_cache(engine):
    """After clear_cache(), total_entries should be 0."""
    engine.search("some query")
    engine.clear_cache()
    stats = engine.cache_stats()
    assert stats["total_entries"] == 0


def test_results_contain_cluster_memberships(engine):
    """Each result should include soft cluster membership info."""
    response = engine.search("baseball statistics")
    for result in response.results:
        assert isinstance(result.cluster_memberships, dict)
        assert len(result.cluster_memberships) > 0


def test_latency_fields_in_response(engine):
    """Response should include non-negative latency breakdown."""
    response = engine.search("space exploration news")
    assert response.total_latency_ms >= 0
    assert response.faiss_latency_ms >= 0
    assert response.cache_lookup_latency_ms >= 0
