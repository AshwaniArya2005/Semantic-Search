"""
src/api/models.py
─────────────────
Pydantic v2 request/response models for the FastAPI service.

Using Pydantic v2 for:
- Automatic JSON schema generation (for /docs Swagger UI)
- Runtime type validation with clear error messages
- model_config and field_serializer for Python dataclass ↔ JSON conversion
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /query request body."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language search query",
        examples=["What are the best graphics cards for gaming?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1–20)",
    )
    bypass_cache: bool = Field(
        default=False,
        description="If true, skip cache lookup and always run FAISS search",
    )


# ── Response models ────────────────────────────────────────────────────────────

class ClusterMembership(BaseModel):
    """Soft cluster assignment for a single document."""
    cluster_id: int
    probability: float = Field(ge=0.0, le=1.0)


class DocumentHit(BaseModel):
    """A single retrieved document in the search response."""

    doc_id: int = Field(description="Internal document index")
    rank: int = Field(description="Result rank (1 = most relevant)")
    score: float = Field(description="Cosine similarity to query [0, 1]")
    category: str = Field(description="20 Newsgroups category, e.g. 'comp.graphics'")
    subject: str = Field(description="Post subject line")
    text_snippet: str = Field(description="First 300 characters of cleaned post body")
    top_clusters: list[ClusterMembership] = Field(
        default_factory=list,
        description="Top-3 soft cluster memberships (GMM posteriors)",
    )

    @classmethod
    def from_document_result(cls, r: Any) -> "DocumentHit":
        """Convert a SearchEngine DocumentResult into this API model."""
        # Sort cluster memberships by probability and keep top-3
        top_clusters = sorted(
            [
                ClusterMembership(cluster_id=int(k), probability=round(float(v), 4))
                for k, v in r.cluster_memberships.items()
            ],
            key=lambda x: x.probability,
            reverse=True,
        )[:3]

        return cls(
            doc_id=r.doc_id,
            rank=r.rank,
            score=round(r.score, 6),
            category=r.category,
            subject=r.subject,
            text_snippet=r.text_snippet,
            top_clusters=top_clusters,
        )


class LatencyBreakdown(BaseModel):
    """Per-component latency for performance diagnostics."""
    total_ms: float
    faiss_ms: float
    cache_lookup_ms: float
    encoding_ms: float = 0.0  # reported separately if needed


class QueryResponse(BaseModel):
    """POST /query response body."""

    query: str
    results: list[DocumentHit]
    total_results: int
    cache_hit: bool = Field(description="Whether this response was served from cache")
    cache_similarity: float = Field(
        description="Cosine similarity of the best matching cached query (0 if cache miss)"
    )
    latency: LatencyBreakdown


class CacheClusterDistribution(BaseModel):
    """Per-cluster entry count in the cache."""
    cluster_id: int
    entry_count: int


class CacheStatsResponse(BaseModel):
    """GET /cache/stats response body."""

    total_entries: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float = Field(description="hits / (hits + misses)")
    avg_lookup_latency_ms: float
    similarity_threshold: float = Field(
        description=(
            "Current cosine similarity threshold for cache hits. "
            "Paraphrases typically score ≥ 0.85."
        )
    )
    top_k_clusters_searched: int
    cluster_distribution: list[CacheClusterDistribution]

    @classmethod
    def from_dict(cls, d: dict) -> "CacheStatsResponse":
        """Build from the dict returned by SemanticCache.stats()."""
        cluster_dist = [
            CacheClusterDistribution(cluster_id=int(k.replace("cluster_", "")), entry_count=v)
            for k, v in d.get("cluster_distribution", {}).items()
        ]
        return cls(
            total_entries=d["total_entries"],
            max_size=d["max_size"],
            hits=d["hits"],
            misses=d["misses"],
            hit_rate=d["hit_rate"],
            avg_lookup_latency_ms=d["avg_lookup_latency_ms"],
            similarity_threshold=d["similarity_threshold"],
            top_k_clusters_searched=d["top_k_clusters_searched"],
            cluster_distribution=cluster_dist,
        )


class CacheClearResponse(BaseModel):
    """DELETE /cache response body."""
    message: str
    cleared_entries: int


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str
    n_docs: int
    n_clusters: int
    cache_entries: int
    version: str
