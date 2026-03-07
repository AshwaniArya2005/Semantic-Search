"""
config/settings.py
──────────────────
Centralised configuration powered by Pydantic BaseSettings.
All values are read from environment variables (or .env file).
This makes every hyperparameter tunable without touching source code.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all hyperparameters and paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    dataset_dir: Path = Field(default=Path("dataset"), description="Raw dataset directory")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Generated artifacts")

    @property
    def faiss_index_path(self) -> Path:
        return self.artifacts_dir / "faiss_index"

    @property
    def embeddings_path(self) -> Path:
        return self.artifacts_dir / "embeddings.npy"

    @property
    def doc_metadata_path(self) -> Path:
        return self.artifacts_dir / "doc_metadata.pkl"

    @property
    def gmm_model_path(self) -> Path:
        return self.artifacts_dir / "gmm_model.pkl"

    @property
    def pca_model_path(self) -> Path:
        return self.artifacts_dir / "pca_model.pkl"

    @property
    def cluster_probs_path(self) -> Path:
        return self.artifacts_dir / "cluster_probs.npy"

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "SentenceTransformer model ID. "
            "all-MiniLM-L6-v2: 384-dim, 22M params, optimal speed/accuracy trade-off. "
            "Trained with cosine similarity objective → perfect for FAISS inner-product search "
            "after L2 normalisation."
        ),
    )
    embedding_batch_size: int = Field(default=64, description="Batch size for encoding")
    embedding_dim: int = Field(default=384, description="Output dimension of the model")

    # ── FAISS Index ────────────────────────────────────────────────────────
    faiss_nlist: int = Field(
        default=100,
        description=(
            "Number of Voronoi cells (IVF). "
            "Rule of thumb: sqrt(N). For ~20K docs → 141; we use 100 for safety margin."
        ),
    )
    faiss_nprobe: int = Field(
        default=10,
        description="Cells to scan per query. Higher = more recall, slower. Sweet spot: 5–15% of nlist.",
    )

    # ── GMM Clustering ─────────────────────────────────────────────────────
    gmm_n_components: int = Field(
        default=20,
        description="Number of Gaussian mixture components. Matches 20 newsgroup topics.",
    )
    gmm_covariance_type: str = Field(
        default="diag",
        description=(
            "'diag' = diagonal covariance matrix. "
            "Tractable in 50-D PCA space (only D params/component vs D²/2 for 'full'). "
            "Captures per-dimension variance while preserving independence assumption."
        ),
    )
    gmm_max_iter: int = Field(default=200, description="EM algorithm max iterations")
    gmm_random_state: int = Field(default=42)

    # ── PCA (pre-processing for GMM) ───────────────────────────────────────
    pca_n_components: int = Field(
        default=50,
        description=(
            "Reduce 384-dim embeddings to 50-D before GMM. "
            "Avoids curse of dimensionality in covariance estimation. "
            "Retains >90% variance empirically."
        ),
    )

    # ── Semantic Cache ─────────────────────────────────────────────────────
    cache_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold for cache hit. "
            "Paraphrases typically score 0.85+. "
            "Different-topic queries score 0.5–0.75. "
            "Recommended range: 0.82–0.88."
        ),
    )
    cache_max_size: int = Field(
        default=10_000,
        description="Maximum cache entries. Oldest entries evicted when exceeded (LRU-like).",
    )
    cache_top_k_clusters: int = Field(
        default=3,
        description="Number of top-probability clusters to search during cache lookup.",
    )

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_title: str = Field(default="Semantic Search — 20 Newsgroups")
    api_version: str = Field(default="1.0.0")

    # ── Search ──────────────────────────────────────────────────────────────
    default_top_k: int = Field(default=5, description="Default number of results returned")


# Global singleton — import this wherever config is needed
settings = Settings()

# Ensure artifact directory exists at import time
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
