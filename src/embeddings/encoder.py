"""
src/embeddings/encoder.py
─────────────────────────
Thin, production-grade wrapper around SentenceTransformers.

Design decisions:
- Model: `all-MiniLM-L6-v2`
    • 384-dim output — small enough for fast FAISS indexing, large enough for
      rich semantic representation.
    • Trained with multi-negative-ranking loss (cosine similarity objective).
      After L2-normalisation, inner product == cosine similarity — which lets
      us use FAISS IndexFlatIP (inner product) as a cosine search engine.
    • 14,200 docs/sec on CPU vs 5,200 for all-mpnet-base-v2. Critical for
      ingestion of ~20K newsgroup docs.
    • MTEB leaderboard: top of the "small" category for semantic similarity.

- L2 normalisation is performed in this layer (not delegated to FAISS).
  This keeps the encoder interface clean: callers always receive unit vectors.

- Lazy loading: the model is only downloaded/loaded on first encode() call,
  preventing import-time side effects.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings


class Encoder:
    """
    Wraps SentenceTransformer for batch text encoding.

    Outputs L2-normalised float32 vectors. With l2-norm applied:
        dot(a, b) ≡ cosine_similarity(a, b)   ∀ ||a||=||b||=1

    This allows reusing FAISS IndexFlatIP (fastest exact search) as
    a cosine similarity search engine.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    def _load(self) -> None:
        """Lazily load the model on first use."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.success(f"Model loaded. Embedding dim: {self.dim}")

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised float32 vectors.

        Args:
            texts:         Input strings (can be queries or documents).
            batch_size:    Encoding batch size. Defaults to settings value.
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype=float32, unit vectors.
        """
        self._load()
        bs = batch_size or settings.embedding_batch_size

        logger.info(f"Encoding {len(texts)} texts in batches of {bs}...")

        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ← L2 normalise here
        )

        # SentenceTransformer returns float32 by default; enforce it
        embeddings = embeddings.astype(np.float32)

        logger.success(f"Encoded {len(texts)} texts → shape {embeddings.shape}")
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text. Avoids overhead of batching for real-time queries.

        Returns:
            np.ndarray of shape (1, dim), dtype=float32, unit vector.
        """
        return self.encode([text], show_progress=False)

    def encode_batch_iter(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ):
        """
        Generator yielding (batch_embeddings, start_idx) for memory-efficient
        ingestion of very large corpora.
        """
        self._load()
        bs = batch_size or settings.embedding_batch_size

        for start in tqdm(range(0, len(texts), bs), desc="Encoding batches"):
            batch = texts[start : start + bs]
            vecs = self._model.encode(  # type: ignore[union-attr]
                batch,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            yield vecs, start


# Module-level singleton for convenient import
encoder = Encoder()
