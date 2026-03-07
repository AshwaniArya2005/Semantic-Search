"""
src/clustering/fuzzy_gmm.py
───────────────────────────
Gaussian Mixture Model (GMM) for soft/fuzzy document clustering.

Why GMM over alternatives:
─────────────────────────────────────────────────────────────────────────────
│ Algorithm        │ Soft memberships │ Probabilistic │ Handles elongated  │
│                  │                  │               │ clusters           │
├──────────────────┼──────────────────┼───────────────┼────────────────────┤
│ Hard K-Means     │ ✗                │ ✗             │ ✗                  │
│ Soft K-Means     │ ✓ (via distance) │ ✗             │ ✗                  │
│ Fuzzy C-Means    │ ✓ (via exponent) │ ✗             │ ✗                  │
│ GMM (this impl)  │ ✓                │ ✓ (Bayes)     │ ✓ (diag cov)      │
─────────────────────────────────────────────────────────────────────────────

GMM models each cluster as a multivariate Gaussian and uses the EM algorithm
to fit parameters. The posterior P(cluster | document) — computed via Bayes'
theorem — gives true probability distributions, not proximity-weighted scores.

This is critical for cross-posted newsgroup articles (e.g., a post about
"Linux graphics drivers" should have high P(comp.os.ms-windows.misc) AND
P(comp.graphics)) — something hard assignments cannot represent.

Covariance type selection:
- 'full':  D²/2 parameters per component → intractable in 384-D space
- 'diag':  D parameters per component → tractable, captures per-dimension
           variance. Applied after PCA which decorrelates features, making
           the diagonal assumption nearly lossless.
- 'tied':  All components share one covariance → too restrictive for newsgroups
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config.settings import settings


class FuzzyGMM:
    """
    Dimensionality-reduced GMM for soft document clustering.

    Pipeline:
        raw embeddings (384-D)
            → StandardScaler (zero mean, unit variance)
            → PCA (50-D)
            → GaussianMixture (20 components, diag covariance)
            → P(cluster | doc) [N × K posterior matrix]

    The StandardScaler → PCA pipeline is critical:
    - SentenceTransformer outputs are NOT zero-centred post-normalisation.
    - PCA requires centred data for correct eigenvector computation.
    - After PCA, the diagonal covariance assumption is cleanest (PCA outputs
      are uncorrelated by construction).
    """

    def __init__(
        self,
        n_components: int = settings.gmm_n_components,
        covariance_type: str = settings.gmm_covariance_type,
        pca_dims: int = settings.pca_n_components,
        random_state: int = settings.gmm_random_state,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.pca_dims = pca_dims
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._pca = PCA(n_components=pca_dims, random_state=random_state)
        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=settings.gmm_max_iter,
            random_state=random_state,
            verbose=1,
            verbose_interval=20,
        )
        self._fitted = False

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> "FuzzyGMM":
        """
        Fit the full pipeline on document embeddings.

        Args:
            embeddings: float32 array of shape (N, D). L2-normalised.

        Returns:
            self (for chaining)
        """
        logger.info(f"Fitting FuzzyGMM: {embeddings.shape[0]} docs, {self.n_components} clusters")

        # Step 1: Standardise (mean=0, std=1 per dimension)
        scaled = self._scaler.fit_transform(embeddings)

        # Step 2: PCA dimensionality reduction
        logger.info(f"Running PCA: {embeddings.shape[1]}-D → {self.pca_dims}-D")
        reduced = self._pca.fit_transform(scaled)
        explained = self._pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explains {explained:.1%} of total variance")

        # Step 3: Fit Gaussian Mixture Model via Expectation-Maximisation
        logger.info(f"Fitting GMM ({self.covariance_type} covariance, EM max_iter={settings.gmm_max_iter})")
        self._gmm.fit(reduced)

        self._fitted = True
        bic = self._gmm.bic(reduced)
        aic = self._gmm.aic(reduced)
        logger.success(
            f"GMM converged={self._gmm.converged_}, "
            f"n_iter={self._gmm.n_iter_}, BIC={bic:.1f}, AIC={aic:.1f}"
        )
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute soft cluster membership probabilities.

        Returns:
            np.ndarray of shape (N, K) where each row sums to 1.
            Entry [i, k] = P(cluster k | document i)
        """
        self._check_fitted()
        reduced = self._transform(embeddings)
        probs = self._gmm.predict_proba(reduced)
        return probs.astype(np.float32)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Hard assignment: argmax over cluster probabilities."""
        return self.predict_proba(embeddings).argmax(axis=1)

    def top_k_clusters(
        self,
        embedding: np.ndarray,
        k: int = settings.cache_top_k_clusters,
    ) -> list[tuple[int, float]]:
        """
        Return the top-k most probable cluster IDs for a single embedding.

        Args:
            embedding: shape (1, D) or (D,)
            k:         number of clusters to return

        Returns:
            List of (cluster_id, probability) sorted by descending probability.
        """
        self._check_fitted()
        vec = np.atleast_2d(embedding).astype(np.float32)
        probs = self.predict_proba(vec)[0]  # shape (K,)
        top_ids = np.argsort(probs)[::-1][:k]
        return [(int(cid), float(probs[cid])) for cid in top_ids]

    # ── Evaluation ─────────────────────────────────────────────────────────

    def score(self, embeddings: np.ndarray) -> dict[str, float]:
        """Return GMM evaluation metrics on a held-out or training set."""
        self._check_fitted()
        reduced = self._transform(embeddings)
        return {
            "log_likelihood": float(self._gmm.score(reduced)),
            "bic": float(self._gmm.bic(reduced)),
            "aic": float(self._gmm.aic(reduced)),
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, gmm_path: Path, pca_path: Path) -> None:
        """Persist GMM + PCA + Scaler to disk."""
        self._check_fitted()
        gmm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gmm_path, "wb") as f:
            pickle.dump({"gmm": self._gmm, "scaler": self._scaler}, f)
        with open(pca_path, "wb") as f:
            pickle.dump(self._pca, f)
        logger.success(f"GMM saved → {gmm_path}, PCA → {pca_path}")

    @classmethod
    def load(cls, gmm_path: Path, pca_path: Path) -> "FuzzyGMM":
        """Load a saved FuzzyGMM from disk."""
        with open(gmm_path, "rb") as f:
            data = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        obj = cls.__new__(cls)
        obj._gmm = data["gmm"]
        obj._scaler = data["scaler"]
        obj._pca = pca
        obj.n_components = obj._gmm.n_components
        obj.covariance_type = obj._gmm.covariance_type
        obj.pca_dims = pca.n_components_
        obj.random_state = settings.gmm_random_state
        obj._fitted = True
        logger.success(f"FuzzyGMM loaded: {obj.n_components} clusters, {obj.pca_dims}-D PCA")
        return obj

    # ── Internals ──────────────────────────────────────────────────────────

    def _transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply Scaler → PCA transformation pipeline."""
        scaled = self._scaler.transform(embeddings.astype(np.float32))
        return self._pca.transform(scaled)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FuzzyGMM not fitted. Call fit() first.")

    @property
    def cluster_labels(self) -> list[str]:
        """Human-readable cluster label placeholders (refined by visualiser)."""
        return [f"cluster_{i}" for i in range(self.n_components)]
