"""
src/clustering/visualizer.py
────────────────────────────
Cluster evaluation and visualisation using UMAP for 2D projection.

Why UMAP over t-SNE:
- UMAP preserves both local AND global structure (t-SNE only local).
- 10–100× faster for large N. Critical for ~20K doc corpus.
- Produces layouts where inter-cluster distances are meaningful.
- Deterministic with fixed random_state.

Visualisation produces:
1. Scatter plot coloured by GMM hard assignment (dominant cluster)
2. Scatter plot coloured by true newsgroup category (ground truth check)
3. Heatmap of mean soft membership probabilities per newsgroup category
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from loguru import logger

from config.settings import settings

# Optional import — UMAP can be slow to import; guard it
try:
    from umap import UMAP
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed. Cluster visualisation unavailable.")


class ClusterVisualizer:
    """
    Generate publication-quality cluster visualisations for the GMM output.

    All plots saved to `artifacts/` directory.
    """

    def __init__(self, artifacts_dir: Path = settings.artifacts_dir) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Consistent colour palette for up to 20 clusters
        self._palette = plt.cm.tab20.colors  # type: ignore

    # ── Main entry point ───────────────────────────────────────────────────

    def visualise_all(
        self,
        embeddings: np.ndarray,
        cluster_probs: np.ndarray,
        categories: list[str],
        category_names: list[str],
        *,
        n_umap_samples: int = 5_000,
    ) -> None:
        """
        Generate all cluster visualisation plots.

        Args:
            embeddings:      float32 (N, D) — original embeddings.
            cluster_probs:   float32 (N, K) — GMM posterior probabilities.
            categories:      list of N category strings (ground truth labels).
            category_names:  sorted unique category names (for legend).
            n_umap_samples:  max docs to include in UMAP (speed vs coverage).
        """
        if not _UMAP_AVAILABLE:
            logger.warning("Skipping visualisation: umap-learn not available.")
            return

        N = len(embeddings)
        if N > n_umap_samples:
            logger.info(f"Sampling {n_umap_samples}/{N} docs for UMAP (speed)")
            idx = np.random.choice(N, n_umap_samples, replace=False)
            emb_sub = embeddings[idx]
            probs_sub = cluster_probs[idx]
            cats_sub = [categories[i] for i in idx]
        else:
            emb_sub, probs_sub, cats_sub = embeddings, cluster_probs, categories

        logger.info("Running UMAP projection (2D)...")
        reducer = UMAP(
            n_components=2,
            n_neighbors=30,         # larger = more global structure preserved
            min_dist=0.1,           # tighter clusters in 2D
            metric="cosine",        # consistent with our embedding distance
            random_state=42,
            verbose=True,
        )
        emb_2d = reducer.fit_transform(emb_sub)
        logger.success("UMAP projection complete")

        # Hard cluster assignment from GMM
        hard_labels = probs_sub.argmax(axis=1)

        self._plot_by_cluster(emb_2d, hard_labels, cluster_probs.shape[1])
        self._plot_by_category(emb_2d, cats_sub, category_names)
        self._plot_membership_heatmap(cluster_probs, categories, category_names)
        self._plot_entropy_scatter(emb_2d, probs_sub)

        logger.success(f"All plots saved to {self.artifacts_dir}")

    # ── Individual plots ───────────────────────────────────────────────────

    def _plot_by_cluster(
        self, emb_2d: np.ndarray, hard_labels: np.ndarray, n_clusters: int
    ) -> None:
        """Scatter coloured by GMM hard cluster assignment."""
        fig, ax = plt.subplots(figsize=(14, 10))
        for k in range(n_clusters):
            mask = hard_labels == k
            if mask.sum() == 0:
                continue
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[self._palette[k % len(self._palette)]],
                label=f"Cluster {k}",
                alpha=0.6,
                s=8,
                linewidths=0,
            )
        ax.set_title("UMAP Projection — GMM Cluster Assignments", fontsize=16, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(
            loc="upper right",
            ncol=2,
            markerscale=3,
            fontsize=7,
            framealpha=0.7,
        )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        path = self.artifacts_dir / "cluster_plot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Cluster scatter saved → {path}")

    def _plot_by_category(
        self,
        emb_2d: np.ndarray,
        categories: list[str],
        category_names: list[str],
    ) -> None:
        """Scatter coloured by true newsgroup category (ground truth)."""
        cat_to_idx = {name: i for i, name in enumerate(category_names)}
        cat_ids = np.array([cat_to_idx.get(c, -1) for c in categories])

        fig, ax = plt.subplots(figsize=(16, 10))
        for i, cat in enumerate(category_names):
            mask = cat_ids == i
            if mask.sum() == 0:
                continue
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                c=[self._palette[i % len(self._palette)]],
                label=cat.replace("comp.", "").replace("rec.", "").replace("sci.", ""),
                alpha=0.65,
                s=8,
                linewidths=0,
            )
        ax.set_title("UMAP Projection — True 20 Newsgroups Categories", fontsize=16, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(
            loc="upper right",
            ncol=2,
            markerscale=3,
            fontsize=7,
            framealpha=0.7,
        )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        path = self.artifacts_dir / "category_plot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Category scatter saved → {path}")

    def _plot_membership_heatmap(
        self,
        cluster_probs: np.ndarray,
        categories: list[str],
        category_names: list[str],
    ) -> None:
        """
        Heatmap: rows=newsgroup categories, cols=GMM clusters.
        Each cell = mean P(cluster | docs in this category).

        A well-trained GMM should show diagonal-dominant blocks, confirming
        that topically-similar newsgroups share dominant clusters.
        """
        K = cluster_probs.shape[1]
        C = len(category_names)
        cat_to_idx = {name: i for i, name in enumerate(category_names)}

        mean_probs = np.zeros((C, K), dtype=np.float32)
        for i, cat in enumerate(category_names):
            mask = np.array([c == cat for c in categories])
            if mask.sum() > 0:
                mean_probs[i] = cluster_probs[mask].mean(axis=0)

        fig, ax = plt.subplots(figsize=(18, 10))
        short_names = [n.split(".")[-1] for n in category_names]
        sns.heatmap(
            mean_probs,
            ax=ax,
            xticklabels=[f"C{k}" for k in range(K)],
            yticklabels=short_names,
            cmap="YlOrRd",
            cbar_kws={"label": "Mean P(cluster | category)"},
            linewidths=0.3,
        )
        ax.set_title(
            "Mean GMM Cluster Membership by Newsgroup Category\n"
            "(Diagonal dominance = well-separated clusters)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("GMM Cluster ID")
        ax.set_ylabel("Newsgroup Category")
        fig.tight_layout()
        path = self.artifacts_dir / "cluster_probabilities.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Membership heatmap saved → {path}")

    def _plot_entropy_scatter(
        self, emb_2d: np.ndarray, cluster_probs: np.ndarray
    ) -> None:
        """
        Scatter coloured by entropy of cluster membership distribution.

        High entropy = ambiguous document (truly cross-posted / multi-topic).
        Low entropy = document firmly in one cluster.

        This is the 'fuzzy clustering diagnostic' — shows where soft memberships
        are actually doing useful work vs. collapsing to hard assignments.
        """
        eps = 1e-10
        # Shannon entropy H(p) = -Σ p_k log(p_k)
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + eps), axis=1)  # shape (N,)

        fig, ax = plt.subplots(figsize=(14, 10))
        sc = ax.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=entropy,
            cmap="plasma",   # perceptually uniform: low=dark, high=bright
            alpha=0.7,
            s=8,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, label="Cluster Membership Entropy")
        ax.set_title(
            "UMAP — Cluster Membership Entropy\n"
            "(Bright = ambiguous multi-topic doc, Dark = strongly mono-topic)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        path = self.artifacts_dir / "entropy_plot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Entropy scatter saved → {path}")
