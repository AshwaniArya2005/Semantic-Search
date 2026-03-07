"""
scripts/cluster.py
──────────────────
Load saved artifacts and generate all cluster visualisation plots.

Produces 4 plots in artifacts/:
  - cluster_plot.png          → UMAP coloured by GMM hard assignment
  - category_plot.png         → UMAP coloured by true newsgroup category
  - cluster_probabilities.png → Heatmap of mean P(cluster | category)
  - entropy_plot.png          → UMAP coloured by cluster membership entropy

Run:
    python scripts/cluster.py [--n-samples 5000]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.clustering.fuzzy_gmm import FuzzyGMM
from src.clustering.visualizer import ClusterVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate cluster visualisation plots")
    p.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Max docs to include in UMAP (larger = slower but more representative)",
    )
    return p.parse_args()


def run_visualisation(n_samples: int) -> None:
    logger.info("=" * 60)
    logger.info("  Cluster Visualisation")
    logger.info("=" * 60)

    # ── Load artifacts ──────────────────────────────────────────────────────
    if not settings.embeddings_path.exists():
        logger.error(
            f"Embeddings not found at {settings.embeddings_path}. "
            "Run `python scripts/ingest.py` first."
        )
        sys.exit(1)

    logger.info("Loading embeddings...")
    embeddings = np.load(settings.embeddings_path)

    logger.info("Loading GMM model...")
    gmm = FuzzyGMM.load(settings.gmm_model_path, settings.pca_model_path)

    logger.info("Loading document metadata...")
    with open(settings.doc_metadata_path, "rb") as f:
        doc_metadata = pickle.load(f)

    # ── Load cluster probs ──────────────────────────────────────────────────
    if settings.cluster_probs_path.exists():
        logger.info("Loading pre-computed cluster probabilities...")
        cluster_probs = np.load(settings.cluster_probs_path)
    else:
        logger.info("Computing cluster probabilities (not cached)...")
        cluster_probs = gmm.predict_proba(embeddings)

    # ── Extract categories ──────────────────────────────────────────────────
    categories = [m["category"] for m in doc_metadata]
    category_names = sorted(set(categories))
    logger.info(f"Found {len(category_names)} newsgroup categories: {category_names[:5]}...")

    # ── Generate plots ──────────────────────────────────────────────────────
    viz = ClusterVisualizer()
    viz.visualise_all(
        embeddings=embeddings,
        cluster_probs=cluster_probs,
        categories=categories,
        category_names=category_names,
        n_umap_samples=n_samples,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.success("✅ Visualisation complete!")
    logger.info(f"   Plots saved to: {settings.artifacts_dir}/")
    logger.info("   - cluster_plot.png          (GMM assignments)")
    logger.info("   - category_plot.png         (true categories)")
    logger.info("   - cluster_probabilities.png (membership heatmap)")
    logger.info("   - entropy_plot.png          (fuzziness diagnostic)")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_visualisation(n_samples=args.n_samples)
