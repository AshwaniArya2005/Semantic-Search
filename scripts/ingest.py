"""
scripts/ingest.py
─────────────────
End-to-end data ingestion pipeline.

Pipeline stages:
  1. Load + clean documents from tar.gz archive
  2. Encode all documents with SentenceTransformer
  3. Build + train FAISS IVFFlat index
  4. Train GMM on PCA-reduced embeddings
  5. Compute per-document soft cluster memberships
  6. Save all artifacts to artifacts/ directory

Run:
    python scripts/ingest.py [--max-docs N] [--corpus mini|full]

Expected runtime:
  - mini corpus (~2K docs):  ~2–3 minutes
  - full corpus (~20K docs): ~10–15 minutes (CPU)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to path (enables running from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.clustering.fuzzy_gmm import FuzzyGMM
from src.data.loader import Document, load_newsgroups
from src.embeddings.encoder import Encoder
from src.vectordb.store import FAISSVectorStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS + GMM artifacts for semantic search")
    p.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit documents (None = all). Use 500 for quick smoke-test.",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=settings.dataset_dir,
        help="Path to directory containing tar.gz files",
    )
    return p.parse_args()


def run_ingestion(max_docs: int | None, dataset_dir: Path) -> None:
    logger.info("=" * 60)
    logger.info("  Semantic Search Ingestion Pipeline")
    logger.info("=" * 60)

    # ── Stage 1: Load documents ────────────────────────────────────────────
    logger.info("Stage 1/5 — Loading documents from dataset archive...")
    docs: list[Document] = list(
        load_newsgroups(dataset_dir, max_docs=max_docs)
    )
    logger.success(f"Loaded {len(docs)} documents")

    if len(docs) < 50:
        logger.error("Too few documents. Check your dataset path.")
        sys.exit(1)

    # ── Stage 2: Embed documents ───────────────────────────────────────────
    logger.info("Stage 2/5 — Encoding documents with SentenceTransformer...")
    encoder = Encoder()
    texts = [d.text for d in docs]
    embeddings = encoder.encode(texts, show_progress=True)  # float32, L2-normalised
    # Shape: (N, 384)

    # Save embeddings for potential re-use (e.g., re-clustering without re-encoding)
    emb_path = settings.embeddings_path
    np.save(emb_path, embeddings)
    logger.success(f"Embeddings saved → {emb_path}  shape={embeddings.shape}")

    # ── Stage 3: Build FAISS index ─────────────────────────────────────────
    logger.info("Stage 3/5 — Building FAISS IVFFlat index...")
    store = FAISSVectorStore()
    store.build(embeddings)
    store.save(settings.faiss_index_path)
    logger.success(f"FAISS index saved → {settings.faiss_index_path}")

    # ── Stage 4: Train GMM ─────────────────────────────────────────────────
    logger.info("Stage 4/5 — Training Gaussian Mixture Model...")
    gmm = FuzzyGMM()
    gmm.fit(embeddings)
    gmm.save(settings.gmm_model_path, settings.pca_model_path)

    eval_metrics = gmm.score(embeddings)
    logger.success(
        f"GMM metrics — "
        f"log-likelihood: {eval_metrics['log_likelihood']:.4f}, "
        f"BIC: {eval_metrics['bic']:.1f}, "
        f"AIC: {eval_metrics['aic']:.1f}"
    )

    # ── Stage 5: Compute cluster probs + save metadata ─────────────────────
    logger.info("Stage 5/5 — Computing soft cluster memberships + saving metadata...")
    cluster_probs = gmm.predict_proba(embeddings)  # shape (N, K)
    np.save(settings.cluster_probs_path, cluster_probs)

    # Build document metadata list (indexed by doc_id for O(1) lookup in engine)
    doc_metadata = []
    for i, doc in enumerate(tqdm(docs, desc="Building metadata")):
        # Convert cluster probs to a dict {cluster_id: prob} keeping only top-5
        # for storage efficiency
        prob_vec = cluster_probs[i]
        top5 = np.argsort(prob_vec)[::-1][:5]
        cluster_dict = {str(int(k)): round(float(prob_vec[k]), 6) for k in top5}

        doc_metadata.append({
            "doc_id": doc.doc_id,
            "category": doc.category,
            "subject": doc.subject,
            "text": doc.text,
            "cluster_probs": cluster_dict,
        })

    with open(settings.doc_metadata_path, "wb") as f:
        pickle.dump(doc_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.success(f"Metadata saved → {settings.doc_metadata_path}  ({len(doc_metadata)} entries)")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.success("✅  Ingestion complete!")
    logger.info(f"   Documents:   {len(docs)}")
    logger.info(f"   Embedding dim: {embeddings.shape[1]}")
    logger.info(f"   GMM clusters:  {gmm.n_components}")
    logger.info(f"   Artifacts in:  {settings.artifacts_dir}/")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  python scripts/cluster.py       # generate cluster plots")
    logger.info("  uvicorn src.api.app:app --reload # start API server")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(max_docs=args.max_docs, dataset_dir=args.dataset_dir)
