"""
src/data/loader.py
──────────────────
Responsible for extracting the 20 Newsgroups dataset from the local tar.gz archive
and yielding clean document records.

Design decisions:
- We parse from the raw tar.gz to avoid sklearn's internet dependency.
- Header/footer stripping removes boilerplate (email headers, signatures) that
  would bias embeddings toward metadata rather than semantic content.
- We yield a dataclass (not raw dicts) for IDE-friendly downstream consumption.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from loguru import logger


# ── Data contract ──────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Document:
    """Immutable document record parsed from the 20 Newsgroups corpus."""
    doc_id: int
    category: str        # e.g. "comp.graphics"
    subject: str         # parsed from header
    text: str            # cleaned body text
    raw_filename: str    # original filename inside the tar


# ── Preprocessing helpers ──────────────────────────────────────────────────────

# Header fields we want to extract before stripping
_SUBJECT_RE = re.compile(r"^Subject:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# Lines that are clearly quoted text from previous posts ("> ...") or signatures ("-- ")
_QUOTE_RE = re.compile(r"^>.*$", re.MULTILINE)
_SIG_MARKER = "\n-- \n"

# Strip email-like header block (everything before the first blank line)
def _strip_headers(text: str) -> str:
    """Remove email-style headers (everything above the first blank line)."""
    # Find the first blank line separating headers from body
    idx = text.find("\n\n")
    if idx == -1:
        return text
    return text[idx:].strip()


def _strip_signature(text: str) -> str:
    """Remove email signatures (convention: '-- ' on its own line)."""
    sig_idx = text.find(_SIG_MARKER)
    if sig_idx != -1:
        text = text[:sig_idx]
    return text


def _clean_text(raw: str) -> tuple[str, str]:
    """
    Returns (subject, cleaned_body).
    Strips headers, signatures, quoted text, and normalises whitespace.
    """
    # Extract subject before stripping headers
    subject_match = _SUBJECT_RE.search(raw)
    subject = subject_match.group(1).strip() if subject_match else ""

    body = _strip_headers(raw)
    body = _strip_signature(body)
    # Remove quoted lines ("> ...")
    body = _QUOTE_RE.sub("", body)
    # Collapse multiple blank lines
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return subject, body


# ── Main loader ────────────────────────────────────────────────────────────────

def load_newsgroups(
    dataset_dir: Path,
    *,
    min_text_len: int = 50,
    max_docs: int | None = None,
) -> Generator[Document, None, None]:
    """
    Parse 20 Newsgroups documents from the local tar.gz archive.

    Tries mini_newsgroups.tar.gz first (fast dev runs), then 20_newsgroups.tar.gz.

    Args:
        dataset_dir:  Path to the directory containing the tar.gz files.
        min_text_len: Skip documents with body shorter than this (noise removal).
        max_docs:     Optional cap for fast iteration / testing.

    Yields:
        Document instances in order of appearance in the archive.
    """
    # Prefer mini corpus for fast dev; fall back to full corpus
    candidates = [
        dataset_dir / "mini_newsgroups.tar.gz",
        dataset_dir / "20_newsgroups.tar.gz",
    ]
    tar_path: Path | None = None
    for c in candidates:
        if c.exists():
            tar_path = c
            break

    if tar_path is None:
        raise FileNotFoundError(
            f"No newsgroup archive found in {dataset_dir}. "
            "Expected 'mini_newsgroups.tar.gz' or '20_newsgroups.tar.gz'."
        )

    logger.info(f"Loading dataset from: {tar_path.name}")

    doc_id = 0
    yielded = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        # Filter to only files (skip directories)
        file_members = [m for m in members if m.isfile()]

        logger.info(f"Found {len(file_members)} files in archive")

        for member in file_members:
            if max_docs is not None and yielded >= max_docs:
                break

            # Archive structure: <root>/<category>/<doc_id>
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue  # skip files at root level

            # The category is the parent directory name
            category = parts[-2]

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read().decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Failed to read {member.name}: {e}")
                continue

            subject, body = _clean_text(raw)

            if len(body) < min_text_len:
                logger.debug(f"Skipping short doc {member.name} ({len(body)} chars)")
                doc_id += 1
                continue

            # Combine subject + body for richer embedding input
            # Subject is often the most information-dense part of a newsgroup post
            full_text = f"{subject}\n\n{body}" if subject else body

            yield Document(
                doc_id=doc_id,
                category=category,
                subject=subject,
                text=full_text,
                raw_filename=member.name,
            )
            doc_id += 1
            yielded += 1

    logger.success(f"Loaded {yielded} documents across {_count_categories(yielded)} categories")


def _count_categories(n: int) -> str:
    """Helper for log message — shows 'up to 20'."""
    return "up to 20"


def get_categories(docs: list[Document]) -> list[str]:
    """Return sorted unique category names from a document list."""
    return sorted({d.category for d in docs})
