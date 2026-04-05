"""
chunker.py
----------
Responsible for splitting raw text into clean, meaningful chunks
before they are stored in the knowledge base.

Why separate? Chunking strategy directly affects retrieval quality.
Keeping it isolated makes it easy to swap or tune without touching
retrieval or LLM logic.
"""

import re
import logging
import hashlib
import config

log = logging.getLogger("rag.chunker")


def split_sentences(text: str) -> list[str]:
    """Break a block of text into individual sentences using punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def make_chunks(text: str) -> list[str]:
    """
    Convert raw text into overlapping sentence-window chunks.
    Each chunk contains CHUNK_SIZE sentences, sliding by (CHUNK_SIZE - CHUNK_OVERLAP).
    """
    sentences = split_sentences(text)
    if not sentences:
        log.warning("No sentences found in provided text")
        return []

    step = max(1, config.CHUNK_SIZE - config.CHUNK_OVERLAP)
    chunks = [
        " ".join(sentences[i: i + config.CHUNK_SIZE])
        for i in range(0, len(sentences), step)
        if sentences[i: i + config.CHUNK_SIZE]
    ]

    log.info("Chunking complete: %d sentences → %d chunks", len(sentences), len(chunks))
    return chunks


# ── Deduplication ─────────────────────────────────────────────────────────────

def chunk_hash(chunk: str) -> str:
    """Generate a short fingerprint for a chunk to detect duplicates."""
    return hashlib.md5(chunk.strip().lower().encode()).hexdigest()


def deduplicate(chunks: list[str], seen_hashes: set[str]) -> list[str]:
    """
    Remove chunks that are identical (or near-identical) to ones already ingested.
    Updates seen_hashes in-place so the caller's set stays current.
    """
    unique = []
    for chunk in chunks:
        h = chunk_hash(chunk)
        if h in seen_hashes:
            log.debug("Duplicate chunk skipped")
            continue
        seen_hashes.add(h)
        unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed:
        log.info("Deduplication removed %d duplicate chunk(s)", removed)
    return unique
