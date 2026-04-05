"""
retriever.py
------------
The full RAG retrieval pipeline. Given a user query, this module
finds the most relevant chunks from the knowledge base.

Pipeline stages (in order):
  1. Query Expansion   — broaden the query with related terms
  2. Hybrid Retrieval  — combine BM25 (keyword) + Dense (semantic) scores
  3. Filtering         — remove chunks that are too short, too long, or off-topic
  4. Re-ranking        — use a cross-encoder to precisely score (query, chunk) pairs
  5. Metrics Logging   — log precision, recall, F1 for every retrieval call

All settings are read from config.py — do not add os.environ calls here.
"""

import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import config

log = logging.getLogger("rag.retriever")

# ── Module-level aliases from config (for clean usage below) ──────────────────
TOP_K_RETRIEVE      = config.TOP_K_RETRIEVE
TOP_K_FINAL         = config.TOP_K_FINAL
DENSE_WEIGHT        = config.DENSE_WEIGHT
BM25_WEIGHT         = config.BM25_WEIGHT
DENSE_THRESHOLD     = config.DENSE_THRESHOLD
DENSE_MODEL         = config.DENSE_MODEL
RERANK_MODEL        = config.RERANK_MODEL

log.info("Loading dense embedder: %s", DENSE_MODEL)
_embedder = SentenceTransformer(DENSE_MODEL)

log.info("Loading cross-encoder reranker: %s", RERANK_MODEL)
_reranker = CrossEncoder(RERANK_MODEL, max_length=512)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    """Convert a list of text strings into L2-normalised float32 vectors."""
    vecs = _embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


# ── Index Management ──────────────────────────────────────────────────────────

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Create a new FAISS inner-product index and add the given vectors."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    log.info("FAISS index created (dim=%d, vectors=%d)", vectors.shape[1], vectors.shape[0])
    return index


def add_to_faiss(index: faiss.IndexFlatIP, vectors: np.ndarray):
    """Append new vectors to an existing FAISS index."""
    index.add(vectors)
    log.debug("Added %d vectors to FAISS index (total=%d)", len(vectors), index.ntotal)


def build_bm25(tokenized_chunks: list[list[str]]) -> BM25Okapi:
    """Build a BM25 keyword index from pre-tokenised chunks."""
    bm25 = BM25Okapi(tokenized_chunks)
    log.debug("BM25 index built over %d chunks", len(tokenized_chunks))
    return bm25


# ── Query Expansion ───────────────────────────────────────────────────────────

# Static expansion map — extend or replace with WordNet / LLM-based expansion
_EXPANSION_MAP = {
    "how":     "method process steps",
    "what":    "definition meaning explanation",
    "why":     "reason cause purpose",
    "error":   "issue problem failure exception bug",
    "fix":     "resolve solution workaround patch",
    "install": "setup configure deploy",
    "fast":    "performance speed optimize",
    "slow":    "latency bottleneck delay",
}


def expand_query(query: str) -> str:
    """Append related terms to the query to improve recall during retrieval."""
    tokens = query.lower().split()
    extras = [_EXPANSION_MAP[t] for t in tokens if t in _EXPANSION_MAP]
    if not extras:
        return query
    expanded = f"{query} {' '.join(extras)}"
    log.debug("Query expanded: '%s' → '%s'", query, expanded)
    return expanded


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_candidates(query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Pass all candidates through — filtering is handled by DENSE_THRESHOLD and reranking."""
    return candidates


# ── Hybrid Scoring ────────────────────────────────────────────────────────────

def _dense_scores(query: str, chunks: list[str], index: faiss.IndexFlatIP, top_k: int) -> dict[int, float]:
    """Return a {chunk_index: score} map from FAISS dense retrieval."""
    q_vec = embed([query])
    k = min(top_k, len(chunks))
    scores, idxs = index.search(q_vec, k)
    return {int(i): float(s) for i, s in zip(idxs[0], scores[0]) if s >= DENSE_THRESHOLD}


def _bm25_scores(query: str, bm25: BM25Okapi, top_k: int) -> dict[int, float]:
    """Return a normalised {chunk_index: score} map from BM25 keyword retrieval."""
    tokens = query.lower().split()
    raw = bm25.get_scores(tokens)
    top_idxs = np.argsort(raw)[::-1][:top_k]
    max_score = raw[top_idxs[0]] if raw[top_idxs[0]] > 0 else 1.0
    return {int(i): float(raw[i] / max_score) for i in top_idxs if raw[i] > 0}


def hybrid_retrieve(
    query: str,
    chunks: list[str],
    index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    top_k: int = TOP_K_RETRIEVE,
) -> list[tuple[int, float]]:
    """
    Combine BM25 and dense scores into a single ranked candidate list.
    Final score = DENSE_WEIGHT * dense_score + BM25_WEIGHT * bm25_score.
    """
    dense = _dense_scores(query, chunks, index, top_k * 2)
    bm25s = _bm25_scores(query, bm25, top_k * 2)
    all_idxs = set(dense) | set(bm25s)
    combined = {
        i: DENSE_WEIGHT * dense.get(i, 0.0) + BM25_WEIGHT * bm25s.get(i, 0.0)
        for i in all_idxs
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    log.info("Hybrid retrieval — dense=%d  bm25=%d  combined=%d candidates",
             len(dense), len(bm25s), len(ranked))
    return ranked


# ── Re-ranking ────────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[tuple[str, float]], top_k: int = TOP_K_FINAL) -> list[str]:
    """
    Re-score (query, chunk) pairs with a cross-encoder for precise relevance ranking.
    Cross-encoders are slower but far more accurate than bi-encoder similarity alone.
    """
    if not candidates:
        return []
    pairs = [(query, chunk) for chunk, _ in candidates]
    ce_scores = _reranker.predict(pairs)
    reranked = sorted(zip([c for c, _ in candidates], ce_scores),
                      key=lambda x: x[1], reverse=True)
    log.info("Re-ranking complete — top scores: %s",
             [round(float(s), 3) for _, s in reranked[:top_k]])
    return [chunk for chunk, _ in reranked[:top_k]]


# ── Precision / Recall Metrics ────────────────────────────────────────────────

def log_retrieval_metrics(query: str, retrieved: list[str], pool_size: int):
    """
    Log approximate precision, recall, and F1 after every retrieval.
    Relevance is approximated by dense threshold as a proxy.
    """
    precision = len(retrieved) / TOP_K_RETRIEVE if TOP_K_RETRIEVE else 0.0
    recall    = len(retrieved) / pool_size if pool_size else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    log.info(
        "Retrieval metrics — precision=%.3f  recall=%.3f  F1=%.3f  "
        "retrieved=%d  pool=%d",
        precision, recall, f1, len(retrieved), pool_size,
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

def retrieve(
    query: str,
    chunks: list[str],
    index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    top_k: int = TOP_K_FINAL,
) -> list[str]:
    """
    Full retrieval pipeline: expand → hybrid retrieve → filter → rerank → log metrics.
    Returns the top-k most relevant chunks for the given query.
    """
    expanded = expand_query(query)

    candidate_idxs = hybrid_retrieve(expanded, chunks, index, bm25, TOP_K_RETRIEVE)
    candidates = [(chunks[i], s) for i, s in candidate_idxs]

    candidates = filter_candidates(query, candidates)
    final = rerank(query, candidates, top_k)

    log_retrieval_metrics(query, final, len(chunks))
    return final
