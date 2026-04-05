"""
chatbot.py
----------
Orchestrator — the single entry point used by app.py (the Flask server).

This file does NOT contain chunking, retrieval, or LLM logic.
It simply wires the three specialist modules together:

  chunker.py   →  splits and deduplicates text
  retriever.py →  finds the most relevant chunks for a query
  llm.py       →  calls the language model and manages chat memory

The in-memory knowledge base (chunks + FAISS index + BM25 index) lives here
so that all modules share one consistent state.
"""

import os
import logging
import faiss
import numpy as np
from dataclasses import dataclass, field

import chunker
import retriever
import llm
import memory
import database

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rag.chatbot")


# ── Shared Knowledge Base State ───────────────────────────────────────────────

@dataclass
class KnowledgeBase:
    """Holds all in-memory state for the vector store and keyword index."""
    chunks: list[str]                   = field(default_factory=list)
    chunk_hashes: set[str]              = field(default_factory=set)
    index: faiss.IndexFlatIP | None     = None   # FAISS dense index
    bm25: object | None                 = None   # BM25 keyword index
    tokenized: list[list[str]]          = field(default_factory=list)

_kb = KnowledgeBase()


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _add_chunks_to_kb(new_chunks: list[str]):
    """Embed new chunks, add them to FAISS and rebuild the BM25 index."""
    vecs = retriever.embed(new_chunks)

    if _kb.index is None:
        _kb.index = retriever.build_faiss_index(vecs)
    else:
        retriever.add_to_faiss(_kb.index, vecs)

    _kb.chunks.extend(new_chunks)
    _kb.tokenized.extend(c.lower().split() for c in new_chunks)
    _kb.bm25 = retriever.build_bm25(_kb.tokenized)

    log.info("Knowledge base updated: %d total chunks", len(_kb.chunks))


# ── Public API (called by app.py) ─────────────────────────────────────────────

def ingest_text(text: str) -> int:
    """
    Chunk, deduplicate, and index raw text into the knowledge base.
    Returns the total number of chunks currently stored.
    """
    raw_chunks = chunker.make_chunks(text)
    new_chunks = chunker.deduplicate(raw_chunks, _kb.chunk_hashes)

    if not new_chunks:
        log.warning("No new content to ingest after deduplication")
        return len(_kb.chunks)

    _add_chunks_to_kb(new_chunks)
    return len(_kb.chunks)


def ingest_file(filepath: str) -> int:
    """
    Read a .txt, .md, or .pdf file and ingest its content into the knowledge base.
    Returns the total number of chunks currently stored.
    """
    ext = os.path.splitext(filepath)[1].lower()
    log.info("Reading file: %s (ext=%s)", filepath, ext)

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return ingest_text(text)


def chat(session_id: str, user_message: str) -> str:
    """
    Full chat flow:
      1. Retrieve relevant chunks from the knowledge base
      2. Save user message to DB + Redis (memory.add_turn)
      3. Call LLM — reads history from Redis (DB fallback inside memory.py)
      4. Save assistant answer to DB + Redis
    """
    log.info("User query (session=%s): '%s'", session_id, user_message)

    context_chunks = []
    if _kb.index is not None and _kb.chunks:
        context_chunks = retriever.retrieve(
            query=user_message,
            chunks=_kb.chunks,
            index=_kb.index,
            bm25=_kb.bm25,
        )
    else:
        log.warning("Knowledge base is empty — answering without context")

    # Write user turn to DB + Redis BEFORE calling LLM
    memory.add_turn(session_id, "user", user_message)

    # LLM reads history from Redis (memory.get_history handles DB fallback)
    answer = llm.call_llm(session_id, context_chunks, user_message)

    # Write assistant answer to DB + Redis AFTER LLM responds
    memory.add_turn(session_id, "assistant", answer)

    return answer


def clear_knowledge_base():
    """Reset the knowledge base and ALL conversation sessions in Redis."""
    global _kb
    _kb = KnowledgeBase()
    memory.clear_all()
    log.info("Knowledge base and all conversation sessions cleared")


def get_stats(session_id: str) -> dict:
    """Return a snapshot of the current system state for the UI stats bar."""
    session_info = memory.get_session_info(session_id)
    return {
        "chunks":          len(_kb.chunks),
        "indexed":         _kb.index is not None,
        "turns":           session_info["turns"],
        "ttl_seconds":     session_info["ttl_seconds"],
        "memory_backend":  session_info["backend"],
        "model":           llm.OPENROUTER_MODEL,
        "dense_model":     retriever.DENSE_MODEL,
        "rerank_model":    retriever.RERANK_MODEL,
        "top_k_retrieve":  retriever.TOP_K_RETRIEVE,
        "top_k_final":     retriever.TOP_K_FINAL,
        "hybrid_weights":  {"dense": retriever.DENSE_WEIGHT, "bm25": retriever.BM25_WEIGHT},
    }
