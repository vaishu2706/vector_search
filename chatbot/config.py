"""
config.py
---------
Single source of truth for ALL application settings.

Every configurable value lives here. Other modules import from this file
instead of calling os.environ.get() themselves.

How to configure (in order of priority — highest first):
  1. Real environment variables  (e.g. set in your terminal or CI/CD)
  2. A .env file in the chatbot/ folder  (easiest for local development)
  3. The defaults defined below  (work out of the box, no setup needed)

To customise, copy .env.example to .env and edit the values.
"""

import os
import logging

log = logging.getLogger("rag.config")

# Load .env file if present (silently ignored if the file doesn't exist)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        log.info("Loaded config from .env file")
    else:
        log.info("No .env file found — using environment variables and defaults")
except ImportError:
    log.warning("python-dotenv not installed — .env file will not be loaded. "
                "Run: pip install python-dotenv")


def _get(key: str, default: str) -> str:
    """Read a string value from environment, falling back to default."""
    return os.environ.get(key, default)


def _int(key: str, default: int) -> int:
    """Read an integer value from environment, falling back to default."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        log.warning("Invalid value for %s, using default %s", key, default)
        return default


def _float(key: str, default: float) -> float:
    """Read a float value from environment, falling back to default."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        log.warning("Invalid value for %s, using default %s", key, default)
        return default


# ── LLM / OpenRouter ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = _get("OPENROUTER_MODEL",   "mistralai/mistral-7b-instruct")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MAX_MEMORY_TURNS   = _int("MAX_MEMORY_TURNS", 6)   # number of user/assistant pairs to remember

# ── Database (permanent chat storage) ───────────────────────────────────────
DB_PATH = _get("DB_PATH", os.path.join(os.path.dirname(__file__), "chat_history.db"))

# ── Redis (conversation memory) ───────────────────────────────────────────────
REDIS_HOST        = _get("REDIS_HOST",     "localhost")
REDIS_PORT        = _int("REDIS_PORT",     6379)
REDIS_PASSWORD    = _get("REDIS_PASSWORD", "")       # leave empty if no auth
REDIS_DB          = _int("REDIS_DB",       0)
REDIS_SESSION_TTL = _int("REDIS_SESSION_TTL", 3600)  # seconds before idle session expires (1 hr)

# ── Embedding & Reranking Models ──────────────────────────────────────────────
DENSE_MODEL  = _get("DENSE_MODEL",  "all-MiniLM-L6-v2")
RERANK_MODEL = _get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = _int("CHUNK_SIZE",    5)   # sentences per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 1)   # sentences shared between adjacent chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE      = _int("TOP_K_RETRIEVE",   10)    # candidates fetched before reranking
TOP_K_FINAL         = _int("TOP_K_FINAL",       4)    # chunks passed to the LLM after reranking
DENSE_WEIGHT        = _float("DENSE_WEIGHT",   0.6)   # semantic score weight in hybrid retrieval
BM25_WEIGHT         = _float("BM25_WEIGHT",    0.4)   # keyword score weight in hybrid retrieval
DENSE_THRESHOLD     = _float("DENSE_THRESHOLD", 0.20) # minimum dense similarity to keep a chunk


def log_active_config():
    """Print all active settings to the console at startup (API key is masked)."""
    masked_key = (OPENROUTER_API_KEY[:6] + "…") if OPENROUTER_API_KEY else "NOT SET ⚠️"
    log.info("=" * 55)
    log.info("Active Configuration")
    log.info("=" * 55)
    log.info("  LLM model        : %s", OPENROUTER_MODEL)
    log.info("  API key          : %s", masked_key)
    log.info("  Dense model      : %s", DENSE_MODEL)
    log.info("  Rerank model     : %s", RERANK_MODEL)
    log.info("  Chunk size       : %d sentences (overlap=%d)", CHUNK_SIZE, CHUNK_OVERLAP)
    log.info("  Top-K retrieve   : %d  →  final: %d", TOP_K_RETRIEVE, TOP_K_FINAL)
    log.info("  Hybrid weights   : dense=%.1f  bm25=%.1f", DENSE_WEIGHT, BM25_WEIGHT)
    log.info("  Dense threshold  : %.2f", DENSE_THRESHOLD)
    log.info("  Memory turns     : %d", MAX_MEMORY_TURNS)
    log.info("  DB path          : %s", DB_PATH)
    log.info("  Redis            : %s:%d (db=%d, ttl=%ds)",
             REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_SESSION_TTL)
    log.info("=" * 55)
