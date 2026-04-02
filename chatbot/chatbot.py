import os
import re
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── In-memory store ──────────────────────────────────────────────────────────
_chunks: list[str] = []
_index: faiss.IndexFlatIP | None = None
_conversation: list[dict] = []          # [{role, content}, ...]


# ── Document ingestion ───────────────────────────────────────────────────────
def _split_chunks(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]


def ingest_text(text: str) -> int:
    """Add raw text to the FAISS index. Returns total chunk count."""
    global _index, _chunks
    new_chunks = _split_chunks(text)
    if not new_chunks:
        return len(_chunks)

    vecs = _embed(new_chunks)
    if _index is None:
        _index = faiss.IndexFlatIP(vecs.shape[1])
    _index.add(vecs)
    _chunks.extend(new_chunks)
    return len(_chunks)


def ingest_file(filepath: str) -> int:
    """Parse .txt / .md / .pdf and ingest into index."""
    ext = os.path.splitext(filepath)[1].lower()
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


def clear_knowledge_base():
    global _chunks, _index, _conversation
    _chunks, _index, _conversation = [], None, []


# ── Retrieval ────────────────────────────────────────────────────────────────
def _embed(texts: list[str]) -> np.ndarray:
    vecs = _embedder.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def retrieve(query: str, top_k: int = 4) -> list[str]:
    if _index is None or len(_chunks) == 0:
        return []
    q = _embed([query])
    scores, idxs = _index.search(q, min(top_k, len(_chunks)))
    return [_chunks[i] for i, s in zip(idxs[0], scores[0]) if s > 0.25]


# ── Conversation memory ──────────────────────────────────────────────────────
def _trim_memory(max_turns: int = 6):
    """Keep only the last N user/assistant pairs."""
    global _conversation
    if len(_conversation) > max_turns * 2:
        _conversation = _conversation[-(max_turns * 2):]


# ── LLM call via OpenRouter ──────────────────────────────────────────────────
def _call_llm(messages: list[dict]) -> str:
    if not OPENROUTER_API_KEY:
        return "⚠️ OPENROUTER_API_KEY is not set. Please set it as an environment variable."
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
    }
    payload = {"model": OPENROUTER_MODEL, "messages": messages}
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Main chat function ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful FAQ assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have information about that in the uploaded documents."
Be concise, clear, and friendly. Do not make up information."""


def chat(user_message: str) -> str:
    context_chunks = retrieve(user_message)
    context = "\n\n".join(context_chunks) if context_chunks else "No relevant documents found."

    system = f"{SYSTEM_PROMPT}\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"

    _conversation.append({"role": "user", "content": user_message})
    _trim_memory()

    messages = [{"role": "system", "content": system}] + _conversation

    answer = _call_llm(messages)
    _conversation.append({"role": "assistant", "content": answer})
    return answer


def get_stats() -> dict:
    return {
        "chunks": len(_chunks),
        "indexed": _index is not None,
        "turns": len(_conversation) // 2,
        "model": OPENROUTER_MODEL,
    }
