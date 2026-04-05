"""
llm.py
------
Handles all communication with the language model via the OpenRouter API.

This module has one job: build the prompt and call the LLM.
It does NOT write to memory — that is done by chatbot.py so the
write order (DB first, then Redis) is always guaranteed.

Read path:
    memory.get_history(session_id)
        → tries Redis first (fast cache)
        → falls back to DB if Redis key is missing or Redis is down

All settings are read from config.py.
"""

import logging
import requests

import config
import memory

log = logging.getLogger("rag.llm")

OPENROUTER_API_KEY = config.OPENROUTER_API_KEY
OPENROUTER_MODEL   = config.OPENROUTER_MODEL
OPENROUTER_URL     = config.OPENROUTER_URL

SYSTEM_PROMPT = (
    "You are a helpful FAQ assistant. Answer the user's question using ONLY the context provided.\n"
    "If the answer is not in the context, say \"I don't have information about that in the uploaded documents.\"\n"
    "Be concise, clear, and friendly. Do not make up information."
)


def call_llm(session_id: str, context_chunks: list[str], user_message: str) -> str:
    """
    Build the full prompt and call the OpenRouter LLM.

    Reads conversation history from Redis (with automatic DB fallback via memory.py).
    Does NOT write turns — chatbot.py calls memory.add_turn() before and after this
    function so DB is always written first.

    Raises requests.HTTPError or requests.Timeout on API failure.
    """
    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY is not set")
        return "⚠️ OPENROUTER_API_KEY is not set. Please set it as an environment variable."

    # Build context block from retrieved chunks
    if context_chunks:
        context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    else:
        log.warning("No context chunks for session=%s query='%s'", session_id, user_message)
        context = "No relevant documents found."

    system_content = f"{SYSTEM_PROMPT}\n\n--- CONTEXT ---\n{context}\n--- END CONTEXT ---"

    # Read history from Redis → DB fallback (handled inside memory.get_history)
    history = memory.get_history(session_id)
    log.debug("LLM using %d history messages from cache (session=%s)", len(history), session_id)

    messages = [{"role": "system", "content": system_content}] + history

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
    }
    payload = {"model": OPENROUTER_MODEL, "messages": messages}

    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        log.info("LLM response received (session=%s, chars=%d)", session_id, len(answer))
        return answer
    except requests.exceptions.Timeout:
        log.error("LLM request timed out (session=%s)", session_id)
        raise
    except requests.exceptions.HTTPError as e:
        log.error("LLM HTTP error %s (session=%s): %s",
                  e.response.status_code, session_id, e.response.text[:200])
        raise
    except Exception as e:
        log.error("Unexpected LLM error (session=%s): %s", session_id, e)
        raise
