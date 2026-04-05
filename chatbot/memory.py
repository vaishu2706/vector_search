"""
memory.py
---------
Two-layer conversation memory: SQLite (permanent) + Redis (fast cache).

Write path  →  every new message is saved to DB first, then pushed to Redis.
Read path   →  LLM always reads from Redis (fast).
                If the Redis key is missing (expired or Redis restarted),
                the recent history is automatically re-loaded from the DB
                so no conversation context is ever lost.

                DB  ←──── source of truth (permanent, all messages)
                Redis ←── hot cache (capped to MAX_MEMORY_TURNS, has TTL)

If Redis is completely unavailable the module falls back to reading
directly from the DB, so the app keeps working without Redis.
"""

import json
import logging

import config
import database

log = logging.getLogger("rag.memory")

# ── Redis connection ──────────────────────────────────────────────────────────

_redis = None

try:
    import redis as _redis_lib
    _client = _redis_lib.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD or None,
        db=config.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _client.ping()
    _redis = _client
    log.info("Redis connected at %s:%d (db=%d)", config.REDIS_HOST, config.REDIS_PORT, config.REDIS_DB)
except ImportError:
    log.warning("redis-py not installed — reads will fall back to DB directly")
except Exception as e:
    log.warning("Redis unavailable (%s) — reads will fall back to DB directly", e)

_KEY_PREFIX = "chat:memory:"


def _key(session_id: str) -> str:
    """Build the Redis cache key for a session."""
    return f"{_KEY_PREFIX}{session_id}"


# ── Write path ────────────────────────────────────────────────────────────────

def add_turn(session_id: str, role: str, content: str):
    """
    Persist a message to the DB (permanent) and push it into the Redis cache.
    DB write always happens first — Redis is best-effort.
    """
    # 1. Always write to DB first
    database.save_message(session_id, role, content)

    # 2. Push to Redis cache if available
    if _redis:
        max_msgs = config.MAX_MEMORY_TURNS * 2
        key = _key(session_id)
        pipe = _redis.pipeline()
        pipe.rpush(key, json.dumps({"role": role, "content": content}))
        pipe.ltrim(key, -max_msgs, -1)           # keep only recent turns in cache
        pipe.expire(key, config.REDIS_SESSION_TTL)
        pipe.execute()
        log.debug("Redis cache updated: session=%s role=%s", session_id, role)


# ── Read path ─────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> list[dict]:
    """
    Return recent conversation history for the LLM.

    Priority:
      1. Redis cache  — fast, already capped to MAX_MEMORY_TURNS
      2. DB fallback  — used when Redis key is missing or Redis is down,
                        re-hydrates the Redis cache for future calls
    """
    max_msgs = config.MAX_MEMORY_TURNS * 2

    if _redis:
        key = _key(session_id)
        cached = _redis.lrange(key, 0, -1)

        if cached:
            log.debug("Redis cache hit: session=%s (%d messages)", session_id, len(cached))
            return [json.loads(m) for m in cached]

        # Cache miss — load from DB and warm Redis
        log.info("Redis cache miss for session=%s — loading from DB", session_id)
        messages = database.get_recent_messages(session_id, limit=max_msgs)
        _warm_cache(session_id, messages)
        return messages

    # Redis not available — read directly from DB
    log.debug("Redis unavailable — reading history from DB for session=%s", session_id)
    return database.get_recent_messages(session_id, limit=max_msgs)


def _warm_cache(session_id: str, messages: list[dict]):
    """Push a list of messages into Redis to warm the cache after a miss."""
    if not _redis or not messages:
        return
    key = _key(session_id)
    pipe = _redis.pipeline()
    pipe.delete(key)
    for msg in messages:
        pipe.rpush(key, json.dumps(msg))
    pipe.expire(key, config.REDIS_SESSION_TTL)
    pipe.execute()
    log.info("Redis cache warmed: session=%s (%d messages)", session_id, len(messages))


# ── Clear ─────────────────────────────────────────────────────────────────────

def clear_session(session_id: str):
    """Delete this session's history from both Redis and the DB."""
    if _redis:
        _redis.delete(_key(session_id))
        log.info("Redis cache cleared for session=%s", session_id)
    database.delete_session(session_id)


def clear_all():
    """Delete ALL sessions from both Redis and the DB."""
    if _redis:
        keys = _redis.keys(f"{_KEY_PREFIX}*")
        if keys:
            _redis.delete(*keys)
        log.info("All Redis cache keys cleared (%d sessions)", len(keys))
    database.delete_all()


# ── Metadata ──────────────────────────────────────────────────────────────────

def is_redis_available() -> bool:
    """Return True if Redis is connected and healthy."""
    return _redis is not None


def get_session_info(session_id: str) -> dict:
    """Return metadata about a session: turn count, TTL, and which backend is active."""
    total_in_db = len(database.get_all_messages(session_id))

    if _redis:
        key   = _key(session_id)
        ttl   = _redis.ttl(key)
        cached = _redis.llen(key)
        return {
            "turns":          total_in_db // 2,
            "cached_turns":   cached // 2,
            "ttl_seconds":    ttl,
            "backend":        "redis+sqlite",
        }
    return {
        "turns":        total_in_db // 2,
        "cached_turns": 0,
        "ttl_seconds":  -1,
        "backend":      "sqlite",
    }
