"""
database.py
-----------
Permanent storage for all conversation messages using SQLite.

Every message (user and assistant) is written here immediately and
kept forever — this is the source of truth for chat history.

Redis is the fast cache that the LLM reads from. If a session's Redis
key has expired or Redis is unavailable, this module re-hydrates Redis
from the DB so no history is ever truly lost.

Schema:
    messages
        id          INTEGER  PRIMARY KEY AUTOINCREMENT
        session_id  TEXT     NOT NULL
        role        TEXT     NOT NULL   ('user' | 'assistant')
        content     TEXT     NOT NULL
        created_at  TEXT     NOT NULL   (ISO-8601 UTC timestamp)
"""

import sqlite3
import logging
from datetime import datetime, timezone
from contextlib import contextmanager

import config

log = logging.getLogger("rag.database")

DB_PATH = config.DB_PATH

# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_db():
    """Create the messages table if it does not already exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages (session_id)")
    log.info("Database ready at %s", DB_PATH)


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _connect():
    """Yield a SQLite connection with WAL mode and auto-commit on success."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads/writes
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str):
    """Persist a single message turn to the database immediately."""
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, ts),
        )
    log.debug("DB saved: session=%s role=%s", session_id, role)


def get_recent_messages(session_id: str, limit: int) -> list[dict]:
    """
    Return the most recent `limit` messages for a session, oldest first.
    Used to re-hydrate Redis when a session cache has expired.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY created_at ASC
            """,
            (session_id, limit),
        ).fetchall()
    messages = [{"role": r["role"], "content": r["content"]} for r in rows]
    log.debug("DB fetched %d messages for session=%s", len(messages), session_id)
    return messages


def get_all_messages(session_id: str) -> list[dict]:
    """Return the complete conversation history for a session from the DB."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
            for r in rows]


def delete_session(session_id: str):
    """Permanently delete all messages for a session from the DB."""
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    log.info("DB messages deleted for session=%s", session_id)


def delete_all():
    """Permanently delete all messages for all sessions."""
    with _connect() as conn:
        conn.execute("DELETE FROM messages")
    log.info("All DB messages deleted")
