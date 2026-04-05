"""
app.py
------
Flask web server — the entry point for the application.

Defines all HTTP routes and delegates work to chatbot.py.
Each browser tab gets a unique session_id cookie which is passed to
every chat/stats/clear call so Redis memory is fully isolated per session.
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template, make_response

from werkzeug.utils import secure_filename

import config
import memory
import database
from chatbot import ingest_file, ingest_text, chat, clear_knowledge_base, get_stats

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXT   = {"txt", "md", "pdf"}
SESSION_COOKIE = "rag_session_id"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# ── Session helper ────────────────────────────────────────────────────────────

def _get_session_id() -> str:
    """Read the session ID from the request cookie, or generate a new one."""
    return request.cookies.get(SESSION_COOKIE) or str(uuid.uuid4())


def _attach_session(response, session_id: str):
    """Write the session ID cookie onto the response if it wasn't already set."""
    if not request.cookies.get(SESSION_COOKIE):
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            max_age=config.REDIS_SESSION_TTL,
            httponly=True,
            samesite="Lax",
        )
    return response


def _allowed(filename: str) -> bool:
    """Check whether the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    session_id = _get_session_id()
    resp = make_response(render_template("index.html"))
    return _attach_session(resp, session_id)


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a file (.txt / .md / .pdf) or raw pasted text and ingest it."""
    if request.is_json:
        text = request.get_json().get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        count = ingest_text(text)
        return jsonify({"message": f"Text ingested. Total chunks: {count}"})

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Only .txt, .md, .pdf files are supported"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        count = ingest_file(path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": f'"{filename}" ingested. Total chunks: {count}'})


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Receive a user message, run the retrieval pipeline, and return the LLM answer."""
    session_id = _get_session_id()
    data       = request.get_json()
    message    = (data or {}).get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        answer = chat(session_id, message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    resp = make_response(jsonify({"answer": answer, "stats": get_stats(session_id)}))
    return _attach_session(resp, session_id)


@app.route("/history")
def history():
    """Return the full conversation history for this session from the DB."""
    session_id = _get_session_id()
    resp = make_response(jsonify(database.get_all_messages(session_id)))
    return _attach_session(resp, session_id)


@app.route("/clear", methods=["POST"])
def clear():
    """Reset the knowledge base and this session's conversation memory."""
    session_id = _get_session_id()
    memory.clear_session(session_id)          # only clears THIS session
    clear_knowledge_base()                    # clears KB + all sessions
    return jsonify({"message": "Knowledge base and conversation cleared."})


@app.route("/stats")
def stats():
    """Return current system stats including Redis memory info for this session."""
    session_id = _get_session_id()
    resp = make_response(jsonify(get_stats(session_id)))
    return _attach_session(resp, session_id)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    database.init_db()          # create DB table if not exists
    config.log_active_config()
    app.run(debug=True, port=5001)
