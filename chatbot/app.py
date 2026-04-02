import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from chatbot import ingest_file, ingest_text, chat, clear_knowledge_base, get_stats

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXT    = {"txt", "md", "pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a file (.txt / .md / .pdf) or raw pasted text."""
    # Pasted text
    if request.is_json:
        text = request.get_json().get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        count = ingest_text(text)
        return jsonify({"message": f"Text ingested. Total chunks: {count}"})

    # File upload
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
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400
    try:
        answer = chat(message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"answer": answer, "stats": get_stats()})


@app.route("/clear", methods=["POST"])
def clear():
    clear_knowledge_base()
    return jsonify({"message": "Knowledge base and conversation cleared."})


@app.route("/stats")
def stats():
    return jsonify(get_stats())


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5001)
