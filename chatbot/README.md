# 🤖 FAQ Knowledge Base Chatbot

A conversational chatbot that lets you upload documents (`.txt`, `.md`, `.pdf`) or paste text, then ask questions about them in natural language. Powered by **FAISS** for semantic retrieval and **OpenRouter** for LLM-generated answers, with full **conversation memory**.

---

## 🗂 Project Structure

```
chatbot/
├── app.py                  # Flask backend (upload, chat, clear, stats endpoints)
├── chatbot.py              # Core logic: FAISS indexing, retrieval, memory, OpenRouter LLM
├── requirements.txt        # Python dependencies
├── uploads/                # Uploaded files stored here (auto-created)
├── templates/
│   └── index.html          # Chat UI with sidebar upload panel
└── README.md
```

---

## 🧠 Architecture

```
User uploads file / pastes text
        │
        ▼
  Text chunking (400-word chunks, 80-word overlap)
        │
        ▼
  Sentence Transformer → 384-dim embeddings
        │
        ▼
  FAISS IndexFlatIP (cosine similarity after L2 norm)
        │
─────────────────────────────────────────────────────
User asks a question
        │
        ▼
  Query → embedding → FAISS top-4 retrieval (score > 0.25)
        │
        ▼
  System prompt + retrieved context + conversation history
        │
        ▼
  OpenRouter LLM (Mistral / any model) → Answer
        │
        ▼
  Answer stored in memory → shown in chat UI
```

---

## 🚀 Quick Start

### 1. Navigate to the chatbot folder

```bash
cd chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenRouter API key

```bash
# Windows
set OPENROUTER_API_KEY=your_key_here

# macOS/Linux
export OPENROUTER_API_KEY=your_key_here
```

Get your free API key at: https://openrouter.ai/keys

### 5. (Optional) Change the model

```bash
set OPENROUTER_MODEL=mistralai/mistral-7b-instruct   # default
# Other options:
# set OPENROUTER_MODEL=openai/gpt-3.5-turbo
# set OPENROUTER_MODEL=anthropic/claude-3-haiku
# set OPENROUTER_MODEL=google/gemma-7b-it
```

Browse all models at: https://openrouter.ai/models

### 6. Run the app

```bash
python app.py
```

### 7. Open in browser

```
http://127.0.0.1:5001
```

---

## 🖥 How to Use

1. **Upload documents** — drag & drop or click the upload zone (`.txt`, `.md`, `.pdf`)
2. **Or paste text** — paste any FAQ content directly into the text area and click "Ingest Text"
3. **Ask questions** — type in the chat box and press Enter
4. The bot answers using only your uploaded content, with memory of the conversation

---

## 🔌 API Reference

### `POST /upload`

**File upload:**
```
Content-Type: multipart/form-data
Body: file=<file>
```

**Paste text:**
```json
{ "text": "Your FAQ content here..." }
```

**Response:**
```json
{ "message": "\"faq.txt\" ingested. Total chunks: 24" }
```

---

### `POST /chat`

```json
{ "message": "What is your refund policy?" }
```

**Response:**
```json
{
  "answer": "Our refund policy allows returns within 30 days...",
  "stats": { "chunks": 24, "indexed": true, "turns": 3, "model": "mistralai/mistral-7b-instruct" }
}
```

---

### `POST /clear`

Clears the FAISS index, all uploaded chunks, and conversation history.

---

### `GET /stats`

```json
{ "chunks": 24, "indexed": true, "turns": 3, "model": "mistralai/mistral-7b-instruct" }
```

---

## 💬 Conversation Memory

- The last **6 turns** (user + assistant pairs) are kept in memory
- Each new question is answered with awareness of previous exchanges
- Memory is cleared when you click "Clear Everything"

---

## ➕ Supported File Types

| Format | Notes |
|--------|-------|
| `.txt` | Plain text |
| `.md`  | Markdown files |
| `.pdf` | Requires `pdfplumber` (included in requirements) |

Max file size: **10 MB**

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | `FAISS` (IndexFlatIP + L2 normalization) |
| LLM | OpenRouter API (Mistral, GPT, Claude, Gemma, etc.) |
| Backend | `Flask` |
| Frontend | HTML + CSS + Vanilla JS |

---

## 📋 Requirements

- Python 3.9+
- OpenRouter API key (free tier available)
- ~500MB disk space (for embedding model download on first run)
- Internet connection on first run
