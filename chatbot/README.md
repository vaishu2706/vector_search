# 🧠 RAG Knowledge Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents (`.txt`, `.md`, `.pdf`) or paste text, then ask questions about them in natural language.

It uses a **multi-stage retrieval pipeline** — keyword search + semantic search + cross-encoder reranking — to find the most relevant content before generating an answer with an LLM via [OpenRouter](https://openrouter.ai).

---

## 🗂 Project Structure

```
chatbot/
├── app.py            # Flask web server — handles HTTP routes (upload, chat, clear, stats)
├── chatbot.py        # Orchestrator — wires chunker, retriever, and LLM together
├── chunker.py        # Text splitting — breaks documents into clean, overlapping chunks
├── retriever.py      # Retrieval pipeline — BM25 + dense search, filtering, reranking, metrics
├── llm.py            # LLM client — calls OpenRouter API, manages conversation memory
├── requirements.txt  # Python dependencies
├── uploads/          # Uploaded files are saved here (auto-created on startup)
├── templates/
│   └── index.html    # Chat UI — sidebar upload, pipeline stats panel, source chunk preview
└── README.md
```

### Why is it split into multiple files?

Each file has one clear job. A non-developer reading the filenames can immediately understand what each part does:

| File | What it does |
|------|-------------|
| `app.py` | Receives requests from the browser, calls the right function, sends back a response |
| `chatbot.py` | The "manager" — coordinates all the other modules |
| `chunker.py` | Cuts documents into small pieces so they can be searched efficiently |
| `retriever.py` | Finds the most relevant pieces for a given question |
| `llm.py` | Sends the question + relevant pieces to the AI model and gets an answer |

---

## 🔄 Retrieval Pipeline (How It Works)

```
User uploads document
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  chunker.py                                         │
│  1. Split text into sentences                       │
│  2. Group into overlapping sentence-window chunks   │
│  3. Remove duplicates (MD5 fingerprint)             │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  retriever.py — Index                               │
│  4. Embed chunks → FAISS dense index                │
│  5. Tokenise chunks → BM25 keyword index            │
└─────────────────────────────────────────────────────┘

User asks a question
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  retriever.py — Query Pipeline                      │
│  6.  Query expansion (add related terms)            │
│  7.  Hybrid retrieval:                              │
│        BM25 score  × 0.4  (keyword match)           │
│      + Dense score × 0.6  (semantic similarity)     │
│  8.  Filter: remove off-topic / wrong-length chunks │
│  9.  Cross-encoder reranking (precise scoring)      │
│  10. Log precision / recall / F1 metrics            │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  llm.py                                             │
│  11. Build prompt: system + context + history       │
│  12. Call OpenRouter LLM → get answer               │
│  13. Store turn in conversation memory              │
└─────────────────────────────────────────────────────┘
        │
        ▼
     Answer shown in chat UI
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
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB) and reranker model (~23 MB) automatically.

### 4. Set your OpenRouter API key

```bash
# Windows
set OPENROUTER_API_KEY=your_key_here

# macOS / Linux
export OPENROUTER_API_KEY=your_key_here
```

Get a free key at: https://openrouter.ai/keys

### 5. Run the app

```bash
python app.py
```

### 6. Open in browser

```
http://127.0.0.1:5001
```

---

## 🖥 How to Use the UI

1. **Upload a file** — drag & drop or click the upload zone (`.txt`, `.md`, `.pdf`, max 10 MB)
2. **Or paste text** — paste any content into the text area and click "Ingest Text"
3. **Ask questions** — type in the chat box and press Enter
4. **View sources** — click "📎 N source chunks" under any bot reply to see which passages were used
5. **Pipeline config** — the sidebar shows live retrieval settings (models, weights, top-K values)
6. **Clear** — click "🗑 Clear Everything" to reset documents and conversation

---

## ⚙️ Configuration (Environment Variables)

All settings have sensible defaults and can be overridden without changing any code.

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `5` | Number of sentences per chunk |
| `CHUNK_OVERLAP` | `1` | Sentences shared between adjacent chunks |
| `MIN_CHUNK_WORDS` | `15` | Chunks shorter than this are discarded |
| `MAX_CHUNK_WORDS` | `300` | Chunks longer than this are hard-split |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K_RETRIEVE` | `10` | Candidate chunks fetched before reranking |
| `TOP_K_FINAL` | `4` | Chunks passed to the LLM after reranking |
| `DENSE_WEIGHT` | `0.6` | Weight for semantic (dense) score in hybrid |
| `BM25_WEIGHT` | `0.4` | Weight for keyword (BM25) score in hybrid |
| `DENSE_THRESHOLD` | `0.20` | Minimum dense similarity score to include a chunk |
| `MIN_LEXICAL_OVERLAP` | `0.05` | Minimum query-term overlap fraction to keep a chunk |
| `DENSE_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct` | LLM model to use for generation |
| `MAX_MEMORY_TURNS` | `6` | Number of conversation turns kept in memory |

**Example — change the LLM model:**
```bash
set OPENROUTER_MODEL=openai/gpt-4o-mini
# or
set OPENROUTER_MODEL=anthropic/claude-3-haiku
```

Browse all available models: https://openrouter.ai/models

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
{ "text": "Your content here..." }
```

**Response:**
```json
{ "message": "\"faq.txt\" ingested. Total chunks: 31" }
```

---

### `POST /chat`

```json
{ "message": "What is the refund policy?" }
```

**Response:**
```json
{
  "answer": "Refunds are accepted within 30 days of purchase...",
  "sources": ["[1] Refunds are accepted within 30 days...", "[2] To request a refund..."],
  "stats": { "chunks": 31, "indexed": true, "turns": 2, "model": "mistralai/mistral-7b-instruct" }
}
```

---

### `POST /clear`

Resets the knowledge base, FAISS index, BM25 index, and conversation memory.

---

### `GET /stats`

```json
{
  "chunks": 31,
  "indexed": true,
  "turns": 2,
  "model": "mistralai/mistral-7b-instruct",
  "dense_model": "all-MiniLM-L6-v2",
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "top_k_retrieve": 10,
  "top_k_final": 4,
  "hybrid_weights": { "dense": 0.6, "bm25": 0.4 }
}
```

---

## 📋 Requirements

- Python 3.10+
- OpenRouter API key (free tier available)
- ~200 MB disk space for model downloads on first run
- Internet connection on first run

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Chunking | Sentence-boundary splitting with overlap |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Semantic search | `sentence-transformers` + `FAISS` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenRouter API (Mistral, GPT, Claude, Gemma, …) |
| Backend | `Flask` |
| Frontend | HTML + CSS + Vanilla JS |

---

## 📊 Logging

Every retrieval call logs structured metrics to the console:

```
2024-01-15 10:23:01 [INFO] rag.retriever — Hybrid retrieval — dense=8  bm25=6  combined=10 candidates
2024-01-15 10:23:01 [INFO] rag.retriever — Filtering: 10 → 7 candidates
2024-01-15 10:23:01 [INFO] rag.retriever — Re-ranking complete — top scores: [0.821, 0.743, 0.612, 0.589]
2024-01-15 10:23:01 [INFO] rag.retriever — Retrieval metrics — precision=1.000  recall=0.875  F1=0.933  retrieved=4  relevant=4  pool=31
```

Logger names: `rag.chatbot`, `rag.chunker`, `rag.retriever`, `rag.llm`
