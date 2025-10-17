# Custom Dataset AI Chatbot

A minimal Retrieval-Augmented Generation (RAG) chatbot over your local `.txt` and `.md` files using FastAPI, FAISS, and sentence-transformers. It provides a streaming chat API and a tiny web UI.

## Project layout

- `app/` — FastAPI app and services
  - `app/main.py` — API server with `/chat`
  - `app/config.py` — configuration via env vars or defaults
  - `app/services/` — chunking, retriever, generator
- `scripts/` — one-off utilities
  - `scripts/ingest.py` — chunk files, build embeddings, write FAISS index
- `data/` — put your `.txt` or `.md` documents here
- `indexes/` — generated FAISS index and metadata
- `web/` — simple static UI (`index.html`)

## Setup

1. Python 3.10+
2. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Optional: copy `.env.example` to `.env` and adjust paths/models.

## Prepare data and build index

Place `.txt` and `.md` files under `data/`, then run:

```bash
python scripts/ingest.py
```

This creates `indexes/chunks.index` and `indexes/chunks.meta.jsonl`.

## Run the server

The default generator model is large. For CPU-only environments it may be slow. You can switch to a smaller model, e.g. `HuggingFaceTB/SmolLM2-1.7B-Instruct` via `GEN_MODEL`.

```bash
uvicorn app.main:app --reload --port 8000
```

Open `web/index.html` in a browser, or serve it via any static server. The UI POSTs to `/chat` on the same origin; if you host it separately, proxy `/chat` to the FastAPI server.

## Notes

- Retrieval uses cosine similarity via FAISS inner product with L2-normalized embeddings.
- The prompt restricts answers to the retrieved contexts; if not found, it will say it doesn't know.
- For production, consider a proper vector DB (Qdrant, Chroma), better chunking, and citation rendering.
