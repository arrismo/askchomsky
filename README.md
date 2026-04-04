---
title: AskChomsky
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# AskChomsky

Ask questions about Noam Chomsky's work, grounded in a curated corpus with citations.

Powered by LightRAG + Next.js.

## Run Locally

### Prerequisites

- Python 3.11+
- Node.js 20+
- npm

### 1) Backend setup (one-time)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Environment variables

Create a `.env` file in the project root and add:

```env
openrouter_key=YOUR_OPENROUTER_KEY
```

Optional (observability):

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3) Frontend setup (one-time)

```bash
cd frontend
npm install
cd ..
```

### 4) Start the app

Use two terminals from the project root:

Terminal A (FastAPI backend on `http://127.0.0.1:8001`):

```bash
make backend
```

Terminal B (Next.js frontend on `http://localhost:3000`):

```bash
make frontend
```

Then open:

- http://localhost:3000

Notes:

- If you want to run the frontend with a custom backend URL, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.
- Default backend URL in the frontend is `http://localhost:8001`.

## Tools Used

- LightRAG (retrieval-augmented generation)
- LlamaIndex (RAG orchestration)
- OpenAI embeddings: `openai/text-embedding-3-small` (via OpenRouter)
- Model for answer generation: `openai/gpt-4o-mini`
- Langfuse (observability and traces)

## Dataset Used

- Hugging Face dataset: [`mmoise00/chomsky-corpus`](https://huggingface.co/datasets/mmoise00/chomsky-corpus)
- Loaded from the `train` split in the app
- Each record includes:
	- `content`
	- `record_id`
	- `section`
	- `article_title`
	- `article_date`
	- `index_url`

### 1) Index the corpus

```bash
source .venv/bin/activate
python ask.py --ingest --doc-limit 200 --working-dir ./lightrag_store
```

### 2) Query with LightRAG

```bash
source .venv/bin/activate
python ask.py --query "How does Chomsky connect corporate power to public discourse?" --mode hybrid --working-dir ./lightrag_store
```

Notes:

- LightRAG uses your OpenRouter key from `.env` (`openrouter_key`) for answer generation.
- Available query modes: `naive`, `local`, `global`, `hybrid`, `mix`.
- In production you can set `RAG_WORKING_DIR` to control where the LightRAG index is stored
  (the backend uses `RAG_WORKING_DIR` or defaults to `./lightrag_store`).
- Identical queries are cached by default (24h TTL, configurable via `QUERY_CACHE_TTL`).

## Deploy to Hugging Face Spaces

Use the bundled `Dockerfile` when configuring the Space (`sdk: docker` is already declared in this README header).

- **Repository:** Push this project to the Space or set it as the linked Git repository; the build looks for `Dockerfile` at the root.
- **Secrets:** In the Space settings add `openrouter_key` (and optional `LANGFUSE_*` keys) under *Variables & secrets*; the container refuses to start without an LLM key.
- **Resources:** The default `INGEST_DOC_LIMIT` is 200; override it in *Environment variables* if you need a smaller corpus for faster cold starts.
- **Networking:** The app listens on `$PORT` (default `7860`) and serves both the FastAPI backend and the statically exported Next.js frontend from the same origin.
- **Persistence:** The LightRAG store lives in `/app/lightrag_store`; Spaces reset storage between restarts, so ingestion runs automatically whenever the cache is empty.

After each push Hugging Face rebuilds the image, runs `start.sh`, ingests the corpus if needed, and exposes the UI at the Space URL.

## Secret Scanning (GitGuardian)

This repository ships with a pre-commit hook configuration that runs GitGuardian's `ggshield` scanner on every commit and push.

1. Provision the dedicated security tooling venv (one-time):
   ```bash
   python3 -m venv .tools/ggshield
   .tools/ggshield/bin/python -m pip install --upgrade pip
   .tools/ggshield/bin/python -m pip install pre-commit ggshield
   .tools/ggshield/bin/ggshield auth login  # already completed
   ```
2. Enable the hooks in your local clone:
   ```bash
   .tools/ggshield/bin/pre-commit install --install-hooks
   ```
3. (Optional) Run a full scan at any time:
   ```bash
   .tools/ggshield/bin/ggshield secret scan repo .
   ```

Commits that introduce high-risk secrets will be blocked until the secret is removed or revoked.
