# askchomsky

AskChomsky is a retrieval-augmented chatbot over a Noam Chomsky corpus.

## Run Locally

### Prerequisites

- Python 3.14+
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
- HuggingFace embeddings: `BAAI/bge-base-en-v1.5`
- Model for answer generation: `openai/gpt-oss-120b`
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
- This setup uses local embeddings with `BAAI/bge-base-en-v1.5`.
- Available query modes: `naive`, `local`, `global`, `hybrid`, `mix`.
