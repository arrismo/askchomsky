# askchomsky

AskChomsky is a retrieval-augmented chatbot over a Noam Chomsky corpus.

## Tools Used

- LlamaIndex (RAG orchestration)
- ChromaDB (local vector database)
- HuggingFace embeddings: `BAAI/bge-base-en-v1.5`
- Model for answer generation: `openai/gpt-oss-120b`
- Langfuse (observability and traces)

## Dataset Used

- Hugging Face dataset: `mmoise00/chomsky-corpus`
- Loaded from the `train` split in the app
- Each record includes:
	- `content`
	- `record_id`
	- `section`
	- `article_title`
	- `article_date`
	- `index_url`

## LightRAG

This repo includes a LightRAG pipeline integrated into `main.py` and `ask.py`.

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