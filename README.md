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
