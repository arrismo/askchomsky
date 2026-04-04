#!/bin/bash
set -e

# Respect Space-provided directory + doc limit overrides
DOC_LIMIT=${INGEST_DOC_LIMIT:-200}
PORT=${PORT:-7860}
RAG_WORKING_DIR=${RAG_WORKING_DIR:-/app/lightrag_store}

mkdir -p "$RAG_WORKING_DIR"

# Ingest corpus if RAG store is empty
if [ ! -d "$RAG_WORKING_DIR/graphml" ]; then
    echo "Ingesting corpus..."
    python ask.py --ingest --doc-limit "$DOC_LIMIT" --working-dir "$RAG_WORKING_DIR"
    echo "Ingestion complete."
else
    echo "RAG store found, skipping ingestion."
fi

# Start FastAPI on configured port (serves both API + static frontend)
exec uvicorn backend.api:app --host 0.0.0.0 --port "$PORT"
