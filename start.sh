#!/bin/bash
set -e

# Respect Space-provided directory
PORT=${PORT:-7860}
RAG_WORKING_DIR=${RAG_WORKING_DIR:-/app/lightrag_store}

mkdir -p "$RAG_WORKING_DIR"

# Ingest corpus if RAG store is empty
if [ ! -f "$RAG_WORKING_DIR/graph_chunk_entity_relation.graphml" ]; then
    echo "Ingesting corpus..."
    python ask.py --ingest --working-dir "$RAG_WORKING_DIR"
    echo "Ingestion complete."
else
    echo "RAG store found, skipping ingestion."
fi

# Start FastAPI on configured port (serves both API + static frontend)
exec uvicorn backend.api:app --host 0.0.0.0 --port "$PORT"
