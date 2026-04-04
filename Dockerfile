FROM python:3.11-slim

ENV NODE_MAJOR=20 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

# System dependencies (Node.js for building frontend + git for HF datasets)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git && \
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (copy metadata first for caching)
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
RUN pip install --upgrade pip && \
    pip install -e .

# Frontend dependencies
COPY frontend/package.json ./frontend/
RUN cd frontend && npm install

# Project sources
COPY frontend/ ./frontend/
COPY main.py ask.py start.sh ./
RUN chmod +x start.sh

# Build static Next.js export
RUN cd frontend && npm run build

# Prepare runtime directories
RUN mkdir -p /app/lightrag_store

ENV PYTHONPATH=/app \
    RAG_WORKING_DIR=/app/lightrag_store \
    PORT=7860

EXPOSE 7860

CMD ["./start.sh"]
