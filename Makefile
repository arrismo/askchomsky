.PHONY: backend frontend dev

# Start the FastAPI backend (port 8001)
backend:
	.venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8001 --reload

# Start the Next.js frontend (port 3000)
frontend:
	cd frontend && npm run dev

# Start both in parallel (requires two terminal tabs, or use a tool like tmux)
dev:
	@echo "Run in two separate terminals:"
	@echo "  make backend"
	@echo "  make frontend"
