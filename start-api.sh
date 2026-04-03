#!/usr/bin/env bash
# Run the FastAPI backend on http://127.0.0.1:8000 (what Vite proxies to).
# Use this in one terminal, then in another: cd frontend && npm run dev
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/venv/bin/activate"
cd "$ROOT/backend"
exec uvicorn main:app --host 127.0.0.1 --port 8000 --reload
