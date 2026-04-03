#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/venv"

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[run] Starting backend on http://localhost:8000 ..."
source "$VENV/bin/activate"
cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$ROOT"

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[run] Starting frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$ROOT"

# ── Cleanup on Ctrl-C ─────────────────────────────────────────────────────────
trap "echo; echo '[run] Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo "[run] Both services running. Press Ctrl-C to stop."
wait
