set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/venv"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 22 2>/dev/null || true

echo "[run] Starting backend on http://localhost:8000 ..."
source "$VENV/bin/activate"
cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$ROOT"

echo "[run] Starting frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$ROOT"

trap "echo; echo '[run] Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo "[run] Both services running. Press Ctrl-C to stop."
wait
