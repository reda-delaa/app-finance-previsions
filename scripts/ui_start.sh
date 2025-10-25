#!/usr/bin/env bash
set -euo pipefail

# Canonical Streamlit launcher to avoid multiple instances on random ports.
# Usage: AF_UI_PORT=8501 ./scripts/ui_start.sh

PORT="${AF_UI_PORT:-5555}"
APP="src/apps/agent_app.py"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[ui] Port $PORT already in use. Refusing to start another instance." >&2
  echo "[ui] Tip: stop existing UI with 'make ui-stop' or choose a different AF_UI_PORT=<port>." >&2
  exit 1
fi

echo "[ui] Starting Streamlit on port $PORT ..."
PYTHONPATH=${PYTHONPATH:-src} streamlit run "$APP" --server.port "$PORT" --server.headless false
