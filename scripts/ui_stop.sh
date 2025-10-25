#!/usr/bin/env bash
set -euo pipefail

# Stop Streamlit instances of this repo (best-effort, non-destructive).

PIDS=$(pgrep -f "streamlit run .*src/apps/agent_app.py" || true)
if [ -z "$PIDS" ]; then
  echo "[ui] No Streamlit instance found for src/apps/agent_app.py"
  exit 0
fi

echo "[ui] Stopping PIDs: $PIDS"
kill $PIDS || true
sleep 1
kill -9 $PIDS 2>/dev/null || true
echo "[ui] Done."

