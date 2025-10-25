#!/usr/bin/env bash
set -euo pipefail

# Start Streamlit UI in background and log output to logs/ui/streamlit_${PORT}.log

PORT="${AF_UI_PORT:-5555}"
APP="${AF_UI_APP:-src/apps/agent_app.py}"
LOGDIR="logs/ui"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/streamlit_${PORT}.log"
PIDFILE="$LOGDIR/streamlit_${PORT}.pid"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[ui] Port $PORT already in use. Refusing to start another instance." >&2
  exit 1
fi

echo "[ui-bg] Starting Streamlit on port $PORT (log: $LOGFILE) ..."
(
  echo "==== $(date '+%F %T') — streamlit start (port $PORT) ===="
  PYTHONPATH=${PYTHONPATH:-src} streamlit run "$APP" --server.port "$PORT" --server.headless false
) >>"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "[ui-bg] PID $(cat "$PIDFILE")"
echo "[ui-bg] Tail logs: tail -f '$LOGFILE'"

