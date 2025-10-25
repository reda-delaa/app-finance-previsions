#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AF_UI_PORT:-5555}"
LOGDIR="$REPO_ROOT/logs/ui"
PIDFILE="$LOGDIR/streamlit_${PORT}.pid"

echo "[ui-status] Port: $PORT"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[ui-status] LISTENING on $PORT"
else
  echo "[ui-status] NOT LISTENING on $PORT"
fi

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" || true)
  if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "[ui-status] PID from pidfile: $PID (running)"
  else
    echo "[ui-status] PID file exists but process not found"
  fi
else
  echo "[ui-status] No PID file (expected at $PIDFILE)"
fi

LOGFILE="$LOGDIR/streamlit_${PORT}.log"
if [ -f "$LOGFILE" ]; then
  echo "[ui-status] Recent log lines:"
  tail -n 10 "$LOGFILE" || true
else
  echo "[ui-status] No log file yet ($LOGFILE)"
fi
