#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AF_DASH_PORT:-8050}"
LOGDIR="$REPO_ROOT/logs/dash"
PIDFILE="$LOGDIR/dash_${PORT}.pid"

echo "[dash-status] Port: $PORT"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[dash-status] LISTENING on $PORT"
else
  echo "[dash-status] NOT LISTENING on $PORT"
fi

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" || true)
  if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "[dash-status] PID from pidfile: $PID (running)"
  else
    echo "[dash-status] PID file exists but process not found"
  fi
else
  echo "[dash-status] No PID file (expected at $PIDFILE)"
fi

LOGFILE="$LOGDIR/dash_${PORT}.log"
if [ -f "$LOGFILE" ]; then
  echo "[dash-status] Recent log lines:"
  tail -n 10 "$LOGFILE" || true
else
  echo "[dash-status] No log file yet ($LOGFILE)"
fi

