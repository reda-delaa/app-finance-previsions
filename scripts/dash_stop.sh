#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AF_DASH_PORT:-8050}"
PIDFILE="$REPO_ROOT/logs/dash/dash_${PORT}.pid"

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" || true)
else
  PID=""
fi

if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
  echo "[dash] Stopping PID: $PID"
  kill "$PID" || true
  sleep 1
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PIDFILE" || true
fi

# Fallback by process name
PIDS=$(pgrep -f "src/dash_app/app.py" || true)
if [ -n "$PIDS" ]; then
  echo "[dash] Stopping PIDs: $PIDS"
  kill $PIDS || true
  sleep 1
  kill -9 $PIDS 2>/dev/null || true
fi

# As a last resort, kill any listeners on the port
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  LPIDS=$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN | tr '\n' ' ')
  if [ -n "$LPIDS" ]; then
    echo "[dash] Forcing stop for port $PORT (PIDs: $LPIDS)"
    kill $LPIDS || true
    sleep 1
    kill -9 $LPIDS 2>/dev/null || true
  fi
fi
rm -f "$PIDFILE" 2>/dev/null || true
echo "[dash] Done."
