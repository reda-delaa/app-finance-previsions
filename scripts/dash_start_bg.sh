#!/usr/bin/env bash
set -euo pipefail

# Start Dash UI in background and log output. Works from any CWD.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AF_DASH_PORT:-8050}"
APP="$REPO_ROOT/src/dash_app/app.py"
LOGDIR="$REPO_ROOT/logs/dash"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/dash_${PORT}.log"
PIDFILE="$LOGDIR/dash_${PORT}.pid"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[dash] Port $PORT already in use. Refusing to start another instance." >&2
  echo "[dash] Hint: use 'make dash-stop' (or bash scripts/dash_stop.sh) to free the port." >&2
  echo "[dash] Current listeners:" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN || true
  exit 1
fi

echo "[dash-bg] Starting Dash on port $PORT (log: $LOGFILE) ..."
(
  echo "==== $(date '+%F %T') — dash start (port $PORT) ===="
  PY_BIN="$REPO_ROOT/.venv/bin/python3"
  if [ -x "$PY_BIN" ]; then
    RUN_PY="$PY_BIN"
  else
    RUN_PY="python3"
  fi
  AF_DASH_DEBUG=${AF_DASH_DEBUG:-false} PYTHONPATH=${PYTHONPATH:-"$REPO_ROOT/src"} "$RUN_PY" "$APP"
) >>"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "[dash-bg] PID $(cat "$PIDFILE")"
echo "[dash-bg] Tail logs: tail -f '$LOGFILE'"
