#!/usr/bin/env bash
set -euo pipefail

# Watcher: ensure the UI listens on AF_UI_PORT; restart if needed.

PORT="${AF_UI_PORT:-5555}"
CHECK_INTERVAL="${AF_UI_WATCH_INTERVAL:-5}"

echo "[ui-watch] Watching port $PORT (interval ${CHECK_INTERVAL}s)"
while true; do
  if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[ui-watch] No listener on $PORT → restarting"
    AF_UI_PORT="$PORT" bash "$(dirname "$0")/ui_start.sh" >/dev/null 2>&1 &
    sleep 3
  fi
  sleep "$CHECK_INTERVAL"
done

