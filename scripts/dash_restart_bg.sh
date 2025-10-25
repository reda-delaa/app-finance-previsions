#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AF_DASH_PORT:-8050}"
LOGDIR="$REPO_ROOT/logs/dash"
mkdir -p "$LOGDIR"
RESTART_LOG="$LOGDIR/restart_${PORT}_$(date '+%Y%m%d_%H%M%S').log"

ATTEMPTS="${AF_DASH_HEALTH_ATTEMPTS:-15}"
SLEEP_SEC="${AF_DASH_HEALTH_SLEEP:-1}"

{
  echo "[dash-restart] $(date '+%F %T') restarting Dash on port $PORT"
  bash "$REPO_ROOT/scripts/dash_stop.sh" || true
  bash "$REPO_ROOT/scripts/dash_start_bg.sh"
  for i in $(seq 1 "$ATTEMPTS"); do
    sleep "$SLEEP_SEC"
    CODE=$(curl -sS -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:${PORT}" 2>/dev/null || echo 000)
    if echo "$CODE" | grep -q '^200$'; then
      echo "[dash-restart] healthy (HTTP 200) on port $PORT"
      exit 0
    fi
    echo "[dash-restart] waiting for Dash (attempt $i, last=$CODE) ..."
  done
  echo "[dash-restart] Dash did not respond with 200 yet; check logs with: tail -n +1 $LOGDIR/dash_${PORT}.log"
} | tee "$RESTART_LOG"

echo "[dash-restart] log saved: $RESTART_LOG"

