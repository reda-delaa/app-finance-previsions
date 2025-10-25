#!/usr/bin/env bash
set -euo pipefail

# Stop current Streamlit (if any) and start in background with logging.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${AF_UI_PORT:-5555}"
LOGDIR="$REPO_ROOT/logs/ui"
mkdir -p "$LOGDIR"
RESTART_LOG="$LOGDIR/restart_${PORT}_$(date '+%Y%m%d_%H%M%S').log"

ATTEMPTS="${AF_UI_HEALTH_ATTEMPTS:-15}"
SLEEP_SEC="${AF_UI_HEALTH_SLEEP:-1}"

{
  echo "[ui-restart] $(date '+%F %T') restarting UI on port $PORT"
  bash "$REPO_ROOT/scripts/ui_stop.sh" || true
  bash "$REPO_ROOT/scripts/ui_start_bg.sh"
  # health probe (best-effort)
  for i in $(seq 1 "$ATTEMPTS"); do
    sleep "$SLEEP_SEC"
    CODE=$(curl -sS -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:${PORT}" 2>/dev/null || echo 000)
    if echo "$CODE" | grep -q '^200$'; then
      echo "[ui-restart] healthy (HTTP 200) on port $PORT"
      exit 0
    fi
    echo "[ui-restart] waiting for UI (attempt $i, last=$CODE) ..."
  done
  echo "[ui-restart] UI did not respond with 200 yet; check logs with: tail -n +1 $LOGDIR/streamlit_${PORT}.log"
} | tee "$RESTART_LOG"

echo "[ui-restart] log saved: $RESTART_LOG"
