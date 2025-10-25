#!/usr/bin/env bash
set -euo pipefail

# Stop current Streamlit (if any) and start in background with logging.

PORT="${AF_UI_PORT:-5555}"
LOGDIR="logs/ui"
mkdir -p "$LOGDIR"
RESTART_LOG="$LOGDIR/restart_${PORT}_$(date '+%Y%m%d_%H%M%S').log"

{
  echo "[ui-restart] $(date '+%F %T') restarting UI on port $PORT"
  bash "$(dirname "$0")/ui_stop.sh" || true
  bash "$(dirname "$0")/ui_start_bg.sh"
  # health probe (best-effort)
  for i in 1 2 3 4 5; do
    sleep 1
    if curl -sS -o /dev/null -w '%{http_code}\n' "http://localhost:${PORT}" 2>/dev/null | grep -q '^200$'; then
      echo "[ui-restart] healthy (HTTP 200) on port $PORT"
      exit 0
    fi
    echo "[ui-restart] waiting for UI (attempt $i) ..."
  done
  echo "[ui-restart] UI did not respond with 200 yet; check logs with: tail -n +1 logs/ui/streamlit_${PORT}.log"
} | tee "$RESTART_LOG"

echo "[ui-restart] log saved: $RESTART_LOG"

