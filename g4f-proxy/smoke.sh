#!/usr/bin/env bash
set -e
BASE="http://127.0.0.1:4000"
OK=$(curl -s "$BASE/v1/working-models" | jq -r '.ok[].variant')
for m in $OK; do
  printf "\n=== %s ===\n" "$m"
  jq -nc --arg m "$m" --arg c "ping" \
    '{model:$m, messages:[{role:"user",content:$c}]}' \
  | curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- \
  | jq -r '.choices[0].message.content|.[:200]'
done
