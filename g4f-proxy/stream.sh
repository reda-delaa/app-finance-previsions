#!/usr/bin/env bash
set -e
m=${1:-command-r}
content=${2:-"Dis bonjour et une astuce Python."}
curl -N http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d "$(jq -nc --arg m "$m" --arg c "$content" '{model:$m,stream:true,messages:[{role:"user",content:$c}]}')"
