#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! docker compose ps >/dev/null 2>&1; then
  docker compose up -d
fi

# Ping le front; si KO, (re)start
if ! curl -sf "http://127.0.0.1:8888/healthz" >/dev/null 2>&1; then
  docker compose up -d --force-recreate
fi