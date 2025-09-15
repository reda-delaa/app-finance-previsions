#!/usr/bin/env bash
set -e
BASE="http://127.0.0.1:4000"
sample=${1:-120}
timeout=${2:-8}
parallel=${3:-16}
curl -s "$BASE/v1/g4f/scan?sample=$sample&timeout=$timeout&parallel=$parallel" | jq '.ok_count, .ok[].variant'
echo "— working models —"
curl -s "$BASE/v1/working-models" | jq '.ok | length, .ok[].variant'
