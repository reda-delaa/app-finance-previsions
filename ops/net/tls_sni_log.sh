#!/usr/bin/env bash
set -euo pipefail

# Log TLS SNI (server_name) and destination IP using tshark if available.
# Passive, read-only. Does not block traffic.

IFACE=${IFACE:-en0}
OUTDIR=${OUTDIR:-artifacts/net}
mkdir -p "$OUTDIR"
OUT="$OUTDIR/tls_sni_$(date +%Y%m%d_%H%M%S).log"

if ! command -v tshark >/dev/null 2>&1; then
  echo "tshark not installed. Install via: brew install wireshark (then ensure tshark in PATH)" >&2
  exit 1
fi

echo "Writing to $OUT (Ctrl+C to stop)" >&2

exec tshark -i "$IFACE" -l \
  -Y 'tls.handshake.extensions_server_name' \
  -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tls.handshake.extensions_server_name \
  2>/dev/null | tee "$OUT"

