Network Observability (scan-only)

Tools here are decoupled from the app to avoid confusion. They do not block traffic; they only log.

Components
- net_observe.py — logs TCP ESTABLISHED connections per process into JSONL under artifacts/net/
- tls_sni_log.sh — logs TLS SNI (server_name) and IP using tshark into artifacts/net/

Usage
- make net-observe
  - env: NET_INTERVAL (seconds, default 5), NET_SAMPLES (0 = continuous), NET_ONLY_PROCS (regex)
- make net-sni-log (requires tshark)
  - env: IFACE (default en0), OUTDIR (default artifacts/net)

Outputs are written under artifacts/net/ by default.

