# SearXNG Local (for reliable finance/news search)

This directory provides a minimal, local SearXNG instance (Docker) tuned for JSON API responses. Use it when public instances are unreliable or rate‑limited.

## Prerequisites
- Docker (and Docker Compose v2)

## Start / Stop / Logs
- Start: `make searx-up`
  - Exposes `http://localhost:8082/`
  - Healthcheck: `/health`
- Stop: `make searx-down`
- Logs: `make searx-logs`

## App integration
- Prioritize the local instance by exporting:
  - `export SEARXNG_LOCAL_URL=http://localhost:8082`
- The navigator (`src/research/web_navigator.py`) will prefer the local URL, then fall back to public instances and (optionally) Serper/Tavily.

## Probe reliability
- Run a quick probe against public/local instances:
  - `make searx-probe` (defaults: `--runs 12 --sleep 0.5`)
- Report written to `data/reports/dt=YYYYMMDD/searxng_probe.json`.

## Settings
- `docker-compose.yml`: uses `searxng/searxng:latest`, binds port 8082→8080 inside container.
- `settings.yml`: minimal JSON/GET friendly config; engines include DuckDuckGo, Google, Bing.

## Notes
- Keep this “ops/” directory decoupled from the application code. Nothing here is required at runtime unless you explicitly enable it.
- If you don’t export `SEARXNG_LOCAL_URL`, the app will use public instances (less reliable) and optional Serper/Tavily (if keys are set).
