Progress & Roadmap (Investor App)

Recent (UI/Ops)
- UI canonical port 5555; single instance policy with `make ui-start/stop/restart` and watcher `make ui-watch`.
- Top‑nav sticky + footer added; Home/sidebar reorganized (Prévisions vs Administration).
- Safer UI copy: removed "lancez scripts/make" prompts from user pages; guidance moved to Admin/Docs.
- Scoreboard page uses header/footer and CSV export; Observability hides sensitive key names.
- SearXNG local stack added (ops/web/searxng-local) and probe script; web navigator prefers local.
- Security & CI: pip‑audit/safety/bandit/secret‑scan; UI smoke (Playwright MCP) added.

ATLAS feedback captured
- UI/Code directions recorded under docs/atlas/
- Backlog extended with F4F agents EPICs
- Next: implement in‑page dates/empty states across pages

Status (done)
- Data + Freshness
  - Harvester: news/macro/prices/fundamentals (watchlist override via data/watchlist.json)
  - Macro FRED parquet; prices parquet; Data Quality scanner + freshness; Alerts page
- LLM models (text-only, reasoning-first)
  - Dynamic watcher (verified + official sources), working.json with provider/latency/source
  - Local API probing and merge; Scoreboard page (uses, avg_agreement, provider, latency, source)
- Forecasting & Analytics
  - Baseline rules + ML; LLM ensemble + arbiter (Investigations/Topics)
  - Fusion final_score (rule 0.65 + ML 0.25 + LLM 0.10)
  - Risk Monitor; Macro Regime; Investment Memos per ticker
- Investor UI
  - Dashboard (Final Top‑5, regime badge, 90‑day metrics, alerts mini‑summary)
  - Signals (weight sliders, CSV); Portfolio (tilt presets, rebalance simulator, exports)
  - Watchlist manager; Alerts (thresholds + CSV); Settings (tilt & alert thresholds)
  - Changes (regime/risk/top‑N/brief deltas); Notes (journal)
- Automation
  - Makefile: factory-run, refresh/probe, risk, memos, fuse, backfill-prices

Recent additions
- Official models source in watcher (G4F_SOURCE=official|verified|both) and Makefile target
- Settings page; Alerts reads alert thresholds; Portfolio tilt reads presets
- LLM Scoreboard shows provider/latency/source; CSV export
- Backfill script for 5y prices; Data Quality coverage check ≥5 years
- Recession Risk agent + page; Makefile target `recession`
- Earnings Calendar agent and UI page; Makefile target `earnings`
- Agents Status dashboard page (freshness of forecasts, regime, risk, earnings, memos, quality)
- Fix: add `import os` in data_quality to avoid NameError in env var read
- LLM agents runner now writes to `data/forecast/dt=YYYYMMDD/llm_agents.json` (consistent with UI)
- Alerts page: section "Earnings à venir" (fenêtre configurable + export CSV)
- Align `agent_daily.py` outputs to `data/forecast/dt=YYYYMMDD` (so Alerts finds `brief.json`)
- Security posture: prefer official MCP servers; disable non‑official Puppeteer MCP by default; add `ops/security/security_scan.sh` and Makefile `sec-audit` to run pip-audit/safety (+ optional semgrep/trivy)
- MCP UI smoke (best‑effort): `ops/ui/mcp_ui_smoke.mjs` + `make ui-smoke-mcp` to navigate/screenshot via @playwright/mcp
- Network observability (scan‑only):
  - `ops/net/net_observe.py` + `make net-observe` — journalise connexions TCP par processus (JSONL sous artifacts/net)
  - `ops/net/tls_sni_log.sh` + `make net-sni-log` — journalise SNI TLS et IP (tshark requis; pas de blocage)
  - Découplé de l’app (répertoire ops/net) pour éviter toute confusion
- Legacy cleanup
  - Déplacement du shim `searxng-local/finnews.py` et de son test sous `ops/legacy/` (exclus de pytest via `norecursedirs=ops`)
 - CI: `.github/workflows/ci.yml` exécute `make test`, UI smoke et `sec-audit`, publie les artifacts
 - SearXNG local: `ops/web/searxng-local/` + Make (`searx-up`, `searx-down`, `searx-logs`), `SEARXNG_LOCAL_URL` prioritaire dans `web_navigator`

Next (high priority)
1) Macro series expansion: add PMI/ISM, LEI, VIX, commodity baskets; quality coverage checks
2) Beginner mode (tooltips + simplified wording on key pages)
3) Hook earnings events into Alerts (per‑ticker upcoming reminders)

Next (nice to have)
- Beginner mode (tooltips + simplified fields across pages)
- Alerts badge in navbar + count
- “Why” tooltips on Portfolio tilt choices (from Regime/Risk agents)
- Official models auto-fetch with richer parsing when network permits

How to run
- Factory run: `make factory-run`
- Keep models fresh: `make g4f-refresh` or `make g4f-refresh-official`
- Backfill 5y prices: `make backfill-prices`
- UI: `PYTHONPATH=src streamlit run src/apps/agent_app.py`
- Forecast agents: `make equity-forecast` then `make forecast-aggregate`
Planned/Started (agents)
- Equity forecast agent: generates dt=YYYYMMDD/forecasts.parquet with baseline (momentum/vol) for 1w/1m/1y.
- Forecast aggregator: reads latest forecasts.parquet, computes final_score, writes dt=YYYYMMDD/final.parquet.
- Makefile targets: `make equity-forecast`, `make forecast-aggregate`.
