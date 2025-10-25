Progress & Roadmap (Investor App)

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

Next (high priority)
1) Recession Risk agent/page: add explicit probability + drivers + plain-language summary
2) Macro series expansion: add PMI/ISM, LEI, VIX, commodity baskets; quality coverage checks
3) Earnings calendar for watchlist (with per-ticker reminders in Alerts)

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

