# Prefer python3, fallback to python
PYTHON := $(shell command -v python3 || command -v python)

.PHONY: test smoke it-integration

test:
	$(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=$$PWD $(PYTHON) scripts/smoke_run.py

it-integration:
	AF_ALLOW_INTERNET=1 PYTHONPATH=$$PWD/src $(PYTHON) -m pytest -m integration -q

# --- Convenience targets that auto-activate the venv ---
.PHONY: venv-install test-venv it-integration-venv

venv-install:
	. .venv/bin/activate && python -m pip install -r requirements.txt

test-venv:
	. .venv/bin/activate && PYTHONPATH=$$PWD/src python -m pytest -q

it-integration-venv:
	. .venv/bin/activate && AF_ALLOW_INTERNET=1 PYTHONPATH=$$PWD/src python -m pytest -m integration -q

# --- LLM model watcher and agents ---
.PHONY: g4f-refresh llm-agents harvester-once

G4F_LIMIT ?= 8

g4f-refresh:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.g4f_model_watcher --refresh --limit $(G4F_LIMIT)

.PHONY: g4f-refresh-official
g4f-refresh-official:
	G4F_SOURCE=official PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.g4f_model_watcher --refresh --limit $(G4F_LIMIT)

.PHONY: g4f-fetch-official
g4f-fetch-official:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/fetch_official_models.py

llm-agents:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_llm_agents.py

harvester-once:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.data_harvester --once

.PHONY: g4f-probe-api g4f-merge-probe
g4f-probe-api:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/g4f_probe_api.py --base $${G4F_API_BASE-http://127.0.0.1:8081} --limit $${G4F_PROBE_LIMIT-40} --update-working

g4f-merge-probe:
	PYTHONPATH=$$PWD/src $(PYTHON) - <<-'PY'
	from src.agents.g4f_model_watcher import merge_from_working_txt
	from pathlib import Path
	p = merge_from_working_txt(Path('data/llm/probe/working_results.txt'))
	print('Updated:', p)
	PY

.PHONY: macro-regime fuse-forecasts factory-run

macro-regime:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_macro_regime.py

fuse-forecasts:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/fuse_forecasts.py

factory-run:
	# Sequential, no orchestrator
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.data_harvester --once || true
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.g4f_model_watcher --refresh --limit $${G4F_LIMIT-8} || true
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_llm_agents.py || true
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_macro_regime.py || true
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/fuse_forecasts.py || true

.PHONY: risk-monitor memos

risk-monitor:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_risk_monitor.py

memos:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_memos.py

.PHONY: recession earnings

recession:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_recession.py

earnings:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_earnings.py

.PHONY: backfill-prices
backfill-prices:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/backfill_prices.py

.PHONY: ui-smoke
ui-smoke:
	# Requires: pip install playwright && python -m playwright install chromium
	UI_BASE=$${UI_BASE-http://localhost:5555} PYTHONPATH=$$PWD/src $(PYTHON) ops/ui/ui_smoke.py || true

.PHONY: ui-smoke-mcp
ui-smoke-mcp:
	# Requires: npm i -D @modelcontextprotocol/sdk and @playwright/mcp available via npx
	UI_BASE=$${UI_BASE-http://localhost:5555} node ops/ui/mcp_ui_smoke.mjs || true

.PHONY: sec-audit
sec-audit:
	bash ops/security/security_scan.sh

.PHONY: ui-start ui-stop ui-restart
ui-start:
	AF_UI_PORT=$${AF_UI_PORT-5555} bash scripts/ui_start.sh

ui-stop:
	bash scripts/ui_stop.sh

ui-restart:
	$(MAKE) ui-stop || true
	AF_UI_PORT=$${AF_UI_PORT-5555} bash scripts/ui_start.sh

.PHONY: ui-start-bg ui-restart-bg ui-status ui-logs
ui-start-bg:
	AF_UI_PORT=$${AF_UI_PORT-5555} bash scripts/ui_start_bg.sh

ui-restart-bg:
	AF_UI_PORT=$${AF_UI_PORT-5555} bash scripts/ui_restart_bg.sh

ui-status:
	AF_UI_PORT=$${AF_UI_PORT-5555} bash scripts/ui_status.sh

ui-logs:
	tail -f logs/ui/streamlit_$${AF_UI_PORT-5555}.log

# --- Dash (experimental UI) ---
.PHONY: dash-start
dash-start:
	AF_DASH_PORT=$${AF_DASH_PORT-8050} PYTHONPATH=$$PWD/src $(PYTHON) src/dash_app/app.py

.PHONY: dash-smoke
dash-smoke:
	PYTHONPATH=$$PWD/src $(PYTHON) ops/ui/dash_smoke.py

.PHONY: ui-watch
ui-watch:
	AF_UI_PORT=$${AF_UI_PORT-5555} AF_UI_WATCH_INTERVAL=$${AF_UI_WATCH_INTERVAL-5} bash scripts/ui_watch.sh

.PHONY: net-observe net-sni-log
net-observe:
	PYTHONPATH=$$PWD/src $(PYTHON) ops/net/net_observe.py --interval $${NET_INTERVAL-5} --samples $${NET_SAMPLES-0}

net-sni-log:
	IFACE=$${IFACE-en0} OUTDIR=$${OUTDIR-artifacts/net} bash ops/net/tls_sni_log.sh

.PHONY: searx-probe
searx-probe:
	PYTHONPATH=$$PWD/src $(PYTHON) ops/web/searxng_probe.py --runs $${SEARX_PROBE_RUNS-12} --sleep $${SEARX_PROBE_SLEEP-0.5}

.PHONY: searx-up searx-down searx-logs
searx-up:
	docker compose -f ops/web/searxng-local/docker-compose.yml up -d
	@echo "SearXNG local on http://localhost:8082 (export SEARXNG_LOCAL_URL=http://localhost:8082)"

searx-down:
	docker compose -f ops/web/searxng-local/docker-compose.yml down

searx-logs:
	docker compose -f ops/web/searxng-local/docker-compose.yml logs -f

# --- Forecast agents (no orchestrator; callable via Makefile/cron) ---
.PHONY: equity-forecast forecast-aggregate

equity-forecast:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.equity_forecast_agent

forecast-aggregate:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.forecast_aggregator_agent

# --- Macro & freshness agents ---
.PHONY: macro-forecast update-monitor

macro-forecast:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.macro_forecast_agent

update-monitor:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.update_monitor_agent
