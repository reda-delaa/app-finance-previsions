PYTHON := $(shell which python)

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

llm-agents:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/run_llm_agents.py

harvester-once:
	PYTHONPATH=$$PWD/src $(PYTHON) -m src.agents.data_harvester --once

.PHONY: g4f-probe-api g4f-merge-probe
g4f-probe-api:
	PYTHONPATH=$$PWD/src $(PYTHON) scripts/g4f_probe_api.py --base $${G4F_API_BASE-http://127.0.0.1:8081} --limit $${G4F_PROBE_LIMIT-40} --update-working

g4f-merge-probe:
	PYTHONPATH=$$PWD/src $(PYTHON) - <<'PY'
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
