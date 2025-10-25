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
