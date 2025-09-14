PYTHON := $(shell which python)

.PHONY: test smoke

test:
	$(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=$$PWD $(PYTHON) scripts/smoke_run.py
