.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip
export PYTHONPATH := src

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

venv:  ## Create the virtualenv and install the package
	python3 -m venv .venv && $(PIP) install -q --upgrade pip && $(PIP) install -q -e ".[dev]"

test:  ## Run the test suite (no external services required)
	$(PY) -m pytest tests/ -q

lint:  ## Ruff
	$(PY) -m ruff check src tests eval examples

fmt:  ## Ruff autofix
	$(PY) -m ruff check --fix src tests eval examples

demo:  ## Show the recall collapse that motivates the whole project
	$(PY) examples/recall_collapse.py

guide:  ## Rebuild the field guide PDF
	$(PY) scripts/build_guide.py

serve:  ## Run the API locally
	.venv/bin/uvicorn sightline.api:app --reload --port 8000

.PHONY: help venv test lint fmt demo guide serve
