# Sightline — every command anybody needs, and nothing else.
#
# Two rules this file follows:
#
#   1. If CI runs it, it is a target here. A build step that only exists inside
#      a YAML file is a build step nobody can reproduce locally, and "works on
#      my machine" then means "works in the runner".
#   2. Nothing here needs a service. `make test` runs against the in-memory
#      tuple store and the numpy vector store, with no Docker, no network and no
#      model download, because a test suite with a setup step is a test suite
#      that gets skipped.
#
# The heavy backends (Qdrant, Postgres, ONNX, FAISS) are optional extras with
# their own targets. `make install` deliberately does not pull them.

SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV ?= .venv
BIN := $(VENV)/bin
PY ?= python3
PYTHON := $(BIN)/python
PYTEST := $(BIN)/pytest

# src/ on the path so every target works in a fresh clone before `pip install`.
export PYTHONPATH := src

# Docs say the reference machine is a 2015 dual-core laptop. Default the eval
# corpus down accordingly; CI overrides these on the command line.
DOCS ?= 4000
QUERIES ?= 30
SEED ?= 20240914
K ?= 10
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help install install-all test test-fast cov lint typecheck check ingest \
        eval selectivity oracle mutation attack headline readme-check serve \
        demo docker-build docker-up docker-down clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-14s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: $(BIN)/python ## Create .venv and install the package plus dev tools
	$(BIN)/pip install -e ".[dev]"

install-all: install ## Also install every optional backend (needs network, ~1 GB)
	$(BIN)/pip install -e ".[dev,qdrant,postgres,embed,oracle,obs]"

# --------------------------------------------------------------------------
# The inner loop
# --------------------------------------------------------------------------

test: ## Run the test suite (no services, no network)
	$(PYTEST) tests/ -q

test-fast: ## Same, minus the slow property sweeps
	$(PYTEST) tests/ -q -k "not hypothesis and not collapses"

cov: ## Test with coverage, failing under the floor CI enforces
	$(PYTEST) tests/ -q --cov=sightline --cov-report=term-missing --cov-report=xml \
		--cov-fail-under=80

lint: ## ruff
	$(BIN)/ruff check src tests eval examples scripts

typecheck: ## mypy over the package and the harness
	$(BIN)/mypy src/sightline eval

check: lint typecheck test ## What CI runs on every push, in one command

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

ingest: ## Ingest the sample corpus into the local index
	$(PYTHON) -m sightline.ingest.pipeline

serve: ## Run the API (override HOST/PORT; default 127.0.0.1:8000)
	$(BIN)/uvicorn sightline.api:app --host $(HOST) --port $(PORT) --reload

demo: ## The recall-collapse example, in one file
	$(PYTHON) examples/recall_collapse.py

# --------------------------------------------------------------------------
# Evaluation
#
# Nothing in here is allowed to publish a number it did not just compute. That
# is the rule `readme-check` enforces and the reason these targets exist
# separately from `test`: a measurement is not a test, and a test is not a
# measurement.
# --------------------------------------------------------------------------

eval: oracle selectivity ## The gate plus the headline chart

selectivity: ## Recall and cost versus permission density
	$(PYTHON) -m eval.selectivity --docs $(DOCS) --queries $(QUERIES) --seed $(SEED) --k $(K)

oracle: ## BLOCKING GATE: the compiled plan must never admit what check() denies
	@$(PYTHON) -c "$$ORACLE_SCRIPT"

mutation: ## Apply the fifteen planted permission bugs; report the kill rate
	$(PYTHON) -m eval.mutation run --json

attack: ## The leak suite (cross-principal, existence probing, injection)
	$(PYTHON) -m eval.attack

headline: ## Recompute every published number and write it into README.md
	$(PYTHON) -m eval.report --write --docs $(DOCS) --queries $(QUERIES) --seed $(SEED)

readme-check: ## Fail if a committed number no longer matches a fresh run
	$(PYTHON) -m eval.report --check --docs $(DOCS) --queries $(QUERIES) --seed $(SEED)

# The oracle has no CLI of its own yet, so the gate lives here rather than in a
# YAML step, per rule 1 at the top of this file. Move it into eval/ the day it
# needs a flag.
define ORACLE_SCRIPT
import sys
import eval  # noqa: F401 - puts src/ on the path
from eval.selectivity import CorpusSpec, build_corpus
from sightline.authz.oracle import run_oracle

corpus = build_corpus(CorpusSpec(n_docs=$(DOCS), n_queries=4, seed=$(SEED)))
result = run_oracle(corpus.tuple_store, n_pairs=5000, seed=$(SEED))
print(result.summary())
if result.breached:
    for pair in result.breaches[:10]:
        print("  BREACH:", pair.to_dict(), file=sys.stderr)
    print("false_allow must be 0. It is not. This is a breach, not a regression.",
          file=sys.stderr)
    sys.exit(1)
sys.exit(0 if result.passed else 2)
endef
export ORACLE_SCRIPT

# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------

docker-build: ## Build the API image
	docker build -t sightline:dev .

docker-up: ## api + qdrant + postgres, for local development only
	docker compose up --build

docker-down: ## Stop and remove the stack, keeping volumes
	docker compose down

clean: ## Remove caches and build artefacts (never data/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
