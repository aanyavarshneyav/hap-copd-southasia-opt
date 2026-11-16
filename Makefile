.PHONY: help setup lint format test check validate-data stage1 stage2 stage3 planning all clean

PYTHON := python3.12
VENV := .venv
BIN := $(VENV)/bin
CONFIG := configs/default.yaml

help:
	@echo "Available targets:"
	@echo "  make setup          Create Python 3.12 venv and install project + dev deps"
	@echo "  make lint           Run ruff linter and format check"
	@echo "  make format         Auto-format src/ and tests/ with ruff"
	@echo "  make test           Run pytest suite"
	@echo "  make check          Run lint + test"
	@echo "  make validate-data  [Phase 1+] Validate raw datasets against DATA_SPEC.md"
	@echo "  make stage1         [Phase 1+] Run Stage 1 historical analysis"
	@echo "  make stage2         [Phase 2+] Run Stage 2 forecasting & CV"
	@echo "  make stage3         [Phase 3+] Run Stage 3 NSGA-II optimisation"
	@echo "  make planning       [Phase 3+] Run MDP, HTN, and PDDL (pyperplan) pipeline"
	@echo "  make all            [Phase 3+] Run full reproducible pipeline"
	@echo "  make clean          Remove caches and temporary build artifacts"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff format src tests

test:
	$(BIN)/pytest

check: lint test

validate-data:
	$(BIN)/python -m src.data.validate --config $(CONFIG)

stage1:
	$(BIN)/python -m src.historical.run --config $(CONFIG)

stage2:
	$(BIN)/python -m src.forecast.run --config $(CONFIG)

stage3:
	$(BIN)/python -m src.optimise.run --config $(CONFIG)

planning:
	$(BIN)/python -m src.planning.run --config $(CONFIG)

all: validate-data stage1 stage2 stage3 planning

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
