# Atalhos. No Windows sem `make`, rode os comandos `uv run ...` diretamente (ver README).
.PHONY: setup submission validate score test lint

TEAM ?= mtah27

setup:
	git submodule update --init --recursive
	uv sync
	uv run pre-commit install

submission:
	uv run python scripts/make_submission.py

validate:
	uv run python scripts/validate.py outputs/submissions/$(TEAM).csv

score:
	uv run python scripts/score.py

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
