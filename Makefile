.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "modebench targets:"
	@echo "  make fmt       Format code with ruff"
	@echo "  make lint      Check code with ruff"
	@echo "  make typecheck Run static type analysis with mypy"
	@echo "  make test      Run unit tests with pytest"
	@echo "  make check     Run lint, typecheck, and test"
	@echo "  make clean     Remove temporary caches and build files"
	@echo "  make smoke     Run a quick dry-run benchmark"

.PHONY: fmt
fmt:
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: lint
lint:
	uv run ruff check .
	uv run ruff format --check .

.PHONY: typecheck
typecheck:
	uv run mypy

.PHONY: test
test:
	uv run pytest

.PHONY: check
check: lint typecheck test

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: smoke
smoke:
	uv run python -m modebench run --profile smoke --dry-run --no-judge
