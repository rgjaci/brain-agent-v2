# Brain Agent v2 — Makefile

.PHONY: help install test test-verbose bench lint lint-fix format type-check run run-tui run-chat clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -e '.[dev,ml,server]'

install-dev: ## Install dev dependencies only
	pip install -e '.[dev]'

test: ## Run tests
	pytest tests/ -q

test-verbose: ## Run tests with verbose output
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=core --cov-report=term-missing

bench: ## Run all benchmarks
	python -m benchmarks.recall_test --verbose
	python -m benchmarks.procedure_test --verbose
	python -m benchmarks.reranker_eval --verbose

lint: ## Run linter
	ruff check .

lint-fix: ## Run linter and fix auto-fixable issues
	ruff check --fix .

format: ## Format code
	ruff format .

type-check: ## Run type checker
	pyright core/ tui/ server/

run: ## Run TUI
	python main.py

run-tui: ## Run TUI (alias)
	python main.py

run-chat: ## Run headless chat
	python main.py chat

bootstrap: ## Run bootstrap
	python main.py bootstrap

stats: ## Show memory statistics
	python main.py stats

clean: ## Clean up cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/
