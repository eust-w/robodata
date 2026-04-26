.PHONY: install install-dev test lint typecheck format clean venv

# Create virtual environment
venv:
	python -m venv .venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Install package in editable mode
install:
	pip install -e .

# Install with all optional dependencies for development
install-dev:
	pip install -e ".[all,dev]"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=robodata --cov-report=term-missing --cov-report=html

# Run linter
lint:
	ruff check robodata/ tests/

# Run type checker
typecheck:
	mypy robodata/

# Format code
format:
	ruff format robodata/ tests/
	ruff check --fix robodata/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build package
build:
	pip install build
	python -m build

# Run all checks
check: lint typecheck test
