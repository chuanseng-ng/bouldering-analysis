#!/bin/bash
set -e
echo "Running mypy..."
mypy .
echo "Running ruff check & format..."
ruff check && ruff format --check
echo "Running pytest..."
pytest --cov=src tests/ 
echo "Running pylint..."
pylint src/ tests/
echo "All QA checks passed!"