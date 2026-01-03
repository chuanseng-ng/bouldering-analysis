#!/bin/bash
set -e
mypy . && pylint src/ tests/ && ruff check && ruff format --check && pytest 