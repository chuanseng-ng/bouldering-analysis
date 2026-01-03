#!/bin/bash
set -e
mypy . && ruff check && ruff format --check && pytest 