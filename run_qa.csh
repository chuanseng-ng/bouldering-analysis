#!/bin/bash

# Track if any errors occurred
errors=0

echo "Running mypy..."
mypy .
if [ $? -ne 0 ]; then
    errors=1
fi

echo "Running ruff check & format..."
ruff check && ruff format --check
if [ $? -ne 0 ]; then
    errors=1
fi

echo "Running pytest..."
pytest_output=$(pytest tests/ --cov-report=term-missing --cov=src/ 2>&1)
pytest_exit_code=$?
echo "$pytest_output"

# Check pytest exit code (tests passed/failed)
if [ $pytest_exit_code -ne 0 ]; then
    errors=1
fi

# Extract the coverage percentage from pytest output
# Format: "TOTAL                                      XXX      X    XX%"
coverage=$(echo "$pytest_output" | grep "^TOTAL" | awk '{print $NF}' | sed 's/%//')

if [ -n "$coverage" ]; then
    echo "Coverage: $coverage%"
    # Check if coverage is less than 99 using awk for float comparison
    if awk "BEGIN {exit !($coverage < 99)}"; then
        echo "ERROR: Coverage $coverage% is below the required threshold of 99%"
        errors=1
    else
        echo "Coverage check passed (>= 99%)"
    fi
else
    echo "WARNING: Could not extract coverage percentage from pytest output"
fi

echo "Running pylint..."
pylint_output=$(pylint src/ tests/ 2>&1)
pylint_exit_code=$?
echo "$pylint_output"

# Extract the score from pylint output
# Format: "Your code has been rated at X.XX/10"
score=$(echo "$pylint_output" | grep "rated at" | sed -n 's/.*rated at \([0-9.]*\)\/10.*/\1/p')

if [ -n "$score" ]; then
    echo "Pylint score: $score/10"
    # Check if score is less than 9.9 using awk for float comparison
    if awk "BEGIN {exit !($score < 9.9)}"; then
        echo "ERROR: Pylint score $score is below the required threshold of 9.9/10"
        errors=1
    else
        echo "Pylint score check passed (>= 9.9/10)"
    fi
else
    echo "WARNING: Could not extract pylint score from output"
fi

# Also check if pylint had errors (exit code)
if [ $pylint_exit_code -ne 0 ]; then
    errors=1
fi

# Exit with appropriate code
if [ $errors -eq 0 ]; then
    echo "All QA checks passed!"
    exit 0
else
    echo "Some QA checks failed!"
    exit 1
fi