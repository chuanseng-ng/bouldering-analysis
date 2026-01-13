# CLAUDE.md - AI Assistant Guide for Bouldering Route Analysis

**Version**: 2026.01.14
**Last Updated**: 2026-01-14
**Architecture**: FastAPI + Supabase (Migration in Progress)
**Repository**: bouldering-analysis
**Purpose**: Guide AI assistants working with this computer vision-based bouldering route grading application

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Status](#architecture-status)
3. [Codebase Structure](#codebase-structure)
4. [Technology Stack](#technology-stack)
5. [Development Workflows](#development-workflows)
6. [Code Conventions](#code-conventions)
7. [Testing Requirements](#testing-requirements)
8. [Configuration Management](#configuration-management)
9. [Quality Standards](#quality-standards)
10. [API Reference](#api-reference)
11. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

### What This Application Does

A web-based system that estimates bouldering route difficulty (V-scale) from images by:

- Detecting and classifying climbing holds using pre-trained perception models
- Constructing movement graphs from hold positions
- Extracting interpretable features for grade estimation
- Providing explainable grade predictions with uncertainty

### Core Design Principles

1. **Backend-first architecture** - FastAPI with Supabase
2. **Route difficulty ≠ pixels** - Use structured features, not end-to-end CNN
3. **Perception models are pre-trained** - Frozen for inference
4. **Grades are ordinal and uncertain** - Probabilistic predictions
5. **One PR ≈ one function** - Modular, testable changes
6. **Every ML output must be explainable** - No black boxes

### Key Technologies

- **Backend**: FastAPI 0.115.6 with Pydantic Settings
- **ML/CV**: PyTorch 2.9.1 + Ultralytics YOLOv8 8.3.233
- **Database**: Supabase (Postgres + Storage) - planned
- **Testing**: pytest 9.0.1 with 85% coverage (current), 90% (final)
- **Quality**: mypy, ruff, pylint (8.5/10 current, 9.0/10 final)

---

## Architecture Status

### Migration Progress

The codebase is being migrated from Flask to FastAPI + Supabase.

| Milestone | Status | PR |
|-----------|--------|-----|
| 1. Backend Foundation | **In Progress** | |
| ├─ FastAPI Bootstrap | Completed | PR-1.1 |
| └─ Supabase Client | Pending | PR-1.2 |
| 2. Image Upload | Pending | PR-2.x |
| 3. Hold Detection | Pending | PR-3.x |
| 4. Hold Classification | Pending | PR-4.x |
| 5. Route Graph | Pending | PR-5.x |
| 6. Feature Extraction | Pending | PR-6.x |
| 7. Grade Estimation | Pending | PR-7.x |
| 8. Explainability | Pending | PR-8.x |
| 9. Database Schema | Pending | PR-9.x |
| 10. Frontend Integration | Pending | PR-10.x |

### Archived Code

Legacy Flask-based code is preserved in:
- `src/archive/legacy/` - Original source files
- `tests/archive/legacy/` - Original test files

**Do not import from archive directories** - they are for reference only.

---

## Codebase Structure

### Current Directory Layout

```text
bouldering-analysis/
├── src/                          # Main application code
│   ├── __init__.py               # Package with create_app export
│   ├── app.py                    # FastAPI application factory
│   ├── config.py                 # Pydantic Settings configuration
│   ├── logging_config.py         # Structured JSON logging
│   ├── routes/                   # API route modules
│   │   ├── __init__.py           # Route exports
│   │   └── health.py             # Health check endpoint
│   └── archive/legacy/           # Archived Flask code (reference only)
│
├── tests/                        # Test suite
│   ├── __init__.py               # Test package
│   ├── conftest.py               # Pytest fixtures
│   ├── test_app.py               # Application tests
│   ├── test_config.py            # Configuration tests
│   ├── test_health.py            # Health endpoint tests
│   ├── test_logging_config.py    # Logging tests
│   └── archive/legacy/           # Archived tests (reference only)
│
├── docs/                         # Documentation
│   ├── DESIGN.md                 # Fundamental design specification
│   └── MODEL_PRETRAIN.md         # Model pretraining specification
│
├── plans/                        # Implementation plans
│   ├── MIGRATION_PLAN.md         # Overall migration roadmap
│   ├── specs/                    # PR specifications
│   │   └── PR-1.1-fastapi-bootstrap.md
│   └── archive/                  # Archived planning docs
│
├── data/                         # Data directory
│   └── hold_classification/      # Hold classification dataset
│
├── models/                       # Trained model weights (gitignored)
│   └── hold_detection/           # Detection model versions
│
├── requirements.txt              # Pip dependencies
├── pyproject.toml                # Pytest and coverage config
├── mypy.ini                      # Type checking config
├── .pylintrc                     # Pylint config
└── .flake8                       # Flake8 config
```

### Critical Files

| File | Purpose | Key Info |
|------|---------|----------|
| `src/app.py` | FastAPI application factory | Entry point, middleware, routes |
| `src/config.py` | Configuration management | Pydantic Settings with env vars |
| `src/logging_config.py` | Structured logging | JSON format for production |
| `src/routes/health.py` | Health check endpoint | `/health`, `/api/v1/health` |
| `tests/conftest.py` | Pytest fixtures | Test app, client, settings |
| `docs/DESIGN.md` | Architecture spec | Milestones, domain model |
| `docs/MODEL_PRETRAIN.md` | ML spec | Detection, classification |
| `plans/MIGRATION_PLAN.md` | Migration roadmap | PR breakdown, phases |

---

## Technology Stack

### Core Dependencies

```text
# Web Framework
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic-settings==2.7.1

# Async Support
httpx==0.28.1
python-multipart==0.0.20

# Structured Logging
python-json-logger==3.2.1

# Machine Learning & Computer Vision
torch==2.9.1
torchvision==0.24.1
ultralytics==8.3.233
Pillow==10.4.0
opencv-python==4.12.0.88

# Data Processing
numpy==2.2.6
PyYAML==6.0.2

# Testing
pytest==9.0.1
pytest-cov==7.0.0
pytest-asyncio==0.25.2

# Quality
pylint==4.0.3
ruff==0.14.7
mypy==1.18.2
```

### Python Version

Tested on: **Python 3.10, 3.11, 3.12, 3.13**

---

## Development Workflows

### Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.app import create_app; print('OK')"
```

### Running the Application

```bash
# Start FastAPI development server
uvicorn src.app:application --reload

# Or using factory pattern
uvicorn src.app:create_app --factory --reload

# The app will be available at http://localhost:8000
# API docs at http://localhost:8000/docs (debug mode only)
```

### Quality Assurance Workflow

**CRITICAL**: All code changes must pass QA before merging.

```bash
# Run all QA checks
mypy src/ tests/ --ignore-missing-imports && \
ruff check src/ tests/ --ignore E501 && \
ruff format --check src/ tests/ && \
pytest tests/ --cov=src --cov-fail-under=85 && \
pylint src/ --ignore=archive
```

#### Individual QA Commands

```bash
# Type checking
mypy src/ tests/ --ignore-missing-imports

# Linting
ruff check src/ tests/ --ignore E501

# Format checking
ruff format --check src/ tests/

# Tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Code quality (minimum 8.5/10 current, 9.0/10 final)
pylint src/ --ignore=archive
```

---

## Code Conventions

### Naming Standards

- **Modules**: snake_case (`config.py`, `logging_config.py`)
- **Classes**: PascalCase (`Settings`, `HealthResponse`)
- **Functions**: snake_case (`create_app()`, `get_settings()`)
- **Constants**: UPPER_SNAKE_CASE (if any)
- **Private/Internal**: Leading underscore (`_configure_middleware()`)

### Documentation Standards

**All modules, classes, and functions must have Google-style docstrings:**

```python
def create_app(config_override: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    This factory function creates a new FastAPI instance with all
    middleware, routes, and configuration applied.

    Args:
        config_override: Optional dictionary to override default
            configuration values. Used primarily for testing.

    Returns:
        Configured FastAPI application instance ready to serve requests.

    Example:
        >>> app = create_app()
        >>> test_app = create_app({"testing": True})
    """
```

### Type Annotations

**Full type hints required on all functions:**

```python
from typing import Any

def get_settings() -> Settings:
    """Get cached application settings."""
    ...

def create_app(config_override: dict[str, Any] | None = None) -> FastAPI:
    """Create FastAPI application."""
    ...
```

### Import Organization

```python
# 1. Standard library
import json
from functools import lru_cache
from typing import Any

# 2. Third-party
from fastapi import FastAPI, Request
from pydantic import BaseModel

# 3. Local/project
from src.config import get_settings
from src.routes import health_router
```

### Code Style

- **Formatter**: ruff format (Black-compatible)
- **Line Length**: Long lines allowed (E501 ignored) but keep reasonable
- **Quotes**: Double quotes preferred
- **Indentation**: 4 spaces
- **Trailing Commas**: Required in multi-line collections

---

## Testing Requirements

### Testing Philosophy

- **Coverage Target**: 85% minimum (current stage), 90% when all features complete
- **Framework**: pytest with comprehensive fixtures
- **Isolation**: Each test uses fresh app instance
- **Current Coverage**: 100%

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_app.py -v

# Run with coverage threshold enforcement
pytest tests/ --cov=src --cov-fail-under=85
```

### Key Test Fixtures (from conftest.py)

```python
@pytest.fixture
def test_settings() -> dict[str, Any]:
    """Test-specific settings overrides."""
    return {"testing": True, "debug": True}

@pytest.fixture
def app(test_settings: dict[str, Any]) -> FastAPI:
    """Create test application instance."""
    return create_app(test_settings)

@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def app_settings(app: FastAPI) -> Settings:
    """Get settings from the test application."""
    return app.state.settings
```

### Test Structure

```python
class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_endpoint_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_response_schema(self, client: TestClient) -> None:
        """Health response should contain required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
```

---

## Configuration Management

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BA_APP_NAME` | `bouldering-analysis` | Application name |
| `BA_APP_VERSION` | `0.1.0` | Application version |
| `BA_DEBUG` | `false` | Enable debug mode |
| `BA_TESTING` | `false` | Enable testing mode |
| `BA_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array) |
| `BA_LOG_LEVEL` | `INFO` | Logging level |

### Example .env File

```bash
BA_APP_NAME=bouldering-analysis
BA_APP_VERSION=0.1.0
BA_DEBUG=true
BA_LOG_LEVEL=DEBUG
BA_CORS_ORIGINS=["http://localhost:3000"]
```

### Accessing Configuration

```python
from src.config import get_settings, get_settings_override

# Get cached settings
settings = get_settings()
print(settings.app_name)

# Override for testing
test_settings = get_settings_override({"testing": True})
```

---

## Quality Standards

### Quality Gates

| Check | Tool | Current | Final |
|-------|------|---------|-------|
| Type Safety | mypy | No errors | No errors |
| Linting | ruff check | No errors | No errors |
| Formatting | ruff format | All files formatted | All files formatted |
| Coverage | pytest-cov | 85% minimum | 90% minimum |
| Code Quality | pylint | 8.5/10 minimum | 9.0/10 minimum |

### Pre-Commit Checklist

- [ ] All tests pass: `pytest tests/`
- [ ] Coverage ≥ 85%: `pytest tests/ --cov=src/`
- [ ] Type checking passes: `mypy src/ tests/`
- [ ] Linting passes: `ruff check .`
- [ ] Formatting passes: `ruff format --check .`
- [ ] Pylint score ≥ 8.5: `pylint src/`

---

## API Reference

### Current Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/health` | Health check (root level) | HealthResponse |
| GET | `/api/v1/health` | Health check (versioned) | HealthResponse |
| GET | `/docs` | Swagger UI (debug only) | HTML |
| GET | `/openapi.json` | OpenAPI schema | JSON |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Response
{
    "status": "healthy",
    "version": "0.1.0",
    "timestamp": "2026-01-14T12:00:00Z"
}
```

### Response Models

```python
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
```

---

## AI Assistant Guidelines

### When Making Changes

1. **Read before modifying**: Always read files before editing
2. **Check specifications**: Review docs/DESIGN.md and plans/MIGRATION_PLAN.md
3. **Follow conventions**: Match existing code style and patterns
4. **Write tests first**: Add tests for new functionality
5. **Maintain coverage**: Ensure 85%+ coverage after changes (90% final)
6. **Run QA suite**: Execute all quality checks before committing
7. **Update documentation**: Keep CLAUDE.md and specs current

### Code Review Checklist

- [ ] Type hints on all functions
- [ ] Google-style docstrings on all public functions
- [ ] Error handling for edge cases
- [ ] Tests added for new functionality
- [ ] Coverage remains ≥ 85% (90% final)
- [ ] Pylint score ≥ 8.5 (9.0 final)
- [ ] All QA checks pass
- [ ] No imports from archive directories

### Common Pitfalls to Avoid

1. **Don't import from archive/** - Use only for reference
2. **Don't bypass QA checks** - All code must pass
3. **Don't ignore type errors** - Fix mypy issues
4. **Don't skip tests** - 85% coverage minimum (90% final)
5. **Don't use print()** - Use logging module
6. **Don't commit .pt files** - Model weights are gitignored

### Key References

- **Design Spec**: [docs/DESIGN.md](docs/DESIGN.md)
- **Model Spec**: [docs/MODEL_PRETRAIN.md](docs/MODEL_PRETRAIN.md)
- **Migration Plan**: [plans/MIGRATION_PLAN.md](plans/MIGRATION_PLAN.md)
- **PR Specs**: [plans/specs/](plans/specs/)

---

## Quick Reference

### Key Commands

```bash
# Start server
uvicorn src.app:application --reload

# Run tests
pytest tests/ --cov=src --cov-report=term-missing

# Type checking
mypy src/ tests/ --ignore-missing-imports

# Linting
ruff check src/ tests/ --ignore E501

# Format
ruff format src/ tests/

# Pylint
pylint src/ --ignore=archive
```

### File Locations

| What | Where |
|------|-------|
| Application factory | `src/app.py` |
| Configuration | `src/config.py` |
| Health endpoint | `src/routes/health.py` |
| Test fixtures | `tests/conftest.py` |
| Design spec | `docs/DESIGN.md` |
| Migration plan | `plans/MIGRATION_PLAN.md` |

---

## Allowed Commands Whitelist

The following commands are pre-approved for AI assistants to execute without additional confirmation. These are safe, commonly-used development commands.

### Quality Assurance Commands

```bash
# Type checking
mypy src/
mypy tests/
mypy src/ tests/

# Linting
ruff check .
ruff check src/
ruff check tests/
ruff check --fix .

# Code formatting
ruff format .
ruff format --check .
ruff format src/
ruff format tests/

# Code quality
pylint src/
pylint src/ --output-format=colorized

# Testing
pytest
pytest tests/
pytest tests/ -v
pytest tests/ --tb=short
pytest tests/ --cov=src/
pytest tests/ --cov-report=term-missing --cov=src/
pytest tests/test_main.py -v
pytest tests/test_models.py -v
pytest -x  # Stop on first failure
pytest -k "test_name"  # Run specific test by name pattern
```

### File System Commands

```bash
# Directory navigation and listing
cd src/
cd tests/
cd ..
ls
ls -la
ls src/
ls tests/
ls -la src/
ls -la tests/

# Directory creation
mkdir -p data/uploads
mkdir -p models/hold_detection

# File operations (read-only)
cat pyproject.toml
cat requirements.txt
cat mypy.ini
cat .pylintrc
head -50 src/main.py
tail -50 src/main.py
wc -l src/*.py
wc -l tests/*.py
```

### Git Commands

```bash
# Status and information (read-only)
git status
git log --oneline -10
git log --oneline -20
git diff
git diff --staged
git branch
git branch -a
git show HEAD

# Safe git operations
git add .
git add src/
git add tests/
git checkout -b feature/branch-name
git fetch origin
git pull origin main
```

### Python Commands

```bash
# Running scripts
python src/main.py
python src/setup_dev.py
python src/train_model.py --help
python src/manage_models.py list
python src/manage_models.py --help

# Package management (read-only)
pip list
pip show flask
pip freeze
```

### Project-Specific Scripts

```bash
# Development scripts
./run.sh
./run_setup_dev.sh
./run_qa.csh
```

### Dangerous Commands (Require Confirmation)

The following commands should NOT be run without explicit user confirmation:

- `rm -rf` - Recursive deletion
- `git push --force` - Force push
- `git reset --hard` - Hard reset
- `DROP TABLE` / `DELETE FROM` - Database destructive operations
- `pip uninstall` - Package removal
- Any command modifying production data or configs

---

**This guide is maintained for AI assistants working with the bouldering-analysis codebase. Keep it updated as the project evolves.**
