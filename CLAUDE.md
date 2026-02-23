# CLAUDE.md - AI Assistant Guide for Bouldering Route Analysis

**Version**: 2026.02.23
**Last Updated**: 2026-02-23
**Architecture**: FastAPI + Supabase (Migration in Progress)
**Repository**: bouldering-analysis

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Status](#architecture-status)
3. [Codebase Structure](#codebase-structure)
4. [Development Workflows](#development-workflows)
5. [Code Conventions](#code-conventions)
6. [Testing & Quality Standards](#testing--quality-standards)
7. [Configuration Management](#configuration-management)
8. [API Reference](#api-reference)
9. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

A web-based system that estimates bouldering route difficulty (V-scale) from images by detecting and classifying climbing holds, constructing movement graphs, and providing explainable grade predictions.

### Core Design Principles

1. **Backend-first architecture** - FastAPI with Supabase
2. **Route difficulty ≠ pixels** - Use structured features, not end-to-end CNN
3. **Perception models are pre-trained** - Frozen for inference
4. **Grades are ordinal and uncertain** - Probabilistic predictions
5. **One PR ≈ one function** - Modular, testable changes
6. **Every ML output must be explainable** - No black boxes

### Key Technologies

- **Backend**: FastAPI 0.128.6 + Pydantic Settings + uvicorn
- **ML/CV**: PyTorch 2.9.1 + Ultralytics YOLOv8 8.3.233 + torchvision
- **Database**: Supabase (Postgres + Storage) — in progress
- **Frontend**: React/Next.js (Lovable → Vercel) + Telegram Bot alternative
- **Testing**: pytest 8.3.5 | Coverage: ≥85% now → ≥90% final
- **Quality**: mypy, ruff, pylint | Score: ≥8.5/10 now → ≥9.0/10 final
- **Python**: 3.10–3.13 | Dependency management: uv

---

## Architecture Status

The codebase is being migrated from Flask to FastAPI + Supabase.

| Milestone | Status | PR | Coverage |
|-----------|--------|-----|----------|
| 1. Backend Foundation | **Completed** | | 98% |
| ├─ FastAPI Bootstrap | ✅ | PR-1.1 | 100% |
| └─ Supabase Client | ✅ | PR-1.2 | 100% |
| 2. Image Upload | **Completed** | | 97% |
| ├─ Upload Route Image | ✅ | PR-2.1 | 97% |
| └─ Create Route Record | ✅ | PR-2.2 | - |
| 3. Hold Detection | **Completed** | PR-3.x | - |
| ├─ Detection Dataset Schema | ✅ | PR-3.1 | - |
| ├─ Detection Model Definition | ✅ | PR-3.2 | - |
| ├─ Detection Training Loop | ✅ | PR-3.3 | - |
| └─ Detection Inference | ✅ | PR-3.4 | 94% |
| 4. Hold Classification | **Completed** | PR-4.x | - |
| ├─ Hold Crop Generator | ✅ | PR-4.1 | - |
| ├─ Classification Dataset Loader | ✅ | PR-4.2 | - |
| ├─ Hold Classifier Model | ✅ | PR-4.3 | - |
| ├─ Classification Training Loop | ✅ | PR-4.4 | 97% |
| └─ Classification Inference | ✅ | PR-4.5 | - |
| 5. Route Graph | Pending | PR-5.x | - |
| 6. Feature Extraction | Pending | PR-6.x | - |
| 7. Grade Estimation | Pending | PR-7.x | - |
| 8. Explainability | Pending | PR-8.x | - |
| 9. Database Schema | Pending | PR-9.x | - |
| 10. Frontend Development | Pending | PR-10.x | - |

### Archived Code

Legacy Flask code in `src/archive/legacy/` and `tests/archive/legacy/`. **Do not import from archive directories** — reference only.

### Implemented Module Summary

**Supabase Client** (`src/database/supabase_client.py`): `get_supabase_client()`, `upload_to_storage()`, `delete_from_storage()`, `get_storage_url()`, `list_storage_files()`. Storage buckets: `route-images`, `model-outputs`. See `docs/SUPABASE_SETUP.md`.

**Database Schema** (❌ PENDING — PR-2.2 & PR-9.x): Tables `routes`, `holds`, `features`, `predictions`, `feedback` are defined in `plans/MIGRATION_PLAN.md` but not yet created in Supabase.

**Classification Training** (`src/training/train_classification.py`): Exports `ClassificationMetrics`, `ClassificationTrainingResult`, `train_hold_classifier()`. ResNet-18/MobileNetV3 backbones, weighted cross-entropy, Adam/AdamW/SGD, StepLR/CosineAnnealingLR. Artifacts: `models/classification/v<YYYYMMDD_HHMMSS>/weights/{best,last}.pt` + `metadata.json`.

**Classification Inference** (`src/inference/classification.py`): Exports `ClassificationInferenceError`, `HoldTypeResult`, `classify_hold()`, `classify_holds()`. Single-model cache with double-checked locking, input size from metadata, center-crop transform matches training validation transform.

---

## Codebase Structure

```text
bouldering-analysis/
├── src/
│   ├── app.py                    # FastAPI application factory
│   ├── config.py                 # Pydantic Settings (all BA_* env vars)
│   ├── logging_config.py         # Structured JSON logging
│   ├── routes/
│   │   ├── health.py             # GET /health, /api/v1/health
│   │   └── upload.py             # POST /api/v1/routes/upload
│   ├── database/
│   │   └── supabase_client.py    # Supabase client & storage helpers
│   ├── training/
│   │   ├── classification_dataset.py
│   │   ├── classification_model.py    # ResNet-18/MobileNetV3, apply_classifier_dropout
│   │   ├── datasets.py               # YOLOv8 detection dataset
│   │   ├── detection_model.py
│   │   ├── exceptions.py
│   │   ├── train_classification.py
│   │   └── train_detection.py
│   ├── inference/
│   │   ├── detection.py          # Hold detection inference
│   │   ├── classification.py     # Hold type classification
│   │   └── crop_extractor.py     # HoldCrop type / crop extraction utilities
│   └── archive/legacy/           # Reference only — do not import
├── tests/
│   ├── conftest.py               # Fixtures: test_settings, app, client, app_settings
│   ├── test_*.py                 # One test file per src module
│   └── archive/legacy/
├── docs/                         # DESIGN.md, MODEL_PRETRAIN.md, setup guides
├── plans/
│   ├── MIGRATION_PLAN.md         # Full roadmap and DB schema definitions
│   └── specs/                    # Per-PR specifications
├── pyproject.toml                # Dependencies, pytest & coverage config
├── uv.lock                       # Locked dependencies (committed)
├── mypy.ini                      # Type checking config
└── .pylintrc                     # Pylint config
```

---

## Development Workflows

### Setup

```bash
# Install all deps including dev (uv recommended)
uv sync --no-install-project --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Verify
uv run python -c "from src.app import create_app; print('OK')"
```

### Full QA — run before every commit

```bash
uv run mypy src/ tests/ && \
uv run ruff check src/ tests/ --ignore E501 && \
uv run ruff format --check src/ tests/ && \
uv run pytest tests/ --cov=src --cov-fail-under=85 && \
uv run pylint src/ --ignore=archive
```

### Individual commands

```bash
uv run mypy src/ tests/                                     # Type checking
uv run ruff check src/ tests/ --ignore E501                 # Linting
uv run ruff format src/ tests/                              # Format
uv run pytest tests/ --cov=src --cov-report=term-missing    # Tests
uv run pylint src/ --ignore=archive                         # Quality score
uv run uvicorn src.app:application --reload                 # Run server (http://localhost:8000)
```

---

## Code Conventions

### Naming

- Modules: `snake_case` | Classes: `PascalCase` | Functions: `snake_case` | Constants: `UPPER_SNAKE_CASE` | Private: `_leading_underscore`

### Docstrings — Google style required on all public modules, classes, and functions

```python
def create_app(config_override: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_override: Optional overrides for testing.

    Returns:
        Configured FastAPI application instance.

    Example:
        >>> app = create_app({"testing": True})
    """
```

### Type Annotations — full hints required on all functions

```python
def get_settings() -> Settings: ...
def create_app(config_override: dict[str, Any] | None = None) -> FastAPI: ...
```

### Import Order

1. Standard library
2. Third-party
3. Local (`from src.config import get_settings`)

### Style

- Formatter: ruff (Black-compatible) | Quotes: double | Indentation: 4 spaces
- Trailing commas in multi-line collections | E501 ignored (long lines allowed)

---

## Testing & Quality Standards

### Thresholds (staged)

| Check | Tool | Now | Final (all features complete) |
|-------|------|-----|-------------------------------|
| Type Safety | mypy | No errors | No errors |
| Linting | ruff check | No errors | No errors |
| Formatting | ruff format | Formatted | Formatted |
| Coverage | pytest-cov | **≥85%** | **≥90%** |
| Quality | pylint | **≥8.5/10** | **≥9.0/10** |

### Test conventions

- Framework: pytest | One test file per source module | Current coverage: 98%+
- Each test uses a fresh app instance (via `app` fixture in `tests/conftest.py`)
- Test class per feature group; method names: `test_<scenario>_<expected_outcome>`

### Pre-Commit Checklist

- [ ] Tests pass + coverage ≥ 85%: `pytest tests/ --cov=src --cov-fail-under=85`
- [ ] Type checking: `mypy src/ tests/`
- [ ] Linting + format: `ruff check . --ignore E501 && ruff format --check .`
- [ ] Pylint ≥ 8.5: `pylint src/ --ignore=archive`
- [ ] Parallel reviews: python-reviewer, code-reviewer, security-reviewer
- [ ] Docs updated: doc-updater

---

## Configuration Management

| Variable | Default | Description |
|----------|---------|-------------|
| `BA_APP_NAME` | `bouldering-analysis` | Application name |
| `BA_APP_VERSION` | `0.1.0` | Application version |
| `BA_DEBUG` | `false` | Enable debug mode |
| `BA_TESTING` | `false` | Enable testing mode |
| `BA_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array) |
| `BA_LOG_LEVEL` | `INFO` | Logging level |
| `BA_SUPABASE_URL` | `""` | Supabase project URL (required) |
| `BA_SUPABASE_KEY` | `""` | Supabase API key (required) |
| `BA_MAX_UPLOAD_SIZE_MB` | `10` | Max file upload size in MB |
| `BA_STORAGE_BUCKET` | `route-images` | Supabase storage bucket |
| `BA_ALLOWED_IMAGE_TYPES` | `["image/jpeg", "image/png"]` | Allowed MIME types |

All variables prefixed `BA_`. Access via `from src.config import get_settings`. See `docs/SUPABASE_SETUP.md`.

---

## API Reference

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/health` | Health check | HealthResponse |
| GET | `/api/v1/health` | Health check (versioned) | HealthResponse |
| POST | `/api/v1/routes/upload` | Upload route image | UploadResponse |
| GET | `/docs` | Swagger UI (debug only) | HTML |
| GET | `/openapi.json` | OpenAPI schema (debug only) | JSON |

**HealthResponse**: `{status: "healthy"|"degraded"|"unhealthy", version, timestamp}`

**UploadResponse**: `{file_id, public_url, file_size, content_type, uploaded_at}`

---

## AI Assistant Guidelines

### When Making Changes

1. **Read before modifying** — always read files first
2. **Check specs** — review `docs/DESIGN.md` and `plans/MIGRATION_PLAN.md`
3. **Follow conventions** — match existing code style and patterns
4. **TDD** — write tests before implementation
5. **Maintain coverage** — ≥85% now, increases to ≥90% when all features complete
6. **Run full QA** — all checks must pass before committing
7. **Update docs** — keep CLAUDE.md and specs current

### Agent Workflow (Mandatory)

Agent types invoked via Task tool (e.g., `everything-claude-code:<name>` or `~/.claude/agents/`).

**Per-feature sequence:**

1. **planner** — before writing code (all PRs touching > 1 file)
2. **tdd-guide** — after planning, before implementation
3. **database-reviewer** — when Supabase schema/SQL is touched; complete before steps 4–6
4. **python-reviewer** } run in parallel
5. **code-reviewer**    }
6. **security-reviewer** }
7. **doc-updater** — after clean reviews (CLAUDE.md, specs, docstrings)

**Additional triggers**: **e2e-runner** at milestone completion | **architect** for design decisions

### Code Review Checklist

- [ ] Type hints on all functions
- [ ] Google-style docstrings on all public functions
- [ ] Error handling for edge cases
- [ ] Tests added for new functionality
- [ ] Coverage ≥ 85% | Pylint ≥ 8.5/10
- [ ] All QA checks pass
- [ ] No imports from archive directories

### Common Pitfalls

1. **Don't import from archive/** — reference only
2. **Don't bypass QA checks** — all code must pass
3. **Don't ignore type errors** — fix mypy issues
4. **Don't use print()** — use logging module
5. **Don't commit .pt files** — model weights are gitignored

### Key References

- `docs/DESIGN.md` — Architecture spec
- `docs/MODEL_PRETRAIN.md` — ML spec
- `plans/MIGRATION_PLAN.md` — Full migration roadmap and DB schemas
- `plans/specs/` — Per-PR specifications
- `docs/PRE_COMMIT_HOOKS.md` — Pre-commit guide

---

**Keep this file updated as the project evolves.**
