# CLAUDE.md - AI Assistant Guide for Bouldering Route Analysis

**Version**: 2026.03.20
**Last Updated**: 2026-03-20
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
- **ML/CV**: PyTorch 2.9.1 + Ultralytics YOLOv8 8.3.233 + torchvision + XGBoost + scikit-learn
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
| 5. Route Graph | **Completed** | PR-5.x | 97% |
| ├─ Route Graph Builder | ✅ | PR-5.1 | 96% |
| └─ Route Constraints | ✅ | PR-5.2 | 97% |
| 6. Feature Extraction | **In Progress** | PR-6.x | - |
| ├─ Geometry Features | ✅ | PR-6.1 | 97% |
| ├─ Hold Features | ✅ | PR-6.2 | 100% |
| 7. Grade Estimation | **In Progress** | PR-7.x | - |
| ├─ Heuristic Grade Estimator | ✅ | PR-7.1 | - |
| └─ ML Grade Estimator | ✅ | PR-7.2 | 97% |
| 8. Explainability | **In Progress** | PR-8.x | 99% |
| ├─ Explanation Engine | ✅ | PR-8.1 | 99% |
| 9. Database Schema | **Completed** | PR-9.x | - |
| ├─ Routes Table | ✅ | PR-9.1 | - |
| ├─ Holds Table | ✅ | PR-9.2 | - |
| ├─ Features Table | ✅ | PR-9.3 | - |
| ├─ Predictions Table | ✅ | PR-9.4 | 97% |
| └─ Feedback Table | ✅ | PR-9.5 | - |
| 10. Frontend Development | Pending | PR-10.x | - |

### Archived Code

Legacy Flask code in `src/archive/legacy/` and `tests/archive/legacy/`. **Do not import from archive directories** — reference only.

### Implemented Module Summary

**Supabase Client** (`src/database/supabase_client.py`): `get_supabase_client()`, `upload_to_storage()`, `delete_from_storage()`, `get_storage_url()`, `list_storage_files()`. Storage buckets: `route-images`, `model-outputs`. See `docs/SUPABASE_SETUP.md`.

**Database Schema — Routes Table** (`migrations/sql/001_create_routes_table.sql`): Created in PR-9.1. UUID primary key (`gen_random_uuid()`), `image_url` (TEXT, max 2048), `wall_angle` (FLOAT, nullable, CHECK -90..90), `status` (VARCHAR(20), CHECK pending/processing/done/failed, DEFAULT 'pending'), `created_at`/`updated_at` (TIMESTAMPTZ NOT NULL DEFAULT NOW()). `moddatetime` trigger auto-updates `updated_at`. Indexes: `idx_routes_created_at` (DESC) for pagination, `idx_routes_status_pending` (partial) for job polling. RLS enabled: public SELECT, service-role INSERT/UPDATE/DELETE. Verifier script: `scripts/migrations/create_routes_table.py` (`verify_routes_table()`, `VerificationResult`, `--dry-run` mode; default = live verify). Keep-alive script: `scripts/ping_supabase.py` (stdlib-only `urllib`, hits `/api/v1/health/db`). All M9 tables (holds, features, predictions, feedback) are now complete.

**Database Schema — Holds Table** (`migrations/sql/002_create_holds_table.sql`): Created in PR-9.2. UUID PK (`gen_random_uuid()`), `route_id` UUID FK with `ON DELETE CASCADE`, `hold_id` INT (`CHECK >= 0`), `x_center`/`y_center`/`width`/`height` FLOAT (all `BETWEEN 0 AND 1`), `detection_class` VARCHAR(10) (`'hold'|'volume'`), `detection_confidence` FLOAT, `hold_type` VARCHAR(20) (`'jug'|'crimp'|'sloper'|'pinch'|'volume'|'unknown'`), `type_confidence` FLOAT, 6 `prob_*` FLOAT columns (per hold class, all `BETWEEN 0 AND 1`), `created_at` TIMESTAMPTZ. `UNIQUE (route_id, hold_id)` doubles as route lookup index (no separate index). No `updated_at`/trigger — write-once; re-run = `DELETE WHERE route_id` + bulk `INSERT`. Probability sum invariant enforced by `ClassifiedHold.type_probabilities` Pydantic validator (PR-9.3 must not bypass it). RLS: public SELECT, service-role INSERT/UPDATE/DELETE. Verifier: `scripts/migrations/create_holds_table.py` (`verify_holds_table()`, `--dry-run` mode; default = live verify). Shared utilities: `scripts/migrations/_migration_utils.py` (`TableVerificationConfig`, `VerificationResult`, `verify_table()`, individual helpers). `create_routes_table.py` refactored to delegate to `verify_table()` (public API unchanged).

**Database Schema — Features Table** (`migrations/sql/003_create_features_table.sql`): Created in PR-9.3. UUID PK (`gen_random_uuid()`), `route_id` UUID FK `UNIQUE` with `ON DELETE CASCADE`, `feature_vector` JSONB NOT NULL (validated at app layer by `RouteFeatures` Pydantic model), `extracted_at` TIMESTAMPTZ NOT NULL DEFAULT NOW(). No CHECK constraints, no trigger — write-once. `UNIQUE (route_id)` doubles as route lookup index (no separate index). RLS: public SELECT, service-role INSERT/UPDATE/DELETE. Re-run contract: `DELETE FROM features WHERE route_id = $1` then INSERT. Verifier: `scripts/migrations/create_features_table.py` (`verify_features_table()`, `--dry-run` mode; default = live verify). Uses shared `_migration_utils.py` utilities.

**Database Schema -- Predictions Table** (`migrations/sql/004_create_predictions_table.sql`): Created in PR-9.4. UUID PK (`gen_random_uuid()`), `route_id` UUID FK with `ON DELETE CASCADE` (no UNIQUE -- multiple predictions per route allowed), `estimator_type` VARCHAR(20) NOT NULL CHECK IN ('heuristic','ml'), `grade` VARCHAR(10) NOT NULL CHECK IN (V0-V17), `grade_index` INT NOT NULL CHECK BETWEEN 0 AND 17, `confidence` FLOAT NOT NULL CHECK BETWEEN 0 AND 1, `difficulty_score` FLOAT NOT NULL CHECK BETWEEN 0 AND 1, `uncertainty` FLOAT nullable CHECK BETWEEN 0 AND 1 (reserved for future calibrated uncertainty output), `explanation` JSONB nullable (validated at app layer by `ExplanationResult` Pydantic model), `model_version` VARCHAR(20) nullable (NULL for heuristic, `v<YYYYMMDD_HHMMSS>` for ML), `predicted_at` TIMESTAMPTZ NOT NULL DEFAULT NOW(). 6 CHECK constraints total. Compound index: `idx_predictions_route_id_predicted_at` ON (route_id, predicted_at DESC) for sorted per-route listing. No `updated_at`/trigger -- append-only immutable history. Write contract: INSERT only on every analysis run; old rows are never deleted or overwritten. RLS: public SELECT, service-role INSERT/UPDATE/DELETE (4-policy pattern). Verifier: `scripts/migrations/create_predictions_table.py` (`verify_predictions_table()`, `--dry-run` mode; default = live verify). Uses shared `_migration_utils.py` utilities.

**Database Schema -- Feedback Table** (`migrations/sql/005_create_feedback_table.sql`): Created in PR-9.5. UUID PK (`gen_random_uuid()`), `route_id` UUID FK with `ON DELETE CASCADE` (no UNIQUE -- multiple feedback per route allowed), `user_grade` VARCHAR(10) nullable CHECK (`IS NULL OR IN (V0-V17)`) -- user may omit grade, `is_accurate` BOOLEAN nullable (no CHECK), `comments` TEXT nullable (no length limit), `created_at` TIMESTAMPTZ NOT NULL DEFAULT NOW(). 1 CHECK constraint (`feedback_user_grade_check`). Explicit index: `idx_feedback_route_id_created_at` ON (route_id, created_at DESC) -- not UNIQUE. No `updated_at`/trigger -- append-only write-once. Key distinction: INSERT policy is `TO PUBLIC` (anonymous submission from frontend), not `TO service_role`. 4 RLS policies: `feedback_select_public`, `feedback_insert_public`, `feedback_update_service`, `feedback_delete_service`. Verifier: `scripts/migrations/create_feedback_table.py` (`verify_feedback_table()`, `--dry-run` mode; default = live verify). Uses shared `_migration_utils.py` utilities.

**Classification Training** (`src/training/train_classification.py`): Exports `ClassificationMetrics`, `ClassificationTrainingResult`, `train_hold_classifier()`. ResNet-18/MobileNetV3 backbones, weighted cross-entropy, Adam/AdamW/SGD, StepLR/CosineAnnealingLR. Artifacts: `models/classification/v<YYYYMMDD_HHMMSS>/weights/{best,last}.pt` + `metadata.json`.

**Classification Inference** (`src/inference/classification.py`): Exports `ClassificationInferenceError`, `HoldTypeResult`, `classify_hold()`, `classify_holds()`. Single-model cache with double-checked locking, input size from metadata, center-crop transform matches training validation transform.

**Route Graph Builder** (`src/graph/`): Exports `RouteGraphError`, `ClassifiedHold`, `make_classified_hold()`, `RouteGraph`, `build_route_graph()`, `apply_route_constraints()`, `NODE_ATTR_IS_START`, `NODE_ATTR_IS_FINISH`. `ClassifiedHold` is a Pydantic model representing a detected hold with classification result. `make_classified_hold()` is a factory for constructing `ClassifiedHold` instances. `RouteGraph` is a Pydantic model wrapping `networkx.Graph` that represents spatial relationships between holds. `build_route_graph()` constructs the graph from a list of classified holds. `apply_route_constraints()` marks start/finish holds and prunes disconnected components. Dependency: networkx 3.4.2.

**Geometry Features** (`src/features/geometry.py`): Exports `FeatureExtractionError`, `GeometryFeatures`, `extract_geometry_features()`. `FeatureExtractionError(ValueError)` is the base exception for all feature extraction failures. `GeometryFeatures` is a Pydantic model with 11 non-negative fields: edge statistics (`avg/max/min/std_move_distance`), path statistics (`path_length_min/max_distance`, `path_length_min/max_hops`), spatial metrics (`hold_density`), and graph topology (`node_count`, `edge_count`). `extract_geometry_features()` accepts a constrained `RouteGraph` (must have start/finish attributes from `apply_route_constraints`) and returns a `GeometryFeatures` instance. Uses `math.fsum` for compensated summation, `nx.single_source_dijkstra` for shortest paths, and bounding-box area for density. No NumPy dependency.

**Hold Features** (`src/features/holds.py`): Exports `HoldFeatures`, `extract_hold_features()`. `HoldFeatures` is a Pydantic model with 23 non-negative fields: hard counts per type (`jug/crimp/sloper/pinch/volume/unknown_count`), hard ratios per type (`*_ratio`), bounding-box area statistics (`avg/max/min/std_hold_size`), and confidence-weighted soft distribution (`*_soft_ratio`). `extract_hold_features()` accepts a list of `ClassifiedHold` instances (must be non-empty) and returns a `HoldFeatures` instance. Uses `math.fsum` for compensated summation. No NumPy dependency.

**Heuristic Grade Estimator** (`src/grading/heuristic.py`): Exports `HeuristicGradeResult`, `estimate_grade_heuristic()`. `GradeEstimationError(ValueError)` is the base exception for all grade estimation failures (`src/grading/exceptions.py`). `HeuristicGradeResult` is a frozen Pydantic model with 4 fields: `grade` (V-scale label), `grade_index` (ordinal 0–17), `confidence` (0.5–1.0), `difficulty_score` (0.0–1.0). `estimate_grade_heuristic()` accepts a `RouteFeatures` instance, computes hold-composition (45%) and geometry (55%) sub-scores via weighted feature combination, maps the combined difficulty score to a V-grade (V0–V17), and returns a `HeuristicGradeResult`. Constants in `src/grading/constants.py` (not re-exported): `V_GRADES` (18-entry tuple V0–V17), `GRADE_THRESHOLDS`, `MAX_HOPS_NORM=20`, `FEATURE_WEIGHTS`. Calibrated conservatively; tends to underestimate above V8. Shared internal helpers (`_clamp`, `_normalize_vector`) extracted to `src/grading/_utils.py` in PR-7.2. Public API re-exports from `src/grading/__init__.py`: `GradeEstimationError`, `HeuristicGradeResult`, `estimate_grade_heuristic`, `MLGradeResult`, `estimate_grade_ml`.

**ML Grade Estimator** (`src/grading/ml_estimator.py`): Exports `MLGradeResult`, `estimate_grade_ml()`. `MLGradeResult` is a frozen Pydantic model with 5 fields: `grade` (V-scale label), `grade_index` (ordinal 0–17), `confidence` (normalized entropy: `1 - H(p)/log(18)`, 0.0–1.0), `difficulty_score` (probability-weighted mean grade index / 17, 0.0–1.0), `grade_probabilities` (full 18-grade dict keyed by V-grade label). `estimate_grade_ml()` accepts a `RouteFeatures` instance and a `model_path` (directory containing `model.pkl` + `metadata.json`), loads (or retrieves from cache) the XGBClassifier, z-score normalizes the feature vector using training statistics from metadata, and returns a full probability distribution over V0–V17. Module-level cache keyed by resolved path; `_clear_model_cache()` for test teardown. Logs a WARNING when running a model trained on synthetic data (`data_source="synthetic"`). `isinstance(classifier, XGBClassifier)` check after `joblib.load` for safety. Dependencies: xgboost, joblib, numpy.

**Explanation Engine** (`src/explanation/engine.py`): Exports `ExplanationError`, `FeatureContribution`, `ExplanationResult`, `generate_explanation`. `ExplanationError(ValueError)` is the base exception for all explanation failures (`src/explanation/exceptions.py`). `FeatureContribution` is a frozen Pydantic model with 4 fields: `name` (human-readable), `value` (raw feature value), `impact` (signed: weight * normalized_value), `description` (one sentence). `ExplanationResult` is a frozen Pydantic model with 6 fields: `grade`, `estimator_type` (Literal["heuristic","ml"]), `confidence_qualifier` (Literal["very confident","confident","uncertain"]), `top_features` (up to 5 ranked by abs(impact)), `summary` (1-2 sentences), `hold_highlights` (top 3 hold types by ratio). `generate_explanation()` accepts a `RouteFeatures` and a `HeuristicGradeResult | MLGradeResult`, wraps `FeatureExtractionError` into `ExplanationError`, computes 5 hold contributions + 3 geometry contributions, ranks by abs(impact), builds highlights and summary. Normalization mirrors `heuristic.py`: ratios as-is for hold features; avg/max distances / MAX_MOVE_DISTANCE, hops / MAX_HOPS_NORM for geometry. Module-level `RuntimeError` guard validates `FEATURE_WEIGHTS` keys match `_FEATURE_DISPLAY_NAMES` at import time. Public API re-exports from `src/explanation/__init__.py`: `ExplanationError`, `FeatureContribution`, `ExplanationResult`, `generate_explanation`.

**Grade Estimator Training** (`src/training/train_grade_estimator.py`): Exports `GradeTrainingMetrics`, `GradeTrainingResult`, `generate_synthetic_training_data()`, `train_grade_estimator()`. `GradeTrainingMetrics` has 3 fields: `train_accuracy`, `val_accuracy`, `mean_absolute_error`. `GradeTrainingResult` has 10 fields including `version`, `model_path`, `metadata_path`, `metrics`, `data_source`, `git_commit`. `generate_synthetic_training_data()` builds routes from a fixed 3x4 spatial grid (12 holds, positions chosen for guaranteed graph connectivity within `BASE_REACH_RADIUS=0.35`), labels via heuristic estimator. `train_grade_estimator()` validates inputs, computes z-score normalization stats (population std, `std=1` fallback), remaps labels to contiguous `[0, n_unique-1]` for XGBoost `multi:softprob`, trains with 80/20 split and early stopping, writes artifacts atomically via `tempfile` + `shutil.move`. Artifacts: `models/grading/v<YYYYMMDD_HHMMSS>/model.pkl` + `metadata.json`. `metadata.json` includes: `feature_names`, `normalization_mean/std`, `classes` (sorted grade indices), `n_classes=18`, `data_source`, `git_commit`, `metrics`, `hyperparameters`.

---

## Codebase Structure

```text
bouldering-analysis/
├── src/
│   ├── app.py                    # FastAPI application factory
│   ├── config.py                 # Pydantic Settings (all BA_* env vars)
│   ├── constants.py              # Shared domain constants (MAX_HOLD_COUNT)
│   ├── logging_config.py         # Structured JSON logging
│   ├── routes/
│   │   ├── health.py             # GET /health, /api/v1/health
│   │   ├── routes.py             # POST /api/v1/routes, GET /api/v1/routes/{id}[/status]
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
│   │   ├── train_detection.py
│   │   └── train_grade_estimator.py   # XGBoost grade estimator training
│   ├── inference/
│   │   ├── detection.py          # Hold detection inference
│   │   ├── classification.py     # Hold type classification
│   │   └── crop_extractor.py     # HoldCrop type / crop extraction utilities
│   ├── graph/
│   │   ├── __init__.py           # Re-exports public API
│   │   ├── exceptions.py         # RouteGraphError(ValueError)
│   │   ├── types.py              # ClassifiedHold model, make_classified_hold()
│   │   ├── constraints.py        # apply_route_constraints(), NODE_ATTR_IS_*
│   │   └── route_graph.py        # RouteGraph model, build_route_graph()
│   ├── features/
│   │   ├── __init__.py           # Re-exports public API
│   │   ├── exceptions.py         # FeatureExtractionError(ValueError)
│   │   ├── geometry.py           # GeometryFeatures model, extract_geometry_features()
│   │   └── holds.py              # HoldFeatures model, extract_hold_features()
│   ├── grading/
│   │   ├── __init__.py           # Re-exports public API
│   │   ├── _utils.py             # Shared internal helpers (_clamp, _normalize_vector)
│   │   ├── exceptions.py         # GradeEstimationError(ValueError)
│   │   ├── constants.py          # V_GRADES, GRADE_THRESHOLDS, FEATURE_WEIGHTS
│   │   ├── heuristic.py          # HeuristicGradeResult, estimate_grade_heuristic()
│   │   └── ml_estimator.py       # MLGradeResult, estimate_grade_ml()
│   ├── explanation/
│   │   ├── __init__.py           # Re-exports public API
│   │   ├── exceptions.py         # ExplanationError(ValueError)
│   │   ├── types.py              # FeatureContribution, ExplanationResult models
│   │   └── engine.py             # generate_explanation() + private helpers
│   └── archive/legacy/           # Reference only — do not import
├── tests/
│   ├── conftest.py               # Fixtures: test_settings, app, client, app_settings
│   ├── test_*.py                 # One test file per src module
│   ├── test_migration_utils.py   # Shared migration utility tests
│   ├── test_migrations_holds.py  # Holds table migration tests
│   ├── test_migrations_features.py  # Features table migration tests
│   ├── test_migrations_predictions.py  # Predictions table migration tests
│   └── archive/legacy/
├── migrations/
│   └── sql/
│       ├── 001_create_routes_table.sql  # Routes table DDL (RLS, triggers, indexes)
│       ├── 002_create_holds_table.sql   # Holds table DDL (FK, RLS, unique constraint)
│       ├── 003_create_features_table.sql # Features table DDL (FK, RLS, JSONB)
│       └── 004_create_predictions_table.sql # Predictions table DDL (FK, RLS, CHECKs, compound index)
├── scripts/
│   ├── ping_supabase.py                 # Keep-alive ping (stdlib urllib)
│   └── migrations/
│       ├── _migration_utils.py          # Shared PostgREST verifier utilities
│       ├── create_routes_table.py       # Routes table verifier (PostgREST)
│       ├── create_holds_table.py        # Holds table verifier (PostgREST)
│       ├── create_features_table.py     # Features table verifier (PostgREST)
│       └── create_predictions_table.py  # Predictions table verifier (PostgREST)
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

- Framework: pytest | One test file per source module | Current coverage: 97%+
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
| `BA_SUPABASE_TIMEOUT_SECONDS` | `10` | Supabase PostgREST request timeout in seconds |
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
| POST | `/api/v1/routes` | Create route record | RouteResponse |
| GET | `/api/v1/routes/{route_id}` | Retrieve route by ID | RouteResponse |
| GET | `/api/v1/routes/{route_id}/status` | Poll route processing status | RouteStatusResponse |
| GET | `/docs` | Swagger UI (debug only) | HTML |
| GET | `/openapi.json` | OpenAPI schema (debug only) | JSON |

**HealthResponse**: `{status: "healthy"|"degraded"|"unhealthy", version, timestamp}`

**UploadResponse**: `{file_id, public_url, file_size, content_type, uploaded_at}`

**RouteResponse**: `{id, image_url, wall_angle, created_at, updated_at, status}`

**RouteStatusResponse**: `{id, status}`

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
