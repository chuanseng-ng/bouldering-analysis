# Archived Legacy Source Code

**Archived**: 2026-01-14
**Reason**: Major refactor to FastAPI + Supabase architecture

This directory contains the original Flask-based implementation that was archived during the migration to the new architecture defined in `docs/DESIGN.md`.

## Contents

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 883 | Flask app + REST API endpoints |
| `models.py` | 290 | SQLAlchemy ORM models (6 models) |
| `config.py` | 323 | Thread-safe YAML configuration loader |
| `train_model.py` | 1,222 | YOLOv8 fine-tuning pipeline |
| `manage_models.py` | 630 | Model version management CLI |
| `grade_prediction_mvp.py` | 618 | Phase 1a MVP grade prediction algorithm |
| `constants.py` | 19 | Hold type constants |
| `setup.py` | 44 | Database initialization |
| `setup_dev.py` | 113 | Dev environment setup |
| `cfg/user_config.yaml` | - | Application configuration |
| `templates/index.html` | - | Web UI template |

## Reusable Patterns

Consider referencing these files for:

1. **Configuration Loading**: `config.py` - Thread-safe YAML loading with caching
2. **Model Management**: `manage_models.py` - CLI pattern for model versioning
3. **Database Patterns**: `models.py` - SQLAlchemy relationship patterns
4. **Test Fixtures**: See `tests/archive/legacy/conftest.py`

## Do Not

- Import from this directory
- Modify these files
- Reference in production code

These files are preserved for reference only.
