# Archived Legacy Tests

**Archived**: 2026-01-14
**Reason**: Major refactor to FastAPI + Supabase architecture

This directory contains the original pytest test suite that was archived during the migration to the new architecture.

## Contents

| Test File | Lines | Coverage |
|-----------|-------|----------|
| `test_main.py` | 1,253 | Flask routes, image analysis |
| `test_train_model.py` | 840 | YOLOv8 training pipeline |
| `test_manage_models.py` | 702 | Model activation/deactivation |
| `test_grade_prediction_mvp.py` | 446 | Grade prediction algorithm |
| `test_config.py` | 526 | Configuration loading |
| `test_models.py` | 498 | Database models CRUD |
| `test_e2e_grade_prediction.py` | 435 | End-to-end pipeline |
| `test_main_proxyfix.py` | 164 | ProxyFix middleware |
| `test_setup_dev.py` | 246 | Environment setup |
| `conftest.py` | 341 | Pytest fixtures |

## Reference Value

These tests demonstrate:

1. **99%+ Coverage Approach**: Comprehensive mocking and edge cases
2. **Fixture Patterns**: Test app, client, sample data fixtures
3. **YOLO Mocking**: How to mock Ultralytics models in tests
4. **Database Isolation**: Fresh database per test

## Do Not

- Run these tests against new code
- Import fixtures from this directory
- Modify these files

Preserved for reference only.
