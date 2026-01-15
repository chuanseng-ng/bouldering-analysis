# PR-1.1: FastAPI Bootstrap — Detailed Specification

**Status**: IMPLEMENTED
**Completed**: 2026-01-14
**Milestone**: 1 — Backend Foundation
**Dependencies**: None

---

## Implementation Summary

| Metric | Target | Actual |
|--------|--------|--------|
| Test Coverage | 85% | 100% |
| Pylint Score | 8.5/10 | 10.00/10 |
| Tests Passing | All | 73/73 |
| mypy | No errors | Passed |
| ruff | No errors | Passed |

---

## 1. Objective

Create a minimal, production-ready FastAPI application with:

- [x] Application factory pattern
- [x] Health check endpoint
- [x] CORS middleware
- [x] API versioning
- [x] Structured JSON logging

---

## 2. File Structure (Implemented)

```text
src/
├── __init__.py              # Package marker with create_app export
├── app.py                   # Application factory (160 lines)
├── config.py                # Configuration management (105 lines)
├── logging_config.py        # Structured logging setup (116 lines)
└── routes/
    ├── __init__.py          # Routes package
    └── health.py            # Health check endpoint (48 lines)

tests/
├── __init__.py              # Tests package
├── conftest.py              # Pytest fixtures (70 lines)
├── test_app.py              # Application tests (140 lines)
├── test_config.py           # Configuration tests (180 lines)
├── test_health.py           # Health endpoint tests (78 lines)
└── test_logging_config.py   # Logging tests (158 lines)
```

---

## 3. Dependencies (Updated)

### requirements.txt

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

# Testing
pytest==9.0.1
pytest-cov==7.0.0
pytest-asyncio==0.25.2

# Quality
pylint==4.0.3
ruff==0.14.7
mypy==1.18.2

# Config
PyYAML==6.0.2
```

---

## 4. Function Contracts (Implemented)

### 4.1 Application Factory

**File**: `src/app.py`

```python
def create_app(config_override: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
```

### 4.2 Health Check Endpoint

**File**: `src/routes/health.py`

```python
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check application health status."""
```

### 4.3 Configuration

**File**: `src/config.py`

```python
class Settings(BaseSettings):
    app_name: str = "bouldering-analysis"
    app_version: str = "0.1.0"
    debug: bool = False
    testing: bool = False
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"

def get_settings() -> Settings:
    """Get cached application settings."""

def get_settings_override(overrides: dict[str, Any]) -> Settings:
    """Create settings with specific overrides for testing."""
```

### 4.4 Logging Configuration

**File**: `src/logging_config.py`

```python
def configure_logging(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structured JSON logging for the application."""

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
```

---

## 5. API Specification (Implemented)

### 5.1 Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| GET | `/health` | Health check (root level) | Implemented |
| GET | `/api/v1/health` | Health check (versioned) | Implemented |
| GET | `/docs` | Swagger UI (debug/testing only) | Implemented |
| GET | `/redoc` | ReDoc (debug/testing only) | Implemented |
| GET | `/openapi.json` | OpenAPI schema | Implemented |

### 5.2 Response Models

**HealthResponse**

```python
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
```

**Example Response**:
```json
{
    "status": "healthy",
    "version": "0.1.0",
    "timestamp": "2026-01-14T12:00:00Z"
}
```

---

## 6. Middleware Configuration (Implemented)

### 6.1 CORS Middleware

- Configurable origins via `BA_CORS_ORIGINS` env var
- Supports all HTTP methods
- Credentials enabled

### 6.2 Request ID Middleware

- Generates UUID if not provided
- Preserves existing `X-Request-ID` header
- Adds `X-Request-ID` to response headers

---

## 7. Configuration (Implemented)

### 7.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BA_APP_NAME` | `bouldering-analysis` | Application name |
| `BA_APP_VERSION` | `0.1.0` | Application version |
| `BA_DEBUG` | `false` | Enable debug mode |
| `BA_TESTING` | `false` | Enable testing mode |
| `BA_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array) |
| `BA_LOG_LEVEL` | `INFO` | Logging level |

### 7.2 Example .env File

```bash
BA_APP_NAME=bouldering-analysis
BA_APP_VERSION=0.1.0
BA_DEBUG=false
BA_LOG_LEVEL=INFO
BA_CORS_ORIGINS=["http://localhost:3000"]
```

---

## 8. Testing (Implemented)

### 8.1 Test Coverage

- **Target**: 85% (current stage), 90% (final)
- **Achieved**: 100%

### 8.2 Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_app.py | 27 | Passed |
| test_config.py | 24 | Passed |
| test_health.py | 8 | Passed |
| test_logging_config.py | 14 | Passed |
| **Total** | **73** | **All Passed** |

### 8.3 Test Classes

- `TestCreateApp` - Application factory tests
- `TestHealthEndpoint` - Health endpoint tests
- `TestVersionedHealthEndpoint` - API versioning tests
- `TestCorsMiddleware` - CORS configuration tests
- `TestRequestIdMiddleware` - Request ID generation tests
- `TestOpenAPISchema` - API documentation tests
- `TestSettings` - Configuration tests
- `TestLogLevelValidation` - Log level validation tests
- `TestCorsOriginsValidation` - CORS origins parsing tests
- `TestHealthResponse` - Response model validation tests
- `TestConfigureLogging` - Logging configuration tests
- `TestCustomJsonFormatter` - JSON formatter tests

---

## 9. Quality Gates (All Passed)

### 9.1 Pre-Merge Checks

- [x] `mypy src/ tests/` passes with no errors
- [x] `ruff check .` passes with no errors
- [x] `ruff format --check .` passes
- [x] `pytest tests/ --cov=src/ --cov-fail-under=85` passes (100%)
- [x] `pylint src/` score >= 8.5/10 (10.00/10)

### 9.2 Manual Verification

```bash
# Start server
uvicorn src.app:application --reload
# Or with factory pattern
uvicorn src.app:create_app --factory --reload

# Verify endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/docs
curl http://localhost:8000/openapi.json
```

---

## 10. Implementation Checklist (Completed)

- [x] Create `src/config.py` with Settings class
- [x] Create `src/logging_config.py` with JSON logging
- [x] Create `src/routes/__init__.py` package
- [x] Create `src/routes/health.py` with health endpoint
- [x] Create `src/app.py` with application factory
- [x] Update `requirements.txt` with new dependencies
- [x] Create `tests/__init__.py` package
- [x] Create `tests/conftest.py` with fixtures
- [x] Create `tests/test_app.py` with test cases
- [x] Create `tests/test_config.py` with config tests
- [x] Create `tests/test_health.py` with model tests
- [x] Create `tests/test_logging_config.py` with logging tests
- [x] Verify all quality gates pass
- [x] Update `pyproject.toml` for pytest config

---

## 11. Future Considerations

Items explicitly NOT in scope for PR-1.1 (deferred to future PRs):

- Database connectivity (PR-1.2)
- File upload handling (PR-2.1)
- Authentication/authorization
- Rate limiting
- Metrics/observability endpoints

---

## 12. References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Uvicorn](https://www.uvicorn.org/)
- [docs/DESIGN.md](../../docs/DESIGN.md) - Project design specification
- [docs/MODEL_PRETRAIN.md](../../docs/MODEL_PRETRAIN.md) - Model pretraining specification
