# CLAUDE.md - AI Assistant Guide for Bouldering Route Analysis

**Version**: 2026.02.23
**Last Updated**: 2026-02-23
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

- **Backend**: FastAPI 0.128.6 with Pydantic Settings
- **ML/CV**: PyTorch 2.9.1 + Ultralytics YOLOv8 8.3.233
- **Database**: Supabase (Postgres + Storage) - planned
- **Frontend**: React/Next.js (via Lovable export), deployed on Vercel
- **Testing**: pytest 8.3.5 with staged coverage requirements (≥85% now, ≥90% final)
- **Quality**: mypy, ruff, pylint with staged requirements (≥8.5/10 now, ≥9.0/10 final)

---

## Architecture Status

### Migration Progress

The codebase is being migrated from Flask to FastAPI + Supabase.

| Milestone | Status | PR | Coverage |
|-----------|--------|-----|----------|
| 1. Backend Foundation | **Completed** | | 98% |
| ├─ FastAPI Bootstrap | ✅ Completed | PR-1.1 | 100% |
| └─ Supabase Client | ✅ Completed | PR-1.2 | 100% |
| 2. Image Upload | **Completed** | | 97% |
| ├─ Upload Route Image | ✅ Completed | PR-2.1 | 97% |
| └─ Create Route Record | ✅ Completed | PR-2.2 | - |
| 3. Hold Detection | **Completed** | PR-3.x | - |
| ├─ Detection Dataset Schema | ✅ Completed | PR-3.1 | - |
| ├─ Detection Model Definition | ✅ Completed | PR-3.2 | - |
| ├─ Detection Training Loop | ✅ Completed | PR-3.3 | - |
| └─ Detection Inference | ✅ Completed | PR-3.4 | 94% |
| 4. Hold Classification | **Completed** | PR-4.x | - |
| ├─ Classification Dataset Schema | ✅ Completed | PR-4.1 | - |
| ├─ Classification Model Definition | ✅ Completed | PR-4.2 | - |
| ├─ Omit Dropout from Model Builder | ✅ Completed | PR-4.3 | - |
| ├─ Classification Training Loop | ✅ Completed | PR-4.4 | 97% |
| └─ Classification Inference | ✅ Completed | PR-4.5 | - |
| 5. Route Graph | Pending | PR-5.x | - |
| 6. Feature Extraction | Pending | PR-6.x | - |
| 7. Grade Estimation | Pending | PR-7.x | - |
| 8. Explainability | Pending | PR-8.x | - |
| 9. Database Schema | Pending | PR-9.x | - |
| 10. Frontend Development | Pending | PR-10.x | - |

### Frontend Development Strategy

The application provides **two frontend interfaces** to serve different use cases:

#### Web Frontend (Primary)

The web frontend follows a two-phase development approach:

**Phase 1: Initial Development (Lovable)**
- Rapid prototyping using [Lovable](https://lovable.dev) platform (no-code/low-code)
- Build core UI components and user flows
- Establish initial design system
- Create working prototype for user testing
- Connect to backend API endpoints

**Phase 2: Enhancement & Deployment (Claude Code + Vercel)**
- Export Lovable project to code repository
- Refine and extend functionality using Claude Code
- Add advanced features and optimizations
- Deploy to [Vercel](https://vercel.com) for production hosting
- Set up continuous deployment from Git

**Technology Stack**:
- **Prototyping**: Lovable (no-code platform for rapid UI development)
- **Framework**: React/Next.js (exported from Lovable)
- **Modifications**: Claude Code (AI-assisted development)
- **Hosting**: Vercel (with automatic Git deployments)
- **API Integration**: FastAPI backend via REST API

**Use Cases**: Full-featured analysis, detailed annotations, route history, advanced features

#### Telegram Bot Frontend (Alternative)

A lightweight Telegram bot for quick, on-the-go route analysis:

**Features**:
- Send route photo directly in Telegram
- Receive grade prediction via message
- Simple text-based interaction
- No installation or signup required
- Works on any device with Telegram

**Technology Stack**:
- **Framework**: Python Telegram Bot (`python-telegram-bot`)
- **Hosting**: TBD (Serverless function or always-on service)
- **API Integration**: FastAPI backend via REST API

**Use Cases**: Quick grade checks, gym climbers on-the-go, minimal friction analysis

See [docs/TELEGRAM_BOT.md](docs/TELEGRAM_BOT.md) for implementation guide.

#### Frontend Environment Variables

**Web Frontend**:
| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |
| `NEXT_PUBLIC_SUPABASE_URL` | `""` | Supabase URL (if frontend uses Supabase directly) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `""` | Supabase anon key (if needed) |

**Telegram Bot**:
| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | `""` | Telegram Bot API token (from @BotFather) |
| `TELEGRAM_API_URL` | `http://localhost:8000` | Backend API base URL |
| `TELEGRAM_WEBHOOK_URL` | `""` | Public webhook URL (for production) |

See [docs/FRONTEND_WORKFLOW.md](docs/FRONTEND_WORKFLOW.md), [docs/VERCEL_SETUP.md](docs/VERCEL_SETUP.md), and [docs/TELEGRAM_BOT.md](docs/TELEGRAM_BOT.md) for detailed guides.

### Archived Code

Legacy Flask-based code is preserved in:
- `src/archive/legacy/` - Original source files
- `tests/archive/legacy/` - Original test files

**Do not import from archive directories** - they are for reference only.

---

## Database Implementation Status

### Supabase Client (✅ COMPLETED - PR-1.2)

**Module**: `src/database/supabase_client.py`

**Implemented Functions**:
- `get_supabase_client()` - Returns cached Supabase client with connection pooling
- `upload_to_storage(bucket, file_path, file_data, content_type)` - Upload files to Supabase Storage
- `delete_from_storage(bucket, file_path)` - Delete files from storage
- `get_storage_url(bucket, file_path)` - Get public URL for stored files
- `list_storage_files(bucket, path)` - List files in a storage bucket

**Configuration** (in `src/config.py`):
- `supabase_url` - Supabase project URL (env: `BA_SUPABASE_URL`)
- `supabase_key` - Supabase API key (env: `BA_SUPABASE_KEY`)
- `storage_bucket` - Default storage bucket name (default: `route-images`)
- `max_upload_size_mb` - Maximum upload size in MB (default: `10`)
- `allowed_image_types` - List of allowed MIME types (default: `["image/jpeg", "image/png"]`)

**Test Coverage**: 100% (`tests/test_supabase_client.py`)

**Connection Test**: Run `python test_supabase_connection.py` to verify Supabase setup

### Image Upload (✅ COMPLETED - PR-2.1)

**Module**: `src/routes/upload.py`

**Endpoint**: `POST /api/v1/routes/upload`

**Features**:
- Multipart file upload handling
- File type validation (JPEG, PNG)
- File size validation (configurable limit)
- Magic byte signature validation (prevents file type spoofing)
- Automatic file organization by year/month
- Unique UUID-based file naming
- Returns public URL and metadata

**Response Model**:
```python
{
    "file_id": "uuid",
    "public_url": "https://...",
    "file_size": 1234,
    "content_type": "image/jpeg",
    "uploaded_at": "2026-01-26T12:00:00Z"
}
```

**Test Coverage**: 97% (`tests/test_upload.py`)

### Database Schema (❌ PENDING - PR-2.2 & PR-9.x)

The following Supabase tables are **planned but not yet implemented**:

#### Table: routes**

```sql
CREATE TABLE routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_url TEXT NOT NULL,
    wall_angle FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: holds**
```sql
CREATE TABLE holds (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id),
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    size FLOAT,
    type VARCHAR(20),
    confidence FLOAT
);
```

**Table: features**
```sql
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id) UNIQUE,
    feature_vector JSONB NOT NULL,
    extracted_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: predictions**
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id),
    grade VARCHAR(10) NOT NULL,
    confidence FLOAT,
    uncertainty FLOAT,
    explanation TEXT,
    model_version VARCHAR(50),
    predicted_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: feedback**
```sql
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id),
    user_grade VARCHAR(10),
    is_accurate BOOLEAN,
    comments TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Status**: Schema defined in `plans/MIGRATION_PLAN.md` but not yet implemented in Supabase.

**Next Steps** (PR-2.2):
1. Create `routes` table in Supabase
2. Implement `POST /api/v1/routes` endpoint to create route records
3. Link uploaded images to route records
4. Add comprehensive tests

### Storage Buckets

**Required Buckets** (to be created in Supabase):
- `route-images` - Storage for uploaded bouldering route images
- `model-outputs` - Storage for model inference outputs (future)

**Current Status**: Buckets must be manually created in Supabase (see `docs/SUPABASE_SETUP.md`)

### Overall Database Coverage

- **Supabase Client**: ✅ 100%
- **Storage Operations**: ✅ 100%
- **Image Upload**: ✅ 97%
- **Database Tables**: ❌ Not implemented
- **Route Records**: ❌ Not implemented

---

## Model Training Implementation Status

### Classification Training (✅ COMPLETED - PR-4.4)

**Module**: `src/training/train_classification.py`

**Features**:
- Full PyTorch training orchestration for hold type classification
- Supports ResNet-18 and MobileNetV3 backbones
- Weighted cross-entropy loss for class imbalance handling
- Optimizers: Adam, AdamW, SGD
- Learning rate schedulers: StepLR, CosineAnnealingLR
- Standard ImageNet augmentations (training) and center-crop evaluation (validation)
- Metrics: Top-1 accuracy, Expected Calibration Error (ECE)
- Versioned artifact layout: `models/classification/<version>/`
- Automatic metadata capture (git commit, timestamp, hyperparameters)

**Exported Classes and Functions**:

```python
ClassificationMetrics       # Metrics from training run
ClassificationTrainingResult  # Full training result with artifact paths
train_hold_classifier()     # Main training entry point
```

**Result Models**:
- `ClassificationMetrics` - Top-1 accuracy, validation loss, ECE, best epoch
- `ClassificationTrainingResult` - Version, architecture, weights paths, metadata path, metrics, dataset root, git commit, trained timestamp, hyperparameters

**Test Coverage**: 97% (`tests/test_train_classification.py`)

**Example Usage**:

```python
from src.training.classification_dataset import load_hold_classification_dataset
from src.training.train_classification import train_hold_classifier

dataset = load_hold_classification_dataset("data/hold_classification")
result = train_hold_classifier(dataset, "data/hold_classification")
print(f"Top-1 Accuracy: {result.metrics.top1_accuracy:.2%}")
print(f"Weights: {result.best_weights_path}")
```

**Artifact Organization**:

```text
models/classification/
└── v<YYYYMMDD_HHMMSS>/
    ├── weights/
    │   ├── best.pt           # Best checkpoint
    │   └── last.pt           # Final checkpoint
    └── metadata.json         # Training metadata & hyperparameters
```

### Classification Inference (✅ COMPLETED - PR-4.5)

**Module**: `src/inference/classification.py`

**Features**:
- Inference API for hold type classification from cropped images
- Single-model caching with double-checked locking pattern (matching `detection.py`)
- Automatic input size discovery from model metadata
- Inference-time transform matches training-time validation transform
- Batch inference support for multiple crops
- Comprehensive error handling and validation

**Exported Classes and Functions**:

```python
ClassificationInferenceError      # Exception for classification failures
HoldTypeResult                    # Result object with type, confidence, probabilities
classify_hold()                   # Single crop classification
classify_holds()                  # Batch crop classification
```

**Result Models**:
- `HoldTypeResult` - Predicted hold type, confidence score (0-1), probability distribution over classes, input size used
- Batch operations return `list[HoldTypeResult]`

**Test Coverage**: 72 tests in `tests/test_inference_classification.py`

**Example Usage**:

```python
from src.inference.classification import classify_hold, classify_holds
from PIL import Image

# Single crop
crop = Image.open("hold_crop.jpg")
result = classify_hold(crop, "models/classification/v20260222_120000/weights/best.pt")
print(f"Type: {result.hold_type}, Confidence: {result.confidence:.2%}")

# Multiple crops
crops = [Image.open(f"crop_{i}.jpg") for i in range(5)]
results = classify_holds(crops, "models/classification/v20260222_120000/weights/best.pt")
for result in results:
    print(f"{result.hold_type}: {result.confidence:.2%}")
```

**Key Design Patterns**:
- Model is loaded once and cached in `_MODEL_CACHE` (double-checked locking)
- Metadata `input_size` is cached in `_INPUT_SIZE_CACHE` for transform consistency
- `apply_classifier_dropout()` (from `src/training/classification_model.py`) reconstructs architecture for any backbone
- Inference transform (center-crop) matches training-time validation transform

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
│   │   ├── health.py             # Health check endpoint
│   │   └── upload.py             # Image upload endpoint
│   ├── database/                 # Database layer (Supabase)
│   │   ├── __init__.py           # Database exports
│   │   └── supabase_client.py    # Supabase client & storage helpers
│   ├── training/                 # Model training module (classification & detection)
│   │   ├── __init__.py           # Training module exports
│   │   ├── classification_dataset.py  # Classification dataset loading
│   │   ├── classification_model.py    # Classification model definition
│   │   ├── datasets.py           # Detection dataset loading
│   │   ├── detection_model.py    # Detection model definition
│   │   ├── exceptions.py         # Training-related exceptions
│   │   ├── train_classification.py   # Classification training loop
│   │   └── train_detection.py    # Detection training loop
│   ├── inference/                # Model inference module (detection & classification)
│   │   ├── __init__.py           # Inference module exports
│   │   ├── detection.py          # Hold detection inference
│   │   └── classification.py     # Hold classification inference
│   └── archive/legacy/           # Archived Flask code (reference only)
│
├── tests/                        # Test suite
│   ├── __init__.py               # Test package
│   ├── conftest.py               # Pytest fixtures
│   ├── test_app.py               # Application tests
│   ├── test_config.py            # Configuration tests
│   ├── test_health.py            # Health endpoint tests
│   ├── test_logging_config.py    # Logging tests
│   ├── test_supabase_client.py   # Supabase client tests
│   ├── test_upload.py            # Upload endpoint tests
│   ├── test_classification_dataset.py  # Classification dataset tests
│   ├── test_classification_model.py    # Classification model tests
│   ├── test_datasets.py          # Detection dataset tests
│   ├── test_detection_model.py   # Detection model tests
│   ├── test_inference_detection.py    # Detection inference tests
│   ├── test_inference_classification.py # Classification inference tests
│   ├── test_train_classification.py    # Classification training tests
│   ├── test_train_detection.py   # Detection training tests
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
├── requirements.txt              # Pip dependencies (can be generated from pyproject.toml)
├── pyproject.toml                # Project metadata, dependencies, pytest & coverage config
├── uv.lock                       # Locked dependency versions (committed to git)
├── .python-version               # Python version specification for uv
├── setup_uv.sh                   # uv setup script (Unix/macOS)
├── setup_uv.ps1                  # uv setup script (Windows)
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
| `src/routes/upload.py` | Image upload endpoint | `/api/v1/routes/upload` |
| `src/database/supabase_client.py` | Supabase client | Connection pooling, storage ops |
| `src/training/__init__.py` | Training module exports | Classification & detection exports |
| `src/training/classification_dataset.py` | Classification dataset | Dataset loading, validation |
| `src/training/classification_model.py` | Classification model | ResNet-18/MobileNetV3 definition, apply_classifier_dropout |
| `src/training/datasets.py` | Detection dataset | YOLOv8 dataset loading |
| `src/training/detection_model.py` | Detection model | YOLOv8 model definition |
| `src/training/exceptions.py` | Training exceptions | Custom exception classes |
| `src/training/train_classification.py` | Classification training | Training loop & artifact management |
| `src/training/train_detection.py` | Detection training | Training loop & artifact management |
| `src/inference/__init__.py` | Inference module exports | Classification & detection exports |
| `src/inference/detection.py` | Detection inference | Hold detection from images |
| `src/inference/classification.py` | Classification inference | Hold type classification from crops |
| `tests/conftest.py` | Pytest fixtures | Test app, client, settings |
| `tests/test_supabase_client.py` | Database tests | Supabase client & storage tests |
| `tests/test_upload.py` | Upload tests | Image validation & upload tests |
| `tests/test_train_classification.py` | Classification training tests | Training loop tests |
| `tests/test_inference_detection.py` | Detection inference tests | Hold detection inference tests |
| `tests/test_inference_classification.py` | Classification inference tests | Hold type classification tests |
| `test_supabase_connection.py` | Connection test script | Verify Supabase setup |
| `.pre-commit-config.yaml` | Pre-commit hooks config | QA automation |
| `docs/DESIGN.md` | Architecture spec | Milestones, domain model |
| `docs/MODEL_PRETRAIN.md` | ML spec | Detection, classification |
| `docs/PRE_COMMIT_HOOKS.md` | Pre-commit guide | Setup, usage, troubleshooting |
| `docs/SUPABASE_SETUP.md` | Supabase setup guide | Step-by-step setup instructions |
| `docs/UV_SETUP.md` | uv setup guide | Fast dependency management with uv |
| `docs/FRONTEND_WORKFLOW.md` | Frontend development guide | Lovable → Claude Code → Vercel |
| `docs/VERCEL_SETUP.md` | Vercel deployment guide | Step-by-step deployment |
| `docs/TELEGRAM_BOT.md` | Telegram bot guide | Bot setup and implementation |
| `plans/MIGRATION_PLAN.md` | Migration roadmap | PR breakdown, phases |
| `setup_uv.sh` | uv setup script (Unix) | Quick setup with uv |
| `setup_uv.ps1` | uv setup script (Windows) | Quick setup with uv |
| `.python-version` | Python version | Version specification for uv |
| `uv.lock` | Dependency lockfile | Reproducible builds |

---

## Technology Stack

### Core Dependencies

```text
# Web Framework
fastapi==0.128.6
uvicorn[standard]==0.40.0
pydantic-settings==2.7.1

# Database & Storage
supabase==2.17.0

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
pytest==8.3.5
pytest-cov==6.0.0
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

**Recommended: Using uv (10-100x faster than pip)**

```bash
# Option 1: Quick setup with setup script
# macOS/Linux:
./setup_uv.sh

# Windows (PowerShell):
.\setup_uv.ps1

# Option 2: Manual setup
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# OR: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create venv and install dependencies (production + dev)
# Note: --no-install-project is used because this is an application, not a library
# --all-extras installs dev dependencies (testing, linting, etc.)
uv sync --no-install-project --all-extras

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR: .venv\Scripts\Activate.ps1  # Windows (PowerShell)

# Or run commands without activation
uv run pytest tests/
uv run uvicorn src.app:application --reload

# Configure Supabase (required for upload functionality)
# 1. Create a .env file in the project root
# 2. Add your Supabase credentials:
#    BA_SUPABASE_URL=https://your-project.supabase.co
#    BA_SUPABASE_KEY=your-anon-or-service-role-key
# See docs/SUPABASE_SETUP.md for detailed instructions

# Install pre-commit hooks (recommended)
uv run pre-commit install

# Verify installation
uv run python -c "from src.app import create_app; print('OK')"

# Test Supabase connection (optional)
uv run python test_supabase_connection.py
```

**Alternative: Traditional pip setup (slower but works)**

```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (recommended)
pre-commit install

# Configure Supabase (see above)
# Verify installation (see above)
```

**See [docs/UV_SETUP.md](docs/UV_SETUP.md) for comprehensive uv documentation.**

### Pre-commit Hooks (Recommended)

This project uses pre-commit hooks to automatically run QA checks before each commit.

```bash
# Install the git hooks (one-time setup)
pre-commit install

# Run checks manually on all files
pre-commit run --all-files

# Run checks on staged files only
pre-commit run
```

The hooks automatically check:

- Code formatting (ruff)
- Linting (ruff)
- Type checking (mypy)
- Tests with coverage (pytest)
- Code quality (pylint)

**See [docs/PRE_COMMIT_HOOKS.md](docs/PRE_COMMIT_HOOKS.md) for detailed usage and troubleshooting.**

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
mypy src/ tests/ && \
ruff check src/ tests/ --ignore E501 && \
ruff format --check src/ tests/ && \
pytest tests/ --cov=src --cov-fail-under=85 && \
pylint src/ --ignore=archive
```

#### Individual QA Commands

```bash
# Type checking
mypy src/ tests/

# Linting
ruff check src/ tests/ --ignore E501

# Format checking
ruff format --check src/ tests/

# Tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Code quality (≥8.5/10 required now, ≥9.0/10 when all features complete)
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

- **Coverage Target (Staged)**:
  - **Current Stage**: 85% minimum required
  - **Final Stage**: 90% minimum required (when all features complete)
- **Framework**: pytest with comprehensive fixtures
- **Isolation**: Each test uses fresh app instance
- **Current Actual Coverage**: 98%+

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_app.py -v

# Run with coverage threshold enforcement (current stage requirement)
pytest tests/ --cov=src --cov-fail-under=85

# Future: When all features complete, use --cov-fail-under=90
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
| `BA_SUPABASE_URL` | `""` | Supabase project URL (required) |
| `BA_SUPABASE_KEY` | `""` | Supabase API key (required) |
| `BA_MAX_UPLOAD_SIZE_MB` | `10` | Maximum file upload size in MB |
| `BA_STORAGE_BUCKET` | `route-images` | Supabase storage bucket name |
| `BA_ALLOWED_IMAGE_TYPES` | `["image/jpeg", "image/png"]` | Allowed image MIME types |

### Example .env File

```bash
BA_APP_NAME=bouldering-analysis
BA_APP_VERSION=0.1.0
BA_DEBUG=true
BA_LOG_LEVEL=DEBUG
BA_CORS_ORIGINS=["http://localhost:3000"]

# Supabase Configuration (required)
BA_SUPABASE_URL=https://your-project.supabase.co
BA_SUPABASE_KEY=your-anon-or-service-role-key

# Upload Configuration (optional)
BA_MAX_UPLOAD_SIZE_MB=10
BA_STORAGE_BUCKET=route-images
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

**IMPORTANT**: Quality targets are staged based on project completion:

- **Current Stage (Backend Foundation)**: 85% coverage, 8.5/10 pylint
- **Final Stage (All Features Complete)**: 90% coverage, 9.0/10 pylint

### Quality Gates

| Check | Tool | Current Stage (Now) | Final Stage (Complete) |
|-------|------|---------------------|------------------------|
| Type Safety | mypy | No errors | No errors |
| Linting | ruff check | No errors | No errors |
| Formatting | ruff format | All files formatted | All files formatted |
| Coverage | pytest-cov | **≥85% required** | **≥90% required** |
| Code Quality | pylint | **≥8.5/10 required** | **≥9.0/10 required** |

### Pre-Commit Checklist

Use these thresholds based on current project stage:

- [ ] All tests pass: `pytest tests/`
- [ ] Coverage ≥ 85% (current): `pytest tests/ --cov=src/ --cov-fail-under=85`
- [ ] Type checking passes: `mypy src/ tests/`
- [ ] Linting passes: `ruff check .`
- [ ] Formatting passes: `ruff format --check .`
- [ ] Pylint score ≥ 8.5 (current): `pylint src/`
- [ ] Agent reviews complete: python-reviewer, code-reviewer, security-reviewer (run in parallel)
- [ ] doc-updater run (CLAUDE.md, specs, docstrings)

**Note**: When all features are complete, thresholds increase to 90% coverage and 9.0/10 pylint.

---

## API Reference

### Current Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/health` | Health check (root level) | HealthResponse |
| GET | `/api/v1/health` | Health check (versioned) | HealthResponse |
| POST | `/api/v1/routes/upload` | Upload route image | UploadResponse |
| GET | `/docs` | Swagger UI (debug only) | HTML |
| GET | `/openapi.json` | OpenAPI schema (debug only) | JSON |

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

# Upload image
curl -X POST http://localhost:8000/api/v1/routes/upload \
  -F "file=@route.jpg"

# Response
{
    "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "public_url": "https://your-project.supabase.co/storage/v1/object/public/route-images/2026/01/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg",
    "file_size": 1048576,
    "content_type": "image/jpeg",
    "uploaded_at": "2026-01-26T12:00:00Z"
}
```

### Response Models

```python
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime

class UploadResponse(BaseModel):
    file_id: str
    public_url: str
    file_size: int
    content_type: str
    uploaded_at: str
```

---

## AI Assistant Guidelines

### When Making Changes

1. **Read before modifying**: Always read files before editing
2. **Check specifications**: Review docs/DESIGN.md and plans/MIGRATION_PLAN.md
3. **Follow conventions**: Match existing code style and patterns
4. **Write tests first**: Add tests for new functionality
5. **Maintain coverage**: Meet current stage requirement (≥85% now, will increase to ≥90%)
6. **Run QA suite**: Execute all quality checks before committing
7. **Update documentation**: Keep CLAUDE.md and specs current

### Agent Workflow (Mandatory)

Agent names below are logical roles. They can be invoked via the Task tool
using any compatible agent provider (e.g., the `everything-claude-code` plugin provides
these as `everything-claude-code:<agent-name>`), or via custom agent definitions in
`~/.claude/agents/` if available.

**Per-feature mandatory sequence:**

1. **planner** — before writing code (all PRs touching >1 file)
2. **tdd-guide** — after planning, before implementation (every new function/endpoint)
3. **database-reviewer** — when Supabase schema/SQL touched (PR-2.2, PR-9.x). If triggered, must complete before the parallel group (steps 4, 5, 6) runs.
4. **python-reviewer** — after writing .py files (type safety, pylint, immutability)
5. **code-reviewer** — after implementation (correctness, architecture alignment)
6. **security-reviewer** — before every commit, parallel with python-reviewer and code-reviewer
7. **doc-updater** — after clean code review (CLAUDE.md, specs, docstrings)

**Additional triggers:**
- At milestone completion: **e2e-runner**
- For design decisions: **architect**
- **Parallel rule**: Steps 4, 5, 6 (python-reviewer, code-reviewer, security-reviewer) run in parallel

### Code Review Checklist

- [ ] Type hints on all functions
- [ ] Google-style docstrings on all public functions
- [ ] Error handling for edge cases
- [ ] Tests added for new functionality
- [ ] Coverage meets current requirement: ≥85% (increases to ≥90% when all features complete)
- [ ] Pylint score meets current requirement: ≥8.5/10 (increases to ≥9.0/10 when all features complete)
- [ ] All QA checks pass
- [ ] No imports from archive directories

### Common Pitfalls to Avoid

1. **Don't import from archive/** - Use only for reference
2. **Don't bypass QA checks** - All code must pass
3. **Don't ignore type errors** - Fix mypy issues
4. **Don't skip tests** - Meet current coverage requirement (≥85% now, ≥90% final)
5. **Don't use print()** - Use logging module
6. **Don't commit .pt files** - Model weights are gitignored

### Key References

- **Design Spec**: [docs/DESIGN.md](docs/DESIGN.md)
- **Model Spec**: [docs/MODEL_PRETRAIN.md](docs/MODEL_PRETRAIN.md)
- **Pre-commit Guide**: [docs/PRE_COMMIT_HOOKS.md](docs/PRE_COMMIT_HOOKS.md)
- **Migration Plan**: [plans/MIGRATION_PLAN.md](plans/MIGRATION_PLAN.md)
- **PR Specs**: [plans/specs/](plans/specs/)

---

## Quick Reference

### Key Commands

**With uv (Recommended)**:

```bash
# Setup
uv sync --no-install-project --all-extras      # Install all dependencies (including dev)
uv sync --no-install-project --all-extras --upgrade  # Update dependencies

# Start server
uv run uvicorn src.app:application --reload

# Run tests
uv run pytest tests/ --cov=src --cov-report=term-missing

# Type checking
uv run mypy src/ tests/

# Linting
uv run ruff check src/ tests/ --ignore E501

# Format
uv run ruff format src/ tests/

# Pylint
uv run pylint src/ --ignore=archive
```

**Without uv (Traditional)**:

```bash
# Setup
pip install -r requirements.txt

# Start server
uvicorn src.app:application --reload

# Run tests
pytest tests/ --cov=src --cov-report=term-missing

# Type checking
mypy src/ tests/

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
| Upload endpoint | `src/routes/upload.py` |
| Supabase client | `src/database/supabase_client.py` |
| Test fixtures | `tests/conftest.py` |
| Supabase tests | `tests/test_supabase_client.py` |
| Upload tests | `tests/test_upload.py` |
| Connection test | `test_supabase_connection.py` |
| Design spec | `docs/DESIGN.md` |
| Supabase setup | `docs/SUPABASE_SETUP.md` |
| uv setup guide | `docs/UV_SETUP.md` |
| Frontend workflow | `docs/FRONTEND_WORKFLOW.md` |
| Vercel setup | `docs/VERCEL_SETUP.md` |
| Telegram bot guide | `docs/TELEGRAM_BOT.md` |
| Migration plan | `plans/MIGRATION_PLAN.md` |
| uv setup scripts | `setup_uv.sh`, `setup_uv.ps1` |

---

## Allowed Commands Whitelist

The following commands are pre-approved for AI assistants to execute without additional confirmation. These are safe, commonly-used development commands.

### Quality Assurance Commands

```bash
# UV Dependency Management
uv sync --no-install-project --all-extras  # Install/update all dependencies (including dev)
uv sync --no-install-project --all-extras --upgrade  # Update all dependencies
uv sync --no-install-project --no-dev   # Install production only
uv add <package>                     # Add dependency
uv add --dev <package>               # Add dev dependency
uv remove <package>                  # Remove dependency
uv lock                              # Update lockfile
uv pip list                          # List installed packages

# Type checking (with or without uv)
mypy src/
mypy tests/
mypy src/ tests/
uv run mypy src/ tests/

# Linting
ruff check .
ruff check src/
ruff check tests/
ruff check --fix .
uv run ruff check .
uv run ruff check --fix .

# Code formatting
ruff format .
ruff format --check .
ruff format src/
ruff format tests/
uv run ruff format .
uv run ruff format --check .

# Code quality
pylint src/
pylint src/ --output-format=colorized
uv run pylint src/

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
uv run pytest tests/
uv run pytest tests/ --cov=src/
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
pip show fastapi
pip freeze
```

### Project-Specific Scripts

```bash
# Development scripts
./run.sh
./run_setup_dev.sh
./run_qa.sh
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
