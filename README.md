# Bouldering Route Analysis

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688)
![Project Status](https://img.shields.io/badge/status-development-yellow)

[![Python QA](https://github.com/chuanseng-ng/bouldering-analysis/actions/workflows/pylint.yml/badge.svg?branch=main&event=push)](https://github.com/chuanseng-ng/bouldering-analysis/actions/workflows/pylint.yml)
[![Python Package](https://github.com/chuanseng-ng/bouldering-analysis/actions/workflows/python-package-uv.yml/badge.svg?branch=main&event=push)](https://github.com/chuanseng-ng/bouldering-analysis/actions/workflows/python-package-uv.yml)

A web-based system that estimates bouldering route difficulty (V-scale) from images using computer vision and machine learning.

## Overview

This application analyzes bouldering route images to:

- **Detect holds** using pretrained YOLOv8 models
- **Classify hold types** (jug, crimp, sloper, pinch, volume)
- **Construct movement graphs** from hold positions
- **Extract interpretable features** for grade estimation
- **Predict difficulty grades** with uncertainty and explanation

## Architecture

Built with a **backend-first, explainable AI** approach:

- **Backend**: FastAPI with Pydantic Settings
- **ML/CV**: PyTorch + Ultralytics YOLOv8
- **Database**: Supabase (Postgres + Storage)
- **Frontends**:
  - **Web**: React/Next.js (developed via Lovable, deployed on Vercel)
  - **Telegram Bot**: Python bot for quick, on-the-go analysis

### Frontend Options

The application provides **two frontend interfaces** to serve different use cases:

#### Web Frontend (Primary)

Full-featured interface for detailed route analysis:

1. **Lovable Prototype**: Rapid UI development using the [Lovable](https://lovable.dev) no-code platform
2. **Code Refinement**: Export to code, enhance with Claude Code, deploy to [Vercel](https://vercel.com)

**Features**: Interactive annotations, route history, detailed explanations, advanced visualizations

See [docs/FRONTEND_WORKFLOW.md](docs/FRONTEND_WORKFLOW.md) for the complete development workflow.

#### Telegram Bot (Alternative)

Lightweight interface for quick grade checks:

- Send route photo directly in Telegram chat
- Receive grade prediction instantly
- No installation or signup required
- Perfect for gym climbers on-the-go

**Features**: Photo upload, instant predictions, simple text interactions

See [docs/TELEGRAM_BOT.md](docs/TELEGRAM_BOT.md) for setup and usage guide.

## Deployment

- **Backend**: TBD (FastAPI deployment strategy to be determined)
- **Web Frontend**: Vercel with automatic Git deployments
- **Telegram Bot**: TBD (Serverless function or dedicated service)

See [docs/VERCEL_SETUP.md](docs/VERCEL_SETUP.md) for web frontend deployment and [docs/TELEGRAM_BOT.md](docs/TELEGRAM_BOT.md) for bot deployment.

### Database Implementation Status

**✅ Completed Features:**

- **Supabase Client** (`src/database/supabase_client.py`) - Connection pooling and storage operations
- **Image Upload** (`POST /api/v1/routes/upload`) - Validates and stores JPEG/PNG files
- **Storage Management** - Upload, delete, list files in Supabase Storage buckets
- **Hold Detection** (`src/inference/detection.py`) - YOLOv8-based hold detection with caching
- **Hold Classification** (`src/inference/classification.py`) - ResNet-18/MobileNetV3 hold type inference

**❌ Pending Features:**

- Database tables (`routes`, `holds`, `features`, `predictions`, `feedback`)
- Route record creation endpoint
- Route movement graph construction
- Grade prediction

See [CLAUDE.md - Database Implementation Status](CLAUDE.md#database-implementation-status) for detailed information.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Supabase account (for storage and database features)

### Installation

**Option 1: Quick Setup with uv (Recommended - 10-100x faster)**

```bash
# Clone repository
git clone https://github.com/yourusername/bouldering-analysis.git
cd bouldering-analysis

# Quick setup script (installs uv if needed)
./setup_uv.sh          # macOS/Linux
# OR: .\setup_uv.ps1   # Windows PowerShell

# Configure Supabase (required for upload functionality)
# Create a .env file in the project root:
cat > .env << EOF
BA_SUPABASE_URL=https://your-project.supabase.co
BA_SUPABASE_KEY=your-anon-or-service-role-key
EOF
# See docs/SUPABASE_SETUP.md for detailed setup instructions

# Install pre-commit hooks (recommended)
uv run pre-commit install

# Test Supabase connection (optional)
uv run python test_supabase_connection.py

# Start the server
uv run uvicorn src.app:application --reload
```

**Option 2: Traditional pip setup**

```bash
# Clone repository
git clone https://github.com/yourusername/bouldering-analysis.git
cd bouldering-analysis

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (recommended)
pre-commit install

# Configure Supabase (see above)

# Test Supabase connection (optional)
python test_supabase_connection.py

# Start the server
uvicorn src.app:application --reload
```

**Keeping requirements.txt in sync with pyproject.toml**:

The `requirements.txt` file is generated from `pyproject.toml` and should be regenerated when dependencies change:

```bash
# Regenerate requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt
```

> **Note**: Consider adding a CI/pre-commit check to ensure `requirements.txt` stays synchronized with `pyproject.toml`. Add this to `.pre-commit-config.yaml` to automatically fail the build if they're out of sync.
>
> 💡 **Why uv?** [uv](https://github.com/astral-sh/uv) is 10-100x faster than pip with automatic dependency locking for reproducible builds. See [docs/UV_SETUP.md](docs/UV_SETUP.md) for details.

The API will be available at [http://localhost:8000](http://localhost:8000)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/health` | GET | Versioned health check |
| `/api/v1/routes/upload` | POST | Upload route image (JPEG/PNG) |
| `/docs` | GET | Swagger UI (debug mode) |

#### Example: Upload an image**

```bash
curl -X POST http://localhost:8000/api/v1/routes/upload \
  -F "file=@route.jpg"
```

## Development

**Current Project Stage**: Hold Classification (Milestone 4 - Completed)
**Completed Milestones**:

- ✅ Milestone 1: Backend Foundation (FastAPI + Supabase Client)
- ✅ Milestone 2: Image Upload (Upload endpoint + route records)
- ✅ Milestone 3: Hold Detection (YOLOv8 training + inference)
- ✅ Milestone 4: Hold Classification (ResNet-18/MobileNetV3 training + inference)

**Quality Targets**: Coverage ≥85%, Pylint ≥8.5/10 (will increase to 90%/9.0 when all features complete)
**Current Coverage**: 97%

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Current requirement: 85% minimum coverage
# Final target (all features): 90% minimum coverage
```

### Quality Checks

#### Automated Pre-commit Hooks (Recommended)

```bash
# Run all QA checks automatically before each commit
pre-commit run --all-files

# Checks include: formatting, linting, type checking, tests, and code quality
# See docs/PRE_COMMIT_HOOKS.md for detailed guide
```

#### Manual QA Commands

```bash
# Type checking
mypy src/ tests/

# Linting
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Code quality
pylint src/ --ignore=archive
# Current requirement: 8.5/10 minimum
# Final target (all features): 9.0/10 minimum
```

### Environment Variables

#### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BA_APP_NAME` | `bouldering-analysis` | Application name |
| `BA_APP_VERSION` | `0.1.0` | Application version |
| `BA_DEBUG` | `false` | Enable debug mode |
| `BA_TESTING` | `false` | Enable testing mode |
| `BA_LOG_LEVEL` | `INFO` | Logging level |
| `BA_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON array) |

#### Database & Storage (Supabase)

| Variable | Default | Description |
|----------|---------|-------------|
| `BA_SUPABASE_URL` | `""` | Supabase project URL (required) |
| `BA_SUPABASE_KEY` | `""` | Supabase API key (required) |
| `BA_STORAGE_BUCKET` | `route-images` | Storage bucket name |
| `BA_MAX_UPLOAD_SIZE_MB` | `10` | Max upload size in MB |
| `BA_ALLOWED_IMAGE_TYPES` | `["image/jpeg", "image/png"]` | Allowed MIME types |

## Project Structure

```text
bouldering-analysis/
├── src/                          # Application code
│   ├── app.py                    # FastAPI application factory
│   ├── config.py                 # Pydantic Settings configuration
│   ├── logging_config.py         # Structured JSON logging
│   ├── routes/                   # API route modules
│   │   ├── health.py             # Health check endpoint
│   │   └── upload.py             # Image upload endpoint
│   ├── database/                 # Database layer (Supabase)
│   │   └── supabase_client.py    # Supabase client & storage
│   ├── training/                 # Model training (detection & classification)
│   │   ├── classification_model.py   # ResNet-18/MobileNetV3 definition
│   │   ├── train_classification.py   # Classification training loop
│   │   ├── detection_model.py    # YOLOv8 model definition
│   │   └── train_detection.py    # Detection training loop
│   ├── inference/                # Model inference
│   │   ├── classification.py     # Hold type classification from crops
│   │   └── detection.py          # Hold detection from images
│   └── archive/legacy/           # Legacy code (reference)
├── tests/                        # Test suite (97% coverage)
│   ├── test_app.py               # Application tests
│   ├── test_config.py            # Configuration tests
│   ├── test_health.py            # Health endpoint tests
│   ├── test_routes.py            # Route module tests
│   ├── test_supabase_client.py   # Supabase client tests
│   ├── test_upload.py            # Upload endpoint tests
│   ├── test_logging_config.py    # Logging config tests
│   ├── test_classification_dataset.py    # Classification dataset tests
│   ├── test_classification_model.py      # Classification model tests
│   ├── test_detection_model.py           # Detection model tests
│   ├── test_training_datasets.py         # Training dataset tests
│   ├── test_train_classification.py      # Classification training tests
│   ├── test_train_detection.py           # Detection training tests
│   ├── test_inference_detection.py       # Detection inference tests
│   ├── test_inference_classification.py  # Classification inference tests
│   ├── test_inference_crop_extractor.py  # Crop extractor inference tests
│   └── conftest.py               # Pytest fixtures
├── docs/                         # Documentation
│   ├── DESIGN.md                 # Architecture spec
│   ├── MODEL_PRETRAIN.md         # ML spec
│   ├── SUPABASE_SETUP.md         # Database setup guide
│   ├── FRONTEND_WORKFLOW.md      # Web frontend development guide
│   ├── VERCEL_SETUP.md           # Vercel deployment guide
│   ├── TELEGRAM_BOT.md           # Telegram bot guide
│   └── PRE_COMMIT_HOOKS.md       # QA automation guide
├── plans/                        # Implementation plans
│   └── MIGRATION_PLAN.md         # Migration roadmap
├── test_supabase_connection.py   # Connection test script
└── CLAUDE.md                     # AI assistant guide
```

## Documentation

- [Design Specification](docs/DESIGN.md) - Architecture and milestones
- [FastAPI Role Explained](docs/FASTAPI_ROLE.md) - What FastAPI does in this app
- [Model Pretraining](docs/MODEL_PRETRAIN.md) - ML model specifications
- [Pre-commit Hooks Guide](docs/PRE_COMMIT_HOOKS.md) - QA automation setup and usage
- [Supabase Setup Guide](docs/SUPABASE_SETUP.md) - Database configuration
- [Web Frontend Workflow](docs/FRONTEND_WORKFLOW.md) - Web UI development guide (Lovable → Claude Code → Vercel)
- [Vercel Setup Guide](docs/VERCEL_SETUP.md) - Web frontend deployment to Vercel
- [Telegram Bot Guide](docs/TELEGRAM_BOT.md) - Telegram bot setup and usage
- [Migration Plan](plans/MIGRATION_PLAN.md) - Implementation roadmap
- [AI Assistant Guide](CLAUDE.md) - Development guidelines

## References

This project draws inspiration from:

- [Bouldering and Computer Vision Blog Post](https://blog.tjtl.io/bouldering-and-computer-vision/)
- [Learn to Climb by Seeing: Climbing Grade Classification with Computer Vision (Stanford CS231N 2024)](https://cs231n.stanford.edu/2024/papers/learn-to-climb-by-seeing-climbing-grade-classification-with-comp.pdf)
- [Rock Climbing Coach PDF](https://kastner.ucsd.edu/ryan/wp-content/uploads/sites/5/2022/06/admin/rock-climbing-coach.pdf)
- [ScienceDirect: Bouldering Route Analysis Article](https://www.sciencedirect.com/science/article/pii/S0952197624015173)
- [Climbing Grade Predictions GitHub Repository](https://github.com/Tetleysteabags/climbing-grade-predictions/tree/main)

## License

MIT License
