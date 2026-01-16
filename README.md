# Bouldering Route Analysis

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![Code Quality](https://img.shields.io/badge/pylint-10.0%2F10-brightgreen)
![Type Checking](https://img.shields.io/badge/mypy-passing-success)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688)
![Project Status](https://img.shields.io/badge/status-development-yellow)

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
- **Frontend**: Lovable (external)

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bouldering-analysis.git
cd bouldering-analysis

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.app:application --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/v1/health` | Versioned health check |
| `GET /docs` | Swagger UI (debug mode) |

## Development

**Current Project Stage**: Backend Foundation (Milestone 1)
**Quality Targets**: Coverage ≥85%, Pylint ≥8.5/10 (will increase to 90%/9.0 when all features complete)

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Current requirement: 85% minimum coverage
# Final target (all features): 90% minimum coverage
```

### Quality Checks

```bash
# Type checking
mypy src/ tests/ --ignore-missing-imports

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

| Variable | Default | Description |
|----------|---------|-------------|
| `BA_DEBUG` | `false` | Enable debug mode |
| `BA_LOG_LEVEL` | `INFO` | Logging level |
| `BA_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

## Project Structure

```text
bouldering-analysis/
├── src/                    # Application code
│   ├── app.py              # FastAPI application
│   ├── config.py           # Configuration
│   ├── routes/             # API routes
│   └── archive/            # Legacy code (reference)
├── tests/                  # Test suite
├── docs/                   # Documentation
│   ├── DESIGN.md           # Architecture spec
│   └── MODEL_PRETRAIN.md   # ML spec
├── plans/                  # Implementation plans
└── CLAUDE.md               # AI assistant guide
```

## Documentation

- [Design Specification](docs/DESIGN.md) - Architecture and milestones
- [FastAPI Role Explained](docs/FASTAPI_ROLE.md) - What FastAPI does in this app
- [Model Pretraining](docs/MODEL_PRETRAIN.md) - ML model specifications
- [Supabase Setup Guide](docs/SUPABASE_SETUP.md) - Database configuration
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
