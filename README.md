# Bouldering Route Analysis

A Flask-based web application that uses computer vision (YOLOv8) to analyze bouldering route images, detect climbing holds, and predict difficulty grades.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.1.2](https://img.shields.io/badge/flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3.233-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)](pytest-coverage.txt)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Quality Assurance](#quality-assurance)
  - [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

## Overview

This application leverages deep learning and computer vision to automatically analyze bouldering routes from images. It detects climbing holds, classifies them into types (crimp, jug, sloper, etc.), and predicts route difficulty grades on the V-scale.

### What It Does

1. **Hold Detection**: Uses fine-tuned YOLOv8 models to detect and classify climbing holds in uploaded images
2. **Grade Prediction**: Analyzes hold distribution and features to predict bouldering route grades (V0-V10+)
3. **Feedback Collection**: Stores user corrections and feedback for continuous model improvement
4. **Model Management**: Tracks multiple model versions with activation/deactivation capabilities
5. **Analytics**: Provides usage statistics and model performance metrics

---

## Features

### Core Functionality

- **Automated Hold Detection**: Detects 8 types of climbing holds:
  - Crimp
  - Jug
  - Sloper
  - Pinch
  - Pocket
  - Foot-hold
  - Start-hold
  - Top-out-hold

- **Route Grading**: Predicts bouldering difficulty grades based on:
  - Hold count and distribution
  - Hold type analysis
  - Spatial arrangement
  - Confidence scoring

- **User Feedback System**:
  - Submit grade corrections
  - Add comments and observations
  - Track prediction accuracy

- **Model Versioning**:
  - Track multiple model versions
  - A/B testing capabilities
  - Performance comparison
  - Easy model activation/deactivation

### Web Interface

- Clean, intuitive UI for image upload
- Drag-and-drop support
- Real-time analysis results
- Visual hold detection overlay
- Interactive feedback submission
- Mobile-responsive design

### REST API

- `/analyze` - Upload and analyze route images
- `/feedback` - Submit user corrections
- `/stats` - View usage statistics
- `/health` - Health check endpoint
- Full API documentation available

---

## Architecture

```text
┌─────────────┐
│    User     │
└──────┬──────┘
       │ Upload Image
       ▼
┌─────────────────────────┐
│   Flask Backend         │
│   (src/main.py)         │
└───────┬─────────────────┘
        │
        ├──► Store Image (data/uploads/)
        │
        ├──► Load Active Model
        │    ┌──────────────────┐
        │    │ Fine-tuned       │
        └───►│ YOLOv8 Model     │
             └────────┬─────────┘
                      │ Inference
                      ▼
             ┌─────────────────┐
             │ Detected Holds  │
             │ + Features      │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │   Database      │
             │   (SQLite/      │
             │   PostgreSQL)   │
             └─────────────────┘
```

---

## Getting Started

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **GPU** (recommended): CUDA-compatible GPU for model training
  - Training: RTX 3080 or better recommended
  - Inference: GTX 1060 or better
- **Storage**: ~2GB for dependencies, ~1GB for trained models

### Installation

#### Option 1: Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/chuanseng-ng/bouldering-analysis.git
cd bouldering-analysis

# Run automated setup
./run_setup_dev.sh
```

#### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize development environment
python src/setup_dev.py
```

#### Option 3: Conda Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate bouldering-analysis
```

### Running the Application

```bash
# Start Flask development server
./run.sh

# Or manually
cd src && python main.py
```

The application will be available at: **<http://localhost:5000>**

---

## Usage

### Web Interface

1. **Open your browser** to `http://localhost:5000`
2. **Upload an image** by clicking "Choose File" or dragging and dropping
3. **View analysis results**:
   - Predicted grade (e.g., V5)
   - Confidence score
   - Detected holds with bounding boxes
   - Hold type distribution
4. **Submit feedback** if the prediction needs correction

### API Endpoints

#### 1. Analyze Route Image

```bash
POST /analyze
```

**Request:**

```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@route.jpg"
```

**Response:**

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "predicted_grade": "V5",
  "confidence_score": 0.87,
  "image_url": "/uploads/route.jpg",
  "holds": [
    {
      "hold_type": "crimp",
      "confidence": 0.92,
      "bbox": {
        "x1": 100,
        "y1": 50,
        "x2": 150,
        "y2": 100
      }
    }
  ],
  "features": {
    "total_holds": 12,
    "hold_types": {
      "crimp": 5,
      "jug": 3,
      "sloper": 2,
      "pinch": 2
    },
    "average_confidence": 0.87
  }
}
```

#### 2. Submit Feedback

```bash
POST /feedback
```

**Request:**

```bash
curl -X POST http://localhost:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_grade": "V6",
    "is_accurate": false,
    "comments": "Felt harder than V5 due to small crimps"
  }'
```

#### 3. Get Statistics

```bash
GET /stats
```

**Response:**

```json
{
  "total_analyses": 1247,
  "total_feedback": 523,
  "average_confidence": 0.85,
  "accuracy_rate": 0.78,
  "grade_distribution": {
    "V0": 45,
    "V1": 89,
    "V2": 156
  }
}
```

#### 4. Health Check

```bash
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2026-01-05T12:34:56Z",
  "model_loaded": true,
  "database_connected": true
}
```

---

## Model Training

### Training a New Hold Detection Model

```bash
python src/train_model.py \
  --model-name hold_detection_v2 \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --data-yaml data/sample_hold/data.yaml \
  --base-weights yolov8n.pt \
  --activate
```

**Parameters:**

- `--model-name`: Name for the new model version
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--data-yaml`: Path to YOLO dataset configuration
- `--base-weights`: Base model weights (default: yolov8n.pt)
- `--activate`: Automatically activate the model after training

### Managing Model Versions

```bash
# List all model versions
python src/manage_models.py list

# Activate a specific version
python src/manage_models.py activate hold_detection 2

# Deactivate a version
python src/manage_models.py deactivate hold_detection 1
```

### Dataset Preparation

Datasets should follow the YOLO format:

```text
data/sample_hold/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

**data.yaml format:**

```yaml
train: data/sample_hold/train/images
val: data/sample_hold/val/images

nc: 8  # number of classes
names: ['crimp', 'jug', 'sloper', 'pinch', 'pocket', 'foot-hold', 'start-hold', 'top-out-hold']
```

---

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src/ --cov-report=term-missing

# Run specific test file
pytest tests/test_main.py -v

# Generate HTML coverage report
pytest tests/ --cov=src/ --cov-report=html
# Open htmlcov/index.html in browser
```

**Coverage Requirement**: 99% minimum (enforced by CI/CD)

### Quality Assurance

Run the comprehensive QA suite before committing:

```bash
./run_qa.csh
```

This script runs:

1. **Type Checking** (mypy)
2. **Code Formatting** (ruff format --check)
3. **Linting** (ruff check)
4. **Tests with Coverage** (pytest with 99% requirement)
5. **Code Quality** (pylint with 9.9/10 minimum score)

#### Individual QA Commands

```bash
# Type checking
mypy src/ tests/

# Check formatting (does not modify files)
ruff format --check .

# Linting
ruff check .

# Auto-fix safe linting issues
ruff check --fix .

# Code quality
pylint src/
```

### Project Structure

```text
bouldering-analysis/
├── src/                          # Main application code
│   ├── cfg/                      # Configuration files
│   │   └── user_config.yaml      # App settings
│   ├── templates/                # HTML templates
│   │   └── index.html            # Web UI
│   ├── main.py                   # Flask app + routes (775 lines)
│   ├── config.py                 # Configuration loader (323 lines)
│   ├── models.py                 # SQLAlchemy ORM models (265 lines)
│   ├── constants.py              # Shared constants
│   ├── train_model.py            # YOLOv8 training pipeline (752 lines)
│   ├── manage_models.py          # Model version management (630 lines)
│   ├── setup.py                  # Database initialization
│   └── setup_dev.py              # Dev environment setup
│
├── tests/                        # Test suite (~4,567 lines)
│   ├── conftest.py               # pytest fixtures
│   ├── test_main.py              # Flask app tests
│   ├── test_config.py            # Configuration tests
│   ├── test_models.py            # Database model tests
│   └── test_train_model.py       # Training pipeline tests
│
├── data/                         # Data directory
│   ├── sample_hold/              # Sample YOLO dataset
│   └── uploads/                  # Uploaded images (gitignored)
│
├── models/                       # Trained models (gitignored)
│   └── hold_detection/           # Hold detection model versions
│
├── docs/                         # Documentation
│   ├── migrations.md             # Database migration guide
│   ├── week3-4_implementation.md # Implementation details
│   └── week3-4_validation.md     # Validation docs
│
├── scripts/                      # Utility scripts
│   └── migrations/               # Database migration scripts
│
├── .github/workflows/            # CI/CD pipelines
│   ├── pylint.yml                # Main QA pipeline
│   ├── python-code-cov.yml       # Coverage testing
│   └── python-package-conda.yml  # Conda build
│
├── CLAUDE.md                     # AI assistant guide
├── PROGRESS.md                   # Project roadmap
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── run.sh                        # Start application
├── run_setup_dev.sh              # Setup script
└── run_qa.csh                    # QA automation
```

---

## Configuration

### Configuration Files

**Primary Config**: `src/cfg/user_config.yaml`

```yaml
model_defaults:
  hold_detection_confidence_threshold: 0.25

model_paths:
  base_yolov8: 'yolov8n.pt'
  fine_tuned_models: 'models/hold_detection/'

data_paths:
  hold_dataset: 'data/sample_hold/'
  uploads: 'data/uploads/'
```

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///bouldering_analysis.db  # Default
# DATABASE_URL=postgresql://user:pass@localhost/bouldering  # Production

# Flask
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-here

# Upload
MAX_CONTENT_LENGTH=16777216  # 16MB

# Proxy (for production deployment)
ENABLE_PROXY_FIX=false
PROXY_FIX_X_FOR=1
PROXY_FIX_X_PROTO=1
```

---

## Database Schema

### Core Models

#### Analysis

Stores image analysis results with predicted grades and features.

```python
id: UUID (Primary Key)
image_filename: str
predicted_grade: str  # e.g., 'V5'
confidence_score: float
features_extracted: JSON
created_at: datetime
updated_at: datetime
```

#### DetectedHold

Individual hold detections with bounding boxes.

```python
id: int (Primary Key)
analysis_id: UUID (Foreign Key -> Analysis)
hold_type_id: int (Foreign Key -> HoldType)
confidence: float
bbox_x1, bbox_y1, bbox_x2, bbox_y2: float
```

#### HoldType

Reference table for hold classifications (8 types).

```python
id: int (Primary Key)
name: str  # 'crimp', 'jug', 'sloper', etc.
description: str
```

#### Feedback

User feedback on predictions.

```python
id: UUID (Primary Key)
analysis_id: UUID (Foreign Key -> Analysis)
user_grade: str
is_accurate: bool
comments: str (optional)
created_at: datetime
```

#### ModelVersion

ML model tracking and versioning.

```python
id: int (Primary Key)
model_type: str  # 'hold_detection'
version: int
model_path: str
accuracy: float (optional)
is_active: bool
created_at: datetime
```

#### UserSession

Analytics and session tracking.

```python
id: UUID (Primary Key)
session_id: str
ip_address: str
user_agent: str
created_at: datetime
last_activity: datetime
```

---

## Future Work

### Phase 1: Enhanced Data Management

- [ ] Expand dataset with diverse bouldering images
- [ ] Implement automated data annotation pipeline
- [ ] Add data augmentation for training robustness
- [ ] Integrate DVC for dataset versioning
- [ ] Implement active learning from user feedback

### Phase 2: Improved Model Architecture

- [ ] Develop advanced route grading algorithm
- [ ] Add uncertainty quantification for predictions
- [ ] Implement ensemble methods for better accuracy
- [ ] Add support for outdoor route analysis
- [ ] Integrate wall angle and hold size detection

### Phase 3: Continuous Learning System

- [ ] Automated model retraining pipeline
- [ ] A/B testing framework for model comparison
- [ ] Performance monitoring and alerting
- [ ] Quality control for user-submitted data
- [ ] Feedback-driven model improvement

### Phase 4: Production Deployment

- [ ] Containerization with Docker
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Scalable inference infrastructure
- [ ] CDN integration for image serving
- [ ] API rate limiting and authentication

### Phase 5: Advanced Features

- [ ] Multi-route analysis in single image
- [ ] Video analysis for dynamic moves
- [ ] 3D route reconstruction
- [ ] Personalized difficulty recommendations
- [ ] Community features and route sharing

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Checklist

Before submitting a pull request:

- [ ] All tests pass: `pytest tests/`
- [ ] Coverage ≥ 99%: `pytest tests/ --cov=src/`
- [ ] Type checking passes: `mypy src/ tests/`
- [ ] Linting passes: `ruff check .`
- [ ] Formatting passes: `ruff format --check .`
- [ ] Pylint score ≥ 9.9: `pylint src/`
- [ ] Documentation updated (if applicable)
- [ ] CLAUDE.md updated (if applicable)

**Or simply run**: `./run_qa.csh`

---

## References

This project draws inspiration from:

- [Bouldering and Computer Vision Blog Post](https://blog.tjtl.io/bouldering-and-computer-vision/)
- [Learn to Climb by Seeing: Climbing Grade Classification with Computer Vision (Stanford CS231N 2024)](https://cs231n.stanford.edu/2024/papers/learn-to-climb-by-seeing-climbing-grade-classification-with-comp.pdf)
- [Rock Climbing Coach PDF](https://kastner.ucsd.edu/ryan/wp-content/uploads/sites/5/2022/06/admin/rock-climbing-coach.pdf)
- [ScienceDirect: Bouldering Route Analysis Article](https://www.sciencedirect.com/science/article/pii/S0952197624015173)
- [Climbing Grade Predictions GitHub Repository](https://github.com/Tetleysteabags/climbing-grade-predictions/tree/main)

### Technologies

- [Flask](https://flask.palletsprojects.com/) - Web framework
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Ultralytics for the YOLOv8 framework
- The climbing community for inspiration and feedback
- Contributors and testers

---

**For detailed development guidelines, see [CLAUDE.md](CLAUDE.md)**

**For project roadmap and progress, see [PROGRESS.md](PROGRESS.md)**
