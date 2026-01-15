# FastAPI's Role in Bouldering Route Analysis

This document explains what FastAPI is used for in this application and how it orchestrates the entire route analysis workflow.

---

## Table of Contents

1. [Overview](#overview)
2. [The Big Picture](#the-big-picture)
3. [Current State vs Future State](#current-state-vs-future-state)
4. [API Endpoints (Planned)](#api-endpoints-planned)
5. [Why FastAPI?](#why-fastapi)
6. [Complete User Flow](#complete-user-flow)
7. [Technical Architecture](#technical-architecture)

---

## Overview

**FastAPI is the backend API server** that:
- Receives requests from the frontend (web interface)
- Orchestrates all the ML/CV processing
- Manages data persistence to Supabase
- Returns results back to the frontend

Think of it as the **"brain" of the application** - it coordinates all the different components (ML models, database, storage) to analyze a bouldering route from start to finish.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Web Browser)                        │
│                      ↓                    ↑                      │
│                   Upload                Return                   │
│                   Image              Grade + Explanation         │
└─────────────────────────────────────────────────────────────────┘
                          ↓                    ↑
                          │                    │
┌─────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  API ENDPOINTS                                              │ │
│  │  • POST /api/v1/routes/upload    (upload route image)      │ │
│  │  • POST /api/v1/routes/analyze   (analyze route)           │ │
│  │  • GET  /api/v1/routes/{id}      (get route details)       │ │
│  │  • POST /api/v1/feedback         (submit feedback)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  BUSINESS LOGIC                                             │ │
│  │  1. Validate input                                          │ │
│  │  2. Call ML models (hold detection, classification)         │ │
│  │  3. Build route graph                                       │ │
│  │  4. Extract features                                        │ │
│  │  5. Estimate grade with uncertainty                         │ │
│  │  6. Generate explanation                                    │ │
│  │  7. Store results in database                               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          ↓                    ↑
                    Call Models          Get/Store Data
                          ↓                    ↑
    ┌─────────────────────────────────────────────────────────────┐
    │                  ML/CV COMPONENTS                            │
    │  • Hold Detection Model (YOLOv8)                            │
    │  • Hold Classification Model (ResNet)                       │
    │  • Route Graph Builder                                      │
    │  • Feature Extractor                                        │
    │  • Grade Estimation Model                                   │
    └─────────────────────────────────────────────────────────────┘
                          ↓                    ↑
                          ↓                    ↑
    ┌─────────────────────────────────────────────────────────────┐
    │                    SUPABASE                                  │
    │  • PostgreSQL Database (route data, predictions)            │
    │  • Storage (route images, model outputs)                    │
    └─────────────────────────────────────────────────────────────┘
```

---

## Current State vs Future State

### ✅ Current State (Milestone 1 - Backend Foundation)

**What's Implemented:**
- ✅ FastAPI application with health check
- ✅ Configuration management (environment variables)
- ✅ Structured logging
- ✅ Supabase client for database and storage
- ✅ CORS middleware for frontend communication
- ✅ API versioning structure (`/api/v1/...`)

**Current Endpoints:**
```
GET  /health              - Health check
GET  /api/v1/health       - Versioned health check
GET  /docs                - API documentation (debug mode)
```

**What You Can Do Now:**
```python
# The backend is running and ready
uvicorn src.app:application --reload

# You can access:
# http://localhost:8000/health       → {"status": "healthy"}
# http://localhost:8000/docs         → Interactive API docs
```

### 🚧 Future State (Milestones 2-10)

**What Will Be Added:**

#### Milestone 2: Image Upload & Persistence
```python
POST /api/v1/routes/upload
{
    "file": <image_file>,
    "wall_angle": 15.0,
    "gym_name": "Planet Granite"
}
→ Returns: {"route_id": "uuid", "image_url": "https://..."}
```

#### Milestone 3-4: Hold Detection & Classification
```python
POST /api/v1/routes/{route_id}/detect-holds
→ Runs ML models to find and classify holds
→ Returns: {
    "holds": [
        {"id": 1, "x": 0.3, "y": 0.5, "type": "crimp", "confidence": 0.92},
        {"id": 2, "x": 0.5, "y": 0.3, "type": "jug", "confidence": 0.98}
    ]
}
```

#### Milestone 5-6: Route Graph & Features
```python
POST /api/v1/routes/{route_id}/build-graph
{
    "start_hold_ids": [1],
    "finish_hold_id": 15
}
→ Builds movement graph and extracts features
→ Returns: {
    "graph": {...},
    "features": {
        "max_reach": 1.8,
        "crimp_count": 5,
        "vertical_distance": 4.2
    }
}
```

#### Milestone 7-8: Grade Estimation & Explanation
```python
POST /api/v1/routes/{route_id}/estimate-grade
→ Predicts route difficulty with explanation
→ Returns: {
    "grade": "V5",
    "confidence": 0.73,
    "uncertainty": "±1 grade",
    "explanation": {
        "difficulty_factors": [
            "High crimp count (5 crimps)",
            "Long reach moves (max 1.8m)",
            "Steep wall angle (45°)"
        ],
        "similar_routes": [...]
    }
}
```

#### Milestone 9: Complete Route Analysis
```python
GET /api/v1/routes/{route_id}
→ Returns complete route analysis:
{
    "id": "uuid",
    "image_url": "https://...",
    "holds": [...],
    "graph": {...},
    "grade_estimate": "V5",
    "confidence": 0.73,
    "explanation": {...},
    "created_at": "2026-01-15T..."
}
```

---

## API Endpoints (Planned)

Here's the complete API that will be built:

### Routes Management

| Method | Endpoint | Description | Milestone |
|--------|----------|-------------|-----------|
| POST | `/api/v1/routes/upload` | Upload route image | M2 |
| GET | `/api/v1/routes/{id}` | Get route details | M2 |
| GET | `/api/v1/routes` | List all routes | M2 |
| DELETE | `/api/v1/routes/{id}` | Delete route | M2 |

### Analysis Pipeline

| Method | Endpoint | Description | Milestone |
|--------|----------|-------------|-----------|
| POST | `/api/v1/routes/{id}/detect-holds` | Detect holds in image | M3 |
| POST | `/api/v1/routes/{id}/classify-holds` | Classify detected holds | M4 |
| POST | `/api/v1/routes/{id}/build-graph` | Build movement graph | M5 |
| POST | `/api/v1/routes/{id}/extract-features` | Extract route features | M6 |
| POST | `/api/v1/routes/{id}/estimate-grade` | Estimate difficulty grade | M7 |
| POST | `/api/v1/routes/{id}/explain` | Generate explanation | M8 |

### One-Shot Analysis

| Method | Endpoint | Description | Milestone |
|--------|----------|-------------|-----------|
| POST | `/api/v1/routes/analyze` | Full pipeline in one call | M9 |

### Feedback & Learning

| Method | Endpoint | Description | Milestone |
|--------|----------|-------------|-----------|
| POST | `/api/v1/feedback` | Submit grade feedback | M9 |
| GET | `/api/v1/feedback/{route_id}` | Get feedback history | M9 |

---

## Why FastAPI?

FastAPI was chosen for this project because:

### 1. **Async Support for ML Models**
```python
@app.post("/api/v1/routes/{route_id}/detect-holds")
async def detect_holds(route_id: str):
    # Can run expensive ML inference without blocking other requests
    holds = await run_hold_detection(route_id)
    return holds
```

### 2. **Automatic API Documentation**
```python
# FastAPI automatically generates interactive docs
# Visit: http://localhost:8000/docs
# You can test all endpoints directly in the browser!
```

### 3. **Type Safety with Pydantic**
```python
class RouteUploadRequest(BaseModel):
    wall_angle: float = Field(ge=0, le=90, description="Wall angle in degrees")
    gym_name: str = Field(min_length=1, max_length=100)

@app.post("/api/v1/routes/upload")
def upload_route(request: RouteUploadRequest):
    # Request is automatically validated
    # Invalid data returns clear error messages
    pass
```

### 4. **High Performance**
- FastAPI is one of the fastest Python frameworks
- Built on Starlette (async) and Pydantic (validation)
- Can handle many concurrent ML inference requests

### 5. **Modern Python Features**
- Python 3.10+ type hints
- Async/await syntax
- Dependency injection
- Background tasks

---

## Complete User Flow

Here's how a complete route analysis works:

### Step 1: User Uploads Route Image

**Frontend:**
```javascript
// User selects image and clicks "Analyze Route"
const formData = new FormData();
formData.append('file', imageFile);
formData.append('wall_angle', 45);

fetch('http://localhost:8000/api/v1/routes/analyze', {
    method: 'POST',
    body: formData
})
```

**FastAPI Backend:**
```python
@app.post("/api/v1/routes/analyze")
async def analyze_route(
    file: UploadFile,
    wall_angle: float,
    start_hold_ids: list[int],
    finish_hold_id: int
):
    # 1. Validate image
    validate_image(file)

    # 2. Upload to Supabase Storage
    image_url = await upload_to_storage(
        bucket="route-images",
        file_path=f"routes/{uuid.uuid4()}.jpg",
        file_data=await file.read()
    )

    # 3. Create route record in database
    route = await create_route_record({
        "image_url": image_url,
        "wall_angle": wall_angle
    })

    # 4. Run hold detection (ML model)
    holds = await detect_holds(image_url)

    # 5. Classify holds (ML model)
    classified_holds = await classify_holds(holds)

    # 6. Build route graph
    graph = build_route_graph(
        classified_holds,
        wall_angle,
        start_hold_ids,
        finish_hold_id
    )

    # 7. Extract features
    features = extract_features(graph, classified_holds)

    # 8. Estimate grade (ML model)
    grade_prediction = estimate_grade(features)

    # 9. Generate explanation
    explanation = generate_explanation(
        features,
        grade_prediction,
        classified_holds
    )

    # 10. Save everything to database
    await save_analysis_results(route.id, {
        "holds": classified_holds,
        "graph": graph,
        "features": features,
        "grade": grade_prediction,
        "explanation": explanation
    })

    # 11. Return results to frontend
    return {
        "route_id": route.id,
        "grade": grade_prediction.grade,
        "confidence": grade_prediction.confidence,
        "explanation": explanation,
        "holds": classified_holds
    }
```

### Step 2: Frontend Displays Results

**What the user sees:**
```
┌─────────────────────────────────────────────┐
│  Your Route Analysis                        │
│                                             │
│  [Image with holds highlighted]             │
│                                             │
│  Estimated Grade: V5 (±1)                   │
│  Confidence: 73%                            │
│                                             │
│  Why this grade?                            │
│  • High crimp count (5 crimps)              │
│  • Long reach moves (max 1.8m)              │
│  • Steep wall angle (45°)                   │
│                                             │
│  Similar routes: [...]                      │
│                                             │
│  [Does this seem right?]                    │
│  [Too Hard] [Just Right] [Too Easy]         │
└─────────────────────────────────────────────┘
```

### Step 3: User Provides Feedback

**Frontend:**
```javascript
// User clicks "Too Hard" - actual grade is V4
fetch('http://localhost:8000/api/v1/feedback', {
    method: 'POST',
    body: JSON.stringify({
        route_id: "uuid",
        predicted_grade: "V5",
        actual_grade: "V4",
        feedback_type: "too_hard"
    })
})
```

**FastAPI Backend:**
```python
@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    # Store feedback for future model improvement
    await save_feedback(feedback)

    # Update route with actual grade
    await update_route_grade(
        feedback.route_id,
        feedback.actual_grade
    )

    return {"message": "Thanks for your feedback!"}
```

---

## Technical Architecture

### How FastAPI Orchestrates Everything

```python
# src/app.py - The Main Application

from fastapi import FastAPI
from src.routes import health, routes, analysis
from src.database import get_supabase_client
from src.ml import load_models

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Bouldering Route Analysis API",
        description="Estimate route difficulty using computer vision",
        version="0.1.0"
    )

    # Configure middleware
    app.add_middleware(CORSMiddleware, ...)
    app.add_middleware(RequestIDMiddleware, ...)

    # Register route handlers
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(routes.router, prefix="/api/v1/routes")
    app.include_router(analysis.router, prefix="/api/v1/analysis")

    # Load ML models on startup
    @app.on_event("startup")
    async def startup():
        app.state.hold_detector = load_hold_detector()
        app.state.hold_classifier = load_hold_classifier()
        app.state.grade_estimator = load_grade_estimator()
        app.state.db = get_supabase_client()

    return app
```

### Request Flow Through FastAPI

```
1. HTTP Request arrives
   ↓
2. CORS Middleware (allow frontend origin)
   ↓
3. Request ID Middleware (add tracking ID)
   ↓
4. Route Handler (e.g., /api/v1/routes/analyze)
   ↓
5. Pydantic Validation (validate request body)
   ↓
6. Business Logic
   ├─→ Call ML models
   ├─→ Process results
   └─→ Store in Supabase
   ↓
7. Response Model (serialize to JSON)
   ↓
8. HTTP Response sent to frontend
```

---

## Summary

**FastAPI is the orchestration layer** that:

1. **Receives requests** from the web frontend
2. **Validates input** using Pydantic models
3. **Runs ML models** (hold detection, classification, grade estimation)
4. **Processes results** (build graphs, extract features)
5. **Generates explanations** (why this grade?)
6. **Stores everything** in Supabase (database + storage)
7. **Returns results** to the frontend in a structured format
8. **Provides API documentation** automatically
9. **Handles async operations** efficiently
10. **Manages ML model lifecycle** (loading, caching)

### What's Working Now (Milestone 1)
- ✅ FastAPI server running
- ✅ Configuration management
- ✅ Supabase connection
- ✅ Health checks
- ✅ API documentation at `/docs`

### What's Coming Next (Milestones 2-10)
- 🚧 Image upload endpoints
- 🚧 ML model integration
- 🚧 Route analysis pipeline
- 🚧 Grade estimation
- 🚧 Explainability features
- 🚧 Feedback collection

**In short:** FastAPI is the "backend brain" that makes the entire route analysis system work by coordinating all the pieces (frontend, ML models, database) into a cohesive application.

---

**Related Documentation:**
- [DESIGN.md](DESIGN.md) - Complete architecture specification
- [SUPABASE_SETUP.md](SUPABASE_SETUP.md) - Database configuration
- [MODEL_PRETRAIN.md](MODEL_PRETRAIN.md) - ML model specifications
