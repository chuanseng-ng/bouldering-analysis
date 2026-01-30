# Bouldering Route Analysis — Fundamental Design Specification

## 1. Project Goal

Build a web-based system that estimates the difficulty of a bouldering route
(V-scale) from a user-provided image by:

- detecting and classifying holds
- constructing a movement graph
- extracting interpretable features
- estimating grade with uncertainty and explanation

This system prioritizes **explainability, modularity, and data efficiency**.

---

## 2. Core Design Principles

1. Backend-first architecture
2. Route difficulty ≠ pixels
3. Perception models are pre-trained and reusable
4. Grades are ordinal and uncertain
5. One pull request ≈ one function or capability
6. Every ML output must be explainable

---

## 3. High-Level Architecture

```text
┌─────────────────────────────────────────────────────┐
│              Web Frontend (Primary)                 │
│         (Lovable → React/Next.js → Vercel)          │
│                                                     │
│  - Image Upload UI                                  │
│  - Hold Annotation Interface                        │
│  - Grade Display with Explanations                  │
│  - User Feedback Forms                              │
│  - Route History & Gallery                          │
└─────────────────────────────────────────────────────┘
                        │
                        │
┌─────────────────────────────────────────────────────┐
│         Telegram Bot (Alternative)                  │
│           (Python Telegram Bot)                     │
│                                                     │
│  - Photo Upload via Chat                            │
│  - Quick Grade Predictions                          │
│  - Text-based Interaction                           │
│  - Minimal Friction                                 │
└─────────────────────────────────────────────────────┘
                        │
                        ├─────────────────────┐
                        │ REST API (HTTPS)    │
                        ├─────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend                        │
│          (Python 3.10+ on TBD platform)             │
│                                                     │
│  - Image Upload & Storage                           │
│  - Hold Detection (YOLOv8)                          │
│  - Hold Classification                              │
│  - Route Graph Construction                         │
│  - Feature Extraction                               │
│  - Grade Estimation                                 │
│  - Explainability                                   │
└─────────────────────────────────────────────────────┘
                        │
                        │ Storage & Database
                        │
┌─────────────────────────────────────────────────────┐
│              Supabase                               │
│        (Postgres + Storage)                         │
│                                                     │
│  - Route Records (routes table)                     │
│  - Hold Data (holds table)                          │
│  - Features (features table)                        │
│  - Predictions (predictions table)                  │
│  - User Feedback (feedback table)                   │
│  - Image Storage (route-images bucket)              │
└─────────────────────────────────────────────────────┘
```

## 4. Domain Model

### 4.1 Route

```python
class Route:
    id: UUID
    image_url: str
    wall_angle: float
    holds: list[Hold]
    start_hold_ids: list[int]
    finish_hold_id: int
```

### 4.2 Hold

```python
class Hold:
    id: int
    x: float          # normalized [0,1]
    y: float
    size: float
    type: HoldType
    confidence: float
```

## 5. Milestone-Based Implementation Plan

### MILESTONE 1 — Backend Foundation

- PR-1.1: FastAPI Bootstrap
  - Function: create_app()
  - Health check
  - API versioning
- PR-1.2: Supabase Client
  - Function: get_supabase_client()
  - Storage + Postgres access

### MILESTONE 2 — Image Upload & Persistence

- PR-2.1: Upload Route Image
  - Function: upload_route_image(file)
  - Validate
  - Store in Supabase Storage
- PR-2.2: Create Route Record
  - Function: create_route_record(image_url)

### MILESTONE 3 — Hold Detection (Pre-Training Phase)

Goal: Train a reusable model to locate holds in arbitrary gym images.

- PR-3.1: Detection Dataset Schema
  - Function: load_hold_detection_dataset()
  - Image
  - Bounding boxes / masks
- PR-3.2: Detection Model Definition
  - Function: build_hold_detector()
  - YOLO / Detectron2
  - Gym-agnostic
- PR-3.3: Detection Training Loop
  - Function: train_hold_detector(dataset)
  - Offline training
  - Model artifact saved
- PR-3.4: Detection Inference
  - Function: detect_holds(image)
  - Uses pre-trained weights
  - Outputs normalized boxes

### MILESTONE 4 — Hold Classification (Pre-Training Phase)

Goal: Classify detected holds into semantic types.

- PR-4.1: Hold Crop Generator
  - Function: extract_hold_crops(image, boxes)
- PR-4.2: Classification Dataset Loader
  - Function: load_hold_classification_dataset()
- PR-4.3: Hold Classifier Model
  - Function: build_hold_classifier()
  - MobileNet / ResNet-18
- PR-4.4: Classification Training
  - Function: train_hold_classifier(dataset)
- PR-4.5: Hold Type Inference
  - Function: classify_hold(crop)
  - Returns (type, confidence)

### MILESTONE 5 — Route Graph Construction

- PR-5.1: Graph Builder
  - Function: build_route_graph(holds, wall_angle)
  - Nodes = holds
  - Edges = feasible moves
- PR-5.2: Start/Finish Constraints
  - Function: apply_route_constraints(graph, start_ids, finish_id)

### MILESTONE 6 — Feature Extraction

- PR-6.1: Geometry Features
  - Function: extract_geometry_features(graph)
- PR-6.2: Hold Composition Features
  - Function: extract_hold_features(holds)
- PR-6.3: Feature Vector Assembly
  - Function: build_feature_vector(route)

### MILESTONE 7 — Grade Estimation

- PR-7.1: Heuristic Estimator
  - Function: estimate_grade_heuristic(features)
- PR-7.2: Ordinal ML Estimator
  - Function: estimate_grade_ml(features)
  - XGBoost / LightGBM
  - Outputs uncertainty

### MILESTONE 8 — Explainability

- PR-8.1: Explanation Engine
  - Function: generate_explanation(features, prediction)

### MILESTONE 9 — Supabase Schema Finalization

- Tables:
  - routes
  - holds
  - features
  - predictions
  - feedback

One migration per PR.

### MILESTONE 10 — Frontend Development & Integration

**Goal**: Build and deploy user-facing interfaces (Web + Telegram Bot)

#### Web Frontend: Three-Phase Approach

**Phase 1: Lovable Prototype (PR-10.1)**
- Rapid UI development using Lovable platform
- Core components: upload, annotation, display
- Backend API integration
- User testing and iteration

**Phase 2: Code Export & Enhancement (PR-10.2)**
- Export Lovable project to Git repository
- Refine with Claude Code
- Performance optimizations
- Comprehensive testing
- Accessibility improvements

**Phase 3: Vercel Deployment (PR-10.3)**
- Deploy to Vercel platform
- Configure environment variables
- Set up continuous deployment
- Configure monitoring and analytics

**Use Cases**: Full-featured analysis, detailed annotations, advanced features

#### Telegram Bot: Simple Implementation (PR-10.4)

**Phase 4: Telegram Bot Frontend (PR-10.4)**
- Build bot using `python-telegram-bot` library
- Implement photo upload and analysis
- Simple text-based interaction
- Deploy to serverless or dedicated service

**Bot Commands**:
- `/start` - Welcome message
- `/help` - Usage guide
- Photo upload → Instant grade prediction
- `/history` - Recent analyses (optional)

**Use Cases**: Quick grade checks, on-the-go analysis, minimal friction

#### Frontend Responsibilities

**Web Frontend**:
- Image upload interface
- Interactive hold annotation
- Grade prediction display with uncertainty
- Feedback submission forms
- Responsive design (mobile + desktop)
- Route history and gallery

**Telegram Bot**:
- Photo upload via chat
- Quick grade predictions
- Text-based explanations
- Simple feedback (optional)

#### Backend API Contract

Both frontends use the same API:

- `POST /api/v1/routes/upload` - Upload image
- `POST /api/v1/routes` - Create route record
- `GET /api/v1/routes/{id}` - Get route details
- `POST /api/v1/routes/{id}/analyze` - Trigger analysis
- `PUT /api/v1/routes/{id}/constraints` - Set start/finish
- `GET /api/v1/routes/{id}/prediction` - Get prediction
- `POST /api/v1/routes/{id}/feedback` - Submit feedback
- `GET /api/v1/routes` - List routes (with pagination)

#### Technology Stack

**Web Frontend**:
- **Development**: Lovable → React/Next.js
- **Hosting**: Vercel (automatic Git deployments)
- **API**: REST (FastAPI backend)

**Telegram Bot**:
- **Development**: Python (`python-telegram-bot`)
- **Hosting**: TBD (Serverless or dedicated service)
- **API**: REST (FastAPI backend)

See [docs/FRONTEND_WORKFLOW.md](FRONTEND_WORKFLOW.md), [docs/VERCEL_SETUP.md](VERCEL_SETUP.md), and [docs/TELEGRAM_BOT.md](TELEGRAM_BOT.md) for detailed guides.

## 6. Non-Goals (Explicit)

- Perfect grade prediction
- End-to-end CNN grading
- Eliminating human uncertainty

## 7. Future Extensions (Post-MVP)

- Graph Neural Networks
- Gym-specific grade normalization
- Beta path prediction
- Setter tooling mode
