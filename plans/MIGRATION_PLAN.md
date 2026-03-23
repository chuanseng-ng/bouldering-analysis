# Bouldering Route Analysis — Migration Plan

**Version**: 1.0
**Created**: 2026-01-14
**Based On**: docs/DESIGN.md and docs/MODEL_PRETRAIN.md

---

## Executive Summary

This plan outlines the migration from the current Flask-based implementation to the new FastAPI + Supabase architecture as defined in the revised specification. The existing codebase (~4,500 lines of source, ~5,100 lines of tests) will be archived and replaced with a milestone-based implementation.

---

## Current State vs Target State

| Aspect | Current State | Target State |
| :----: | :-----------: | :----------: |
| **Backend Framework** | Flask 3.1.2 | FastAPI |
| **Database** | SQLite/SQLAlchemy | Supabase (Postgres + Storage) |
| **Hold Types (Detection)** | 8 types (crimp, jug, sloper, etc.) | 2 types (hold, volume) |
| **Hold Types (Classification)** | N/A (combined with detection) | 6 types (jug, crimp, sloper, pinch, volume, unknown) |
| **Grade Prediction** | Heuristic 4-factor algorithm | Route Graph + Feature Extraction + Ordinal ML |
| **Model Architecture** | Single YOLO detection model | Separate detection + classification models |
| **Storage** | Local filesystem | Supabase Storage |
| **Frontend** | Flask templates (index.html) | Web (React/Next.js via Lovable → Vercel) + Telegram Bot |
| **Deployment** | Not defined | Backend TBD, Web Frontend on Vercel, Bot TBD |

---

## What Gets Archived

### Source Code (`src/` → `src/archive/legacy/`)

| File | Lines | Reason for Archive |
| :--: | :---: | :---------------: |
| `main.py` | 883 | Flask → FastAPI migration |
| `models.py` | 290 | SQLAlchemy → Supabase migration |
| `config.py` | 323 | May partially reuse config pattern |
| `train_model.py` | 1,222 | Refactor for new detection + classification pipeline |
| `manage_models.py` | 630 | Adapt for new model versioning system |
| `grade_prediction_mvp.py` | 618 | Replace with Route Graph approach |
| `constants.py` | 19 | New hold type taxonomies |
| `setup.py` | 44 | New initialization approach |
| `setup_dev.py` | 113 | New dev environment setup |
| `cfg/user_config.yaml` | - | New configuration structure |
| `templates/index.html` | - | Frontend moved to Lovable |

### Tests (`tests/` → `tests/archive/legacy/`)

| Test File | Lines | Notes |
| :-------: | :---: | :---: |
| `test_main.py` | 1,253 | Reference for new FastAPI tests |
| `test_train_model.py` | 840 | Reference for new training tests |
| `test_manage_models.py` | 702 | Reference for new model management |
| `test_grade_prediction_mvp.py` | 446 | Replace with new grade estimation tests |
| `test_config.py` | 526 | Reference for config tests |
| `test_models.py` | 498 | Reference for Supabase tests |
| `test_e2e_grade_prediction.py` | 435 | Reference for e2e tests |
| `test_main_proxyfix.py` | 164 | May not be needed with FastAPI |
| `test_setup_dev.py` | 246 | New setup tests |
| `conftest.py` | 341 | New fixture approach for Supabase |

---

## Milestone Implementation Plan

### MILESTONE 1 — Backend Foundation

**Goal**: Establish FastAPI backend with Supabase connectivity

#### PR-1.1: FastAPI Bootstrap

- **Function**: `create_app()`
- **Tasks**:
  1. Create `src/app.py` with FastAPI application factory
  2. Implement `/health` endpoint
  3. Configure CORS middleware
  4. Set up API versioning (`/api/v1/`)
  5. Configure logging (structured JSON)
- **Dependencies**: None
- **Estimated Effort**: Small

#### PR-1.2: Supabase Client

- **Function**: `get_supabase_client()`
- **Tasks**:
  1. Install supabase-py dependency
  2. Create `src/database/supabase_client.py`
  3. Implement connection pooling
  4. Add storage bucket access helpers
  5. Environment-based configuration (SUPABASE_URL, SUPABASE_KEY)
- **Dependencies**: PR-1.1
- **Estimated Effort**: Small

---

### MILESTONE 2 — Image Upload & Persistence

**Goal**: Enable image upload and storage in Supabase

#### PR-2.1: Upload Route Image

- **Function**: `upload_route_image(file)`
- **Tasks**:
  1. Create `src/routes/upload.py`
  2. Implement multipart file handling
  3. Validate image format (JPEG, PNG)
  4. Validate file size (configurable limit)
  5. Upload to Supabase Storage bucket
  6. Return public URL
- **Dependencies**: PR-1.2
- **Estimated Effort**: Small

#### PR-2.2: Create Route Record

- **Function**: `create_route_record(image_url)`
- **Tasks**:
  1. Create `routes` table in Supabase
  2. Implement `src/routes/routes.py`
  3. Generate UUID for route
  4. Store image_url and metadata
  5. Return route ID
- **Dependencies**: PR-2.1
- **Estimated Effort**: Small

---

### MILESTONE 3 — Hold Detection (Pre-Training Phase)

**Goal**: Train reusable hold/volume detection model

#### PR-3.1: Detection Dataset Schema

- **Function**: `load_hold_detection_dataset()`
- **Tasks**:
  1. Define dataset structure (YOLOv8 format)
  2. Create `src/training/datasets.py`
  3. Support Roboflow exports
  4. Implement dataset versioning metadata
  5. Validate class taxonomy: `[hold, volume]`
- **Dependencies**: None (can run parallel with M1/M2)
- **Estimated Effort**: Small

#### PR-3.2: Detection Model Definition

- **Function**: `build_hold_detector()`
- **Tasks**:
  1. Create `src/training/detection_model.py`
  2. Configure YOLOv8m architecture
  3. Set input resolution (640x640)
  4. Define hyperparameter schema
- **Dependencies**: PR-3.1
- **Estimated Effort**: Small

#### PR-3.3: Detection Training Loop

- **Function**: `train_hold_detector(dataset)`
- **Tasks**:
  1. Create `src/training/train_detection.py`
  2. Implement training loop with YOLO detection loss
  3. Configure AdamW optimizer
  4. Add augmentations (rotation, brightness, perspective)
  5. Track metrics (Recall@IoU0.5, mAP50)
  6. Save artifacts to `models/detection/`
  7. Generate `metadata.json` (dataset version, commit hash, date)
- **Dependencies**: PR-3.2
- **Estimated Effort**: Medium

#### PR-3.4: Detection Inference

- **Function**: `detect_holds(image) -> list[DetectedHold]`
- **Tasks**:
  1. Create `src/inference/detection.py`
  2. Load pre-trained weights
  3. Return normalized bounding boxes
  4. Include class (hold/volume) and confidence
  5. Implement batch inference option
- **Dependencies**: PR-3.3
- **Estimated Effort**: Small

---

### MILESTONE 4 — Hold Classification (Pre-Training Phase)

**Goal**: Classify detected holds into semantic types

#### PR-4.1: Hold Crop Generator (✅ COMPLETED)

- **Function**: `extract_hold_crops(image, boxes)`
- **Tasks**:
  1. ✅ Create `src/inference/crop_extractor.py`
  2. ✅ Extract hold regions from detection boxes
  3. ✅ Resize to 224x224
  4. ✅ Handle edge cases (partial crops, small holds)
- **Dependencies**: PR-3.4
- **Status**: Completed

#### PR-4.2: Classification Dataset Loader (✅ COMPLETED)

- **Function**: `load_hold_classification_dataset()`
- **Tasks**:
  1. ✅ Create `src/training/classification_dataset.py`
  2. ✅ Support image classification format
  3. ✅ Define taxonomy: `[jug, crimp, sloper, pinch, volume, unknown]`
  4. ✅ Implement class balancing (weighted loss or oversampling)
- **Dependencies**: None (can run parallel)
- **Status**: Completed

#### PR-4.3: Hold Classifier Model (✅ COMPLETED)

- **Function**: `build_hold_classifier()`
- **Tasks**:
  1. ✅ Create `src/training/classification_model.py`
  2. ✅ Configure ResNet-18 or MobileNetV3
  3. ✅ Set input size (224x224 RGB)
  4. ✅ Add softmax output with label smoothing
- **Dependencies**: PR-4.2
- **Status**: Completed

#### PR-4.4: Classification Training (✅ COMPLETED)

- **Function**: `train_hold_classifier(dataset)`
- **Tasks**:
  1. ✅ Create `src/training/train_classification.py`
  2. ✅ Implement cross-entropy loss with class weights
  3. ✅ Configure Adam optimizer
  4. ✅ Add augmentations (rotation, color jitter, cutout)
  5. ✅ Track metrics (Top-1 accuracy, ECE)
  6. ✅ Save artifacts to `models/classification/`
- **Dependencies**: PR-4.3
- **Status**: Completed

#### PR-4.5: Hold Type Inference (✅ COMPLETED)

- **Function**: `classify_hold(crop) -> HoldTypeResult`
- **Tasks**:
  1. ✅ Create `src/inference/classification.py`
  2. ✅ Load pre-trained weights with caching (double-checked locking)
  3. ✅ Return predicted type, probability distribution, confidence
  4. ✅ Implement batch inference via `classify_holds()`
  5. ✅ Cache model and input size for consistency across calls
- **Dependencies**: PR-4.4
- **Status**: Completed with 72 comprehensive tests
- **Test Coverage**: Full test suite in `tests/test_inference_classification.py`
- **Key Exports**: `ClassificationInferenceError`, `HoldTypeResult`, `classify_hold`, `classify_holds`
- **Architecture Pattern**: Model caching with double-checked locking (matches PR-3.4 detection.py)

---

### MILESTONE 5 — Route Graph Construction

**Goal**: Build movement graph from detected holds

#### PR-5.1: Graph Builder

- **Function**: `build_route_graph(holds, wall_angle)`
- **Tasks**:
  1. Create `src/graph/route_graph.py`
  2. Define graph structure (nodes = holds, edges = feasible moves)
  3. Implement reachability heuristics
  4. Consider wall angle in edge weights
  5. Use NetworkX or custom graph class
- **Dependencies**: PR-4.5
- **Estimated Effort**: Medium

#### PR-5.2: Start/Finish Constraints

- **Function**: `apply_route_constraints(graph, start_ids, finish_id)`
- **Tasks**:
  1. Create `src/graph/constraints.py`
  2. Mark start holds
  3. Mark finish hold
  4. Validate path existence
  5. Prune unreachable nodes
- **Dependencies**: PR-5.1
- **Estimated Effort**: Small

---

### MILESTONE 6 — Feature Extraction

**Goal**: Extract interpretable features for grade estimation

#### PR-6.1: Geometry Features

- **Function**: `extract_geometry_features(graph)`
- **Tasks**:
  1. Create `src/features/geometry.py`
  2. Calculate average move distance
  3. Calculate max reach required
  4. Compute path length statistics
  5. Measure hold density
- **Dependencies**: PR-5.2
- **Estimated Effort**: Medium

#### PR-6.2: Hold Composition Features

- **Function**: `extract_hold_features(holds)`
- **Tasks**:
  1. Create `src/features/holds.py`
  2. Count holds by type
  3. Calculate type ratios
  4. Compute size statistics
  5. Measure confidence-weighted type distribution
- **Dependencies**: PR-4.5
- **Estimated Effort**: Small

#### PR-6.3: Feature Vector Assembly

- **Function**: `build_feature_vector(route)`
- **Tasks**:
  1. Create `src/features/assembler.py`
  2. Combine geometry and hold features
  3. Normalize feature ranges
  4. Handle missing values
  5. Return structured feature dict
- **Dependencies**: PR-6.1, PR-6.2
- **Estimated Effort**: Small

---

### MILESTONE 7 — Grade Estimation

**Goal**: Predict route grade with uncertainty

#### PR-7.1: Heuristic Estimator

- **Function**: `estimate_grade_heuristic(features)`
- **Tasks**:
  1. Create `src/grading/heuristic.py`
  2. Implement rule-based grade estimation
  3. Use feature thresholds for V-scale mapping
  4. Provide confidence interval
  5. Enable easy debugging/explanation
- **Dependencies**: PR-6.3
- **Estimated Effort**: Medium

#### PR-7.2: Ordinal ML Estimator (✅ COMPLETED)

- **Function**: `estimate_grade_ml(features, model_path)`
- **Tasks**:
  1. ✅ Create `src/grading/ml_estimator.py`
  2. ✅ Train XGBoost multi-class model (`multi:softprob`)
  3. ✅ Output full probability distribution over 18 V-grades
  4. ✅ Calculate normalized-entropy confidence and difficulty score
  5. ✅ Support model versioning (`models/grading/v<YYYYMMDD_HHMMSS>/`)
  6. ✅ Extract shared helpers to `src/grading/_utils.py`
  7. ✅ Create `src/training/train_grade_estimator.py` with synthetic data generation
  8. ✅ Atomic model write via tempfile + shutil.move
- **Dependencies**: PR-7.1 (heuristic used as label source for synthetic data)
- **Status**: Completed with 96 tests (16 + 44 + 36), 96.95% coverage, pylint 9.95/10
- **Key Exports**: `MLGradeResult`, `estimate_grade_ml`, `GradeTrainingMetrics`, `GradeTrainingResult`, `generate_synthetic_training_data`, `train_grade_estimator`
- **New Dependencies**: xgboost >=2.0.0, scikit-learn >=1.3.0, joblib >=1.3.0

---

### MILESTONE 8 — Explainability

**Goal**: Generate human-readable explanations

#### PR-8.1: Explanation Engine

- **Function**: `generate_explanation(features, prediction)`
- **Tasks**:
  1. Create `src/explanation/engine.py`
  2. Identify top contributing features
  3. Generate natural language explanations
  4. Include confidence qualifiers
  5. Highlight key holds in visualization
- **Dependencies**: PR-7.2
- **Estimated Effort**: Medium

---

### MILESTONE 9 — Supabase Schema Finalization

**Goal**: Complete database schema for production

#### PR-9.1: Routes Table — ✅ COMPLETED (2026-03-15)

Migration file: `migrations/sql/001_create_routes_table.sql`
Verifier script: `scripts/migrations/create_routes_table.py`
Tests: `tests/test_migrations_routes.py`

#### Schema Design

##### Table: routes (✅ COMPLETED — PR-9.1)

```sql
-- migrations/sql/001_create_routes_table.sql
CREATE EXTENSION IF NOT EXISTS moddatetime;

CREATE TABLE IF NOT EXISTS routes (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    image_url   TEXT        NOT NULL
                            CHECK (char_length(image_url) <= 2048),
    wall_angle  FLOAT       CHECK (wall_angle IS NULL OR wall_angle BETWEEN -90 AND 90),
    status      VARCHAR(20) NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE TRIGGER set_routes_updated_at
    BEFORE UPDATE ON routes
    FOR EACH ROW
    EXECUTE FUNCTION moddatetime(updated_at);

CREATE INDEX IF NOT EXISTS idx_routes_created_at ON routes (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routes_status_pending ON routes (status)
    WHERE status IN ('pending', 'processing');
```

Key deviations from original plan:
- `status VARCHAR(20)` column added (load-bearing in `src/routes/routes.py`)
- `gen_random_uuid()` instead of `uuid_generate_v4()` (no extension dependency)
- `moddatetime` trigger instead of custom PL/pgSQL function
- CHECK constraints on `wall_angle`, `status`, and `image_url` length
- Partial index on active `status` values for background job polling

#### PR-9.2: Holds Table — ✅ COMPLETED

Migration file: `migrations/sql/002_create_holds_table.sql`
Verifier script: `scripts/migrations/create_holds_table.py`
Shared utilities: `scripts/migrations/_migration_utils.py`
Tests: `tests/test_migrations_holds.py`, `tests/test_migration_utils.py`

##### Table: holds (✅ COMPLETED — PR-9.2)

```sql
-- migrations/sql/002_create_holds_table.sql
CREATE EXTENSION IF NOT EXISTS moddatetime;

CREATE TABLE IF NOT EXISTS holds (
    id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id             UUID        NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    hold_id              INT         NOT NULL CHECK (hold_id >= 0),
    x_center             FLOAT       NOT NULL CHECK (x_center BETWEEN 0 AND 1),
    y_center             FLOAT       NOT NULL CHECK (y_center BETWEEN 0 AND 1),
    width                FLOAT       NOT NULL CHECK (width BETWEEN 0 AND 1),
    height               FLOAT       NOT NULL CHECK (height BETWEEN 0 AND 1),
    detection_class      VARCHAR(10) NOT NULL CHECK (detection_class IN ('hold', 'volume')),
    detection_confidence FLOAT       NOT NULL CHECK (detection_confidence BETWEEN 0 AND 1),
    hold_type            VARCHAR(20) NOT NULL CHECK (hold_type IN ('jug', 'crimp', 'sloper', 'pinch', 'volume', 'unknown')),
    type_confidence      FLOAT       NOT NULL CHECK (type_confidence BETWEEN 0 AND 1),
    prob_jug             FLOAT       NOT NULL CHECK (prob_jug BETWEEN 0 AND 1),
    prob_crimp           FLOAT       NOT NULL CHECK (prob_crimp BETWEEN 0 AND 1),
    prob_sloper          FLOAT       NOT NULL CHECK (prob_sloper BETWEEN 0 AND 1),
    prob_pinch           FLOAT       NOT NULL CHECK (prob_pinch BETWEEN 0 AND 1),
    prob_volume          FLOAT       NOT NULL CHECK (prob_volume BETWEEN 0 AND 1),
    prob_unknown         FLOAT       NOT NULL CHECK (prob_unknown BETWEEN 0 AND 1),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (route_id, hold_id)
);
-- No updated_at / trigger — holds are write-once (delete + reinsert on re-run)
```

Key design decisions vs. original plan:
- UUID PK (`gen_random_uuid()`) instead of `SERIAL` — consistent with `routes` table
- `ON DELETE CASCADE` — holds are meaningless without their route
- `UNIQUE (route_id, hold_id)` doubles as the route lookup index — no separate `idx_holds_route_id`
- 6 flat `prob_*` columns instead of `JSONB` — explicit CHECK per column, queryable, type-safe
- No `updated_at` / trigger — write-once design; re-run = `DELETE WHERE route_id` + bulk `INSERT`
- Probability sum invariant enforced by `ClassifiedHold.type_probabilities` Pydantic validator

Re-run contract (forward obligation for PR-9.3):
- Before re-inserting holds: `DELETE FROM holds WHERE route_id = $1`, then bulk `INSERT`
- Never bypass `ClassifiedHold` Pydantic validation when inserting (raw dicts skip the sum-to-1.0 check)

Shared utilities (`scripts/migrations/_migration_utils.py`) extracted in PR-9.2 to avoid
N-file duplication as more tables are added.  `create_routes_table.py` was refactored to
delegate to `verify_table()` from this module; public API unchanged.

#### PR-9.3: Features Table — ✅ COMPLETED

Migration file: `migrations/sql/003_create_features_table.sql`
Verifier script: `scripts/migrations/create_features_table.py`
Tests: `tests/test_migrations_features.py`

##### Table: features (✅ COMPLETED — PR-9.3)

```sql
CREATE TABLE IF NOT EXISTS features (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id       UUID        NOT NULL UNIQUE REFERENCES routes(id) ON DELETE CASCADE,
    feature_vector JSONB       NOT NULL,
    extracted_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    -- No CHECK constraints: JSONB contents validated by RouteFeatures Pydantic model.
    -- UNIQUE (route_id) doubles as the route lookup index (no separate index needed).
    -- Write-once: re-run = DELETE WHERE route_id + INSERT.
);
```

#### PR-9.4: Predictions Table — ✅ COMPLETED (2026-03-20)

Migration file: `migrations/sql/004_create_predictions_table.sql`
Verifier script: `scripts/migrations/create_predictions_table.py`
Tests: `tests/test_migrations_predictions.py`

##### Table: predictions (✅ COMPLETED — PR-9.4)

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id         UUID        NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    estimator_type   VARCHAR(20) NOT NULL CHECK (estimator_type IN ('heuristic', 'ml')),
    grade            VARCHAR(10) NOT NULL CHECK (grade IN (
                         'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                         'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17'
                     )),
    grade_index      INT         NOT NULL CHECK (grade_index BETWEEN 0 AND 17),
    confidence       FLOAT       NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    difficulty_score FLOAT       NOT NULL CHECK (difficulty_score BETWEEN 0 AND 1),
    -- NULL for all current estimators (reserved for future calibrated uncertainty output).
    uncertainty      FLOAT                CHECK (uncertainty BETWEEN 0 AND 1),
    -- JSONB: contents validated at app layer by ExplanationResult Pydantic model.
    explanation      JSONB,
    -- NULL for heuristic estimator; v<YYYYMMDD_HHMMSS> for ML models.
    model_version    VARCHAR(20),
    predicted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    -- Multiple predictions per route allowed (immutable history / model versioning).
    -- No UNIQUE on route_id. Append-only: never DELETE before INSERT on re-run.
);

CREATE INDEX IF NOT EXISTS idx_predictions_route_id_predicted_at
    ON predictions (route_id, predicted_at DESC);
```

Key deviations from original plan:
- `confidence NOT NULL` — always computed by both estimators (heuristic and ML)
- `explanation JSONB` instead of `TEXT` — queryable structure, consistent with `feature_vector`
- `model_version VARCHAR(20)` instead of `VARCHAR(50)` — matches `v<YYYYMMDD_HHMMSS>` format (16 chars)
- Added `estimator_type VARCHAR(20)` — distinguishes heuristic vs ML predictions
- Added `grade_index INT` — ordinal index (0–17) for efficient sorting/range queries
- Added `difficulty_score FLOAT` — continuous difficulty metric for filtering/sorting
- 6 CHECK constraints on all scalar domain columns (grade IN list, grade_index range, confidence, difficulty_score, uncertainty, estimator_type enum)
- Compound index `(route_id, predicted_at DESC)` — optimised for sorted per-route listing
- Write contract: append-only (INSERT only); no DELETE+INSERT re-run contract (unlike holds/features)

#### PR-9.5: Feedback Table — Completed

##### Table: feedback (Completed — PR-9.5)

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id    UUID        NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    user_grade  VARCHAR(10) CHECK (user_grade IS NULL OR user_grade IN (
                    'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17'
                )),
    is_accurate BOOLEAN,
    comments    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_route_id_created_at
    ON feedback (route_id, created_at DESC);
```

Key implementation notes:
- `user_grade` nullable + `IS NULL OR IN (V0..V17)` CHECK — user may omit grade; 1 constraint `feedback_user_grade_check`
- `is_accurate` plain nullable BOOLEAN — no CHECK needed
- `comments` plain nullable TEXT — no length limit
- No `UNIQUE (route_id)` — explicit `CREATE INDEX` (not UNIQUE) for route lookup
- No `updated_at` / trigger — append-only, write-once
- INSERT policy `TO PUBLIC` (not `TO service_role`) — anonymous submission from frontend
- 4 RLS policies: `feedback_select_public`, `feedback_insert_public`, `feedback_update_service`, `feedback_delete_service`
- Verifier: `scripts/migrations/create_feedback_table.py` (`verify_feedback_table()`, `--dry-run` mode; default = live verify)
- 52 tests in `tests/test_migrations_feedback.py` (Layer 1: 35 SQL, Layer 2: 16 verifier + config, Layer 3: 1 integration skipped)

**One migration per PR** as specified.

---

### MILESTONE 10 — Frontend Development & Integration

**Goal**: Build and deploy user-facing interface for bouldering route analysis

This milestone follows a three-phase development workflow:

#### Phase 1: Lovable Prototype (PR-10.1)

**Goal**: Rapidly build functional UI prototype using Lovable platform

- **Function**: N/A (no-code development)
- **Tasks**:
  1. Set up Lovable project with route analysis theme
  2. Build core UI components:
     - Image upload interface with drag-and-drop
     - Route image display with interactive hold overlay
     - Hold annotation tools (mark start/finish)
     - Grade prediction display with uncertainty visualization
     - Feedback submission form
     - Route history/gallery
  3. Connect to backend API endpoints (see Backend API Endpoints below)
  4. Implement responsive design for mobile and desktop
  5. Add loading states and error handling
  6. User testing and design iteration
- **Dependencies**: PR-2.2 (Route records), PR-7.2 (Grade estimation)
- **Estimated Effort**: Medium
- **Deliverables**:
  - Working Lovable prototype
  - Design system documentation
  - API integration documentation
  - User feedback summary

#### Phase 2: Code Export & Enhancement (PR-10.2)

**Goal**: Export prototype to Git and refine with Claude Code

- **Function**: `enhance_frontend_features()`
- **Tasks**:
  1. Export Lovable project to Git repository (likely Next.js/React)
  2. Set up local development environment
  3. Install dependencies and configure build pipeline
  4. Refine components using Claude Code:
     - Performance optimizations (code splitting, lazy loading)
     - Advanced interactions (keyboard shortcuts, touch gestures)
     - Improved error handling and validation
     - Responsive design enhancements
     - Accessibility improvements (ARIA labels, keyboard navigation)
  5. Add comprehensive frontend tests:
     - Unit tests for components (Jest/Vitest)
     - Integration tests for API calls
     - E2E tests for critical flows (Playwright/Cypress)
  6. Optimize bundle size and loading performance
  7. Add analytics and monitoring
- **Dependencies**: PR-10.1
- **Estimated Effort**: Medium
- **Deliverables**:
  - Cleaned, optimized codebase
  - Test suite with ≥80% coverage
  - Performance benchmarks
  - Documentation for local development

#### Phase 3: Vercel Deployment (PR-10.3)

**Goal**: Deploy frontend to production with continuous deployment

- **Function**: `deploy_to_vercel()`
- **Tasks**:
  1. Create Vercel project linked to Git repository
  2. Configure environment variables:
     - `NEXT_PUBLIC_API_URL` (backend API endpoint)
     - `NEXT_PUBLIC_SUPABASE_URL` (if frontend uses Supabase)
     - `NEXT_PUBLIC_SUPABASE_ANON_KEY` (if needed)
  3. Configure build settings:
     - Framework preset (Next.js/React)
     - Build command and output directory
     - Install command customization
  4. Set up preview deployments for pull requests
  5. Configure production deployment on main branch
  6. Set up custom domain (optional)
  7. Configure CORS on backend to allow Vercel domain
  8. Set up monitoring and analytics:
     - Vercel Analytics
     - Error tracking (Sentry integration)
     - Performance monitoring
  9. Document deployment process and troubleshooting
- **Dependencies**: PR-10.2
- **Estimated Effort**: Small
- **Deliverables**:
  - Production deployment on Vercel
  - Automatic preview deployments
  - Deployment documentation
  - Monitoring dashboards

#### Backend API Endpoints

These endpoints must be implemented in the backend to support the frontend:

| Endpoint | Method | Purpose | Status |
| :------: | :----: | :-----: | :----: |
| `POST /api/v1/routes/upload` | POST | Upload route image | ✅ Completed (PR-2.1) |
| `POST /api/v1/routes` | POST | Create route record | ✅ Completed (PR-2.2) |
| `GET /api/v1/routes/{id}` | GET | Get route details | ✅ Completed (PR-2.2) |
| `GET /api/v1/routes/{id}/status` | GET | Poll route processing status | ✅ Completed (PR-2.2) |
| `POST /api/v1/routes/{id}/analyze` | POST | Trigger hold detection & analysis | Pending |
| `GET /api/v1/routes/{id}/holds` | GET | Get detected holds | Pending |
| `PUT /api/v1/routes/{id}/constraints` | PUT | Set start/finish holds | Pending (PR-5.x) |
| `GET /api/v1/routes/{id}/prediction` | GET | Get grade prediction | Pending (PR-7.x) |
| `POST /api/v1/routes/{id}/feedback` | POST | Submit user feedback | Pending |
| `GET /api/v1/routes` | GET | List routes (with pagination) | Pending |

#### Frontend Responsibilities

The frontend will handle:

- **User Interface**:
  - Responsive, mobile-friendly design
  - Interactive hold annotation
  - Real-time feedback on API operations
- **Client-Side Logic**:
  - Form validation
  - Image preprocessing (resize, format conversion)
  - State management (route data, UI state)
- **API Integration**:
  - REST API calls to backend
  - Error handling and retries
  - Loading states and optimistic updates
- **User Experience**:
  - Smooth animations and transitions
  - Helpful error messages
  - Onboarding and tutorials

#### Non-Goals (Handled by Backend)

- Hold detection/classification logic
- Grade prediction algorithms
- Database operations
- Image storage and processing
- Business logic and validation

#### Phase 4: Telegram Bot Frontend (PR-10.4)

**Goal**: Provide lightweight alternative interface via Telegram

- **Function**: `handle_telegram_message(update, context)`
- **Tasks**:
  1. Create Telegram bot using `python-telegram-bot` library
  2. Implement bot commands:
     - `/start` - Welcome message and instructions
     - `/help` - Usage guide
     - Photo upload handler - Analyze route from photo
     - `/status {route_id}` - Check analysis status
     - `/history` - View recent analyses
  3. Integrate with backend API:
     - Upload image via `/api/v1/routes/upload`
     - Create route record
     - Trigger analysis
     - Retrieve and format prediction
  4. Implement conversation flow:
     - Receive photo from user
     - Show "analyzing..." status
     - Return grade prediction with explanation
     - Offer feedback options
  5. Add error handling and user-friendly messages
  6. Deploy bot (webhook or polling mode)
  7. Document bot setup and usage
- **Dependencies**: PR-2.2 (Route creation), PR-7.2 (Grade prediction)
- **Estimated Effort**: Small-Medium
- **Deliverables**:
  - Working Telegram bot
  - Deployment configuration
  - User guide with screenshots
  - Bot command documentation

**Bot Features**:

- Simple photo upload (no annotations initially)
- Quick grade prediction
- Text-based explanations
- Optional: Feedback submission via inline buttons
- Optional: Route history with thumbnails

**Technology Stack**:

- `python-telegram-bot` (v20+)
- FastAPI backend integration
- Deployment: AWS Lambda, Google Cloud Functions, or dedicated server
- Webhook mode for production (polling for development)

See [docs/TELEGRAM_BOT.md](../docs/TELEGRAM_BOT.md) for detailed implementation guide.

---

#### PR-10.10 — TODO Review (post all PR-10 items)

**Goal**: Review [`TODO.md`](../TODO.md) once all PR-10 sub-items (10.1–10.9)
are complete and decide which deferred items must be addressed before the
project is considered production-ready.

- **Function**: N/A (review + decision task)
- **Tasks**:
  1. Open `TODO.md` and read every item in full.
  2. For each item, assess current state: has the trigger condition been met
     (e.g. production URL known, deployment topology decided, user base exists)?
  3. Categorise each item as one of:
     - **Act now** — condition met, implement in this PR or create a follow-up PR immediately.
     - **Defer again** — condition not yet met; update the entry with the new reason.
     - **Close** — no longer relevant (e.g. superseded by a different decision).
  4. For every **Act now** item, either implement the fix inline or create a
     clearly scoped follow-up PR with a spec in `plans/specs/`.
  5. Remove closed items from `TODO.md`; update deferred items with new context.
- **Specific items to evaluate**:
  - **S1** (rate limiting): Is the deployment topology known? Single-process or
    multi-worker? Choose `_UploadRateLimiter` vs. `slowapi` + Redis.
  - **S2** (CORS): Is the Vercel/frontend URL known? Set `BA_CORS_ORIGINS` and
    remove `allow_credentials=True` if cookies are not used.
  - **S3** (exception leakage in 422s): Is the API now public-facing? If yes,
    map exceptions to generic user messages.
  - **S4** (race condition): Is concurrent analysis traffic expected? If yes,
    implement `update_record_if_status()` in `supabase_client.py`.
- **Dependencies**: PR-10.1 through PR-10.9 all complete
- **Estimated Effort**: Small (review) + variable (any follow-up implementation)
- **Deliverables**:
  - Updated `TODO.md` with disposition notes for every item
  - Zero or more new scoped follow-up PRs or inline fixes

---

## Implementation Order

```text
Phase 1: Foundation (M1 + M2) — ✅ COMPLETED
├── ✅ PR-1.1: FastAPI Bootstrap
├── ✅ PR-1.2: Supabase Client
├── ✅ PR-2.1: Upload Route Image
└── ✅ PR-2.2: Create Route Record

Phase 2: Perception Pre-training (M3 + M4) — ✅ COMPLETED
├── ✅ PR-3.1: Detection Dataset Schema
├── ✅ PR-3.2: Detection Model Definition
├── ✅ PR-3.3: Detection Training Loop
├── ✅ PR-3.4: Detection Inference
├── ✅ PR-4.1: Hold Crop Generator
├── ✅ PR-4.2: Classification Dataset Loader
├── ✅ PR-4.3: Hold Classifier Model
├── ✅ PR-4.4: Classification Training
└── ✅ PR-4.5: Hold Type Inference

Phase 3: Intelligence (M5 + M6 + M7) — ✅ COMPLETED
├── ✅ PR-5.1: Graph Builder
├── ✅ PR-5.2: Start/Finish Constraints
├── ✅ PR-6.1: Geometry Features
├── ✅ PR-6.2: Hold Composition Features
├── ✅ PR-6.3: Feature Vector Assembly
├── ✅ PR-7.1: Heuristic Estimator
└── ✅ PR-7.2: Ordinal ML Estimator

Phase 4: Polish (M8 + M9 + M10)
├── ✅ PR-8.1: Explanation Engine
├── ✅ PR-9.1: Routes Table
├── ✅ PR-9.2: Holds Table
├── ✅ PR-9.3: Features Table
├── ✅ PR-9.4: Predictions Table
├── ✅ PR-9.5: Feedback Table
├── ✅ PR-10.1: Supabase Client Extensions + Config
├── ✅ PR-10.2: Routes Table Extension
├── ✅ PR-10.3: Analysis Endpoint
├── ✅ PR-10.4: Constraints & Prediction Endpoints
├── ✅ PR-10.5: Feedback + List Routes
├── ✅ PR-10.6: Lovable Prototype
├── PR-10.7: Next.js Frontend (deferred — Vite SPA sufficient)
├── ✅ PR-10.8: Vercel Deployment (frontend) + Render Deployment (backend)
├── PR-10.9: Telegram Bot
├── PR-10.10: TODO Review (post all PR-10 items)
└── PR-10.11: VPS Migration (post initial ML models — Hetzner + Coolify)
```

---

## Quality Gates Per PR

Each PR must satisfy:

1. **Type Safety**: `mypy src/ tests/` passes
2. **Linting**: `ruff check .` passes
3. **Formatting**: `ruff format --check .` passes
4. **Testing**: 85%+ coverage (current stage), 90% when all features complete
5. **Quality**: pylint score 8.5/10 (current stage), 9.0/10 when all features complete
6. **Documentation**: Google-style docstrings on all functions
7. **Agent Reviews**: python-reviewer + code-reviewer + security-reviewer (mandatory, parallel); database-reviewer (mandatory for any Supabase change)

---

## Agent Requirements Per PR

> Agent names are logical roles invokable via the Task tool with any compatible provider
> (e.g., `everything-claude-code:<agent-name>`).

### Mandatory for Every PR

| Agent | When | Notes |
| ------- | ------ | ------- |
| python-reviewer | After writing .py files | Type safety, pylint, immutability |
| code-reviewer | After implementation | Correctness, architecture alignment |
| security-reviewer | Before every commit | Run in parallel with python-reviewer and code-reviewer |
| doc-updater | After clean code review | CLAUDE.md, specs, docstrings |

### Mandatory When Applicable

| Agent | Applicable PRs / Condition | Trigger |
| ------- | --------------------------- | --------- |
| planner | PRs touching >1 file | Before writing code |
| tdd-guide | Every new function/endpoint | After planning, before implementation |
| database-reviewer | PR-2.2, PR-9.x | Any Supabase schema or SQL change |
| e2e-runner | End of M3, M4, M7, M10 | Milestone completion |
| architect | PR-5.1, PR-7.1, PR-7.2 | Design decisions required |

### Parallel Execution Rule

python-reviewer + code-reviewer + security-reviewer launch simultaneously after implementation is complete. If database-reviewer is triggered, it must complete successfully before the parallel group launches.

---

## Risks & Mitigations

| Risk | Mitigation |
| :--: | :--------: |
| Supabase latency | Use connection pooling, batch operations |
| Model size for serverless | Consider ONNX conversion for inference |
| Feature extraction complexity | Start with heuristic, iterate with ML |
| Hold type label noise | Use `unknown` class, confidence thresholds |
| Graph construction perf | Limit edge count with reachability heuristics |

---

## References

- [docs/DESIGN.md](../docs/DESIGN.md) — Fundamental design specification
- [docs/MODEL_PRETRAIN.md](../docs/MODEL_PRETRAIN.md) — Model pretraining specification
- [Archived Code](../src/archive/legacy/) — Previous implementation for reference

---

## Changelog

- **2026-03-23**: Marked PR-10.6 (Lovable Prototype) and PR-10.8 (Vercel + Render Deployment) as completed; deferred PR-10.7 (Next.js — Vite SPA sufficient); added PR-10.11 (VPS Migration, post ML models); chose Render (free tier) as backend hosting for current stage, Hetzner + Coolify as target after ML models are trained; created docs/RENDER_SETUP.md
- **2026-03-21**: Marked PR-10.1–10.5 (Frontend Integration API) as completed; added PR-10.10 (TODO Review) as a post-PR-10 gate; created TODO.md with four deferred security items (S1 rate limiting, S2 CORS, S3 exception leakage, S4 race condition)
- **2026-03-20**: Marked PR-9.5 (Feedback Table) as completed; added feedback schema with nullable user_grade CHECK, public INSERT policy, explicit index, and verifier/test details
- **2026-03-20**: Marked PR-9.4 (Predictions Table) as completed; updated predictions schema to match implementation (added estimator_type, grade_index, difficulty_score; changed explanation TEXT→JSONB, confidence nullable→NOT NULL, model_version VARCHAR(50)→VARCHAR(20)); added compound index, 6 CHECK constraints, and key deviation notes
- **2026-03-19**: Marked PR-9.2 (Holds Table) as completed; marked PR-9.3 (Features Table) as completed; added PR-9.3 section with verifier/test details; refined PR-9.4/PR-9.5 schemas to use UUID PKs, NOT NULL, ON DELETE CASCADE; updated Phase 4 implementation order
- **2026-03-10**: Marked PR-7.2 (Ordinal ML Estimator) as completed; marked Phase 3 (Intelligence) as completed; updated PR-7.2 task list with final implementation details
- **2026-02-23**: Updated endpoint statuses to reflect Phase 1 (M1+M2) and Phase 2 (M3+M4) completion; marked POST /api/v1/routes/upload and POST /api/v1/routes as Completed; corrected POST /api/v1/routes/{id}/analyze and GET /api/v1/routes/{id}/holds back to Pending (HTTP handlers not yet implemented)
- **2026-02-21**: Added "Agent Reviews" quality gate (item 7) and "Agent Requirements Per PR" section with mandatory/conditional agent tables and parallel execution rule
- **2026-01-14**: Initial migration plan created
