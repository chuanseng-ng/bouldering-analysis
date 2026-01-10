# End-to-End Testing and Staging Deployment Guide

## Overview

This document provides step-by-step instructions for performing end-to-end (E2E) testing of the Phase 1a MVP grade prediction algorithm and deploying to a staging environment.

---

## Part 1: End-to-End Testing

### 1.1 Prerequisites

Before running E2E tests, ensure:

```bash
# 1. All unit tests pass
pytest tests/ --cov=src/ --cov-report=term-missing

# 2. QA checks pass
./run_qa.csh

# 3. Database is initialized
python src/setup_dev.py
```

### 1.2 Manual E2E Test Checklist

#### Test Case 1: Basic Image Upload and Analysis

**Steps:**

1. Start the Flask development server:

   ```bash
   ./run.sh
   # Or: cd src && python main.py
   ```

2. Open browser to `http://localhost:5000`
3. Upload a bouldering route image (JPG/PNG)
4. Select wall incline from dropdown (e.g., "Vertical")
5. Click "Analyze Route"

**Expected Results:**

- [ ] Image uploads successfully
- [ ] Loading spinner displays during analysis
- [ ] Predicted grade displays (V0-V12)
- [ ] Score breakdown shows all 4 factors:
  - Hold Difficulty (35%)
  - Hold Density (25%)
  - Distance (20%)
  - Wall Incline (20%)
- [ ] Detected holds count displays (handholds + footholds)
- [ ] Confidence score displays

#### Test Case 2: Wall Incline Variations

**Steps:**

1. Upload the same image 5 times with different wall inclines:
   - Slab
   - Vertical
   - Slight Overhang
   - Moderate Overhang
   - Steep Overhang

**Expected Results:**

- [ ] Slab produces lowest wall incline score (3.0)
- [ ] Steep overhang produces highest wall incline score (11.0)
- [ ] Predicted grades increase with steeper angles
- [ ] Score breakdown reflects different wall incline values

#### Test Case 3: Edge Cases

| Test | Action | Expected Result |
| :--: | :----: | :-------------: |
| No holds detected | Upload image with no climbing holds | Handles gracefully, returns prediction |
| Invalid file type | Upload .txt or .pdf file | Error message displayed |
| Large file | Upload image > 16MB | Error message about file size |
| Invalid wall incline | Send API request with invalid wall_incline | Error message with valid options |

#### Test Case 4: API Endpoint Testing

**Using curl or Postman:**

```bash
# Test /analyze endpoint
curl -X POST http://localhost:5000/analyze \
  -F "file=@test_route.jpg" \
  -F "wall_incline=vertical"

# Expected response:
# {
#   "analysis_id": "...",
#   "predicted_grade": "V5",
#   "confidence": 0.85,
#   "breakdown": {
#     "hold_difficulty": 6.5,
#     "hold_density": 4.2,
#     "distance": 5.1,
#     "wall_incline": 6.0,
#     ...
#   }
# }

# Test /health endpoint
curl http://localhost:5000/health

# Expected: {"status": "healthy", ...}

# Test /stats endpoint
curl http://localhost:5000/stats

# Expected: {"total_analyses": N, ...}
```

#### Test Case 5: Database Persistence

**Steps:**

1. Analyze an image
2. Note the analysis_id from response
3. Query database directly (or use `/stats` endpoint for API-based verification):

   ```bash
   sqlite3 bouldering_analysis.db "SELECT * FROM analyses WHERE id='<analysis_id>';"
   ```

**Expected Results:**

- [ ] Analysis record exists in database
- [ ] `wall_incline` field is populated correctly
- [ ] `predicted_grade` matches UI display
- [ ] `features_extracted` contains score breakdown JSON

#### Test Case 6: Score Breakdown Validation

**Steps:**

1. Analyze an image and capture the breakdown
2. Manually verify weighted sum:

   ```text
   final_score = (hold_difficulty × 0.35) +
                 (hold_density × 0.25) +
                 (distance × 0.20) +
                 (wall_incline × 0.20)
   ```

**Expected Results:**

- [ ] Calculated final_score matches breakdown.final_score
- [ ] Grade mapping is correct for the final_score

### 1.3 Automated E2E Tests

Create automated E2E tests using the test client:

```python
# tests/test_e2e_grade_prediction.py

import io
from PIL import Image
import pytest

def test_e2e_upload_and_analyze(test_client, mocker):
    """E2E test: Upload image and get grade prediction."""
    # Mock YOLO model to return consistent detections
    mock_model = mocker.patch('src.main.load_active_hold_detection_model')
    mock_model.return_value.predict.return_value = [create_mock_yolo_result()]

    # Create test image
    img = Image.new('RGB', (800, 1200), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Test each wall incline
    for wall_incline in ['slab', 'vertical', 'slight_overhang',
                          'moderate_overhang', 'steep_overhang']:
        response = test_client.post(
            '/analyze',
            data={
                'file': (img_bytes, 'test.jpg'),
                'wall_incline': wall_incline
            },
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        data = response.get_json()

        # Verify response structure
        assert 'predicted_grade' in data
        assert 'confidence' in data
        assert 'breakdown' in data
        assert data['breakdown']['wall_angle'] == wall_incline

        img_bytes.seek(0)  # Reset for next iteration


def test_e2e_grade_increases_with_overhang(test_client, mocker):
    """E2E test: Verify grades increase with steeper wall angles."""
    # ... similar setup ...

    grades = []
    for wall_incline in ['slab', 'vertical', 'steep_overhang']:
        response = test_client.post('/analyze', ...)
        data = response.get_json()
        grades.append(data['breakdown']['final_score'])

    # Slab should have lowest score, steep overhang highest
    assert grades[0] < grades[1] < grades[2]
```

### 1.4 Performance Testing

**Test prediction time requirement (<100ms):**

```python
import time

def test_prediction_performance(test_client, mocker):
    """Verify prediction completes within 100ms."""
    # Setup mocks...

    start = time.perf_counter()
    response = test_client.post('/analyze', ...)
    elapsed = (time.perf_counter() - start) * 1000  # ms

    assert response.status_code == 200
    assert elapsed < 100, f"Prediction took {elapsed:.2f}ms (>100ms limit)"
```

---

## Part 2: Staging Deployment

### 2.1 Staging Environment Options

#### Option A: Local Staging (Recommended for Initial Testing)

Run the app in production-like mode locally:

```bash
# 1. Create staging environment
python -m venv venv_staging
source venv_staging/bin/activate  # Linux/Mac
# Or: venv_staging\Scripts\activate  # Windows

# 2. Install production dependencies
pip install -r requirements.txt

# 3. Set production-like environment variables
export FLASK_ENV=production
export DATABASE_URL=sqlite:///staging_bouldering.db

# 4. Initialize staging database
python src/setup.py

# 5. Run with production server (gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.main:app
```

#### Option B: Docker Staging

Create a `Dockerfile.staging`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application
COPY src/ ./src/
COPY data/ ./data/

# Set environment
ENV FLASK_ENV=production
ENV DATABASE_URL=sqlite:///data/staging_bouldering.db

# Initialize and run
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.main:app"]
```

Build and run:

```bash
docker build -f Dockerfile.staging -t bouldering-staging .
docker run -p 5000:5000 -v $(pwd)/models:/app/models bouldering-staging
```

#### Option C: Cloud Staging (Heroku/Railway/Render)

Example for **Render.com**:

1. Create `render.yaml`:

   ```yaml
   services:
     - type: web
       name: bouldering-staging
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: gunicorn -w 4 src.main:app
       envVars:
         - key: FLASK_ENV
           value: production
         - key: DATABASE_URL
           value: sqlite:///data/staging.db
   ```

2. Connect GitHub repository to Render
3. Deploy from branch `python/grade_predict_phase1_check`

### 2.2 Staging Deployment Checklist

#### Pre-Deployment

- [ ] All unit tests pass locally
- [ ] All QA checks pass (`./run_qa.csh`)
- [ ] E2E tests pass locally
- [ ] Database migrations applied (if any)
- [ ] Configuration validated (`src/cfg/user_config.yaml`)

#### Deployment Steps

```bash
# 1. Create staging branch (if not using current branch)
git checkout -b staging/phase1a-mvp

# 2. Verify CI/CD passes
# Push to GitHub and wait for workflow completion

# 3. Deploy to staging environment
# (Commands depend on your hosting choice)

# 4. Verify deployment
curl https://staging.your-domain.com/health
```

#### Post-Deployment Verification

- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] Homepage loads correctly
- [ ] Image upload works
- [ ] Wall incline dropdown displays all 5 options
- [ ] Analysis returns predicted grade
- [ ] Score breakdown displays correctly
- [ ] Database writes succeed (check `/stats`)

### 2.3 Staging Test Plan

Run these tests on the staging environment:

| Test | Command/Action | Success Criteria |
| :--: | :------------: | :--------------: |
| Health check | `curl /health` | Returns 200 OK |
| Upload image | Web UI upload | Grade displayed |
| API analyze | `curl -X POST /analyze` | JSON response |
| Different wall angles | Test all 5 options | Scores vary correctly |
| Error handling | Invalid file upload | Error message shown |
| Performance | Multiple requests | < 500ms response |
| Database | Check `/stats` | Count increases |

### 2.4 Rollback Plan

If staging deployment fails:

```bash
# 1. Revert to previous version
git checkout main
git push origin main --force-with-lease  # Only if needed

# 2. For Docker deployments
docker run -p 5000:5000 bouldering:previous-tag

# 3. For cloud deployments
# Use platform's rollback feature (Render, Heroku, etc.)
```

**Post-Rollback Verification Checklist:**

- [ ] Health check passes: `curl /health` returns 200
- [ ] Data integrity: Verify recent analyses are intact
- [ ] Smoke tests: Upload and analyze one test image
- [ ] Monitor error logs for 15 minutes after rollback

**Stakeholder Communication:**

1. Notify team via Slack/email immediately after rollback
2. Document root cause and affected timeframe
3. Create incident report if rollback was due to user-facing issues

---

## Part 3: Monitoring and Logging

### 3.1 Key Metrics to Monitor

During staging:

1. **Prediction Distribution**: Track predicted grades

   ```sql
   SELECT predicted_grade, COUNT(*)
   FROM analyses
   GROUP BY predicted_grade
   ORDER BY predicted_grade;
   ```

2. **Wall Incline Usage**: Track which angles are selected

   ```sql
   SELECT wall_incline, COUNT(*)
   FROM analyses
   GROUP BY wall_incline;
   ```

3. **Factor Score Ranges**: Ensure scores are in expected ranges
   - Hold Difficulty: 1-13
   - Hold Density: 0-12
   - Distance: 1-12
   - Wall Incline: 3-11

### 3.2 Logging Configuration

Ensure proper logging in staging:

```python
# In src/main.py or config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('staging.log'),
        logging.StreamHandler()
    ]
)
```

Key log points for Phase 1a:

- Prediction requests (wall_incline, hold counts)
- Factor scores (all 4 factors)
- Final predicted grade
- Any errors or edge cases

### 3.3 Alert Thresholds

Configure alerts for anomalous patterns:

| Metric | Threshold | Alert Condition |
| :----: | :-------: | :-------------: |
| V12 predictions | > 10% of total | May indicate scoring bug |
| Average confidence | < 0.5 | Detection quality issue |
| Prediction latency | > 500ms p95 | Performance degradation |
| Error rate | > 5% | Application instability |

**Log Retention Policy:**

- Staging logs: 30 days
- Production logs: 90 days
- Calibration data: Indefinite (for model improvement)

**Dashboard Metrics to Display:**

- Prediction distribution histogram (V0-V12)
- Wall incline selection breakdown (pie chart)
- Average response time (line chart, hourly)
- Error rate over time (line chart, daily)

---

## Summary Checklist

### Before Marking Phase 1a E2E Testing Complete

- [ ] All manual E2E tests pass
- [ ] Automated E2E tests added and pass
- [ ] Performance requirement met (<100ms prediction)
- [ ] All 5 wall incline options work correctly
- [ ] Score breakdown displays correctly
- [ ] Database persistence verified
- [ ] Edge cases handled gracefully

### Before Marking Staging Deployment Complete

- [ ] Staging environment set up
- [ ] Application deployed successfully
- [ ] Health check passes
- [ ] Full E2E test suite passes on staging
- [ ] Logging and monitoring configured
- [ ] Rollback plan documented and tested
- [ ] Ready for user feedback collection

---

## Next Steps After Staging

1. **Deploy to Production** with feature flag (optional)
2. **Collect User Feedback** - Target 50+ route analyses
3. **Monitor Prediction Accuracy** - Track user-reported grades vs predictions
4. **Begin Phase 1b Calibration** - Adjust thresholds based on feedback
