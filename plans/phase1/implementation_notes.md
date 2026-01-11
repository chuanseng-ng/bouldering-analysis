# Phase 1 Implementation Notes

## Overview

This document provides practical technical guidance for implementing the Phase 1 grade prediction algorithm, including calibration strategies, common pitfalls, and iteration approaches.

## Staged Implementation Approach

### Phase 1a: Basic 4-Factor Model (Weeks 1-4) - IMPLEMENTED ✅

**Goal**: Working prediction system, even if accuracy is initially moderate

**Implemented** (see `src/grade_prediction_mvp.py`):

1. [x] Factor 1: Hold difficulty (handhold + foothold, basic size tiers)
2. [x] Factor 2: Hold density (handhold + foothold counts)
3. [x] Factor 3: Inter-hold distances (average distance score)
4. [x] Factor 4: Wall incline (manual user input)
5. [x] Combined scoring with initial weights (0.35, 0.25, 0.20, 0.20)
6. [x] Grade mapping (score → V0-V12)
7. [ ] User feedback collection (actual difficulty vs predicted)

**DO NOT implement yet**:

- ❌ Hold slant angle adjustments (add in Phase 1b)
- ❌ Complexity multipliers (add in Phase 1c)
- ❌ Advanced distance analysis (weighted average, directional)
- ❌ Wall segments (multi-angle routes)

**Success Criteria**:

- Predictions complete in < 100ms
- No crashes on edge cases (0 holds, missing data)
- User feedback system functional
- Detailed logging for calibration

**Deployment Strategy**:

- Deploy to staging environment
- Test with 20-50 routes
- Collect initial feedback
- Don't expect high accuracy yet (target: ≥40% exact match)

### Phase 1b: Calibration and Refinement (Weeks 5-8)

**Goal**: Improve accuracy through calibration and add slant consideration

**Activities**:

1. **Analyze Feedback Patterns**:
   - Which grades are consistently over/under-predicted?
   - Do slabs/overhangs have systematic bias?
   - Are sparse vs dense routes mis-predicted?

2. **Calibrate Factor Weights**:
   - Adjust weights (0.35, 0.25, 0.20, 0.20) based on correlation with feedback
   - Example: If hold types dominate difficulty, increase Factor 1 weight to 0.40

3. **Calibrate Difficulty Tiers**:
   - Adjust hold type scores (crimps = 10, jugs = 1, etc.)
   - Adjust size thresholds (500px², 1500px², etc.)
   - Adjust wall incline scores (slab = 3.0, overhang = 9.0, etc.)

4. **Add Hold Slant Adjustments**:
   - Implement slant angle detection or manual annotation
   - Apply slant multipliers to hold scores
   - Validate improvement in accuracy

**Success Criteria**:

- ✅ Exact match: ≥60%
- ✅ Within ±1 grade: ≥80%
- ✅ No systematic bias by wall angle
- ✅ User satisfaction >3.0/5.0

### Phase 1b Calibration Workflow

#### 7-Step Calibration Procedure

##### Step 1: Deploy with Logging

- Deploy Phase 1a MVP with comprehensive logging enabled
- Ensure all predictions log: wall angle, factor scores, final prediction
- Verify user feedback collection is functional

##### Step 2: Data Collection (Minimum Requirements)

- **Sample size**: ≥100 analyzed routes with user feedback
- **Wall angle coverage**: ≥3 angle categories with 20+ samples each
- **Collection period**: ≥2 weeks to capture diverse route types
- **Quality check**: Exclude obvious outliers (user error, mislabeled routes)

##### Step 3: Bias Analysis

**Prerequisites**: Create the `calibration_logs` view before running calibration queries.

The view joins `analyses` with `feedback`, extracts factor scores from the `features_extracted` JSON column (populated by `src/grade_prediction_mvp.py`), and converts V-grades to numeric values.

**Migration: Create calibration_logs View**

```sql
-- scripts/migrations/create_calibration_logs_view.sql
-- Grade-to-numeric conversion function (SQLite)
-- V0=0, V1=1, ..., V12=12

CREATE VIEW IF NOT EXISTS calibration_logs AS
SELECT
    a.id AS analysis_id,
    a.wall_incline,
    a.predicted_grade,
    f.user_grade,
    a.confidence_score,
    a.created_at,

    -- Extract factor scores from features_extracted JSON
    -- Note: features_extracted stores the score_breakdown from grade_prediction_mvp.py
    CAST(json_extract(a.features_extracted, '$.hold_difficulty') AS REAL) AS factor1_score,
    CAST(json_extract(a.features_extracted, '$.hold_density') AS REAL) AS factor2_score,
    CAST(json_extract(a.features_extracted, '$.distance') AS REAL) AS factor3_score,
    CAST(json_extract(a.features_extracted, '$.wall_incline') AS REAL) AS factor4_score,
    CAST(json_extract(a.features_extracted, '$.final_score') AS REAL) AS final_score,

    -- Convert predicted_grade (V0-V12) to numeric
    CAST(REPLACE(a.predicted_grade, 'V', '') AS INTEGER) AS predicted_grade_num,

    -- Convert user_grade (V0-V12) to numeric (NULL if not provided)
    CASE
        WHEN f.user_grade IS NOT NULL AND f.user_grade LIKE 'V%'
        THEN CAST(REPLACE(f.user_grade, 'V', '') AS INTEGER)
        ELSE NULL
    END AS user_grade_num,

    -- Prediction error (positive = over-predicted, negative = under-predicted)
    CASE
        WHEN f.user_grade IS NOT NULL AND f.user_grade LIKE 'V%'
        THEN CAST(REPLACE(a.predicted_grade, 'V', '') AS INTEGER) -
             CAST(REPLACE(f.user_grade, 'V', '') AS INTEGER)
        ELSE NULL
    END AS error,

    f.is_accurate,
    f.comments

FROM analyses a
LEFT JOIN feedback f ON a.id = f.analysis_id
WHERE a.features_extracted IS NOT NULL;
```

**Apply the migration:**

```bash
sqlite3 bouldering_analysis.db < scripts/migrations/create_calibration_logs_view.sql
```

**Calibration Queries** (using the view):

```sql
-- Error distribution by wall angle
SELECT
    wall_incline,
    COUNT(*) as total,
    SUM(CASE WHEN error > 0 THEN 1 ELSE 0 END) as over_predicted,
    SUM(CASE WHEN error < 0 THEN 1 ELSE 0 END) as under_predicted,
    SUM(CASE WHEN error = 0 THEN 1 ELSE 0 END) as accurate,
    AVG(ABS(error)) as mae
FROM calibration_logs
WHERE user_grade_num IS NOT NULL
GROUP BY wall_incline;

-- Factor-specific accuracy by angle
SELECT
    wall_incline,
    AVG(factor1_score) as avg_hold_difficulty,
    AVG(factor2_score) as avg_hold_density,
    AVG(factor3_score) as avg_distance,
    AVG(factor4_score) as avg_wall_incline
FROM calibration_logs
WHERE user_grade_num IS NOT NULL
GROUP BY wall_incline;
```

**Schema Dependencies:**

- `analyses.features_extracted`: JSON field populated by `src/grade_prediction_mvp.py` with keys: `hold_difficulty`, `hold_density`, `distance`, `wall_incline`, `final_score`
- `feedback.user_grade`: V-grade string (e.g., "V5") provided by user

##### Step 4: Identify Adjustment Targets

| Bias Pattern | Primary Adjustment | Secondary Check |
| :----------: | :----------------: | :-------------: |
| Slab over-predicted | Increase slab foothold weight | Check Factor 1 hold scores |
| Overhang under-predicted | Increase overhang handhold weight | Check Factor 4 wall score |
| All angles biased similarly | Adjust factor weights | Check grade mapping thresholds |
| High variance, no pattern | Collect more data | Check detection quality |

##### Step 5: Apply Adjustments

Update `src/cfg/user_config.yaml`:

```yaml
grade_prediction:
  wall_angle_weights:
    slab:              { handhold: 0.40, foothold: 0.60 }  # Adjust if slab biased
    vertical:          { handhold: 0.55, foothold: 0.45 }
    slight_overhang:   { handhold: 0.60, foothold: 0.40 }
    moderate_overhang: { handhold: 0.70, foothold: 0.30 }
    steep_overhang:    { handhold: 0.75, foothold: 0.25 }  # Adjust if overhang biased
```

Adjustment increments: ±0.05 per iteration (e.g., 0.60 → 0.65)

##### Step 6: Validation

- Deploy adjusted config to staging
- Test with 20-30 routes across affected angle categories
- Compare accuracy metrics before/after
- Acceptance criteria: ≥5% improvement in affected category, no regression elsewhere

##### Step 7: Independent Calibration Trigger (If Needed)

Consider splitting Factor 1 and Factor 2 weights only if:

- Shared weight adjustment improves one factor but degrades another
- Factor-specific bias ≥15% for a wall angle category
- Sample size ≥150 routes with feedback

If triggered, create factor-specific config:

```yaml
# Phase 1b+ (if required) - NOT default
grade_prediction:
  factor1_wall_angle_weights:  # Hold difficulty weights
    slab: { handhold: 0.40, foothold: 0.60 }
    # ...
  factor2_wall_angle_weights:  # Hold density weights
    slab: { handhold: 0.45, foothold: 0.55 }  # Different from Factor 1
    # ...
```

#### Rollback Strategy

If calibration degrades accuracy:

1. Revert `user_config.yaml` to previous version (git checkout)
2. Redeploy with original config
3. Analyze what went wrong (insufficient data? wrong adjustment direction?)
4. Wait for more data before re-attempting

### Wall-Angle Weight Configuration

#### Shared Configuration Structure (Phase 1b Default)

```yaml
# src/cfg/user_config.yaml
grade_prediction:
  # Shared wall-angle weights used by Factor 1 and Factor 2
  wall_angle_weights:
    slab:
      handhold: 0.40
      foothold: 0.60
    vertical:
      handhold: 0.55
      foothold: 0.45
    slight_overhang:
      handhold: 0.60
      foothold: 0.40
    moderate_overhang:
      handhold: 0.70
      foothold: 0.30
    steep_overhang:
      handhold: 0.75
      foothold: 0.25
```

**Access Pattern**:

```python
# Both factors use the same config path
def get_wall_angle_weights(wall_angle: str) -> tuple[float, float]:
    config = load_config()
    weights = config["grade_prediction"]["wall_angle_weights"][wall_angle]
    return weights["handhold"], weights["foothold"]

# Factor 1
hand_weight, foot_weight = get_wall_angle_weights(wall_angle)
factor1_score = (handhold_score * hand_weight) + (foothold_score * foot_weight)

# Factor 2 (same weights)
hand_weight, foot_weight = get_wall_angle_weights(wall_angle)
factor2_score = (handhold_density * hand_weight) + (foothold_density * foot_weight)
```

#### Independent Weights Structure (Phase 1b+ If Required)

Only implement if shared weights create conflicting optimization (see Decision Criteria in Factor 2 documentation):

```yaml
# Future-state structure - NOT default implementation
grade_prediction:
  # Factor 1: Hold difficulty weights
  factor1_wall_angle_weights:
    slab:              { handhold: 0.40, foothold: 0.60 }
    vertical:          { handhold: 0.55, foothold: 0.45 }
    slight_overhang:   { handhold: 0.60, foothold: 0.40 }
    moderate_overhang: { handhold: 0.70, foothold: 0.30 }
    steep_overhang:    { handhold: 0.75, foothold: 0.25 }

  # Factor 2: Hold density weights (may differ)
  factor2_wall_angle_weights:
    slab:              { handhold: 0.45, foothold: 0.55 }
    vertical:          { handhold: 0.55, foothold: 0.45 }
    slight_overhang:   { handhold: 0.58, foothold: 0.42 }
    moderate_overhang: { handhold: 0.68, foothold: 0.32 }
    steep_overhang:    { handhold: 0.72, foothold: 0.28 }
```

#### Fallback Defaults

If config is missing or invalid, use hardcoded defaults:

```python
DEFAULT_WALL_ANGLE_WEIGHTS = {
    "slab": (0.40, 0.60),
    "vertical": (0.55, 0.45),
    "slight_overhang": (0.60, 0.40),
    "moderate_overhang": (0.70, 0.30),
    "steep_overhang": (0.75, 0.25),
}
```

### Calibration Logging

#### Per-Prediction Breakdown Logging

Every prediction must log structured data for calibration analysis:

```python
# Log format: JSON for programmatic analysis
calibration_log = {
    "timestamp": "2024-01-15T14:30:00Z",
    "analysis_id": "uuid-here",

    # Input features
    "wall_incline": "moderate_overhang",
    "handhold_count": 8,
    "foothold_count": 4,
    "image_height": 1200,

    # Factor scores
    "factor1_score": 7.2,
    "factor2_score": 5.8,
    "factor3_score": 6.1,
    "factor4_score": 9.0,

    # Weights applied
    "factor_weights": {
        "hold_difficulty": 0.35,
        "hold_density": 0.25,
        "distance": 0.20,
        "wall_incline": 0.20
    },
    "wall_angle_weights": {
        "handhold": 0.70,
        "foothold": 0.30
    },

    # Scoring
    "base_score": 6.89,
    "final_score": 6.89,
    "predicted_grade": "V6",
    "confidence": 0.82,

    # Factor contributions (for analysis)
    "contributions": {
        "factor1_contribution": 2.52,  # 7.2 * 0.35
        "factor2_contribution": 1.45,  # 5.8 * 0.25
        "factor3_contribution": 1.22,  # 6.1 * 0.20
        "factor4_contribution": 1.80   # 9.0 * 0.20
    },

    # User feedback (when available)
    "user_grade": "V7",  # null if not provided
    "error": -1,  # predicted - actual (null if no feedback)
    "is_accurate": false  # |error| <= 0
}

logger.info("CALIBRATION_LOG: %s", json.dumps(calibration_log))
```

#### Aggregation Queries for Calibration

**Note**: These queries use the `calibration_logs` view created in Step 3. The view extracts factor scores from `features_extracted` JSON and converts V-grades to numeric values.

**Error Distribution by Angle**:

```sql
SELECT
    wall_incline,
    COUNT(*) as sample_count,
    AVG(error) as mean_error,  -- Negative = under-predicted
    AVG(ABS(error)) as mae,
    SUM(CASE WHEN error > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_over,
    SUM(CASE WHEN error < 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_under,
    SUM(CASE WHEN error = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_exact
FROM calibration_logs
WHERE user_grade_num IS NOT NULL
GROUP BY wall_incline
ORDER BY mae DESC;
```

**Factor-Specific Accuracy by Angle**:

```sql
-- Note: SQLite doesn't have CORR(). Use this simplified query for factor averages.
-- For correlation analysis, export to Python/pandas.
SELECT
    wall_incline,
    AVG(factor1_score) as avg_f1,
    AVG(factor2_score) as avg_f2,
    AVG(factor3_score) as avg_f3,
    AVG(factor4_score) as avg_f4,
    AVG(final_score) as avg_final,
    AVG(error) as avg_error
FROM calibration_logs
WHERE user_grade_num IS NOT NULL
GROUP BY wall_incline;
```

**Bias Direction Analysis**:

```sql
-- Identify angles with systematic over/under-prediction
SELECT
    wall_incline,
    CASE
        WHEN AVG(error) > 0.3 THEN 'OVER_PREDICTED'
        WHEN AVG(error) < -0.3 THEN 'UNDER_PREDICTED'
        ELSE 'BALANCED'
    END as bias_direction,
    AVG(error) as mean_error,
    COUNT(*) as sample_count
FROM calibration_logs
WHERE user_grade_num IS NOT NULL
GROUP BY wall_incline
HAVING COUNT(*) >= 20;  -- Minimum sample size
```

### Phase 1c: Advanced Features (Weeks 9-12, Optional)

**Goal**: Add complexity multipliers if accuracy targets met

**Prerequisites**:

- ✅ Phase 1b accuracy criteria achieved
- ✅ Systematic under-prediction of complex routes identified
- ✅ Resources available for additional development

**Implement**:

1. ✅ Wall segment tracking (multi-angle routes)
2. ✅ Wall transition detection and multiplier
3. ✅ Hold type entropy calculation and multiplier
4. ✅ Combined multiplier application
5. ✅ A/B testing (with vs without multipliers)

**Success Criteria**:

- ✅ Accuracy improves by ≥5% on complex routes
- ✅ No regression on simple routes
- ✅ Overall exact match: ≥70%

## Technical Implementation Guidance

### Database Schema Updates

```python
# Add to Analysis model (in src/models.py)
class Analysis(Base):
    # ... existing fields ...

    # Phase 1a: Basic wall incline
    wall_incline = Column(String(20), default='vertical')

    # Phase 1b: Store slant data if detected
    # (Can reuse features_extracted JSON field)

    # Phase 1c: Wall segments for complex routes
    wall_segments = Column(JSON, nullable=True)
    # Format: [{"angle": "vertical", "proportion": 0.6}, ...]
```

**Migration**:

```bash
# Create migration
flask db migrate -m "Add wall_incline and wall_segments to Analysis"

# Review migration file, then apply
flask db upgrade
```

### Configuration Management

Store calibratable values in `user_config.yaml`:

```yaml
grade_prediction:
  algorithm_version: "v2"

  # Factor weights
  factor_weights:
    hold_difficulty: 0.35
    hold_density: 0.25
    hold_distance: 0.20
    wall_incline: 0.20

  # Hold type difficulty scores
  hold_type_scores:
    crimp: 10
    pocket: 10
    sloper: 7
    pinch: 7
    start_hold: 4
    top_out_hold: 4
    jug: 1

  # Size thresholds (in pixels²)
  size_thresholds:
    crimp_extra_small: 500
    crimp_small: 1000
    crimp_medium: 2000
    sloper_small: 1500
    sloper_medium: 3000
    jug_small: 2000
    foothold_very_small: 800
    foothold_small: 1500
    foothold_medium: 3000

  # Wall incline scores
  wall_incline_scores:
    slab: 3.0
    vertical: 6.0
    slight_overhang: 7.5
    moderate_overhang: 9.0
    steep_overhang: 11.0

  # Foothold weighting by wall angle
  foothold_weights:
    slab: {hand: 0.35, foot: 0.65}
    vertical: {hand: 0.55, foot: 0.45}
    slight_overhang: {hand: 0.60, foot: 0.40}
    moderate_overhang: {hand: 0.70, foot: 0.30}
    steep_overhang: {hand: 0.75, foot: 0.25}

  # Phase 1c: Complexity multipliers
  complexity_multipliers:
    enabled: false  # Feature flag
    transition_max: 1.5
    variability_max: 1.3
```

**Benefits**:

- Easy to adjust without code changes
- Version control for calibration history
- A/B testing different configurations
- Rollback if calibration hurts accuracy

### Prediction Function Structure

```python
# src/grade_prediction.py (new file)

def predict_grade_v2(
    features: dict,
    detected_holds: list,
    wall_incline: str = 'vertical',
    wall_segments: list = None,
    config: dict = None
) -> tuple[str, float, dict]:
    """
    Phase 1 grade prediction algorithm.

    Args:
        features: Extracted features from image analysis
        detected_holds: List of DetectedHold objects
        wall_incline: Wall angle category
        wall_segments: Optional list of wall segments
        config: Optional config override (for A/B testing)

    Returns:
        tuple: (predicted_grade, confidence, breakdown_dict)
    """
    # Load config
    cfg = config or load_config()

    # Separate handholds and footholds
    handholds = [h for h in detected_holds if h.hold_type.name != 'foot-hold']
    footholds = [h for h in detected_holds if h.hold_type.name == 'foot-hold']

    # Factor 1: Hold difficulty
    factor1_score = calculate_factor1_score(
        handholds, footholds, wall_incline, cfg
    )

    # Factor 2: Hold density
    factor2_score = calculate_factor2_score(
        len(handholds), len(footholds), wall_incline, cfg
    )

    # Factor 3: Hold distances
    distances = calculate_inter_hold_distances(detected_holds)
    factor3_score = calculate_factor3_score(distances, cfg)

    # Factor 4: Wall incline
    factor4_score = cfg['wall_incline_scores'][wall_incline]

    # Combine with weights
    weights = cfg['factor_weights']
    base_score = (
        factor1_score * weights['hold_difficulty'] +
        factor2_score * weights['hold_density'] +
        factor3_score * weights['hold_distance'] +
        factor4_score * weights['wall_incline']
    )

    # Phase 1c: Apply complexity multipliers if enabled
    if cfg.get('complexity_multipliers', {}).get('enabled', False):
        base_score = apply_complexity_multipliers(
            base_score, wall_segments, handholds, cfg
        )

    # Map to V-grade
    predicted_grade = map_score_to_grade(base_score)

    # Calculate confidence
    confidence = calculate_confidence(base_score, detected_holds, cfg)

    # Breakdown for explainability
    breakdown = {
        'base_score': base_score,
        'predicted_grade': predicted_grade,
        'confidence': confidence,
        'factors': {
            'hold_difficulty': factor1_score,
            'hold_density': factor2_score,
            'hold_distance': factor3_score,
            'wall_incline': factor4_score
        },
        'weights': weights,
        'algorithm_version': cfg['algorithm_version']
    }

    return predicted_grade, confidence, breakdown
```

### Integration with Main Application

```python
# src/main.py - Update analyze_image() function

def analyze_image(...):
    # ... existing hold detection code ...

    # NEW: Get wall incline from form
    wall_incline = request.form.get('wall_incline', 'vertical')

    # NEW: Use v2 algorithm
    predicted_grade, confidence, breakdown = predict_grade_v2(
        features=features,
        detected_holds=detected_holds,
        wall_incline=wall_incline
    )

    # Store prediction breakdown
    analysis.features_extracted = json.dumps(breakdown)

    # ... rest of existing code ...
```

## Calibration Strategy

### Data Collection

**Log for every prediction**:

- Input features (hold counts, types, sizes, distances, wall angle)
- Predicted grade and confidence
- Factor scores (breakdown)
- User feedback (if provided)

**Store in database or log files**:

```python
# Add to feedback mechanism
class UserFeedback(Base):
    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'))
    predicted_grade = Column(String(10))
    user_perceived_grade = Column(String(10))
    feedback_text = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Analysis Approach

**Weekly calibration review**:

1. **Aggregate Feedback**:

   ```sql
   -- Over-predictions: predicted > perceived
   SELECT predicted_grade, user_perceived_grade, COUNT(*)
   FROM user_feedback
   WHERE predicted_grade > user_perceived_grade
   GROUP BY predicted_grade, user_perceived_grade;

   -- Under-predictions: predicted < perceived
   SELECT predicted_grade, user_perceived_grade, COUNT(*)
   FROM user_feedback
   WHERE predicted_grade < user_perceived_grade
   GROUP BY predicted_grade, user_perceived_grade;
   ```

2. **Identify Systematic Bias**:
   - Are slabs consistently over-predicted? → Reduce slab wall score
   - Are overhangs under-predicted? → Increase overhang wall score
   - Are crimp routes under-predicted? → Increase crimp difficulty score
   - Are routes with few footholds under-predicted? → Increase foothold density penalty

3. **Adjust Configuration**:
   - Update values in `user_config.yaml`
   - Deploy updated config (no code changes needed)
   - Monitor next week's predictions

4. **A/B Testing** (advanced):
   - Split traffic 50/50 between old and new config
   - Compare accuracy metrics
   - Keep better-performing config

### Iteration Cycle

```text
Week 1-2: Deploy Phase 1a → Collect feedback
Week 3: Analyze patterns → Identify biases
Week 4: Adjust config → Deploy Phase 1b
Week 5-6: Collect feedback on Phase 1b
Week 7: Analyze accuracy improvement
Week 8: Fine-tune or proceed to Phase 1c
```

**Target**: Achieve Phase 1b success criteria (≥60% exact, ≥80% ±1) by Week 8.

## Common Pitfalls and Solutions

### Pitfall 1: Overfitting to Few Data Points

**Problem**: Adjusting calibration based on 5-10 feedback samples

**Solution**:

- Wait for ≥50 feedback samples before major adjustments
- Look for consistent patterns, not individual outliers
- Use statistical significance tests

### Pitfall 2: Ignoring Edge Cases

**Problem**: Crashes on 0 holds, missing wall angle, etc.

**Solution**:

- Add extensive error handling
- Default to neutral values (wall angle = 'vertical', etc.)
- Log edge cases for review

### Pitfall 3: Circular Calibration

**Problem**: Adjusting Factor 1 to fix Factor 4 issues

**Solution**:

- Calibrate factors independently when possible
- Analyze which factor is actually causing bias
- Use breakdown logging to identify root cause

### Pitfall 4: Complexity Too Early

**Problem**: Implementing complexity multipliers before base model works

**Solution**:

- Strictly follow phased approach
- DO NOT add complexity multipliers in Phase 1a
- Validate base model first

### Pitfall 5: Ignoring Image Resolution Variance

**Problem**: Size thresholds (500px², 1500px²) don't work across different image resolutions

**Solution**:

- Normalize hold sizes by image dimensions
- Or: Separate thresholds per resolution range
- Or: Collect image resolution statistics and adjust

## Testing Strategy

### Unit Tests

```python
# tests/test_grade_prediction.py

def test_factor1_crimps_harder_than_jugs():
    """Verify crimps score higher than jugs."""
    crimps = [create_mock_hold('crimp', area=600) for _ in range(5)]
    jugs = [create_mock_hold('jug', area=2500) for _ in range(5)]

    crimp_score = calculate_handhold_difficulty(crimps)
    jug_score = calculate_handhold_difficulty(jugs)

    assert crimp_score > jug_score

def test_factor2_fewer_holds_harder():
    """Verify fewer holds increases difficulty."""
    score_5_holds = calculate_handhold_density_score(5)
    score_15_holds = calculate_handhold_density_score(15)

    assert score_5_holds > score_15_holds

def test_no_footholds_campusing_penalty():
    """Verify campusing (0 footholds) gets maximum penalty."""
    score = calculate_foothold_density_score(0)
    assert score == 12.0

def test_combined_score_range():
    """Verify final score is in valid range."""
    # Test with various input combinations
    for _ in range(100):
        score = predict_grade_v2(...)
        assert 0 <= score <= 12
```

### Integration Tests

```python
def test_v5_route_prediction():
    """Test prediction on known V5 route."""
    # Create mock route with V5 characteristics
    route = create_mock_v5_route()

    grade, confidence, breakdown = predict_grade_v2(route)

    # Should predict V4-V6 (within ±1)
    assert grade in ['V4', 'V5', 'V6']
```

### Regression Tests

```python
def test_no_regression_on_calibration_routes():
    """Ensure calibration doesn't hurt known-good predictions."""
    # Load routes that were accurately predicted in previous version
    calibration_routes = load_calibration_routes()

    for route in calibration_routes:
        new_grade = predict_grade_v2(route)
        old_grade = route['previous_prediction']

        # New prediction should be within ±1 of old prediction
        assert abs(grade_to_num(new_grade) - grade_to_num(old_grade)) <= 1
```

## Performance Optimization

### Target: < 100ms per prediction

**Optimization strategies**:

1. **Precompute hold features**:
   - Calculate hold areas during detection
   - Store in DetectedHold model
   - Avoid recalculating in prediction

2. **Cache config loading**:
   - Load `user_config.yaml` once at startup
   - Cache in memory
   - Only reload when config changes

3. **Optimize distance calculations**:
   - Use NumPy for vectorized distance computation
   - Avoid nested loops

4. **Profile slow sections**:

   ```python
   import cProfile
   cProfile.run('predict_grade_v2(...)')
   ```

## Deployment Checklist

**Before deploying Phase 1a** - COMPLETED ✅:

- [x] All unit tests passing
- [x] Database migration tested
- [ ] User feedback mechanism functional
- [x] Logging infrastructure in place
- [x] Configuration file validated
- [x] Edge cases handled (0 holds, missing data)

**Before deploying Phase 1b**:

- [ ] ≥50 feedback samples collected
- [ ] Calibration analysis complete
- [ ] New config values tested in staging
- [ ] Regression tests passing

**Before deploying Phase 1c**:

- [ ] Phase 1b accuracy criteria met
- [ ] Complexity multipliers tested independently
- [ ] A/B test results favorable
- [ ] No regression on simple routes

## Summary

Phase 1 implementation requires:

1. ✅ **Staged approach**: 1a (basic) → 1b (calibration) → 1c (complexity)
2. ✅ **Configuration-driven**: Easy to adjust without code changes
3. ✅ **Data-driven calibration**: Collect feedback, analyze patterns, iterate
4. ✅ **Avoid complexity too early**: Get basic model working first
5. ✅ **Extensive testing**: Unit, integration, regression tests

**Key Success Factor**: Treat all specification values as starting hypotheses. Calibrate with real data.

**Next Steps**:

1. Implement Phase 1a (basic 4-factor model)
2. Deploy to staging and collect feedback
3. Calibrate and refine (Phase 1b)
4. Optionally add complexity (Phase 1c)
5. Only then consider Phase 1.5 (personas) or Phase 2 (video)
