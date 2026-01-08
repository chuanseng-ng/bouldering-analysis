# Phase 1 Implementation Notes

## Overview

This document provides practical technical guidance for implementing the Phase 1 grade prediction algorithm, including calibration strategies, common pitfalls, and iteration approaches.

## Staged Implementation Approach

### Phase 1a: Basic 4-Factor Model (Weeks 1-4)

**Goal**: Working prediction system, even if accuracy is initially moderate

**Implement**:
1. ✅ Factor 1: Hold difficulty (handhold + foothold, basic size tiers)
2. ✅ Factor 2: Hold density (handhold + foothold counts)
3. ✅ Factor 3: Inter-hold distances (average distance score)
4. ✅ Factor 4: Wall incline (manual user input)
5. ✅ Combined scoring with initial weights (0.35, 0.25, 0.20, 0.20)
6. ✅ Grade mapping (score → V0-V12)
7. ✅ User feedback collection (actual difficulty vs predicted)

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

**Before deploying Phase 1a**:
- [ ] All unit tests passing
- [ ] Database migration tested
- [ ] User feedback mechanism functional
- [ ] Logging infrastructure in place
- [ ] Configuration file validated
- [ ] Edge cases handled (0 holds, missing data)

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

