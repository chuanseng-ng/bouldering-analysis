# Phase 1a: MVP Route-Based Grade Prediction - Simplified Specification

## Executive Summary

This document defines the **Minimum Viable Product (MVP)** for Phase 1 route-based grade prediction - a simplified, achievable implementation that can be completed in **4-6 weeks** and provides immediate value.

**Key Principle**: Start simple, validate early, iterate based on real data.

## ⚠️ CRITICAL: What's Different from Full Phase 1 Spec

The full Phase 1 specification ([phase1_route_based_prediction.md](../phase1_route_based_prediction.md)) includes many advanced features that are **DEFERRED** in Phase 1a:

**Phase 1a MVP (THIS DOCUMENT)**:

- 4 basic factors with simplified scoring
- Constant foothold weighting (60/40 split)
- NO slanted hold detection
- NO complexity multipliers
- NO wall segments (single angle only)
- Manual wall angle input

**Deferred to Phase 1b** (Calibration & Refinement):

- Wall-angle-dependent foothold weighting
- Slanted hold orientation adjustments
- Advanced foothold scarcity multipliers
- Empirical threshold calibration

**Deferred to Phase 1c** (Advanced Features):

- Complexity multipliers (transitions, variability)
- Multi-segment wall support
- Entropy-based hold type analysis

---

## Objectives

### Primary Goal

Replace the current simplified grade prediction with a **basic multi-factor algorithm** that:
1. Considers hold types, sizes, counts, and spacing
2. Incorporates wall angle
3. Treats footholds as important (not afterthought)
4. Provides explainable predictions

### Success Criteria

- ✅ **Accuracy**: ≥50% exact match, ≥75% within ±1 grade (lower bar for MVP)
- ✅ **Performance**: <100ms per route prediction
- ✅ **Stability**: No crashes, handles edge cases
- ✅ **Explainability**: Score breakdown available
- ✅ **Deployable**: Ready for user feedback collection

### Non-Goals for Phase 1a

- ❌ Perfect accuracy (that comes from calibration)
- ❌ Complex difficulty models (keep it simple)
- ❌ Automatic hold orientation detection
- ❌ Multi-segment wall angle support

---

## Algorithm Design

### Simplified Scoring System

```text
Base Score = f(Hold Difficulty, Hold Density, Distance, Wall Incline)

Final Score = Base Score  (NO complexity multipliers in MVP)

V-Grade = map(Final Score, 0-12 range)
```

**Weighting** (same as full spec):

```text
Base_Score = (
    Hold_Difficulty_Score × 0.35 +
    Hold_Density_Score × 0.25 +
    Distance_Score × 0.20 +
    Wall_Incline_Score × 0.20
)
```

---

## Factor 1: Hold Type & Size Analysis (SIMPLIFIED)

### Objective

Score holds based on type and size, considering both handholds and footholds.

### ⚠️ SIMPLIFICATIONS FOR MVP

1. **NO slanted hold detection** - Assume all holds are horizontal
2. **Simplified size thresholds** - Use coarse categories only
3. **Constant foothold weighting** - 60% handholds, 40% footholds (no wall angle dependency)

### Handhold Difficulty Scoring

#### Hold Type Base Scores

```python
HANDHOLD_BASE_SCORES = {
    'crimp': 10,
    'pocket': 10,
    'sloper': 7,
    'pinch': 7,
    'start-hold': 4,
    'top-out-hold': 4,
    'jug': 1
}
```

#### Size Adjustment (SIMPLIFIED)

Calculate hold area:

```python
hold_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
```

**Simplified size categories** (only 3 instead of 5):

```python
def get_size_modifier(hold_type: str, area: float) -> float:
    """
    Simplified size modifier - only 3 categories.
    
    Returns modifier to add to base score.
    """
    if hold_type in ['crimp', 'pocket']:
        if area < 1000:    # Small
            return 2
        elif area < 2500:  # Medium
            return 1
        else:              # Large
            return 0
    
    elif hold_type == 'sloper':
        if area < 2000:    # Small
            return 2
        else:              # Large
            return 0
    
    elif hold_type == 'jug':
        if area < 2000:    # Small jug
            return 1
        else:              # True jug
            return 0
    
    else:  # pinch, start, top-out
        return 0
```

#### Handhold Difficulty Calculation

```python
def calculate_handhold_difficulty(handholds: list) -> float:
    """
    Calculate handhold difficulty score.
    
    SIMPLIFIED: No hard_hold_ratio multiplier for MVP.
    """
    if len(handholds) == 0:
        return 6.0  # Default neutral
    
    total_score = 0
    for hold in handholds:
        base_score = HANDHOLD_BASE_SCORES.get(hold.type, 5)
        size_modifier = get_size_modifier(hold.type, hold.area)
        total_score += (base_score + size_modifier)
    
    # Normalize by count
    avg_difficulty = total_score / len(handholds)
    
    # Clamp to 1-13 range
    return max(1.0, min(13.0, avg_difficulty))
```

### Foothold Difficulty Scoring (SIMPLIFIED)

```python
def calculate_foothold_difficulty(footholds: list) -> float:
    """
    Calculate foothold difficulty score.
    
    SIMPLIFIED: Basic size categories, simple scarcity multiplier.
    """
    if len(footholds) == 0:
        # NO FOOTHOLDS = CAMPUSING
        return 12.0
    
    # Size-based scoring
    total_score = 0
    for fh in footholds:
        if fh.area < 1000:     # Small
            total_score += 8
        elif fh.area < 2000:   # Medium
            total_score += 5
        else:                  # Large
            total_score += 2
    
    avg_difficulty = total_score / len(footholds)
    
    # Simplified scarcity multiplier
    if len(footholds) <= 2:
        scarcity = 1.4
    elif len(footholds) <= 4:
        scarcity = 1.2
    else:
        scarcity = 1.0
    
    return avg_difficulty * scarcity
```

### Combined Hold Difficulty

**SIMPLIFIED: Constant 60/40 weighting** (no wall angle dependency):

```python
def calculate_combined_hold_difficulty(handholds: list, footholds: list) -> float:
    """
    Combine handhold and foothold difficulty.
    
    SIMPLIFIED MVP: Use constant 60/40 weighting.
    """
    handhold_score = calculate_handhold_difficulty(handholds)
    foothold_score = calculate_foothold_difficulty(footholds)
    
    # Constant weights for MVP
    HANDHOLD_WEIGHT = 0.60
    FOOTHOLD_WEIGHT = 0.40
    
    combined = (handhold_score * HANDHOLD_WEIGHT) + (foothold_score * FOOTHOLD_WEIGHT)
    
    return combined
```

---

## Factor 2: Hold Count Analysis (SIMPLIFIED)

### Objective

Assess difficulty based on number of available holds.

### Handhold Density

```python
def calculate_handhold_density_score(handhold_count: int) -> float:
    """
    Logarithmic relationship: fewer holds = harder.
    
    Same as full spec.
    """
    if handhold_count == 0:
        return 12.0
    
    score = 12 - (math.log2(handhold_count) * 2.5)
    return max(0, min(12, score))
```

### Foothold Density (SIMPLIFIED)

```python
def calculate_foothold_density_score(foothold_count: int) -> float:
    """
    Foothold scarcity scoring.
    
    SIMPLIFIED: Coarse categories instead of fine-grained.
    """
    if foothold_count == 0:
        return 12.0
    elif foothold_count <= 2:
        return 9.0
    elif foothold_count <= 5:
        return 6.0
    elif foothold_count <= 8:
        return 3.5
    else:
        return 1.5
```

### Combined Hold Density

**SIMPLIFIED: Constant 60/40 weighting**:

```python
def calculate_combined_hold_density(handhold_count: int, foothold_count: int) -> float:
    """
    Combine handhold and foothold density.
    
    SIMPLIFIED MVP: Use constant 60/40 weighting.
    """
    handhold_density = calculate_handhold_density_score(handhold_count)
    foothold_density = calculate_foothold_density_score(foothold_count)
    
    # Constant weights for MVP
    combined = (handhold_density * 0.60) + (foothold_density * 0.40)
    
    return combined
```

---

## Factor 3: Hold Distance Analysis (SIMPLIFIED)

### Objective

Measure spacing between holds to assess reach difficulty.

### Distance Calculation

```python
def calculate_hold_distances(holds: list, image_height: float) -> dict:
    """
    Calculate sequential distances between holds.
    
    Same as full spec - this part is simple enough.
    """
    if len(holds) < 2:
        return {
            'avg_distance': 0,
            'max_distance': 0,
            'normalized_avg': 0,
            'normalized_max': 0
        }
    
    # Sort by y-coordinate
    sorted_holds = sorted(holds, key=lambda h: h.bbox_y1)
    
    distances = []
    for i in range(len(sorted_holds) - 1):
        h1, h2 = sorted_holds[i], sorted_holds[i + 1]
        
        # Calculate centers
        x1 = (h1.bbox_x1 + h1.bbox_x2) / 2
        y1 = (h1.bbox_y1 + h1.bbox_y2) / 2
        x2 = (h2.bbox_x1 + h2.bbox_x2) / 2
        y2 = (h2.bbox_y1 + h2.bbox_y2) / 2
        
        # Euclidean distance
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(dist)
    
    avg_dist = statistics.mean(distances)
    max_dist = max(distances)
    
    return {
        'avg_distance': avg_dist,
        'max_distance': max_dist,
        'normalized_avg': avg_dist / image_height,
        'normalized_max': max_dist / image_height
    }
```

### Distance Scoring (SIMPLIFIED)

```python
def calculate_distance_score(distance_metrics: dict) -> float:
    """
    Score based on hold spacing.
    
    SIMPLIFIED: Coarse thresholds, simple formula.
    """
    if len(distance_metrics.get('distances', [])) == 0:
        return 12.0  # No holds to measure = campusing
    
    normalized_avg = distance_metrics['normalized_avg']
    normalized_max = distance_metrics['normalized_max']
    
    # Average distance component (0-8)
    if normalized_avg < 0.12:
        avg_component = 1
    elif normalized_avg < 0.20:
        avg_component = 3
    elif normalized_avg < 0.30:
        avg_component = 5
    else:
        avg_component = 8
    
    # Max distance component (crux bonus: 0-4)
    if normalized_max < 0.25:
        max_component = 0
    elif normalized_max < 0.40:
        max_component = 2
    else:
        max_component = 4
    
    return min(avg_component + max_component, 12)
```

### Combined Distance Score

**SIMPLIFIED: Constant 60/40 weighting**:

```python
def calculate_combined_distance_score(handholds: list, footholds: list, image_height: float) -> float:
    """
    Combine handhold and foothold distance scores.
    
    SIMPLIFIED MVP: Use constant 60/40 weighting.
    """
    handhold_distances = calculate_hold_distances(handholds, image_height)
    foothold_distances = calculate_hold_distances(footholds, image_height)
    
    handhold_score = calculate_distance_score(handhold_distances)
    foothold_score = calculate_distance_score(foothold_distances)
    
    # Constant weights for MVP
    combined = (handhold_score * 0.60) + (foothold_score * 0.40)
    
    return combined
```

---

## Factor 4: Wall Incline Analysis (SIMPLIFIED)

### Objective

Account for wall angle impact on difficulty.

### Wall Incline Categories

```python
WALL_INCLINE_CATEGORIES = {
    'slab': 0.70,           # Easier
    'vertical': 1.00,       # Baseline
    'slight_overhang': 1.20,
    'moderate_overhang': 1.50,
    'steep_overhang': 1.80  # Harder
}
```

### Wall Incline Scoring

```python
def calculate_wall_incline_score(wall_incline: str) -> float:
    """
    Score based on wall angle.
    
    SIMPLIFIED: Manual input, single angle only (no segments).
    """
    base_score = 6.0  # Neutral baseline
    
    multiplier = WALL_INCLINE_CATEGORIES.get(wall_incline, 1.0)
    
    return base_score * multiplier
```

**User Input**: Add dropdown in UI with 5 options:

- Slab (leaning back)
- Vertical (straight up)
- Slight overhang (leaning forward slightly)
- Moderate overhang (leaning forward)
- Steep overhang (roof-like)

---

## Final Score Calculation

### Combining Factors

```python
def predict_grade_v2_mvp(
    detected_holds: list,
    wall_incline: str = 'vertical',
    image_height: float = 1080
) -> tuple[str, float, dict]:
    """
    MVP grade prediction - simplified Phase 1a.
    
    Args:
        detected_holds: List of DetectedHold objects
        wall_incline: One of 5 categories (manual input)
        image_height: Image height for distance normalization
    
    Returns:
        tuple: (predicted_grade, confidence, score_breakdown)
    """
    # Separate handholds and footholds
    handholds = [h for h in detected_holds if h.hold_type not in ['foot-hold']]
    footholds = [h for h in detected_holds if h.hold_type == 'foot-hold']
    
    # Calculate 4 factor scores
    hold_difficulty_score = calculate_combined_hold_difficulty(handholds, footholds)
    hold_density_score = calculate_combined_hold_density(len(handholds), len(footholds))
    distance_score = calculate_combined_distance_score(handholds, footholds, image_height)
    wall_incline_score = calculate_wall_incline_score(wall_incline)
    
    # Weighted combination
    base_score = (
        hold_difficulty_score * 0.35 +
        hold_density_score * 0.25 +
        distance_score * 0.20 +
        wall_incline_score * 0.20
    )
    
    # NO MULTIPLIERS IN MVP - keep it simple
    final_score = base_score
    
    # Confidence based on detection quality
    confidence_avg = statistics.mean([h.confidence for h in detected_holds])
    confidence = min(confidence_avg / 0.7, 1.0)
    
    # Map to grade
    predicted_grade = map_score_to_grade(final_score)
    
    # Score breakdown
    breakdown = {
        'hold_difficulty': hold_difficulty_score,
        'hold_density': hold_density_score,
        'distance': distance_score,
        'wall_incline': wall_incline_score,
        'base_score': base_score,
        'final_score': final_score,
        'handhold_count': len(handholds),
        'foothold_count': len(footholds),
        'wall_angle': wall_incline
    }
    
    return predicted_grade, confidence, breakdown
```

### Grade Mapping

```python
def map_score_to_grade(score: float) -> str:
    """
    Map score (0-12) to V-grade.
    
    Same as full spec.
    """
    if score < 1.0:
        return "V0"
    elif score < 2.0:
        return "V1"
    elif score < 3.0:
        return "V2"
    elif score < 4.0:
        return "V3"
    elif score < 4.5:
        return "V4"
    elif score < 5.5:
        return "V5"
    elif score < 6.5:
        return "V6"
    elif score < 7.5:
        return "V7"
    elif score < 8.5:
        return "V8"
    elif score < 9.5:
        return "V9"
    elif score < 10.5:
        return "V10"
    elif score < 11.5:
        return "V11"
    else:
        return "V12"
```

---

## Data Requirements

### Database Changes

**Add to Analysis model**:

```python
# In models.py
class Analysis(Base):
    # ... existing fields ...
    wall_incline = Column(String(20), default='vertical')
    # Options: 'slab', 'vertical', 'slight_overhang', 'moderate_overhang', 'steep_overhang'
```

**Migration**:

```python
# Migration script
def upgrade():
    op.add_column('analyses', sa.Column('wall_incline', sa.String(20), default='vertical'))
```

### Configuration File

**Add to `user_config.yaml`**:

```yaml
grade_prediction:
  algorithm_version: "v2_mvp"
  
  # Factor weights
  weights:
    hold_difficulty: 0.35
    hold_density: 0.25
    distance: 0.20
    wall_incline: 0.20
  
  # Hold/foot weighting (constant for MVP)
  handhold_weight: 0.60
  foothold_weight: 0.40
  
  # Size thresholds (pixels²)
  size_thresholds:
    crimp_small: 1000
    crimp_large: 2500
    sloper_small: 2000
    jug_small: 2000
    foothold_small: 1000
    foothold_large: 2000
  
  # Distance thresholds (normalized by image height)
  distance_thresholds:
    close: 0.12
    moderate: 0.20
    wide: 0.30
    crux_small: 0.25
    crux_large: 0.40
  
  # Wall incline multipliers
  wall_incline_multipliers:
    slab: 0.70
    vertical: 1.00
    slight_overhang: 1.20
    moderate_overhang: 1.50
    steep_overhang: 1.80
  
  # Performance settings
  confidence_threshold: 0.5
  default_wall_incline: "vertical"
```

---

## UI Changes

### Upload Form Addition

**In `templates/index.html`**, add after image upload:

```html
<div class="form-group">
    <label for="wall_incline">Wall Angle:</label>
    <select id="wall_incline" name="wall_incline" class="form-control">
        <option value="vertical" selected>Vertical (straight up)</option>
        <option value="slab">Slab (leaning back)</option>
        <option value="slight_overhang">Slight Overhang</option>
        <option value="moderate_overhang">Moderate Overhang</option>
        <option value="steep_overhang">Steep Overhang</option>
    </select>
    <small class="form-text text-muted">
        Select the wall angle for more accurate grade prediction
    </small>
</div>
```

### Results Display

Show score breakdown:

```html
<div class="grade-breakdown">
    <h4>Score Breakdown</h4>
    <ul>
        <li>Hold Difficulty: {{ breakdown.hold_difficulty | round(1) }} (35%)</li>
        <li>Hold Density: {{ breakdown.hold_density | round(1) }} (25%)</li>
        <li>Distance: {{ breakdown.distance | round(1) }} (20%)</li>
        <li>Wall Angle: {{ breakdown.wall_incline | round(1) }} (20%)</li>
    </ul>
    <p><strong>Final Score:</strong> {{ breakdown.final_score | round(1) }}</p>
</div>
```

---

## Implementation Checklist

### Week 1-2: Foundation

- [x] Database migration: Add `wall_incline` field
- [x] Create `src/grade_prediction_mvp.py` module
- [x] Implement hold separation (handholds vs footholds)
- [x] Implement hold dimension calculation
- [x] Write unit tests for utilities

### Week 2-3: Core Factors

- [x] Implement Factor 1: Hold difficulty (simplified)
- [x] Implement Factor 2: Hold density (simplified)
- [x] Implement Factor 3: Distances (simplified)
- [x] Implement Factor 4: Wall incline
- [x] Unit tests for each factor

### Week 3-4: Integration

- [x] Implement `predict_grade_v2_mvp()` main function
- [x] Update `src/main.py` to use new function
- [x] Add configuration loading
- [x] Integration tests

### Week 4-5: UI & Deployment

- [x] Add wall_incline dropdown to upload form
- [x] Update results display with breakdown
- [ ] Test end-to-end flow
- [ ] Deploy to staging

### Week 5-6: Feedback Collection

- [ ] Deploy to production with feature flag
- [ ] Collect user feedback (target: 50+ routes)
- [ ] Monitor prediction distribution
- [ ] Document issues and edge cases

---

## Testing Strategy

### Unit Tests

```python
# tests/test_grade_prediction_mvp.py

def test_handhold_difficulty_all_crimps():
    """Test handhold difficulty with all crimps."""
    # Create mock crimp holds
    crimps = [create_mock_hold('crimp', area=800) for _ in range(6)]
    score = calculate_handhold_difficulty(crimps)
    assert 10 <= score <= 13  # High difficulty

def test_foothold_difficulty_campusing():
    """Test foothold difficulty with no footholds."""
    score = calculate_foothold_difficulty([])
    assert score == 12.0  # Maximum difficulty

def test_combined_hold_difficulty_balanced():
    """Test combined scoring with balanced holds."""
    handholds = [create_mock_hold('jug', area=3000) for _ in range(8)]
    footholds = [create_mock_hold('foot-hold', area=2500) for _ in range(6)]
    score = calculate_combined_hold_difficulty(handholds, footholds)
    assert 2 <= score <= 5  # Low-moderate difficulty

def test_wall_incline_overhang():
    """Test wall incline scoring for overhang."""
    score = calculate_wall_incline_score('moderate_overhang')
    assert score == 9.0  # 6.0 * 1.5

def test_predict_grade_mvp_integration():
    """Test full prediction pipeline."""
    holds = create_mock_route(handhold_count=10, foothold_count=6)
    grade, confidence, breakdown = predict_grade_v2_mvp(
        holds, 
        wall_incline='vertical',
        image_height=1080
    )
    assert grade in ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 
                     'V7', 'V8', 'V9', 'V10', 'V11', 'V12']
    assert 0 <= confidence <= 1.0
    assert 'hold_difficulty' in breakdown
```

### Edge Case Tests

```python
def test_no_holds_detected():
    """Handle case with no holds."""
    grade, confidence, breakdown = predict_grade_v2_mvp([])
    assert grade == 'V0'  # Default to easiest

def test_only_footholds():
    """Handle case with only footholds (unusual)."""
    footholds = [create_mock_hold('foot-hold', area=2000) for _ in range(6)]
    grade, confidence, breakdown = predict_grade_v2_mvp(footholds)
    # Should handle gracefully

def test_campusing_route():
    """Handle route with no footholds."""
    handholds = [create_mock_hold('crimp', area=800) for _ in range(5)]
    grade, confidence, breakdown = predict_grade_v2_mvp(handholds)
    # Should have high difficulty from foothold penalty
```

---

## Calibration Plan

### Phase 1a Calibration Process

**After deployment, iterate weekly**:

1. **Week 6**: Collect ≥50 route feedback samples
2. **Week 7**: Analyze systematic biases
   - Are we over-predicting certain route types?
   - Are we under-predicting certain route types?
   - Which factor contributes most to errors?
3. **Week 8**: Adjust configuration values
   - Modify `size_thresholds` if needed
   - Adjust `distance_thresholds` if needed
   - Tweak `wall_incline_multipliers` if needed
4. **Week 9**: Deploy updated config, monitor improvement
5. **Repeat** until accuracy targets met

### Expected Calibration Adjustments

Based on climbing domain knowledge, expect to adjust:

- **Size thresholds**: ±200-500 pixels
- **Distance thresholds**: ±0.03-0.08 (normalized)
- **Wall incline multipliers**: ±0.10-0.20
- **Factor weights**: ±0.05

### Success Criteria for Phase 1a

After calibration:

- ✅ Accuracy: ≥50% exact match, ≥75% within ±1 grade
- ✅ User satisfaction: >3.0/5.0
- ✅ System stability: No crashes, <100ms predictions
- ✅ Feedback collection: ≥100 route samples collected

**Then proceed to Phase 1b** (wall-angle-dependent weighting, slant detection)

---

## What's Deferred to Phase 1b/1c

### Phase 1b: Calibration & Refinement (2-3 weeks after 1a)

**Add after MVP validated**:

- ✅ Wall-angle-dependent foothold weighting (65% on slabs, 25% on overhangs)
- ✅ Slanted hold detection (if YOLO model updated) OR manual annotation
- ✅ Advanced foothold scarcity multipliers
- ✅ Empirical threshold recalibration

**Prerequisites**:

- Phase 1a deployed, collecting feedback
- ≥100 route samples analyzed
- Systematic biases identified

### Phase 1c: Advanced Features (2-3 weeks after 1b)

**Add after refined model validated**:

- ✅ Complexity multipliers (wall transitions, hold type variability)
- ✅ Multi-segment wall support
- ✅ Shannon entropy for hold type analysis
- ✅ Advanced scoring formulas

**Prerequisites**:

- Phase 1b accuracy ≥60% exact, ≥80% within ±1 grade
- User feedback positive (>3.5/5.0)
- Development resources available

---

## Comparison: MVP vs Full Spec

| Feature | Phase 1a MVP | Full Phase 1 Spec | Complexity Reduction |
| :-----: | :----------: | :---------------: | :------------------: |
| Slanted holds | ❌ Deferred | ✅ Required | HIGH - no detection method |
| Foothold weighting | Constant (60/40) | Wall-angle-dependent | MEDIUM - simpler logic |
| Size categories | 3 categories | 5+ categories | LOW - easier thresholds |
| Complexity multipliers | ❌ Deferred | ✅ Included | HIGH - entire subsystem |
| Wall segments | Single angle | Multi-segment | MEDIUM - simpler UI/data |
| Foothold scarcity | Simple (3 levels) | Complex (7 levels) | LOW - fewer thresholds |
| Hold type entropy | ❌ Deferred | ✅ Included | LOW - simple calculation |
| **Total LOC estimate** | ~400 lines | ~1200 lines | **67% reduction** |
| **Implementation time** | 4-6 weeks | 8-12 weeks | **50% faster** |

---

## Summary

### Phase 1a Delivers

✅ **Functional multi-factor algorithm** that's better than current system
✅ **Foothold consideration** without over-complexity
✅ **Wall angle awareness** via manual input
✅ **Deployable in 4-6 weeks** with realistic scope
✅ **Foundation for iteration** and calibration

### Phase 1a Defers

⏸️ Slanted hold detection (no method available)
⏸️ Wall-angle-dependent weighting (add after validation)
⏸️ Complexity multipliers (add after base model works)
⏸️ Advanced foothold analysis (add during calibration)

### Key Success Factors

1. **Start simple** - Basic implementation first
2. **Deploy fast** - Get to user feedback quickly
3. **Iterate weekly** - Adjust config based on data
4. **Measure everything** - Log all predictions for analysis
5. **User feedback** - Drive calibration with real experiences

---

## Next Steps

1. **Review & approve** this simplified specification
2. **Begin implementation** using this as guide (not full spec)
3. **Deploy Phase 1a MVP** within 4-6 weeks
4. **Collect feedback** for 2-3 weeks
5. **Calibrate & iterate** weekly
6. **Decide on Phase 1b** based on MVP results

**Do NOT** implement full Phase 1 spec - use this MVP spec instead.
