# Complexity Multipliers

## Overview

Complexity multipliers are **advanced features** that amplify base difficulty scores to account for route characteristics that increase mental and physical load beyond simple hold/distance analysis.

**Status**: **Implement in Phase 1c (Refinement)**, AFTER basic 4-factor model is calibrated and validated.

**Rationale**: Complexity multipliers add significant implementation complexity. They should NOT block initial development. Implement only after core algorithm is working and you have real data to validate their impact.

## Two Complexity Factors

### 1. Wall Angle Transitions

**Concept**: Routes that change wall angles require climbers to adapt technique mid-route, increasing mental load and transition difficulty.

**Example**: Starting on vertical wall, transitioning to overhang requires switching from balanced climbing to power-dominant moves.

### 2. Hold Type Variability

**Concept**: Routes with diverse hold types (crimps, slopers, pinches) require more technical adaptability and mental load than routes with uniform hold types.

**Example**: A route with all jugs is mentally simpler than a route alternating between crimps, slopers, and pinches.

## Wall Angle Transitions Multiplier

### Detection

```python
def detect_wall_transitions(wall_segments: list) -> dict:
    """
    Detect wall angle transitions in route.

    Args:
        wall_segments: List of (angle_category, proportion) tuples
            Example: [('vertical', 0.6), ('overhang', 0.4)]

    Returns:
        Dictionary with transition analysis
    """
    if len(wall_segments) <= 1:
        return {
            'has_transitions': False,
            'transition_count': 0,
            'transition_severity': 0
        }

    # Count transitions
    transition_count = len(wall_segments) - 1

    # Calculate severity (difference in angle categories)
    severity_map = {
        'slab': 1,
        'vertical': 2,
        'slight_overhang': 3,
        'moderate_overhang': 4,
        'steep_overhang': 5
    }

    max_severity = 0
    for i in range(len(wall_segments) - 1):
        angle1, _ = wall_segments[i]
        angle2, _ = wall_segments[i + 1]

        severity_diff = abs(severity_map[angle1] - severity_map[angle2])
        max_severity = max(max_severity, severity_diff)

    return {
        'has_transitions': True,
        'transition_count': transition_count,
        'transition_severity': max_severity
    }
```

### Multiplier Calculation

```python
def calculate_transition_multiplier(transition_data: dict) -> float:
    """
    Calculate difficulty multiplier based on wall transitions.

    Returns multiplier in range 1.0 - 1.5
    """
    if not transition_data['has_transitions']:
        return 1.0  # No transitions

    count = transition_data['transition_count']
    severity = transition_data['transition_severity']

    # Base multiplier from count
    count_multiplier = 1.0 + (min(count, 3) * 0.05)  # Up to +15% for 3+ transitions

    # Severity adjustment
    severity_multiplier = 1.0 + (severity * 0.05)  # Up to +25% for max severity

    # Combine (capped at 1.5)
    total_multiplier = min(1.5, count_multiplier * severity_multiplier)

    return total_multiplier
```

**Multiplier Range**: 1.0 - 1.5 (0% to +50% difficulty increase)

**Examples**:

- Single wall angle: 1.0 (no penalty)
- Vertical → Slight overhang (1 transition, severity 1): 1.05 × 1.05 = 1.10
- Slab → Moderate overhang (1 transition, severity 3): 1.05 × 1.15 = 1.21
- Multiple transitions (3+, severity 2): 1.15 × 1.10 = 1.27

**Calibration Note**: Multiplier ranges are initial estimates. Adjust based on whether routes with transitions are systematically under/over-predicted.

## Hold Type Variability Multiplier

### Entropy Calculation

Use **Shannon entropy** to measure hold type diversity:

```python
def calculate_hold_type_entropy(hold_types: dict) -> float:
    """
    Calculate Shannon entropy of hold type distribution.

    Args:
        hold_types: Dictionary of hold type counts
            Example: {'crimp': 5, 'jug': 3, 'sloper': 2}

    Returns:
        Entropy value (0 to ~2.8 for 7 hold types)
    """
    total_holds = sum(hold_types.values())

    if total_holds == 0:
        return 0

    entropy = 0
    for count in hold_types.values():
        if count > 0:
            proportion = count / total_holds
            entropy -= proportion * log2(proportion)

    return entropy
```

**Entropy Interpretation**:

- **Entropy = 0**: All holds same type (uniform, simple)
- **Entropy ~ 1.5**: Moderate diversity (2-3 dominant types)
- **Entropy > 2.5**: High diversity (many different types)

### Multiplier Calculation

```python
def calculate_variability_multiplier(entropy: float) -> float:
    """
    Calculate difficulty multiplier based on hold type variability.

    Returns multiplier in range 1.0 - 1.3
    """
    # Entropy typically ranges 0 - 2.8
    # Map to 1.0 - 1.3 multiplier

    # Low entropy (< 1.0): No penalty (uniform holds)
    if entropy < 1.0:
        return 1.0

    # High entropy (> 2.5): Max penalty (+30%)
    elif entropy > 2.5:
        return 1.3

    # Linear scaling between 1.0 and 2.5
    else:
        return 1.0 + ((entropy - 1.0) / 1.5) * 0.3
```

**Multiplier Range**: 1.0 - 1.3 (0% to +30% difficulty increase)

**Examples**:

- All jugs (entropy ≈ 0): 1.0 (no penalty)
- Mostly crimps with some jugs (entropy ≈ 0.8): 1.0 (below threshold)
- Mixed 3 types (entropy ≈ 1.5): 1.10
- Very diverse 5+ types (entropy ≈ 2.6): 1.30

**Calibration Note**: Adjust threshold (1.0) and max multiplier (1.3) if mixed-type routes are systematically mis-predicted.

## Combining Multipliers

### Application Order

```python
def apply_complexity_multipliers(
    base_score: float,
    wall_transitions: dict,
    hold_type_entropy: float
) -> float:
    """
    Apply both complexity multipliers to base score.

    Multipliers are multiplicative (amplify each other).
    """
    transition_mult = calculate_transition_multiplier(wall_transitions)
    variability_mult = calculate_variability_multiplier(hold_type_entropy)

    final_score = base_score * transition_mult * variability_mult

    return final_score
```

### Example: Complex Route

**Setup**:

- Base score (Factors 1-4): 7.5
- Wall transitions: 2 transitions, severity 2 → multiplier 1.20
- Hold diversity: entropy 2.2 → multiplier 1.24

**Calculation**:

- Final score: 7.5 × 1.20 × 1.24 = **11.16**

**Impact**: Complexity multipliers increased predicted grade by ~50% (V6 base → V9+ final).

## Implementation Approach

### Phase 1a-1b: DO NOT IMPLEMENT

Focus on getting basic 4-factor model working and calibrated.

**Reasoning**:

- Complexity multipliers add implementation time
- Require wall segment and hold type tracking
- Difficult to debug if accuracy is poor
- May mask issues in base factors

### Phase 1c: Implement if Accuracy Targets Met

**Prerequisites for implementing complexity multipliers**:

1. ✅ Basic 4-factor model deployed
2. ✅ Accuracy ≥60% exact match, ≥80% within ±1 grade
3. ✅ Base factors calibrated with user feedback
4. ✅ Systematic patterns identified (complex routes under-predicted)

**Implementation Steps**:

1. Add wall segment tracking to data model
2. Implement transition detection
3. Implement entropy calculation
4. Apply multipliers to predictions
5. A/B test: with vs without multipliers
6. Validate accuracy improvement

**Success Criteria**:

- Accuracy improves by ≥5% on routes with transitions
- Accuracy improves by ≥5% on routes with diverse holds
- No regression on simple routes (single wall angle, uniform holds)

## Edge Cases

**Single wall angle, uniform holds:**

- Both multipliers = 1.0
- No difficulty increase (correct)

**Complex route with easy holds:**

- Base score might be low (Factor 1)
- Multipliers increase it moderately
- Captures mental/technical load even with good holds

**Simple route with hard holds:**
- Base score high (Factor 1)
- Multipliers = 1.0 (no amplification)
- Difficulty driven by hold quality alone

**Extreme complexity:**

- Multiple transitions AND high entropy
- Multipliers can combine to 1.5 × 1.3 = 1.95 (nearly double difficulty)
- Rare but possible (very complex problem)

## Calibration Strategy

### Validation Questions

After implementing complexity multipliers, validate:

1. **Do routes with wall transitions feel harder than single-angle routes with same holds?**
   - Compare user feedback on transition vs non-transition routes
   - Adjust transition multiplier range if needed

2. **Do routes with diverse holds feel more mentally demanding?**
   - Compare user feedback on uniform vs mixed-type routes
   - Adjust entropy threshold and multiplier range

3. **Are complexity multipliers improving accuracy overall?**
   - A/B test with/without multipliers
   - Measure accuracy improvement by route type

### Iteration

If complexity multipliers **reduce** accuracy:

- Multiplier ranges too aggressive (reduce max values)
- Thresholds too sensitive (raise entropy threshold, etc.)
- Complexity not actually correlating with difficulty (remove feature)

If complexity multipliers have **no effect** on accuracy:

- Multiplier ranges too conservative (increase max values)
- Sample size too small (collect more data)
- Base factors already capturing complexity (feature redundant)

## Summary

Complexity multipliers enhance the 4-factor model by accounting for:

1. ✅ **Wall angle transitions** - Technique adaptation required
2. ✅ **Hold type variability** - Mental load from diverse techniques

**CRITICAL**: Implement complexity multipliers **only in Phase 1c**, AFTER basic model is validated.

**Multiplier Ranges**:

- Transition multiplier: 1.0 - 1.5 (up to +50%)
- Variability multiplier: 1.0 - 1.3 (up to +30%)
- Combined: Can amplify base score by up to ~2x for extremely complex routes

**Goal**: Improve prediction accuracy on complex routes without regressing on simple routes.

**Next**: See [Implementation Notes](implementation_notes.md) for technical guidance and calibration strategy.
