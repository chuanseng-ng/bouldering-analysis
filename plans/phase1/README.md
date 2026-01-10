# Phase 1: Route-Based Grade Prediction

## Overview

Phase 1 implements a sophisticated, multi-factor algorithm to predict climbing route difficulty (V0-V12) from detected route characteristics.

**Status**: Phase 1a MVP implemented ✅ - Calibration and testing in progress

**Objective**: Replace the current simplified grade prediction with a climbing domain-aware algorithm that considers multiple difficulty factors.

## Algorithm Structure

### Four Primary Factors

The algorithm evaluates difficulty based on four independent factors:

1. **[Hold Analysis](factor1_hold_analysis.md)** (Weight: ~35%)
   - Hold types (crimps, jugs, slopers, pinches, pockets)
   - Hold sizes (physical dimensions)
   - **Hold orientation/slant angle** (critical for difficulty)
   - Foothold availability and quality

2. **[Hold Density](factor2_hold_density.md)** (Weight: ~25%)
   - Total hold count (handhold and foothold separately)
   - Spacing between holds
   - Hold availability for balance and rest

3. **[Hold Distances](factor3_hold_distances.md)** (Weight: ~20%)
   - Inter-hold spacing and reach requirements
   - Movement span analysis
   - Dynamic vs static move identification

4. **[Wall Incline](factor4_wall_incline.md)** (Weight: ~20%)
   - Wall angle impact on difficulty
   - Slab vs vertical vs overhang
   - Biomechanical considerations

### Complexity Multipliers (Refinement Phase)

Two additional multipliers amplify base difficulty for complex routes:

1. **[Wall Angle Transitions](complexity_multipliers.md#wall-transitions)**
   - Multiple wall angles in single route
   - Transition difficulty

2. **[Hold Type Variability](complexity_multipliers.md#hold-variability)**
   - Diverse hold types requiring different techniques
   - Mental load and adaptation requirements

**Implementation Note**: Add complexity multipliers **after** basic 4-factor model is calibrated and validated.

## Scoring Formula

### Basic Model (Initial Implementation)

```text
Base Score = (
    Hold_Difficulty_Score × 0.35 +
    Hold_Density_Score × 0.25 +
    Distance_Score × 0.20 +
    Wall_Incline_Score × 0.20
)

V-Grade = map(Base_Score, grade_thresholds)
```

### Full Model (Refinement Phase)

```text
Base Score = (computed as above)

Final Score = Base Score × Transition_Multiplier × Variability_Multiplier

V-Grade = map(Final_Score, grade_thresholds)
```

**IMPORTANT**: All weight values (0.35, 0.25, etc.) are **starting points** requiring empirical calibration with real route data and user feedback.

## Implementation Approach

### Stage 1: Basic 4-Factor Model

Implement the core algorithm:

1. Extract hold data (types, sizes, positions)
2. Calculate 4 factor scores
3. Combine with initial weights
4. Map to V-grade
5. Log predictions and collect user feedback

**Goal**: Working prediction system, even if accuracy is initially moderate.

### Stage 2: Calibration

Refine the algorithm:

1. Analyze user feedback on predictions
2. Adjust factor weights based on accuracy data
3. Calibrate size thresholds and difficulty tiers
4. Iterate until accuracy targets met

**Goal**: ≥60% exact match, ≥80% within ±1 grade.

### Stage 3: Complexity Multipliers

Add advanced features:

1. Implement wall transition detection
2. Calculate hold variability entropy
3. Apply multipliers to base score
4. Validate improvement in accuracy

**Goal**: Improved accuracy on complex routes.

## Critical Design Principles

### 1. Calibration-First Mindset

**All threshold values in specifications are starting points**, including:

- Hold size thresholds (e.g., "area < 500px²")
- Difficulty tier scores (e.g., "crimps = 10")
- Factor weights (e.g., "0.35, 0.25, 0.20, 0.20")
- Multiplier ranges (e.g., "1.0-1.5")

**Why**: These values depend on:

- Image resolution and camera angles
- Detection model calibration
- Gym-specific route setting styles
- User population skill distribution

**Action**: Implement extensive logging, collect user feedback, iterate.

### 2. Slanted Hold Consideration

Hold orientation significantly affects difficulty and **must** be considered:

**Handholds:**

- Downward-slanting (negative): +2-3 difficulty points
- Horizontal/slight positive: Neutral
- Upward-slanting (positive): -1-2 difficulty points

**Footholds:**

- Downward-slanting: +3-4 difficulty points (can easily slip)
- Horizontal: Neutral
- Upward-slanting: -2-3 difficulty points (positive platform)

See [`factor1_hold_analysis.md`](factor1_hold_analysis.md) for full integration approach.

### 3. Staged Complexity

**Start simple, add complexity incrementally:**

Phase 1a: Basic 4-factor model
Phase 1b: Calibration and refinement
Phase 1c: Complexity multipliers
Phase 1d: Advanced features (if needed)

**Rationale**: Prevents complexity from blocking initial progress, allows focused debugging.

### 4. Foothold Parity

Footholds are **as important as handholds** for difficulty assessment:

- Footholds enable balance, rest positions, weight transfer
- Missing or tiny footholds drastically increase difficulty
- Foothold importance varies by wall angle (critical on slabs, less so on overhangs)

See [`factor1_hold_analysis.md`](factor1_hold_analysis.md) for wall-angle-dependent foothold weighting.

## File Structure

```text
phase1/
├── README.md                          (this file - overview)
├── factor1_hold_analysis.md           (hold types, sizes, slant angles)
├── factor2_hold_density.md            (hold count and spacing)
├── factor3_hold_distances.md          (inter-hold distances, reaches)
├── factor4_wall_incline.md            (wall angle impact)
├── complexity_multipliers.md          (transitions and variability)
└── implementation_notes.md            (technical guidance, calibration)
```

## Success Criteria

### Minimum Viable (Phase 1a) - IMPLEMENTED ✅

- [x] All 4 factors implemented
- [x] Basic prediction working (even if accuracy moderate)
- [x] Prediction time < 100ms
- [ ] User feedback collection operational
- [x] Detailed logging for calibration

### Target Accuracy (Phase 1b) - PENDING

- [ ] Exact match: ≥60%
- [ ] Within ±1 grade: ≥80%
- [ ] No regressions from current system
- [x] Clear explanations for predictions

### Advanced Features (Phase 1c) - PENDING

- [ ] Complexity multipliers integrated
- [ ] Exact match: ≥70%
- [ ] Within ±1 grade: ≥85%
- [ ] User satisfaction >3.5/5.0

## Next Steps

1. **Read Factor Specifications**: Review each factor file to understand scoring approach
2. **Review Implementation Notes**: [`implementation_notes.md`](implementation_notes.md) for technical guidance
3. **Implement Basic Model**: Start with 4-factor algorithm (no complexity multipliers)
4. **Collect Feedback**: Essential for calibration
5. **Iterate**: Refine based on real-world data

## Related Documentation

- **System Overview**: [`../overview.md`](../overview.md)
- **Current Implementation**: [`../../src/main.py:707`](../../src/main.py:707)
- **Database Models**: [`../../src/models.py`](../../src/models.py)
- **Next Phase**: [`../phase1.5/README.md`](../phase1.5/README.md) - Persona personalization
