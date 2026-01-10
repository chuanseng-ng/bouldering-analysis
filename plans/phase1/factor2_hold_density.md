# Factor 2: Hold Density Analysis

## Objective

Assess route difficulty based on the number of available holds, considering both handhold and foothold availability separately.

**Weight in Overall Score**: ~25% (starting point - requires calibration)

## Core Principle

**Inverse relationship with difficulty (non-linear):**

- **Fewer holds** → Fewer movement options → Higher difficulty
- **More holds** → Multiple sequences available → Lower difficulty

**CRITICAL**: Foothold scarcity has different implications than handhold scarcity:

- **Handholds**: Fewer can indicate powerful, athletic problems
- **Footholds**: Fewer almost always increases difficulty (balance limitations)

## Handhold Density Score

### Formula

Use logarithmic decay to model non-linear relationship:

```text
Handhold_Density_Score = 12 - (log₂(handhold_count) × 2.5)
```

Clamp result between 0 and 12.

### Mapping

| Hold Count | Calculation | Score | Interpretation |
| :--------: | :---------: | :---: | :------------: |
| 3 | 12 - (1.58 × 2.5) | ~8.2 | Extremely difficult (V8-V12) |
| 5 | 12 - (2.32 × 2.5) | ~6.5 | Hard (V6-V8) |
| 8 | 12 - (3.0 × 2.5) | ~4.5 | Moderate-hard (V4-V6) |
| 12 | 12 - (3.58 × 2.5) | ~3.0 | Moderate (V3-V5) |
| 16 | 12 - (4.0 × 2.5) | ~2.0 | Moderate-easy (V1-V3) |
| 20+ | 12 - (4.32+ × 2.5) | ~1.2 | Easy (V0-V2) |

**Rationale**:

- Doubling holds doesn't halve difficulty
- Each additional hold provides diminishing marginal benefit
- Very few holds create exponentially harder problems

**Calibration Note**: This formula is a starting hypothesis. Monitor predictions and adjust the multiplier (2.5) or offset (12) based on user feedback.

## Foothold Density Score

### Formula

Model foothold scarcity with emphasis on severe difficulty from missing footholds:

```python
def calculate_foothold_density_score(foothold_count: int) -> float:
    """
    Calculate foothold density score.

    Steeper penalty for few footholds than handholds.
    """
    if foothold_count == 0:
        return 12.0  # Campusing - extreme difficulty

    elif foothold_count <= 2:
        return 10.0  # Very few footholds - severe constraint

    elif foothold_count <= 4:
        return 7.0   # Limited footholds - significant constraint

    elif foothold_count <= 6:
        return 4.5   # Some footholds - moderate constraint

    elif foothold_count <= 10:
        return 2.5   # Adequate footholds

    else:  # 11+
        return 1.0   # Many footholds - abundant options
```

| Foothold Count | Score | Interpretation |
| :------------: | :---: | :------------: |
| 0 | 12.0 | Campusing required (extreme) |
| 1-2 | 10.0 | Very limited balance options |
| 3-4 | 7.0 | Significant balance constraints |
| 5-6 | 4.5 | Moderate constraints |
| 7-10 | 2.5 | Adequate options |
| 11+ | 1.0 | Abundant options |

**Rationale**:

- Footholds enable balance, rest, and efficient movement
- Missing footholds forces campusing (elite-only technique)
- Few footholds severely limits movement sequences
- More footholds matter less on overhangs (see Factor 1 weighting)

**Calibration Note**: Thresholds (2, 4, 6, 10) are initial estimates. Adjust based on gym-specific route setting patterns.

## Combining Handhold and Foothold Density

### Wall-Angle-Dependent Weighting

Foothold density importance varies by wall angle:

```python
def calculate_factor2_score(
    handhold_count: int,
    foothold_count: int,
    wall_angle_category: str
) -> float:
    """
    Calculate Factor 2: Hold Density Score.

    Combines handhold and foothold density with wall-angle weighting.
    """
    # Calculate individual density scores
    handhold_density = 12 - (log2(max(handhold_count, 1)) * 2.5)
    handhold_density = max(0, min(12, handhold_density))

    foothold_density = calculate_foothold_density_score(foothold_count)

    # Wall-angle-dependent weights
    # (Reuse weights from Factor 1 for consistency)
    weights = {
        'slab': (0.40, 0.60),              # Foothold density more important
        'vertical': (0.55, 0.45),          # Balanced
        'slight_overhang': (0.60, 0.40),
        'moderate_overhang': (0.70, 0.30),
        'steep_overhang': (0.75, 0.25)     # Handhold density more important
    }
    hand_weight, foot_weight = weights.get(wall_angle_category, (0.55, 0.45))

    # Combine scores
    combined_score = (handhold_density * hand_weight) + (foothold_density * foot_weight)

    return combined_score
```

**Weighting Rationale**:

- **Slabs**: Foothold availability critical (60% weight)
- **Vertical**: Balanced importance (45% foothold weight)
- **Overhangs**: Handhold density matters more (25% foothold weight)

**Note**: These weights are similar to Factor 1 but can be independently calibrated if needed.

## Context and Exceptions

### Hold Density Must Be Interpreted with Hold Types

**Hold count alone is insufficient:**

- **5 jugs** (few holds) might be easier than **15 tiny crimps** (many holds)
- **3 large footholds** easier than **8 micro footholds**
- This interaction is captured by combining Factor 2 with Factor 1

**Example**:

- Route A: 10 handholds (all jugs), density score = 4.75
- Route B: 10 handholds (all crimps), density score = 4.75
- But Route B is much harder due to hold types (captured in Factor 1)

### Route Setting Patterns

**Boulder problems vs Long routes:**

- **Short boulders** (3-5 hard moves): Typically 5-10 handholds
- **Long boulders** (10+ moves): Typically 12-20 handholds
- Adjust expectations based on route length if available

**Gym-specific calibration:**

- Some gyms set sparse routes (difficulty from limited holds)
- Other gyms set dense routes (difficulty from hold types)
- Monitor gym-specific patterns in feedback

## Example Calculations

### Example 1: Sparse Powerful Route

**Setup:**

- 5 handholds (all jugs for power moves)
- 3 footholds (medium size)
- Wall angle: Steep overhang

**Calculation:**

- Handhold density: 12 - (log₂(5) × 2.5) = 12 - 5.8 = **6.2**
- Foothold density: **7.0** (3-4 footholds tier)
- Weights (steep overhang): 75% hands, 25% feet
- Combined: (6.2 × 0.75) + (7.0 × 0.25) = 4.65 + 1.75 = **6.4**

**Interpretation**: Moderate-high density difficulty, powered by few handholds on overhang.

### Example 2: Technical Slab with Sparse Footholds

**Setup:**

- 12 handholds (mixed types)
- 4 footholds (small size)
- Wall angle: Slab

**Calculation:**

- Handhold density: 12 - (log₂(12) × 2.5) = 12 - 8.96 = **3.04**
- Foothold density: **7.0** (3-4 footholds tier)
- Weights (slab): 40% hands, 60% feet
- Combined: (3.04 × 0.40) + (7.0 × 0.60) = 1.22 + 4.20 = **5.42**

**Interpretation**: Sparse footholds dominate difficulty on slab despite adequate handholds.

### Example 3: Campusing Route (No Footholds)

**Setup:**

- 8 handholds (crimps and pockets)
- 0 footholds (campusing)
- Wall angle: Vertical

**Calculation:**

- Handhold density: 12 - (log₂(8) × 2.5) = 12 - 7.5 = **4.5**
- Foothold density: **12.0** (campusing)
- Weights (vertical): 55% hands, 45% feet
- Combined: (4.5 × 0.55) + (12.0 × 0.45) = 2.48 + 5.40 = **7.88**

**Interpretation**: Campusing adds massive difficulty, even with moderate handhold density.

## Implementation Notes

### Minimum Viable Implementation

**Phase 1a** - IMPLEMENTED ✅ (see `src/grade_prediction_mvp.py`):

1. [x] Count handholds and footholds separately
2. [x] Apply density formulas (logarithmic for handholds, tiered for footholds)
3. [x] Combine with constant 60/40 weights (wall-angle-dependent deferred to Phase 1b)
4. [x] Log predictions with score breakdown

**Phase 1b (Calibration)** - PENDING:

1. [ ] Analyze prediction accuracy by hold count
2. [ ] Adjust logarithmic multiplier (2.5) if needed
3. [ ] Refine foothold density thresholds (2, 5, 8)
4. [ ] Implement wall-angle-dependent weights (40/60 for slabs, 75/25 for overhangs)
5. [ ] Calibrate wall-angle weights independently if helpful

### Data Collection for Calibration

**Log for each route:**

- Handhold count and types
- Foothold count and sizes
- Wall angle
- Predicted grade
- User feedback (actual difficulty perceived)

**Analyze patterns:**

- Are sparse routes (5-8 holds) consistently over/under-predicted?
- Is campusing (0 footholds) adequately penalized?
- Do foothold density weights align with slab vs overhang feedback?

### Edge Cases

**Single hold routes (contrived):**

- handhold_count = 1: Score = 12.0 (max difficulty)
- Likely unrealistic in practice

**Very dense routes (20+ holds):**

- Score approaches 1.0 (easy)
- May indicate beginner route or long endurance problem

**Misdetection (holds missed by CV):**

- Density scores will be artificially high
- Monitor detection completeness alongside calibration
- Consider confidence scores in hold detection

## Relationship to Other Factors

### Factor 1 (Hold Types/Sizes)

- Factor 1: "What kind of holds are they?"
- Factor 2: "How many holds are there?"
- **Interaction**: Few hard holds is much harder than few easy holds
- Combined in weighted scoring model

### Factor 3 (Hold Distances)

- Factor 2: Total hold availability
- Factor 3: How far apart are they?
- **Interaction**: Sparse holds spaced far apart is especially hard
- Distance factor accounts for spatial distribution

### Factor 4 (Wall Incline)

- Wall angle determines foothold density importance
- Weights adjusted per wall angle (Factor 2 formula)
- Steep overhangs: handhold density matters more
- Slabs: foothold density critical

## Summary

Factor 2 evaluates hold availability through:

1. [x] **Handhold density** - Logarithmic penalty for sparse holds - IMPLEMENTED
2. [x] **Foothold density** - Steeper penalty for missing footholds - IMPLEMENTED
3. [ ] **Wall-angle weighting** - Importance varies by terrain - DEFERRED to Phase 1b (using constant 60/40 in MVP)
4. [x] **Campusing detection** - Maximum penalty for zero footholds - IMPLEMENTED

**Result**: Hold density score (range ~1-12) reflecting movement options and balance constraints.

**Next**: Combine with [Factor 1 (Hold Analysis)](factor1_hold_analysis.md), [Factor 3 (Distances)](factor3_hold_distances.md), and [Factor 4 (Wall Incline)](factor4_wall_incline.md).
