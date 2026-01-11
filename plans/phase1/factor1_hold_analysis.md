# Factor 1: Hold Type, Size, and Orientation Analysis

## Objective

Evaluate the technical difficulty of holds based on:

1. **Hold type** (crimp, jug, sloper, pinch, pocket)
2. **Physical size** (bounding box area)
3. **Orientation/slant angle** (upward, horizontal, downward)
4. **Foothold availability and quality**

**Weight in Overall Score**: ~35% (starting point - requires calibration)

## Critical Design Principles

### 1. Footholds Are as Important as Handholds

Footholds are a **primary difficulty determinant**, not an afterthought:

- Enable balance, rest positions, weight transfer
- Missing or tiny footholds drastically increase difficulty
- Foothold importance varies by wall angle

### 2. Hold Orientation Matters Significantly

**NEW CONSIDERATION**: Slanted holds significantly affect difficulty:

**Downward-slanting holds** (negative angle):

- Much harder to grip/stand on
- Require more grip strength or precise foot placement
- Force open-hand positions or heel hooks

**Upward-slanting holds** (positive angle):

- Easier to grip/stand on
- Provide positive surface to push against
- Allow more secure positions

**Side-slanting holds**:

- Affect lateral stability
- Harder for precise footwork
- Can force specific body positions

### 3. Wall Angle Determines Foothold Importance

Foothold impact on difficulty varies dramatically by wall angle:

- **Slabs**: Footholds contribute 60-70% of difficulty
- **Vertical**: Footholds contribute 40-50% of difficulty
- **Overhangs**: Footholds contribute 25-35% of difficulty

## Handhold Difficulty Classification

### Base Difficulty Tiers

**Tier 1 - Very Hard (Base Score: 10)**

- **Crimps**: Small, narrow holds requiring finger strength
- **Pockets**: Small holes requiring specific finger positioning

**Tier 2 - Hard (Base Score: 7)**

- **Slopers**: Round holds requiring open-handed grip and body tension
- **Pinches**: Require thumb opposition and pinch strength

**Tier 3 - Moderate (Base Score: 4)**

- **Start-holds**: Typically good holds to begin route
- **Top-out-holds**: Final holds, usually accessible

**Tier 4 - Easy (Base Score: 1)**

- **Jugs**: Large, easy-to-grip holds

**IMPORTANT**: These base scores are **starting points** requiring calibration with real route data.

### Size-Based Adjustments

Calculate hold size from bounding box:

```text
hold_width = bbox_x2 - bbox_x1
hold_height = bbox_y2 - bbox_y1
hold_area = hold_width × hold_height
```

**Size Modifiers (Handholds):**

**Crimps & Pockets:**

- Extra small (area < 500px²): +3 difficulty
- Small (area 500-1000px²): +2 difficulty
- Medium (area 1000-2000px²): +1 difficulty
- Large (area > 2000px²): 0 (easier, not a true crimp)

**Slopers:**

- Small (area < 1500px²): +2 difficulty
- Medium (area 1500-3000px²): +1 difficulty
- Large (area > 3000px²): 0

**Jugs:**

- Small (area < 2000px²): +1 difficulty (not truly a jug)
- Large (area > 2000px²): 0 (remains easy)

**Calibration Note**: Pixel thresholds will vary by image resolution, camera angle, and distance. Monitor and adjust based on feedback.

### Orientation/Slant Adjustments (NEW)

**Detecting Slant Angle:**

If hold orientation can be detected (from hold contour shape, orientation vector, or manual annotation):

```python
def calculate_slant_adjustment(slant_angle: float, hold_type: str) -> float:
    """
    Calculate difficulty adjustment based on hold slant angle.

    Args:
        slant_angle: Angle in degrees
            - 0° = horizontal
            - Positive = upward-slanting
            - Negative = downward-slanting
        hold_type: Type of hold (affects slant impact)

    Returns:
        Adjustment multiplier (0.7 - 1.4 range)
    """
    # Downward-slanting (negative angles)
    if slant_angle < -15:  # Strong downward slant
        if hold_type in ['crimp', 'pocket']:
            return 1.4  # 40% harder
        elif hold_type == 'sloper':
            return 1.5  # 50% harder (very difficult)
        else:
            return 1.3  # 30% harder

    elif slant_angle < -5:  # Slight downward slant
        if hold_type in ['crimp', 'pocket']:
            return 1.2  # 20% harder
        elif hold_type == 'sloper':
            return 1.3  # 30% harder
        else:
            return 1.15  # 15% harder

    # Horizontal or neutral (-5° to +5°)
    elif -5 <= slant_angle <= 5:
        return 1.0  # Neutral

    # Upward-slanting (positive angles)
    elif slant_angle <= 15:  # Slight upward slant
        if hold_type == 'jug':
            return 0.95  # 5% easier (already easy)
        else:
            return 0.85  # 15% easier

    else:  # Strong upward slant (>15°)
        if hold_type == 'jug':
            return 0.9  # 10% easier
        else:
            return 0.75  # 25% easier
```

**Implementation Approaches:**

1. **Manual Annotation (MVP)**: Users can optionally mark holds as "slanted" during upload
2. **Computer Vision (Future)**: Detect hold orientation from contour analysis
3. **Default Neutral (Safe)**: If orientation unknown, assume horizontal (1.0 multiplier)

**Rationale for Ranges:**

- Downward-slanting crimps/pockets are significantly harder (up to 50% increase)
- Upward-slanting holds provide substantial advantage (up to 25% reduction)
- Jugs are less affected by slant (already easy)
- Slopers are most affected (friction-dependent)

### Handhold Difficulty Score Formula

```text
For each handhold:
    base_score = TIER_SCORE[hold_type]
    size_adjustment = SIZE_MODIFIER[hold_area]
    slant_adjustment = SLANT_MULTIPLIER[slant_angle, hold_type]

    hold_score = (base_score + size_adjustment) × slant_adjustment

Total handhold score = Σ(hold_scores)
Normalized handhold score = Total / handhold_count
```

### Handhold Type Distribution

Calculate proportion of hard holds:

```text
hard_hold_ratio = (count_crimps + count_pockets + count_slopers) / total_handholds
```

Apply non-linear difficulty multiplier:

```text
Final Handhold Score = Normalized_Score × (1 + hard_hold_ratio × 0.5)
```

**Rationale**: Routes with many hard holds are disproportionately harder (mental load, fatigue accumulation).

## Foothold Difficulty Analysis

### Foothold Importance by Wall Angle

Define wall-angle-dependent weights:

```python
def get_hold_weights_by_wall_angle(wall_angle_category: str) -> tuple:
    """
    Return (handhold_weight, foothold_weight) for combining scores.

    Weights sum to 1.0, reflecting biomechanical importance.
    """
    weights = {
        'slab': (0.35, 0.65),              # 65% foothold importance
        'vertical': (0.55, 0.45),          # 45% foothold importance
        'slight_overhang': (0.60, 0.40),   # 40% foothold importance
        'moderate_overhang': (0.70, 0.30), # 30% foothold importance
        'steep_overhang': (0.75, 0.25)     # 25% foothold importance
    }
    return weights.get(wall_angle_category, (0.55, 0.45))
```

| Wall Angle | Hand % | Foot % | Rationale |
| :--------: | :----: | :----: | :-------: |
| Slab | 35% | **65%** | Footwork-dominant, balance-critical |
| Vertical | 55% | **45%** | Balanced load between hands and feet |
| Slight Overhang | 60% | **40%** | Upper body load increases |
| Moderate Overhang | 70% | **30%** | Power-focused, feet assist positioning |
| Steep Overhang | 75% | **25%** | Upper body dominant, feet for positioning |

**Calibration Note**: These percentages are biomechanically-informed starting points. Adjust based on user feedback.

### Foothold Difficulty Tiers

**Tier 1 - No Footholds (Campusing)**

- **Score: 12** (EXTREME difficulty)
- Forces climbing without feet
- Adds +2 to +4 V-grades for most climbers

**Tier 2 - Very Small Footholds**

- **Size**: area < 800px²
- **Score: 9**
- Requires precise toe placement and balance

**Tier 3 - Small Footholds**

- **Size**: area 800-1500px²
- **Score: 6**
- Moderate precision required

**Tier 4 - Medium Footholds**

- **Size**: area 1500-3000px²
- **Score: 3**
- Standard footwork

**Tier 5 - Large Footholds**

- **Size**: area > 3000px²
- **Score: 1**
- Easy to stand on

### Foothold Scarcity Multiplier

```python
def calculate_foothold_scarcity_multiplier(foothold_count: int) -> float:
    """
    Apply difficulty multiplier based on foothold scarcity.

    Fewer footholds = limited balance options = harder
    """
    if foothold_count == 0:
        return None  # Special case: campusing (score = 12.0)
    elif foothold_count <= 2:
        return 1.5   # Very few footholds
    elif foothold_count <= 4:
        return 1.25  # Limited footholds
    elif foothold_count <= 6:
        return 1.1   # Some constraint
    else:
        return 1.0   # Adequate options
```

**Rationale**: Few footholds limit:

- Balance options during moves
- Rest position availability
- Movement sequence choices
- Dynamic move generation (need feet to push)

### Foothold Slant Adjustments (NEW)

Footholds are **even more affected** by slant angle than handholds:

```python
def calculate_foothold_slant_adjustment(slant_angle: float, foothold_size: float) -> float:
    """
    Calculate difficulty adjustment for slanted footholds.

    Footholds are more sensitive to slant than handholds.
    """
    # Downward-slanting footholds are VERY difficult
    if slant_angle < -15:  # Strong downward slant
        if foothold_size < 1000:  # Small foothold
            return 1.6  # 60% harder (can easily slip)
        else:
            return 1.4  # 40% harder

    elif slant_angle < -5:  # Slight downward slant
        if foothold_size < 1000:
            return 1.4  # 40% harder
        else:
            return 1.2  # 20% harder

    # Horizontal (-5° to +5°)
    elif -5 <= slant_angle <= 5:
        return 1.0  # Neutral

    # Upward-slanting (positive)
    elif slant_angle <= 15:  # Slight upward
        return 0.8  # 20% easier (positive platform)

    else:  # Strong upward (>15°)
        return 0.7  # 30% easier (very positive)
```

**Key Differences from Handholds:**

- Downward-slanting footholds are MORE penalizing (up to 60% harder)
- Small downward-slanting footholds are extremely difficult
- Upward-slanting footholds provide greater advantage (up to 30% easier)
- Slant angle affects balance and slip risk more than grip

### Foothold Difficulty Score Formula

```python
def calculate_foothold_difficulty(footholds: list, wall_angle: str) -> float:
    """
    Calculate foothold difficulty score.
    """
    if len(footholds) == 0:
        return 12.0  # Campusing - maximum difficulty

    total_score = 0
    for fh in footholds:
        # Base score from size
        if fh.area < 800:
            base_score = 9
        elif fh.area < 1500:
            base_score = 6
        elif fh.area < 3000:
            base_score = 3
        else:
            base_score = 1

        # Apply slant adjustment if available
        if fh.slant_angle is not None:
            slant_mult = calculate_foothold_slant_adjustment(fh.slant_angle, fh.area)
            base_score *= slant_mult

        total_score += base_score

    # Normalize by count
    avg_score = total_score / len(footholds)

    # Apply scarcity multiplier
    scarcity_mult = calculate_foothold_scarcity_multiplier(len(footholds))

    return avg_score * scarcity_mult
```

## Combined Hold Difficulty Score

Integrate handholds and footholds with wall-angle-dependent weighting:

```python
def calculate_factor1_score(
    handholds: list,
    footholds: list,
    wall_angle_category: str
) -> float:
    """
    Calculate Factor 1: Hold Difficulty Score.

    Combines handhold and foothold difficulty with wall-angle weighting.
    """
    # Calculate individual scores
    handhold_score = calculate_handhold_difficulty(handholds)  # ~1-13 range
    foothold_score = calculate_foothold_difficulty(footholds, wall_angle_category)  # ~1-12 range

    # Get wall-angle-dependent weights
    hand_weight, foot_weight = get_hold_weights_by_wall_angle(wall_angle_category)

    # Combine scores
    combined_score = (handhold_score * hand_weight) + (foothold_score * foot_weight)

    # Result range: approximately 1-13
    return combined_score
```

## Example Calculations

### Example 1: V5 Slab with Small Footholds and Slanted Holds

**Setup:**

- 8 handholds: 5 crimps (1200px², horizontal), 3 jugs (2500px², upward +10°)
- 5 footholds: all small (1000px²), 3 horizontal, 2 downward-slanted (-10°)
- Wall angle: Slab

**Handhold Calculation:**

- Crimps: (10 + 1) × 1.0 = 11 each → 5 × 11 = 55
- Jugs (upward-slanted): (1 + 0) × 0.85 = 0.85 each → 3 × 0.85 = 2.55
- Total: 57.55 / 8 = 7.19
- Hard hold ratio: 5/8 = 0.625
- Final: 7.19 × (1 + 0.625 × 0.5) = 7.19 × 1.3125 = **9.44**

**Foothold Calculation:**

- 3 horizontal small: 6 × 1.0 = 6 each → 18
- 2 downward-slanted small: 6 × 1.4 = 8.4 each → 16.8
- Total: 34.8 / 5 = 6.96
- Scarcity (5 footholds): 1.1x
- Final: 6.96 × 1.1 = **7.66**

**Combined (Slab: 35% hands, 65% feet):**

- (9.44 × 0.35) + (7.66 × 0.65) = 3.30 + 4.98 = **8.28**
- **Impact**: Slanted footholds dominate difficulty on slab

### Example 2: V7 Overhang with Downward-Slanted Crimps

**Setup:**

- 7 handholds: all crimps (600px²), 4 downward-slanted (-12°), 3 horizontal
- 8 footholds: large (3500px²), all horizontal
- Wall angle: Moderate overhang

**Handhold Calculation:**

- Downward crimps: (10 + 2) × 1.3 = 15.6 each → 4 × 15.6 = 62.4
- Horizontal crimps: (10 + 2) × 1.0 = 12 each → 3 × 12 = 36
- Total: 98.4 / 7 = 14.06
- Hard hold ratio: 7/7 = 1.0
- Final: 14.06 × (1 + 1.0 × 0.5) = 14.06 × 1.5 = **21.09** (very hard!)

**Foothold Calculation:**

- Large horizontal: 1 × 1.0 = 1 each → 8 × 1 = 8
- Average: 8 / 8 = 1.0
- Scarcity: 1.0x
- Final: **1.0**

**Combined (Moderate Overhang: 70% hands, 30% feet):**

- (21.09 × 0.70) + (1.0 × 0.30) = 14.76 + 0.30 = **15.06**
- **Impact**: Downward-slanted crimps make route extremely difficult

## Implementation Notes

### Minimum Viable Implementation

**Phase 1a (Basic)** - IMPLEMENTED ✅:

1. [x] Implement handhold and foothold size-based scoring (`src/grade_prediction_mvp.py`)
2. [x] Use neutral slant multiplier (1.0) for all holds
3. [x] Apply constant 60/40 weighting (wall-angle-dependent deferred to Phase 1b)
4. [ ] Collect feedback on predictions

**Phase 1b (Slant Integration)** - PENDING:

1. [ ] Add slant angle detection or manual annotation
2. [ ] Implement slant adjustment multipliers
3. [ ] Implement wall-angle-dependent foothold weighting
4. [ ] Calibrate slant impact based on user feedback
5. [ ] Monitor accuracy improvement

### Shared Wall-Angle Weight Configuration

**Phase 1b Starting Values** (shared with Factor 2):

| Wall Angle | Handhold Weight | Foothold Weight |
| :--------: | :-------------: | :-------------: |
| Slab | 0.40 | 0.60 |
| Vertical | 0.55 | 0.45 |
| Slight Overhang | 0.60 | 0.40 |
| Moderate Overhang | 0.70 | 0.30 |
| Steep Overhang | 0.75 | 0.25 |

**Key Points**:

- Factor 1 and Factor 2 share the same wall-angle weights from a single configuration source
- Weights are stored in `grade_prediction.wall_angle_weights` in `user_config.yaml`
- This follows the "validate-then-diverge" principle: shared weights simplify initial calibration
- Independent calibration (separate weights per factor) only considered if shared weights create conflicting optimization

**Configuration Access**:

```python
# Factor 1 reads from shared config
hand_weight, foot_weight = get_wall_angle_weights(wall_angle)
factor1_score = (handhold_score * hand_weight) + (foothold_score * foot_weight)
```

**Cross-Reference**: See [Factor 2 Hold Density - Wall-Angle Weight Calibration Strategy](factor2_hold_density.md#wall-angle-weight-calibration-strategy) for:

- Decision criteria for independent calibration
- Systematic bias thresholds
- Factor-specific bias detection workflow

See [Implementation Notes - Wall-Angle Weight Configuration](implementation_notes.md#wall-angle-weight-configuration) for:

- YAML configuration structure
- Fallback defaults
- Future-state structure for independent weights

### Calibration Strategy

**Initial Deployment:**

- Use specified tier scores and size thresholds as starting points
- Log all hold characteristics and predicted grades
- Collect user feedback ("too hard", "too easy", "accurate")
- Analyze systematic biases

**Iteration:**

- Adjust tier scores based on feedback
- Refine size thresholds for image resolution
- Calibrate slant multipliers
- Test on known routes

**Target Metrics:**

- Factor 1 predictions correlate with user ratings
- Size adjustments improve accuracy over base tiers alone
- Slant adjustments improve accuracy by 5-10%

### Edge Cases

**No footholds detected (campusing):**

- Return foothold_score = 12.0
- Combined score will be very high
- Likely V7+ route

**All holds same type:**

- hard_hold_ratio can be 0.0 or 1.0
- Formula handles both cases

**Very large holds:**

- Thresholds prevent negative adjustments
- Jugs remain easy regardless of size

**Missing slant data:**

- Default to 1.0 multiplier (neutral)
- System degrades gracefully

## Summary

Factor 1 evaluates hold difficulty through:

1. [x] **Hold type tiers** (crimps hardest, jugs easiest) - IMPLEMENTED
2. [x] **Size adjustments** (smaller = harder) - IMPLEMENTED
3. [ ] **Slant angle multipliers** (downward harder, upward easier) - DEFERRED to Phase 1b
4. [x] **Foothold quality** (size, scarcity) - IMPLEMENTED (slant deferred)
5. [ ] **Wall-angle weighting** (60% footholds on slabs, 25% on overhangs) - DEFERRED to Phase 1b (using constant 60/40 in MVP, shared config with Factor 2)

**Result**: Comprehensive hold difficulty score (range ~1-13) that properly accounts for hands, feet, and orientation.

**Configuration Note**:

- **Phase 1a (MVP)**: Uses hardcoded constant 60/40 handhold/foothold weights (wall-angle-independent)
- **Phase 1b**: Will read wall-angle-dependent weights from `grade_prediction.wall_angle_weights` config, shared with Factor 2

See [Implementation Notes](implementation_notes.md#wall-angle-weight-configuration) for full configuration details.

**Next**: Combine with [Factor 2 (Hold Density)](factor2_hold_density.md), [Factor 3 (Distances)](factor3_hold_distances.md), and [Factor 4 (Wall Incline)](factor4_wall_incline.md).
