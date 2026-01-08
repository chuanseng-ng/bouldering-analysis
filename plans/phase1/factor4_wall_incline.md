# Factor 4: Wall Incline Analysis

## Objective

Evaluate how wall angle affects climbing difficulty based on biomechanical principles.

**Weight in Overall Score**: ~20% (starting point - requires calibration)

## Core Principle

**Wall angle fundamentally changes climbing biomechanics:**
- **Slabs** (< 90°): Footwork-dominant, balance-critical, technical
- **Vertical** (90°): Balanced between hands and feet, standard baseline
- **Overhangs** (> 90°): Upper body dominant, power and core strength

**Key Insight**: The same holds at different wall angles create vastly different difficulty. A V3 on a slab might be V6 on a steep overhang.

## Wall Angle Categories

### Classification

**Slab**
- **Angle**: 70° - 89° (leaning back from climber)
- **Biomechanics**: Weight over feet, friction climbing, balance-critical
- **Primary skills**: Footwork, balance, body positioning
- **Difficulty modifier**: **0.65 - 0.80** (easier than vertical)

**Vertical**
- **Angle**: 90° (perfectly upright)
- **Biomechanics**: Balanced load, standard climbing
- **Primary skills**: All-around technique
- **Difficulty modifier**: **1.0** (baseline)

**Slight Overhang**
- **Angle**: 91° - 105° (leaning toward climber)
- **Biomechanics**: Increased upper body load, core engagement
- **Primary skills**: Pulling strength, core tension
- **Difficulty modifier**: **1.15 - 1.30**

**Moderate Overhang**
- **Angle**: 106° - 120° (significant overhang)
- **Biomechanics**: Upper body dominant, sustained core tension
- **Primary skills**: Power, lock-offs, core strength
- **Difficulty modifier**: **1.40 - 1.60**

**Steep Overhang (Roof)**
- **Angle**: 121° - 135°+ (approaching horizontal)
- **Biomechanics**: Nearly horizontal, extreme upper body and core demands
- **Primary skills**: Campus strength, lock-offs, coordination
- **Difficulty modifier**: **1.70 - 2.00**

**Calibration Note**: Modifiers are approximate. Calibrate based on user feedback and route setter assessments.

## Wall Incline Score Formula

### Direct Mapping Approach

```python
def calculate_wall_incline_score(wall_angle_category: str) -> float:
    """
    Calculate difficulty score based on wall angle.

    Returns score in range 1-12 based on angle category.
    """
    scores = {
        'slab': 3.0,                # Easier (technical but lower physical load)
        'vertical': 6.0,            # Baseline
        'slight_overhang': 7.5,     # Moderate increase
        'moderate_overhang': 9.0,   # Significant increase
        'steep_overhang': 11.0      # Very hard (roof climbing)
    }

    return scores.get(wall_angle_category, 6.0)  # Default to vertical
```

### Multiplier Approach (Alternative)

Instead of absolute scores, apply as multiplier to combined score of other factors:

```python
def get_wall_incline_multiplier(wall_angle_category: str) -> float:
    """
    Return difficulty multiplier based on wall angle.

    Applied to base score from Factors 1-3.
    """
    multipliers = {
        'slab': 0.75,               # 25% easier
        'vertical': 1.0,            # Baseline
        'slight_overhang': 1.20,    # 20% harder
        'moderate_overhang': 1.50,  # 50% harder
        'steep_overhang': 1.85      # 85% harder
    }

    return multipliers.get(wall_angle_category, 1.0)
```

**Recommendation**: Use **direct mapping approach** for simplicity in Phase 1a. Can switch to multiplier approach if it improves calibration.

## Wall Angle Input Method

### Manual User Input (Phase 1 MVP)

**Approach**: User selects wall angle from dropdown during route upload.

**UI Options:**
- Slab (< 90°)
- Vertical (90°)
- Slight Overhang (90° - 105°)
- Moderate Overhang (105° - 120°)
- Steep Overhang / Roof (> 120°)

**Pros**:
- Simple, accurate
- No computer vision needed
- User knows wall angle when climbing

**Cons**:
- Requires user input (minor friction)
- May be forgotten or mis-selected

### Automatic Detection (Future Enhancement)

**Approach**: Use computer vision to detect wall angle from image.

**Challenges**:
- Requires reference lines or known geometry
- Camera angle affects perception
- Multiple wall angles in one image (transitions)

**Recommendation**: Defer automatic detection to future enhancement. Manual input is sufficient for MVP.

## Example Calculations

### Example 1: V4 Slab Route

**Setup:**
- Wall angle: Slab
- Factors 1-3 combined base score: 8.5

**Calculation (Direct Mapping):**
- Wall incline score: **3.0**
- Factor 4 weight: 20%
- Contribution: 3.0 × 0.20 = **0.60**

**Calculation (Multiplier Approach):**
- Base score (Factors 1-3, weight 80%): 8.5 × 0.80 = 6.8
- Wall multiplier: 0.75
- Final: 6.8 × 0.75 = **5.1**

**Interpretation**: Slab significantly reduces difficulty compared to vertical.

### Example 2: V8 Steep Overhang

**Setup:**
- Wall angle: Steep overhang
- Factors 1-3 combined base score: 7.5

**Calculation (Direct Mapping):**
- Wall incline score: **11.0**
- Factor 4 weight: 20%
- Contribution: 11.0 × 0.20 = **2.20**

**Calculation (Multiplier Approach):**
- Base score (Factors 1-3, weight 80%): 7.5 × 0.80 = 6.0
- Wall multiplier: 1.85
- Final: 6.0 × 1.85 = **11.1**

**Interpretation**: Steep overhang nearly doubles difficulty.

## Relationship to Other Factors

### Foothold Importance (Factor 1)

Wall angle determines foothold weighting:
- **Slabs**: Footholds contribute 65% to Factor 1 score
- **Steep overhangs**: Footholds contribute 25% to Factor 1 score

See [Factor 1: Hold Analysis](factor1_hold_analysis.md) for weighting details.

### Dynamic Threshold (Factor 3)

Wall angle affects when reaches become dynamic:
- **Slabs**: Balance enables longer static reaches
- **Overhangs**: Core tension limits static reach distance

**Advanced refinement** (optional): Adjust dynamic threshold by wall angle.

### Complexity Multipliers

Wall angle transitions (changing angles within route) apply additional complexity multiplier. See [Complexity Multipliers](complexity_multipliers.md).

## Wall Segments and Transitions

### Single Wall Angle (Simple Case)

Most routes have one dominant wall angle:
- Entire route on slab wall
- Entire route on overhang feature

**Implementation**: Use single wall angle category for whole route.

### Multiple Wall Angles (Complex Case)

Some routes traverse multiple wall angles:
- Start on vertical, finish on overhang
- Slab section followed by vertical section

**Advanced Implementation** (Phase 1c):

```python
def calculate_segmented_wall_score(wall_segments: list) -> float:
    """
    Calculate wall incline score for routes with multiple segments.

    Args:
        wall_segments: List of (angle_category, proportion) tuples
            Example: [('vertical', 0.6), ('overhang', 0.4)]

    Returns:
        Weighted average wall score
    """
    total_score = 0
    for angle_category, proportion in wall_segments:
        segment_score = calculate_wall_incline_score(angle_category)
        total_score += segment_score * proportion

    return total_score
```

**Example**:
- 60% vertical (score 6.0), 40% moderate overhang (score 9.0)
- Weighted: (6.0 × 0.6) + (9.0 × 0.4) = 3.6 + 3.6 = **7.2**

**Recommendation**: Start with single wall angle (Phase 1a). Add segment support in refinement (Phase 1c) if needed.

## Implementation Notes

### Phase 1a (MVP)

1. Add wall angle dropdown to route upload form
2. Store wall angle in database (`wall_incline` field)
3. Calculate wall incline score using direct mapping
4. Combine with Factors 1-3 using 20% weight

### Phase 1b (Calibration)

1. Analyze user feedback by wall angle
2. Adjust difficulty scores (3.0, 6.0, 7.5, 9.0, 11.0)
3. Optionally adjust wall angle weight (20%)
4. Monitor slab vs overhang prediction accuracy separately

### Phase 1c (Advanced Features)

1. Implement wall segment support
2. Add wall transition complexity multiplier
3. Test automatic wall angle detection (optional)

### Data Model

```python
# Add to Analysis model
wall_incline = Column(String(20), default='vertical')

# Optional: For multi-segment routes
wall_segments = Column(JSON, nullable=True)
# Format: [{"angle": "vertical", "proportion": 0.6}, {"angle": "overhang", "proportion": 0.4}]
```

### UI Mockup

```text
┌─────────────────────────────────────────┐
│ Upload Route Image                      │
├─────────────────────────────────────────┤
│                                         │
│ [Choose File] route_photo.jpg           │
│                                         │
│ Wall Angle:                             │
│ ┌──────────────────────────────────┐   │
│ │ ⚠ Vertical (90°)               ▼ │   │
│ └──────────────────────────────────┘   │
│   • Slab (< 90°)                        │
│   • Vertical (90°)                      │
│   • Slight Overhang (90° - 105°)        │
│   • Moderate Overhang (105° - 120°)     │
│   • Steep Overhang / Roof (> 120°)      │
│                                         │
│ [Analyze Route]                         │
└─────────────────────────────────────────┘
```

## Edge Cases

**Unknown wall angle:**
- Default to "vertical" (1.0 multiplier / 6.0 score)
- Log as missing data for calibration review

**Extreme angles (> 135°):**
- Treat as "steep overhang" category
- Very rare in practice

**Slight variations within category:**
- 88° vs 85° slab: Both use same "slab" score
- Category-based approach is sufficiently granular

## Summary

Factor 4 evaluates wall angle impact through:

1. ✅ **Wall angle categories** - Slab to steep overhang
2. ✅ **Biomechanical difficulty scaling** - 0.75x to 1.85x multipliers
3. ✅ **Manual user input** - Simple dropdown selection
4. ✅ **Future: Segment support** - Routes with multiple angles

**Result**: Wall incline difficulty score (range 1-12) or multiplier (0.75-1.85) reflecting biomechanical demands.

**Integration**: Wall angle also determines foothold weighting in Factor 1 and affects dynamic threshold in Factor 3.

**Next**: Combine all 4 factors in weighted scoring model. Optionally add [Complexity Multipliers](complexity_multipliers.md) in refinement phase.

