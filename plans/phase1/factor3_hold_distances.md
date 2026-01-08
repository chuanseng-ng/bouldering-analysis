# Factor 3: Hold Distance Analysis

## Objective

Evaluate difficulty based on the spatial distribution of holds - specifically, how far apart holds are and what movement types are required (static vs dynamic).

**Weight in Overall Score**: ~20% (starting point - requires calibration)

## Core Principle

**Larger distances between holds increase difficulty:**
- **Long reaches** require span, flexibility, or dynamic moves
- **Dynamic moves** require power, coordination, and higher risk
- **Spatial clustering** indicates rest positions vs continuous difficulty

**Key Insight**: Distance must be normalized by climber dimensions. A 6-foot reach is easier for a 6'2" climber than a 5'2" climber. Since individual climber data is usually unavailable, normalize by wall dimensions.

## Distance Calculation

### Inter-Hold Distance

Calculate Euclidean distance between consecutive holds:

```python
def calculate_inter_hold_distance(hold1, hold2):
    """
    Calculate distance between hold centers.
    """
    # Hold centers from bounding boxes
    center1_x = (hold1.bbox_x1 + hold1.bbox_x2) / 2
    center1_y = (hold1.bbox_y1 + hold1.bbox_y2) / 2

    center2_x = (hold2.bbox_x1 + hold2.bbox_x2) / 2
    center2_y = (hold2.bbox_y1 + hold2.bbox_y2) / 2

    # Euclidean distance
    distance = sqrt((center2_x - center1_x)^2 + (center2_y - center1_y)^2)

    return distance
```

### Normalized Distance

Normalize by wall height to make distances comparable across images:

```text
normalized_distance = pixel_distance / image_height
```

**Example**:
- Distance: 400 pixels
- Image height: 2000 pixels
- Normalized: 400 / 2000 = 0.20 (20% of wall height)

**Calibration Note**: Normalization approach will vary based on camera setup (fixed camera vs user photos). Adjust as needed.

## Distance Difficulty Tiers

### Classification

**Tier 1 - Very Short Distances (Easy)**
- Normalized distance: < 0.10 (< 10% wall height)
- **Score: 1-2**
- Interpretation: Holds close together, easy to reach
- Movement: Static, controlled

**Tier 2 - Short Distances (Moderate-Easy)**
- Normalized distance: 0.10 - 0.20
- **Score: 3-4**
- Interpretation: Comfortable static reaches
- Movement: Standard climbing moves

**Tier 3 - Moderate Distances (Moderate)**
- Normalized distance: 0.20 - 0.30
- **Score: 5-7**
- Interpretation: Extended reaches, requires technique
- Movement: Static or dynamic depending on climber

**Tier 4 - Long Distances (Hard)**
- Normalized distance: 0.30 - 0.45
- **Score: 8-10**
- Interpretation: Large reaches, likely dynamic
- Movement: Dynamic or powerful static (lockoffs)

**Tier 5 - Very Long Distances (Very Hard)**
- Normalized distance: > 0.45 (> 45% wall height)
- **Score: 11-12**
- Interpretation: Dyno or coordination moves
- Movement: Dynamic, high power and precision

**Calibration Note**: These thresholds are starting estimates. Adjust based on gym-specific route setting and user feedback.

### Distance Score Formula

```python
def calculate_distance_difficulty_score(normalized_distance: float) -> float:
    """
    Map normalized distance to difficulty score.

    Uses piecewise linear mapping with steeper penalty for long reaches.
    """
    if normalized_distance < 0.10:
        # Very short: linear from 1-2
        return 1 + (normalized_distance / 0.10)

    elif normalized_distance < 0.20:
        # Short: linear from 2-4
        return 2 + ((normalized_distance - 0.10) / 0.10) * 2

    elif normalized_distance < 0.30:
        # Moderate: linear from 4-7
        return 4 + ((normalized_distance - 0.20) / 0.10) * 3

    elif normalized_distance < 0.45:
        # Long: linear from 7-10
        return 7 + ((normalized_distance - 0.30) / 0.15) * 3

    else:
        # Very long: linear from 10-12, capped at 12
        return min(12, 10 + ((normalized_distance - 0.45) / 0.20) * 2)
```

## Aggregating Multiple Distances

### Approach Options

**Option A: Average Distance (Simple)**
```text
avg_distance = mean(all_inter_hold_distances)
Distance_Score = calculate_distance_difficulty_score(avg_distance)
```

**Pros**: Simple, fast
**Cons**: Ignores one very hard move that defines route difficulty

**Option B: Weighted Average (Emphasize Hard Moves)**
```python
def calculate_weighted_distance_score(distances: list) -> float:
    """
    Weighted average emphasizing the hardest reaches.

    Top 30% of distances weighted more heavily.
    """
    sorted_distances = sorted(distances, reverse=True)

    # Top 30% get 2x weight
    top_30_count = max(1, int(len(distances) * 0.3))
    top_distances = sorted_distances[:top_30_count]
    remaining_distances = sorted_distances[top_30_count:]

    weighted_sum = sum(top_distances) * 2 + sum(remaining_distances)
    weight_total = len(top_distances) * 2 + len(remaining_distances)

    weighted_avg = weighted_sum / weight_total

    return calculate_distance_difficulty_score(weighted_avg)
```

**Pros**: Captures crux moves
**Cons**: More complex

**Recommendation**: Start with Option A (average), upgrade to Option B if crux moves are under-predicted.

### Vertical vs Lateral Distances

**Consideration**: Vertical and lateral reaches have different difficulty:
- **Vertical reaches (upward)**: Require pulling strength, harder
- **Lateral reaches (sideways)**: Require core tension, balance
- **Downward reaches**: Unusual, typically easier

**Advanced Implementation** (optional future enhancement):
```python
def calculate_directional_distance_score(dx, dy, total_distance):
    """
    Adjust difficulty based on reach direction.
    """
    base_score = calculate_distance_difficulty_score(total_distance)

    vertical_ratio = abs(dy) / max(total_distance, 1)
    lateral_ratio = abs(dx) / max(total_distance, 1)

    # Vertical emphasis (pulling harder than traversing)
    if vertical_ratio > 0.7:  # Mostly vertical
        return base_score * 1.1
    elif lateral_ratio > 0.7:  # Mostly lateral
        return base_score * 1.05
    else:
        return base_score
```

**Calibration Note**: Start without directional adjustments. Add only if accuracy improves.

## Movement Type Detection

### Static vs Dynamic Classification

**Simple threshold-based approach:**

```python
def classify_movement_type(normalized_distance: float) -> str:
    """
    Classify movement as static or dynamic based on distance.

    Threshold is approximate - climber-dependent in reality.
    """
    DYNAMIC_THRESHOLD = 0.35  # 35% of wall height

    if normalized_distance < DYNAMIC_THRESHOLD:
        return "static"
    else:
        return "dynamic"  # Likely requires dyno or powerful static
```

**Calibration Note**: Dynamic threshold varies by:
- Climber height and wingspan
- Wall angle (overhangs require dynos at shorter distances)
- Hold types (good holds enable longer static reaches)

### Dynamic Move Penalty

If movement is classified as dynamic, apply additional difficulty:

```python
def calculate_factor3_score(distances: list) -> float:
    """
    Calculate Factor 3: Distance Score.

    Includes penalty for dynamic moves.
    """
    # Calculate base distance score
    avg_distance = mean(distances)
    base_score = calculate_distance_difficulty_score(avg_distance)

    # Count dynamic moves
    dynamic_count = sum(1 for d in distances if d > DYNAMIC_THRESHOLD)
    dynamic_ratio = dynamic_count / len(distances)

    # Apply dynamic penalty (up to +20% difficulty)
    dynamic_penalty = 1 + (dynamic_ratio * 0.2)

    final_score = base_score * dynamic_penalty

    return min(12, final_score)
```

**Rationale**: Dynamic moves add:
- Coordination requirements
- Higher fall risk
- Greater power demands
- Mental difficulty (committing to dynamic moves)

## Example Calculations

### Example 1: Close Static Moves

**Setup:**
- 10 inter-hold distances
- Average normalized distance: 0.15 (15% wall height)
- All moves static

**Calculation:**
- Base score: 2 + ((0.15 - 0.10) / 0.10) × 2 = 2 + 1 = **3.0**
- Dynamic ratio: 0/10 = 0.0
- Dynamic penalty: 1 + (0.0 × 0.2) = 1.0
- Final score: 3.0 × 1.0 = **3.0**

**Interpretation**: Easy, controlled movement.

### Example 2: Mixed Distances with Some Dynos

**Setup:**
- 8 inter-hold distances: [0.12, 0.18, 0.25, 0.28, 0.22, 0.38, 0.42, 0.20]
- Average: 0.256
- Dynamic count (>0.35): 2 moves

**Calculation:**
- Base score: 4 + ((0.256 - 0.20) / 0.10) × 3 = 4 + 1.68 = **5.68**
- Dynamic ratio: 2/8 = 0.25
- Dynamic penalty: 1 + (0.25 × 0.2) = 1.05
- Final score: 5.68 × 1.05 = **5.96**

**Interpretation**: Moderate difficulty with a couple hard dynamic moves.

### Example 3: Long Dyno Problem

**Setup:**
- 5 inter-hold distances: [0.15, 0.50, 0.45, 0.18, 0.55]
- Average: 0.366
- Dynamic count (>0.35): 3 moves

**Calculation:**
- Base score: 7 + ((0.366 - 0.30) / 0.15) × 3 = 7 + 1.32 = **8.32**
- Dynamic ratio: 3/5 = 0.60
- Dynamic penalty: 1 + (0.60 × 0.2) = 1.12
- Final score: 8.32 × 1.12 = **9.32**

**Interpretation**: Hard problem with multiple powerful dynos.

## Implementation Notes

### Minimum Viable Implementation

**Phase 1a:**
1. Calculate inter-hold distances (consecutive holds)
2. Normalize by wall height
3. Compute average distance score
4. No dynamic penalty initially (simplify)

**Phase 1b (Refinement):**
1. Add dynamic move detection
2. Apply dynamic penalty
3. Optionally implement weighted average (emphasize crux)
4. Calibrate thresholds based on feedback

### Determining Hold Sequence

**Challenge**: Detected holds are unordered. How to determine movement sequence?

**Approach Options:**

**Option A: Vertical Ordering (Simple)**
- Assume climbers move bottom to top
- Sort holds by y-coordinate (bbox_y position)
- Calculate consecutive distances

**Option B: Nearest Neighbor (Better)**
- Build graph of hold-to-hold distances
- Use minimum spanning tree or greedy nearest-neighbor
- Approximates likely climbing path

**Option C: User Annotation (Future)**
- Allow users to mark sequence during upload
- Most accurate, but requires user effort

**Recommendation**: Start with Option A (vertical ordering). Upgrade to Option B if sequence errors cause prediction issues.

### Edge Cases

**Single hold (start to finish):**
- No inter-hold distances
- Default to minimal score or use wall height as proxy

**Clustered holds (rest position):**
- Many short distances followed by one long distance
- Weighted average approach captures this better

**Traversing routes:**
- Vertical ordering fails
- May require nearest-neighbor approach

## Relationship to Other Factors

### Factor 1 (Hold Types)

- Long reaches to good holds: Easier
- Long reaches to crimps: Much harder
- **Interaction captured** in combined score

### Factor 2 (Hold Density)

- Sparse holds + long distances: Very hard
- Dense holds + short distances: Easier
- **Distance complements density** analysis

### Factor 4 (Wall Incline)

- Overhangs: Shorter reaches can force dynos
- Slabs: Balance makes long static reaches possible
- **Wall angle affects dynamic threshold** (future refinement)

## Summary

Factor 3 evaluates movement demands through:

1. ✅ **Inter-hold distances** - Normalized by wall height
2. ✅ **Difficulty tiers** - Short to very long reaches
3. ✅ **Dynamic move detection** - Threshold-based classification
4. ✅ **Dynamic penalty** - Additional difficulty for dynamic moves

**Result**: Distance difficulty score (range ~1-12) reflecting reach requirements and movement type.

**Next**: Combine with [Factor 1 (Hold Analysis)](factor1_hold_analysis.md), [Factor 2 (Density)](factor2_hold_density.md), and [Factor 4 (Wall Incline)](factor4_wall_incline.md).

