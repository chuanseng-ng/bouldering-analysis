# Phase 1: Route-Based Grade Prediction - Implementation Guide

## Executive Summary

This document outlines the **Phase 1 implementation** of the grade prediction algorithm - a sophisticated, climbing domain-aware system that predicts V-grades (V0-V12) from detected route characteristics.

**Core Objective**: Replace the current simplified grade prediction logic in [`predict_grade()`](../src/main.py:707) with a multi-factor algorithm that considers:

- **Four Primary Factors**: Hold types/sizes, hold count, hold distances, and wall incline
- **Two Complexity Multipliers**: Wall angle transitions and hold type variability

**Phase 1.5 Extension**: Optional persona-based personalization to provide customized difficulty predictions based on individual climbing styles and strengths.

**Status**: Core implementation focus. This phase must be completed, deployed, and validated before proceeding to Phase 2 (video analysis).

---

# PHASE 1: ROUTE-BASED GRADE PREDICTION (Core Implementation)

## Current State Analysis

### Existing Implementation

The current [`predict_grade()`](../src/main.py:707) function uses:

- **Simple hold count thresholds**: Direct mapping of hold count to base grades
- **Basic difficulty multipliers**: Small additive adjustments for crimps, pockets, and slopers
- **No distance analysis**: Hold spacing is completely ignored

### Limitations

1. No consideration of hold distances (critical for difficulty assessment)
2. Oversimplified hold type scoring (doesn't account for hold combinations)
3. Linear relationship assumptions (reality is non-linear)
4. No consideration of route sequence or spatial distribution
5. No wall incline factor (overhang vs slab significantly affects difficulty)
6. Caps at V10 instead of V12

### Available Data

From [`DetectedHold`](../src/models.py:150) model:

- Bounding box coordinates (bbox_x1, bbox_y1, bbox_x2, bbox_y2)
- Hold type classification (via hold_type_id)
- Detection confidence scores
- Analysis relationship (all holds for a given route)

From [`HOLD_TYPES`](../src/constants.py:10):

- 8 distinct hold types: crimp, jug, sloper, pinch, pocket, foot-hold, start-hold, top-out-hold

---

## Algorithm Design

### Four-Component Scoring System with Complexity Multipliers

The new algorithm uses a **weighted scoring model** that combines four independent factors, enhanced by two complexity multipliers:

```text
Base Score = f(Hold Difficulty Score, Hold Density Score, Distance Score, Wall Incline Score)
Final Score = Base Score × Wall Transition Multiplier × Hold Variability Multiplier
```

The four factors contribute to a base difficulty score through weighted averaging. The two complexity multipliers then amplify this base score to account for route characteristics that increase mental and physical load.

---

## Factor 1: Hold Type & Size Analysis

### Objective

Evaluate the technical difficulty of holds based on type and physical size.

### Hold Difficulty Classification

#### Tier 1 - Very Hard (Base Score: 10)

- **Crimps**: Small, narrow holds requiring finger strength
- **Pockets**: Small holes requiring specific finger positioning
- Size consideration: Smaller crimps/pockets (bbox area < threshold) add difficulty

#### Tier 2 - Hard (Base Score: 7)

- **Slopers**: Round holds requiring open-handed grip and body tension
- **Pinches**: Require thumb opposition and pinch strength

#### Tier 3 - Moderate (Base Score: 4)

- **Foot-holds**: Used for feet, moderate difficulty when used as handholds

#### Tier 4 - Easy (Base Score: 1)

- **Jugs**: Large, easy-to-grip holds
- **Start-holds**: Typically good holds to begin route
- **Top-out-holds**: Final holds, usually accessible

### Size Calculation

For each detected hold, calculate physical size:

```text
hold_width = bbox_x2 - bbox_x1
hold_height = bbox_y2 - bbox_y1
hold_area = hold_width * hold_height
```

### Size-Based Adjustments

Apply size modifiers based on hold type:

**Crimps & Pockets:**

- Extra small (area < 500px²): +3 difficulty bonus
- Small (area 500-1000px²): +2 difficulty bonus
- Medium (area 1000-2000px²): +1 difficulty bonus
- Large (area > 2000px²): 0 bonus (easier)

**Slopers:**

- Small (area < 1500px²): +2 difficulty bonus
- Medium (area 1500-3000px²): +1 difficulty bonus
- Large (area > 3000px²): 0 bonus

**Jugs:**

- Small jugs (area < 2000px²): +1 difficulty (not truly a jug)
- Large jugs: 0 bonus (remains easy)

### Hold Type Distribution Analysis

Calculate the **proportion** of hard holds vs easy holds:

```text
hard_hold_ratio = (count_crimps + count_pockets + count_slopers) / total_handholds
```

Where `total_handholds` excludes foot-holds, start-holds, and top-out-holds.

### Hold Difficulty Score Formula

```text
Hold_Difficulty_Score = Σ(hold_base_score × size_modifier) × (1 + hard_hold_ratio × 0.5)
```

The hard_hold_ratio multiplier creates a non-linear effect: routes with many hard holds are disproportionately harder.

### Normalization

Normalize by dividing by total handhold count:

```text
Normalized_Hold_Score = Hold_Difficulty_Score / total_handholds
```

This creates a per-hold difficulty average ranging approximately 1-13.

---

## Factor 2: Hold Count Analysis

### Objective

Assess route difficulty based on the number of available holds, with context-aware interpretation.

### Core Principle

**Inverse relationship with difficulty, but non-linear:**

- Very few holds (3-5): Extremely difficult (V8-V12) - requires powerful, precise moves
- Few holds (6-8): Hard (V5-V7) - limited options, technical sequences
- Moderate holds (9-12): Intermediate (V3-V5) - some options available
- Many holds (13-16): Moderate-easy (V1-V3) - multiple path options
- Very many holds (17+): Easy (V0-V2) - abundant options, likely easier sequences

### Exceptions & Context

**Hold count must be interpreted with hold types:**

- 5 jugs is easier than 15 crimps
- This interaction is captured by combining with Factor 1

### Hold Density Score Formula

Use a **logarithmic decay function** to model the non-linear relationship:

```text
Hold_Density_Score = 12 - (log₂(total_handholds) × 2.5)
```

This maps:

- 3 holds → ~8.2 score
- 5 holds → ~6.5 score
- 8 holds → ~5.0 score
- 12 holds → ~3.4 score
- 16 holds → ~2.0 score
- 20+ holds → ~1.0 score

Clamp the result between 0 and 12.

---

## Factor 3: Hold Distance Analysis

### Objective

Measure route difficulty based on hold spacing, capturing both average difficulty and crux moves.

### Spatial Analysis Required

#### Step 1: Sort Holds Vertically

Since bouldering routes typically ascend, sort detected holds by y-coordinate (bottom to top):

```text
sorted_holds = sort(holds, key=bbox_y1, ascending=True)
```

#### Step 2: Calculate Sequential Distances

For each consecutive pair of holds, calculate Euclidean distance:

```text
distance = √[(x₂ - x₁)² + (y₂ - y₁)²]
```

Where center points are:

```text
x_center = (bbox_x1 + bbox_x2) / 2
y_center = (bbox_y1 + bbox_y2) / 2
```

#### Step 3: Extract Distance Metrics

Calculate:

1. **Average distance**: Mean of all sequential distances
2. **Maximum distance**: Largest gap between consecutive holds (captures crux)
3. **Distance variance**: Standard deviation (measures consistency)

### Distance Interpretation (pixels at standard image resolution)

**Average Distance Ranges:**

- Close spacing (< 150px): Easy moves, static climbing → V0-V2
- Moderate spacing (150-300px): Standard reaches → V2-V5  
- Wide spacing (300-500px): Long reaches, dynamic moves → V5-V9
- Very wide spacing (> 500px): Powerful dynos required → V9-V12

**Maximum Distance (Crux Move):**

- < 200px: No significant crux → 0 bonus
- 200-400px: Moderate crux → +1 grade bonus
- 400-600px: Hard crux → +2 grade bonus
- > 600px: Extreme crux → +3 grade bonus

### Normalization for Image Resolution

Since images may vary in resolution, normalize distances:

```text
image_height = max(bbox_y2) for all holds
normalized_distance = raw_distance / image_height
```

Use normalized ratios:

- Close: < 0.15 (15% of image height)
- Moderate: 0.15-0.30
- Wide: 0.30-0.50
- Very wide: > 0.50

### Distance Score Formula

```text
Avg_Distance_Component = (normalized_avg_distance / 0.15) × 3
Max_Distance_Component = crux_bonus (0-3 based on max distance)
Distance_Score = Avg_Distance_Component + Max_Distance_Component
```

Clamp Distance_Score between 0 and 12.

---

## Factor 4: Wall Incline Analysis

### Objective

Assess difficulty based on the wall angle, accounting for how gravity affects the climb.

### Climbing Biomechanics of Wall Incline

Wall angle fundamentally changes how climbers interact with holds:

**Slab (< 90°, inclined away from climber):**

- Gravity helps keep climber on the wall
- Can use legs more effectively for support
- Better weight distribution over feet
- Easier to rest and maintain balance
- Lower difficulty relative to vertical

**Vertical (90°, perpendicular to ground):**

- Standard baseline for difficulty
- Balanced mix of upper body and lower body work
- Neutral gravity impact

**Overhang (> 90°, inclined toward climber):**

- Gravity pulls climber away from wall
- Requires significantly more upper body strength
- Harder to rest between moves
- Core tension required to stay on wall
- Higher difficulty relative to vertical

### Wall Incline Measurement Approaches

#### Option 1: Manual User Input (Recommended for Initial Implementation)

- **Method**: User specifies wall angle when uploading route image
- **Input format**: Angle in degrees or category (slab/vertical/overhang)
- **Pros**: Simple, accurate, no computer vision required
- **Cons**: Requires user input, potential for user error

#### Option 2: Computer Vision Detection (Future Enhancement)

- **Method**: Analyze image perspective and vanishing points
- **Technique**: Detect wall plane orientation relative to camera
- **Pros**: Automatic, no user input needed
- **Cons**: Complex, requires calibration, may be inaccurate

#### Option 3: Image Metadata (If Available)

- **Method**: Extract inclinometer data from smartphone sensors
- **Format**: EXIF metadata or companion sensor data
- **Pros**: Accurate, automatic if device supports it
- **Cons**: Requires device with sensors, not universally available

**Recommended Initial Approach**: Option 1 (Manual Input)

- Add wall_incline field to [`Analysis`](../src/models.py:32) model
- Provide UI dropdown with options: slab, vertical, slight_overhang, moderate_overhang, steep_overhang
- Map user selections to angle ranges for scoring

### Wall Incline Categories & Angles

Define standard categories with angle ranges:

| Category | Angle Range | Description | Typical Location |
|----------|-------------|-------------|------------------|
| **Slab** | 70°-89° | Inclined away from climber | Beginner walls, outdoor slabs |
| **Vertical** | 90° | Perpendicular to ground | Standard gym walls |
| **Slight Overhang** | 91°-105° | Gentle overhang | Intermediate gym sections |
| **Moderate Overhang** | 106°-120° | Noticeable overhang | Advanced gym walls, cave routes |
| **Steep Overhang** | 121°-135° | Severe overhang | Competition walls, roofs |
| **Roof** | 136°+ | Near horizontal | Elite routes, ceiling problems |

**Note**: For initial implementation, focus on the five main categories (excluding roof, which is rare).

### Wall Incline Scoring Function

Create a difficulty multiplier based on wall angle:

#### Scoring Formula

```text
Wall_Incline_Score = Base_Score × Angle_Multiplier

Where:
Base_Score = 6.0 (neutral baseline for vertical wall)

Angle_Multiplier:
- Slab (70°-89°):            0.65  →  Score: 3.9
- Vertical (90°):            1.00  →  Score: 6.0
- Slight Overhang (91°-105°): 1.25  →  Score: 7.5
- Moderate Overhang (106°-120°): 1.60  →  Score: 9.6
- Steep Overhang (121°-135°): 2.00  →  Score: 12.0
```

**Rationale for multipliers:**

- **0.65 for slabs**: Reduces difficulty by ~35% (V5 vertical becomes ~V3 on slab)
- **1.00 for vertical**: Baseline, no adjustment
- **1.25 for slight overhang**: Modest increase (~V5 becomes ~V6)
- **1.60 for moderate overhang**: Significant increase (~V5 becomes ~V7-8)
- **2.00 for steep overhang**: Major increase (~V5 becomes ~V8-9)

These multipliers align with climbing consensus that overhangs add 1-3 grades of difficulty.

#### Continuous Angle Scoring (Alternative)

For more granular scoring when exact angles are available:

```python
def calculate_incline_multiplier(angle_degrees: float) -> float:
    """
    Calculate difficulty multiplier based on wall angle.
    
    Args:
        angle_degrees: Wall angle (90° = vertical, >90° = overhang, <90° = slab)
    
    Returns:
        Multiplier value (0.5 to 2.5)
    """
    if angle_degrees < 90:
        # Slab: decreases difficulty
        # 70° → 0.65, 85° → 0.90, 90° → 1.00
        return 0.65 + (angle_degrees - 70) * 0.0175
    elif angle_degrees == 90:
        # Vertical: baseline
        return 1.00
    else:
        # Overhang: increases difficulty exponentially
        # 95° → 1.15, 105° → 1.40, 120° → 1.80, 135° → 2.20
        overhang_amount = angle_degrees - 90
        return 1.00 + (overhang_amount / 45) * 1.2
```

### Interaction with Other Factors

Wall incline interacts with hold difficulty:

**Slopers on overhang**: Extremely difficult (gravity pulls hand off hold)
**Crimps on slab**: Relatively easier (can use feet more)
**Jugs on overhang**: Still challenging (require sustained grip)

**Implementation Note**: Initial version uses independent scoring. Future enhancement could add interaction multipliers.

### Normalization

Wall_Incline_Score ranges from ~3.9 to 12.0, matching the 0-12 scale of other factors.

---

## Complexity Multipliers

### Overview

Complexity multipliers are **multiplicative factors** (range 1.0-1.5x) that amplify the base difficulty score to account for route characteristics that increase mental and physical demands beyond what the four base factors capture. Unlike the additive weighted factors, these multipliers model the exponential increase in difficulty when climbers must constantly adapt their technique.

### Climbing Domain Rationale

Routes with consistent characteristics allow climbers to:

- Develop rhythm and flow
- Settle into familiar movement patterns
- Conserve mental energy by relying on muscle memory
- Predict upcoming moves based on pattern recognition

Routes with transitions and variability force climbers to:

- Constantly reassess body position and technique
- Adapt to changing physical demands mid-route
- Maintain heightened mental focus throughout
- Manage multiple technical challenges simultaneously

This added complexity amplifies the baseline difficulty in a non-linear way, justifying multiplicative rather than additive scoring.

---

## Multiplier 1: Wall Angle Transitions

### Objective

Detect and quantify difficulty added when the wall angle changes during the route, forcing climbers to adapt body position and technique mid-climb.

### Transition Detection Approach

#### Required Data

To detect wall angle transitions, we need **spatial information** about where the wall changes angle along the route path. This requires one of the following:

**Option 1: Multi-segment User Input (Recommended for Initial Implementation)**

- User specifies wall segments and their angles when uploading route
- Example input format:
  
  ```json
  {
    "segments": [
      {"y_start": 0, "y_end": 400, "angle": "vertical"},
      {"y_start": 400, "y_end": 800, "angle": "moderate_overhang"},
      {"y_start": 800, "y_end": 1080, "angle": "steep_overhang"}
    ]
  }
  ```

- Map holds to segments based on y-coordinates
- Detect transitions when consecutive holds fall in different segments

**Option 2: Wall Feature Detection (Future Enhancement)**

- Use computer vision to detect wall boundaries and angle changes
- Analyze image geometry for perspective shifts
- Automatic segmentation of wall regions
- More complex but requires no user input

**Option 3: 3D Reconstruction (Advanced Future)**

- Use multiple images or depth sensors
- Build 3D model of climbing wall
- Extract precise angle data for each section
- Most accurate but technically demanding

**Recommended Initial Approach**: Option 1 (Multi-segment Input)

- Extend wall_incline field to support multiple segments
- Provide UI for adding segments with y-coordinate ranges
- Fall back to single angle if no segments specified

#### Transition Detection Algorithm

```python
def detect_wall_transitions(holds: list, wall_segments: list) -> dict:
    """
    Detect transitions between wall segments.
    
    Args:
        holds: List of DetectedHold objects sorted by y-coordinate
        wall_segments: List of wall segments with y_range and angle
    
    Returns:
        dict with transition count and magnitude information
    """
    transitions = []
    
    # Assign each hold to a segment
    for i in range(len(holds) - 1):
        current_hold = holds[i]
        next_hold = holds[i + 1]
        
        current_segment = find_segment(current_hold.y_center, wall_segments)
        next_segment = find_segment(next_hold.y_center, wall_segments)
        
        if current_segment != next_segment:
            # Transition detected
            angle_diff = abs(next_segment.angle - current_segment.angle)
            transitions.append({
                'from_angle': current_segment.angle,
                'to_angle': next_segment.angle,
                'magnitude': angle_diff,
                'hold_index': i
            })
    
    return {
        'transition_count': len(transitions),
        'transitions': transitions,
        'max_magnitude': max([t['magnitude'] for t in transitions]) if transitions else 0,
        'avg_magnitude': mean([t['magnitude'] for t in transitions]) if transitions else 0
    }
```

### Transition Magnitude Classification

Transitions vary in difficulty based on the angle change magnitude:

| Magnitude | Angle Change | Difficulty Impact | Example |
|-----------|--------------|-------------------|---------|
| **Minor** | 5°-15° | Low additional difficulty | Vertical → Slight overhang |
| **Moderate** | 16°-30° | Noticeable difficulty spike | Slight → Moderate overhang |
| **Major** | 31°-45° | Significant difficulty spike | Vertical → Steep overhang |
| **Extreme** | >45° | Very high difficulty spike | Slab → Roof transition |

### Wall Transition Scoring Function

Calculate multiplier based on transition count and magnitude:

```python
def calculate_transition_multiplier(transition_data: dict) -> float:
    """
    Calculate difficulty multiplier from wall transitions.
    
    Returns multiplier in range [1.0, 1.5]
    """
    if transition_data['transition_count'] == 0:
        return 1.0  # No transitions, no multiplier
    
    # Base multiplier from transition count
    count_factor = min(transition_data['transition_count'] * 0.08, 0.25)
    
    # Additional multiplier from magnitude
    avg_magnitude = transition_data['avg_magnitude']
    max_magnitude = transition_data['max_magnitude']
    
    if avg_magnitude < 15:
        magnitude_factor = 0.05
    elif avg_magnitude < 30:
        magnitude_factor = 0.15
    elif avg_magnitude < 45:
        magnitude_factor = 0.25
    else:
        magnitude_factor = 0.35
    
    # Bonus for extreme single transition
    extreme_bonus = 0.1 if max_magnitude > 45 else 0
    
    # Combine factors
    total_multiplier = 1.0 + count_factor + magnitude_factor + extreme_bonus
    
    # Clamp to maximum 1.5x
    return min(total_multiplier, 1.5)
```

**Examples:**

- No transitions → 1.0x (no change)
- 1 minor transition (10°) → 1.13x
- 2 moderate transitions (25° avg) → 1.31x
- 3 major transitions (40° avg) → 1.5x (capped)
- 1 extreme transition (50°) → 1.5x (capped)

### Edge Cases for Transitions

**Case 1: No Segment Data Provided**

- Default to single wall angle (no transitions)
- Multiplier = 1.0x
- Document this assumption in prediction metadata

**Case 2: Holds Concentrated in Single Segment**

- Even with multiple segments, no transitions if all holds in one section
- Multiplier = 1.0x

**Case 3: Very Short Segments**

- If segment height < 200px, may be annotation error
- Validate segment sizes, merge small segments
- Or flag for user review

**Case 4: Ambiguous Hold Assignment**

- Hold spans segment boundary (y-coordinate at transition point)
- Assign to segment containing hold centroid
- Or use weighted assignment if hold is large

---

## Multiplier 2: Hold Type Variability

### Objective

Quantify difficulty increase when routes use diverse hold types, preventing climbers from settling into consistent technique.

### Variability Measurement Approaches

#### Approach 1: Hold Type Entropy (Recommended)

Use Shannon entropy to measure distribution uniformity:

```python
def calculate_hold_type_entropy(hold_type_counts: dict) -> float:
    """
    Calculate Shannon entropy of hold type distribution.
    
    Higher entropy = more uniform distribution = more variability
    
    Args:
        hold_type_counts: Dictionary mapping hold types to counts
                         (exclude foot-holds, start-holds, top-out-holds)
    
    Returns:
        Entropy value (0 to ~2.08 for 5 hand hold types)
    """
    import math
    
    # Get handhold types only
    handhold_types = ['crimp', 'jug', 'sloper', 'pinch', 'pocket']
    counts = [hold_type_counts.get(ht, 0) for ht in handhold_types]
    total = sum(counts)
    
    if total == 0:
        return 0
    
    # Calculate Shannon entropy
    entropy = 0
    for count in counts:
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    
    return entropy
```

**Entropy Interpretation:**

- Entropy = 0: All same hold type (e.g., all crimps) → Consistent technique
- Entropy = 1.0: Some variety (e.g., mostly crimps, few jugs) → Moderate variety
- Entropy = 2.0+: High variety (e.g., even mix of 4+ types) → High variability

#### Approach 2: Standard Deviation of Hold Difficulty Scores

Measure variance in hold difficulty:

```python
def calculate_hold_difficulty_variance(holds: list) -> float:
    """
    Calculate standard deviation of hold difficulty scores.
    
    High variance = mix of easy and hard holds = more variability
    """
    from statistics import stdev
    
    difficulty_scores = [get_hold_base_score(h.hold_type) for h in holds]
    
    if len(difficulty_scores) < 2:
        return 0
    
    return stdev(difficulty_scores)
```

**Standard Deviation Interpretation:**

- StdDev < 2: Consistent difficulty (e.g., all jugs or all crimps)
- StdDev 2-4: Moderate variety (e.g., mix of similar difficulty)
- StdDev > 4: High variety (e.g., jugs mixed with crimps)

#### Approach 3: Unique Hold Type Count

Simple count of distinct hold types used:

```python
def count_unique_hold_types(hold_type_counts: dict) -> int:
    """Count number of distinct handhold types present."""
    handhold_types = ['crimp', 'jug', 'sloper', 'pinch', 'pocket']
    return sum(1 for ht in handhold_types if hold_type_counts.get(ht, 0) > 0)
```

**Unique Count Interpretation:**

- 1 type: No variability
- 2-3 types: Moderate variability
- 4-5 types: High variability

**Recommended Approach**: Entropy (Approach 1)

- Most sophisticated measure of distribution
- Accounts for both variety and balance
- Standard information theory metric
- Normalizes across different hold counts

### Hold Variability Scoring Function

Calculate multiplier based on entropy:

```python
def calculate_variability_multiplier(entropy: float, unique_types: int) -> float:
    """
    Calculate difficulty multiplier from hold type variability.
    
    Returns multiplier in range [1.0, 1.5]
    
    Args:
        entropy: Shannon entropy of hold type distribution (0-2.32)
        unique_types: Number of distinct hold types used
    """
    # Base multiplier from entropy
    # Map entropy [0, 2.32] to multiplier [1.0, 1.4]
    max_entropy = 2.32  # log2(5) for 5 hold types perfectly distributed
    entropy_multiplier = 1.0 + (entropy / max_entropy) * 0.4
    
    # Bonus for using many different types
    if unique_types >= 4:
        type_bonus = 0.1
    elif unique_types >= 3:
        type_bonus = 0.05
    else:
        type_bonus = 0
    
    # Combine and clamp
    total_multiplier = entropy_multiplier + type_bonus
    return min(total_multiplier, 1.5)
```

**Examples:**

- All crimps (entropy=0, types=1) → 1.0x
- Mostly crimps, some jugs (entropy=0.8, types=2) → 1.14x
- Even mix of 3 types (entropy=1.58, types=3) → 1.32x
- Even mix of 4 types (entropy=2.0, types=4) → 1.44x
- Even mix of 5 types (entropy=2.32, types=5) → 1.5x (capped)

### Interaction Considerations

**Hold variability interacts with hold difficulty:**

- Mix of jugs and slopers (easy + hard) → High variability, moderate base difficulty
- Mix of crimps and pockets (hard + hard) → High variability, high base difficulty
- The multiplier applies to whatever the base score is, so both cases get harder

**Wall incline affects hold variability impact:**

- Slopers on overhang are harder than on vertical
- But variability multiplier applies consistently
- The base score already accounts for wall angle via Factor 4

### Edge Cases for Hold Variability

**Case 1: Very Few Holds (<4)**

- Limited data for entropy calculation
- Entropy may be artificially low
- Apply multiplier conservatively (cap at 1.2x for <4 holds)

**Case 2: All Holds Same Type**

- Entropy = 0, multiplier = 1.0x
- Correct behavior: no variability penalty

**Case 3: Dominated by One Type (>80%)**

- Low entropy despite multiple types present
- Multiplier ~1.05-1.15x
- Correct: slight variability, slight penalty

**Case 4: Foot-holds Misclassified as Handholds**

- Could artificially inflate variability
- Mitigate by excluding foot-holds from calculation
- Use confidence scores to filter uncertain classifications

---

## Combined Scoring & Grade Mapping

### Weighted Composite Score with Multipliers

The scoring process happens in two stages:

**Stage 1: Calculate Base Score**

Combine the four factor scores using empirically-derived weights:

```text
Base_Score = (
    Hold_Difficulty_Score × 0.35 +
    Hold_Density_Score × 0.25 +
    Distance_Score × 0.20 +
    Wall_Incline_Score × 0.20
)
```

**Rationale for weights:**

- **35% Hold Difficulty**: Primary determinant - a route of all crimps is fundamentally harder
- **25% Hold Density**: Significant impact - fewer holds means fewer options
- **20% Distance**: Important for distinguishing grades - wide spacing requires power/technique
- **20% Wall Incline**: Critical modifier - overhang vs slab can change grade by 2-3 levels

**Stage 2: Apply Complexity Multipliers**

Apply multiplicative factors for route complexity:

```text
Transition_Multiplier = calculate_transition_multiplier(wall_transitions)  # 1.0-1.5x
Variability_Multiplier = calculate_variability_multiplier(hold_entropy)    # 1.0-1.5x

Final_Score = Base_Score × Transition_Multiplier × Variability_Multiplier
```

**Multiplier Impact Examples:**

| Base Score | Transitions | Variability | Final Score | Impact |
|------------|-------------|-------------|-------------|--------|
| 5.0 (V5)   | None (1.0x) | Low (1.0x)  | 5.0         | No change |
| 5.0 (V5)   | 2 moderate (1.3x) | Low (1.0x) | 6.5 (V6-V7) | +1-2 grades |
| 5.0 (V5)   | None (1.0x) | High (1.4x) | 7.0 (V7)    | +2 grades |
| 5.0 (V5)   | 2 major (1.5x) | High (1.4x) | 10.5 (V10-V11) | +5-6 grades |
| 8.0 (V8)   | 1 moderate (1.2x) | Moderate (1.2x) | 11.5 (V11-V12) | +3-4 grades |

**Combined Multiplier Range:**

- Minimum: 1.0x × 1.0x = 1.0x (no complexity)
- Maximum: 1.5x × 1.5x = 2.25x (extreme complexity)
- Typical: 1.1x - 1.4x combined (most routes have some complexity)

**Why Multiplicative?**

- Complexity factors amplify baseline difficulty exponentially, not linearly
- A hard route with transitions becomes disproportionately harder
- Matches climber experience: "V5 with transitions feels like V7"
- Prevents over-scoring: Easy routes (V0-V2) don't become V5 from complexity alone

### Score to V-Grade Mapping

Map the composite score (range 0-12) to V-grades (V0-V12):

```text
V0:  0.0 - 1.0
V1:  1.0 - 2.0
V2:  2.0 - 3.0
V3:  3.0 - 4.0
V4:  4.0 - 5.0
V5:  5.0 - 6.0
V6:  6.0 - 7.0
V7:  7.0 - 8.0
V8:  8.0 - 9.0
V9:  9.0 - 10.0
V10: 10.0 - 11.0
V11: 11.0 - 11.5
V12: 11.5 - 12.0
```

### Confidence Score Adjustment

Integrate the detection confidence scores:

```text
if average_confidence < 0.5:
    add_grade_uncertainty_flag = True
    
confidence_factor = min(average_confidence / 0.7, 1.0)
adjusted_score = Composite_Score × confidence_factor
```

Low confidence detections should reduce predicted grade (conservative approach).

---

## Algorithm Structure

### High-Level Flow

```text
┌───────────────────────────────────────────────────────┐
│  Input: Detected Holds + Wall Segments (optional)     │
│  (from YOLO detection + user input)                   │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│  Preprocessing                                        │
│  - Filter by confidence                               │
│  - Categorize holds                                   │
│  - Calculate dimensions                               │
│  - Parse wall incline/segments                        │
│  - Calculate hold type entropy                        │
└───────────────┬───────────────────────────────────────┘
                │
                ├────────┬──────────┬──────────┬──────────┐
                ▼        ▼          ▼          ▼          │
        ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
        │Hold    │  │Hold    │  │Distance│  │Wall    │  │
        │Type &  │  │Count   │  │Analysis│  │Incline │  │
        │Size    │  │Analysis│  │        │  │Analysis│  │
        │Score 1 │  │Score 2 │  │Score 3 │  │Score 4 │  │
        └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  │
            │           │            │           │        │
            └───────────┴────────────┴───────────┴────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Weighted Combination │
                    │ Base Score = Σ(Si×Wi)│
                    └──────────┬───────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
    ┌──────────────┐                    ┌──────────────┐
    │ Wall Angle   │                    │ Hold Type    │
    │ Transitions  │                    │ Variability  │
    │ Multiplier   │                    │ Multiplier   │
    │ (1.0-1.5x)   │                    │ (1.0-1.5x)   │
    └──────┬───────┘                    └──────┬───────┘
           │                                   │
           └────────────┬──────────────────────┘
                        ▼
              ┌──────────────────┐
              │ Apply Multipliers│
              │ Final Score =    │
              │ Base × M1 × M2   │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Confidence       │
              │ Adjustment       │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Grade Mapping    │
              │ Score → V-Grade  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Output: V-Grade  │
              │ (V0 - V12)       │
              └──────────────────┘
```

### Pseudocode Structure

```python
def predict_grade_v2(
    features: dict,
    detected_holds: list,
    wall_segments: list = None,
    wall_incline: str = 'vertical'
) -> tuple[str, float, dict]:
    """
    Predict climbing grade using sophisticated multi-factor analysis with complexity multipliers.
    
    Args:
        features: Dictionary with hold counts and types
        detected_holds: List of DetectedHold objects with bbox coordinates
        wall_segments: Optional list of wall segments with angles (for transitions)
                      If None, assumes single wall angle
        wall_incline: Default wall angle category if wall_segments not provided
    
    Returns:
        tuple: (predicted_grade, confidence_score, score_breakdown)
    """
    # Preprocessing
    handholds = filter_handholds(detected_holds)
    confidence_avg = calculate_average_confidence(handholds)
    hold_type_counts = count_hold_types(handholds)
    
    # Stage 1: Calculate Base Score from 4 Factors
    
    # Factor 1: Hold Type & Size Analysis
    hold_difficulty_score = analyze_hold_difficulty(handholds)
    
    # Factor 2: Hold Count Analysis
    hold_density_score = analyze_hold_density(len(handholds))
    
    # Factor 3: Distance Analysis
    distance_score = analyze_hold_distances(handholds)
    
    # Factor 4: Wall Incline Analysis
    wall_incline_score = analyze_wall_incline(wall_segments or wall_incline)
    
    # Combine scores into base score
    base_score = (
        hold_difficulty_score * 0.35 +
        hold_density_score * 0.25 +
        distance_score * 0.20 +
        wall_incline_score * 0.20
    )
    
    # Stage 2: Calculate Complexity Multipliers
    
    # Multiplier 1: Wall Angle Transitions
    if wall_segments and len(wall_segments) > 1:
        transition_data = detect_wall_transitions(handholds, wall_segments)
        transition_multiplier = calculate_transition_multiplier(transition_data)
    else:
        transition_data = {'transition_count': 0}
        transition_multiplier = 1.0
    
    # Multiplier 2: Hold Type Variability
    hold_entropy = calculate_hold_type_entropy(hold_type_counts)
    unique_types = count_unique_hold_types(hold_type_counts)
    variability_multiplier = calculate_variability_multiplier(hold_entropy, unique_types)
    
    # Apply multipliers to base score
    final_score = base_score * transition_multiplier * variability_multiplier
    
    # Stage 3: Confidence Adjustment & Grade Mapping
    
    # Adjust for confidence
    adjusted_score = apply_confidence_adjustment(final_score, confidence_avg)
    
    # Map to grade
    predicted_grade = map_score_to_grade(adjusted_score)
    
    # Prepare detailed breakdown for debugging/explainability
    score_breakdown = {
        'base_factors': {
            'hold_difficulty': hold_difficulty_score,
            'hold_density': hold_density_score,
            'distance': distance_score,
            'wall_incline': wall_incline_score
        },
        'base_score': base_score,
        'multipliers': {
            'wall_transitions': transition_multiplier,
            'hold_variability': variability_multiplier,
            'combined': transition_multiplier * variability_multiplier
        },
        'final_score': final_score,
        'adjusted_score': adjusted_score,
        'transition_data': transition_data,
        'entropy': hold_entropy,
        'unique_types': unique_types
    }
    
    return predicted_grade, confidence_avg, score_breakdown
```

---

## Data Requirements

### Required Data Available

✅ Bounding box coordinates (from [`DetectedHold`](../src/models.py:150))
✅ Hold type classifications (from [`HoldType`](../src/models.py:125))
✅ Confidence scores (from detection)
✅ Analysis relationship (all holds for a route)

### New Data Required for Wall Incline and Complexity Multipliers

❗ **Wall incline/angle and segments**: Not currently captured in database

**Implementation Options:**

1. **Add to Analysis Model (Recommended)**
   - Add `wall_incline` field to [`Analysis`](../src/models.py:32) table for single-angle routes
   - Add `wall_segments` JSON field for multi-segment routes with transitions
   - Type: String (enum) for single angle, JSON for segments
   - Default: 'vertical' for wall_incline, null for wall_segments
   - Nullable: Yes (defaults to vertical if not provided)

2. **Database Schema Change**

   ```python
   # In Analysis model
   wall_incline = Column(String(20), default='vertical')
   # Options: 'slab', 'vertical', 'slight_overhang', 'moderate_overhang', 'steep_overhang'
   
   wall_segments = Column(JSON, nullable=True)
   # Format: [
   #   {"y_start": 0, "y_end": 400, "angle": 90, "category": "vertical"},
   #   {"y_start": 400, "y_end": 800, "angle": 110, "category": "moderate_overhang"}
   # ]
   
   # OR for precise angles (alternative):
   wall_angle_degrees = Column(Float, default=90.0)
   # Range: 70-135 degrees
   ```

3. **UI Input Mechanism**
   - Add dropdown/slider to upload form in [`templates/index.html`](../src/templates/index.html)
   - For simple routes: Single wall_incline selector
   - For complex routes: Option to add multiple segments with y-coordinate ranges
   - Submit wall_incline or wall_segments with image upload
   - Pass to [`analyze_image()`](../src/main.py:655) function

**Wall Segments Data Structure:**

```json
{
  "wall_segments": [
    {
      "segment_id": 1,
      "y_start": 0,
      "y_end": 500,
      "angle_degrees": 90,
      "category": "vertical"
    },
    {
      "segment_id": 2,
      "y_start": 500,
      "y_end": 1000,
      "angle_degrees": 115,
      "category": "moderate_overhang"
    }
  ]
}
```

**Validation Rules:**

- Segments must cover continuous y-ranges without gaps
- y_start of segment N+1 should equal y_end of segment N
- If wall_segments is null or empty, fall back to single wall_incline
- Minimum segment height: 200px (to avoid micro-segments)

### Data Preprocessing Needed

1. **Hold Filtering**
   - Exclude holds with confidence < threshold (already done in [`_process_detection_results`](../src/main.py:520))
   - Separate handholds from foot-holds, start-holds, top-out-holds

2. **Dimension Calculation**
   - Calculate width, height, area for each hold
   - Store in features dictionary for efficiency

3. **Spatial Organization**
   - Sort holds by vertical position (ascending y-coordinate)
   - Calculate hold centroid positions

4. **Wall Incline Parsing**
   - Convert category string to angle or multiplier
   - Validate input (handle invalid values with default to vertical)
   - Store in features for scoring

5. **Image Metadata**
   - Extract image dimensions (height, width) for normalization
   - Can be obtained when loading image in [`analyze_image`](../src/main.py:655)

### New Data to Capture

Add to [`features_extracted`](../src/models.py:47) JSON field:

```json
{
  "total_holds": 12,
  "hold_types": {...},
  "average_confidence": 0.85,
  "wall_incline": "moderate_overhang",
  "wall_angle_degrees": 110.0,
  "wall_segments": [
    {"y_start": 0, "y_end": 500, "angle": 90, "category": "vertical"},
    {"y_start": 500, "y_end": 1080, "angle": 115, "category": "moderate_overhang"}
  ],
  "hold_dimensions": [
    {"hold_id": 1, "width": 45, "height": 30, "area": 1350},
    ...
  ],
  "distance_metrics": {
    "average_distance": 245.5,
    "max_distance": 520.0,
    "normalized_avg": 0.22,
    "normalized_max": 0.47
  },
  "image_dimensions": {
    "width": 1920,
    "height": 1080
  },
  "complexity_analysis": {
    "wall_transitions": {
      "count": 1,
      "transitions": [
        {"from_angle": 90, "to_angle": 115, "magnitude": 25, "hold_index": 6}
      ],
      "max_magnitude": 25,
      "avg_magnitude": 25
    },
    "hold_variability": {
      "entropy": 1.85,
      "unique_types": 4,
      "type_distribution": {
        "crimp": 5,
        "jug": 3,
        "sloper": 2,
        "pinch": 2
      }
    }
  },
  "score_breakdown": {
    "base_factors": {
      "hold_difficulty": 7.2,
      "hold_density": 4.1,
      "distance": 5.8,
      "wall_incline": 9.6
    },
    "base_score": 7.1,
    "multipliers": {
      "wall_transitions": 1.31,
      "hold_variability": 1.28,
      "combined": 1.68
    },
    "final_score": 11.9,
    "adjusted_score": 11.3,
    "predicted_grade": "V11"
  }
}
```

This detailed breakdown enables:

- Debugging and validation
- User feedback correlation
- Algorithm refinement
- Explainability (show why a grade was assigned and how multipliers affected it)
- Analysis of which factors contribute most to difficulty
- Identification of routes with high complexity vs high base difficulty

### Data Migration Considerations

**For existing routes without wall_incline:**

- Default to 'vertical' (safest assumption for gym routes)
- Flag these routes for potential user correction
- Allow users to update wall_incline retroactively
- Recalculate grade when wall_incline is updated

---

## Edge Cases & Special Scenarios

### Edge Case 1: Very Few Holds Detected (< 3)

**Problem**: Insufficient data for reliable grading

**Solution**:

- Return low confidence grade (V0-V2)
- Set confidence_score < 0.3
- Add warning flag in response
- Consider as incomplete route detection

### Edge Case 2: Only Easy Holds Detected (all jugs)

**Problem**: May indicate detection failure (missed smaller holds)

**Solution**:

- Apply hold_difficulty_score normally (will be low)
- Check if this pattern is common in training data
- Flag for potential reprocessing with lower confidence threshold

### Edge Case 3: Unusual Hold Distributions

**Problem**: E.g., all holds in horizontal line (traverse), all clustered

**Solution**:

- Calculate distance variance
- High variance → likely normal route
- Low variance + horizontal → could be traverse (different grading)
- For v1: Treat all routes as vertical; future enhancement for traverse detection

### Edge Case 4: Start/Top-out Holds Far from Route

**Problem**: These holds might be spatially separated, skewing distance calculations

**Solution**:

- Exclude start-hold and top-out-hold from distance calculations
- Include only intermediate handholds for distance metrics
- Still consider them for hold count (they exist on route)

### Edge Case 5: Image Resolution Variance

**Problem**: Distance thresholds may not scale properly

**Solution**:

- Always normalize distances by image height
- Use ratio-based thresholds (0.15, 0.30, etc.)
- Validate normalized approach with various resolution test images

### Edge Case 6: Overlapping Holds

**Problem**: Two holds detected very close together (distance < 50px)

**Solution**:

- Could indicate detection error or actual close holds
- Don't filter out - close holds are legitimate
- Small distances reduce difficulty score appropriately

### Edge Case 7: High Difficulty Score but Many Holds

**Problem**: 20 crimps detected - conflicting signals

**Solution**:

- Weighted system handles this: many holds reduces density score
- But high difficulty score from crimps increases overall
- Net result: Moderate grade (V4-V6) - reasonable for "juggy" crimp route

### Edge Case 8: Missing Wall Incline Data

**Problem**: User doesn't provide wall incline information

**Solution**:

- Default to 'vertical' (most common gym wall type)
- Assign neutral score (6.0) - no difficulty adjustment
- Flag prediction with lower confidence or add note
- Prompt user to add wall incline for better accuracy

### Edge Case 9: Extreme Wall Angles

**Problem**: User inputs unusual angles (roof >135°, extreme slab <70°)

**Solution**:

- Validate input range (70°-135° recommended)
- Cap multipliers at defined bounds (0.65-2.00)
- For angles outside range, use closest valid value
- Consider flagging for manual review

### Edge Case 10: Missing Wall Segment Data

**Problem**: User wants to specify transitions but doesn't provide y-coordinates

**Solution**:

- Fall back to single wall_incline (no transitions)
- Transition multiplier = 1.0x
- Log warning for incomplete data
- Suggest adding segment data for better accuracy

### Edge Case 11: Overlapping or Invalid Segments

**Problem**: Segments have overlapping y-ranges or gaps

**Solution**:

- Validate segments during input
- Reject invalid configurations with clear error message
- Auto-merge adjacent segments with same angle
- Fill small gaps (<50px) automatically

### Edge Case 12: All Holds in One Segment Despite Multiple Segments

**Problem**: Wall has multiple segments but all holds are in one section

**Solution**:

- Transition count = 0 even though segments exist
- Transition multiplier = 1.0x (correct behavior)
- Route effectively has no transitions from climbing perspective

### Edge Case 13: Extreme Complexity Multipliers

**Problem**: Route with both high transitions and high variability could exceed score cap

**Solution**:

- Combined multiplier can reach 2.25x (1.5 × 1.5)
- This could push easy routes (V2 base) to V4-5
- Consider capping final_score at 12.0 before mapping to grade
- Document that extreme complexity routes may hit ceiling

### Edge Case 14: Very Low Variability but Multiple Hold Types

**Problem**: Route has 3 types but one dominates (e.g., 90% crimps, few jugs)

**Solution**:

- Entropy will be low (~0.5)
- Variability multiplier ~1.09x (small increase)
- Correct behavior: slight variety doesn't significantly increase difficulty

### Edge Case 15: Transitions Between Similar Angles

**Problem**: Small angle changes (e.g., 90° → 92°) detected as transitions

**Solution**:

- Set minimum magnitude threshold (e.g., 5°)
- Ignore transitions below threshold
- Prevents noise from affecting multiplier
- Document threshold in configuration

---

## Validation & Calibration Strategy

### Calibration Data Needed

1. **Ground truth grades**: Collect user feedback from [`Feedback`](../src/models.py:87) table
2. **Route diversity**: Ensure calibration across V0-V12 range
3. **Minimum samples**: At least 10 routes per grade level (120 total)

### Validation Approach

1. **Weight Tuning**:
   - Initial weights: 35/25/20/20 (hold/density/distance/incline)
   - Use grid search or optimization to find best weights
   - Minimize mean absolute error against ground truth grades
   - Test weight sensitivity for wall incline factor

2. **Threshold Adjustment**:
   - Distance thresholds (150px, 300px, 500px) are initial estimates
   - Adjust based on actual route data
   - May need separate thresholds for gym vs outdoor routes
   - Validate wall incline multipliers against climber consensus

3. **Score Mapping Refinement**:
   - Linear mapping (0-12 → V0-V12) is starting point
   - May need non-linear mapping if score distribution is skewed
   - Consider using percentile-based mapping
   - Test if wall incline affects score distribution

### Testing Strategy

1. **Unit Tests**:
   - Test each component function independently
   - Verify edge case handling
   - Check score boundaries

2. **Integration Tests**:
   - Test with synthetic hold data of known difficulty
   - Verify full pipeline from DetectedHold → grade

3. **Accuracy Testing**:
   - Compare predictions against user feedback
   - Calculate accuracy within ±1 grade tolerance
   - Target: >70% within ±1 grade for initial version

---

## Implementation Considerations

### Code Organization

Create new module: `src/grade_prediction.py`

```text
src/grade_prediction.py
├── # Base factor analysis functions
├── calculate_hold_dimensions()
├── filter_handholds()
├── analyze_hold_difficulty()
├── analyze_hold_density()
├── analyze_hold_distances()
├── analyze_wall_incline()
├──
├── # Complexity multiplier functions (NEW)
├── detect_wall_transitions()
├── calculate_transition_multiplier()
├── calculate_hold_type_entropy()
├── count_unique_hold_types()
├── calculate_variability_multiplier()
├──
├── # Scoring and mapping functions
├── combine_scores()
├── apply_multipliers()
├── apply_confidence_adjustment()
├── map_score_to_grade()
├──
├── # Main entry point
└── predict_grade_v2()
```

Update [`src/main.py`](../src/main.py):

- Replace call to current `predict_grade()` with `predict_grade_v2()`
- Pass `features` dict, `detected_holds` list, `wall_segments`, and `wall_incline` parameter
- Update to handle new return format (grade, confidence, score_breakdown)
- Extract wall_incline and wall_segments from Analysis model or request parameters
- Store complexity analysis data in features_extracted JSON field

### Backward Compatibility

**Option 1: Gradual Migration**

- Keep old `predict_grade()` as `predict_grade_v1()`
- Add feature flag to switch between algorithms
- Compare both outputs during transition period

**Option 2: Direct Replacement**

- Replace `predict_grade()` entirely
- Update all tests
- Accept that historical predictions used different algorithm

**Recommendation**: Option 2 - clean break, simpler codebase

### Performance Considerations

**Computational Complexity**:

- Hold dimension calculation: O(n) where n = number of holds
- Distance calculation: O(n log n) for sorting + O(n) for distances
- Overall: O(n log n) - acceptable for typical routes (5-30 holds)

**Optimization Opportunities**:

- Cache hold dimensions in DetectedHold model (future enhancement)
- Pre-calculate centroids during detection phase
- Vectorize distance calculations using NumPy

### Dependencies

**Required Libraries** (already available):

- `math` (for sqrt, log calculations)
- `statistics` (for mean, stdev)
- Database models (DetectedHold, HoldType)

**No New Dependencies Needed** ✅

### Configuration Management

Add to [`src/cfg/user_config.yaml`](../src/cfg/user_config.yaml):

```yaml
grade_prediction:
  algorithm_version: "v2"
  weights:
    hold_difficulty: 0.35
    hold_density: 0.25
    distance: 0.20
    wall_incline: 0.20
  distance_thresholds:
    close: 150
    moderate: 300
    wide: 500
    very_wide: 600
  size_thresholds:
    crimp_small: 500
    crimp_medium: 1000
    crimp_large: 2000
    sloper_small: 1500
    sloper_large: 3000
  wall_incline_multipliers:
    slab: 0.65
    vertical: 1.00
    slight_overhang: 1.25
    moderate_overhang: 1.60
    steep_overhang: 2.00
  # Complexity multiplier configuration
  complexity_multipliers:
    transition:
      enabled: true
      max_multiplier: 1.5
      count_factor: 0.08  # Per transition
      magnitude_thresholds:
        minor: 15  # degrees
        moderate: 30
        major: 45
      magnitude_factors:
        minor: 0.05
        moderate: 0.15
        major: 0.25
        extreme: 0.35
      extreme_bonus: 0.1
      min_magnitude: 5  # Ignore transitions smaller than this
      min_segment_height: 200  # pixels
    variability:
      enabled: true
      max_multiplier: 1.5
      entropy_weight: 0.4  # Map max entropy to this multiplier contribution
      type_bonuses:
        three_types: 0.05
        four_plus_types: 0.1
      min_holds_for_full_multiplier: 4  # Cap multiplier for very few holds
  grade_range:
    min: 0  # V0
    max: 12  # V12
  confidence_threshold: 0.5
  default_wall_incline: "vertical"
  max_final_score: 12.0  # Cap to prevent overflow from extreme multipliers
```

This enables:

- Easy adjustment of parameters without code changes
- Different configurations for different climbing contexts
- A/B testing of algorithm variants
- Fine-tuning of complexity multipliers independently
- Enabling/disabling multipliers for comparison testing

---

## Testing & Validation Plan

### Test Data Requirements

1. **Synthetic Test Cases**:
   - Create controlled DetectedHold data with known characteristics
   - Test each component independently
   - Verify edge cases

2. **Real Route Data**:
   - Use existing analyses from database
   - Require user feedback for ground truth
   - Cover diverse grade range

### Test Coverage Areas

1. **Unit Tests for Components**:
   - `test_calculate_hold_dimensions()`: Verify area calculations
   - `test_filter_handholds()`: Ensure correct hold type filtering
   - `test_analyze_hold_difficulty()`: Test difficulty scoring
   - `test_analyze_hold_density()`: Verify logarithmic relationship
   - `test_analyze_hold_distances()`: Test distance calculations
   - `test_analyze_wall_incline()`: Test incline scoring and multipliers
   - `test_map_score_to_grade()`: Verify grade boundaries
   - `test_detect_wall_transitions()`: Test transition detection logic
   - `test_calculate_transition_multiplier()`: Verify multiplier calculation
   - `test_calculate_hold_type_entropy()`: Test entropy calculation
   - `test_calculate_variability_multiplier()`: Verify variability scoring

2. **Integration Tests**:
   - `test_predict_grade_v2_integration()`: Full pipeline test with wall incline
   - `test_backward_compatibility()`: Ensure API compatibility
   - `test_confidence_adjustment()`: Verify confidence handling
   - `test_wall_incline_defaults()`: Test missing wall incline data handling
   - `test_complexity_multipliers_integration()`: Test with transitions and variability
   - `test_single_vs_multi_segment()`: Compare single wall vs segmented wall
   - `test_multiplier_combinations()`: Test various multiplier scenarios

3. **Validation Tests**:
   - `test_grade_accuracy()`: Compare against ground truth
   - `test_grade_distribution()`: Ensure reasonable grade spread
   - `test_consistency()`: Same input → same output
   - `test_wall_incline_impact()`: Verify wall incline changes grade appropriately
   - `test_transition_impact()`: Verify transitions increase difficulty
   - `test_variability_impact()`: Verify hold variety increases difficulty
   - `test_multiplier_bounds()`: Ensure multipliers stay within 1.0-1.5x range
   - `test_extreme_complexity()`: Test routes with max complexity don't break

### Acceptance Criteria

✅ All unit tests pass  
✅ Integration tests pass  
✅ Accuracy ≥60% exact match, ≥80% within ±1 grade (on validation set)  
✅ No regression in existing test cases  
✅ Edge cases handled gracefully (no crashes)  
✅ Performance acceptable (< 100ms per prediction)  
✅ Code review approved  
✅ Documentation complete  

---

## Migration & Deployment

### Phase 1: Development

- [ ] Add `wall_incline` field to Analysis model (database migration)
- [ ] Add `wall_segments` JSON field to Analysis model
- [ ] Update UI to capture wall incline input (simple mode)
- [ ] Update UI to capture wall segments input (advanced mode)
- [ ] Implement `analyze_wall_incline()` function
- [ ] Implement `detect_wall_transitions()` function
- [ ] Implement `calculate_transition_multiplier()` function
- [ ] Implement `calculate_hold_type_entropy()` function
- [ ] Implement `calculate_variability_multiplier()` function
- [ ] Implement core algorithm in `grade_prediction.py`
- [ ] Write unit tests for each component (including complexity multipliers)
- [ ] Integrate with existing codebase
- [ ] Update `analyze_image()` to pass detected holds, wall segments, and wall incline

### Phase 2: Testing

- [ ] Create synthetic test cases (various wall angles and transitions)
- [ ] Create test cases for hold variability scenarios
- [ ] Run integration tests
- [ ] Test wall incline default handling
- [ ] Test complexity multiplier edge cases
- [ ] Collect user feedback on existing routes
- [ ] Calibrate weights and thresholds (including complexity multipliers)

### Phase 3: Validation

- [ ] Compare v1 vs v2 predictions on historical data
- [ ] Analyze accuracy metrics
- [ ] Refine algorithm based on results
- [ ] Document differences and improvements

### Phase 4: Deployment

- [ ] Update production `predict_grade()` function
- [ ] Update API documentation
- [ ] Monitor prediction distribution
- [ ] Collect user feedback on new predictions

### Phase 5: Iteration

- [ ] Analyze user feedback patterns
- [ ] Identify systematic errors
- [ ] Refine weights/thresholds
- [ ] Consider machine learning approach for weight optimization

---

## Future Enhancements

### Short-term (Next Iteration)

1. **Route Type Detection**:
   - Detect traverses (horizontal routes) vs vertical routes
   - Apply different grading logic for each type

2. **Hold Sequence Analysis**:
   - Analyze spatial patterns (e.g., alternating hands)
   - Detect coordination requirements

3. **Advanced Wall Incline Features**:
   - Computer vision-based wall angle detection
   - Interaction multipliers (e.g., slopers on overhang extra penalty)
   - Segment-wise incline analysis for routes that change angle

### Medium-term

1. **Machine Learning Model**:
   - Train supervised model on user feedback
   - Use features from current algorithm as inputs
   - Learn optimal weights automatically

2. **Multi-image Analysis**:
   - Process multiple angles of same route
   - Combine predictions for higher confidence

3. **Climber Attribute Consideration**:
   - Height/reach affects distance difficulty
   - Could provide personalized grades (future user feature)

### Long-term

1. **Video Analysis** (See Phase 2 - separate document):
   - Analyze climbing videos to see actual movement
   - Detect dynamic vs static moves
   - Incorporate movement patterns into grading

2. **Community Consensus**:
   - Weight predictions by user agreement
   - Continuously refine based on aggregate feedback

3. **Style Classification**:
   - Identify route style (powerful, technical, endurance)
   - Provide style-specific difficulty ratings

---

## Summary & Next Steps

### Key Design Decisions

1. **Four-factor weighted model with complexity multipliers**: Balances multiple aspects of difficulty including wall angle, enhanced by transition and variability multipliers
2. **Two-stage scoring approach**: Base score from weighted factors, then multiplicative complexity adjustments
3. **Non-linear relationships**: Reflects real climbing physics (logarithmic hold density, multiplicative complexity)
4. **Normalized metrics**: Handles varying image resolutions
5. **Configurable parameters**: Enables tuning without code changes
6. **Manual wall incline and segment input**: Pragmatic approach for initial implementation
7. **Shannon entropy for variability**: Standard information theory metric for hold type distribution
8. **Spatial transition detection**: Detect wall angle changes based on hold positions along route

### Critical Success Factors

✅ Accurate hold dimension calculation
✅ Proper distance normalization
✅ Appropriate weight calibration (including wall incline weight)
✅ Robust edge case handling (including missing wall incline data)
✅ Sufficient validation data (across different wall angles and complexity levels)
✅ Realistic wall incline multipliers validated by climbing community
✅ Accurate transition detection without false positives
✅ Entropy calculation correctly reflects hold type diversity
✅ Multiplier bounds prevent over-scoring (1.0-1.5x range)
✅ Combined multipliers produce realistic grade increases

### Implementation Priority

**High Priority** (Core Algorithm):

1. Database schema update for wall_incline and wall_segments fields
2. UI update for wall incline input (simple mode)
3. Hold dimension calculation
4. Distance analysis implementation
5. Hold difficulty scoring
6. Wall incline scoring function
7. Score combination logic (4 factors)
8. Hold type entropy calculation
9. Hold variability multiplier

**Medium Priority** (Refinement):

10. UI update for wall segments input (advanced mode)
11. Wall transition detection algorithm
12. Transition multiplier calculation
13. Configuration management (including complexity multiplier config)
14. Detailed feature storage (including complexity analysis)
15. Confidence adjustment
16. Default handling for missing wall incline

**Lower Priority** (Enhancement):

17. A/B testing framework
18. Advanced validation metrics
19. User feedback integration loop
20. Computer vision wall angle detection
21. Automatic transition detection from image analysis

### Recommended Next Action

Begin implementation in **Code mode** with the core algorithm components, starting with:

1. **Base factors**: Hold dimension calculation, distance analysis, hold difficulty scoring
2. **Complexity multipliers**: Entropy calculation for hold variability, transition detection for wall segments

Prioritize hold variability multiplier first (simpler, no additional data requirements) before transition multiplier (requires wall segments input).

---

# PHASE 1.5: PERSONA-BASED PERSONALIZATION (Optional Enhancement)

## Overview

**Status**: Feasibility analysis complete - See detailed analysis in [`persona_personalization_analysis.md`](persona_personalization_analysis.md)

**Purpose**: Enhance the Phase 1 route-based grade prediction with user persona/climbing style preferences to provide personalized difficulty predictions.

**Key Concept**: Different climbers have different strengths. A route's difficulty is subjective based on the climber's specialization. For example:

- A V5 slab route might feel like V4 to a slab specialist (but V6 to a power climber)
- A V8 overhang with wide spacing might feel like V6 to a power climber (but V9 to a slab specialist)

**Positioning**: Implement as Phase 1.5 - after Phase 1 deployment and validation, before Phase 2 (video analysis).

---

## Why Persona Personalization?

### Problem Statement

The current algorithm (Phase 1) predicts an "objective" grade based on route characteristics. However, climbers experience routes differently based on their individual strengths and weaknesses:

- **Slab specialists** find balance-dependent routes easier
- **Power climbers** excel on steep terrain with wide spacing
- **Technical climbers** handle complex sequences better
- **Crimp specialists** thrive on small holds

A single grade doesn't capture these subjective experiences.

### Proposed Solution

Allow users to select their climbing style persona(s) to receive personalized grade predictions that reflect how the route will feel **for them** specifically.

**Example Output**:

```text
Standard Grade: V7
Your Grade (Power Climber): V5
Why? This route's steep overhang and wide spacing match your strengths
```

---

## Technical Approach

### Integration with Existing Algorithm

The persona system **layers on top** of the Phase 1 algorithm without modifying its core structure:

1. **Route Classification**: Analyze detected features to classify route characteristics (slab/overhang, crimp-heavy, wide spacing, etc.)

2. **Persona Matching**: Match user's selected persona(s) to route characteristics

3. **Adjustment Application**: Apply persona-specific multipliers to factor scores:

   ```python
   adjusted_factor_score = base_factor_score × persona_multiplier
   ```

4. **Personalized Prediction**: Recalculate grade with adjusted scores

### 7 Core Personas Defined

See [`persona_personalization_analysis.md`](persona_personalization_analysis.md) for complete details:

1. **🦶 Slab Specialist** - Balance, footwork, friction climbing
2. **💪 Power/Campus Climber** - Explosive strength, dynamic moves, steep terrain
3. **🧠 Technical/Beta Reader** - Route reading, efficient sequences, adaptability
4. **🤏 Crimp Specialist** - Finger strength, small holds, static control
5. **🤸 Flexibility/Mobility Specialist** - High steps, compression, creative positions
6. **⏱️ Endurance Specialist** - Sustained climbing, recovery, longer problems
7. **⚖️ Balanced/All-Arounder** - Well-rounded, no major strengths/weaknesses (default)

### Adjustment Mechanism

Each persona has defined adjustment multipliers for different route types:

```yaml
slab_specialist:
  slab_routes:
    wall_incline: 0.65    # 35% easier
    distance: 0.85        # 15% easier
    hold_density: 0.90    # 10% easier
  overhang_routes:
    wall_incline: 1.35    # 35% harder
    distance: 1.15        # 15% harder
```

**Adjustment Range**: 0.65-1.35 (±35% maximum, typically ±15-20%)

---

## Implementation Scope

### Phase 1.5A: Basic Persona System (8-9 weeks)

**Core Features**:

- [ ] Single persona selection (one of 7 personas)
- [ ] Route characteristic classification
- [ ] Persona adjustment engine
- [ ] Basic UI integration (persona selector)
- [ ] Display both standard and personalized grades
- [ ] Strength level control (light/medium/strong)

**Deliverables**:

- All 7 personas implemented
- Adjustment configuration in `user_config.yaml`
- Updated `predict_grade_v2_personalized()` function
- UI for persona selection
- Testing framework

### Phase 1.5B: Advanced Features (Future)

**Optional Enhancements**:

- [ ] Multi-persona support (primary + secondary with weights)
- [ ] Persona quiz/assessment tool
- [ ] ML-based personalization learning from feedback
- [ ] Persona evolution tracking
- [ ] Community consensus adjustments

---

## Data Model Changes

### Option A: Store in Analysis Model (Recommended Initial)

```python
class Analysis(Base):
    # ... existing fields ...
    user_persona_applied = db.Column(db.JSON, nullable=True)
    # Format:
    # {
    #   "primary": "power_climber",
    #   "secondary": null,
    #   "strength": "medium",
    #   "enabled": true
    # }
```

### Option B: User Profile Model (Future)

```python
class UserProfile(Base):
    """Stores user preferences and persona settings."""
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.String(36), primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('user_sessions.session_id'))
    persona_config = db.Column(db.JSON, nullable=True)
    preferences = db.Column(db.JSON, nullable=True)
```

---

## Key Benefits

✅ **Personalized User Experience**: Climbers see grades that reflect their actual experience

✅ **Non-Destructive**: Doesn't modify core algorithm, can be toggled on/off

✅ **Educational**: Helps climbers understand their strengths and weaknesses

✅ **Motivational**: "This V7 feels like V5 for you!" encourages attempts

✅ **Training Insights**: Identifies weak areas to improve

✅ **No Additional Data Collection**: Works with existing route detection

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Users misjudge their persona | Provide detailed descriptions, quiz tool, examples |
| Adjustments feel inaccurate | Start conservative, calibrate with feedback, allow strength control |
| Adds confusion | Clear UI showing both grades, educational content |
| Calibration without ground truth | Use domain expertise, iterate with user feedback, A/B testing |

---

## Success Metrics

**Adoption**:

- Target: >40% of users select a persona
- Track: Persona selection rates, distribution across personas

**Satisfaction**:

- Target: >3.5/5.0 user satisfaction with personalized grades
- Track: Feedback surveys, agreement rates

**Accuracy**:

- Target: >70% of users report personalized grades "feel accurate"
- Track: Feedback on personalized predictions

**Engagement**:

- Track: Repeat usage, feature retention, grade comparisons

---

## Dependencies

### Prerequisites (Must Complete First)

✅ Phase 1 core algorithm deployed and validated
✅ User feedback system operational
✅ Baseline accuracy ≥60% within ±1 grade
✅ At least 100 analyses in database for testing

### Required Resources

- Development time: 8-9 weeks
- Climbing domain expertise for calibration
- Beta testers representing different personas
- UX design for persona selection interface

---

## Timeline & Phasing

**Recommended Start**: 2-3 months after Phase 1 deployment

**Implementation**:

- Weeks 1-2: Foundation (persona definitions, database)
- Weeks 3-4: Route classification
- Weeks 4-5: Adjustment engine
- Weeks 5-6: UI integration
- Weeks 6-8: Testing and calibration
- Weeks 8-9: Deployment and monitoring

**Rollout Strategy**:

- Internal testing (2 weeks)
- Beta users 10% (2 weeks)
- Expanded 50% (2 weeks)
- Full rollout 100%

---

## Configuration Example

Add to `user_config.yaml`:

```yaml
personas:
  enabled: true
  default: "balanced"
  strength_levels: ["light", "medium", "strong"]
  max_adjustment_magnitude: 0.35
  
  profiles:
    slab_specialist:
      name: "Slab Specialist"
      icon: "🦶"
      description: "Excels at balance, footwork, and friction climbing"
      adjustments:
        slab_routes:
          wall_incline: 0.65
          distance: 0.85
          hold_density: 0.90
        overhang_routes:
          wall_incline: 1.35
          distance: 1.15
    # ... other personas ...
```

---

## Next Steps

1. **Review Analysis**: Read [`persona_personalization_analysis.md`](persona_personalization_analysis.md) for complete design
2. **Stakeholder Approval**: Decide whether to implement Phase 1.5
3. **Timeline Decision**: Determine when to start (post-Phase 1 validation)
4. **Resource Allocation**: Assign development resources
5. **Beta Recruitment**: Identify testers for different personas

---

## Reference Documents

**Detailed Analysis**: [`persona_personalization_analysis.md`](persona_personalization_analysis.md)

- Complete persona definitions with strengths/weaknesses
- Persona correlation matrix
- Detailed adjustment mechanism design
- Multi-persona support architecture
- Data model specifications
- Implementation timeline
- Risk analysis and mitigation
- Testing and calibration strategy

**Integration Point**: This phase enhances [`predict_grade_v2()`](../src/main.py:707) with persona-aware adjustments

---

## Appendix: References

### Related Files

- [`src/main.py`](../src/main.py) - Current implementation, analyze_image function
- [`src/models.py`](../src/models.py) - Database models (Analysis, DetectedHold, HoldType)
- [`src/constants.py`](../src/constants.py) - Hold type definitions
- [`src/cfg/user_config.yaml`](../src/cfg/user_config.yaml) - Configuration file

### Climbing Grade References

- V-Scale ranges from V0 (beginner) to V17 (elite)
- This implementation targets V0-V12 (covers most gym routes)
- Grade progression is non-linear (V5 is not "half" of V10)

### Algorithm Inspirations

- Weighted scoring models from ranking systems
- Logarithmic scaling for diminishing returns (hold count)
- Multi-factor analysis from expert systems
- Normalized metrics from computer vision best practices
- **Shannon entropy** from information theory for measuring distribution uniformity
- **Multiplicative difficulty factors** from game design and psychophysics
- **Spatial clustering analysis** for transition detection
