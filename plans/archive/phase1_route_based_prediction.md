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

Evaluate the technical difficulty of holds based on type and physical size, **including both handholds and footholds as critical difficulty factors**.

**CRITICAL DESIGN PRINCIPLE**: Footholds are as important as handholds for route difficulty. Their absence, size, and availability significantly impact balance, body positioning, and overall climbing technique required.

### Handhold Difficulty Classification

#### Tier 1 - Very Hard (Base Score: 10)

- **Crimps**: Small, narrow holds requiring finger strength
- **Pockets**: Small holes requiring specific finger positioning
- Size consideration: Smaller crimps/pockets (bbox area < threshold) add difficulty

#### Tier 2 - Hard (Base Score: 7)

- **Slopers**: Round holds requiring open-handed grip and body tension
- **Pinches**: Require thumb opposition and pinch strength

#### Tier 3 - Moderate (Base Score: 4)

- **Start-holds**: Typically good holds to begin route
- **Top-out-holds**: Final holds, usually accessible

#### Tier 4 - Easy (Base Score: 1)

- **Jugs**: Large, easy-to-grip holds

### Size Calculation

For each detected hold (both handholds and footholds), calculate physical size:

```text
hold_width = bbox_x2 - bbox_x1
hold_height = bbox_y2 - bbox_y1
hold_area = hold_width * hold_height
```

### Handhold Size-Based Adjustments

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

### Handhold Type Distribution Analysis

Calculate the **proportion** of hard holds vs easy holds:

```text
hard_hold_ratio = (count_crimps + count_pockets + count_slopers) / total_handholds
```

Where `total_handholds` excludes foot-holds, start-holds, and top-out-holds.

### Handhold Difficulty Score Formula

```text
Handhold_Difficulty_Score = Σ(hold_base_score × size_modifier) × (1 + hard_hold_ratio × 0.5)
```

The hard_hold_ratio multiplier creates a non-linear effect: routes with many hard holds are disproportionately harder.

### Handhold Normalization

Normalize by dividing by total handhold count:

```text
Normalized_Handhold_Score = Handhold_Difficulty_Score / total_handholds
```

This creates a per-hold difficulty average ranging approximately 1-13.

---

### Foothold Difficulty Analysis

**CRITICAL IMPORTANCE**: Footholds are a primary difficulty determinant. The availability, size, and spacing of footholds fundamentally affects balance, body positioning, and whether normal climbing technique is even possible.

#### Foothold Difficulty Classification

Footholds are scored based on size and availability:

**Tier 1 - No Footholds Detected (Campusing)**

- **Base Score: 12** (EXTREME difficulty)
- **Condition**: Zero footholds detected on route
- **Impact**: Forces campusing (no feet), drastically increases difficulty for normal climbers
- **Grade Impact**: Typically adds +2 to +4 V-grades for non-elite climbers
- **Rationale**: Most climbers rely heavily on feet for support and balance

**Tier 2 - Very Small Footholds (Base Score: 9)**

- **Size**: area < 800px²
- **Difficulty**: Requires precise footwork, toe precision, balance
- **Common on**: Technical face climbs, advanced routes

**Tier 3 - Small Footholds (Base Score: 6)**

- **Size**: area 800-1500px²
- **Difficulty**: Moderate footwork precision required
- **Common on**: Intermediate to advanced routes

**Tier 4 - Medium Footholds (Base Score: 3)**

- **Size**: area 1500-3000px²
- **Difficulty**: Standard footwork, good support
- **Common on**: Most gym routes, beginner to intermediate

**Tier 5 - Large Footholds (Base Score: 1)**

- **Size**: area > 3000px²
- **Difficulty**: Easy to stand on, excellent support
- **Common on**: Beginner routes, jugs for feet

#### Foothold Size-Based Scoring

```python
def calculate_foothold_difficulty(footholds: list) -> float:
    """
    Calculate foothold difficulty score.
    
    CRITICAL: No footholds = campusing = extreme difficulty
    """
    if len(footholds) == 0:
        # NO FOOTHOLDS = CAMPUSING
        return 12.0  # Maximum difficulty
    
    total_score = 0
    for fh in footholds:
        area = fh.area
        if area < 800:
            score = 9
        elif area < 1500:
            score = 6
        elif area < 3000:
            score = 3
        else:
            score = 1
        total_score += score
    
    # Normalize by foothold count
    avg_foothold_difficulty = total_score / len(footholds)
    
    # Apply foothold scarcity penalty
    # Few footholds = limited balance options = harder
    if len(footholds) <= 2:
        scarcity_multiplier = 1.5  # Very few footholds
    elif len(footholds) <= 4:
        scarcity_multiplier = 1.25  # Few footholds
    elif len(footholds) <= 6:
        scarcity_multiplier = 1.1  # Limited footholds
    else:
        scarcity_multiplier = 1.0  # Adequate footholds
    
    return avg_foothold_difficulty * scarcity_multiplier
```

**Foothold Scarcity Impact:**

- 0 footholds → Score: 12.0 (campusing, extreme)
- 1-2 footholds → 1.5x multiplier (very limited balance options)
- 3-4 footholds → 1.25x multiplier (limited options, technical sequences)
- 5-6 footholds → 1.1x multiplier (some options, still constrained)
- 7+ footholds → 1.0x multiplier (adequate options for balance)

**Design Rationale:**

Footholds enable:

- Weight transfer to legs (more efficient than arms)
- Balance and stability during moves
- Rest positions between difficult moves
- Hip positioning for reach optimization
- Dynamic movement generation from legs

Without footholds or with very small/sparse footholds, climbers must:

- Support more weight on arms (rapid fatigue)
- Maintain constant core tension
- Execute precise, powerful moves without feet
- Manage significantly higher physical demands

---

### Wall-Angle-Dependent Foothold Weighting

**CRITICAL DESIGN PRINCIPLE**: Foothold importance varies dramatically by wall angle.

#### Climbing Biomechanics by Wall Angle

**Slabs (70°-89°):**

- **Foothold importance: HIGHEST (60-70% of difficulty)**
- Climbers push with legs, weight over feet
- Balance and footwork are primary skills
- Small footholds drastically increase difficulty
- Handholds often used for balance, not pulling

**Vertical (90°):**

- **Foothold importance: HIGH (40-50% of difficulty)**
- Balanced load between hands and feet
- Good footwork reduces arm strain
- Footholds essential for efficient climbing
- Standard baseline for difficulty assessment

**Slight Overhang (91°-105°):**

- **Foothold importance: MODERATE (35-45% of difficulty)**
- Upper body load increases
- Footholds still important for body positioning
- Core tension becomes more critical
- Foot placement affects hip positioning

**Moderate Overhang (106°-120°):**

- **Foothold importance: MODERATE-LOW (25-35% of difficulty)**
- Upper body dominant
- Footholds used for body positioning, not weight support
- Core tension and pulling strength primary
- Large footholds help, small footholds less impactful

**Steep Overhang (121°-135°):**

- **Foothold importance: LOW (20-30% of difficulty)**
- Upper body and core dominant
- Footholds mainly for body positioning
- Route difficulty driven by handholds and power
- Campus-style climbing more common

#### Foothold Weight Function

Define wall-angle-dependent weights for combining handhold and foothold scores:

```python
def get_foothold_weight(wall_angle_category: str) -> tuple[float, float]:
    """
    Return (handhold_weight, foothold_weight) for wall angle.
    
    Weights sum to 1.0, reflecting relative importance.
    
    Returns:
        tuple: (handhold_weight, foothold_weight)
    """
    weights = {
        'slab': (0.35, 0.65),              # 65% foothold importance
        'vertical': (0.55, 0.45),          # 45% foothold importance
        'slight_overhang': (0.60, 0.40),   # 40% foothold importance
        'moderate_overhang': (0.70, 0.30), # 30% foothold importance
        'steep_overhang': (0.75, 0.25)     # 25% foothold importance
    }
    
    return weights.get(wall_angle_category, (0.55, 0.45))  # Default: vertical
```

**Weight Rationale:**

| Wall Angle | Handhold % | Foothold % | Climbing Style |
| :--------: | :--------: | :--------: | :------------: |
| Slab | 35% | **65%** | Footwork-dominant, balance-critical |
| Vertical | 55% | **45%** | Balanced, standard climbing |
| Slight Overhang | 60% | **40%** | Upper body increases |
| Moderate Overhang | 70% | **30%** | Power-focused, feet assist |
| Steep Overhang | 75% | **25%** | Upper body dominant, feet position |

---

### Combined Hold Difficulty Score

Integrate handhold and foothold difficulty with wall-angle-dependent weighting:

```python
def calculate_combined_hold_difficulty(
    handholds: list,
    footholds: list,
    wall_angle_category: str
) -> float:
    """
    Calculate combined hold difficulty considering both hands and feet.
    
    Uses wall-angle-dependent weighting to reflect climbing biomechanics.
    """
    # Calculate individual scores
    handhold_score = calculate_handhold_difficulty(handholds)  # 1-13 range
    foothold_score = calculate_foothold_difficulty(footholds)  # 1-12 range
    
    # Get wall-angle-dependent weights
    hand_weight, foot_weight = get_foothold_weight(wall_angle_category)
    
    # Combine with weights
    combined_score = (handhold_score * hand_weight) + (foothold_score * foot_weight)
    
    # Combined score range: approximately 1-13
    return combined_score
```

**Example Calculations:**

**Example 1: V5 Slab with Small Footholds**

- Handholds: 8 medium jugs/crimps → Handhold score: 5.5
- Footholds: 5 small footholds (1200px² avg) → Foothold score: 7.5
- Wall angle: Slab
- Weights: 35% hands, 65% feet
- Combined: (5.5 × 0.35) + (7.5 × 0.65) = 1.93 + 4.88 = **6.81**
- **Impact**: Small footholds dominate difficulty on slab

**Example 2: V7 Vertical Route with No Footholds (Campus)**

- Handholds: 6 crimps/pockets → Handhold score: 9.0
- Footholds: 0 detected → Foothold score: 12.0 (campusing)
- Wall angle: Vertical
- Weights: 55% hands, 45% feet
- Combined: (9.0 × 0.55) + (12.0 × 0.45) = 4.95 + 5.40 = **10.35**
- **Impact**: Campusing adds massive difficulty

**Example 3: V6 Overhang with Large Footholds**

- Handholds: 8 slopers/pinches → Handhold score: 8.0
- Footholds: 8 large footholds (3500px² avg) → Foothold score: 1.0
- Wall angle: Moderate overhang
- Weights: 70% hands, 30% feet
- Combined: (8.0 × 0.70) + (1.0 × 0.30) = 5.60 + 0.30 = **5.90**
- **Impact**: Good footholds don't help much on steep terrain

**Example 4: V4 Vertical Route with Good Footholds**

- Handholds: 12 mixed holds → Handhold score: 6.0
- Footholds: 10 medium footholds (2000px² avg) → Foothold score: 3.0
- Wall angle: Vertical
- Weights: 55% hands, 45% feet
- Combined: (6.0 × 0.55) + (3.0 × 0.45) = 3.30 + 1.35 = **4.65**
- **Impact**: Good footholds reduce overall difficulty

### Summary of Factor 1 Updates

**Key Changes:**

1. ✅ **Footholds elevated to primary difficulty factor** (not tier 3 afterthought)
2. ✅ **Campusing penalty implemented** (no footholds = score 12.0)
3. ✅ **Foothold size scoring defined** (very small to large)
4. ✅ **Foothold scarcity multipliers added** (few footholds = harder)
5. ✅ **Wall-angle-dependent weighting** (65% on slabs, 25% on overhangs)
6. ✅ **Combined scoring formula** integrates hands and feet appropriately

**Result**: Factor 1 now properly reflects that footholds are as important as handholds, with importance varying by wall angle per climbing biomechanics.

---

## Factor 2: Hold Count Analysis

### Objective

Assess route difficulty based on the number of available holds, **considering both handhold and foothold availability with wall-angle-dependent weighting**.

**CRITICAL DESIGN PRINCIPLE**: Hold count must account for both handholds and footholds. Sparse footholds severely limit balance options and movement choices, increasing difficulty.

### Core Principle - Handhold Count

**Inverse relationship with difficulty, but non-linear:**

- Very few holds (3-5): Extremely difficult (V8-V12) - requires powerful, precise moves
- Few holds (6-8): Hard (V5-V7) - limited options, technical sequences
- Moderate holds (9-12): Intermediate (V3-V5) - some options available
- Many holds (13-16): Moderate-easy (V1-V3) - multiple path options
- Very many holds (17+): Easy (V0-V2) - abundant options, likely easier sequences

### Core Principle - Foothold Count

**CRITICAL: Foothold availability affects balance options and movement freedom:**

- **No footholds (0)**: EXTREME difficulty - campusing required, very limited to elite climbers
- **Very few footholds (1-2)**: Severe constraint - very limited balance options, technical sequences
- **Few footholds (3-4)**: Significant constraint - limited balance options, precise foot placement required
- **Moderate footholds (5-7)**: Some constraint - adequate but not abundant options
- **Many footholds (8+)**: Adequate options - multiple balance positions available

**Key Insight**: Unlike handholds where fewer can indicate powerful problems, fewer footholds almost always increases difficulty due to balance limitations.

### Exceptions & Context

**Hold count must be interpreted with hold types:**

- 5 jugs is easier than 15 crimps (for handholds)
- 3 large footholds easier than 8 tiny footholds
- This interaction is captured by combining with Factor 1

### Handhold Density Score Formula

Use a **logarithmic decay function** to model the non-linear relationship:

```text
Handhold_Density_Score = 12 - (log₂(total_handholds) × 2.5)
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

### Foothold Density Score Formula

Model foothold scarcity with emphasis on the severe difficulty increase from missing or very sparse footholds:

```python
def calculate_foothold_density_score(foothold_count: int) -> float:
    """
    Calculate difficulty score based on foothold count.
    
    CRITICAL: No footholds = campusing = extreme difficulty
    Sparse footholds = limited balance options = high difficulty
    
    Returns score in range [0, 12]
    """
    if foothold_count == 0:
        # NO FOOTHOLDS = CAMPUSING
        return 12.0  # Maximum difficulty
    elif foothold_count == 1:
        return 11.0  # Single foothold, extremely limited
    elif foothold_count == 2:
        return 9.5   # Two footholds, very limited balance
    elif foothold_count <= 4:
        return 8.0   # Few footholds, limited options
    elif foothold_count <= 6:
        return 6.0   # Moderate footholds, some constraint
    elif foothold_count <= 8:
        return 4.0   # Adequate footholds
    elif foothold_count <= 12:
        return 2.5   # Many footholds, good options
    else:
        return 1.0   # Abundant footholds, many options
```

**Foothold Density Mapping:**

| Foothold Count | Density Score | Difficulty Impact | Climbing Constraint |
| :------------: | :-----------: | :---------------: | :-----------------: |
| 0 | 12.0 | EXTREME | Campusing, elite only |
| 1 | 11.0 | Very High | Single balance point |
| 2 | 9.5 | High | Very limited movement |
| 3-4 | 8.0 | Moderate-High | Limited options, technical |
| 5-6 | 6.0 | Moderate | Some constraint |
| 7-8 | 4.0 | Low-Moderate | Adequate options |
| 9-12 | 2.5 | Low | Good options |
| 13+ | 1.0 | Very Low | Abundant options |

**Design Rationale:**

Foothold scarcity has a **steeper penalty curve** than handhold scarcity because:

1. **Balance Dependency**: Climbers need feet for balance more than hands
2. **Weight Distribution**: Legs can support more weight efficiently than arms
3. **Rest Positions**: Feet enable rest positions to recover from hard moves
4. **Movement Options**: More footholds = more beta options = easier route reading
5. **Fatigue Management**: Without footholds, arms fatigue rapidly

---

### Wall-Angle-Dependent Hold Density Weighting

**CRITICAL DESIGN PRINCIPLE**: Foothold density importance varies by wall angle, matching the biomechanics principle from Factor 1.

```python
def calculate_combined_hold_density(
    handhold_count: int,
    foothold_count: int,
    wall_angle_category: str
) -> float:
    """
    Calculate combined hold density score with wall-angle-dependent weighting.
    
    Uses same weighting as Factor 1 for consistency.
    """
    # Calculate individual density scores
    handhold_density_score = 12 - (math.log2(max(handhold_count, 1)) * 2.5)
    handhold_density_score = max(0, min(12, handhold_density_score))
    
    foothold_density_score = calculate_foothold_density_score(foothold_count)
    
    # Get wall-angle-dependent weights (same as Factor 1)
    hand_weight, foot_weight = get_foothold_weight(wall_angle_category)
    
    # Combine with weights
    combined_density_score = (
        handhold_density_score * hand_weight +
        foothold_density_score * foot_weight
    )
    
    return combined_density_score
```

**Weight Application:**

| Wall Angle | Handhold Weight | Foothold Weight | Rationale |
| :--------: | :-------------: | :-------------: | :-------: |
| Slab | 35% | **65%** | Foothold availability critical for balance |
| Vertical | 55% | **45%** | Balanced importance |
| Slight Overhang | 60% | **40%** | Handholds more important |
| Moderate Overhang | 70% | **30%** | Upper body dominant |
| Steep Overhang | 75% | **25%** | Handholds far more critical |

---

### Example Calculations

**Example 1: Slab with Sparse Footholds**

- Handholds: 10 holds → Handhold density: 3.7
- Footholds: 3 holds → Foothold density: 8.0
- Wall angle: Slab (35% hand, 65% foot)
- Combined: (3.7 × 0.35) + (8.0 × 0.65) = 1.30 + 5.20 = **6.50**
- **Impact**: Sparse footholds dominate difficulty on slab

**Example 2: Vertical Route with No Footholds (Campus Problem)**

- Handholds: 6 holds → Handhold density: 6.5
- Footholds: 0 holds → Foothold density: 12.0
- Wall angle: Vertical (55% hand, 45% foot)
- Combined: (6.5 × 0.55) + (12.0 × 0.45) = 3.58 + 5.40 = **8.98**
- **Impact**: Campusing adds massive difficulty through density score

**Example 3: Steep Overhang with Few Handholds but Good Footholds**

- Handholds: 4 holds → Handhold density: 7.5
- Footholds: 8 holds → Foothold density: 4.0
- Wall angle: Steep Overhang (75% hand, 25% foot)
- Combined: (7.5 × 0.75) + (4.0 × 0.25) = 5.63 + 1.00 = **6.63**
- **Impact**: Handhold scarcity dominates on overhang, footholds help less

**Example 4: Beginner Vertical Route with Many Holds**

- Handholds: 18 holds → Handhold density: 1.4
- Footholds: 14 holds → Foothold density: 1.0
- Wall angle: Vertical (55% hand, 45% foot)
- Combined: (1.4 × 0.55) + (1.0 × 0.45) = 0.77 + 0.45 = **1.22**
- **Impact**: Abundant holds of both types = easy route

**Example 5: Technical Slab with Adequate Holds**

- Handholds: 12 holds → Handhold density: 3.4
- Footholds: 10 holds → Foothold density: 2.5
- Wall angle: Slab (35% hand, 65% foot)
- Combined: (3.4 × 0.35) + (2.5 × 0.65) = 1.19 + 1.63 = **2.82**
- **Impact**: Good foothold availability reduces difficulty on slab

---

### Summary of Factor 2 Updates

**Key Changes:**

1. ✅ **Foothold density scoring added** with steeper penalty curve
2. ✅ **Campusing penalty reinforced** (0 footholds = 12.0 score)
3. ✅ **Foothold scarcity levels defined** (1-2 = extreme, 3-4 = high, etc.)
4. ✅ **Wall-angle-dependent weighting** applied to density (consistent with Factor 1)
5. ✅ **Combined density formula** integrates handhold and foothold availability
6. ✅ **Example calculations** demonstrate real-world scenarios

**Result**: Factor 2 now properly accounts for foothold availability as a critical difficulty factor, with appropriate wall-angle-dependent importance.

---

## Factor 3: Hold Distance Analysis

### Objective

Measure route difficulty based on hold spacing for **both handholds and footholds**, capturing reach difficulty, high-step challenges, and crux moves.

**CRITICAL DESIGN PRINCIPLE**: Foothold spacing matters as much as handhold spacing. Large vertical foothold gaps force high-steps, requiring flexibility and balance. Sparse footholds with wide spacing dramatically increase difficulty.

### Spatial Analysis Required

#### Step 1: Separate and Sort Holds

Separate handholds and footholds, then sort each by vertical position:

```python
def separate_and_sort_holds(all_holds: list) -> tuple[list, list]:
    """
    Separate holds into handholds and footholds, sort vertically.
    
    Returns:
        tuple: (sorted_handholds, sorted_footholds)
    """
    handholds = [h for h in all_holds if h.hold_type not in ['foot-hold']]
    footholds = [h for h in all_holds if h.hold_type == 'foot-hold']
    
    # Sort by y-coordinate (bottom to top)
    handholds_sorted = sorted(handholds, key=lambda h: h.bbox_y1)
    footholds_sorted = sorted(footholds, key=lambda h: h.bbox_y1)
    
    return handholds_sorted, footholds_sorted
```

#### Step 2: Calculate Sequential Distances

For each consecutive pair of holds (separately for hands and feet), calculate Euclidean distance:

```python
def calculate_hold_distances(holds: list) -> dict:
    """
    Calculate distance metrics for a list of holds.
    
    Returns:
        dict with avg_distance, max_distance, distance_variance
    """
    if len(holds) < 2:
        return {
            'avg_distance': 0,
            'max_distance': 0,
            'distance_variance': 0,
            'distances': []
        }
    
    distances = []
    for i in range(len(holds) - 1):
        h1, h2 = holds[i], holds[i + 1]
        
        # Calculate center points
        x1 = (h1.bbox_x1 + h1.bbox_x2) / 2
        y1 = (h1.bbox_y1 + h1.bbox_y2) / 2
        x2 = (h2.bbox_x1 + h2.bbox_x2) / 2
        y2 = (h2.bbox_y1 + h2.bbox_y2) / 2
        
        # Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(distance)
    
    return {
        'avg_distance': statistics.mean(distances),
        'max_distance': max(distances),
        'distance_variance': statistics.stdev(distances) if len(distances) > 1 else 0,
        'distances': distances
    }
```

---

### Handhold Distance Analysis

#### Distance Interpretation (pixels at standard image resolution)

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

#### Normalization for Image Resolution

Since images may vary in resolution, normalize distances:

```python
def normalize_distance(raw_distance: float, image_height: float) -> float:
    """Normalize distance by image height for resolution independence."""
    return raw_distance / image_height
```

Use normalized ratios:

- Close: < 0.15 (15% of image height)
- Moderate: 0.15-0.30
- Wide: 0.30-0.50
- Very wide: > 0.50

#### Handhold Distance Score Formula

```python
def calculate_handhold_distance_score(
    handhold_metrics: dict,
    image_height: float
) -> float:
    """
    Calculate difficulty score from handhold spacing.
    
    Returns score in range [0, 12]
    """
    if len(handhold_metrics['distances']) == 0:
        return 0
    
    # Normalize distances
    normalized_avg = handhold_metrics['avg_distance'] / image_height
    normalized_max = handhold_metrics['max_distance'] / image_height
    
    # Average distance component (0-9 range)
    avg_component = (normalized_avg / 0.15) * 3
    avg_component = min(avg_component, 9)
    
    # Maximum distance component (crux bonus: 0-3)
    if normalized_max < 0.18:  # ~200px at 1080p
        max_component = 0
    elif normalized_max < 0.37:  # ~400px at 1080p
        max_component = 1
    elif normalized_max < 0.55:  # ~600px at 1080p
        max_component = 2
    else:
        max_component = 3
    
    total_score = avg_component + max_component
    return min(total_score, 12)
```

---

### Foothold Distance Analysis

**CRITICAL IMPORTANCE**: Foothold spacing determines high-step difficulty and balance transitions.

#### Foothold Spacing Biomechanics

**Vertical Foothold Gaps:**

Large vertical gaps between footholds force high-steps:

- Requires hip flexibility
- Demands precise balance during transition
- Increases core tension requirements
- Can be as limiting as wide handhold spacing

**Horizontal Foothold Gaps:**

Wide horizontal spacing between footholds:

- Forces wide stances or single-foot balance
- Limits hip positioning options
- Affects reach optimization
- Increases balance difficulty

#### Foothold Distance Interpretation

**Critical Distinction**: Foothold analysis focuses more on **vertical spacing** (high-steps) since climbers ascend routes:

**Average Vertical Spacing:**

- Close spacing (< 120px / < 0.11 normalized): Easy stepping, natural stride → Low difficulty
- Moderate spacing (120-200px / 0.11-0.18): Standard steps → Moderate difficulty
- Wide spacing (200-300px / 0.18-0.28): High-steps required → High difficulty
- Very wide spacing (> 300px / > 0.28): Extreme high-steps, flexibility critical → Very high difficulty

**Impact on Climbing:**

- Close foothold spacing: Climber can choose optimal foot positions
- Moderate spacing: Some constraint, but manageable
- Wide spacing: Forces specific high-step moves, requires flexibility
- Very wide spacing: May require dynamic foot placements or skipping footholds

#### Foothold Distance Score Formula

```python
def calculate_foothold_distance_score(
    foothold_metrics: dict,
    image_height: float
) -> float:
    """
    Calculate difficulty score from foothold spacing.
    
    Emphasis on vertical spacing (high-steps).
    No footholds = campusing = maximum score.
    
    Returns score in range [0, 12]
    """
    if len(foothold_metrics['distances']) == 0:
        # NO FOOTHOLDS = CAMPUSING
        return 12.0
    
    # Normalize distances
    normalized_avg = foothold_metrics['avg_distance'] / image_height
    normalized_max = foothold_metrics['max_distance'] / image_height
    
    # Average spacing component (0-9 range)
    # Steeper curve than handholds - foothold spacing more impactful
    if normalized_avg < 0.11:  # < 120px
        avg_component = 1.0
    elif normalized_avg < 0.18:  # 120-200px
        avg_component = 3.0
    elif normalized_avg < 0.28:  # 200-300px
        avg_component = 6.0
    else:  # > 300px
        avg_component = 9.0
    
    # Maximum spacing component (high-step crux: 0-3)
    if normalized_max < 0.18:  # < 200px
        max_component = 0
    elif normalized_max < 0.28:  # 200-300px
        max_component = 1
    elif normalized_max < 0.37:  # 300-400px
        max_component = 2
    else:  # > 400px
        max_component = 3
    
    total_score = avg_component + max_component
    return min(total_score, 12)
```

**Design Rationale:**

Foothold distance scoring uses a **steeper curve** than handholds because:

1. **High-steps are highly constraining** - cannot be avoided like handhold reaches
2. **Flexibility limitations** - many climbers struggle with extreme high-steps
3. **Balance difficulty** - longer transitions between feet = more instability
4. **Compound effect** - wide foothold spacing + sparse footholds = extreme difficulty

---

### Wall-Angle-Dependent Distance Weighting

**CRITICAL DESIGN PRINCIPLE**: Foothold spacing importance varies by wall angle.

```python
def calculate_combined_distance_score(
    handhold_distance_score: float,
    foothold_distance_score: float,
    wall_angle_category: str
) -> float:
    """
    Calculate combined distance score with wall-angle-dependent weighting.
    
    Uses same weighting as Factors 1 and 2 for consistency.
    """
    # Get wall-angle-dependent weights
    hand_weight, foot_weight = get_foothold_weight(wall_angle_category)
    
    # Combine with weights
    combined_score = (
        handhold_distance_score * hand_weight +
        foothold_distance_score * foot_weight
    )
    
    return combined_score
```

**Weight Application:**

| Wall Angle | Handhold Weight | Foothold Weight | Distance Impact |
| :--------: | :-------------: | :-------------: | :-------------: |
| Slab | 35% | **65%** | High-steps dominate difficulty |
| Vertical | 55% | **45%** | Balanced reach importance |
| Slight Overhang | 60% | **40%** | Hand reaches more critical |
| Moderate Overhang | 70% | **30%** | Power moves dominate |
| Steep Overhang | 75% | **25%** | Dynamic hand moves critical |

---

### Example Calculations

**Example 1: Slab with Large High-Steps**

- Handhold avg: 200px (norm: 0.18) → Score: 3.6
- Handhold max: 350px (norm: 0.32) → Crux bonus: 1
- Handhold total: 4.6
- Foothold avg: 280px (norm: 0.26) → Score: 6.0
- Foothold max: 380px (norm: 0.35) → Crux bonus: 2
- Foothold total: 8.0
- Wall angle: Slab (35% hand, 65% foot)
- Combined: (4.6 × 0.35) + (8.0 × 0.65) = 1.61 + 5.20 = **6.81**
- **Impact**: High-steps dominate difficulty on slab

**Example 2: Vertical Campus Problem (No Footholds)**

- Handhold avg: 280px (norm: 0.26) → Score: 5.2
- Handhold max: 450px (norm: 0.42) → Crux bonus: 2
- Handhold total: 7.2
- Foothold: None → Score: 12.0
- Wall angle: Vertical (55% hand, 45% foot)
- Combined: (7.2 × 0.55) + (12.0 × 0.45) = 3.96 + 5.40 = **9.36**
- **Impact**: Campusing adds extreme difficulty through distance factor

**Example 3: Overhang with Wide Hand Reaches, Close Footholds**

- Handhold avg: 420px (norm: 0.39) → Score: 7.8
- Handhold max: 580px (norm: 0.54) → Crux bonus: 2
- Handhold total: 9.8
- Foothold avg: 150px (norm: 0.14) → Score: 1.0
- Foothold max: 220px (norm: 0.20) → Crux bonus: 0
- Foothold total: 1.0
- Wall angle: Moderate Overhang (70% hand, 30% foot)
- Combined: (9.8 × 0.70) + (1.0 × 0.30) = 6.86 + 0.30 = **7.16**
- **Impact**: Wide handhold spacing dominates on overhang

**Example 4: Beginner Vertical Route with Close Holds**

- Handhold avg: 180px (norm: 0.17) → Score: 3.4
- Handhold max: 250px (norm: 0.23) → Crux bonus: 1
- Handhold total: 4.4
- Foothold avg: 160px (norm: 0.15) → Score: 3.0
- Foothold max: 200px (norm: 0.18) → Crux bonus: 0
- Foothold total: 3.0
- Wall angle: Vertical (55% hand, 45% foot)
- Combined: (4.4 × 0.55) + (3.0 × 0.45) = 2.42 + 1.35 = **3.77**
- **Impact**: Close spacing for both hands and feet = easier route

---

### Summary of Factor 3 Updates

**Key Changes:**

1. ✅ **Foothold distance analysis added** with focus on vertical spacing (high-steps)
2. ✅ **Separate distance calculations** for handholds and footholds
3. ✅ **Campusing penalty reinforced** (no footholds = 12.0 distance score)
4. ✅ **High-step difficulty emphasized** with steeper scoring curve for footholds
5. ✅ **Wall-angle-dependent weighting** applied (consistent with Factors 1 & 2)
6. ✅ **Combined distance formula** integrates handhold reaches and foothold spacing
7. ✅ **Example calculations** demonstrate realistic scenarios

**Result**: Factor 3 now properly accounts for foothold spacing as a critical difficulty factor, especially for high-step moves and balance transitions, with appropriate wall-angle-dependent importance.

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
| :------: | :---------: | :---------: | :--------------: |
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
| --------- | ------------ | ----------------- | ------- |
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
    Combined_Hold_Difficulty_Score × 0.35 +
    Combined_Hold_Density_Score × 0.25 +
    Combined_Distance_Score × 0.20 +
    Wall_Incline_Score × 0.20
)
```

**CRITICAL UPDATE**: All "Combined" scores now integrate both handhold and foothold components with wall-angle-dependent weighting as defined in Factors 1, 2, and 3.

**Rationale for weights:**

- **35% Combined Hold Difficulty**: Primary determinant - includes both handhold quality and foothold availability/size (weighted by wall angle)
- **25% Combined Hold Density**: Significant impact - accounts for both handhold and foothold availability (weighted by wall angle)
- **20% Combined Distance**: Important for distinguishing grades - includes both handhold reaches and foothold spacing/high-steps (weighted by wall angle)
- **20% Wall Incline**: Critical modifier - overhang vs slab can change grade by 2-3 levels

**Stage 2: Apply Complexity Multipliers**

Apply multiplicative factors for route complexity:

```text
Transition_Multiplier = calculate_transition_multiplier(wall_transitions)  # 1.0-1.5x
Variability_Multiplier = calculate_variability_multiplier(hold_entropy)    # 1.0-1.5x

Final_Score = Base_Score × Transition_Multiplier × Variability_Multiplier
```

**Multiplier Impact Examples:**

| Base Score | Transitions       | Variability     | Final Score    | Impact      |
|:----------:|:-----------:      |:-----------:    |:-----------:   |:------:     |
| 5.0 (V5)   | None (1.0x)       | Low (1.0x)      | 5.0            | No change   |
| 5.0 (V5)   | 2 moderate (1.3x) | Low (1.0x)      | 6.5 (V6-V7)    | +1-2 grades |
| 5.0 (V5)   | None (1.0x)       | High (1.4x)     | 7.0 (V7)       | +2 grades   |
| 5.0 (V5)   | 2 major (1.5x)    | High (1.4x)     | 10.5 (V10-V11) | +5-6 grades |
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
│  - SEPARATE handholds and footholds (NOT discard)    │
│  - Calculate dimensions for both                      │
│  - Parse wall incline/segments                        │
│  - Calculate hold type entropy                        │
└───────────────┬───────────────────────────────────────┘
                │
                ├────────────────┬──────────────────┬──────────────────┬──────────────────┐
                ▼                ▼                  ▼                  ▼                  │
        ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐              │
        │Factor 1:   │   │Factor 2:   │   │Factor 3:   │   │Factor 4:   │              │
        │Combined    │   │Combined    │   │Combined    │   │Wall Incline│              │
        │Hold Diff.  │   │Hold Density│   │Distance    │   │Analysis    │              │
        │            │   │            │   │            │   │            │              │
        │ Handhold ──┼─┐ │ Handhold ──┼─┐ │ Handhold ──┼─┐ │            │              │
        │ Difficulty │ │ │ Density    │ │ │ Distance   │ │ │            │              │
        │            │ │ │            │ │ │            │ │ │            │              │
        │ +          │ │ │ +          │ │ │ +          │ │ │            │              │
        │            │ │ │            │ │ │            │ │ │            │              │
        │ Foothold ──┼─┤ │ Foothold ──┼─┤ │ Foothold ──┼─┤ │            │              │
        │ Difficulty │ │ │ Density    │ │ │ Distance   │ │ │            │              │
        │            │ │ │            │ │ │            │ │ │            │              │
        │ Wall-angle │ │ │ Wall-angle │ │ │ Wall-angle │ │ │            │              │
        │ weighted   │ │ │ weighted   │ │ │ weighted   │ │ │            │              │
        └─────┬──────┘ │ └─────┬──────┘ │ └─────┬──────┘ │ └─────┬──────┘              │
              └────────┘       └────────┘       └────────┘       │                     │
                  │                 │                 │           │                     │
                  └─────────────────┴─────────────────┴───────────┴─────────────────────┘
                                              │
                                              ▼
                                  ┌──────────────────────┐
                                  │ Weighted Combination │
                                  │ Base Score = Σ(Si×Wi)│
                                  │ All factors include  │
                                  │ foothold integration │
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
    
    CRITICAL: Fully integrates foothold analysis throughout all factors.
    
    Args:
        features: Dictionary with hold counts and types
        detected_holds: List of DetectedHold objects with bbox coordinates
        wall_segments: Optional list of wall segments with angles (for transitions)
                      If None, assumes single wall angle
        wall_incline: Default wall angle category if wall_segments not provided
    
    Returns:
        tuple: (predicted_grade, confidence_score, score_breakdown)
    """
    # Preprocessing - SEPARATE handholds and footholds (DO NOT DISCARD footholds)
    handholds, footholds = separate_holds(detected_holds)
    
    confidence_avg = calculate_average_confidence(detected_holds)
    hold_type_counts = count_hold_types(handholds)
    
    # Get wall angle category for weighting
    wall_angle_cat = get_wall_angle_category(wall_segments or wall_incline)
    
    # Stage 1: Calculate Base Score from 4 Factors (with foothold integration)
    
    # Factor 1: Combined Hold Type & Size Analysis
    # Includes both handhold difficulty and foothold difficulty
    # with wall-angle-dependent weighting
    handhold_difficulty = analyze_handhold_difficulty(handholds)
    foothold_difficulty = analyze_foothold_difficulty(footholds)
    combined_hold_difficulty_score = combine_with_wall_weights(
        handhold_difficulty,
        foothold_difficulty,
        wall_angle_cat
    )
    
    # Factor 2: Combined Hold Count Analysis
    # Includes both handhold density and foothold density
    # with wall-angle-dependent weighting
    handhold_density = analyze_handhold_density(len(handholds))
    foothold_density = analyze_foothold_density(len(footholds))
    combined_hold_density_score = combine_with_wall_weights(
        handhold_density,
        foothold_density,
        wall_angle_cat
    )
    
    # Factor 3: Combined Distance Analysis
    # Includes both handhold distances and foothold distances (high-steps)
    # with wall-angle-dependent weighting
    handhold_distance = analyze_handhold_distances(handholds)
    foothold_distance = analyze_foothold_distances(footholds)
    combined_distance_score = combine_with_wall_weights(
        handhold_distance,
        foothold_distance,
        wall_angle_cat
    )
    
    # Factor 4: Wall Incline Analysis (unchanged)
    wall_incline_score = analyze_wall_incline(wall_segments or wall_incline)
    
    # Combine scores into base score
    base_score = (
        combined_hold_difficulty_score * 0.35 +
        combined_hold_density_score * 0.25 +
        combined_distance_score * 0.20 +
        wall_incline_score * 0.20
    )
    
    # Stage 2: Calculate Complexity Multipliers
    
    # Multiplier 1: Wall Angle Transitions
    if wall_segments and len(wall_segments) > 1:
        # Use all holds (handholds + footholds) for transition detection
        all_holds_sorted = sort_holds_vertically(handholds + footholds)
        transition_data = detect_wall_transitions(all_holds_sorted, wall_segments)
        transition_multiplier = calculate_transition_multiplier(transition_data)
    else:
        transition_data = {'transition_count': 0}
        transition_multiplier = 1.0
    
    # Multiplier 2: Hold Type Variability (handholds only)
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
            'combined_hold_difficulty': combined_hold_difficulty_score,
            'handhold_difficulty': handhold_difficulty,
            'foothold_difficulty': foothold_difficulty,
            'combined_hold_density': combined_hold_density_score,
            'handhold_density': handhold_density,
            'foothold_density': foothold_density,
            'combined_distance': combined_distance_score,
            'handhold_distance': handhold_distance,
            'foothold_distance': foothold_distance,
            'wall_incline': wall_incline_score
        },
        'hold_counts': {
            'handholds': len(handholds),
            'footholds': len(footholds),
            'total': len(detected_holds)
        },
        'wall_angle_weights': get_foothold_weight(wall_angle_cat),
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

**Key Pseudocode Updates:**

1. ✅ **`separate_holds()`** replaces filtering - footholds are preserved
2. ✅ **All three main factors** now calculate separate handhold and foothold components
3. ✅ **`combine_with_wall_weights()`** applies wall-angle-dependent weighting consistently
4. ✅ **Foothold metrics tracked** separately in score_breakdown
5. ✅ **Wall angle weights** included in output for transparency
6. ✅ **Hold counts** show handhold/foothold breakdown

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
  "handhold_count": 8,
  "foothold_count": 4,
  "hold_types": {...},
  "average_confidence": 0.85,
  "wall_incline": "moderate_overhang",
  "wall_angle_degrees": 110.0,
  "wall_angle_weights": {
    "handhold_weight": 0.70,
    "foothold_weight": 0.30
  },
  "wall_segments": [
    {"y_start": 0, "y_end": 500, "angle": 90, "category": "vertical"},
    {"y_start": 500, "y_end": 1080, "angle": 115, "category": "moderate_overhang"}
  ],
  "handhold_dimensions": [
    {"hold_id": 1, "width": 45, "height": 30, "area": 1350, "type": "crimp"},
    {"hold_id": 2, "width": 60, "height": 40, "area": 2400, "type": "jug"}
  ],
  "foothold_dimensions": [
    {"hold_id": 3, "width": 35, "height": 25, "area": 875, "type": "foot-hold"},
    {"hold_id": 4, "width": 40, "height": 30, "area": 1200, "type": "foot-hold"}
  ],
  "handhold_distance_metrics": {
    "average_distance": 245.5,
    "max_distance": 520.0,
    "normalized_avg": 0.22,
    "normalized_max": 0.47,
    "distance_variance": 85.3
  },
  "foothold_distance_metrics": {
    "average_distance": 210.0,
    "max_distance": 350.0,
    "normalized_avg": 0.19,
    "normalized_max": 0.32,
    "distance_variance": 65.2
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
      "combined_hold_difficulty": 7.2,
      "handhold_difficulty": 8.5,
      "foothold_difficulty": 4.2,
      "combined_hold_density": 4.1,
      "handhold_density": 5.0,
      "foothold_density": 2.5,
      "combined_distance": 5.8,
      "handhold_distance": 6.5,
      "foothold_distance": 4.0,
      "wall_incline": 9.6
    },
    "hold_counts": {
      "handholds": 8,
      "footholds": 4,
      "total": 12
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

**Key Data Updates:**

1. ✅ **Separate hold counts** for handholds and footholds
2. ✅ **Wall angle weights** showing foothold importance for this route
3. ✅ **Separate dimension arrays** for handholds and footholds with types
4. ✅ **Separate distance metrics** for handhold reaches and foothold spacing
5. ✅ **Separate difficulty/density/distance scores** in breakdown
6. ✅ **Combined scores** showing weighted integration
7. ✅ **Hold count breakdown** showing hand/foot/total

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

### Edge Case 16: No Footholds Detected (Campusing)

**Problem**: Route has zero footholds detected - forces campusing

**Solution**:

- **CRITICAL**: Apply maximum difficulty penalty (score 12.0) across all three factors
- Factor 1: Foothold difficulty = 12.0
- Factor 2: Foothold density = 12.0
- Factor 3: Foothold distance = 12.0
- Combined with wall-angle weights, this appropriately increases difficulty
- On vertical: adds ~5.4 points to final score (45% of 12.0)
- On slab: adds ~7.8 points to final score (65% of 12.0)
- Flag route as "Campus Problem" in metadata
- User should verify this is intentional (not detection failure)

### Edge Case 17: Very Few Footholds (1-2)

**Problem**: Extremely sparse footholds severely limit balance options

**Solution**:

- Apply high difficulty scores (11.0 for 1, 9.5 for 2)
- Apply scarcity multipliers (1.5x) to foothold difficulty
- Result: Very high difficulty contribution from foothold factors
- Appropriate behavior: routes with 1-2 footholds are extremely technical
- Document that this is expected behavior for balance-dependent routes

### Edge Case 18: All Footholds Very Small (< 800px²)

**Problem**: Route has adequate foothold count but all are tiny

**Solution**:

- High foothold difficulty score (avg ~9) despite adequate count
- Foothold density score remains moderate (based on count)
- Combined effect: high difficulty from Factor 1, moderate from Factor 2
- Correct behavior: small footholds significantly increase technical difficulty
- Example: 6 tiny footholds harder than 3 large footholds

### Edge Case 19: Footholds Misclassified as Handholds

**Problem**: Detection model incorrectly classifies footholds as handholds

**Solution**:

- Cannot be detected algorithmically without additional context
- Mitigate by training better detection model
- User feedback mechanism to report misclassifications
- For now: trust detection model output
- Future: implement confidence-based hold type reassignment

### Edge Case 20: Extreme High-Steps (Foothold Spacing > 400px)

**Problem**: Very large vertical gaps between footholds

**Solution**:

- Apply maximum foothold distance score (9.0 + 3.0 crux = 12.0)
- On slabs (65% foothold weight): adds ~7.8 to combined distance score
- Appropriate behavior: extreme high-steps require flexibility
- May indicate detection gap (missed intermediate foothold)
- Flag for user review if spacing exceeds threshold (e.g., >0.40 normalized)

### Edge Case 21: Foothold-Heavy vs Handhold-Heavy Routes

**Problem**: Route has 15 footholds but only 5 handholds

**Solution**:

- Each factor calculates separate hand/foot scores
- Density: handhold density = high (few holds), foothold density = low (many holds)
- Combined with wall-angle weights produces appropriate result
- Vertical example: (high_hand × 0.55) + (low_foot × 0.45) = moderate score
- Correct behavior: system balances hand and foot availability

### Edge Case 22: Slab Route with No Footholds Detected

**Problem**: Slab + campusing is extremely rare and likely detection error

**Solution**:

- Slab campusing would score: foothold penalty × 65% = extreme difficulty
- Flag as "UNLIKELY - Review Detection" in metadata
- Recommend user review: slabs almost never require campusing
- If confirmed correct: route is elite-level (V10+)
- Consider lowering detection confidence threshold for slab images

### Edge Case 23: Mixed Foothold Sizes

**Problem**: Route has mix of large and tiny footholds

**Solution**:

- Average foothold difficulty reflects mix
- Example: 3 large (score 1) + 3 tiny (score 9) = avg 5.0
- Scarcity multiplier applied to average: 5.0 × 1.1 = 5.5
- Correct behavior: mixed sizes create moderate foothold difficulty
- Variance captured in overall route complexity

### Edge Case 24: Foothold Distance Calculation with <2 Footholds

**Problem**: Cannot calculate distance metrics with 0-1 footholds

**Solution**:

- 0 footholds: distance score = 12.0 (campusing)
- 1 foothold: distance score = 11.0 (single balance point, no spacing calc)
- Both cases handled in `calculate_foothold_distance_score()`
- Prevents division by zero or empty list errors
- Appropriate high difficulty scores reflect reality

### Edge Case 25: Wall Angle Changes Affect Foothold Importance

**Problem**: Same foothold configuration scores differently on slab vs overhang

**Solution**:

- **This is correct behavior, not a bug**
- Wall-angle-dependent weighting reflects climbing biomechanics
- Slab: sparse/small footholds weighted heavily (65%)
- Overhang: same footholds weighted lightly (25%)
- Document this in UI/explainability: "foothold importance varies by wall angle"
- Example: 3 small footholds on slab vs overhang = different difficulty

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
  algorithm_version: "v2_with_footholds"
  weights:
    hold_difficulty: 0.35
    hold_density: 0.25
    distance: 0.20
    wall_incline: 0.20
  
  # Handhold configuration
  handhold_distance_thresholds:
    close: 150
    moderate: 300
    wide: 500
    very_wide: 600
  handhold_size_thresholds:
    crimp_small: 500
    crimp_medium: 1000
    crimp_large: 2000
    sloper_small: 1500
    sloper_large: 3000
    jug_small: 2000
  
  # Foothold configuration (NEW - CRITICAL)
  foothold_enabled: true
  foothold_distance_thresholds:
    close: 120
    moderate: 200
    wide: 300
    very_wide: 400
  foothold_size_thresholds:
    very_small: 800
    small: 1500
    medium: 3000
  foothold_scarcity_multipliers:
    zero_footholds: 12.0
    one_to_two: 1.5
    three_to_four: 1.25
    five_to_six: 1.1
    seven_plus: 1.0
  foothold_density_scores:
    zero: 12.0
    one: 11.0
    two: 9.5
    three_to_four: 8.0
    five_to_six: 6.0
    seven_to_eight: 4.0
    nine_to_twelve: 2.5
    thirteen_plus: 1.0
  
  # Wall-angle-dependent foothold weighting (CRITICAL)
  wall_angle_foothold_weights:
    slab:
      handhold_weight: 0.35
      foothold_weight: 0.65
    vertical:
      handhold_weight: 0.55
      foothold_weight: 0.45
    slight_overhang:
      handhold_weight: 0.60
      foothold_weight: 0.40
    moderate_overhang:
      handhold_weight: 0.70
      foothold_weight: 0.30
    steep_overhang:
      handhold_weight: 0.75
      foothold_weight: 0.25
  
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
   - `test_calculate_hold_dimensions()`: Verify area calculations for both handholds and footholds
   - `test_separate_holds()`: Ensure correct separation of handholds and footholds
   - `test_analyze_handhold_difficulty()`: Test handhold difficulty scoring
   - `test_analyze_foothold_difficulty()`: Test foothold difficulty scoring (including campusing)
   - `test_analyze_handhold_density()`: Verify logarithmic relationship
   - `test_analyze_foothold_density()`: Test foothold scarcity scoring
   - `test_analyze_handhold_distances()`: Test handhold distance calculations
   - `test_analyze_foothold_distances()`: Test foothold distance calculations (high-steps)
   - `test_get_foothold_weight()`: Verify wall-angle-dependent weights
   - `test_combine_with_wall_weights()`: Test hand/foot score integration
   - `test_analyze_wall_incline()`: Test incline scoring and multipliers
   - `test_map_score_to_grade()`: Verify grade boundaries
   - `test_detect_wall_transitions()`: Test transition detection logic
   - `test_calculate_transition_multiplier()`: Verify multiplier calculation
   - `test_calculate_hold_type_entropy()`: Test entropy calculation
   - `test_calculate_variability_multiplier()`: Verify variability scoring
   - `test_campusing_penalty()`: Verify no-foothold extreme difficulty across all factors
   - `test_foothold_scarcity_multipliers()`: Test 1-2, 3-4, 5-6 foothold scenarios

2. **Integration Tests**:
   - `test_predict_grade_v2_integration()`: Full pipeline test with footholds and wall incline
   - `test_foothold_integration()`: Test all three factors with foothold data
   - `test_wall_angle_weight_application()`: Verify weights applied correctly across factors
   - `test_slab_vs_overhang_foothold_impact()`: Compare same footholds on different angles
   - `test_campusing_route_integration()`: Full pipeline with zero footholds
   - `test_sparse_footholds_integration()`: Test routes with 1-3 footholds
   - `test_backward_compatibility()`: Ensure API compatibility
   - `test_confidence_adjustment()`: Verify confidence handling
   - `test_wall_incline_defaults()`: Test missing wall incline data handling
   - `test_complexity_multipliers_integration()`: Test with transitions and variability
   - `test_single_vs_multi_segment()`: Compare single wall vs segmented wall
   - `test_multiplier_combinations()`: Test various multiplier scenarios
   - `test_separate_vs_combined_scores()`: Verify hand/foot scores combine correctly

3. **Validation Tests**:
   - `test_grade_accuracy()`: Compare against ground truth (including foothold routes)
   - `test_grade_distribution()`: Ensure reasonable grade spread
   - `test_consistency()`: Same input → same output
   - `test_wall_incline_impact()`: Verify wall incline changes grade appropriately
   - `test_foothold_impact_by_wall_angle()`: Verify foothold importance varies by angle
   - `test_campusing_difficulty_increase()`: Verify no footholds significantly increases grade
   - `test_sparse_foothold_difficulty()`: Verify 1-3 footholds increases difficulty appropriately
   - `test_foothold_size_impact()`: Verify small vs large footholds affect difficulty
   - `test_high_step_difficulty()`: Verify large foothold spacing increases difficulty
   - `test_transition_impact()`: Verify transitions increase difficulty
   - `test_variability_impact()`: Verify hold variety increases difficulty
   - `test_multiplier_bounds()`: Ensure multipliers stay within 1.0-1.5x range
   - `test_extreme_complexity()`: Test routes with max complexity don't break
   - `test_slab_campus_detection_warning()`: Verify unlikely scenarios are flagged

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
| :--: | :--------: |
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

