# Persona-Based Grade Prediction Personalization - Analysis & Design

## Executive Summary

This document analyzes the feasibility and design approach for adding **user persona/climbing style preferences** to personalize grade predictions in the bouldering analysis system.

**Status**: ✅ **FEASIBLE** with existing algorithm structure

**Key Finding**: The current 4-factor + 2-multiplier algorithm is well-suited for persona-based adjustments. Personas can modify factor weights, base scores, and multiplier sensitivities to reflect individual climbing strengths.

**Recommended Approach**: Implement as **Phase 1.5** - after core algorithm validation but before video analysis (Phase 2).

---

## 1. Technical Feasibility Analysis

### 1.1 Compatibility with Existing Algorithm

**Current Algorithm Structure** (from [`grade_prediction_algorithm.md`](grade_prediction_algorithm.md)):

```text
Base Score = f(Hold Difficulty, Hold Density, Distance, Wall Incline) 
            with weights [0.35, 0.25, 0.20, 0.20]

Final Score = Base Score × Wall Transition Multiplier × Hold Variability Multiplier
```

**✅ Excellent Compatibility** - The algorithm supports personalization through:

1. **Factor-level adjustments**: Modify how each factor contributes to the score
2. **Multiplier sensitivity**: Adjust how complexity affects perceived difficulty
3. **Non-destructive**: Persona adjustments layer on top without changing base algorithm
4. **Reversible**: Can disable personalization to see "objective" grade

### 1.2 Integration Points

#### Option A: Adjust Factor Weights (Recommended Initial Approach)

Modify the weight distribution based on persona:

```python
# Base weights (neutral climber)
base_weights = {
    'hold_difficulty': 0.35,
    'hold_density': 0.25,
    'distance': 0.20,
    'wall_incline': 0.20
}

# Slab expert adjustments
slab_expert_weights = {
    'hold_difficulty': 0.38,  # More sensitive to hold quality
    'hold_density': 0.27,     # Slightly more important
    'distance': 0.15,         # Less affected by reaches (better balance)
    'wall_incline': 0.20      # Same importance
}
# Weights must sum to 1.0
```

**Pros**: Simple, interpretable, preserves algorithm structure
**Cons**: Limited personalization range

#### Option B: Adjust Factor Scores with Multipliers (Recommended Final Approach)

Apply persona-specific multipliers to each factor score:

```python
# Example: Slab expert on a slab route
persona_adjustments = {
    'hold_difficulty': 1.0,    # Neutral
    'hold_density': 1.0,       # Neutral
    'distance': 0.85,          # 15% easier (better balance on slabs)
    'wall_incline': 0.70       # 30% easier when wall_incline is slab
}

# Apply to each factor score before combining
adjusted_distance_score = base_distance_score * 0.85
```

**Pros**: More granular control, can create significant personalization
**Cons**: Requires careful calibration

#### Option C: Hybrid Approach (Recommended for Implementation)

Combine both weight adjustment AND score multipliers:

1. Start with persona-adjusted weights
2. Apply score multipliers for specific conditions (e.g., slab on slab routes)
3. Provides both global preferences and context-specific adjustments

### 1.3 Challenges & Limitations

#### Challenge 1: Calibration Complexity

**Issue**: How to determine appropriate adjustment values for each persona?

**Solutions**:

- Start with conservative adjustments (±10-15%)
- Use climber feedback to calibrate over time
- A/B test different adjustment magnitudes
- Provide user control over "personalization strength" (light/medium/strong)

#### Challenge 2: Objective Truth vs Subjective Experience

**Issue**: Grade is meant to be objective, but personas make it subjective

**Solutions**:

- Always show both "standard grade" and "personalized grade"
- Label clearly: "V5 (feels like V4 for slab climbers)"
- Use for user guidance, not official route grading
- Allow users to toggle personalization on/off

#### Challenge 3: Overfitting to User Self-Assessment

**Issue**: Users may incorrectly identify their strengths

**Solutions**:

- Provide descriptions and examples for each persona
- Allow multiple personas with confidence weights
- Use feedback data to suggest persona adjustments
- Start with "balanced" persona as default

#### Challenge 4: Data Sparsity

**Issue**: Limited feedback data for each persona type

**Solutions**:

- Group similar personas for initial calibration
- Use domain knowledge (climbing expertise) for initial values
- Iteratively refine based on user feedback
- Document uncertainty in predictions

### 1.4 Feasibility Conclusion

**✅ TECHNICALLY FEASIBLE**

The existing algorithm structure is well-suited for persona-based personalization. The primary challenges are calibration and UX design, not technical implementation.

**Recommended Path**:

1. Implement basic persona system with domain-knowledge-based adjustments
2. Collect user feedback and calibrate
3. Iteratively refine adjustment values
4. Consider ML-based personalization in future

---

## 2. Climbing Persona Definitions

### 2.1 Persona Research Methodology

Based on climbing domain knowledge, coaching literature, and common climbing specializations, I've identified 7 core personas that represent distinct climbing styles and strengths.

**Key Principle**: Personas represent **relative strengths**, not absolute skill levels. A V5 power climber and V8 power climber share similar difficulty perception patterns, just at different grade ranges.

### 2.2 The 7 Core Climbing Personas

---

#### Persona 1: **Slab Specialist** 🦶

**Primary Strengths**:

- Exceptional balance and body positioning
- Precise footwork on small edges
- Excellent friction climbing technique
- Static, controlled movement
- Mental composure on delicate sequences

**Primary Weaknesses**:

- Struggles on steep overhangs (>105°)
- Limited upper body power
- Difficulty with dynamic moves
- Challenges with campus-style climbing
- Less effective on powerful, compression problems

**Climbing Style**: Methodical, calculated, relies on lower body strength and balance

**Typical Training**: Balance exercises, slab climbing, edging technique, mental focus

**Factor Sensitivities**:

- **Hold Difficulty**: Neutral to slightly higher sensitivity (relies on precise edges)
- **Hold Density**: Slightly lower sensitivity (can use minimal holds efficiently)
- **Distance**: Lower sensitivity on slabs (excellent balance compensates for reaches)
- **Wall Incline**: HIGHLY sensitive - slabs feel much easier, overhangs much harder

**Adjustment Profile**:

```json
{
  "name": "Slab Specialist",
  "code": "slab_specialist",
  "description": "Excels at balance, footwork, and friction climbing on low-angle walls",
  "strengths": ["balance", "footwork", "friction", "static_movement"],
  "weaknesses": ["overhang", "power", "dynamic", "campus"],
  "factor_adjustments": {
    "slab_routes": {
      "wall_incline": 0.65,     // 35% easier on slabs
      "distance": 0.85,          // 15% easier reaches (better balance)
      "hold_density": 0.90,      // 10% easier (efficient movement)
      "hold_difficulty": 1.0     // Neutral
    },
    "overhang_routes": {
      "wall_incline": 1.35,      // 35% harder on overhangs
      "distance": 1.15,          // 15% harder reaches (poor position)
      "hold_difficulty": 1.10,   // 10% harder (less powerful grip)
      "hold_density": 1.05       // 5% harder (needs more options)
    }
  }
}
```

---

#### Persona 2: **Power/Campus Climber** 💪

**Primary Strengths**:

- Explosive upper body strength
- Dynamic movement capability
- Powerful lock-offs and pulls
- Campus board proficiency
- Strong contact strength (ability to stick poor holds dynamically)

**Primary Weaknesses**:

- Limited endurance on long routes
- Struggles with technical slab climbing
- Poor on delicate, balance-dependent problems
- Less efficient movement (uses power over technique)
- Foot precision may be lacking

**Climbing Style**: Aggressive, powerful, prefers short, intense problems

**Typical Training**: Campus board, weighted pull-ups, plyometric training, finger strength

**Factor Sensitivities**:

- **Hold Difficulty**: Lower sensitivity (can muscle through poor holds)
- **Hold Density**: Lower sensitivity (can make big moves between holds)
- **Distance**: MUCH lower sensitivity (powerful reaches and dynos are strengths)
- **Wall Incline**: Lower sensitivity on overhangs, higher on slabs

**Adjustment Profile**:

```json
{
  "name": "Power/Campus Climber",
  "code": "power_climber",
  "description": "Excels at powerful, dynamic movements and steep terrain",
  "strengths": ["power", "dynamic", "campus", "lockoff", "overhang"],
  "weaknesses": ["slab", "endurance", "balance", "delicate_footwork"],
  "factor_adjustments": {
    "overhang_routes": {
      "wall_incline": 0.75,      // 25% easier on overhangs
      "distance": 0.70,          // 30% easier (big reaches are strength)
      "hold_difficulty": 0.85,   // 15% easier (strong grip)
      "hold_density": 0.80       // 20% easier (can skip holds)
    },
    "slab_routes": {
      "wall_incline": 1.25,      // 25% harder on slabs
      "distance": 1.10,          // 10% harder (balance issues)
      "hold_difficulty": 1.15,   // 15% harder (can't muscle through)
      "hold_density": 1.10       // 10% harder (needs technique)
    },
    "high_distance_routes": {
      "distance": 0.70           // 30% easier on long reaches
    }
  }
}
```

---

#### Persona 3: **Technical/Beta Reader** 🧠

**Primary Strengths**:

- Excellent sequencing and route reading
- Efficient movement patterns
- Body positioning optimization
- Adaptability to different styles
- Problem-solving under pressure

**Primary Weaknesses**:

- May lack raw power for physical cruxes
- Can struggle with pure strength problems
- Less effective when perfect beta isn't enough
- May overthink simple powerful moves

**Climbing Style**: Cerebral, efficient, focuses on optimal movement

**Typical Training**: Varied climbing, flash attempts, route analysis, flexibility

**Factor Sensitivities**:

- **Hold Difficulty**: Neutral (adapts technique to holds)
- **Hold Density**: Lower sensitivity (finds efficient sequences)
- **Distance**: Neutral (optimizes body position)
- **Wall Incline**: Neutral (adapts to all angles)
- **Complexity Multipliers**: LOWER sensitivity (handles complexity well)

**Adjustment Profile**:

```json
{
  "name": "Technical/Beta Reader",
  "code": "technical_climber",
  "description": "Excels at route reading, efficient sequences, and adaptable technique",
  "strengths": ["route_reading", "efficiency", "adaptability", "sequencing"],
  "weaknesses": ["raw_power", "pure_strength_problems"],
  "factor_adjustments": {
    "all_routes": {
      "hold_difficulty": 0.95,   // 5% easier (efficient technique)
      "hold_density": 0.90,      // 10% easier (finds sequences)
      "distance": 0.95,          // 5% easier (optimizes positioning)
      "wall_incline": 1.0        // Neutral
    }
  },
  "multiplier_adjustments": {
    "wall_transitions": 0.80,    // 20% less affected by transitions
    "hold_variability": 0.75     // 25% less affected by variety (adapts well)
  }
}
```

---

#### Persona 4: **Crimp Specialist** 🤏

**Primary Strengths**:

- Exceptional finger strength on small holds
- Static pulling power
- Precise hand placement
- Patience and control
- Excellent on vertical to slightly overhanging terrain

**Primary Weaknesses**:

- Vulnerable to tweaky holds (injury risk)
- Struggles with slopers and open-hand holds
- Limited on juggy, powerful problems
- May lack dynamic movement capability
- Less effective on compression problems

**Climbing Style**: Controlled, precise, relies on finger strength

**Typical Training**: Hangboard, finger strength protocols, half-crimp technique

**Factor Sensitivities**:

- **Hold Difficulty**: MUCH lower on crimps, higher on slopers
- **Hold Density**: Neutral
- **Distance**: Slightly higher (prefers static over dynamic)
- **Wall Incline**: Neutral to slightly lower on vertical

**Adjustment Profile**:

```json
{
  "name": "Crimp Specialist",
  "code": "crimp_specialist",
  "description": "Excels at small holds requiring finger strength and precise control",
  "strengths": ["finger_strength", "crimps", "static_movement", "precision"],
  "weaknesses": ["slopers", "dynamic", "open_hand", "compression"],
  "factor_adjustments": {
    "crimp_heavy_routes": {
      "hold_difficulty": 0.70,   // 30% easier on crimpy routes
      "distance": 1.05,          // 5% harder (prefers static)
      "hold_density": 0.95,      // 5% easier (efficient on small holds)
      "wall_incline": 0.95       // 5% easier on vertical
    },
    "sloper_heavy_routes": {
      "hold_difficulty": 1.35,   // 35% harder on slopers
      "distance": 1.10,          // 10% harder (can't static crimp)
      "hold_density": 1.10,      // 10% harder (needs options)
      "wall_incline": 1.05       // 5% harder
    }
  }
}
```

---

#### Persona 5: **Flexibility/Mobility Specialist** 🤸

**Primary Strengths**:

- Exceptional range of motion
- High steps and heel hooks
- Stemming and splits positions
- Creative body positioning
- Effective on compression and wide problems

**Primary Weaknesses**:

- May lack pure finger strength
- Struggles with small crimpy holds
- Less effective on powerful, locked-off positions
- Can be challenged by standard beta (uses unconventional sequences)

**Climbing Style**: Creative, flowing, uses flexibility advantages

**Typical Training**: Yoga, flexibility work, hip mobility, creative problems

**Factor Sensitivities**:

- **Hold Difficulty**: Higher on crimps, lower on volumes/compression
- **Hold Density**: Lower (can use holds in unconventional ways)
- **Distance**: Much lower (can make wide moves easily)
- **Wall Incline**: Lower on overhangs (good core and hip flexibility)

**Adjustment Profile**:

```json
{
  "name": "Flexibility/Mobility Specialist",
  "code": "flexibility_specialist",
  "description": "Excels at high steps, compression, and creative body positions",
  "strengths": ["flexibility", "high_steps", "compression", "creativity", "stemming"],
  "weaknesses": ["crimps", "finger_strength", "standard_beta"],
  "factor_adjustments": {
    "all_routes": {
      "hold_difficulty": 1.10,   // 10% harder (may lack finger strength)
      "hold_density": 0.85,      // 15% easier (uses holds creatively)
      "distance": 0.75,          // 25% easier (wide moves are easy)
      "wall_incline": 0.90       // 10% easier on overhangs (core/hip flexibility)
    },
    "compression_routes": {
      "hold_difficulty": 0.80,   // 20% easier on volumes/compression
      "distance": 0.70           // 30% easier (flexibility shines)
    }
  }
}
```

---

#### Persona 6: **Endurance Specialist** ⏱️

**Primary Strengths**:

- Excellent recovery ability
- Sustained climbing capability
- Mental fortitude on long problems
- Efficient energy management
- Strong aerobic capacity

**Primary Weaknesses**:

- Limited on short, powerful boulder problems
- Lacks peak power for single hard moves
- Struggles with pure strength cruxes
- Less effective on low-hold-count powerful problems

**Climbing Style**: Steady, paced, prefers longer sequences

**Typical Training**: ARCing, volume climbing, long routes, circuit training

**Factor Sensitivities**:

- **Hold Difficulty**: Neutral
- **Hold Density**: LOWER on high-density routes (endurance shines)
- **Distance**: Neutral
- **Wall Incline**: Slightly lower (sustained climbing on all angles)

**Note**: This persona is less relevant for typical bouldering (which emphasizes power over endurance), but becomes important for longer gym routes or outdoor problems with many moves.

**Adjustment Profile**:

```json
{
  "name": "Endurance Specialist",
  "code": "endurance_specialist",
  "description": "Excels at sustained climbing and recovery on longer problems",
  "strengths": ["endurance", "recovery", "mental_fortitude", "efficiency"],
  "weaknesses": ["power", "single_move_strength", "short_powerful_problems"],
  "factor_adjustments": {
    "high_hold_count_routes": {
      "hold_density": 0.85,      // 15% easier on many holds
      "hold_difficulty": 0.95,   // 5% easier (paces effort)
      "distance": 1.0,           // Neutral
      "wall_incline": 0.95       // 5% easier (sustained effort)
    },
    "low_hold_count_routes": {
      "hold_density": 1.20,      // 20% harder (lacks peak power)
      "hold_difficulty": 1.10,   // 10% harder (can't power through)
      "distance": 1.10           // 10% harder (needs power)
    }
  }
}
```

---

#### Persona 7: **Balanced/All-Arounder** ⚖️

**Primary Strengths**:

- Well-rounded skillset
- Adaptable to most route styles
- No major weaknesses
- Consistent performance across styles
- Good foundation in all areas

**Primary Weaknesses**:

- Lacks specialization advantages
- No exceptional strengths
- May be outperformed by specialists on specific styles
- Jack of all trades, master of none

**Climbing Style**: Versatile, adaptive, standard technique

**Typical Training**: Varied climbing, well-rounded training program

**Factor Sensitivities**:

- All factors: Neutral (baseline)
- This is the "default" persona

**Adjustment Profile**:

```json
{
  "name": "Balanced/All-Arounder",
  "code": "balanced",
  "description": "Well-rounded climber with no major strengths or weaknesses",
  "strengths": ["versatility", "adaptability", "consistency"],
  "weaknesses": ["lacks_specialization"],
  "factor_adjustments": {
    "all_routes": {
      "hold_difficulty": 1.0,
      "hold_density": 1.0,
      "distance": 1.0,
      "wall_incline": 1.0
    }
  },
  "multiplier_adjustments": {
    "wall_transitions": 1.0,
    "hold_variability": 1.0
  }
}
```

---

## 3. Persona Correlation Matrix

### 3.1 Skill Overlap Analysis

This matrix shows how personas share similar skills and adaptations. Higher correlation means the personas have overlapping strengths.

```text
                    Slab  Power  Tech  Crimp  Flex  Endur  Bal
Slab Specialist      1.0   0.1   0.5   0.4    0.3   0.4   0.6
Power Climber        0.1   1.0   0.3   0.2    0.5   0.1   0.5
Technical            0.5   0.3   1.0   0.6    0.6   0.6   0.8
Crimp Specialist     0.4   0.2   0.6   1.0    0.2   0.5   0.6
Flexibility          0.3   0.5   0.6   0.2    1.0   0.4   0.6
Endurance            0.4   0.1   0.6   0.5    0.4   1.0   0.7
Balanced             0.6   0.5   0.8   0.6    0.6   0.7   1.0
```

### 3.2 Correlation Insights

**High Correlation Pairs** (>0.6):

- **Technical ↔ Balanced** (0.8): Technical climbers are well-rounded
- **Endurance ↔ Balanced** (0.7): Endurance climbers need versatility
- **Technical ↔ Crimp** (0.6): Both value precision and control
- **Technical ↔ Flexibility** (0.6): Both use creative problem-solving

**Low Correlation Pairs** (<0.3):

- **Slab ↔ Power** (0.1): Opposite styles (balance vs strength)
- **Power ↔ Endurance** (0.1): Opposite energy systems
- **Crimp ↔ Flexibility** (0.2): Different physical adaptations
- **Power ↔ Crimp** (0.2): Different strength types

**Practical Implications**:

1. Users can combine compatible personas (e.g., Technical + Crimp)
2. Incompatible personas should not be combined (e.g., Slab + Power)
3. Balanced persona can blend with any specialization
4. Technical skills complement most physical specializations

### 3.3 Persona Groupings

Based on correlation analysis, personas group into 3 categories:

**Group A: Physical Specialists** (focus on body positioning/movement style)

- Slab Specialist (balance/footwork)
- Power Climber (strength/dynamics)
- Flexibility Specialist (mobility/creativity)

**Group B: Technical Specialists** (focus on hold interaction/precision)

- Crimp Specialist (finger strength)
- Technical Climber (efficiency)
- Endurance Specialist (sustained effort)

**Group C: Generalists**

- Balanced All-Arounder (baseline)

**Recommendation**: Allow users to select ONE from Group A and ONE from Group B, or just Balanced. This prevents contradictory persona combinations.

---

## 4. Persona Adjustment System Design

### 4.1 Adjustment Mechanism Architecture

#### Layer 1: Route Classification

Before applying persona adjustments, classify the route based on detected features:

```python
def classify_route_characteristics(features: dict) -> dict:
    """
    Classify route based on detected features to determine
    which persona adjustments apply.
    """
    characteristics = {
        'wall_angle_category': None,  # slab, vertical, overhang, steep
        'hold_type_dominant': None,   # crimp, sloper, jug, mixed
        'distance_category': None,    # close, moderate, wide
        'hold_count_category': None,  # few, moderate, many
        'complexity_level': None      # low, medium, high
    }
    
    # Classify wall angle
    wall_incline = features.get('wall_incline', 'vertical')
    if wall_incline in ['slab']:
        characteristics['wall_angle_category'] = 'slab'
    elif wall_incline in ['vertical']:
        characteristics['wall_angle_category'] = 'vertical'
    elif wall_incline in ['slight_overhang', 'moderate_overhang']:
        characteristics['wall_angle_category'] = 'overhang'
    else:
        characteristics['wall_angle_category'] = 'steep'
    
    # Classify dominant hold type
    hold_types = features.get('hold_types', {})
    total_handholds = sum(hold_types.get(ht, 0) for ht in ['crimp', 'jug', 'sloper', 'pinch', 'pocket'])
    
    if total_handholds > 0:
        crimp_ratio = hold_types.get('crimp', 0) / total_handholds
        sloper_ratio = hold_types.get('sloper', 0) / total_handholds
        jug_ratio = hold_types.get('jug', 0) / total_handholds
        
        if crimp_ratio > 0.5:
            characteristics['hold_type_dominant'] = 'crimp'
        elif sloper_ratio > 0.4:
            characteristics['hold_type_dominant'] = 'sloper'
        elif jug_ratio > 0.5:
            characteristics['hold_type_dominant'] = 'jug'
        else:
            characteristics['hold_type_dominant'] = 'mixed'
    
    # Classify distance
    distance_metrics = features.get('distance_metrics', {})
    avg_normalized_distance = distance_metrics.get('normalized_avg', 0.2)
    
    if avg_normalized_distance < 0.15:
        characteristics['distance_category'] = 'close'
    elif avg_normalized_distance < 0.30:
        characteristics['distance_category'] = 'moderate'
    else:
        characteristics['distance_category'] = 'wide'
    
    # Classify hold count
    total_holds = features.get('total_holds', 10)
    if total_holds < 6:
        characteristics['hold_count_category'] = 'few'
    elif total_holds < 13:
        characteristics['hold_count_category'] = 'moderate'
    else:
        characteristics['hold_count_category'] = 'many'
    
    # Classify complexity
    complexity = features.get('complexity_analysis', {})
    transitions = complexity.get('wall_transitions', {}).get('count', 0)
    entropy = complexity.get('hold_variability', {}).get('entropy', 0)
    
    complexity_score = (transitions * 0.3) + (entropy * 0.7)
    if complexity_score < 0.5:
        characteristics['complexity_level'] = 'low'
    elif complexity_score < 1.5:
        characteristics['complexity_level'] = 'medium'
    else:
        characteristics['complexity_level'] = 'high'
    
    return characteristics
```

#### Layer 2: Persona Adjustment Lookup

Match route characteristics to persona adjustment profiles:

```python
def get_persona_adjustments(persona: str, route_chars: dict) -> dict:
    """
    Get adjustment multipliers for each factor based on persona
    and route characteristics.
    """
    # Load persona profiles from configuration
    persona_profiles = load_persona_profiles()
    
    if persona not in persona_profiles:
        # Default to balanced persona
        return {
            'hold_difficulty': 1.0,
            'hold_density': 1.0,
            'distance': 1.0,
            'wall_incline': 1.0,
            'transition_multiplier': 1.0,
            'variability_multiplier': 1.0
        }
    
    profile = persona_profiles[persona]
    adjustments = {'hold_difficulty': 1.0, 'hold_density': 1.0, 
                   'distance': 1.0, 'wall_incline': 1.0,
                   'transition_multiplier': 1.0, 'variability_multiplier': 1.0}
    
    # Apply adjustments based on route characteristics
    wall_angle = route_chars['wall_angle_category']
    hold_type = route_chars['hold_type_dominant']
    distance_cat = route_chars['distance_category']
    hold_count = route_chars['hold_count_category']
    
    # Example: Slab specialist on slab route
    if persona == 'slab_specialist' and wall_angle == 'slab':
        adjustments['wall_incline'] = 0.65
        adjustments['distance'] = 0.85
        adjustments['hold_density'] = 0.90
    
    # Example: Power climber on overhang with wide spacing
    if persona == 'power_climber' and wall_angle in ['overhang', 'steep']:
        adjustments['wall_incline'] = 0.75
        adjustments['distance'] = 0.70
        adjustments['hold_difficulty'] = 0.85
        if distance_cat == 'wide':
            adjustments['distance'] = 0.65  # Even easier
    
    # Add more persona-specific logic...
    
    return adjustments
```

#### Layer 3: Apply to Prediction Algorithm

Integrate adjustments into the grade prediction flow:

```python
def predict_grade_v2_personalized(
    features: dict,
    detected_holds: list,
    wall_segments: list = None,
    wall_incline: str = 'vertical',
    user_persona: dict = None  # NEW: persona configuration
) -> tuple[str, float, dict]:
    """
    Enhanced grade prediction with persona personalization.
    """
    # Standard Phase 1 prediction (baseline)
    base_grade, base_confidence, base_breakdown = predict_grade_v2(
        features, detected_holds, wall_segments, wall_incline
    )
    
    # If no persona selected, return baseline
    if not user_persona or user_persona.get('persona') == 'balanced':
        return base_grade, base_confidence, base_breakdown
    
    # Classify route characteristics
    route_chars = classify_route_characteristics(features)
    
    # Get persona adjustments
    persona_name = user_persona.get('persona', 'balanced')
    persona_strength = user_persona.get('strength', 'medium')  # light/medium/strong
    
    adjustments = get_persona_adjustments(persona_name, route_chars)
    
    # Apply strength scaling (light=50%, medium=100%, strong=150% adjustment)
    strength_scales = {'light': 0.5, 'medium': 1.0, 'strong': 1.5}
    scale = strength_scales.get(persona_strength, 1.0)
    
    scaled_adjustments = {}
    for factor, multiplier in adjustments.items():
        # Scale adjustment toward 1.0 (neutral) based on strength
        deviation = multiplier - 1.0
        scaled_adjustments[factor] = 1.0 + (deviation * scale)
    
    # Recalculate with adjusted factor scores
    personalized_score = calculate_personalized_score(
        base_breakdown, scaled_adjustments
    )
    
    # Map to grade
    personalized_grade = map_score_to_grade(personalized_score)
    
    # Calculate confidence (lower when grade differs significantly)
    grade_diff = abs(grade_to_numeric(personalized_grade) - grade_to_numeric(base_grade))
    confidence_penalty = min(grade_diff * 0.1, 0.3)
    personalized_confidence = base_confidence * (1 - confidence_penalty)
    
    # Prepare response
    result_breakdown = {
        **base_breakdown,
        'persona_applied': persona_name,
        'persona_strength': persona_strength,
        'route_characteristics': route_chars,
        'adjustments': scaled_adjustments,
        'base_grade': base_grade,
        'personalized_grade': personalized_grade,
        'personalized_score': personalized_score
    }
    
    return personalized_grade, personalized_confidence, result_breakdown
```

### 4.2 Adjustment Range Guidelines

**Conservative Approach** (Recommended for Initial Implementation):

- **Maximum adjustment**: ±35% per factor (multiplier range: 0.65-1.35)
- **Typical adjustment**: ±15-20% per factor (multiplier range: 0.80-1.20)
- **Combined effect**: Can shift grade by ±2-3 grades in extreme cases
- **Calibration target**: Most adjustments result in ±1 grade change

**Adjustment Philosophy**:

- Personas reflect **perceived difficulty**, not absolute difficulty
- Large adjustments only for perfect persona-route matches
- Partial matches get proportionally smaller adjustments
- Always preserve base algorithm as ground truth

### 4.3 Adjustment Examples

#### Example 1: Slab Specialist on Slab Route (V5 Base)

```text
Base Factors:
- Hold Difficulty: 7.2 × 0.35 = 2.52
- Hold Density: 4.1 × 0.25 = 1.03
- Distance: 5.8 × 0.20 = 1.16
- Wall Incline: 3.9 × 0.20 = 0.78  (slab angle)
Base Score: 5.49

Persona Adjustments (Slab Specialist on Slab):
- Hold Difficulty: 7.2 × 1.0 × 0.35 = 2.52
- Hold Density: 4.1 × 0.90 × 0.25 = 0.92
- Distance: 5.8 × 0.85 × 0.20 = 0.99
- Wall Incline: 3.9 × 0.65 × 0.20 = 0.51
Personalized Base Score: 4.94

Multipliers: 1.0 × 1.0 = 1.0 (same)
Final Personalized Score: 4.94

Grade Mapping:
- Base Grade: V5 (score 5.49)
- Personalized Grade: V4-V5 (score 4.94)
- Feels like: "V5 feels like V4-V5 for slab climbers"
```

#### Example 2: Power Climber on Steep Overhang with Wide Spacing (V8 Base)

```text
Base Factors:
- Hold Difficulty: 8.5 × 0.35 = 2.98
- Hold Density: 6.0 × 0.25 = 1.50
- Distance: 7.5 × 0.20 = 1.50
- Wall Incline: 12.0 × 0.20 = 2.40
Base Score: 8.38

Persona Adjustments (Power Climber on Steep Overhang + Wide):
- Hold Difficulty: 8.5 × 0.85 × 0.35 = 2.53
- Hold Density: 6.0 × 0.80 × 0.25 = 1.20
- Distance: 7.5 × 0.65 × 0.20 = 0.98  (wide spacing is easy for power)
- Wall Incline: 12.0 × 0.75 × 0.20 = 1.80
Personalized Base Score: 6.51

Multipliers: 1.0 × 1.0 = 1.0
Final Personalized Score: 6.51

Grade Mapping:
- Base Grade: V8 (score 8.38)
- Personalized Grade: V6 (score 6.51)
- Feels like: "V8 feels like V6 for power climbers" (significant advantage)
```

---

## 5. Multi-Persona and Hybrid Profile Support

### 5.1 Design Rationale

Real climbers rarely fit perfectly into one persona. A climber might be:

- 70% technical, 30% crimp specialist
- Balanced but with slight power preference
- Slab-focused with good technical skills

**Solution**: Support **weighted persona combinations**

### 5.2 Combination Approaches

#### Option A: Primary + Secondary Persona (Recommended)

Users select a primary persona (70% weight) and optional secondary (30% weight):

```python
user_persona = {
    'primary': {
        'persona': 'technical_climber',
        'weight': 0.7
    },
    'secondary': {
        'persona': 'crimp_specialist',
        'weight': 0.3
    },
    'strength': 'medium'
}
```

**Adjustment Calculation**:

```python
def combine_persona_adjustments(primary_adj, secondary_adj, weights):
    """Blend adjustments from multiple personas."""
    combined = {}
    for factor in primary_adj.keys():
        primary_val = primary_adj[factor]
        secondary_val = secondary_adj.get(factor, 1.0)
        
        # Weighted average (geometric mean for multipliers)
        combined[factor] = (
            primary_val ** weights['primary'] * 
            secondary_val ** weights['secondary']
        )
    return combined
```

#### Option B: Slider-Based Spectrum (Advanced UX)

Present as spectrum sliders between opposite personas:

```text
Slab <-------|--------> Power      [Neutral]
Technical <-------|--------> Physical    [Slightly Technical]
Crimp <-------|--------> Flexibility  [Balanced]
```

**Pros**: Intuitive UI, allows fine-tuning
**Cons**: More complex to implement and explain

#### Option C: Skill Attribute System (Future Enhancement)

Rate individual skills on 1-5 scale:

- Balance: ⭐⭐⭐⭐⭐
- Power: ⭐⭐⭐
- Finger Strength: ⭐⭐⭐⭐
- Flexibility: ⭐⭐

Then algorithmically map to persona adjustments.

**Pros**: Most granular, most accurate
**Cons**: Complex calibration, high user effort

### 5.3 Recommended Implementation

**Phase 1**: Single persona selection only (including "balanced")

- Simplest UX
- Easier calibration
- Clear user choice

**Phase 2**: Primary + Secondary with fixed 70/30 split

- Addresses most hybrid cases
- Still manageable calibration
- Reasonable UX complexity

**Phase 3** (Future): Custom attribute sliders or adjustable weights

- Advanced users only
- Requires extensive validation data

### 5.4 Persona Compatibility Rules

Not all persona combinations make sense:

**Compatible Pairs** (can combine):

- Technical + Any (technical skills complement everything)
- Balanced + Any specialist (slight specialization)
- Crimp + Endurance (precision + staying power)
- Flexibility + Power (mobility + strength)
- Slab + Technical (technique-focused)

**Incompatible Pairs** (should warn user):

- Slab + Power (contradictory styles)
- Power + Endurance (opposite energy systems)
- Crimp + Flexibility (competing physical adaptations)

**Validation Logic**:

```python
INCOMPATIBLE_PAIRS = [
    ('slab_specialist', 'power_climber'),
    ('power_climber', 'endurance_specialist'),
    ('crimp_specialist', 'flexibility_specialist')
]

def validate_persona_combination(primary, secondary):
    """Check if persona combination is valid."""
    if (primary, secondary) in INCOMPATIBLE_PAIRS or \
       (secondary, primary) in INCOMPATIBLE_PAIRS:
        return False, "These personas have contradictory styles"
    return True, "Compatible combination"
```

---

## 6. Data Model Changes

### 6.1 User Persona Storage

#### Option A: Store in Analysis Model (Per-Analysis)

**Use Case**: User wants to see how route feels for different personas

```python
# Add to Analysis model
class Analysis(Base):
    # ... existing fields ...
    user_persona = db.Column(db.JSON, nullable=True)
    # Format:
    # {
    #   "primary": "slab_specialist",
    #   "secondary": "technical_climber",
    #   "weights": {"primary": 0.7, "secondary": 0.3},
    #   "strength": "medium"
    # }
```

**Pros**: Historical record of persona used, can compare different personas
**Cons**: Duplicates data if user has consistent persona

#### Option B: Store in User Model (Global Preference)

**Use Case**: User sets their persona once, applies to all analyses

```python
# New User/Profile model
class UserProfile(Base):
    """Stores user preferences and persona settings."""
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), db.ForeignKey('user_sessions.session_id'))
    
    # Persona configuration
    persona_config = db.Column(db.JSON, nullable=True)
    # Format:
    # {
    #   "primary": "power_climber",
    #   "secondary": null,
    #   "weights": {"primary": 1.0, "secondary": 0.0},
    #   "strength": "medium",
    #   "enabled": true
    # }
    
    # User preferences
    preferences = db.Column(db.JSON, nullable=True)
    # Format:
    # {
    #   "show_both_grades": true,
    #   "default_persona_strength": "medium"
    # }
    
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)
```

**Pros**: Cleaner user experience, one-time setup
**Cons**: Harder to compare personas, requires user management

#### Option C: Hybrid Approach (Recommended)

- Store default persona in UserProfile (if user authentication exists)
- Store applied persona in Analysis (for historical tracking)
- Allow override per analysis

```python
# Analysis model - store what was actually used
user_persona_applied = db.Column(db.JSON, nullable=True)

# UserProfile model - store user's default preference
persona_config = db.Column(db.JSON, nullable=True)
```

### 6.2 Configuration Storage

Add persona definitions to `user_config.yaml`:

```yaml
personas:
  enabled: true  # Feature flag
  default: "balanced"
  strength_levels: ["light", "medium", "strong"]
  
  # Persona definitions
  profiles:
    slab_specialist:
      name: "Slab Specialist"
      description: "Excels at balance, footwork, and friction climbing on low-angle walls"
      icon: "🦶"
      strengths: ["balance", "footwork", "friction", "static_movement"]
      weaknesses: ["overhang", "power", "dynamic", "campus"]
      adjustments:
        slab_routes:
          wall_incline: 0.65
          distance: 0.85
          hold_density: 0.90
          hold_difficulty: 1.0
        overhang_routes:
          wall_incline: 1.35
          distance: 1.15
          hold_difficulty: 1.10
          hold_density: 1.05
    
    power_climber:
      name: "Power/Campus Climber"
      description: "Excels at powerful, dynamic movements and steep terrain"
      icon: "💪"
      # ... etc
    
    # ... all 7 personas ...
  
  # Compatibility matrix
  incompatible_pairs:
    - ["slab_specialist", "power_climber"]
    - ["power_climber", "endurance_specialist"]
    - ["crimp_specialist", "flexibility_specialist"]
```

### 6.3 Migration Requirements

**If implementing UserProfile model**:

```python
# Migration script
def upgrade():
    # Create user_profiles table
    op.create_table(
        'user_profiles',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('user_sessions.session_id')),
        sa.Column('persona_config', sa.JSON, nullable=True),
        sa.Column('preferences', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow)
    )
    
    # Add persona field to analyses table
    op.add_column('analyses', sa.Column('user_persona_applied', sa.JSON, nullable=True))
```

**If using Analysis model only**:

```python
# Migration script
def upgrade():
    # Add persona field to existing analyses table
    op.add_column('analyses', sa.Column('user_persona', sa.JSON, nullable=True))
```

### 6.4 API Request/Response Format

**Request** (upload image with persona):

```json
POST /analyze
{
  "image": "<file>",
  "wall_incline": "moderate_overhang",
  "persona": {
    "primary": "power_climber",
    "secondary": null,
    "strength": "medium",
    "enabled": true
  }
}
```

**Response** (with persona results):

```json
{
  "analysis_id": "abc-123",
  "base_grade": "V8",
  "personalized_grade": "V6",
  "confidence": 0.72,
  "persona_applied": {
    "primary": "power_climber",
    "strength": "medium"
  },
  "route_characteristics": {
    "wall_angle_category": "overhang",
    "hold_type_dominant": "mixed",
    "distance_category": "wide",
    "complexity_level": "medium"
  },
  "grade_explanation": "Base grade V8, feels like V6 for power climbers due to overhang advantage and wide spacing",
  "display_format": "V8 → V6 (for power climbers)"
}
```

---

## 7. Implementation Plan & Phasing

### 7.1 Positioning in Overall Plan

**Recommendation**: Implement as **Phase 1.5** - between core algorithm and video analysis

```text
Phase 1: Core Grade Prediction Algorithm ✅
  ├─ 4 base factors
  ├─ 2 complexity multipliers
  └─ Wall incline support

Phase 1.5: Persona Personalization ⬅️ NEW
  ├─ Persona definitions
  ├─ Single persona selection
  ├─ Route classification
  ├─ Adjustment application
  └─ UI integration

Phase 2: Video Analysis Validation (Lower Priority)
  ├─ Pose estimation
  ├─ Performance metrics
  └─ Cross-validation
```

**Rationale**:

- Personas enhance core algorithm without requiring new data collection
- Simpler than video analysis
- Provides immediate user value
- Can be validated with user feedback
- Does not block Phase 2 development

### 7.2 Implementation Stages

#### Stage 1: Foundation (Week 1-2)

**Deliverables**:

- [ ] Define all 7 personas in `user_config.yaml`
- [ ] Create persona configuration schema
- [ ] Add `user_persona` field to Analysis model
- [ ] Database migration
- [ ] Unit tests for persona loading

**Acceptance Criteria**:

- Personas load correctly from config
- Schema validation works
- Database can store persona selections

#### Stage 2: Route Classification (Week 2-3)

**Deliverables**:

- [ ] Implement `classify_route_characteristics()`
- [ ] Test route classification accuracy
- [ ] Validate classification thresholds
- [ ] Unit tests for all route types

**Acceptance Criteria**:

- Routes correctly classified by wall angle
- Hold type dominance detected accurately
- Distance categories align with expectations

#### Stage 3: Adjustment Engine (Week 3-4)

**Deliverables**:

- [ ] Implement `get_persona_adjustments()`
- [ ] Create adjustment lookup logic for all 7 personas
- [ ] Implement adjustment strength scaling (light/medium/strong)
- [ ] Add adjustment application to `predict_grade_v2()`
- [ ] Comprehensive unit tests

**Acceptance Criteria**:

- Adjustments correctly retrieved based on persona + route
- Strength scaling works properly
- Grade predictions change appropriately with personas

#### Stage 4: Multi-Persona Support (Week 4-5)

**Deliverables**:

- [ ] Implement primary/secondary persona blending
- [ ] Add compatibility validation
- [ ] Test various persona combinations
- [ ] Edge case handling

**Acceptance Criteria**:

- Weighted blending produces sensible results
- Incompatible pairs rejected with clear error
- Compatible pairs combine smoothly

#### Stage 5: UI Integration (Week 5-6)

**Deliverables**:

- [ ] Add persona selection dropdown/form in upload page
- [ ] Create persona description cards
- [ ] Add strength level selector
- [ ] Display both base and personalized grades
- [ ] Add explanation text for why grade differs
- [ ] Optional: persona toggle to compare

**Acceptance Criteria**:

- Users can easily select persona
- UI clearly explains persona impact
- Both grades displayed without confusion
- Mobile-friendly design

#### Stage 6: Testing & Calibration (Week 6-8)

**Deliverables**:

- [ ] Integration tests for full pipeline
- [ ] User acceptance testing
- [ ] Collect initial feedback
- [ ] Calibrate adjustment values based on feedback
- [ ] Performance testing
- [ ] Documentation

**Acceptance Criteria**:

- All tests pass
- User feedback positive
- Adjustments feel accurate to test users
- No performance degradation
- Documentation complete

#### Stage 7: Deployment & Monitoring (Week 8-9)

**Deliverables**:

- [ ] Deploy persona feature with feature flag
- [ ] Monitor usage and feedback
- [ ] A/B test adjustment magnitudes
- [ ] Collect user persona selections for analysis
- [ ] Iterate based on data

**Acceptance Criteria**:

- Feature deployed successfully
- No critical bugs
- Users adopting feature
- Feedback mechanism working

**Total Timeline**: 8-9 weeks (~2 months)

### 7.3 Feature Flags & Gradual Rollout

**Configuration**:

```yaml
personas:
  enabled: true
  rollout_percentage: 100  # 0-100, controls who sees feature
  allow_multi_persona: false  # Enable multi-persona in Phase 2
  show_base_grade: true  # Always show non-personalized grade
  default_strength: "medium"
```

**Rollout Strategy**:

1. Week 1-2: Internal testing only (rollout: 0%)
2. Week 3-4: Beta users (rollout: 10%)
3. Week 5-6: Expanded testing (rollout: 50%)
4. Week 7+: Full rollout (rollout: 100%)

### 7.4 Success Metrics

**Quantitative Metrics**:

- Persona feature adoption rate (target: >40% of users)
- User satisfaction with personalized grades (survey: target >3.5/5)
- Feedback agreement rate (personalized vs base)
- Grade prediction accuracy improvement with personas

**Qualitative Metrics**:

- User comments on persona accuracy
- Feature usability feedback
- Persona description clarity
- Confusion or misunderstanding incidents

---

## 8. Risks, Challenges & Mitigation

### 8.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Users misjudge their persona | High | Medium | Provide clear descriptions, examples, quiz |
| Adjustments feel inaccurate | Medium | High | Calibrate with feedback, allow strength control |
| Feature adds confusion | Medium | Medium | Clear UI/UX, show both grades, educational content |
| Calibration data insufficient | High | Medium | Start with domain expertise, iterate with data |
| Performance impact | Low | Low | Minimal computation added, optimize if needed |
| Users expect perfection | Medium | Medium | Set expectations, explain subjectivity |

### 8.2 Key Challenges

#### Challenge 1: Calibrating Without Ground Truth

**Problem**: No objective way to know if "V5 feels like V4 to slab climbers" is accurate

**Mitigation**:

- Start with climbing domain knowledge and expert input
- Collect user feedback: "Does this personalized grade feel accurate?"
- Iteratively adjust based on aggregate feedback
- A/B test different adjustment magnitudes
- Provide strength slider so users can tune sensitivity

#### Challenge 2: User Self-Assessment Bias

**Problem**: Users may overestimate their strengths or choose personas aspirationally

**Mitigation**:

- Provide detailed persona descriptions with examples
- Create persona quiz/assessment tool
- Suggest persona based on feedback patterns
- Allow users to try different personas and compare
- Emphasize "preference" over "absolute strength"

#### Challenge 3: Route Classification Edge Cases

**Problem**: Some routes don't fit neatly into categories

**Mitigation**:

- Use fuzzy boundaries, not hard thresholds
- Apply partial adjustments for borderline cases
- Default to conservative (smaller) adjustments when uncertain
- Log edge cases for manual review

#### Challenge 4: Explaining Grade Differences

**Problem**: Users may not understand why personalized grade differs

**Mitigation**:

- Provide clear explanation text: "This route has [characteristics] which match your [persona] strengths"
- Visual indicators (icons) for route characteristics
- Tooltip explanations for adjustments
- Link to persona description and documentation

#### Challenge 5: Maintenance Burden

**Problem**: 7 personas × multiple route types = many adjustment combinations

**Mitigation**:

- Use structured configuration (YAML) for easy updates
- Build adjustment testing framework
- Document adjustment rationale clearly
- Version persona profiles for tracking changes
- Consider ML-based adjustment learning (future)

### 8.3 Fallback Strategies

**If persona feature underperforms**:

1. Reduce adjustment magnitudes (make it subtle)
2. Limit to primary persona only (disable multi-persona)
3. Add "confidence" indicator for personalized grades
4. Make feature opt-in instead of default
5. Gather more calibration data before expanding
6. Worst case: Deprecate feature gracefully

**Circuit Breakers**:

```python
# In configuration
personas:
  enabled: true
  max_adjustment_magnitude: 0.35  # Cap at ±35%
  min_confidence_threshold: 0.4   # Don't personalize if base confidence low
  emergency_disable: false        # Kill switch
```

---

## 9. User Experience Considerations

### 9.1 UX Principles

**Principle 1: Transparency**

- Always show both base and personalized grades
- Clearly label which is which
- Explain why they differ

**Principle 2: Simplicity**

- Default to balanced persona (no personalization)
- Simple selection interface
- Hide advanced options (multi-persona) until requested

**Principle 3: Education**

- Persona descriptions with examples
- Tooltips explaining adjustments
- Help documentation

**Principle 4: User Control**

- Easy to change persona
- Strength level slider
- Toggle personalization on/off
- See how different personas would rate the route

### 9.2 UI Mockup Concepts

**Upload Form - Persona Selection**:

```text
┌─────────────────────────────────────────────────┐
│  Upload Climbing Route Image                   │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Choose File] route_photo.jpg                  │
│                                                 │
│  Wall Angle: [Moderate Overhang ▼]             │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │ 🎯 Your Climbing Style (Optional)         │ │
│  ├───────────────────────────────────────────┤ │
│  │ Select your primary strength to get a     │ │
│  │ personalized difficulty prediction:       │ │
│  │                                            │ │
│  │ Primary Persona:                           │ │
│  │ [Balanced (Default) ▼]                     │ │
│  │   • 🦶 Slab Specialist                     │ │
│  │   • 💪 Power/Campus Climber                │ │
│  │   • 🧠 Technical/Beta Reader                │ │
│  │   • 🤏 Crimp Specialist                     │ │
│  │   • 🤸 Flexibility Specialist               │ │
│  │   • ⏱️  Endurance Specialist                │ │
│  │   • ⚖️  Balanced/All-Arounder               │ │
│  │                                            │ │
│  │ Personalization Strength:                  │ │
│  │ Light ◯──●──◯ Strong                       │ │
│  │        Medium                               │ │
│  │                                            │ │
│  │ [?] What does this mean?                   │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  [Analyze Route]                                │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Results Display - Personalized Grade**:

```text
┌─────────────────────────────────────────────────┐
│  Analysis Results                               │
├─────────────────────────────────────────────────┤
│                                                 │
│  🎯 Grade Prediction                            │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │  Standard Grade:        V8              │   │
│  │  Your Grade (💪 Power): V6              │   │
│  │                                          │   │
│  │  ⚡ This route feels easier for you!     │   │
│  │                                          │   │
│  │  Why? This route has:                    │   │
│  │  • Steep overhang (your strength! 💪)    │   │
│  │  • Wide spacing between holds           │   │
│  │  • Powerful moves                        │   │
│  │                                          │   │
│  │  Confidence: ████████░░ 78%             │   │
│  │                                          │   │
│  │  [ Show Details ] [ Try Different Style ]│   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 9.3 Persona Selection Helper

**Persona Quiz** (optional feature):

```text
┌─────────────────────────────────────────────────┐
│  🎯 Find Your Climbing Style                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Answer a few questions to find your persona:   │
│                                                 │
│  1. What type of holds do you prefer?           │
│     ◯ Small crimps and edges                    │
│     ◯ Big jugs and volumes                      │
│     ◯ Slopers                                   │
│     ◯ I adapt to whatever is available          │
│                                                 │
│  2. What wall angle feels most comfortable?     │
│     ◯ Slab (leaning back)                       │
│     ◯ Vertical                                  │
│     ◯ Overhang (leaning forward)                │
│     ◯ All angles equally                        │
│                                                 │
│  3. What's your preferred movement style?       │
│     ◯ Static, controlled, precise               │
│     ◯ Dynamic, powerful, athletic               │
│     ◯ Creative, using flexibility               │
│     ◯ Efficient, technical sequences            │
│                                                 │
│  4. What's your biggest strength?               │
│     ◯ Balance and footwork                      │
│     ◯ Upper body power                          │
│     ◯ Finger strength                           │
│     ◯ Flexibility and mobility                  │
│     ◯ Problem-solving and beta reading          │
│     ◯ Endurance and recovery                    │
│                                                 │
│  [Get My Persona]                               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 10. Future Enhancements

### 10.1 Machine Learning Personalization

**Concept**: Learn user preferences from feedback instead of manual persona selection

**Approach**:

1. Track user feedback on routes
2. Identify patterns: which routes feel easier/harder than predicted
3. Train ML model to predict personal difficulty adjustment
4. Apply learned adjustments automatically

**Benefits**:

- No manual persona selection needed
- More accurate, individualized predictions
- Adapts as climber improves

**Challenges**:

- Requires significant feedback data per user
- Cold start problem for new users
- More complex to implement and explain

**Timeline**: Post-Phase 1.5, after collecting sufficient feedback data

### 10.2 Persona Evolution Tracking

**Concept**: Track how user's persona changes over time as they train

**Features**:

- Persona history graph
- "Improvement areas" suggestions
- Training recommendations based on weaknesses
- Celebration of persona milestones

**Example**:

```text
Your Climbing Journey:
Jan 2026: Balanced → Apr 2026: Slab Specialist → Aug 2026: Technical + Slab

You've improved your overhang climbing! Consider updating your persona.
```

### 10.3 Community Consensus Personas

**Concept**: Aggregate feedback from many users to refine persona adjustments

**Approach**:

1. Collect persona selections and feedback
2. Analyze which personas agree/disagree with standard grades
3. Use consensus to calibrate adjustment values
4. Identify new personas from clustering

**Benefits**:

- Data-driven calibration
- Discover new persona patterns
- Community validation

### 10.4 Route-Specific Persona Recommendations

**Concept**: Suggest which persona would find each route easiest

**UI Example**:

```text
This V7 route is:
• Easiest for: 💪 Power Climbers (feels like V5)
• Hardest for: 🦶 Slab Specialists (feels like V8)
• Standard for: ⚖️ Balanced Climbers (V7)
```

**Use Cases**:

- Help climbers find routes matching their style
- Identify weaknesses to train
- Route setting insights

---

## 11. Validation & Testing Strategy

### 11.1 Testing Approach

#### Unit Tests

**Persona Loading & Validation**:

```python
def test_load_personas_from_config():
    """Test persona definitions load correctly."""
    personas = load_persona_profiles()
    assert len(personas) == 7
    assert 'slab_specialist' in personas
    assert personas['slab_specialist']['name'] == "Slab Specialist"

def test_persona_adjustment_structure():
    """Test adjustment values are valid."""
    personas = load_persona_profiles()
    for name, profile in personas.items():
        for route_type, adjustments in profile['adjustments'].items():
            for factor, multiplier in adjustments.items():
                assert 0.5 <= multiplier <= 1.5, f"Invalid multiplier {multiplier}"
```

**Route Classification**:

```python
def test_classify_slab_route():
    """Test slab route classification."""
    features = {'wall_incline': 'slab', 'hold_types': {...}}
    chars = classify_route_characteristics(features)
    assert chars['wall_angle_category'] == 'slab'

def test_classify_crimp_route():
    """Test crimp-heavy route detection."""
    features = {'hold_types': {'crimp': 8, 'jug': 2}}
    chars = classify_route_characteristics(features)
    assert chars['hold_type_dominant'] == 'crimp'
```

**Adjustment Application**:

```python
def test_slab_specialist_on_slab():
    """Test slab specialist gets advantage on slab routes."""
    route_chars = {'wall_angle_category': 'slab'}
    adjustments = get_persona_adjustments('slab_specialist', route_chars)
    assert adjustments['wall_incline'] < 1.0  # Easier
    assert adjustments['wall_incline'] == 0.65

def test_power_climber_on_overhang():
    """Test power climber gets advantage on overhangs."""
    route_chars = {'wall_angle_category': 'overhang', 'distance_category': 'wide'}
    adjustments = get_persona_adjustments('power_climber', route_chars)
    assert adjustments['wall_incline'] < 1.0
    assert adjustments['distance'] < 1.0
```

#### Integration Tests

**Full Prediction Pipeline**:

```python
def test_personalized_prediction_slab():
    """Test full pipeline with slab persona."""
    features = create_slab_route_features()
    persona = {'persona': 'slab_specialist', 'strength': 'medium'}
    
    base_grade, _, _ = predict_grade_v2(features, ...)
    pers_grade, _, breakdown = predict_grade_v2_personalized(features, ..., persona)
    
    # Slab specialist should find slab routes easier
    assert grade_to_numeric(pers_grade) <= grade_to_numeric(base_grade)
    assert breakdown['persona_applied'] == 'slab_specialist'
```

**Multi-Persona Blending**:

```python
def test_multi_persona_blend():
    """Test primary + secondary persona combination."""
    persona = {
        'primary': {'persona': 'technical_climber', 'weight': 0.7},
        'secondary': {'persona': 'crimp_specialist', 'weight': 0.3}
    }
    
    adjustments = get_combined_adjustments(persona, route_chars)
    # Should be weighted blend
    assert 0.5 < adjustments['hold_difficulty'] < 1.0
```

#### Validation Tests

**Sanity Checks**:

```python
def test_adjustments_bounded():
    """Ensure all adjustments stay within reasonable bounds."""
    for persona in ALL_PERSONAS:
        for route_type in ALL_ROUTE_TYPES:
            adjustments = get_persona_adjustments(persona, route_type)
            for factor, mult in adjustments.items():
                assert 0.5 <= mult <= 1.5, f"Unbounded adjustment: {mult}"

def test_grade_shift_reasonable():
    """Ensure personalized grades don't shift too drastically."""
    # Generate 100 random routes
    for _ in range(100):
        features = generate_random_route()
        base_grade = predict_grade_v2(features, ...)[0]
        
        for persona in ALL_PERSONAS:
            pers_grade = predict_grade_v2_personalized(
                features, ..., {'persona': persona}
            )[0]
            
            diff = abs(grade_to_numeric(pers_grade) - grade_to_numeric(base_grade))
            assert diff <= 3, f"Grade shift too large: {diff} grades"
```

### 11.2 Calibration Strategy

#### Phase 1: Domain Knowledge Baseline

**Week 1-2**: Set initial adjustment values based on:

- Climbing coaching expertise
- Training literature
- Consensus from experienced climbers
- Conservative values (±15-20% typical)

#### Phase 2: Internal Testing

**Week 3-4**: Test with known routes:

- Team members rate routes with their actual personas
- Compare personalized predictions to subjective experience
- Adjust values based on systematic patterns

#### Phase 3: Beta User Calibration

**Week 5-6**: Expand testing:

- Recruit 20-30 beta users with diverse personas
- Collect structured feedback on 5+ routes each
- Statistical analysis of adjustment accuracy
- Refine adjustment values

#### Phase 4: Continuous Improvement

**Ongoing**: Monitor and adjust:

- Collect feedback from all users
- Aggregate persona satisfaction scores
- A/B test adjustment variations
- Quarterly calibration reviews

### 11.3 Acceptance Criteria

**Minimum Viability**:

- [ ] All 7 personas implemented and tested
- [ ] Route classification accuracy >80%
- [ ] Adjustment values within 0.65-1.35 range
- [ ] Personalized grades differ from base by ≤3 grades
- [ ] User feedback satisfaction >3.0/5.0
- [ ] No critical bugs
- [ ] Documentation complete

**Success Targets**:

- [ ] >40% of users select a persona
- [ ] User feedback satisfaction >3.5/5.0
- [ ] Personalized grades feel accurate >70% of time (user survey)
- [ ] Grade shifts feel appropriate >80% of time
- [ ] Feature usage grows month-over-month

---

## 12. Recommendations & Next Steps

### 12.1 Summary of Recommendations

✅ **PROCEED with persona personalization as Phase 1.5**

**Key Recommendations**:

1. **Start Simple**: Single persona selection only in Phase 1
   - Implement all 7 personas
   - Use medium strength default
   - Add multi-persona support later

2. **Conservative Adjustments**: Begin with modest adjustment ranges
   - Typical: ±15-20% (multipliers 0.80-1.20)
   - Maximum: ±35% (multipliers 0.65-1.35)
   - Allow strength scaling for user control

3. **Hybrid Data Model**: Store in both UserProfile and Analysis
   - UserProfile for default preference
   - Analysis for historical tracking
   - Allow per-analysis override

4. **Transparent UX**: Always show both grades
   - Label clearly: "Standard: V8, Your Grade: V6"
   - Explain why they differ
   - Provide toggle to disable personalization

5. **Iterative Calibration**: Start with domain knowledge, refine with data
   - Use climbing expertise for initial values
   - Collect user feedback continuously
   - A/B test adjustment magnitudes
   - Quarterly calibration reviews

6. **Graceful Degradation**: Build safety mechanisms
   - Feature flag for quick disable
   - Confidence thresholds
   - Bounded adjustments
   - Fallback to standard grade

### 12.2 Implementation Priority

**High Priority (Must Have)**:

1. All 7 persona definitions
2. Route classification system
3. Adjustment engine for single persona
4. Basic UI integration
5. Testing framework

**Medium Priority (Should Have)**:

6. Strength level scaling
7. Persona quiz/helper
8. Detailed grade explanations
9. UserProfile model
10. Calibration tools

**Low Priority (Nice to Have)**:

11. Multi-persona support
12. Persona evolution tracking
13. ML-based personalization
14. Community consensus features

### 12.3 Decision Points

**Decision 1: When to implement?**
→ **Recommendation**: After Phase 1 deployment and initial validation (2-3 months post-Phase 1)

**Decision 2: Single or multi-persona?**
→ **Recommendation**: Start with single, add multi later if demand exists

**Decision 3: User profiles or per-analysis?**
→ **Recommendation**: Hybrid approach - both for flexibility

**Decision 4: Adjustment magnitude?**
→ **Recommendation**: Conservative start (±15-20%), expand based on feedback

**Decision 5: Feature flag strategy?**
→ **Recommendation**: Gradual rollout with 10% → 50% → 100% over 4 weeks

### 12.4 Success Criteria for Go/No-Go

**PROCEED with implementation if**:

- ✅ Phase 1 algorithm validated and stable
- ✅ User feedback system operational
- ✅ Development resources available
- ✅ At least 100 analyses in database for testing
- ✅ Climbing domain expertise accessible for calibration

**NO-GO or delay if**:

- ❌ Phase 1 accuracy below 60%
- ❌ Major bugs in core algorithm
- ❌ Insufficient user adoption of base system
- ❌ Limited development resources
- ❌ No path to calibration/validation

### 12.5 Next Steps

**Immediate Actions**:

1. Review this analysis with team/stakeholders
2. Decide on implementation timeline
3. Validate persona definitions with climbing experts
4. Create detailed UI mockups
5. Set up testing framework

**Before Implementation**:

1. Ensure Phase 1 is deployed and validated
2. Collect baseline user feedback
3. Recruit beta testers representing different personas
4. Define success metrics and monitoring
5. Create feature flag configuration

**During Implementation**:

1. Follow 8-week implementation timeline
2. Weekly progress check-ins
3. Continuous testing and validation
4. User feedback collection
5. Iterative calibration

**Post-Launch**:

1. Monitor adoption and feedback
2. Weekly calibration reviews
3. Monthly adjustment refinements
4. Plan for Phase 2 (multi-persona, ML)
5. Document learnings

---

## Appendix A: Configuration Examples

### Example Persona Profile (Complete)

```yaml
slab_specialist:
  name: "Slab Specialist"
  code: "slab_specialist"
  icon: "🦶"
  description: "Excels at balance, footwork, and friction climbing on low-angle walls"
  long_description: |
    Slab specialists have exceptional balance and precise footwork. They excel
    at delicate, technical climbing on low-angle walls where body positioning
    and friction are key. However, they may struggle on steep overhangs that
    require significant upper body strength.
  
  strengths:
    - "Balance and body positioning"
    - "Precise footwork on small edges"
    - "Friction climbing technique"
    - "Static, controlled movement"
    - "Mental composure on delicate sequences"
  
  weaknesses:
    - "Steep overhangs (>105°)"
    - "Upper body power moves"
    - "Dynamic movements"
    - "Campus-style climbing"
    - "Compression problems"
  
  # Adjustment multipliers for different route types
  adjustments:
    slab_routes:
      condition:
        wall_angle_category: "slab"
      factors:
        wall_incline: 0.65    # 35% easier
        distance: 0.85        # 15% easier
        hold_density: 0.90    # 10% easier
        hold_difficulty: 1.0  # Neutral
      multipliers:
        transition: 1.0       # Neutral
        variability: 1.0      # Neutral
    
    overhang_routes:
      condition:
        wall_angle_category: ["overhang", "steep"]
      factors:
        wall_incline: 1.35    # 35% harder
        distance: 1.15        # 15% harder
        hold_difficulty: 1.10 # 10% harder
        hold_density: 1.05    # 5% harder
      multipliers:
        transition: 1.10      # 10% more affected
        variability: 1.05     # 5% more affected
    
    vertical_routes:
      condition:
        wall_angle_category: "vertical"
      factors:
        wall_incline: 1.0     # Neutral
        distance: 0.95        # 5% easier
        hold_difficulty: 1.0  # Neutral
        hold_density: 1.0     # Neutral
  
  # Examples for user education
  examples:
    - route: "Delicate vertical face with small footholds"
      base_grade: "V5"
      personalized_grade: "V4"
      explanation: "Your balance and footwork skills give you an advantage"
    
    - route: "Steep overhang with powerful moves"
      base_grade: "V5"
      personalized_grade: "V7"
      explanation: "This route requires upper body power, which isn't your strength"
  
  # Training recommendations
  training_recommendations:
    maintain:
      - "Continue practicing slab climbing"
      - "Refine footwork precision"
      - "Build mental focus for delicate sequences"
    improve:
      - "Develop upper body strength for overhangs"
      - "Practice dynamic movements"
      - "Train on steep terrain 1-2x per week"
```

---

## Appendix B: Glossary

**Persona**: A climbing style archetype representing specific strengths and weaknesses

**Adjustment Multiplier**: A factor (0.5-1.5) applied to base scores to reflect persona-specific difficulty perception

**Route Characteristic**: Detected features of a route (wall angle, hold types, spacing) used to determine which adjustments apply

**Personalization Strength**: User-selectable intensity of persona adjustments (light/medium/strong)

**Base Grade**: Standard grade prediction without persona adjustments

**Personalized Grade**: Grade prediction adjusted for user's climbing style persona

**Factor Score**: Individual component scores (hold difficulty, density, distance, incline) in the grade prediction algorithm

**Complexity Multiplier**: Amplification factors for route complexity (transitions, variability)

**Hybrid Persona**: Combination of multiple personas with weighted blending

**Balanced Persona**: Default persona with neutral adjustments (no personalization)

---

## Document Metadata

**Version**: 1.0
**Created**: 2026-01-04
**Author**: Roo (Architect Mode)
**Status**: Analysis Complete - Awaiting Review
**Related Documents**:

- [`grade_prediction_algorithm.md`](grade_prediction_algorithm.md)
- Implementation to be added to main plan upon approval

**Approval Required From**:

- Technical lead (feasibility review)
- Climbing domain expert (persona validation)
- Product owner (prioritization decision)
- UX designer (user experience review)

**Next Action**: Review with stakeholders and decide on Phase 1.5 implementation timeline
