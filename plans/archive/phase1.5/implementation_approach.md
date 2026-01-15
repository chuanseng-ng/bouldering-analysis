# Phase 1.5 Implementation Approach

## Overview

This document provides technical guidance for implementing the persona-based personalization system.

**Prerequisites**: Phase 1 must be deployed, validated, and achieving ≥60% accuracy before starting Phase 1.5.

## Implementation Stages

### Stage 1: Persona Configuration (Week 1-2)

**Define personas in `user_config.yaml`:**

```yaml
personas:
  enabled: false  # Feature flag
  default: "balanced"
  strength_levels: ["light", "medium", "strong"]

  profiles:
    slab_specialist:
      name: "Slab Specialist"
      icon: "🦶"
      description: "Excels at balance, footwork, and friction climbing on low-angle walls"
      strengths: ["balance", "footwork", "friction"]
      weaknesses: ["overhang", "power", "dynamic"]

      adjustments:
        slab_routes:
          condition: {wall_angle_category: "slab"}
          factors:
            wall_incline: [0.65, 0.75]  # Range: conservative to aggressive
            distance: [0.85, 0.90]
            hold_density: [0.90, 0.95]

        overhang_routes:
          condition: {wall_angle_category: ["overhang", "steep"]}
          factors:
            wall_incline: [1.25, 1.35]
            distance: [1.10, 1.15]
            hold_difficulty: [1.05, 1.10]

    # ... other personas ...
```

**Add database field:**

```python
# Add to Analysis model
user_persona_applied = Column(JSON, nullable=True)
# Format: {"persona": "power_climber", "strength": "medium"}
```

### Stage 2: Route Classification (Week 2-3)

**Implement route characteristic detection:**

```python
def classify_route_characteristics(features: dict, wall_angle: str) -> dict:
    """
    Classify route based on detected features.

    Returns dict with classification for persona matching.
    """
    # Classify wall angle
    wall_angle_category = wall_angle  # Already classified by user input

    # Classify dominant hold type
    hold_types = features.get('hold_types', {})
    total_handholds = sum(hold_types.get(ht, 0)
                          for ht in ['crimp', 'jug', 'sloper', 'pinch', 'pocket'])

    if total_handholds > 0:
        crimp_ratio = hold_types.get('crimp', 0) / total_handholds
        sloper_ratio = hold_types.get('sloper', 0) / total_handholds

        if crimp_ratio > 0.5:
            hold_type_dominant = 'crimp'
        elif sloper_ratio > 0.4:
            hold_type_dominant = 'sloper'
        else:
            hold_type_dominant = 'mixed'
    else:
        hold_type_dominant = 'mixed'

    # Classify distance category
    distance_metrics = features.get('distance_metrics', {})
    avg_distance = distance_metrics.get('normalized_avg', 0.2)

    if avg_distance < 0.20:
        distance_category = 'close'
    elif avg_distance < 0.35:
        distance_category = 'moderate'
    else:
        distance_category = 'wide'

    # Classify hold count
    total_holds = features.get('total_holds', 10)
    if total_holds < 8:
        hold_count_category = 'few'
    elif total_holds < 13:
        hold_count_category = 'moderate'
    else:
        hold_count_category = 'many'

    return {
        'wall_angle_category': wall_angle_category,
        'hold_type_dominant': hold_type_dominant,
        'distance_category': distance_category,
        'hold_count_category': hold_count_category
    }
```

### Stage 3: Adjustment Engine (Week 3-5)

**Get persona adjustments based on route characteristics:**

```python
def get_persona_adjustments(
    persona_name: str,
    route_chars: dict,
    strength: str = 'medium',
    config: dict = None
) -> dict:
    """
    Get adjustment multipliers for factors based on persona and route.

    Args:
        persona_name: Selected persona (e.g., "slab_specialist")
        route_chars: Route characteristics from classification
        strength: Personalization strength ("light", "medium", "strong")
        config: Persona configuration

    Returns:
        Dict of factor adjustment multipliers
    """
    cfg = config or load_config()
    personas = cfg.get('personas', {}).get('profiles', {})

    if persona_name not in personas or persona_name == 'balanced':
        return {
            'hold_difficulty': 1.0,
            'hold_density': 1.0,
            'distance': 1.0,
            'wall_incline': 1.0
        }

    profile = personas[persona_name]
    adjustments = {
        'hold_difficulty': 1.0,
        'hold_density': 1.0,
        'distance': 1.0,
        'wall_incline': 1.0
    }

    # Match route characteristics to adjustment conditions
    for adjustment_key, adjustment_data in profile.get('adjustments', {}).items():
        condition = adjustment_data.get('condition', {})

        # Check if condition matches
        matches = True
        for key, value in condition.items():
            if isinstance(value, list):
                if route_chars.get(key) not in value:
                    matches = False
            else:
                if route_chars.get(key) != value:
                    matches = False

        if matches:
            # Apply adjustments
            factors = adjustment_data.get('factors', {})
            for factor, value_range in factors.items():
                # Use midpoint of range for "medium" strength
                if isinstance(value_range, list) and len(value_range) == 2:
                    midpoint = (value_range[0] + value_range[1]) / 2
                    adjustments[factor] = midpoint
                else:
                    adjustments[factor] = value_range

    # Apply strength scaling
    strength_scales = {'light': 0.5, 'medium': 1.0, 'strong': 1.5}
    scale = strength_scales.get(strength, 1.0)

    scaled_adjustments = {}
    for factor, multiplier in adjustments.items():
        deviation = multiplier - 1.0
        scaled_adjustments[factor] = 1.0 + (deviation * scale)

    return scaled_adjustments
```

**Apply to prediction:**

```python
def predict_grade_v2_personalized(
    features: dict,
    detected_holds: list,
    wall_incline: str = 'vertical',
    user_persona: dict = None,
    config: dict = None
) -> tuple[str, float, dict]:
    """
    Enhanced prediction with persona personalization.

    Args:
        features: Extracted features
        detected_holds: List of DetectedHold objects
        wall_incline: Wall angle category
        user_persona: {"persona": "power_climber", "strength": "medium"}
        config: Optional config override

    Returns:
        (personalized_grade, confidence, breakdown)
    """
    # Get base prediction from Phase 1
    base_grade, base_confidence, base_breakdown = predict_grade_v2(
        features, detected_holds, wall_incline, config=config
    )

    # If no persona or balanced, return base prediction
    if not user_persona or user_persona.get('persona') == 'balanced':
        return base_grade, base_confidence, base_breakdown

    # Classify route characteristics
    route_chars = classify_route_characteristics(features, wall_incline)

    # Get persona adjustments
    persona_name = user_persona.get('persona', 'balanced')
    strength = user_persona.get('strength', 'medium')

    adjustments = get_persona_adjustments(
        persona_name, route_chars, strength, config
    )

    # Apply adjustments to factor scores
    adjusted_factors = {}
    for factor_name, base_score in base_breakdown['factors'].items():
        adjustment = adjustments.get(factor_name, 1.0)
        adjusted_factors[factor_name] = base_score * adjustment

    # Recalculate combined score with adjusted factors
    weights = base_breakdown['weights']
    personalized_score = sum(
        adjusted_factors[f] * weights[f]
        for f in ['hold_difficulty', 'hold_density', 'distance', 'wall_incline']
    )

    # Map to grade
    personalized_grade = map_score_to_grade(personalized_score)

    # Adjust confidence (lower if grade differs significantly)
    grade_diff = abs(grade_to_numeric(personalized_grade) - grade_to_numeric(base_grade))
    confidence_penalty = min(grade_diff * 0.1, 0.3)
    personalized_confidence = base_confidence * (1 - confidence_penalty)

    # Breakdown
    breakdown = {
        **base_breakdown,
        'persona_applied': persona_name,
        'persona_strength': strength,
        'route_characteristics': route_chars,
        'adjustments': adjustments,
        'adjusted_factors': adjusted_factors,
        'base_grade': base_grade,
        'personalized_grade': personalized_grade,
        'personalized_score': personalized_score
    }

    return personalized_grade, personalized_confidence, breakdown
```

### Stage 4: UI Integration (Week 5-6)

**Add persona selection to upload form:**

```html
<!-- In upload form -->
<div class="persona-selection">
  <label for="persona">Your Climbing Style (Optional)</label>
  <select name="persona" id="persona">
    <option value="balanced">⚖️ Balanced (Default)</option>
    <option value="slab_specialist">🦶 Slab Specialist</option>
    <option value="power_climber">💪 Power/Campus Climber</option>
    <option value="technical_climber">🧠 Technical/Beta Reader</option>
    <option value="crimp_specialist">🤏 Crimp Specialist</option>
    <option value="flexibility_specialist">🤸 Flexibility Specialist</option>
    <option value="endurance_specialist">⏱️ Endurance Specialist</option>
  </select>

  <label for="persona_strength">Personalization Strength</label>
  <select name="persona_strength" id="persona_strength">
    <option value="light">Light</option>
    <option value="medium" selected>Medium</option>
    <option value="strong">Strong</option>
  </select>
</div>
```

**Display dual grades in results:**

```html
<!-- In results page -->
<div class="grade-prediction">
  <div class="standard-grade">
    <label>Standard Grade:</label>
    <span class="grade">V{{ base_grade }}</span>
  </div>

  {% if personalized_grade != base_grade %}
  <div class="personalized-grade">
    <label>Your Grade ({{ persona_icon }} {{ persona_name }}):</label>
    <span class="grade">V{{ personalized_grade }}</span>
  </div>

  <div class="grade-explanation">
    <strong>Why?</strong> This route has:
    <ul>
      {% for characteristic in route_characteristics %}
      <li>{{ characteristic }}</li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}
</div>
```

### Stage 5: Calibration (Week 6-8)

**Collect persona-specific feedback:**

```python
# Add to feedback model
class PersonaFeedback(Base):
    __tablename__ = 'persona_feedback'

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'))
    persona_applied = Column(String(50))
    persona_strength = Column(String(20))
    base_grade = Column(String(10))
    personalized_grade = Column(String(10))
    user_rating = Column(Integer)  # 1-5 scale
    feels_accurate = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Analyze and iterate:**

```sql
-- Per-persona satisfaction
SELECT persona_applied, AVG(user_rating) as avg_rating, COUNT(*) as count
FROM persona_feedback
GROUP BY persona_applied;

-- Accuracy by persona
SELECT persona_applied,
       SUM(CASE WHEN feels_accurate = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy_pct
FROM persona_feedback
GROUP BY persona_applied;
```

**Adjust configuration:**

- If slab specialists report grades too easy → Increase multiplier lower bounds
- If power climbers report grades too hard → Decrease multiplier upper bounds
- Iterate weekly until satisfaction targets met

## Deployment Strategy

### Gradual Rollout

**Week 1-2**: Internal testing (rollout: 0%)
**Week 3-4**: Beta users (rollout: 10%)
**Week 5-6**: Expanded testing (rollout: 50%)
**Week 7+**: Full rollout (rollout: 100%)

**Feature flag control:**

```yaml
personas:
  enabled: true
  rollout_percentage: 100  # 0-100
  show_base_grade: true  # Always show non-personalized grade
  default_strength: "medium"
```

## Success Metrics

**Track weekly:**

- Persona adoption rate (% of users selecting non-balanced)
- Persona distribution (which personas are popular)
- Average satisfaction rating per persona
- Accuracy ("feels accurate") per persona
- Grade shift distribution (how much personas change predictions)

**Target metrics**:

- Adoption: >40%
- Satisfaction: >3.5/5.0
- Accuracy: >70% "feels accurate"

## Edge Cases and Fallbacks

**Unknown persona name:**

- Default to "balanced" (no adjustments)

**Missing route characteristics:**

- Use default/neutral classifications
- Log for review

**Extreme adjustments (>3 grade shift):**

- Cap adjustments to prevent unrealistic personalization
- Log for calibration review

**Persona feedback collection failures:**

- Gracefully degrade, continue with feature
- Log errors for investigation

## Summary

Phase 1.5 implementation requires:

1. ✅ **Persona configuration** - Define 7 personas in config
2. ✅ **Route classification** - Detect route characteristics
3. ✅ **Adjustment engine** - Apply persona-specific multipliers
4. ✅ **UI integration** - Persona selection and dual-grade display
5. ✅ **Calibration** - Iterate based on user feedback

**Critical Success Factor**: Start with conservative adjustments, iterate based on persona-specific feedback data.

**Timeline**: 6-8 weeks after Phase 1 validation

**Next**: See [persona_definitions.md](persona_definitions.md) for detailed persona specifications.
