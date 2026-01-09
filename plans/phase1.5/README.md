# Phase 1.5: Persona-Based Grade Personalization

## Overview

Phase 1.5 adds **optional persona-based personalization** to provide customized difficulty predictions based on individual climbing styles and strengths.

**Status**: Optional enhancement - implement **only after** Phase 1 validation

**Objective**: Allow climbers to see how route difficulty aligns with their personal strengths and weaknesses.

## Core Concept

Different climbers find different routes easier or harder based on their style:

- **Slab specialists** excel on low-angle technical routes but struggle on overhangs
- **Power climbers** dominate overhangs but find slabs challenging
- **Technical climbers** adapt well to varied styles through efficient beta reading

**Persona system** adjusts difficulty factors based on route characteristics and climber strengths.

## Seven Climbing Personas

1. **[Slab Specialist](persona_definitions.md#slab-specialist)** 🦶
   - Strengths: Balance, footwork, friction climbing
   - Weaknesses: Overhangs, power moves, dynamic climbing

2. **[Power/Campus Climber](persona_definitions.md#power-climber)** 💪
   - Strengths: Explosive strength, dynamic moves, steep terrain
   - Weaknesses: Slabs, technical footwork, endurance

3. **[Technical/Beta Reader](persona_definitions.md#technical-climber)** 🧠
   - Strengths: Route reading, efficient sequences, adaptability
   - Weaknesses: Pure strength problems, lack of specialization

4. **[Crimp Specialist](persona_definitions.md#crimp-specialist)** 🤏
   - Strengths: Finger strength on small holds, static pulling
   - Weaknesses: Slopers, open-hand holds, dynamic moves

5. **[Flexibility Specialist](persona_definitions.md#flexibility-specialist)** 🤸
   - Strengths: High steps, compression, creative body positions
   - Weaknesses: Crimp strength, standard beta, power moves

6. **[Endurance Specialist](persona_definitions.md#endurance-specialist)** ⏱️
   - Strengths: Sustained climbing, recovery, mental fortitude
   - Weaknesses: Short powerful problems, peak power moves

7. **[Balanced All-Arounder](persona_definitions.md#balanced)** ⚖️
   - Strengths: Versatility, consistency across styles
   - Weaknesses: Lacks specialization advantages

## How Personas Work

### Route Classification

Routes are classified by characteristics:

- Wall angle: Slab, vertical, overhang, steep
- Hold types: Crimp-heavy, sloper-heavy, mixed
- Spacing: Close, moderate, wide
- Complexity: Low, medium, high

### Persona Adjustments

Each persona has adjustment multipliers for different route types:

**Example: Slab Specialist on Slab Route**

- Wall incline factor: ×0.65 (35% easier)
- Distance factor: ×0.85 (15% easier - better balance)
- Result: V5 standard → V4 personalized

**Example: Power Climber on Overhang with Wide Spacing**

- Wall incline factor: ×0.75 (25% easier)
- Distance factor: ×0.70 (30% easier - powerful reaches)
- Result: V8 standard → V6 personalized

**IMPORTANT**: Adjustment values are **starting points** requiring calibration with user feedback.

## Prerequisites for Implementation

**DO NOT implement Phase 1.5 until**:

- ✅ Phase 1 deployed and validated
- ✅ Baseline accuracy ≥60% within ±1 grade
- ✅ User feedback system operational
- ✅ At least 100 route analyses completed
- ✅ Resources available for development and calibration

## User Experience

### Persona Selection

Users optionally select their persona during route upload or in profile settings:

```text
┌──────────────────────────────────────┐
│ Your Climbing Style (Optional)       │
├──────────────────────────────────────┤
│ Select your primary strength:        │
│ ┌──────────────────────────────────┐ │
│ │ ⚖️ Balanced (Default)          ▼ │ │
│ └──────────────────────────────────┘ │
│   • 🦶 Slab Specialist              │
│   • 💪 Power/Campus Climber          │
│   • 🧠 Technical/Beta Reader          │
│   • 🤏 Crimp Specialist               │
│   • 🤸 Flexibility Specialist         │
│   • ⏱️ Endurance Specialist           │
│   • ⚖️ Balanced All-Arounder          │
└──────────────────────────────────────┘
```

### Dual-Grade Display

Results show both standard and personalized grades:

```text
┌──────────────────────────────────────┐
│ Grade Prediction                     │
├──────────────────────────────────────┤
│ Standard Grade:        V8            │
│ Your Grade (💪 Power): V6            │
│                                      │
│ ⚡ This route feels easier for you!  │
│                                      │
│ Why? This route has:                 │
│ • Steep overhang (your strength!)    │
│ • Wide spacing between holds         │
│ • Powerful dynamic moves             │
└──────────────────────────────────────┘
```

## Implementation Approach

See [`implementation_approach.md`](implementation_approach.md) for technical details.

### High-Level Steps

1. **Define Personas** (Week 1-2)
   - Finalize 7 persona definitions in config
   - Define adjustment multipliers (starting values)
   - Create persona selection UI

2. **Route Classification** (Week 2-3)
   - Implement route characteristic detection
   - Classify routes by wall angle, hold types, spacing
   - Test classification accuracy

3. **Adjustment Engine** (Week 3-5)
   - Apply persona-specific adjustments to factor scores
   - Provide strength level scaling (light/medium/strong)
   - Generate explanation text

4. **UI Integration** (Week 5-6)
   - Add persona selection to upload form
   - Display dual grades (standard + personalized)
   - Show explanation for grade differences

5. **Calibration** (Week 6-8)
   - Collect user feedback on personalized grades
   - Adjust multipliers based on feedback
   - Iterate until satisfaction targets met

## Success Criteria

**Adoption**:

- Target: >40% of users select a persona
- Track: Persona distribution and usage patterns

**Satisfaction**:

- Target: >3.5/5.0 user satisfaction with personalized grades
- Track: Feedback surveys, agreement rates

**Accuracy**:

- Target: >70% report personalized grades "feel accurate"
- Track: User feedback on personalized predictions

## Key Design Principles

### 1. Non-Destructive

- Personas layer **on top of** Phase 1 algorithm
- Can be toggled on/off without affecting base predictions
- Always show both standard and personalized grades

### 2. Conservative Adjustments

- Start with modest adjustment ranges (±15-20%)
- Maximum adjustments capped at ±35%
- Avoid extreme personalization that feels unrealistic

### 3. Transparent

- Clearly explain why grades differ
- Show route characteristics that trigger adjustments
- Allow users to understand personalization logic

### 4. Iterative Calibration

- Adjustment multipliers are **starting hypotheses**
- Require extensive user feedback for validation
- Iterate based on persona-specific satisfaction data

## Risks and Mitigation

**Risk: Users misjudge their persona**

- **Mitigation**: Provide detailed descriptions, examples, optional quiz

**Risk: Adjustments feel inaccurate**

- **Mitigation**: Allow strength level scaling, collect feedback, iterate

**Risk: Feature adds confusion**

- **Mitigation**: Make optional (default: balanced), clear UI/UX, educational content

**Risk: Insufficient calibration data**

- **Mitigation**: Start with climbing domain expertise, group similar personas initially

## Relationship to Other Phases

**Phase 1**: Personas build on Phase 1 factor scores
**Phase 2**: Video analysis can validate persona adjustments (e.g., do power climbers actually perform better on overhangs?)

## Summary

Phase 1.5 provides optional personalized grade predictions based on climbing style:

- ✅ **7 distinct personas** covering major climbing styles
- ✅ **Route-specific adjustments** based on characteristics
- ✅ **Dual-grade display** (standard + personalized)
- ✅ **Optional feature** that doesn't interfere with base predictions

**Timeline**: Implement only after Phase 1 proves successful

**Outcome**: Enhanced user engagement and more relevant difficulty predictions for diverse climbing styles.

**Next Steps**:

1. Review [persona definitions](persona_definitions.md)
2. Read [implementation approach](implementation_approach.md)
3. Only proceed if Phase 1 prerequisites are met
