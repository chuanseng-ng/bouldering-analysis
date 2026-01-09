# Grade Prediction System - Overview

## Executive Summary

This system predicts climbing route difficulty (V0-V12) using computer vision and climbing domain knowledge through a **phased implementation approach**.

### System Objectives

1. **Predict climbing route difficulty** using multi-factor analysis
2. **Provide accurate, explainable predictions** based on route characteristics
3. **Validate predictions** through multiple data sources
4. **Continuously improve** using user feedback and performance data

### Phased Implementation Strategy

The implementation is divided into distinct phases to manage complexity and validate each component:

| Phase | Name | Status | Priority |
|-------|------|--------|----------|
| **Phase 1** | Route-Based Grade Prediction | Core Implementation | **HIGH** |
| **Phase 1.5** | Persona-Based Personalization | Optional Enhancement | Medium |
| **Phase 2** | Video Analysis Validation | Future Enhancement | Low |

## Phase Overview

### Phase 1: Route-Based Grade Prediction ⭐ **CORE IMPLEMENTATION**

**Objective**: Build a sophisticated, multi-factor algorithm to predict grade from route characteristics.

**Key Features**:
- Four primary difficulty factors: hold types/sizes, hold count, hold distances, wall incline
- Two complexity multipliers: wall angle transitions, hold type variability (add in refinement phase)
- Weighted scoring model with calibratable parameters
- Detailed score breakdowns for explainability

**Algorithm Approach**:
```text
Base Score = f(Hold Difficulty, Hold Density, Distance, Wall Incline)
Final Score = Base Score × Transition Multiplier × Variability Multiplier
V-Grade = map(Final Score, 0-12 range)
```

**📄 Full Documentation**: [`phase1/`](phase1/)

---

### Phase 1.5: Persona-Based Personalization 🎯 **OPTIONAL ENHANCEMENT**

**Objective**: Provide personalized grade predictions based on individual climbing styles and strengths.

**Key Features**:
- 7 climbing personas (Slab Specialist, Power Climber, Technical Climber, etc.)
- Persona-specific adjustments to difficulty factors
- Dual-grade display (standard + personalized)

**Prerequisites**:
- ✅ Phase 1 deployed and validated
- ✅ Baseline accuracy ≥60% within ±1 grade
- ✅ User feedback system operational

**📄 Full Documentation**: [`phase1.5/`](phase1.5/)

---

### Phase 2: Video Analysis Validation 🎥 **FUTURE ENHANCEMENT**

**Objective**: Add video-based performance analysis to cross-validate route-based predictions.

**Key Features**:
- Pose estimation from climbing videos
- Body mechanics analysis (angles, positions, movement patterns)
- Performance metrics (flow, pace, rest positions, struggle indicators)
- Cross-validation between route and video predictions

**Prerequisites**:
- ✅ Phase 1 deployed to production
- ✅ Phase 1 accuracy ≥70% within ±1 grade
- ✅ Development resources available
- ✅ Sample climbing videos collected

**📄 Full Documentation**: [`phase2/`](phase2/)

---

## Implementation Priority & Approach

### Current Focus: Phase 1 (HIGH PRIORITY)

**Why Phase 1 First?**
1. **Foundation**: Core algorithm must work before adding complexity
2. **Validation**: Need baseline accuracy to measure improvements
3. **User Value**: Immediate benefit from improved route-based predictions
4. **Lower Risk**: Fewer dependencies, simpler implementation
5. **Resource Efficient**: No video processing infrastructure needed

**Staged Implementation Approach:**
1. Implement basic 4-factor model first
2. Calibrate and validate with user feedback
3. Add complexity multipliers in refinement phase
4. Iterate based on real-world data

### Success Metrics

**Phase 1 Success Criteria:**
- ✅ Exact match: ≥60%
- ✅ Within ±1 grade: ≥80%
- ✅ Prediction time: <100ms per route
- ✅ Clear explanations for predictions

**Phase 1.5 Success Criteria:**
- Target: >40% of users select a persona
- Target: >70% report personalized grades "feel accurate"

**Phase 2 Success Criteria:**
- ✅ Route vs video agreement: ≥70% within ±1 grade
- ✅ Video processing time: <2 minutes per video

---

## Key Design Decisions

### Multi-Factor Scoring (Phase 1)

**Decision**: Use weighted combination of 4 factors + 2 multipliers
**Rationale**: Captures multiple aspects of difficulty; weights tunable based on feedback
**Alternative Considered**: Machine learning model (deferred - need training data)

**IMPORTANT**: Initial weight values should be treated as **starting points requiring empirical calibration**, not final values.

### Staged Complexity (Phase 1)

**Decision**: Implement basic 4-factor model first, add complexity multipliers later
**Rationale**: Prevents complexity from blocking initial development; allows faster iteration
**Alternative Considered**: All features at once (rejected - too complex to debug)

### Manual Wall Incline Input (Phase 1)

**Decision**: User provides wall angle via UI dropdown
**Rationale**: Simple, accurate, no computer vision needed for MVP
**Alternative Considered**: CV-based detection (deferred to future enhancement)

### Persona-Based Adjustments (Phase 1.5)

**Decision**: Layer on top of Phase 1 without modifying core
**Rationale**: Non-destructive, can be toggled on/off, validates Phase 1 first
**Alternative Considered**: Integrated from start (rejected - too complex initially)

### Video as Validation, Not Primary (Phase 2)

**Decision**: Video analysis validates route predictions, not replaces them
**Rationale**: Route analysis is faster, simpler; video adds validation layer
**Alternative Considered**: Video as primary (rejected - too resource intensive)

---

## Critical Design Principles

### 1. Iterative Calibration Over Prescriptive Values

All threshold values, multipliers, and weights in these specifications should be treated as **initial hypotheses** requiring:
- Empirical validation with real route data
- User feedback collection and analysis
- A/B testing and iterative refinement
- Continuous monitoring and adjustment

**Do NOT treat specification values as final** - they are calibration starting points.

### 2. Slanted Hold Consideration

Hold orientation (slant angle) significantly affects difficulty and **must** be considered:

**For Handholds:**
- Downward-slanting: Harder (requires more grip strength)
- Horizontal/Positive: Neutral difficulty
- Upward-slanting: Easier (more positive surface to grip)

**For Footholds:**
- Downward-slanting: Much harder (foot can slip off)
- Horizontal: Standard difficulty
- Upward-slanting: Easier (foot has positive platform)

This factor should be integrated into Factor 1 (Hold Analysis).

### 3. Complexity Added Incrementally

The complexity multipliers (wall transitions, hold variability) are valuable but should be:
- Added **after** basic 4-factor model is working
- Implemented in a refinement phase
- Not allowed to block initial development
- Validated independently before integration

### 4. Avoid Over-Engineering Initial Implementation

Start simple:
- Basic 4-factor model first
- Conservative calibration values
- Extensive logging for calibration
- User feedback collection from day 1
- Iterate based on real data

---

## Related Documentation

### Phase Documentation
- **Phase 1**: [`phase1/README.md`](phase1/README.md) - Core route-based prediction
- **Phase 1.5**: [`phase1.5/README.md`](phase1.5/README.md) - Persona personalization
- **Phase 2**: [`phase2/README.md`](phase2/README.md) - Video analysis validation

### Source Code
- **Main Application**: [`../src/main.py`](../src/main.py) - Current implementation
- **Database Models**: [`../src/models.py`](../src/models.py) - Analysis, DetectedHold, HoldType
- **Constants**: [`../src/constants.py`](../src/constants.py) - Hold type definitions
- **Configuration**: [`../src/cfg/user_config.yaml`](../src/cfg/user_config.yaml) - System configuration

---

## Getting Started

### For Developers Implementing Phase 1

1. **Read Phase 1 Documentation**: [`phase1/README.md`](phase1/README.md)
2. **Review Factor Specifications**: Understand each of the 4 factors
3. **Review Current Code**: [`../src/main.py:707`](../src/main.py:707)
4. **Plan Staged Implementation**: Basic 4-factor model → Calibration → Complexity multipliers
5. **Set Up Feedback Collection**: Essential for calibration

### For Project Managers

1. **Phase 1 is the Priority**: Focus resources on completing Phase 1 first
2. **Plan for Iteration**: Initial release will require calibration
3. **Set Up Feedback Mechanisms**: User feedback drives calibration
4. **Defer Phase 1.5 and Phase 2**: Only after Phase 1 proves successful

### For Stakeholders

**Current State**: Basic grade prediction using hold counts
**Phase 1 Goal**: Sophisticated multi-factor algorithm
**Success Metric**: ≥80% accuracy within ±1 grade
**User Benefit**: More accurate route difficulty predictions
**Future Enhancements**: Personalization (Phase 1.5), Video validation (Phase 2)

---

## FAQ

### Why not implement all phases at once?

**Risk Management**: Each phase adds complexity. Validating Phase 1 before Phase 2 reduces risk of building on faulty foundation.

### Can we skip Phase 1 and go straight to video analysis?

**No**: Video analysis requires known-good route predictions for cross-validation. Phase 1 must succeed first.

### Is Phase 1.5 required?

**No**: Persona personalization is optional. Evaluate after Phase 1 deployment based on user feedback and resources.

### When should we start Phase 2?

**Conditions**: Phase 1 accuracy ≥70%, algorithm stable, resources available for video processing infrastructure.

### What if Phase 1 accuracy is low?

**Iteration**: Use user feedback to calibrate weights and thresholds. Collect more ground truth data, adjust factor formulas, iterate.

---

## Summary

This grade prediction system uses a **phased approach** to manage complexity:

1. **Phase 1 (HIGH PRIORITY)**: Route-based prediction with 4 factors - **Start immediately**
2. **Phase 1.5 (MEDIUM PRIORITY)**: Persona-based personalization - **After Phase 1 validation**
3. **Phase 2 (LOW PRIORITY)**: Video analysis validation - **After Phase 1 proves successful**

**Next Action**: Begin Phase 1 implementation with basic 4-factor model.

**Key Success Factor**: Treat all values as calibration starting points, iterate based on user feedback.

