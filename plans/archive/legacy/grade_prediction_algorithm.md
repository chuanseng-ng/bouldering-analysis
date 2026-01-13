# Grade Prediction Algorithm - Overview & Implementation Plan

## Executive Summary

This document provides a high-level overview of the grade prediction algorithm implementation strategy. The system uses a **phased approach** to build a sophisticated climbing route difficulty prediction system that evolves from route-based analysis to video-based validation.

### System Objectives

1. **Predict climbing route difficulty** (V0-V12) using computer vision and climbing domain knowledge
2. **Provide accurate, explainable predictions** based on route characteristics
3. **Validate predictions** through multiple data sources
4. **Continuously improve** using user feedback and performance data

### Phased Implementation Strategy

The implementation is divided into distinct phases to manage complexity and validate each component before adding the next:

| Phase | Name | Status | Priority | Timeline |
|-------|------|--------|----------|----------|
| **Phase 1** | Route-Based Grade Prediction | Core Implementation | **HIGH** | 8-12 weeks |
| **Phase 1.5** | Persona-Based Personalization | Optional Enhancement | Medium | 8-9 weeks |
| **Phase 2** | Video Analysis Validation | Future Enhancement | **LOW** | 13-18 weeks |

---

## Phase Overview

### Phase 1: Route-Based Grade Prediction ⭐ **CORE IMPLEMENTATION**

**Status**: Primary focus - must be completed first

**Objective**: Replace the current simplified grade prediction with a sophisticated, multi-factor algorithm.

**Key Features**:

- Four primary difficulty factors: hold types/sizes, hold count, hold distances, wall incline
- Two complexity multipliers: wall angle transitions, hold type variability
- Weighted scoring model with empirically-derived weights
- Configurable thresholds and parameters
- Detailed score breakdowns for explainability

**Algorithm Approach**:

```text
Base Score = f(Hold Difficulty, Hold Density, Distance, Wall Incline)
Final Score = Base Score × Transition Multiplier × Variability Multiplier
V-Grade = map(Final Score, 0-12 range)
```

**Deliverables**:

- ✅ Multi-factor scoring algorithm
- ✅ Wall incline analysis
- ✅ Complexity multiplier system
- ✅ Comprehensive testing framework
- ✅ Configuration management
- ✅ User documentation

**📄 Full Documentation**: [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md)

---

### Phase 1.5: Persona-Based Personalization 🎯 **OPTIONAL ENHANCEMENT**

**Status**: Optional - implement after Phase 1 validation

**Objective**: Provide personalized grade predictions based on individual climbing styles and strengths.

**Key Features**:

- 7 climbing personas: Slab Specialist, Power Climber, Technical Climber, Crimp Specialist, Flexibility Specialist, Endurance Specialist, Balanced All-Arounder
- Persona-specific adjustments to difficulty factors
- Route characteristic classification
- Dual-grade display (standard + personalized)
- Strength level controls

**Example Use Case**:

```text
Standard Grade: V7
Your Grade (Power Climber): V5
Why? This route's steep overhang and wide spacing match your strengths
```

**Prerequisites**:

- ✅ Phase 1 deployed and validated
- ✅ Baseline accuracy ≥60% within ±1 grade
- ✅ User feedback system operational

**📄 Full Documentation**: See Phase 1.5 section in [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md)

**📄 Detailed Analysis**: [`persona_personalization_analysis.md`](persona_personalization_analysis.md)

---

### Phase 2: Video Analysis Validation 🎥 **FUTURE ENHANCEMENT**

**Status**: Lower priority - implement only after Phase 1 success

**Objective**: Add video-based performance analysis to cross-validate route-based predictions.

**Key Features**:

- Pose estimation from climbing videos
- Body mechanics analysis (angles, positions, movement patterns)
- Performance metrics (flow, pace, rest positions, struggle indicators)
- Independent video-based grade prediction
- Cross-validation between route and video predictions
- Discrepancy detection and logging

**Cross-Validation Logic**:

```text
Route Prediction (Phase 1) → V5
Video Prediction (Phase 2) → V6
Grade Difference: 1 grade → VALID ✅
Combined Confidence: High
```

**Prerequisites**:

- ✅ Phase 1 deployed to production
- ✅ Phase 1 accuracy ≥70% within ±1 grade
- ✅ Phase 1 algorithm stable (no major bugs)
- ✅ Development resources available
- ✅ Sample climbing videos collected

**📄 Full Documentation**: [`phase2_video_analysis_validation.md`](phase2_video_analysis_validation.md)

---

## Implementation Priority & Roadmap

### Current Focus: Phase 1 (HIGH PRIORITY)

**Why Phase 1 First?**

1. **Foundation**: Core algorithm must work before adding complexity
2. **Validation**: Need baseline accuracy to measure improvements
3. **User Value**: Immediate benefit from improved route-based predictions
4. **Lower Risk**: Fewer dependencies, simpler implementation
5. **Resource Efficient**: No video processing infrastructure needed

**Phase 1 Timeline** (8-12 weeks):

```text
Weeks 1-2:  Database schema updates (wall_incline, wall_segments)
Weeks 2-4:  Core algorithm implementation (4 factors + 2 multipliers)
Weeks 4-6:  UI integration (wall incline input, grade display)
Weeks 6-8:  Testing and calibration
Weeks 8-10: Integration and validation
Weeks 10-12: Deployment and monitoring
```

### Next: Phase 1.5 (MEDIUM PRIORITY)

**Timing**: 2-3 months **after** Phase 1 deployment

**Conditions**:

- Phase 1 accuracy validated (≥60% within ±1 grade)
- User feedback system operational
- At least 100 route analyses completed
- Resources available for 8-9 week development

### Future: Phase 2 (LOW PRIORITY)

**Timing**: 6+ months **after** Phase 1 deployment

**Conditions**:

- Phase 1 accuracy ≥70% within ±1 grade
- Algorithm stable and proven in production
- Video processing infrastructure available
- Sample climbing videos collected
- Resources available for 13-18 week development

---

## Success Metrics

### Phase 1 Success Criteria

**Accuracy**:

- ✅ Exact match: ≥60%
- ✅ Within ±1 grade: ≥80%
- ✅ No regressions from current system

**Performance**:

- ✅ Prediction time: <100ms per route
- ✅ No crashes on edge cases
- ✅ All unit tests passing

**User Satisfaction**:

- ✅ User feedback collection operational
- ✅ Clear explanations for predictions
- ✅ Configurable parameters working

### Phase 1.5 Success Criteria

**Adoption**:

- Target: >40% of users select a persona
- Track: Persona distribution and usage patterns

**Satisfaction**:

- Target: >3.5/5.0 user satisfaction with personalized grades
- Track: Feedback surveys, agreement rates

**Accuracy**:

- Target: >70% report personalized grades "feel accurate"
- Track: User feedback on personalized predictions

### Phase 2 Success Criteria

**Cross-Validation**:

- ✅ Route vs video agreement: ≥70% within ±1 grade
- ✅ Discrepancy detection functional
- ✅ Systematic bias identification working

**Performance**:

- ✅ Video processing time: <2 minutes per video
- ✅ Pose estimation accuracy: ≥85%
- ✅ Storage management operational

---

## Key Design Decisions

### Multi-Factor Scoring (Phase 1)

**Decision**: Use weighted combination of 4 factors + 2 multipliers
**Rationale**: Captures multiple aspects of difficulty; weights tunable based on feedback
**Alternative Considered**: Machine learning model (deferred - need training data)

### Two-Stage Scoring (Phase 1)

**Decision**: Base score (additive) → Apply multipliers (multiplicative)
**Rationale**: Complexity amplifies baseline difficulty exponentially, matching climber experience
**Alternative Considered**: All additive (rejected - doesn't capture non-linear difficulty)

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

## Technical Architecture

### Current Implementation

**Location**: [`src/main.py:707`](../src/main.py:707) - `predict_grade()` function

**Current Logic**:

- Simple hold count thresholds
- Basic difficulty multipliers for specific hold types
- No distance analysis
- No wall incline consideration
- Caps at V10

### Phase 1 Architecture

**New Module**: `src/grade_prediction.py`

**Components**:

```text
src/grade_prediction.py
├── Factor Analysis Functions
│   ├── analyze_hold_difficulty()
│   ├── analyze_hold_density()
│   ├── analyze_hold_distances()
│   └── analyze_wall_incline()
├── Complexity Multiplier Functions
│   ├── detect_wall_transitions()
│   ├── calculate_transition_multiplier()
│   ├── calculate_hold_type_entropy()
│   └── calculate_variability_multiplier()
├── Scoring Functions
│   ├── combine_scores()
│   ├── apply_multipliers()
│   └── map_score_to_grade()
└── Main Entry Point
    └── predict_grade_v2()
```

**Database Changes**:

```python
# Add to Analysis model
wall_incline = Column(String(20), default='vertical')
wall_segments = Column(JSON, nullable=True)
```

**Configuration** (`user_config.yaml`):

```yaml
grade_prediction:
  algorithm_version: "v2"
  weights: {hold_difficulty: 0.35, hold_density: 0.25, ...}
  complexity_multipliers: {transition: {...}, variability: {...}}
  wall_incline_multipliers: {slab: 0.65, vertical: 1.00, ...}
```

---

## Related Documentation

### Phase Documentation

- **Phase 1 Implementation Guide**: [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md) - Complete technical specification for route-based prediction
- **Phase 2 Implementation Guide**: [`phase2_video_analysis_validation.md`](phase2_video_analysis_validation.md) - Complete technical specification for video analysis
- **Persona Analysis**: [`persona_personalization_analysis.md`](persona_personalization_analysis.md) - Detailed persona system design

### Source Code

- **Main Application**: [`src/main.py`](../src/main.py) - Current implementation
- **Database Models**: [`src/models.py`](../src/models.py) - Analysis, DetectedHold, HoldType
- **Constants**: [`src/constants.py`](../src/constants.py) - Hold type definitions
- **Configuration**: [`src/cfg/user_config.yaml`](../src/cfg/user_config.yaml) - System configuration

### Migration Documentation

- **Database Migrations**: [`docs/migrations.md`](../docs/migrations.md) - Schema change procedures

---

## Getting Started

### For Developers Implementing Phase 1

1. **Read Phase 1 Documentation**: [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md)
2. **Review Current Code**: [`src/main.py:707`](../src/main.py:707)
3. **Plan Database Migration**: Add `wall_incline` and `wall_segments` fields
4. **Create Grade Prediction Module**: `src/grade_prediction.py`
5. **Implement Core Functions**: Start with hold difficulty and distance analysis
6. **Add Configuration**: Update `user_config.yaml`
7. **Write Tests**: Unit tests for each component
8. **Integrate with Main**: Update `analyze_image()` to use `predict_grade_v2()`
9. **UI Updates**: Add wall incline input form
10. **Deploy and Monitor**: Collect user feedback for calibration

### For Project Managers

1. **Phase 1 is the Priority**: Allocate resources to complete Phase 1 first
2. **Set Success Criteria**: Define accuracy thresholds before deployment
3. **Plan User Feedback**: Set up feedback collection mechanism
4. **Monitor Metrics**: Track accuracy, performance, user satisfaction
5. **Evaluate Phase 1.5**: Decide if persona system adds value after Phase 1 validation
6. **Defer Phase 2**: Only consider after Phase 1 proves successful (6+ months)

### For Stakeholders

**Current State**: Basic grade prediction using hold counts
**Phase 1 Goal**: Sophisticated multi-factor algorithm with wall angle support
**Timeline**: 8-12 weeks to production
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

**Timeline**: 6+ months after Phase 1 deployment
**Conditions**: Phase 1 accuracy ≥70%, algorithm stable, resources available

### How do we measure success?

**Metrics**:

- Prediction accuracy (exact match and ±1 grade)
- User feedback agreement rate
- System performance (prediction time)
- User satisfaction scores

### What if Phase 1 accuracy is low?

**Iteration**: Use user feedback to calibrate weights and thresholds. Consider collecting more ground truth data or adjusting factor formulas.

---

## Version History

| Version | Date | Changes | Author |
| ------- | ---- | ------- | ------ |
| 1.0 | 2026-01-04 | Split comprehensive plan into phased documents | System |
| 0.9 | 2026-01-03 | Added Phase 1.5 persona personalization | System |
| 0.8 | 2026-01-02 | Added complexity multipliers to Phase 1 | System |
| 0.7 | 2026-01-01 | Added wall incline factor to Phase 1 | System |
| 0.6 | 2025-12-31 | Added Phase 2 video analysis validation | System |
| 0.5 | 2025-12-30 | Initial comprehensive plan | System |

---

## Summary

This grade prediction system uses a **phased approach** to manage complexity and validate each component:

1. **Phase 1 (HIGH PRIORITY)**: Route-based prediction with 4 factors + 2 multipliers - **Start immediately**
2. **Phase 1.5 (MEDIUM PRIORITY)**: Persona-based personalization - **After Phase 1 validation**
3. **Phase 2 (LOW PRIORITY)**: Video analysis validation - **6+ months after Phase 1**

**Next Action**: Begin Phase 1 implementation with database schema updates and core algorithm development.

**Key Success Factor**: Complete and validate Phase 1 before proceeding to subsequent phases.

---

**For detailed technical specifications, see**:

- 📘 **Phase 1**: [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md)
- 📗 **Phase 2**: [`phase2_video_analysis_validation.md`](phase2_video_analysis_validation.md)
- 📙 **Personas**: [`persona_personalization_analysis.md`](persona_personalization_analysis.md)

