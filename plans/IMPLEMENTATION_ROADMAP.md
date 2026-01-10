# Implementation Roadmap - RECOMMENDED PATH

## Executive Summary

This document provides the **recommended implementation sequence** based on feasibility analysis of the planning documents. It simplifies the original Phase 1 specification into achievable stages.

## ⚠️ CRITICAL FINDINGS FROM FEASIBILITY ANALYSIS

### Major Issues Identified

1. **Slanted Hold Detection**: Marked as "critical" but no detection method exists
   - Current YOLO model doesn't output hold orientation
   - Would require model retraining or manual annotation
   - **SOLUTION**: Defer to Phase 1b, assume horizontal holds for MVP

2. **Over-Specified Initial Scope**: Full Phase 1 spec includes features better suited for refinement
   - Wall-angle-dependent foothold weighting (complex)
   - Complexity multipliers (advanced)
   - Multi-segment wall support (complex UI/data)
   - **SOLUTION**: Staged implementation (1a → 1b → 1c)

3. **Prescriptive Values**: Threshold values treated as final instead of calibration starting points
   - **SOLUTION**: Added calibration warnings to all specifications

## Recommended Implementation Sequence

### Stage 1a: MVP (4-6 weeks) ⭐ **START HERE**

**Document**: [`phase1/phase1a_mvp_specification.md`](phase1/phase1a_mvp_specification.md)

**Scope**: Basic 4-factor model with simplified scoring

**Features**:

- ✅ Basic hold difficulty (NO slant detection)
- ✅ Hold density (simplified categories)
- ✅ Hold distances (basic calculation)
- ✅ Wall incline (manual input, single angle)
- ✅ Simple handhold/foothold separation
- ✅ **Constant 60/40 foothold weighting** (no wall angle dependency)

**Success Criteria**:

- Accuracy: ≥50% exact, ≥75% within ±1 grade
- Performance: <100ms per route
- Deployable with user feedback collection

**Deliverables**:

- `src/grade_prediction_mvp.py` module
- Database migration for `wall_incline` field
- UI dropdown for wall angle input
- Configuration file updates
- Unit tests
- Integration tests

**Timeline**: 4-6 weeks (1 developer, full-time)

---

### Stage 1b: Calibration & Refinement (2-3 weeks)

**Timing**: After Phase 1a deployed and collecting feedback (≥100 route samples)

**Scope**: Calibrate MVP based on real data, add refinements

**Features**:

- ✅ Empirical threshold calibration (adjust config values)
- ✅ Wall-angle-dependent foothold weighting (if data supports it)
- ✅ Slanted hold detection OR manual annotation (if feasible)
- ✅ Advanced foothold scarcity multipliers (if needed)

**Success Criteria**:

- Accuracy: ≥60% exact, ≥80% within ±1 grade
- User satisfaction: >3.5/5.0
- No regressions from Phase 1a

**Deliverables**:

- Updated configuration with calibrated values
- Optional: Slant angle detection or annotation UI
- Enhanced foothold analysis
- Calibration report and documentation

**Timeline**: 2-3 weeks (after 2-3 weeks of feedback collection)

---

### Stage 1c: Advanced Features (2-3 weeks) - OPTIONAL

**Timing**: After Phase 1b validated and stable

**Scope**: Add complexity multipliers and advanced analysis

**Features**:

- ✅ Complexity multipliers (wall transitions, hold type variability)
- ✅ Multi-segment wall support (if useful)
- ✅ Shannon entropy for hold type analysis
- ✅ Advanced scoring formulas

**Success Criteria**:

- Accuracy: ≥70% exact, ≥85% within ±1 grade
- User satisfaction: >4.0/5.0
- Measurable improvement over Phase 1b

**Deliverables**:

- Wall segment input UI (advanced mode)
- Transition detection algorithm
- Hold type entropy calculation
- Updated configuration

**Timeline**: 2-3 weeks (after Phase 1b proven successful)

---

### Stage 2: Persona Personalization (6-8 weeks) - OPTIONAL

**Timing**: 2-3 months after Phase 1 deployment

**Prerequisites**:

- ✅ Phase 1 (1a + 1b minimum) deployed and validated
- ✅ Baseline accuracy ≥60% within ±1 grade
- ✅ User feedback system operational
- ✅ ≥100 route analyses completed

**Scope**: Add persona-based difficulty adjustments

**Document**: [`phase1.5/README.md`](phase1.5/README.md)

**Timeline**: 6-8 weeks

---

### Stage 3: Video Analysis (13-18 weeks) - FUTURE

**Timing**: 6+ months after Phase 1 deployment

**Prerequisites**:

- ✅ Phase 1 accuracy ≥70% within ±1 grade
- ✅ Stable in production for 6+ months
- ✅ Resources for video processing infrastructure

**Scope**: Add video-based performance validation

**Document**: [`phase2/README.md`](phase2/README.md)

**Timeline**: 13-18 weeks

---

## Week-by-Week Breakdown (Phase 1a MVP)

### Weeks 1-2: Foundation

- [ ] **Database**: Add `wall_incline` field to Analysis model
- [ ] **Module**: Create `src/grade_prediction_mvp.py`
- [ ] **Utilities**: Hold separation, dimension calculation
- [ ] **Tests**: Unit tests for utilities
- [ ] **Config**: Add MVP configuration to `user_config.yaml`

### Weeks 2-3: Core Factors

- [ ] **Factor 1**: Implement hold difficulty (simplified, no slant)
- [ ] **Factor 2**: Implement hold density (simplified categories)
- [ ] **Factor 3**: Implement distances (basic calculation)
- [ ] **Factor 4**: Implement wall incline
- [ ] **Tests**: Unit tests for each factor

### Weeks 3-4: Integration

- [ ] **Main Function**: `predict_grade_v2_mvp()`
- [ ] **Integration**: Update `src/main.py` to use new function
- [ ] **Config Loading**: Load configuration values
- [ ] **Tests**: Integration tests, end-to-end tests

### Weeks 4-5: UI & Deployment

- [ ] **UI**: Add wall_incline dropdown to upload form
- [ ] **UI**: Update results display with score breakdown
- [ ] **Testing**: Manual QA, edge case testing
- [ ] **Deployment**: Deploy to staging environment
- [ ] **Documentation**: User-facing documentation

### Weeks 5-6: Production & Feedback

- [ ] **Production**: Deploy with feature flag
- [ ] **Monitoring**: Set up prediction logging
- [ ] **Feedback**: Collect user feedback (target: 50+ routes)
- [ ] **Analysis**: Document issues, edge cases, biases
- [ ] **Planning**: Plan Phase 1b calibration

---

## Critical Success Factors

### 1. Start Simple

- ✅ Implement Phase 1a MVP first
- ✅ NO slant detection initially
- ✅ NO complexity multipliers initially
- ✅ Constant foothold weighting (60/40)

### 2. Deploy Fast

- ✅ Target 4-6 weeks to production
- ✅ Get user feedback early
- ✅ Iterate based on real data

### 3. Measure Everything

- ✅ Log all predictions with full breakdown
- ✅ Track prediction distribution by grade
- ✅ Monitor systematic biases (route type patterns)
- ✅ Collect user feedback on every prediction

### 4. Iterate Weekly

- ✅ After deployment, review feedback weekly
- ✅ Adjust configuration values based on data
- ✅ Document what changes improved accuracy
- ✅ Deploy config updates without code changes

### 5. Treat Values as Hypotheses

- ✅ All thresholds are starting points
- ✅ Expect ±30-50% adjustment range
- ✅ Calibration is part of the process
- ✅ Real data beats specifications

---

## What NOT to Implement (Yet)

### ❌ Deferred to Phase 1b

- Slanted hold detection/adjustment
- Wall-angle-dependent foothold weighting
- Advanced foothold scarcity multipliers (7 levels → simplified to 3)
- Fine-grained size categories (5+ → simplified to 3)

### ❌ Deferred to Phase 1c

- Complexity multipliers (wall transitions)
- Hold type variability (entropy-based)
- Multi-segment wall support
- Advanced scoring formulas

### ❌ Deferred to Phase 1.5

- Persona-based personalization
- Multi-persona blending
- User profile system

### ❌ Deferred to Phase 2

- Video analysis
- Pose estimation
- Cross-validation system

---

## Decision Matrix: When to Add Features

| Feature | Add When... | Success Criteria |
| :-----: | :---------: | :--------------: |
| **Basic 4 factors** | Now (Phase 1a) | N/A - core requirement |
| **Constant foothold weighting** | Now (Phase 1a) | N/A - core requirement |
| **Manual wall angle input** | Now (Phase 1a) | N/A - core requirement |
| **Configuration calibration** | After 50+ feedback samples | Systematic bias identified |
| **Wall-angle-dependent weighting** | After MVP validated | ≥100 samples, clear pattern |
| **Slanted hold detection** | When YOLO model supports it | Model retrained with orientation |
| **Complexity multipliers** | After 1b calibrated | Accuracy ≥60%, user satisfaction >3.5 |
| **Persona system** | After Phase 1 stable | Accuracy ≥60%, ≥3 months production |
| **Video analysis** | After Phase 1 proven | Accuracy ≥70%, ≥6 months production |

---

## Comparison: MVP vs Full Spec

### Complexity Reduction

| Metric | Full Phase 1 Spec | Phase 1a MVP | Reduction |
| :----: | :---------------: | :----------: | :-------: |
| **Lines of Code** | ~1200 | ~400 | 67% |
| **Implementation Time** | 8-12 weeks | 4-6 weeks | 50% |
| **Configuration Values** | ~40 | ~15 | 63% |
| **Database Fields** | 3-4 | 1 | 75% |
| **UI Components** | 5-7 | 1 dropdown | 86% |
| **Test Cases** | 80+ | 30+ | 63% |

### Feature Comparison

| Feature | Full Phase 1 | Phase 1a MVP | Notes |
| :-----: | :----------: | :----------: | :---: |
| Slanted holds | Required | ❌ Deferred | No detection method |
| Foothold weighting | Wall-angle-dependent | Constant (60/40) | Simpler |
| Size categories | 5+ levels | 3 levels | Easier thresholds |
| Scarcity levels | 7 levels | 3 levels | Simpler logic |
| Wall segments | Multi-segment | Single angle | Simpler UI |
| Complexity multipliers | Included | ❌ Deferred | Entire subsystem |
| Hold type entropy | Included | ❌ Deferred | Simple calculation |

---

## FAQ

### Q: Why not implement the full Phase 1 spec?

**A**: The full spec includes features that:

1. Can't be implemented yet (slanted holds - no detection)
2. Add complexity without proven value (wall-angle weighting before validation)
3. Should be calibrated first (complexity multipliers before base model works)

**MVP approach**: Start simple, validate, iterate based on real data.

### Q: Will Phase 1a be accurate enough?

**A**: Initial accuracy may be lower (50-60% exact match), but:

- Better than current system
- Provides foundation for calibration
- Enables feedback collection
- Quick iteration improves accuracy weekly

**Phase 1b calibration** will bring accuracy to 60-80% exact match.

### Q: When should we add slanted hold detection?

**A**: Options:

1. **Phase 1b**: If YOLO model retrained with orientation labels
2. **Phase 1b**: If manual annotation UI added
3. **Later**: If not critical for accuracy (may be less important than thought)

**Recommend**: Wait for Phase 1a feedback to see if users mention hold orientation issues.

### Q: What if we have more resources?

**A**: Even with more resources:

1. Start with Phase 1a MVP (risk mitigation)
2. Validate approach early
3. Then parallelize Phase 1b features
4. Avoid implementing unvalidated complexity

**Do NOT skip MVP** even with more resources.

### Q: How do we handle user expectations?

**A**: Communication:

- "This is version 1.0 of improved grade prediction"
- "Grades will improve as we collect feedback"
- "Help us calibrate by providing feedback on predictions"
- Show score breakdown so predictions are explainable

---

## Next Steps

### Immediate Actions

1. **Review** this roadmap with team/stakeholders
2. **Approve** Phase 1a MVP specification
3. **Set up** development environment
4. **Begin** Phase 1a implementation (Week 1-2 tasks)

### Do NOT Start With

- ❌ Full Phase 1 specification
- ❌ Slanted hold detection research
- ❌ Complexity multiplier implementation
- ❌ Persona system design

### Reference Documents

**For Implementation**:

- 📘 **Phase 1a MVP Spec**: [`phase1/phase1a_mvp_specification.md`](phase1/phase1a_mvp_specification.md) ⭐ **START HERE**
- 📗 **Phase 1 Full Spec**: [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md) - Reference only, DO NOT implement yet
- 📙 **Overview**: [`overview.md`](overview.md) - System context

**For Later Phases**:

- 📕 **Phase 1.5**: [`phase1.5/README.md`](phase1.5/README.md) - After Phase 1 validated
- 📔 **Phase 2**: [`phase2/README.md`](phase2/README.md) - Distant future

---

## Summary

### Recommended Path

1. **Phase 1a** (4-6 weeks): Implement MVP per simplified spec
2. **Feedback** (2-3 weeks): Collect 100+ route samples
3. **Phase 1b** (2-3 weeks): Calibrate and refine
4. **Phase 1c** (2-3 weeks): Add complexity multipliers if needed
5. **Phase 1.5** (6-8 weeks): Personas (if desired)
6. **Phase 2** (13-18 weeks): Video analysis (distant future)

### Key Principles

✅ **Start simple** - Basic 4-factor model
✅ **Deploy fast** - 4-6 weeks to production
✅ **Iterate weekly** - Calibrate based on feedback
✅ **Measure everything** - Log all predictions
✅ **Treat values as hypotheses** - Expect adjustments

### Success Metrics

**Phase 1a MVP**:

- Accuracy: ≥50% exact, ≥75% within ±1 grade
- Deployment: <6 weeks
- Feedback: ≥50 routes collected

**Phase 1b Calibrated**:

- Accuracy: ≥60% exact, ≥80% within ±1 grade
- User satisfaction: >3.5/5.0
- Stable, no regressions

**Phase 1c Advanced** (if pursued):

- Accuracy: ≥70% exact, ≥85% within ±1 grade
- User satisfaction: >4.0/5.0
- Measurable improvement over 1b

---

**Ready to implement? Start with**: [`phase1/phase1a_mvp_specification.md`](phase1/phase1a_mvp_specification.md)
