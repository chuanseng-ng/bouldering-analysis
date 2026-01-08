# Bouldering Grade Prediction - Planning Documents

This directory contains the planning and specification documents for the bouldering route grade prediction system.

## Quick Start

**New to this project?** Start here:

1. Read [`overview.md`](overview.md) - System goals, phased approach, key principles
2. Explore [`phase1/`](phase1/) - Core route-based prediction algorithm (HIGH PRIORITY)
3. Review other phases only after Phase 1 validation

**Ready to implement?** Go directly to:

- **Phase 1**: [`phase1/README.md`](phase1/README.md) - Start here for implementation
- **Phase 1 Factors**: Individual factor specifications in [`phase1/`](phase1/) directory
- **Implementation Notes**: [`phase1/implementation_notes.md`](phase1/implementation_notes.md) - Technical guidance

## Document Structure

### Overview & System Design

- **[overview.md](overview.md)** - High-level system overview, objectives, phased approach, and key design principles

### Phase 1: Route-Based Prediction (Core Implementation) ⭐ **START HERE**

The [`phase1/`](phase1/) directory contains the core route-based grade prediction algorithm:

| File | Description |
| :--: | :---------: |
| **[README.md](phase1/README.md)** | Phase 1 overview, algorithm structure, implementation approach |
| **[factor1_hold_analysis.md](phase1/factor1_hold_analysis.md)** | Hold types, sizes, **slant angles**, foothold importance |
| **[factor2_hold_density.md](phase1/factor2_hold_density.md)** | Hold count analysis and spacing density |
| **[factor3_hold_distances.md](phase1/factor3_hold_distances.md)** | Inter-hold distances and reach requirements |
| **[factor4_wall_incline.md](phase1/factor4_wall_incline.md)** | Wall angle impact on difficulty |
| **[complexity_multipliers.md](phase1/complexity_multipliers.md)** | Advanced features (add in refinement phase) |
| **[implementation_notes.md](phase1/implementation_notes.md)** | Technical guidance, calibration, testing strategy |

**Priority**: **HIGHEST** - Implement Phase 1 first before considering other phases.

### Phase 1.5: Persona-Based Personalization (Optional Enhancement)

The [`phase1.5/`](phase1.5/) directory contains specifications for optional persona-based difficulty adjustments:

| File | Description |
| :--: | :---------: |
| **[README.md](phase1.5/README.md)** | Persona system overview, prerequisites, UX design |
| **[persona_definitions.md](phase1.5/persona_definitions.md)** | 7 climbing persona archetypes and adjustment profiles |
| **[implementation_approach.md](phase1.5/implementation_approach.md)** | Technical integration, calibration strategy |

**Prerequisites**: Phase 1 deployed and validated (≥60% accuracy within ±1 grade)

### Phase 2: Video Analysis Validation (Future Enhancement)

The [`phase2/`](phase2/) directory contains specifications for future video-based validation:

| File | Description |
| :--: | :---------: |
| **[README.md](phase2/README.md)** | Video analysis overview, prerequisites, cross-validation approach |
| **[technical_specification.md](phase2/technical_specification.md)** | Body mechanics metrics, pose estimation, implementation details |

**Prerequisites**: Phase 1 accuracy ≥70% within ±1 grade, stable in production for 6+ months

## Reading Order

### For Developers Starting Implementation

1. **[overview.md](overview.md)** - Understand system goals and phased approach
2. **[phase1/README.md](phase1/README.md)** - Algorithm structure and implementation stages
3. **Phase 1 factor files** - Detailed scoring logic for each difficulty factor:
   - [factor1_hold_analysis.md](phase1/factor1_hold_analysis.md) - **Read this first** (includes slanted holds)
   - [factor2_hold_density.md](phase1/factor2_hold_density.md)
   - [factor3_hold_distances.md](phase1/factor3_hold_distances.md)
   - [factor4_wall_incline.md](phase1/factor4_wall_incline.md)
4. **[implementation_notes.md](phase1/implementation_notes.md)** - Practical implementation guidance
5. **[complexity_multipliers.md](phase1/complexity_multipliers.md)** - Add only in Phase 1c (refinement)
6. **Phase 1.5 and Phase 2** - Only after Phase 1 validation

### For Project Stakeholders & Managers

1. **[overview.md](overview.md)** - High-level understanding, success criteria, phased approach
2. **[phase1/README.md](phase1/README.md)** - Core implementation scope and staged approach
3. **Phase-specific READMEs** as needed - Prerequisites and timelines for each phase

### For Algorithm Calibration

1. **Phase 1 factor files** - Understand difficulty scoring for each factor
2. **[implementation_notes.md](phase1/implementation_notes.md)** - Calibration strategy and iteration approach
3. Monitor: All threshold values require empirical validation with real route data

## Key Principles

### 1. Iterative Calibration Over Prescriptive Values

**CRITICAL**: All threshold values, multipliers, and weights in these specifications are **initial hypotheses requiring empirical calibration**.

Treat specification values as:

- ✅ **Starting points** for implementation
- ✅ **Calibration targets** to validate with real data
- ✅ **Iteration baselines** to adjust based on user feedback

Do **NOT** treat as:

- ❌ **Final production values**
- ❌ **Immutable constants**
- ❌ **Guaranteed accurate thresholds**

**Why**: Values depend on image resolution, camera angles, gym-specific route setting, detection model calibration, and user population characteristics.

**Action**: Implement extensive logging, collect user feedback, iterate weekly.

### 2. Slanted Hold Consideration

Hold orientation (slant angle) significantly affects difficulty and **must** be considered:

**For Handholds**:

- **Downward-slanting**: Harder (requires more grip strength) - adjust difficulty +20-40%
- **Horizontal/Positive**: Neutral difficulty
- **Upward-slanting**: Easier (more positive surface) - adjust difficulty -15-25%

**For Footholds**:

- **Downward-slanting**: Much harder (foot can slip off) - adjust difficulty +40-60%
- **Horizontal**: Standard difficulty
- **Upward-slanting**: Easier (positive platform) - adjust difficulty -20-30%

See [factor1_hold_analysis.md](phase1/factor1_hold_analysis.md) for integration approach.

### 3. Complexity Added Incrementally

**Implementation Staging**:

- **Phase 1a**: Basic 4-factor model (hold difficulty, density, distances, wall incline)
- **Phase 1b**: Calibration and slant angle integration
- **Phase 1c**: Complexity multipliers (wall transitions, hold variability)

**Rationale**: Prevents complexity from blocking initial development, allows focused debugging.

### 4. Avoid Over-Engineering Initial Implementation

**Start simple**:

- Basic 4-factor model first
- Conservative calibration values
- Extensive logging for calibration
- User feedback collection from day 1
- Iterate based on real data

**Avoid**:

- Implementing all features at once
- Perfect accuracy on first attempt
- ML approaches before collecting training data
- Complexity multipliers before basic model works

### 5. Foothold Parity with Handholds

Footholds are **as important as handholds** for difficulty assessment:

- Footholds enable balance, rest positions, weight transfer
- Missing or tiny footholds drastically increase difficulty
- Foothold importance varies by wall angle (65% on slabs, 25% on overhangs)

See [factor1_hold_analysis.md](phase1/factor1_hold_analysis.md) for wall-angle-dependent foothold weighting.

## Implementation Roadmap

### Phase 1: Route-Based Prediction (Core) - **IMMEDIATE PRIORITY**

**Stage 1a: Basic Model**

- Implement 4-factor algorithm
- Manual wall angle input
- Basic hold size/type scoring
- Deploy and collect feedback

**Stage 1b: Calibration & Slant Integration**

- Analyze user feedback
- Adjust factor weights and thresholds
- Add hold slant angle detection/adjustment
- Achieve ≥60% exact match, ≥80% within ±1 grade

**Stage 1c: Advanced Features** (Optional)

- Add complexity multipliers
- Wall segment support (multi-angle routes)
- Target ≥70% exact match

### Phase 1.5: Persona Personalization (Optional) - **AFTER PHASE 1 VALIDATION**

**Prerequisites**:

- Phase 1 deployed and achieving accuracy targets
- User feedback system operational
- 100+ route analyses completed

**Duration**: ~6-8 weeks

### Phase 2: Video Analysis Validation (Future) - **DEFER 6+ MONTHS**

**Prerequisites**:

- Phase 1 accuracy ≥70% within ±1 grade
- Stable in production for 6+ months
- Resources allocated for video processing infrastructure

**Duration**: ~13-18 weeks

## Success Criteria

### Phase 1 Targets

**Accuracy**:

- ✅ Exact match: ≥60% (Phase 1b), ≥70% (Phase 1c)
- ✅ Within ±1 grade: ≥80% (Phase 1b), ≥85% (Phase 1c)
- ✅ No regressions from current system

**Performance**:

- ✅ Prediction time: <100ms per route
- ✅ No crashes on edge cases

**User Satisfaction**:

- ✅ User feedback collection operational
- ✅ Clear explanations for predictions
- ✅ User satisfaction >3.0/5.0 (Phase 1b), >3.5/5.0 (Phase 1c)

### Phase 1.5 Targets

- Adoption: >40% of users select a persona
- Satisfaction: >3.5/5.0 with personalized grades
- Accuracy: >70% report "feels accurate"

### Phase 2 Targets

- Route vs video agreement: ≥70% within ±1 grade
- Video processing time: <2 minutes per video
- Pose estimation accuracy: ≥85%

## Related Documentation

### Source Code

- **Main Application**: [`../src/main.py`](../src/main.py) - Current implementation
- **Database Models**: [`../src/models.py`](../src/models.py) - Analysis, DetectedHold, HoldType
- **Constants**: [`../src/constants.py`](../src/constants.py) - Hold type definitions
- **Configuration**: [`../src/cfg/user_config.yaml`](../src/cfg/user_config.yaml) - System configuration

### External Resources

- **MediaPipe Pose** (for Phase 2): https://google.github.io/mediapipe/solutions/pose.html
- **V-Scale Grading**: Standard bouldering difficulty scale (V0-V12)

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 2.0 | 2026-01-07 | Restructured into modular files, added slanted hold considerations, removed deterministic timelines, emphasized calibration | Claude Code |
| 1.0 | 2026-01-04 | Initial comprehensive plan | System |

## FAQ

### Why are the plans so detailed if values need calibration?

**Answer**: Specifications provide:

- Starting hypotheses based on climbing domain knowledge
- Implementation structure and approach
- Calibration targets and success criteria
- Edge case considerations

They are **not** meant to be final production values. Treat as research prototypes requiring validation.

### What if Phase 1 accuracy is low initially?

**Answer**: Expected! Use iterative calibration:

1. Collect ≥50 feedback samples
2. Analyze systematic biases (over/under-predictions by route type)
3. Adjust configuration values (weights, thresholds, scores)
4. Deploy updated config
5. Monitor improvement
6. Repeat weekly until targets met

### Can we skip Phase 1 and use video analysis (Phase 2) directly?

**Answer**: No. Video analysis requires a known-good route prediction baseline for cross-validation. Phase 1 must succeed first.

### Should we implement all 4 factors + complexity multipliers at once?

**Answer**: No. **Staged approach**:

- Phase 1a: Basic 4 factors only
- Phase 1b: Calibration and slant integration
- Phase 1c: Complexity multipliers (if needed)

Starting simple allows faster iteration and focused debugging.

### How do we handle slanted holds if we can't detect them automatically?

**Answer**: **Phase 1a**: Assume neutral (horizontal) for all holds
**Phase 1b**: Add manual annotation option OR implement computer vision detection
**Result**: System degrades gracefully when slant data unavailable

### What if specifications conflict with real-world testing?

**Answer**: **Real-world data always wins**. Specifications are hypotheses. Adjust values based on empirical validation.

## Navigation

- **System Overview**: [overview.md](overview.md)
- **Phase 1 (Start here)**: [phase1/README.md](phase1/README.md)
- **Phase 1.5 (Optional)**: [phase1.5/README.md](phase1.5/README.md)
- **Phase 2 (Future)**: [phase2/README.md](phase2/README.md)

---

**Next Action**: Read [overview.md](overview.md), then proceed to [phase1/README.md](phase1/README.md) to begin implementation.
