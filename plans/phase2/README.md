# Phase 2: Video Analysis Validation

## Overview

Phase 2 adds video-based performance analysis to cross-validate route-based predictions from Phase 1.

**Status**: Lower priority future enhancement - implement **only after** Phase 1 deployment and extensive validation

**Objective**: Analyze climber body mechanics and movement patterns from video footage to generate an independent grade prediction, then cross-validate with route-based prediction.

## Core Concept

**Video analysis serves as a validation mechanism, not a replacement** for the route-based algorithm.

By analyzing how climbers actually perform on a route (body mechanics, movement patterns, struggle indicators), we can:

1. **Validate** the accuracy of route-based predictions
2. **Identify** systematic biases in the Phase 1 algorithm
3. **Detect** unusual route characteristics not captured by visual hold analysis
4. **Improve** the algorithm over time using performance data
5. **Build confidence** in predictions through dual-source validation

## How It Works

### Step 1: Analyze Climbing Performance

Extract metrics from climbing video using pose estimation (MediaPipe or similar):

**Body Mechanics**:

- Hip-shoulder angle (body lean from wall)
- Arm extension ratio (locked off vs controlled)
- Knee-hip-shoulder angle (body compression)
- Core tension indicators

**Movement Patterns**:

- Dynamic vs static movement classification
- Movement flow and hesitation
- Climb pace analysis
- Struggle indicators

**Rest & Recovery**:

- Rest position detection and quality
- Shake-out frequency
- Recovery time needed

**Foot & Hand Analysis**:

- Foot placement precision
- Weight distribution (hand-heavy vs foot-heavy)
- Grip duration analysis
- Reaching distances

### Step 2: Generate Video-Based Grade Prediction

Combine extracted metrics into a performance difficulty score:

```python
performance_score = (
    body_position_score × 0.25 +
    movement_score × 0.20 +
    rest_score × 0.20 +
    precision_score × 0.15 +
    reach_score × 0.10 +
    tempo_score × 0.10
)

video_predicted_grade = map(performance_score, V0-V12)
```

**IMPORTANT**: Metric weights and thresholds are **highly speculative** and will require extensive calibration with real climbing videos.

### Step 3: Cross-Validate

Compare route-based and video-based predictions:

```text
Route Prediction (Phase 1): V5
Video Prediction (Phase 2): V6
Grade Difference: 1 grade → VALID ✅
Combined Confidence: High

Route Prediction: V4
Video Prediction: V8
Grade Difference: 4 grades → SIGNIFICANT DISCREPANCY ⚠️
Requires Investigation
```

**Thresholds**:

- **Acceptable**: ±1 grade difference (natural variance)
- **Review**: ±2 grades (flag for manual review)
- **Significant Discrepancy**: ±3+ grades (high-priority investigation)

## Prerequisites for Implementation

**DO NOT begin Phase 2 until**:

- ✅ Phase 1 deployed to production
- ✅ Phase 1 accuracy ≥70% within ±1 grade
- ✅ Phase 1 algorithm stable (no major bugs)
- ✅ User feedback collection operational
- ✅ Development resources available for video processing
- ✅ Sample climbing videos collected (20-50 diverse examples)
- ✅ Storage and compute resources allocated

**Why wait?**: Video analysis requires a known-good route prediction baseline. Building Phase 2 on a faulty Phase 1 foundation compounds errors.

## Technical Requirements

### Pose Estimation

**Recommended Library**: MediaPipe Pose

- Free, open-source
- Fast inference (real-time capable)
- Good accuracy for full-body pose
- 33 body landmarks
- Cross-platform support

**Alternative**: OpenPose (higher accuracy, slower, GPU-required)

### Video Requirements

**Minimum specifications**:

- Resolution: 720p (1280×720) or higher
- Frame rate: 24 FPS or higher
- Duration: Full route completion
- Angle: Side view or slight angle (see climber body clearly)
- Lighting: Climber clearly visible

### Storage Requirements

- **Video files**: 10-100 MB each
- **Pose data**: ~1-5 MB per video (stored as JSON)
- **Retention policy**: Define based on storage capacity (e.g., 90 days)

### Dependencies

```python
# requirements.txt additions for Phase 2
mediapipe>=0.10.0  # Pose estimation
opencv-python>=4.8.0  # Video processing
moviepy>=1.0.3  # Video editing utilities
```

## Implementation Stages

See [`technical_specification.md`](technical_specification.md) for detailed metrics and formulas.

**High-level stages**:

1. **Research & Prototyping** (~2-3 weeks)
   - Collect sample videos
   - Test MediaPipe accuracy
   - Prototype metric extraction

2. **Core Video Analysis** (~3-4 weeks)
   - Implement video upload and validation
   - Integrate pose estimation
   - Extract all metrics

3. **Cross-Validation System** (~2-3 weeks)
   - Implement comparison logic
   - Build discrepancy detection
   - Create logging system

4. **UI Integration** (~2 weeks)
   - Add video upload form
   - Display video analysis results
   - Show cross-validation comparison

5. **Testing & Calibration** (~3-4 weeks)
   - Test on diverse videos
   - Calibrate metric weights
   - Validate accuracy

6. **Deployment & Monitoring** (~1-2 weeks)
   - Deploy with feature flag
   - Set up monitoring
   - Collect user feedback

**Total Estimated Duration**: 13-18 weeks (~3-4.5 months)

## Success Criteria

**Cross-Validation**:

- ✅ Route vs video agreement: ≥70% within ±1 grade
- ✅ Discrepancy detection functional
- ✅ Systematic bias identification working

**Performance**:

- ✅ Video processing time: <2 minutes per video
- ✅ Pose estimation accuracy: ≥85%
- ✅ Storage management operational

**User Experience**:

- ✅ Video upload simple and reliable
- ✅ Processing status visible
- ✅ Results clearly explained

## Key Design Principles

### 1. Video as Validation, Not Primary

Route-based prediction (Phase 1) remains the primary method:

- Faster processing
- No video upload required
- Works for route planning (before attempting)

Video analysis provides:

- Secondary validation
- Bias detection
- Performance feedback
- Algorithm improvement data

### 2. Extensive Calibration Required

**All threshold values are highly speculative**:

- Body position metrics (hip-shoulder angle thresholds, etc.)
- Movement classification (dynamic threshold, etc.)
- Rest quality scores
- Performance score weights

**Expect**:

- Initial accuracy will be low
- Extensive iteration needed
- May require person-specific calibration (climber height, style)
- Consider ML-based approach after collecting sufficient data

### 3. Graceful Degradation

System must handle:

- Poor video quality
- Occlusions (climber blocked by holds/wall)
- Non-standard camera angles
- Partial route completion

Provide confidence scores and warnings when video quality is insufficient.

### 4. Privacy and Storage

- Store videos securely
- Provide deletion options
- Consider anonymization of pose data
- Implement retention policies

## Risks and Challenges

**High Complexity**:

- Video processing infrastructure
- Pose estimation accuracy depends on video quality
- Many calibratable parameters (high-dimensional space)

**Resource Intensive**:

- Storage (videos + pose data)
- Compute (pose estimation)
- Development time (13-18 weeks)

**Accuracy Uncertainty**:

- No guarantee video analysis will improve overall accuracy
- May only work for specific scenarios
- Climber-specific variations (height, style, skill level)

**User Friction**:

- Requires video upload (additional step)
- Privacy concerns
- Processing time (1-2 minutes)

## Relationship to Other Phases

**Phase 1**: Video analysis validates Phase 1 predictions
**Phase 1.5**: Video performance can validate persona adjustments (e.g., do power climbers actually perform better on overhangs?)

## Summary

Phase 2 provides video-based validation of route predictions:

- ✅ **Pose estimation** from climbing videos
- ✅ **Body mechanics analysis** (angles, positions, movement patterns)
- ✅ **Independent grade prediction** from performance metrics
- ✅ **Cross-validation** with route-based predictions
- ✅ **Discrepancy detection** for algorithm improvement

**Priority**: Lower than Phase 1 and Phase 1.5

**Timeline**: 3-4.5 months, starting 6+ months after Phase 1 deployment

**Recommendation**: Focus on Phase 1 first. Consider Phase 2 only after Phase 1 has proven highly successful in production.

**Next**: See [`technical_specification.md`](technical_specification.md) for detailed metrics, formulas, and implementation guidance.
