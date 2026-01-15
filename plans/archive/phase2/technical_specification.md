# Phase 2 Technical Specification

## Overview

This document outlines the technical approach for video-based performance analysis and grade prediction.

**CRITICAL NOTE**: All metrics, thresholds, and formulas in this document are **highly speculative starting points**. Expect extensive calibration with real climbing videos. Consider this a research prototype, not a production-ready specification.

## Body Mechanics Metrics

### 1. Body Position & Angles

**Hip-Shoulder Angle** (Body Lean):

```python
# Calculate angle between body centerline and wall normal
angle = calculate_angle(hip_pos - shoulder_pos, wall_normal)

# Interpretation:
# 0°-30°: Close to wall (slab style)
# 30°-60°: Moderate lean
# 60°-90°: Far from wall (overhang, requires core tension)
```

**Arm Extension Ratio**:

```python
ratio = wrist_to_shoulder_distance / (upper_arm + forearm_length)

# Interpretation:
# <0.7: Bent arms (controlled, easier)
# 0.7-0.9: Moderate extension
# >0.9: Nearly straight arms (harder, locked off)
```

**Knee-Hip-Shoulder Angle** (Body Compression):

```python
angle = calculate_angle_3_points(knee, hip, shoulder)

# Interpretation:
# 60-90°: Compressed (good footwork, efficient)
# 120-180°: Extended (reaching, dynamic, harder)
```

### 2. Movement Patterns

**Dynamic vs Static Classification**:

```python
# Classify based on acceleration
if max(accelerations) > DYNAMIC_THRESHOLD:
    movement_type = "dynamic"
else:
    movement_type = "static"

# DYNAMIC_THRESHOLD: ~2.0 m/s² (HIGHLY SPECULATIVE - requires calibration)
```

**Movement Flow**:

```python
# Lower flow = more hesitation = harder
flow_score = 1.0 / (1 + pause_count * 0.1 + jerkiness * 0.5)
```

**Climb Pace**:

```python
pace = route_height_meters / time_to_complete_seconds

# Interpretation:
# <0.5 m/s: Struggling, high difficulty
# 0.5-1.0 m/s: Moderate pace
# >1.5 m/s: Fast, comfortable
```

### 3. Rest & Recovery

**Rest Position Detection**:

```python
# Detect frames where velocity < threshold
is_resting = velocity < 0.1  # m/s (SPECULATIVE)

rest_periods = identify_continuous_periods(is_resting)
```

**Rest Quality Score**:

```python
# Good rest: straight arms, hips close to wall, weight on feet
quality = hip_shoulder_angle * 0.5 + (1 - arm_extension) * 0.5

# Lower score = better rest (relaxed position)
```

### 4. Foot & Hand Analysis

**Foot Placement Precision**:

```python
# Count foot adjustments (multiple movements to same hold)
precision_score = foot_adjustments * 0.5 + alignment_errors * 0.5

# Higher score = less precise (harder holds or less skilled)
```

**Weight Distribution Estimation**:

```python
# Approximate from body center of mass relative to supports
weight_on_feet = sum(1/foot_distances) / (sum(1/hand_distances) + sum(1/foot_distances))

# Higher ratio (>0.6) = foot-heavy (more efficient)
# Lower ratio (<0.4) = hand-heavy (overhang or poor technique)
```

## Performance-Based Grade Prediction

### Scoring Model

```python
def predict_grade_from_video(video_metrics: dict) -> tuple:
    """
    Predict grade from video performance metrics.

    ALL WEIGHTS ARE SPECULATIVE - REQUIRE EXTENSIVE CALIBRATION.
    """
    # Score individual components (each 0-12 range)
    body_position_score = score_body_position(video_metrics)
    movement_score = score_movement_quality(video_metrics)
    rest_score = score_rest_analysis(video_metrics)
    precision_score = score_foot_precision(video_metrics)
    reach_score = score_reaching(video_metrics)
    tempo_score = score_climb_pace(video_metrics)

    # Weighted combination (SPECULATIVE WEIGHTS)
    performance_score = (
        body_position_score * 0.25 +
        movement_score * 0.20 +
        rest_score * 0.20 +
        precision_score * 0.15 +
        reach_score * 0.10 +
        tempo_score * 0.10
    )

    # Map to V-grade
    predicted_grade = map_score_to_grade(performance_score)

    # Calculate confidence based on video quality
    confidence = calculate_video_confidence(video_metrics)

    breakdown = {
        'performance_score': performance_score,
        'components': {
            'body_position': body_position_score,
            'movement': movement_score,
            'rest': rest_score,
            'precision': precision_score,
            'reach': reach_score,
            'tempo': tempo_score
        }
    }

    return predicted_grade, confidence, breakdown
```

**CRITICAL**: The 0.25, 0.20, 0.15 weights are **starting guesses**. Expect to iterate extensively.

### Component Scoring Functions

Each component scorer maps metrics to 0-12 difficulty scale. Examples:

**Body Position Scorer**:

```python
def score_body_position(metrics):
    avg_hip_shoulder_angle = mean(metrics['hip_shoulder_angles'])

    # Large angles = harder (body far from wall)
    if avg_hip_shoulder_angle > 70:
        return 10-12  # Very hard
    elif avg_hip_shoulder_angle > 50:
        return 7-10   # Hard
    elif avg_hip_shoulder_angle > 30:
        return 4-7    # Moderate
    else:
        return 1-4    # Easier (slab)

    # Thresholds (70, 50, 30) are SPECULATIVE
```

**Movement Quality Scorer**:

```python
def score_movement_quality(metrics):
    flow = metrics['movement_flow']
    dynamic_ratio = metrics['dynamic_move_count'] / metrics['total_moves']

    base_score = 12 - (flow * 10)  # Low flow = high score (harder)
    dynamic_penalty = dynamic_ratio * 3  # Dynamic moves add difficulty

    return min(12, base_score + dynamic_penalty)

    # Formula is SPECULATIVE - requires validation
```

## Cross-Validation Logic

```python
def cross_validate_predictions(
    route_grade: str,
    video_grade: str,
    route_confidence: float,
    video_confidence: float
) -> dict:
    """
    Compare route-based and video-based predictions.
    """
    grade_diff = abs(grade_to_numeric(route_grade) - grade_to_numeric(video_grade))

    # Determine status
    if grade_diff <= 1:
        status = "VALID"
        message = "Predictions agree within acceptable margin"
    elif grade_diff == 2:
        status = "REVIEW"
        message = "Moderate discrepancy - flagged for review"
    else:  # grade_diff >= 3
        status = "SIGNIFICANT_DISCREPANCY"
        message = "Large discrepancy - requires investigation"

    # Adjust combined confidence
    combined_confidence = (route_confidence + video_confidence) / 2
    if grade_diff <= 1:
        combined_confidence *= 1.1  # Boost when agree
    else:
        combined_confidence *= 0.8  # Reduce when disagree

    combined_confidence = min(1.0, combined_confidence)

    return {
        'status': status,
        'message': message,
        'route_grade': route_grade,
        'video_grade': video_grade,
        'grade_difference': grade_diff,
        'combined_confidence': combined_confidence,
        'recommended_grade': determine_recommended_grade(
            route_grade, video_grade, route_confidence, video_confidence
        )
    }
```

**Thresholds (±1, ±2, ±3)** are reasonable starting points but may need adjustment based on empirical agreement rates.

## Implementation Guidance

### Video Processing Pipeline

1. **Upload & Validation**:
   - Check file size (<200MB recommended)
   - Check format (MP4, MOV, AVI, WEBM)
   - Check resolution (≥720p)
   - Check duration (reasonable length)

2. **Pose Estimation**:

   ```python
   import mediapipe as mp

   mp_pose = mp.solutions.pose
   pose = mp_pose.Pose(
       static_image_mode=False,
       model_complexity=2,
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5
   )

   # Process video frame by frame
   for frame in video_frames:
       results = pose.process(frame)
       if results.pose_landmarks:
           store_landmarks(results.pose_landmarks)
   ```

3. **Metric Extraction**:
   - Calculate body angles from landmark positions
   - Detect movement patterns from landmark trajectories
   - Identify rest periods from velocity analysis
   - Compute aggregate statistics

4. **Grade Prediction**:
   - Score each component (body position, movement, rest, etc.)
   - Combine with weights
   - Map to V-grade

5. **Cross-Validation**:
   - Compare with route-based prediction
   - Flag discrepancies
   - Log for algorithm improvement

### Data Storage

**Database Extensions**:

```python
# Store video analysis results in features_extracted JSON field
{
  "video_analysis": {
    "video_filename": "climb_20260107.mp4",
    "video_duration": 45.3,
    "pose_data_file": "data/pose_landmarks/abc123.json",
    "metrics": {
      "body_position": {...},
      "movement": {...},
      "rest": {...}
    },
    "grade_prediction": {
      "predicted_grade": "V7",
      "confidence": 0.72,
      "performance_score": 7.8
    }
  },
  "cross_validation": {
    "route_grade": "V5",
    "video_grade": "V7",
    "grade_difference": 2,
    "status": "REVIEW"
  }
}
```

### Configuration

```yaml
video_analysis:
  enabled: false  # Feature flag
  max_video_size_mb: 200
  allowed_formats: ['.mp4', '.mov', '.avi', '.webm']
  min_resolution: [1280, 720]
  pose_model: "mediapipe"
  target_fps: 10  # Extract 10 frames per second

  # Cross-validation thresholds
  validation:
    acceptable_margin: 1  # grades
    review_threshold: 2
    significant_discrepancy: 3

  # Component weights (SPECULATIVE)
  performance_weights:
    body_position: 0.25
    movement: 0.20
    rest: 0.20
    precision: 0.15
    reach: 0.10
    tempo: 0.10
```

## Calibration Strategy

### Phase 2a: Prototype & Research

**Goal**: Validate pose estimation works on climbing videos

**Activities**:

1. Collect 20-30 diverse climbing videos (different grades, styles, gyms)
2. Run pose estimation, visualize landmarks
3. Manually assess pose accuracy
4. Identify failure modes (occlusions, lighting, angles)
5. Document video quality requirements

**Success**: Pose estimation ≥85% landmark detection accuracy

### Phase 2b: Metric Extraction

**Goal**: Extract meaningful metrics from pose data

**Activities**:

1. Implement metric calculation functions
2. Manually review metrics for 10-20 videos
3. Validate metrics align with visual assessment
4. Identify outliers and edge cases
5. Refine metric formulas

**Success**: Metrics correlate with visual difficulty assessment

### Phase 2c: Grade Prediction Calibration

**Goal**: Map performance metrics to grade predictions

**Activities**:

1. Collect ground truth grades for 50+ videos
2. Train/calibrate scoring model
3. Test prediction accuracy
4. Iterate on weights and thresholds
5. Compare accuracy to route-based predictions

**Success**: Video predictions ≥50% exact match, ≥70% within ±1 grade

### Phase 2d: Cross-Validation

**Goal**: Validate agreement between route and video predictions

**Activities**:

1. Analyze discrepancy patterns
2. Identify systematic biases
3. Refine both route and video algorithms
4. Build confidence in dual-source predictions

**Success**: Route vs video agreement ≥70% within ±1 grade

## Common Challenges

**Challenge: Poor Video Quality**

- **Impact**: Low pose estimation accuracy, unreliable metrics
- **Mitigation**: Provide video quality guidelines, confidence scoring, reject poor videos

**Challenge: Occlusions**

- **Impact**: Missing landmarks, broken trajectories
- **Mitigation**: Interpolate missing frames, use confidence thresholds, graceful degradation

**Challenge: Climber Variability**

- **Impact**: Height, wingspan, style differences affect metrics
- **Mitigation**: Normalize by body dimensions (if available), collect diverse training data

**Challenge: Camera Angle**

- **Impact**: Side view best, other angles distort metrics
- **Mitigation**: Detect camera angle, adjust metrics or reject unsuitable videos

**Challenge: Computation Time**

- **Impact**: Pose estimation slow (1-2 minutes per video)
- **Mitigation**: Async processing, progress updates, GPU acceleration

## Edge Cases

**Partial Route Completion**:

- Climber falls mid-route
- Use completed portion for analysis
- Adjust confidence based on completion percentage

**Multiple Climbers in Frame**:

- Pose estimation may track wrong person
- Require single-climber videos or implement person tracking

**Unusual Movements**:

- Sitting, adjusting, chatting (non-climbing)
- Detect and filter non-climbing segments

**Camera Movement**:

- Handheld cameras, panning
- May affect metric accuracy
- Prefer static camera or stabilization

## Summary

Phase 2 video analysis requires:

1. ✅ **Pose estimation** - MediaPipe or OpenPose
2. ✅ **Metric extraction** - Body mechanics, movement, rest, precision
3. ✅ **Performance scoring** - Weighted combination of metrics
4. ✅ **Cross-validation** - Compare with route predictions
5. ✅ **Extensive calibration** - All values are starting points

**CRITICAL SUCCESS FACTOR**: Treat all thresholds, weights, and formulas as research hypotheses. Plan for extensive iteration.

**Development Approach**: Prototype → Research → Iterate → Validate → Deploy

**Next**: Only proceed if Phase 1 prerequisites are met and resources allocated for 13-18 week development cycle.
