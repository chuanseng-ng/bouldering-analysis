# Phase 2: Video Analysis Validation - Future Enhancement

## Executive Summary

This document outlines **Phase 2** of the grade prediction system - a video-based validation mechanism that analyzes actual climb performance to cross-check route-based predictions from Phase 1.

**Core Objective**: Add a secondary validation system that analyzes climber body mechanics and movement patterns from video footage to generate an independent grade prediction. Cross-validation between route-based and video-based predictions helps identify systematic biases, validate accuracy, and improve the algorithm over time.

**Status**: Lower priority future enhancement - to be implemented **only after** Phase 1 deployment and validation

**Priority**: This phase is marked as lower-priority future enhancement. Phase 1 must be completed, deployed, and validated before beginning Phase 2.

---

# PHASE 2: VIDEO ANALYSIS VALIDATION (Future Enhancement - Lower Priority)

## Overview

**Status**: Lower priority future enhancement - to be implemented after Phase 1 deployment and validation

**Purpose**: Add a secondary validation system that analyzes actual climb performance from video footage to cross-check the route-based grade prediction from Phase 1.

**Key Principle**: Video analysis serves as a **validation mechanism**, not a replacement for the route-based algorithm. By analyzing how climbers actually perform on a route (body mechanics, movement patterns, struggle indicators), we can:

1. Validate the accuracy of route-based predictions
2. Identify systematic biases in the Phase 1 algorithm
3. Detect unusual route characteristics not captured by visual hold analysis
4. Improve the algorithm over time using performance data
5. Build confidence in predictions through dual-source validation

**Relationship to Phase 1**: Phase 2 is complementary to Phase 1. The route-based algorithm (Phase 1) predicts difficulty from route characteristics, while the video-based system (Phase 2) measures difficulty from climber performance. Agreement between the two increases confidence; disagreement triggers review.

---

## Video Analysis Objectives

### Primary Objective

Generate an independent grade prediction based on climber performance during the actual climb, focusing on:

- **Body position and posture**: Hip angles, body lean, center of gravity
- **Limb placement**: Hand and foot positioning precision
- **Joint angles**: Elbow, knee, hip, shoulder flexion/extension
- **Movement quality**: Dynamic vs static moves, hesitation, flow
- **Struggle indicators**: Extended rest positions, repeated attempts, grip releases

### Secondary Objectives

1. **Cross-validate route predictions**: Compare video-based grade to route-based grade
2. **Identify edge cases**: Flag routes where predictions diverge significantly (±2 grades)
3. **Collect performance data**: Build dataset for algorithm improvement
4. **Detect route anomalies**: Find routes with unusual characteristics
5. **Provide user feedback**: Explain why grades differ when discrepancies occur

---

## Body Mechanics Metrics

### Pose Estimation Foundation

Use pose estimation models to extract skeletal keypoints from video frames:

**Key body landmarks** (following MediaPipe or OpenPose conventions):

- **Head**: Nose, eyes, ears
- **Torso**: Shoulders, hips, spine
- **Arms**: Shoulders, elbows, wrists, hands
- **Legs**: Hips, knees, ankles, feet

**Frame extraction rate**: 10-15 FPS (sufficient for movement analysis without excessive processing)

### Metric 1: Body Position & Angles

#### Hip-Shoulder Angle (Body Lean)

Measures how far the climber's body is from the wall:

```python
def calculate_hip_shoulder_angle(shoulder_pos, hip_pos, wall_normal):
    """
    Calculate angle between body centerline and wall.
    
    Returns:
        angle: 0° = body flat against wall (slab)
               90° = body perpendicular to wall (overhang)
    """
    body_vector = hip_pos - shoulder_pos
    angle = calculate_angle(body_vector, wall_normal)
    return angle
```

**Difficulty indicators**:

- **Slab climbing**: Body close to wall (angle < 30°) → easier
- **Overhang climbing**: Body far from wall (angle > 60°) → harder, more core tension required
- **Sustained large angles**: Indicates continuous difficulty

#### Knee-Hip-Shoulder Angle (Body Compression)

Measures how compressed/extended the climber's body is:

```text
Compressed body (small angle 60-90°): Using footholds effectively, technical climbing
Extended body (large angle 120-180°): Reaching, dynamic moves, more difficult
```

**Calculation**:

```python
knee_hip_shoulder_angle = calculate_angle_3_points(knee, hip, shoulder)
```

**Difficulty indicators**:

- **Frequent compression**: Good footwork, lower difficulty
- **Extended positions**: Long reaches, higher difficulty
- **Inability to compress**: Poor footholds or overhang, higher difficulty

#### Arm Extension Ratio

Ratio of extended arm length to bent arm:

```python
arm_extension_ratio = (wrist_to_shoulder_distance) / (upper_arm_length + forearm_length)
# Ratio > 0.9 = nearly straight arm (harder, less control)
# Ratio < 0.7 = bent arm (easier, more control)
```

**Difficulty indicators**:

- **Extended arms (ratio > 0.9)**: Struggling to reach, locked off positions → harder
- **Bent arms (ratio < 0.7)**: Controlled movement → easier
- **Frequent full extension**: Indicates route is at climber's limit

### Metric 2: Movement Patterns

#### Dynamic vs Static Movement Detection

Analyze velocity and acceleration of body center of mass:

```python
def classify_movement_type(body_positions_over_time):
    """
    Classify movement as dynamic or static based on velocity.
    
    Dynamic: Rapid acceleration, jumping, swinging
    Static: Controlled, gradual position changes
    """
    velocities = calculate_velocity(body_positions_over_time)
    accelerations = calculate_acceleration(velocities)
    
    if max(accelerations) > DYNAMIC_THRESHOLD:
        return "dynamic"
    else:
        return "static"
```

**Difficulty indicators**:

- **High proportion of dynamic moves**: Harder route (requires power and coordination)
- **Static climbing**: Easier route (controlled technique)
- **Forced dynamic moves**: Route difficulty exceeds climber's static ability

#### Movement Flow & Hesitation

Measure smoothness and continuity of movement:

```python
def calculate_movement_flow(keypoint_positions_over_time):
    """
    Calculate flow score based on movement continuity.
    
    High flow: Smooth, continuous movement
    Low flow: Frequent pauses, hesitation, repositioning
    """
    pauses = detect_pauses(keypoint_positions_over_time)
    jerkiness = calculate_jerk(keypoint_positions_over_time)  # 3rd derivative of position
    
    flow_score = 1.0 / (1 + pauses * 0.1 + jerkiness * 0.5)
    return flow_score
```

**Difficulty indicators**:

- **Low flow score**: Hesitation, uncertainty → route is difficult for climber
- **Frequent pauses**: Figuring out sequences → higher difficulty
- **Smooth flow**: Familiar movements → lower difficulty

#### Pace Analysis

Track time spent on route normalized by route height:

```python
climb_pace = route_height_meters / time_to_complete_seconds
# Slow pace (< 0.5 m/s): Struggling, high difficulty
# Fast pace (> 1.5 m/s): Comfortable, lower difficulty
```

**Difficulty indicators**:

- **Slow pace**: Route is challenging
- **Accelerating pace**: Easier sections or gaining confidence
- **Decelerating pace**: Fatigue or difficult sections

### Metric 3: Rest Positions & Recovery

#### Rest Position Detection

Identify when climber is resting vs actively moving:

```python
def detect_rest_positions(keypoint_positions, velocity_threshold=0.1):
    """
    Detect frames where climber is stationary (resting).
    
    Returns:
        rest_periods: List of (start_frame, end_frame, duration)
    """
    velocities = calculate_velocity(keypoint_positions)
    is_resting = velocities < velocity_threshold
    
    rest_periods = identify_continuous_periods(is_resting)
    return rest_periods
```

**Difficulty indicators**:

- **Frequent long rests**: Route is difficult, climber needs recovery
- **No rest positions**: Either easy route (continuous movement) or very hard (no rest available)
- **Shaking hands during rest**: Pump, fatigue → high difficulty

#### Rest Position Quality

Analyze body position during rests:

```text
Good rest: Straight arms, hips close to wall, weight on feet → route has rest holds
Poor rest: Bent arms, body far from wall → no good rests available, harder route
```

**Scoring**:

```python
def score_rest_quality(hip_shoulder_angle, arm_extension_ratio):
    """
    Score quality of rest position.
    
    Good rest: Low score (body relaxed, straight arms)
    Poor rest: High score (body tense, locked off)
    """
    quality_score = hip_shoulder_angle * 0.5 + (1 - arm_extension_ratio) * 0.5
    return quality_score
```

### Metric 4: Foot Placement & Precision

#### Foot Movement Precision

Analyze precision of foot placements:

```python
def analyze_foot_precision(foot_positions_over_time, hold_positions):
    """
    Measure precision of foot placements.
    
    Returns:
        precision_score: Lower = more precise (fewer adjustments)
    """
    # Detect foot adjustments (multiple movements to same hold)
    adjustments = count_foot_repositionings(foot_positions_over_time)
    
    # Measure foot-hold alignment
    alignment_errors = calculate_alignment_errors(foot_positions_over_time, hold_positions)
    
    precision_score = adjustments * 0.5 + alignment_errors * 0.5
    return precision_score
```

**Difficulty indicators**:

- **Low precision (many adjustments)**: Small footholds, technical difficulty
- **High precision (few adjustments)**: Good footholds or skilled climber
- **Foot slips**: Very small holds or at climber's limit

#### Weight Distribution

Analyze how weight is distributed between hands and feet:

```text
Foot-heavy climbing (>60% weight on feet): Good technique, easier
Hand-heavy climbing (>60% weight on hands): Overhang or poor footwork, harder
```

**Estimation** (from pose):

```python
def estimate_weight_distribution(body_center, hand_positions, foot_positions):
    """
    Estimate weight distribution based on body position relative to supports.
    
    Approximation: Weight shifts toward supports closer to center of mass
    """
    hand_distances = calculate_distances(body_center, hand_positions)
    foot_distances = calculate_distances(body_center, foot_positions)
    
    weight_on_feet_ratio = sum(1/foot_distances) / (sum(1/hand_distances) + sum(1/foot_distances))
    return weight_on_feet_ratio
```

### Metric 5: Grip & Hand Analysis

#### Grip Duration Analysis

Track how long climber holds each grip:

```python
def analyze_grip_durations(hand_positions, hold_positions, timestamps):
    """
    Measure time spent on each hold.
    
    Returns:
        grip_durations: List of hold durations
    """
    grip_events = detect_hand_to_hold_matches(hand_positions, hold_positions)
    durations = calculate_durations(grip_events, timestamps)
    return durations
```

**Difficulty indicators**:

- **Short grip durations (< 1 second)**: Quick releases, hard crimps/pockets
- **Long grip durations (> 5 seconds)**: Good holds or resting
- **Shaking hands**: Pump, fatigue → difficult holds

#### Hand-Hold Distance (Reaching)

Measure distance of reaches:

```python
normalized_reach_distance = reach_distance / arm_span
# Large normalized reaches (> 0.8): Difficult dynos or long moves
# Small reaches (< 0.5): Comfortable static moves
```

**Difficulty indicators**:

- **Frequent long reaches**: Route requires significant span or dynamic moves
- **Maximal reaches**: At climber's physical limit

### Metric 6: Body Tension & Core Engagement

#### Flagging Detection

Detect when climber extends leg outward for balance (flagging):

```python
def detect_flagging(leg_positions, wall_plane):
    """
    Detect flagging (leg extended away from wall for balance).
    
    Flagging indicates technical difficulty and need for precise balance.
    """
    leg_to_wall_distance = calculate_distance_to_plane(leg_positions, wall_plane)
    is_flagging = leg_to_wall_distance > FLAGGING_THRESHOLD
    return is_flagging
```

**Difficulty indicators**:

- **Frequent flagging**: Technical, balanced moves required → higher difficulty
- **No flagging**: Easier route or different technique

#### Core Tension Indicator

Measure body rigidity during moves:

```python
def estimate_core_tension(spine_angle_variance):
    """
    Estimate core engagement from spine stability.
    
    Low variance: Rigid core, lots of tension → harder
    High variance: Relaxed core → easier
    """
    core_tension_score = 1.0 / (1 + spine_angle_variance)
    return core_tension_score
```

---

## Performance-Based Grade Prediction

### Scoring Model

Combine metrics into a performance difficulty score:

```python
def predict_grade_from_video(video_metrics: dict) -> tuple[str, float, dict]:
    """
    Predict grade based on climber performance in video.
    
    Args:
        video_metrics: Dictionary of extracted metrics from video analysis
    
    Returns:
        tuple: (predicted_grade, confidence_score, metric_breakdown)
    """
    # Extract individual metric scores
    body_position_score = score_body_position(video_metrics['hip_shoulder_angles'])
    movement_score = score_movement_quality(video_metrics['movement_flow'], 
                                            video_metrics['dynamic_ratio'])
    rest_score = score_rest_analysis(video_metrics['rest_periods'], 
                                     video_metrics['rest_quality'])
    precision_score = score_foot_precision(video_metrics['foot_precision'])
    reach_score = score_reaching(video_metrics['reach_distances'])
    tempo_score = score_climb_pace(video_metrics['climb_pace'])
    
    # Weighted combination
    performance_score = (
        body_position_score * 0.25 +
        movement_score * 0.20 +
        rest_score * 0.20 +
        precision_score * 0.15 +
        reach_score * 0.10 +
        tempo_score * 0.10
    )
    
    # Map to V-grade
    predicted_grade = map_performance_score_to_grade(performance_score)
    
    # Calculate confidence based on video quality and consistency
    confidence = calculate_video_confidence(video_metrics)
    
    return predicted_grade, confidence, {
        'performance_score': performance_score,
        'body_position': body_position_score,
        'movement_quality': movement_score,
        'rest_analysis': rest_score,
        'foot_precision': precision_score,
        'reaching': reach_score,
        'tempo': tempo_score
    }
```

### Performance Score to Grade Mapping

```python
def map_performance_score_to_grade(performance_score):
    """
    Map performance score (0-12) to V-grade.
    
    Similar mapping to Phase 1, but based on performance indicators.
    """
    grade_mapping = {
        (0, 1.5): "V0",
        (1.5, 2.5): "V1",
        (2.5, 3.5): "V2",
        (3.5, 4.5): "V3",
        (4.5, 5.5): "V4",
        (5.5, 6.5): "V5",
        (6.5, 7.5): "V6",
        (7.5, 8.5): "V7",
        (8.5, 9.5): "V8",
        (9.5, 10.5): "V9",
        (10.5, 11.25): "V10",
        (11.25, 11.75): "V11",
        (11.75, 12): "V12"
    }
    
    for (min_score, max_score), grade in grade_mapping.items():
        if min_score <= performance_score < max_score:
            return grade
    
    return "V12"  # Maximum grade
```

---

## Cross-Validation Mechanism

### Comparison Logic

Compare route-based prediction (Phase 1) to video-based prediction (Phase 2):

```python
def cross_validate_predictions(route_grade, video_grade, route_confidence, video_confidence):
    """
    Compare route-based and video-based predictions.
    
    Returns:
        validation_result: Dict with comparison analysis
    """
    grade_diff = abs(grade_to_numeric(route_grade) - grade_to_numeric(video_grade))
    
    # Determine validation status
    if grade_diff <= 1:
        status = "VALID"
        message = "Route and performance predictions agree within acceptable margin"
    elif grade_diff == 2:
        status = "REVIEW"
        message = "Moderate discrepancy detected - flagged for review"
    else:  # grade_diff >= 3
        status = "SIGNIFICANT_DISCREPANCY"
        message = "Large discrepancy detected - requires investigation"
    
    # Calculate combined confidence
    combined_confidence = (route_confidence + video_confidence) / 2
    if grade_diff <= 1:
        combined_confidence *= 1.1  # Boost confidence when predictions agree
    else:
        combined_confidence *= 0.8  # Reduce confidence when predictions disagree
    
    combined_confidence = min(combined_confidence, 1.0)
    
    return {
        'status': status,
        'message': message,
        'route_grade': route_grade,
        'video_grade': video_grade,
        'grade_difference': grade_diff,
        'route_confidence': route_confidence,
        'video_confidence': video_confidence,
        'combined_confidence': combined_confidence,
        'recommended_grade': determine_recommended_grade(
            route_grade, video_grade, route_confidence, video_confidence
        )
    }
```

### Threshold Definitions

**Acceptable Margin**: ±1 grade difference

- Routes where video-based and route-based predictions differ by 0-1 grades
- **Action**: Accept as valid, high confidence
- **Reasoning**: Natural variance in climbing

**Review Threshold**: ±2 grades difference

- Routes where predictions differ by exactly 2 grades
- **Action**: Flag for manual review, medium confidence
- **Reasoning**: Could indicate route characteristics not captured or climber skill variance

**Significant Discrepancy**: ±3+ grades difference

- Routes where predictions differ by 3 or more grades
- **Action**: Flag as high-priority for investigation, low confidence
- **Reasoning**: Likely indicates systematic error or unusual route

### Discrepancy Logging & Reporting

Store all discrepancies in database:

```python
# GradeDiscrepancy model (add to models.py)
class GradeDiscrepancy(Base):
    """Log grade prediction discrepancies for algorithm improvement."""
    __tablename__ = 'grade_discrepancies'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'))
    route_grade = Column(String(10))
    video_grade = Column(String(10))
    grade_difference = Column(Integer)
    status = Column(String(50))  # VALID, REVIEW, SIGNIFICANT_DISCREPANCY
    probable_causes = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## Technical Requirements

### Pose Estimation Libraries

**Recommended: MediaPipe** (Option 1)

- Free, open-source
- Fast inference (real-time capable)
- Good accuracy for full-body pose
- 33 body landmarks including hands
- Cross-platform support

**Alternative: OpenPose** (if higher accuracy needed)

- Very accurate multi-person pose
- Slower inference, requires GPU
- Detailed keypoints

**Alternative: AlphaPose** (advanced implementation)

- Excellent accuracy
- Handles occlusions well
- More complex installation

### Video Processing Requirements

**Supported formats**: MP4, MOV, AVI, WEBM

**Minimum requirements**:

- Resolution: 720p (1280x720) or higher
- Frame rate: 24 FPS or higher
- Duration: Full route completion
- Angle: Side view or slight angle
- Lighting: Climber clearly visible

### Storage Requirements

**Video files**: 10-100 MB each

- Local filesystem for processing
- Optional cloud archive (AWS S3, Google Cloud Storage)
- Retention policy (e.g., 90 days)

**Pose data**: ~1-5 MB per video

- Store landmarks in separate JSON file
- Link via analysis_id

### Dependencies

```python
# requirements.txt additions for Phase 2
mediapipe>=0.10.0  # Pose estimation
opencv-python>=4.8.0  # Video processing
moviepy>=1.0.3  # Video editing utilities
```

---

## Data Model Additions

### Analysis Model Extensions

Use existing `features_extracted` JSON field to store video analysis data:

```json
{
  "video_analysis": {
    "video_filename": "route_climb_20260104.mp4",
    "video_duration_seconds": 45.3,
    "video_quality_score": 0.85,
    "pose_data_file": "data/pose_landmarks/abc123.json",
    "metrics": {
      "body_position": {...},
      "movement": {...},
      "rest_analysis": {...}
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
    "status": "REVIEW",
    "recommended_grade": "V6"
  }
}
```

### Configuration Storage

Add to `user_config.yaml`:

```yaml
video_analysis:
  enabled: false  # Phase 2 feature flag
  max_video_size_mb: 200
  allowed_formats: ['.mp4', '.mov', '.avi', '.webm']
  min_resolution: [1280, 720]
  pose_model: "mediapipe"
  target_analysis_fps: 10
  
  # Cross-validation thresholds
  validation:
    acceptable_margin: 1  # grades
    review_threshold: 2  # grades
    significant_discrepancy: 3  # grades
```

---

## Implementation Timeline

### Phase 2 Development Stages

**Important**: Phase 2 should only begin after Phase 1 is deployed, validated, and stable.

#### Stage 1: Research & Prototyping (2-3 weeks)

- Collect sample climbing videos
- Test MediaPipe accuracy
- Prototype metric extraction
- Document findings

#### Stage 2: Core Video Analysis Implementation (3-4 weeks)

- Implement video upload and validation
- Integrate pose estimation
- Extract all metrics
- Generate performance-based predictions
- Create unit tests

#### Stage 3: Cross-Validation System (2-3 weeks)

- Implement comparison logic
- Build discrepancy detection
- Create logging system
- User notification system

#### Stage 4: UI Integration (2 weeks)

- Add video upload form
- Display video analysis results
- Show cross-validation comparison

#### Stage 5: Testing & Calibration (3-4 weeks)

- Test on diverse videos
- Calibrate metric weights
- Validate accuracy
- Edge case testing

#### Stage 6: Deployment & Monitoring (1-2 weeks)

- Deploy Phase 2 system
- Set up monitoring
- Collect user feedback

**Total Estimated Timeline**: 13-18 weeks (3-4.5 months)

---

## Priority & Dependencies

### Priority Classification

**Phase 2 is LOWER PRIORITY than Phase 1** for these reasons:

1. **Phase 1 must be proven first**: Route-based algorithm needs validation before adding complexity
2. **Resource intensive**: Video processing requires significant compute and storage
3. **User adoption**: Users need to upload videos (additional friction)
4. **Complexity**: More moving parts, more potential failure points

### Success Criteria for Phase 2 Start

**Do NOT begin Phase 2 until**:

✅ Phase 1 deployed to production
✅ Phase 1 accuracy ≥ 70% within ±1 grade
✅ Phase 1 algorithm stable (no major bugs)
✅ User feedback collection operational
✅ Development resources available
✅ Sample climbing videos collected
✅ Storage and compute resources allocated

### Optional Feature Flag

Deploy Phase 2 as **optional feature** initially:

```python
# In configuration
enable_video_analysis = False  # Default: disabled
```

This allows:

- Gradual rollout to select users
- Testing in production without affecting core functionality
- Easy rollback if issues arise
- Separate monitoring and optimization

---

## Appendix: References

### Related Files

- [`src/main.py`](../src/main.py) - Current implementation, analyze_image function
- [`src/models.py`](../src/models.py) - Database models (Analysis, DetectedHold, HoldType)
- [`src/cfg/user_config.yaml`](../src/cfg/user_config.yaml) - Configuration file
- [`phase1_route_based_prediction.md`](phase1_route_based_prediction.md) - Phase 1 implementation guide

### Video Analysis References

- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose.html
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- AlphaPose: https://github.com/MVIG-SJTU/AlphaPose

### Movement Analysis Concepts

- Biomechanics of climbing performance
- Pose estimation from computer vision
- Performance metrics from sports science
- Movement pattern recognition

### Cross-Validation Methodology

- Multi-source prediction validation
- Discrepancy detection and resolution
- Confidence score combination
- Algorithmic bias identification

---

## Summary

**Phase 2 Status**: Lower priority future enhancement

**Key Features**:

- Video-based performance analysis using pose estimation
- Independent grade prediction from climber movement patterns
- Cross-validation between route-based and video-based predictions
- Discrepancy detection and logging for algorithm improvement

**Prerequisites**:

- Phase 1 must be deployed and validated
- Sufficient resources for video processing infrastructure
- Sample climbing videos for testing and calibration

**Benefits**:

- Dual-source validation increases prediction confidence
- Identifies systematic biases in route-based algorithm
- Provides performance data for continuous improvement
- Detects unusual routes or climbing styles

**Timeline**: 13-18 weeks after Phase 1 validation

**Recommendation**: Focus on Phase 1 first. Consider Phase 2 only after Phase 1 has proven successful in production.

