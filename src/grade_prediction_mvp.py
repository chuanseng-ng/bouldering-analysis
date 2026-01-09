"""
Phase 1a MVP Grade Prediction Algorithm.

This module implements a multi-factor algorithm to predict bouldering route difficulty
(V0-V12) based on detected hold characteristics, density, distances, and wall angle.

Implementation follows the Phase 1a MVP specification with simplifications:
- No slanted hold detection (assumes horizontal)
- Constant 60/40 handhold/foothold weighting (not wall-angle-dependent)
- No complexity multipliers
- Single wall angle (no segments)

Factors evaluated:
1. Hold Difficulty (35% weight): Hold types and sizes
2. Hold Density (25% weight): Number of available holds
3. Hold Distances (20% weight): Spacing between holds
4. Wall Incline (20% weight): Wall angle impact

See plans/phase1/phase1a_mvp_specification.md for full details.
"""

from __future__ import annotations

import math
import statistics
import logging
from typing import Any

from src.config import load_config

logger = logging.getLogger(__name__)


# ===========================
# Factor 1: Hold Difficulty
# ===========================


def get_size_modifier(hold_type: str, area: float) -> float:
    """
    Calculate size-based difficulty modifier for a hold.

    Simplified size categories (3 instead of 5) for MVP.

    Args:
        hold_type: Type of hold (crimp, jug, sloper, etc.)
        area: Hold area in pixels² (bbox width × height)

    Returns:
        Modifier to add to base score (0-2 range)
    """
    config = load_config()
    thresholds = config.get("grade_prediction", {}).get("size_thresholds", {})

    if hold_type in ["crimp", "pocket"]:
        if area < thresholds.get("crimp_small", 1000):  # Small
            return 2
        if area < thresholds.get("crimp_large", 2500):  # Medium
            return 1
        return 0  # Large

    if hold_type == "sloper":
        if area < thresholds.get("sloper_small", 2000):  # Small
            return 2
        return 0  # Large

    if hold_type == "jug":
        if area < thresholds.get("jug_small", 2000):  # Small jug
            return 1
        return 0  # True jug

    # pinch, start-hold, top-out-hold
    return 0


def calculate_handhold_difficulty(handholds: list[Any]) -> float:
    """
    Calculate handhold difficulty score.

    SIMPLIFIED MVP: No hard_hold_ratio multiplier.

    Args:
        handholds: List of handhold objects with .name and area attributes

    Returns:
        Difficulty score in range 1-13
    """
    if len(handholds) == 0:
        return 6.0  # Default neutral

    config = load_config()
    base_scores = config.get("grade_prediction", {}).get("handhold_base_scores", {})

    total_score = 0.0
    for hold in handholds:
        # Get hold type name
        hold_type = hold.name if hasattr(hold, "name") else hold.get("name", "jug")

        # Calculate hold area
        if hasattr(hold, "bbox_x1"):
            area = (hold.bbox_x2 - hold.bbox_x1) * (hold.bbox_y2 - hold.bbox_y1)
        else:
            area = hold.get("area", 2000)  # Default medium size

        base_score = base_scores.get(hold_type, 5)
        size_modifier = get_size_modifier(hold_type, area)
        total_score += base_score + size_modifier

    # Normalize by count
    avg_difficulty = total_score / len(handholds)

    # Clamp to 1-13 range
    return max(1.0, min(13.0, avg_difficulty))


def calculate_foothold_difficulty(footholds: list[Any]) -> float:
    """
    Calculate foothold difficulty score.

    SIMPLIFIED MVP: Basic size categories, simple scarcity multiplier.

    Args:
        footholds: List of foothold objects

    Returns:
        Difficulty score in range 1-12
    """
    if len(footholds) == 0:
        # NO FOOTHOLDS = CAMPUSING (extreme difficulty)
        return 12.0

    config = load_config()
    thresholds = config.get("grade_prediction", {}).get("size_thresholds", {})

    # Size-based scoring
    total_score = 0.0
    for fh in footholds:
        # Calculate foothold area
        if hasattr(fh, "bbox_x1"):
            area = (fh.bbox_x2 - fh.bbox_x1) * (fh.bbox_y2 - fh.bbox_y1)
        else:
            area = fh.get("area", 2000)

        if area < thresholds.get("foothold_small", 1000):  # Small
            total_score += 8
        elif area < thresholds.get("foothold_large", 2000):  # Medium
            total_score += 5
        else:  # Large
            total_score += 2

    avg_difficulty = total_score / len(footholds)

    # Simplified scarcity multiplier
    if len(footholds) <= 2:
        scarcity = 1.4
    elif len(footholds) <= 4:
        scarcity = 1.2
    else:
        scarcity = 1.0

    return avg_difficulty * scarcity


def calculate_combined_hold_difficulty(
    handholds: list[Any], footholds: list[Any]
) -> float:
    """
    Combine handhold and foothold difficulty.

    SIMPLIFIED MVP: Use constant 60/40 weighting.

    Args:
        handholds: List of handhold objects
        footholds: List of foothold objects

    Returns:
        Combined difficulty score in range ~1-13
    """
    config = load_config()
    handhold_weight = float(
        config.get("grade_prediction", {}).get("handhold_weight", 0.60)
    )
    foothold_weight = float(
        config.get("grade_prediction", {}).get("foothold_weight", 0.40)
    )

    handhold_score = calculate_handhold_difficulty(handholds)
    foothold_score = calculate_foothold_difficulty(footholds)

    combined = (handhold_score * handhold_weight) + (foothold_score * foothold_weight)

    return combined


# ===========================
# Factor 2: Hold Density
# ===========================


def calculate_handhold_density_score(handhold_count: int) -> float:
    """
    Calculate handhold density score using logarithmic relationship.

    Fewer holds = harder (non-linear).

    Args:
        handhold_count: Number of handholds detected

    Returns:
        Density score in range 0-12
    """
    if handhold_count == 0:
        return 12.0

    score = 12 - (math.log2(handhold_count) * 2.5)
    return max(0.0, min(12.0, score))


def calculate_foothold_density_score(foothold_count: int) -> float:
    """
    Calculate foothold density score.

    SIMPLIFIED MVP: Coarse categories instead of fine-grained.

    Args:
        foothold_count: Number of footholds detected

    Returns:
        Density score in range 1-12
    """
    if foothold_count == 0:
        return 12.0
    if foothold_count <= 2:
        return 9.0
    if foothold_count <= 5:
        return 6.0
    if foothold_count <= 8:
        return 3.5
    # 9+
    return 1.5


def calculate_combined_hold_density(handhold_count: int, foothold_count: int) -> float:
    """
    Combine handhold and foothold density scores.

    SIMPLIFIED MVP: Use constant 60/40 weighting.

    Args:
        handhold_count: Number of handholds
        foothold_count: Number of footholds

    Returns:
        Combined density score in range ~1-12
    """
    config = load_config()
    handhold_weight = float(
        config.get("grade_prediction", {}).get("handhold_weight", 0.60)
    )
    foothold_weight = float(
        config.get("grade_prediction", {}).get("foothold_weight", 0.40)
    )

    handhold_density = calculate_handhold_density_score(handhold_count)
    foothold_density = calculate_foothold_density_score(foothold_count)

    combined = (handhold_density * handhold_weight) + (
        foothold_density * foothold_weight
    )

    return combined


# ===========================
# Factor 3: Hold Distances
# ===========================


def calculate_hold_distances(holds: list[Any], image_height: float) -> dict[str, Any]:
    """
    Calculate sequential distances between holds.

    Holds are sorted vertically (bottom to top) and consecutive distances computed.

    Args:
        holds: List of hold objects with bbox coordinates
        image_height: Image height for normalization

    Returns:
        Dictionary with distance metrics:
        {
            'avg_distance': float,
            'max_distance': float,
            'normalized_avg': float,
            'normalized_max': float,
            'distances': list[float]
        }
    """
    if len(holds) < 2:
        return {
            "avg_distance": 0,
            "max_distance": 0,
            "normalized_avg": 0,
            "normalized_max": 0,
            "distances": [],
        }

    # Sort holds by y-coordinate (bottom to top)
    sorted_holds = sorted(
        holds, key=lambda h: h.bbox_y1 if hasattr(h, "bbox_y1") else h.get("bbox_y1", 0)
    )

    distances = []
    for i in range(len(sorted_holds) - 1):
        h1, h2 = sorted_holds[i], sorted_holds[i + 1]

        # Calculate centers
        if hasattr(h1, "bbox_x1"):
            x1 = (h1.bbox_x1 + h1.bbox_x2) / 2
            y1 = (h1.bbox_y1 + h1.bbox_y2) / 2
            x2 = (h2.bbox_x1 + h2.bbox_x2) / 2
            y2 = (h2.bbox_y1 + h2.bbox_y2) / 2
        else:
            x1, y1 = h1.get("center_x", 0), h1.get("center_y", 0)
            x2, y2 = h2.get("center_x", 0), h2.get("center_y", 0)

        # Euclidean distance
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(dist)

    if not distances:
        return {
            "avg_distance": 0,
            "max_distance": 0,
            "normalized_avg": 0,
            "normalized_max": 0,
            "distances": [],
        }

    avg_dist = statistics.mean(distances)
    max_dist = max(distances)

    return {
        "avg_distance": avg_dist,
        "max_distance": max_dist,
        "normalized_avg": avg_dist / image_height if image_height > 0 else 0,
        "normalized_max": max_dist / image_height if image_height > 0 else 0,
        "distances": distances,
    }


def calculate_distance_score(distance_metrics: dict[str, Any]) -> float:
    """
    Score based on hold spacing.

    SIMPLIFIED MVP: Coarse thresholds, simple formula.

    Args:
        distance_metrics: Output from calculate_hold_distances()

    Returns:
        Distance difficulty score in range 1-12
    """
    if not distance_metrics.get("distances"):
        return 12.0  # No holds to measure = campusing

    normalized_avg = distance_metrics["normalized_avg"]
    normalized_max = distance_metrics["normalized_max"]

    # Average distance component (0-8)
    if normalized_avg < 0.12:
        avg_component = 1
    elif normalized_avg < 0.20:
        avg_component = 3
    elif normalized_avg < 0.30:
        avg_component = 5
    else:
        avg_component = 8

    # Max distance component (crux bonus: 0-4)
    if normalized_max < 0.25:
        max_component = 0
    elif normalized_max < 0.40:
        max_component = 2
    else:
        max_component = 4

    return min(avg_component + max_component, 12)


def calculate_combined_distance_score(
    handholds: list[Any], footholds: list[Any], image_height: float
) -> float:
    """
    Combine handhold and foothold distance scores.

    SIMPLIFIED MVP: Use constant 60/40 weighting.

    Args:
        handholds: List of handhold objects
        footholds: List of foothold objects
        image_height: Image height for normalization

    Returns:
        Combined distance score in range ~1-12
    """
    config = load_config()
    handhold_weight = float(
        config.get("grade_prediction", {}).get("handhold_weight", 0.60)
    )
    foothold_weight = float(
        config.get("grade_prediction", {}).get("foothold_weight", 0.40)
    )

    handhold_distances = calculate_hold_distances(handholds, image_height)
    foothold_distances = calculate_hold_distances(footholds, image_height)

    handhold_score = calculate_distance_score(handhold_distances)
    foothold_score = calculate_distance_score(foothold_distances)

    combined = (handhold_score * handhold_weight) + (foothold_score * foothold_weight)

    return combined


# ===========================
# Factor 4: Wall Incline
# ===========================


def calculate_wall_incline_score(wall_incline: str) -> float:
    """
    Score based on wall angle.

    SIMPLIFIED MVP: Manual input, single angle only (no segments).

    Args:
        wall_incline: Wall angle category ('slab', 'vertical', 'slight_overhang',
                      'moderate_overhang', 'steep_overhang')

    Returns:
        Wall incline difficulty score in range 3-11
    """
    config = load_config()
    wall_scores = config.get("grade_prediction", {}).get("wall_incline_scores", {})

    # Default to vertical if unknown
    return float(wall_scores.get(wall_incline, 6.0))


# ===========================
# Grade Mapping
# ===========================


def map_score_to_grade(score: float) -> str:
    """
    Map final score (0-12) to V-grade (V0-V12).

    Args:
        score: Final combined difficulty score

    Returns:
        V-grade string (e.g., "V5")
    """
    if score < 1.0:
        return "V0"
    if score < 2.0:
        return "V1"
    if score < 3.0:
        return "V2"
    if score < 4.0:
        return "V3"
    if score < 4.5:
        return "V4"
    if score < 5.5:
        return "V5"
    if score < 6.5:
        return "V6"
    if score < 7.5:
        return "V7"
    if score < 8.5:
        return "V8"
    if score < 9.5:
        return "V9"
    if score < 10.5:
        return "V10"
    if score < 11.5:
        return "V11"
    # score >= 11.5
    return "V12"


# ===========================
# Main Prediction Function
# ===========================


def predict_grade_v2_mvp(
    detected_holds: list[Any],
    wall_incline: str = "vertical",
    image_height: float = 1080,
) -> tuple[str, float, dict[str, Any]]:
    """
    MVP grade prediction - simplified Phase 1a.

    Args:
        detected_holds: List of DetectedHold objects with attributes:
                       - name: hold type name
                       - bbox_x1, bbox_y1, bbox_x2, bbox_y2: bounding box
                       - confidence: detection confidence
        wall_incline: One of ['slab', 'vertical', 'slight_overhang',
                      'moderate_overhang', 'steep_overhang']
        image_height: Image height for distance normalization

    Returns:
        tuple: (predicted_grade, confidence, score_breakdown)
        - predicted_grade: V-grade string (e.g., "V5")
        - confidence: Prediction confidence (0-1)
        - score_breakdown: Dictionary with detailed scoring info
    """
    logger.info(
        "Starting grade prediction for %d holds, wall: %s",
        len(detected_holds),
        wall_incline,
    )

    # Separate handholds and footholds
    handholds = []
    footholds = []
    for h in detected_holds:
        # Check if object or dict
        if hasattr(h, "name"):
            hold_name = h.name
        elif isinstance(h, dict):
            hold_name = h.get("name", "")
        else:
            hold_name = ""

        if hold_name == "foot-hold":
            footholds.append(h)
        else:
            handholds.append(h)

    logger.info(
        "Separated into %d handholds and %d footholds", len(handholds), len(footholds)
    )

    # Calculate 4 factor scores
    hold_difficulty_score = calculate_combined_hold_difficulty(handholds, footholds)
    hold_density_score = calculate_combined_hold_density(len(handholds), len(footholds))
    distance_score = calculate_combined_distance_score(
        handholds, footholds, image_height
    )
    wall_incline_score = calculate_wall_incline_score(wall_incline)

    logger.info(
        "Factor scores - Hold: %.2f, Density: %.2f, Distance: %.2f, Wall: %.2f",
        hold_difficulty_score,
        hold_density_score,
        distance_score,
        wall_incline_score,
    )

    # Load weights from config
    config = load_config()
    weights = config.get("grade_prediction", {}).get("weights", {})

    # Weighted combination
    base_score = (
        hold_difficulty_score * weights.get("hold_difficulty", 0.35)
        + hold_density_score * weights.get("hold_density", 0.25)
        + distance_score * weights.get("distance", 0.20)
        + wall_incline_score * weights.get("wall_incline", 0.20)
    )

    # NO MULTIPLIERS IN MVP - keep it simple
    final_score = base_score

    logger.info("Base score: %.2f, Final score: %.2f", base_score, final_score)

    # Confidence based on detection quality
    if detected_holds:
        confidences = [
            h.confidence if hasattr(h, "confidence") else h.get("confidence", 0.7)
            for h in detected_holds
        ]
        confidence_avg = statistics.mean(confidences)
        confidence = min(confidence_avg / 0.7, 1.0)
    else:
        confidence = 0.5  # Low confidence with no detections

    # Map to grade
    predicted_grade = map_score_to_grade(final_score)

    logger.info(
        "Predicted grade: %s (score: %.2f, confidence: %.2f)",
        predicted_grade,
        final_score,
        confidence,
    )

    # Score breakdown for explainability
    breakdown = {
        "hold_difficulty": round(hold_difficulty_score, 2),
        "hold_density": round(hold_density_score, 2),
        "distance": round(distance_score, 2),
        "wall_incline": round(wall_incline_score, 2),
        "base_score": round(base_score, 2),
        "final_score": round(final_score, 2),
        "handhold_count": len(handholds),
        "foothold_count": len(footholds),
        "wall_angle": wall_incline,
        "algorithm_version": "v2_mvp",
        "weights": weights,
    }

    return predicted_grade, confidence, breakdown
