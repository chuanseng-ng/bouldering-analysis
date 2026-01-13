"""
Tests for the Phase 1a MVP grade prediction algorithm.

Tests cover all four factors and the main prediction function.
"""

from __future__ import annotations

from src.grade_prediction_mvp import (
    calculate_handhold_difficulty,
    calculate_foothold_difficulty,
    calculate_combined_hold_difficulty,
    calculate_handhold_density_score,
    calculate_foothold_density_score,
    calculate_combined_hold_density,
    calculate_hold_distances,
    calculate_distance_score,
    calculate_combined_distance_score,
    calculate_wall_incline_score,
    map_score_to_grade,
    predict_grade_v2_mvp,
    get_size_modifier,
)


# Helper class for creating mock hold objects
class MockHold:
    """Mock hold object for testing."""

    def __init__(
        self,
        name: str,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        confidence: float = 0.8,
    ):
        self.name = name
        self.bbox_x1 = bbox_x1
        self.bbox_y1 = bbox_y1
        self.bbox_x2 = bbox_x2
        self.bbox_y2 = bbox_y2
        self.confidence = confidence


# ===========================
# Factor 1: Hold Difficulty Tests
# ===========================


def test_get_size_modifier_crimp_small():
    """Test size modifier for small crimps."""
    modifier = get_size_modifier("crimp", 800)  # Small crimp
    assert modifier == 2


def test_get_size_modifier_crimp_medium():
    """Test size modifier for medium crimps."""
    modifier = get_size_modifier("crimp", 1500)  # Medium crimp
    assert modifier == 1


def test_get_size_modifier_crimp_large():
    """Test size modifier for large crimps."""
    modifier = get_size_modifier("crimp", 3000)  # Large crimp
    assert modifier == 0


def test_get_size_modifier_jug():
    """Test size modifier for jugs."""
    modifier = get_size_modifier("jug", 1500)  # Small jug
    assert modifier == 1
    modifier = get_size_modifier("jug", 2500)  # Large jug
    assert modifier == 0


def test_calculate_handhold_difficulty_all_crimps():
    """Test handhold difficulty with all crimps."""
    crimps = [MockHold("crimp", 0, i * 100, 30, i * 100 + 30) for i in range(6)]
    score = calculate_handhold_difficulty(crimps)
    assert 10 <= score <= 13  # High difficulty expected


def test_calculate_handhold_difficulty_all_jugs():
    """Test handhold difficulty with all jugs."""
    jugs = [MockHold("jug", 0, i * 100, 50, i * 100 + 50) for i in range(8)]
    score = calculate_handhold_difficulty(jugs)
    assert 1 <= score <= 3  # Low difficulty expected


def test_calculate_handhold_difficulty_empty():
    """Test handhold difficulty with no holds."""
    score = calculate_handhold_difficulty([])
    assert score == 6.0  # Default neutral


def test_calculate_foothold_difficulty_campusing():
    """Test foothold difficulty with no footholds (campusing)."""
    score = calculate_foothold_difficulty([])
    assert score == 12.0  # Maximum difficulty


def test_calculate_foothold_difficulty_small_footholds():
    """Test foothold difficulty with small footholds."""
    small_footholds = [
        MockHold("foot-hold", 0, i * 100, 25, i * 100 + 25) for i in range(3)
    ]
    score = calculate_foothold_difficulty(small_footholds)
    assert score > 8  # High difficulty expected


def test_calculate_foothold_difficulty_large_footholds():
    """Test foothold difficulty with large footholds."""
    large_footholds = [
        MockHold("foot-hold", 0, i * 100, 60, i * 100 + 60) for i in range(8)
    ]
    score = calculate_foothold_difficulty(large_footholds)
    assert score < 3  # Low difficulty expected


def test_calculate_combined_hold_difficulty():
    """Test combined handhold and foothold difficulty."""
    handholds = [MockHold("jug", 0, i * 100, 50, i * 100 + 50) for i in range(8)]
    footholds = [MockHold("foot-hold", 0, i * 100, 50, i * 100 + 50) for i in range(6)]

    score = calculate_combined_hold_difficulty(handholds, footholds)
    assert 1 <= score <= 3  # Low difficulty expected (jugs + large footholds)


# ===========================
# Factor 2: Hold Density Tests
# ===========================


def test_calculate_handhold_density_few_holds():
    """Test handhold density with few holds."""
    score = calculate_handhold_density_score(3)
    assert score > 8  # High difficulty (few holds)


def test_calculate_handhold_density_many_holds():
    """Test handhold density with many holds."""
    score = calculate_handhold_density_score(20)
    assert score < 2  # Low difficulty (many holds)


def test_calculate_handhold_density_zero_holds():
    """Test handhold density with no holds."""
    score = calculate_handhold_density_score(0)
    assert score == 12.0  # Maximum difficulty


def test_calculate_foothold_density_categories():
    """Test foothold density across different categories."""
    assert calculate_foothold_density_score(0) == 12.0
    assert calculate_foothold_density_score(2) == 9.0
    assert calculate_foothold_density_score(5) == 6.0
    assert calculate_foothold_density_score(8) == 3.5
    assert calculate_foothold_density_score(10) == 1.5


def test_calculate_combined_hold_density():
    """Test combined handhold and foothold density."""
    score = calculate_combined_hold_density(10, 6)
    assert 3 <= score <= 6  # Mid-range expected


# ===========================
# Factor 3: Hold Distances Tests
# ===========================


def test_calculate_hold_distances_vertical():
    """Test distance calculation between vertically arranged holds."""
    holds = [
        MockHold("jug", 100, 100, 150, 150),
        MockHold("jug", 100, 300, 150, 350),
        MockHold("jug", 100, 500, 150, 550),
    ]
    image_height = 1000

    metrics = calculate_hold_distances(holds, image_height)

    assert metrics["avg_distance"] > 0
    assert metrics["max_distance"] > 0
    assert 0 < metrics["normalized_avg"] < 1
    assert 0 < metrics["normalized_max"] < 1
    assert len(metrics["distances"]) == 2


def test_calculate_hold_distances_single_hold():
    """Test distance calculation with single hold."""
    holds = [MockHold("jug", 100, 100, 150, 150)]
    metrics = calculate_hold_distances(holds, 1000)

    assert metrics["avg_distance"] == 0
    assert metrics["max_distance"] == 0
    assert len(metrics["distances"]) == 0


def test_calculate_hold_distances_empty():
    """Test distance calculation with no holds."""
    metrics = calculate_hold_distances([], 1000)

    assert metrics["avg_distance"] == 0
    assert metrics["max_distance"] == 0


def test_calculate_distance_score_close():
    """Test distance score for close holds."""
    metrics = {
        "normalized_avg": 0.10,  # Very close
        "normalized_max": 0.15,
        "distances": [100, 110, 90],
    }
    score = calculate_distance_score(metrics)
    assert score < 4  # Low difficulty


def test_calculate_distance_score_far():
    """Test distance score for far apart holds."""
    metrics = {
        "normalized_avg": 0.35,  # Far
        "normalized_max": 0.50,  # Very far
        "distances": [350, 400, 500],
    }
    score = calculate_distance_score(metrics)
    assert score > 8  # High difficulty


def test_calculate_combined_distance_score():
    """Test combined distance score."""
    handholds = [MockHold("jug", 100, i * 200, 150, i * 200 + 50) for i in range(5)]
    footholds = [MockHold("foot-hold", 50, i * 150, 80, i * 150 + 30) for i in range(6)]

    score = calculate_combined_distance_score(handholds, footholds, 1000)
    assert 2 <= score <= 10  # Should be in reasonable range


# ===========================
# Factor 4: Wall Incline Tests
# ===========================


def test_calculate_wall_incline_vertical():
    """Test wall incline score for vertical wall."""
    score = calculate_wall_incline_score("vertical")
    assert score == 6.0


def test_calculate_wall_incline_slab():
    """Test wall incline score for slab."""
    score = calculate_wall_incline_score("slab")
    assert score == 3.0


def test_calculate_wall_incline_overhang():
    """Test wall incline score for moderate overhang."""
    score = calculate_wall_incline_score("moderate_overhang")
    assert score == 9.0


def test_calculate_wall_incline_steep_overhang():
    """Test wall incline score for steep overhang."""
    score = calculate_wall_incline_score("steep_overhang")
    assert score == 11.0


def test_calculate_wall_incline_unknown():
    """Test wall incline score for unknown category."""
    score = calculate_wall_incline_score("unknown_category")
    assert score == 6.0  # Should default to vertical


# ===========================
# Grade Mapping Tests
# ===========================


def test_map_score_to_grade_v0():
    """Test grade mapping for V0."""
    assert map_score_to_grade(0.5) == "V0"


def test_map_score_to_grade_v5():
    """Test grade mapping for V5."""
    assert map_score_to_grade(5.0) == "V5"


def test_map_score_to_grade_v12():
    """Test grade mapping for V12."""
    assert map_score_to_grade(12.0) == "V12"


def test_map_score_to_grade_boundaries():
    """Test grade mapping at boundaries."""
    assert map_score_to_grade(1.0) == "V1"
    assert map_score_to_grade(0.99) == "V0"
    assert map_score_to_grade(11.5) == "V12"
    assert map_score_to_grade(11.49) == "V11"


# ===========================
# Integration Tests
# ===========================


def test_predict_grade_v2_mvp_easy_route():
    """Test prediction for an easy route (jugs, many holds)."""
    holds = [MockHold("jug", 100, i * 100, 150, i * 100 + 50) for i in range(15)]
    # Add footholds
    holds.extend(
        [MockHold("foot-hold", 50, i * 80, 80, i * 80 + 30) for i in range(10)]
    )

    grade, confidence, breakdown = predict_grade_v2_mvp(
        detected_holds=holds, wall_incline="vertical", image_height=1080
    )

    # Should predict lower grades
    assert grade in ["V0", "V1", "V2", "V3"]
    assert 0 <= confidence <= 1.0
    assert "hold_difficulty" in breakdown
    assert "final_score" in breakdown


def test_predict_grade_v2_mvp_hard_route():
    """Test prediction for a hard route (crimps, few holds, overhang)."""
    holds = [MockHold("crimp", 100, i * 200, 130, i * 200 + 30) for i in range(5)]
    # Add few footholds
    holds.extend(
        [MockHold("foot-hold", 50, i * 250, 75, i * 250 + 25) for i in range(2)]
    )

    grade, confidence, breakdown = predict_grade_v2_mvp(
        detected_holds=holds, wall_incline="moderate_overhang", image_height=1080
    )

    # Should predict higher grades
    assert grade in ["V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12"]
    assert 0 <= confidence <= 1.0
    assert breakdown["handhold_count"] == 5
    assert breakdown["foothold_count"] == 2


def test_predict_grade_v2_mvp_campusing():
    """Test prediction for campusing route (no footholds)."""
    holds = [MockHold("crimp", 100, i * 180, 130, i * 180 + 30) for i in range(7)]

    grade, confidence, breakdown = predict_grade_v2_mvp(
        detected_holds=holds, wall_incline="vertical", image_height=1080
    )

    # Should have high difficulty due to campusing
    assert breakdown["foothold_count"] == 0
    # Final score should be influenced by campusing penalty


def test_predict_grade_v2_mvp_slab():
    """Test prediction for slab route."""
    holds = [MockHold("sloper", 100, i * 120, 140, i * 120 + 40) for i in range(10)]
    holds.extend(
        [MockHold("foot-hold", 50, i * 100, 80, i * 100 + 30) for i in range(8)]
    )

    grade, confidence, breakdown = predict_grade_v2_mvp(
        detected_holds=holds, wall_incline="slab", image_height=1080
    )

    # Slab should reduce difficulty
    assert breakdown["wall_angle"] == "slab"
    assert breakdown["wall_incline"] == 3.0  # Slab score


def test_predict_grade_v2_mvp_no_holds():
    """Test prediction with no detected holds."""
    grade, confidence, breakdown = predict_grade_v2_mvp(
        detected_holds=[], wall_incline="vertical", image_height=1080
    )

    # Should handle gracefully with default grade
    assert grade in [
        "V0",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
    ]
    assert 0 <= confidence <= 1.0


def test_predict_grade_v2_mvp_breakdown_structure():
    """Test that breakdown has all expected fields."""
    holds = [MockHold("jug", 100, i * 100, 150, i * 100 + 50) for i in range(8)]

    grade, confidence, breakdown = predict_grade_v2_mvp(
        detected_holds=holds, wall_incline="vertical", image_height=1080
    )

    # Check breakdown structure
    assert "hold_difficulty" in breakdown
    assert "hold_density" in breakdown
    assert "distance" in breakdown
    assert "wall_incline" in breakdown
    assert "base_score" in breakdown
    assert "final_score" in breakdown
    assert "handhold_count" in breakdown
    assert "foothold_count" in breakdown
    assert "wall_angle" in breakdown
    assert "algorithm_version" in breakdown
    assert breakdown["algorithm_version"] == "v2_mvp"


def test_predict_grade_v2_mvp_confidence_calculation():
    """Test confidence calculation based on detection quality."""
    # High confidence detections
    high_conf_holds = [
        MockHold("jug", 100, i * 100, 150, i * 100 + 50, confidence=0.95)
        for i in range(8)
    ]

    grade1, confidence1, _ = predict_grade_v2_mvp(
        detected_holds=high_conf_holds, wall_incline="vertical", image_height=1080
    )

    # Low confidence detections
    low_conf_holds = [
        MockHold("jug", 100, i * 100, 150, i * 100 + 50, confidence=0.5)
        for i in range(8)
    ]

    grade2, confidence2, _ = predict_grade_v2_mvp(
        detected_holds=low_conf_holds, wall_incline="vertical", image_height=1080
    )

    # High confidence detections should have higher prediction confidence
    assert confidence1 > confidence2
