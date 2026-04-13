"""Tests for src.grading.constants module.

Covers:
- src/grading/constants.py  — V_GRADES, GRADE_THRESHOLDS, MAX_HOPS_NORM,
                               MAX_MOVE_DISTANCE, FEATURE_WEIGHTS
"""

import math

import pytest

from src.grading.constants import (
    FEATURE_WEIGHTS,
    GRADE_THRESHOLDS,
    MAX_HOPS_NORM,
    MAX_MOVE_DISTANCE,
    V_GRADES,
)


# ---------------------------------------------------------------------------
# TestVGrades
# ---------------------------------------------------------------------------


class TestVGrades:
    """Tests for V_GRADES constant."""

    def test_v_grades_has_18_entries(self) -> None:
        """V_GRADES must contain exactly 18 entries."""
        assert len(V_GRADES) == 18

    def test_v_grades_starts_at_v0(self) -> None:
        """First entry must be 'V0'."""
        assert V_GRADES[0] == "V0"

    def test_v_grades_ends_at_v17(self) -> None:
        """Last entry must be 'V17'."""
        assert V_GRADES[-1] == "V17"

    def test_v_grades_are_sequential(self) -> None:
        """V_GRADES entries must follow sequential V-scale labelling."""
        for i, grade in enumerate(V_GRADES):
            assert grade == f"V{i}"

    def test_v_grades_is_tuple(self) -> None:
        """V_GRADES must be a tuple (immutable)."""
        assert isinstance(V_GRADES, tuple)


# ---------------------------------------------------------------------------
# TestGradeThresholds
# ---------------------------------------------------------------------------


class TestGradeThresholds:
    """Tests for GRADE_THRESHOLDS constant."""

    def test_grade_thresholds_has_18_entries(self) -> None:
        """GRADE_THRESHOLDS must contain exactly 18 entries."""
        assert len(GRADE_THRESHOLDS) == 18

    def test_grade_thresholds_spacing(self) -> None:
        """Consecutive thresholds must differ by approximately 1/18."""
        expected_step = 1.0 / 18
        for i in range(1, len(GRADE_THRESHOLDS)):
            diff = GRADE_THRESHOLDS[i] - GRADE_THRESHOLDS[i - 1]
            assert diff == pytest.approx(expected_step, rel=1e-6)

    def test_grade_thresholds_starts_at_zero(self) -> None:
        """First threshold must be 0.0."""
        assert GRADE_THRESHOLDS[0] == pytest.approx(0.0)

    def test_grade_thresholds_length_matches_v_grades(self) -> None:
        """GRADE_THRESHOLDS length must equal V_GRADES length."""
        assert len(GRADE_THRESHOLDS) == len(V_GRADES)


# ---------------------------------------------------------------------------
# TestMaxHopsNorm
# ---------------------------------------------------------------------------


class TestMaxHopsNorm:
    """Tests for MAX_HOPS_NORM constant."""

    def test_is_positive_integer(self) -> None:
        """MAX_HOPS_NORM must be a positive integer."""
        assert isinstance(MAX_HOPS_NORM, int)
        assert MAX_HOPS_NORM > 0

    def test_expected_value(self) -> None:
        """MAX_HOPS_NORM must equal 20 (empirical upper bound)."""
        assert MAX_HOPS_NORM == 20


# ---------------------------------------------------------------------------
# TestMaxMoveDistance
# ---------------------------------------------------------------------------


class TestMaxMoveDistance:
    """Tests for MAX_MOVE_DISTANCE constant."""

    def test_is_positive_float(self) -> None:
        """MAX_MOVE_DISTANCE must be a positive float."""
        assert isinstance(MAX_MOVE_DISTANCE, float)
        assert MAX_MOVE_DISTANCE > 0.0

    def test_expected_value_is_sqrt2(self) -> None:
        """MAX_MOVE_DISTANCE must equal sqrt(2) (diagonal of unit square)."""
        assert MAX_MOVE_DISTANCE == pytest.approx(math.sqrt(2.0))

    def test_greater_than_one(self) -> None:
        """MAX_MOVE_DISTANCE must exceed 1.0 (distances in [0,1]^2 can exceed 1)."""
        assert MAX_MOVE_DISTANCE > 1.0


# ---------------------------------------------------------------------------
# TestFeatureWeights
# ---------------------------------------------------------------------------


class TestFeatureWeights:
    """Tests for FEATURE_WEIGHTS constant."""

    def test_contains_all_hold_type_keys(self) -> None:
        """FEATURE_WEIGHTS must have an entry for each hold type used."""
        for key in (
            "crimp_ratio",
            "sloper_ratio",
            "pinch_ratio",
            "jug_ratio",
            "edges_ratio",
            "pocket_ratio",
        ):
            assert key in FEATURE_WEIGHTS, f"Missing hold type key: {key!r}"

    def test_contains_all_geometry_keys(self) -> None:
        """FEATURE_WEIGHTS must have an entry for each geometry feature used."""
        for key in ("avg_move_distance", "max_move_distance", "path_length_max_hops"):
            assert key in FEATURE_WEIGHTS, f"Missing geometry key: {key!r}"

    def test_contains_mixing_keys(self) -> None:
        """FEATURE_WEIGHTS must have hold_weight and geometry_weight entries."""
        assert "hold_weight" in FEATURE_WEIGHTS
        assert "geometry_weight" in FEATURE_WEIGHTS

    def test_geometry_sub_weights_sum_to_one(self) -> None:
        """avg_move_distance + max_move_distance + path_length_max_hops weights must sum to 1.0."""
        total = (
            FEATURE_WEIGHTS["avg_move_distance"]
            + FEATURE_WEIGHTS["max_move_distance"]
            + FEATURE_WEIGHTS["path_length_max_hops"]
        )
        assert total == pytest.approx(1.0)

    def test_mixing_weights_sum_to_one(self) -> None:
        """hold_weight + geometry_weight must sum to 1.0."""
        total = FEATURE_WEIGHTS["hold_weight"] + FEATURE_WEIGHTS["geometry_weight"]
        assert total == pytest.approx(1.0)

    def test_jug_weight_is_negative(self) -> None:
        """jug_ratio weight must be negative (jugs reduce difficulty)."""
        assert FEATURE_WEIGHTS["jug_ratio"] < 0.0
