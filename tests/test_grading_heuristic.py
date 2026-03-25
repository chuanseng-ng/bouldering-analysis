"""Tests for src.grading.heuristic module.

Covers:
- src/grading/heuristic.py  — HeuristicGradeResult, estimate_grade_heuristic()
                               and all private helpers (_clamp, _combine_scores,
                               _compute_confidence, _compute_geometry_difficulty,
                               _compute_hold_difficulty, _score_to_grade_index)
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.features.assembler import RouteFeatures, assemble_features
from src.features.exceptions import FeatureExtractionError
from src.graph.constraints import apply_route_constraints
from src.graph.route_graph import RouteGraph, build_route_graph
from src.graph.types import ClassifiedHold
from src.grading.constants import (
    FEATURE_WEIGHTS,
    MAX_HOPS_NORM,
    MAX_MOVE_DISTANCE,
    V_GRADES,
)
from src.grading.exceptions import GradeEstimationError
from src.grading.heuristic import (
    HeuristicGradeResult,
    _clamp,
    _combine_scores,
    _compute_confidence,
    _compute_geometry_difficulty,
    _compute_hold_difficulty,
    _score_to_grade_index,
    estimate_grade_heuristic,
)
from tests.conftest import make_classified_hold_for_tests as _make_classified_hold


# ---------------------------------------------------------------------------
# Local test helpers
# ---------------------------------------------------------------------------


def _make_constrained_graph(
    holds: list[ClassifiedHold],
    start_ids: list[int],
    finish_id: int,
    wall_angle: float = 0.0,
) -> RouteGraph:
    """Build a constrained RouteGraph for heuristic tests.

    Args:
        holds: List of ClassifiedHold instances.
        start_ids: List of hold IDs designating start holds.
        finish_id: Hold ID designating the finish hold.
        wall_angle: Wall inclination in degrees (default 0.0).

    Returns:
        A constrained RouteGraph with start/finish attributes set.
    """
    rg = build_route_graph(holds, wall_angle)
    return apply_route_constraints(rg, list(start_ids), finish_id)


def _make_route_features(
    holds: list[ClassifiedHold] | None = None,
    start_ids: list[int] | None = None,
    finish_id: int = 1,
) -> RouteFeatures:
    """Build a RouteFeatures instance for heuristic tests.

    Args:
        holds: List of ClassifiedHold instances. Defaults to two jug holds.
        start_ids: Start hold IDs. Defaults to [0].
        finish_id: Finish hold ID. Defaults to 1.

    Returns:
        A RouteFeatures instance.
    """
    if holds is None:
        holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.35, y_center=0.5, hold_type="jug"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.65, y_center=0.5, hold_type="jug"
            ),
        ]
    if start_ids is None:
        start_ids = [0]
    crg = _make_constrained_graph(holds, start_ids=start_ids, finish_id=finish_id)
    return assemble_features(crg)


def _make_vec(**overrides: float) -> dict[str, float]:
    """Build a minimal feature vector with sensible defaults.

    Returns all-zero values except for the provided overrides.

    Args:
        **overrides: Key-value pairs to set in the vector.

    Returns:
        A dict[str, float] suitable for passing to private helpers.
    """
    base: dict[str, float] = {
        "crimp_ratio": 0.0,
        "sloper_ratio": 0.0,
        "pinch_ratio": 0.0,
        "jug_ratio": 0.0,
        "edges_ratio": 0.0,
        "pocket_ratio": 0.0,
        "avg_move_distance": 0.0,
        "max_move_distance": 0.0,
        "path_length_max_hops": 0.0,
        "node_count": 2.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestHeuristicGradeResult
# ---------------------------------------------------------------------------


class TestHeuristicGradeResult:
    """Tests for HeuristicGradeResult Pydantic model validation."""

    def test_valid_model_creates_instance(self) -> None:
        """Valid fields must construct a HeuristicGradeResult."""
        result = HeuristicGradeResult(
            grade="V3",
            grade_index=3,
            confidence=0.8,
            difficulty_score=0.19,
        )
        assert isinstance(result, HeuristicGradeResult)

    def test_grade_field_stored_correctly(self) -> None:
        """grade field must be returned as provided."""
        result = HeuristicGradeResult(
            grade="V5", grade_index=5, confidence=0.9, difficulty_score=0.3
        )
        assert result.grade == "V5"

    def test_grade_index_min_boundary_accepted(self) -> None:
        """grade_index=0 must be accepted."""
        result = HeuristicGradeResult(
            grade="V0", grade_index=0, confidence=0.7, difficulty_score=0.01
        )
        assert result.grade_index == 0

    def test_grade_index_max_boundary_accepted(self) -> None:
        """grade_index=17 must be accepted."""
        result = HeuristicGradeResult(
            grade="V17", grade_index=17, confidence=0.6, difficulty_score=0.99
        )
        assert result.grade_index == 17

    def test_grade_index_below_min_raises(self) -> None:
        """grade_index=-1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            HeuristicGradeResult(
                grade="V0", grade_index=-1, confidence=0.7, difficulty_score=0.0
            )

    def test_grade_index_above_max_raises(self) -> None:
        """grade_index=18 must raise ValidationError."""
        with pytest.raises(ValidationError):
            HeuristicGradeResult(
                grade="V17", grade_index=18, confidence=0.7, difficulty_score=1.0
            )

    def test_confidence_min_boundary_accepted(self) -> None:
        """confidence=0.5 must be accepted."""
        result = HeuristicGradeResult(
            grade="V0", grade_index=0, confidence=0.5, difficulty_score=0.0
        )
        assert result.confidence == pytest.approx(0.5)

    def test_confidence_below_min_raises(self) -> None:
        """confidence=0.49 must raise ValidationError."""
        with pytest.raises(ValidationError):
            HeuristicGradeResult(
                grade="V0", grade_index=0, confidence=0.49, difficulty_score=0.0
            )

    def test_frozen_model_raises_on_assign(self) -> None:
        """Assigning a field must raise ValidationError (frozen=True)."""
        result = HeuristicGradeResult(
            grade="V3", grade_index=3, confidence=0.8, difficulty_score=0.19
        )
        with pytest.raises(ValidationError):
            result.grade = "V4"  # type: ignore[misc]

    def test_inconsistent_grade_and_index_raises_validation_error(self) -> None:
        """grade/grade_index mismatch must raise ValidationError."""
        with pytest.raises(ValidationError):
            HeuristicGradeResult(
                grade="V0", grade_index=17, confidence=0.7, difficulty_score=0.9
            )


# ---------------------------------------------------------------------------
# TestClamp
# ---------------------------------------------------------------------------


class TestClamp:
    """Tests for _clamp() private helper."""

    def test_value_below_lo_returns_lo(self) -> None:
        """Value below lower bound must be clamped to lo."""
        assert _clamp(-1.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_value_above_hi_returns_hi(self) -> None:
        """Value above upper bound must be clamped to hi."""
        assert _clamp(2.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_value_within_range_returned_unchanged(self) -> None:
        """Value within [lo, hi] must be returned unchanged."""
        assert _clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_value_at_lo_boundary_returned(self) -> None:
        """Value equal to lo must be returned unchanged."""
        assert _clamp(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_value_at_hi_boundary_returned(self) -> None:
        """Value equal to hi must be returned unchanged."""
        assert _clamp(1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_negative_range_clamped_correctly(self) -> None:
        """_clamp works with negative bounds."""
        assert _clamp(-5.0, -3.0, -1.0) == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# TestComputeHoldDifficulty
# ---------------------------------------------------------------------------


class TestComputeHoldDifficulty:
    """Tests for _compute_hold_difficulty() private helper."""

    def test_all_jugs_returns_zero(self) -> None:
        """All jugs → negative raw → clamped to 0.0."""
        vec = _make_vec(jug_ratio=1.0)
        assert _compute_hold_difficulty(vec) == pytest.approx(0.0)

    def test_all_crimps_returns_weight(self) -> None:
        """All crimps → raw = crimp_ratio weight = 0.35."""
        vec = _make_vec(crimp_ratio=1.0)
        assert _compute_hold_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["crimp_ratio"]
        )

    def test_all_slopers_returns_weight(self) -> None:
        """All slopers → raw = sloper_ratio weight = 0.25."""
        vec = _make_vec(sloper_ratio=1.0)
        assert _compute_hold_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["sloper_ratio"]
        )

    def test_all_pinches_returns_weight(self) -> None:
        """All pinches → raw = pinch_ratio weight = 0.20."""
        vec = _make_vec(pinch_ratio=1.0)
        assert _compute_hold_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["pinch_ratio"]
        )

    def test_all_edges_returns_weight(self) -> None:
        """All edges → raw = edges_ratio weight."""
        vec = _make_vec(edges_ratio=1.0)
        assert _compute_hold_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["edges_ratio"]
        )

    def test_mixed_hold_types_correct_combination(self) -> None:
        """Mixed holds must compute weighted sum correctly."""
        vec = _make_vec(crimp_ratio=0.5, jug_ratio=0.5)
        expected = (
            FEATURE_WEIGHTS["crimp_ratio"] * 0.5 + FEATURE_WEIGHTS["jug_ratio"] * 0.5
        )
        assert _compute_hold_difficulty(vec) == pytest.approx(max(0.0, expected))

    def test_result_clamped_to_one(self) -> None:
        """Result must not exceed 1.0 even with all weight types = 1.0."""
        vec = _make_vec(
            crimp_ratio=1.0,
            sloper_ratio=1.0,
            pinch_ratio=1.0,
            edges_ratio=1.0,
            pocket_ratio=1.0,
        )
        assert _compute_hold_difficulty(vec) <= 1.0


# ---------------------------------------------------------------------------
# TestComputeGeometryDifficulty
# ---------------------------------------------------------------------------


class TestComputeGeometryDifficulty:
    """Tests for _compute_geometry_difficulty() private helper."""

    def test_all_zeros_returns_zero(self) -> None:
        """All-zero geometry → 0.0."""
        vec = _make_vec()
        assert _compute_geometry_difficulty(vec) == pytest.approx(0.0)

    def test_max_avg_move_distance_contributes(self) -> None:
        """avg_move_distance=MAX_MOVE_DISTANCE → normalised to 1.0 → 0.50 contribution."""
        vec = _make_vec(avg_move_distance=MAX_MOVE_DISTANCE)
        assert _compute_geometry_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["avg_move_distance"]
        )

    def test_max_max_move_distance_contributes(self) -> None:
        """max_move_distance=MAX_MOVE_DISTANCE → normalised to 1.0 → 0.30 contribution."""
        vec = _make_vec(max_move_distance=MAX_MOVE_DISTANCE)
        assert _compute_geometry_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["max_move_distance"]
        )

    def test_hops_normalized_by_max_hops_norm(self) -> None:
        """path_length_max_hops=MAX_HOPS_NORM → norm_hops=1.0 → 0.20 contribution."""
        vec = _make_vec(path_length_max_hops=float(MAX_HOPS_NORM))
        assert _compute_geometry_difficulty(vec) == pytest.approx(
            FEATURE_WEIGHTS["path_length_max_hops"]
        )

    def test_hops_beyond_max_capped_at_one(self) -> None:
        """path_length_max_hops > MAX_HOPS_NORM → norm_hops capped at 1.0."""
        vec_max = _make_vec(path_length_max_hops=float(MAX_HOPS_NORM))
        vec_over = _make_vec(path_length_max_hops=float(MAX_HOPS_NORM * 2))
        assert _compute_geometry_difficulty(vec_max) == pytest.approx(
            _compute_geometry_difficulty(vec_over)
        )

    def test_distances_beyond_max_capped_at_one(self) -> None:
        """avg/max move distances > MAX_MOVE_DISTANCE → normalised to 1.0."""
        vec_max = _make_vec(avg_move_distance=MAX_MOVE_DISTANCE)
        vec_over = _make_vec(avg_move_distance=MAX_MOVE_DISTANCE * 2)
        assert _compute_geometry_difficulty(vec_max) == pytest.approx(
            _compute_geometry_difficulty(vec_over)
        )

    def test_all_max_returns_one(self) -> None:
        """All geometry at maximum → score = 1.0."""
        vec = _make_vec(
            avg_move_distance=MAX_MOVE_DISTANCE,
            max_move_distance=MAX_MOVE_DISTANCE,
            path_length_max_hops=float(MAX_HOPS_NORM),
        )
        assert _compute_geometry_difficulty(vec) == pytest.approx(1.0)

    def test_partial_distance_normalised_correctly(self) -> None:
        """avg_move_distance=MAX_MOVE_DISTANCE/2 → 0.25 contribution."""
        vec = _make_vec(avg_move_distance=MAX_MOVE_DISTANCE / 2.0)
        expected = FEATURE_WEIGHTS["avg_move_distance"] * 0.5
        assert _compute_geometry_difficulty(vec) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestCombineScores
# ---------------------------------------------------------------------------


class TestCombineScores:
    """Tests for _combine_scores() private helper."""

    def test_both_zero_returns_zero(self) -> None:
        """hold=0, geometry=0 → 0.0."""
        assert _combine_scores(0.0, 0.0) == pytest.approx(0.0)

    def test_both_one_returns_one(self) -> None:
        """hold=1, geometry=1 → 1.0."""
        assert _combine_scores(1.0, 1.0) == pytest.approx(1.0)

    def test_weights_sum_to_correct_value(self) -> None:
        """_combine_scores(1,0) = hold_weight, _combine_scores(0,1) = geometry_weight."""
        assert _combine_scores(1.0, 0.0) == pytest.approx(
            FEATURE_WEIGHTS["hold_weight"]
        )
        assert _combine_scores(0.0, 1.0) == pytest.approx(
            FEATURE_WEIGHTS["geometry_weight"]
        )

    def test_mixed_scores_correct(self) -> None:
        """_combine_scores(0.4, 0.6) = 0.45*0.4 + 0.55*0.6 = 0.51."""
        expected = 0.45 * 0.4 + 0.55 * 0.6
        assert _combine_scores(0.4, 0.6) == pytest.approx(expected)

    def test_result_clamped_to_zero_one(self) -> None:
        """Result must be in [0, 1]."""
        score = _combine_scores(0.3, 0.5)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestScoreToGradeIndex
# ---------------------------------------------------------------------------


class TestScoreToGradeIndex:
    """Tests for _score_to_grade_index() private helper."""

    def test_zero_maps_to_index_zero(self) -> None:
        """score=0.0 → grade_index=0."""
        assert _score_to_grade_index(0.0) == 0

    def test_one_maps_to_index_seventeen(self) -> None:
        """score=1.0 → grade_index=17 (not 18)."""
        assert _score_to_grade_index(1.0) == 17

    def test_near_one_maps_to_index_seventeen(self) -> None:
        """score=0.9999 → grade_index=17."""
        assert _score_to_grade_index(0.9999) == 17

    def test_mid_range_correct(self) -> None:
        """score=0.5 → grade_index=9 (V9)."""
        assert _score_to_grade_index(0.5) == 9

    def test_all_18_indices_reachable(self) -> None:
        """Every grade index 0–17 must be reachable."""
        indices = {_score_to_grade_index(i / 18.0 + 0.001 / 18) for i in range(18)}
        assert len(indices) == 18

    def test_boundary_between_grades(self) -> None:
        """score at exact grade boundary must map to the higher grade's index."""
        # 1/18 boundary: index 0 ends at 1/18, so score=1/18 maps to index 1
        boundary = 1.0 / 18
        assert _score_to_grade_index(boundary) == 1


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Tests for _compute_confidence() private helper."""

    def test_center_of_grade_interval_returns_one(self) -> None:
        """Score at the exact centre of grade interval → confidence=1.0."""
        interval = 1.0 / 18
        center = 0.5 * interval  # centre of grade 0 interval
        conf = _compute_confidence(center, 0)
        assert conf == pytest.approx(1.0)

    def test_boundary_returns_half(self) -> None:
        """Score at the grade boundary → confidence=0.5."""
        interval = 1.0 / 18
        boundary = interval  # right boundary of grade 0
        conf = _compute_confidence(boundary, 0)
        assert conf == pytest.approx(0.5)

    def test_confidence_clamp_minimum(self) -> None:
        """Confidence must never drop below 0.5."""
        conf = _compute_confidence(1.0, 0)  # extreme misalignment
        assert conf >= 0.5

    def test_confidence_clamp_maximum(self) -> None:
        """Confidence must never exceed 1.0."""
        conf = _compute_confidence(0.0, 0)
        assert conf <= 1.0

    def test_symmetry_around_center(self) -> None:
        """Score equidistant above/below grade centre must yield same confidence."""
        interval = 1.0 / 18
        center = 0.5 * interval
        delta = interval * 0.25
        conf_above = _compute_confidence(center + delta, 0)
        conf_below = _compute_confidence(center - delta, 0)
        assert conf_above == pytest.approx(conf_below)

    def test_mid_grade_center_returns_one(self) -> None:
        """Centre of grade 9 interval → confidence=1.0."""
        interval = 1.0 / 18
        center = (9 + 0.5) * interval
        conf = _compute_confidence(center, 9)
        assert conf == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestEstimateGradeHeuristic
# ---------------------------------------------------------------------------


class TestEstimateGradeHeuristic:
    """Integration tests for estimate_grade_heuristic()."""

    def test_returns_heuristic_grade_result(self) -> None:
        """estimate_grade_heuristic must return a HeuristicGradeResult."""
        rf = _make_route_features()
        result = estimate_grade_heuristic(rf)
        assert isinstance(result, HeuristicGradeResult)

    def test_grade_in_v_grades(self) -> None:
        """result.grade must be a member of V_GRADES."""
        rf = _make_route_features()
        result = estimate_grade_heuristic(rf)
        assert result.grade in V_GRADES

    def test_grade_matches_grade_index(self) -> None:
        """result.grade must equal V_GRADES[result.grade_index]."""
        rf = _make_route_features()
        result = estimate_grade_heuristic(rf)
        assert result.grade == V_GRADES[result.grade_index]

    def test_difficulty_score_in_range(self) -> None:
        """difficulty_score must be in [0, 1]."""
        rf = _make_route_features()
        result = estimate_grade_heuristic(rf)
        assert 0.0 <= result.difficulty_score <= 1.0

    def test_confidence_in_range(self) -> None:
        """confidence must be in [0.5, 1.0]."""
        rf = _make_route_features()
        result = estimate_grade_heuristic(rf)
        assert 0.5 <= result.confidence <= 1.0

    def test_reproducible_same_input(self) -> None:
        """Same RouteFeatures must always produce the same result."""
        rf = _make_route_features()
        r1 = estimate_grade_heuristic(rf)
        r2 = estimate_grade_heuristic(rf)
        assert r1.grade == r2.grade
        assert r1.difficulty_score == pytest.approx(r2.difficulty_score)

    def test_harder_route_scores_higher_or_equal(self) -> None:
        """Crimp-heavy route must score >= jug-heavy route."""
        jug_holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.35, y_center=0.5, hold_type="jug"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.65, y_center=0.5, hold_type="jug"
            ),
        ]
        crimp_holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.35, y_center=0.5, hold_type="crimp"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.65, y_center=0.5, hold_type="crimp"
            ),
        ]
        rf_jug = _make_route_features(holds=jug_holds)
        rf_crimp = _make_route_features(holds=crimp_holds)
        result_jug = estimate_grade_heuristic(rf_jug)
        result_crimp = estimate_grade_heuristic(rf_crimp)
        assert result_crimp.difficulty_score >= result_jug.difficulty_score

    def test_long_moves_increase_difficulty(self) -> None:
        """Route with longer moves must score higher than short moves."""
        short_holds = [
            _make_classified_hold(hold_id=0, x_center=0.47, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.53, y_center=0.5),
        ]
        long_holds = [
            _make_classified_hold(hold_id=0, x_center=0.35, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.65, y_center=0.5),
        ]
        rf_short = _make_route_features(holds=short_holds)
        rf_long = _make_route_features(holds=long_holds)
        r_short = estimate_grade_heuristic(rf_short)
        r_long = estimate_grade_heuristic(rf_long)
        assert r_long.difficulty_score >= r_short.difficulty_score

    def test_all_jugs_tiny_moves_maps_to_v0(self) -> None:
        """All-jug route with tiny moves must map to V0."""
        holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.49, y_center=0.5, hold_type="jug"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.51, y_center=0.5, hold_type="jug"
            ),
        ]
        rf = _make_route_features(holds=holds)
        result = estimate_grade_heuristic(rf)
        assert result.grade_index == 0

    def test_multi_hold_route_succeeds(self) -> None:
        """Route with multiple holds of varied types must succeed."""
        holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.1, y_center=0.1, hold_type="jug"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.3, y_center=0.3, hold_type="crimp"
            ),
            _make_classified_hold(
                hold_id=2, x_center=0.5, y_center=0.5, hold_type="sloper"
            ),
            _make_classified_hold(
                hold_id=3, x_center=0.7, y_center=0.7, hold_type="pinch"
            ),
            _make_classified_hold(
                hold_id=4, x_center=0.9, y_center=0.9, hold_type="pocket"
            ),
        ]
        rf = _make_route_features(holds=holds, start_ids=[0], finish_id=4)
        result = estimate_grade_heuristic(rf)
        assert isinstance(result, HeuristicGradeResult)


# ---------------------------------------------------------------------------
# TestEstimateGradeHeuristicErrors
# ---------------------------------------------------------------------------


class TestEstimateGradeHeuristicErrors:
    """Tests for estimate_grade_heuristic() error handling."""

    def test_feature_extraction_error_wrapped_as_grade_estimation_error(self) -> None:
        """FeatureExtractionError from to_vector() must be wrapped in GradeEstimationError."""
        rf = _make_route_features()
        original = FeatureExtractionError("simulated taxonomy drift")
        with patch.object(type(rf), "to_vector", side_effect=original):
            with pytest.raises(GradeEstimationError) as exc_info:
                estimate_grade_heuristic(rf)
        assert exc_info.value.__cause__ is original

    def test_wrapped_error_message_contains_context(self) -> None:
        """Wrapped GradeEstimationError.message must include the original message."""
        rf = _make_route_features()
        original = FeatureExtractionError("simulated taxonomy drift")
        with patch.object(type(rf), "to_vector", side_effect=original):
            with pytest.raises(GradeEstimationError) as exc_info:
                estimate_grade_heuristic(rf)
        assert "simulated taxonomy drift" in exc_info.value.message
