"""Tests for src.explanation module.

Covers:
- src/explanation/exceptions.py  — ExplanationError
- src/explanation/types.py       — FeatureContribution, ExplanationResult
- src/explanation/engine.py      — generate_explanation() and all private helpers
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.explanation.engine import (
    _build_hold_highlights,
    _build_summary,
    _compute_geometry_contributions,
    _compute_hold_contributions,
    _generate_geometry_description,
    _generate_hold_description,
    _get_confidence_qualifier,
    _get_estimator_type,
    _rank_feature_contributions,
    generate_explanation,
)
from src.explanation.exceptions import ExplanationError
from src.explanation.types import ExplanationResult, FeatureContribution
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
from src.grading.heuristic import HeuristicGradeResult
from src.grading.ml_estimator import MLGradeResult
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
    """Build a constrained RouteGraph for explanation tests."""
    rg = build_route_graph(holds, wall_angle)
    return apply_route_constraints(rg, list(start_ids), finish_id)


def _make_route_features(
    holds: list[ClassifiedHold] | None = None,
    start_ids: list[int] | None = None,
    finish_id: int = 1,
) -> RouteFeatures:
    """Build a RouteFeatures instance for explanation tests.

    Defaults to two jug holds at x=0.35/0.65 (distance=0.30, within reach).
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
    """Build a minimal feature vector with all-zero defaults."""
    base: dict[str, float] = {
        "crimp_ratio": 0.0,
        "sloper_ratio": 0.0,
        "pinch_ratio": 0.0,
        "jug_ratio": 0.0,
        "volume_ratio": 0.0,
        "avg_move_distance": 0.0,
        "max_move_distance": 0.0,
        "path_length_max_hops": 0.0,
        "node_count": 2.0,
    }
    base.update(overrides)
    return base


def _make_heuristic_result(
    grade: str = "V3",
    grade_index: int = 3,
    confidence: float = 0.8,
    difficulty_score: float = 0.19,
) -> HeuristicGradeResult:
    """Build a HeuristicGradeResult for tests."""
    return HeuristicGradeResult(
        grade=grade,
        grade_index=grade_index,
        confidence=confidence,
        difficulty_score=difficulty_score,
    )


def _make_ml_result(
    grade: str = "V4",
    grade_index: int = 4,
    confidence: float = 0.73,
    difficulty_score: float = 0.24,
) -> MLGradeResult:
    """Build an MLGradeResult with uniform probabilities for tests."""
    probs = {g: (1.0 / 18) for g in V_GRADES}
    # Give the predicted grade a higher probability to satisfy model_validator
    probs[grade] = 0.5
    # Renormalize
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return MLGradeResult(
        grade=grade,
        grade_index=grade_index,
        confidence=confidence,
        difficulty_score=difficulty_score,
        grade_probabilities=probs,
    )


# ---------------------------------------------------------------------------
# TestExplanationError
# ---------------------------------------------------------------------------


class TestExplanationError:
    """Tests for ExplanationError exception."""

    def test_is_value_error_subclass(self) -> None:
        """ExplanationError must be a subclass of ValueError."""
        assert issubclass(ExplanationError, ValueError)

    def test_message_stored_on_instance(self) -> None:
        """ExplanationError.message must equal the constructor argument."""
        err = ExplanationError("test message")
        assert err.message == "test message"

    def test_str_representation_equals_message(self) -> None:
        """str(ExplanationError) must equal the message."""
        err = ExplanationError("some error")
        assert str(err) == "some error"


# ---------------------------------------------------------------------------
# TestFeatureContribution
# ---------------------------------------------------------------------------


class TestFeatureContribution:
    """Tests for FeatureContribution Pydantic model."""

    def test_valid_model_creates_instance(self) -> None:
        """Valid fields must construct a FeatureContribution."""
        fc = FeatureContribution(
            name="Crimp ratio",
            value=0.4,
            impact=0.14,
            description="40% of holds are crimps.",
        )
        assert isinstance(fc, FeatureContribution)

    def test_frozen_raises_on_assign(self) -> None:
        """Assigning a field must raise ValidationError (frozen=True)."""
        fc = FeatureContribution(
            name="Crimp ratio", value=0.4, impact=0.14, description="test"
        )
        with pytest.raises(ValidationError):
            fc.name = "Other"  # type: ignore[misc]

    def test_negative_impact_stored_correctly(self) -> None:
        """Negative impact (e.g. jugs) must be stored as-is."""
        fc = FeatureContribution(
            name="Jug ratio", value=0.5, impact=-0.15, description="test"
        )
        assert fc.impact == pytest.approx(-0.15)


# ---------------------------------------------------------------------------
# TestExplanationResult
# ---------------------------------------------------------------------------


class TestExplanationResult:
    """Tests for ExplanationResult Pydantic model."""

    def test_valid_heuristic_model_creates_instance(self) -> None:
        """Valid heuristic ExplanationResult must be created."""
        result = ExplanationResult(
            grade="V3",
            estimator_type="heuristic",
            confidence_qualifier="confident",
            top_features=[],
            summary="Summary text.",
            hold_highlights=["crimps (40%)"],
        )
        assert isinstance(result, ExplanationResult)

    def test_valid_ml_model_creates_instance(self) -> None:
        """Valid ML ExplanationResult must be created."""
        result = ExplanationResult(
            grade="V5",
            estimator_type="ml",
            confidence_qualifier="very confident",
            top_features=[],
            summary="Summary text.",
            hold_highlights=[],
        )
        assert result.estimator_type == "ml"

    def test_invalid_estimator_type_raises(self) -> None:
        """Invalid estimator_type must raise ValidationError."""
        with pytest.raises(ValidationError):
            ExplanationResult(
                grade="V3",
                estimator_type="unknown",  # type: ignore[arg-type]
                confidence_qualifier="confident",
                top_features=[],
                summary="Summary.",
                hold_highlights=[],
            )

    def test_invalid_confidence_qualifier_raises(self) -> None:
        """Invalid confidence_qualifier must raise ValidationError."""
        with pytest.raises(ValidationError):
            ExplanationResult(
                grade="V3",
                estimator_type="heuristic",
                confidence_qualifier="extremely confident",  # type: ignore[arg-type]
                top_features=[],
                summary="Summary.",
                hold_highlights=[],
            )

    def test_frozen_raises_on_assign(self) -> None:
        """Assigning a field must raise ValidationError (frozen=True)."""
        result = ExplanationResult(
            grade="V3",
            estimator_type="heuristic",
            confidence_qualifier="confident",
            top_features=[],
            summary="Summary.",
            hold_highlights=[],
        )
        with pytest.raises(ValidationError):
            result.grade = "V5"  # type: ignore[misc]

    def test_top_features_list_stored_correctly(self) -> None:
        """top_features list must be stored as provided."""
        fc = FeatureContribution(
            name="Crimp ratio", value=0.4, impact=0.14, description="desc"
        )
        result = ExplanationResult(
            grade="V3",
            estimator_type="heuristic",
            confidence_qualifier="confident",
            top_features=[fc],
            summary="Summary.",
            hold_highlights=[],
        )
        assert len(result.top_features) == 1
        assert result.top_features[0].name == "Crimp ratio"


# ---------------------------------------------------------------------------
# TestGetConfidenceQualifier
# ---------------------------------------------------------------------------


class TestGetConfidenceQualifier:
    """Tests for _get_confidence_qualifier() private helper."""

    def test_confidence_above_085_returns_very_confident(self) -> None:
        """confidence=0.90 must return 'very confident'."""
        assert _get_confidence_qualifier(0.90) == "very confident"

    def test_confidence_at_085_boundary_returns_very_confident(self) -> None:
        """confidence=0.85 (inclusive >=) must return 'very confident'."""
        assert _get_confidence_qualifier(0.85) == "very confident"

    def test_confidence_below_085_above_065_returns_confident(self) -> None:
        """confidence=0.75 must return 'confident'."""
        assert _get_confidence_qualifier(0.75) == "confident"

    def test_confidence_at_065_boundary_returns_confident(self) -> None:
        """confidence=0.65 (inclusive >=) must return 'confident'."""
        assert _get_confidence_qualifier(0.65) == "confident"

    def test_confidence_below_065_returns_uncertain(self) -> None:
        """confidence=0.60 must return 'uncertain'."""
        assert _get_confidence_qualifier(0.60) == "uncertain"

    def test_confidence_zero_returns_uncertain(self) -> None:
        """confidence=0.0 (ML uniform distribution) must return 'uncertain'."""
        assert _get_confidence_qualifier(0.0) == "uncertain"

    def test_confidence_one_returns_very_confident(self) -> None:
        """confidence=1.0 must return 'very confident'."""
        assert _get_confidence_qualifier(1.0) == "very confident"

    def test_confidence_just_below_085_returns_confident(self) -> None:
        """confidence=0.8499 must return 'confident', not 'very confident'."""
        assert _get_confidence_qualifier(0.8499) == "confident"


# ---------------------------------------------------------------------------
# TestComputeHoldContributions
# ---------------------------------------------------------------------------


class TestComputeHoldContributions:
    """Tests for _compute_hold_contributions() private helper."""

    def test_returns_five_contributions(self) -> None:
        """Must return exactly 5 FeatureContribution instances."""
        vec = _make_vec()
        contribs = _compute_hold_contributions(vec)
        assert len(contribs) == 5

    def test_all_zeros_impact_is_zero(self) -> None:
        """All-zero hold ratios → all impacts must be 0.0."""
        vec = _make_vec()
        contribs = _compute_hold_contributions(vec)
        for c in contribs:
            assert c.impact == pytest.approx(0.0)

    def test_crimp_ratio_impact_matches_weight(self) -> None:
        """crimp_ratio=1.0 → crimp impact = FEATURE_WEIGHTS['crimp_ratio']."""
        vec = _make_vec(crimp_ratio=1.0)
        contribs = {c.name: c for c in _compute_hold_contributions(vec)}
        assert contribs["Crimp ratio"].impact == pytest.approx(
            FEATURE_WEIGHTS["crimp_ratio"]
        )

    def test_jug_ratio_impact_is_negative(self) -> None:
        """jug_ratio=1.0 → jug impact must be negative (reduces difficulty)."""
        vec = _make_vec(jug_ratio=1.0)
        contribs = {c.name: c for c in _compute_hold_contributions(vec)}
        assert contribs["Jug ratio"].impact < 0.0

    def test_all_contributions_have_description(self) -> None:
        """Every contribution must have a non-empty description."""
        vec = _make_vec(crimp_ratio=0.5)
        contribs = _compute_hold_contributions(vec)
        for c in contribs:
            assert len(c.description) > 0


# ---------------------------------------------------------------------------
# TestComputeGeometryContributions
# ---------------------------------------------------------------------------


class TestComputeGeometryContributions:
    """Tests for _compute_geometry_contributions() private helper."""

    def test_returns_three_contributions(self) -> None:
        """Must return exactly 3 FeatureContribution instances."""
        vec = _make_vec()
        contribs = _compute_geometry_contributions(vec)
        assert len(contribs) == 3

    def test_all_zeros_impact_is_zero(self) -> None:
        """All-zero geometry features → all impacts must be 0.0."""
        vec = _make_vec()
        contribs = _compute_geometry_contributions(vec)
        for c in contribs:
            assert c.impact == pytest.approx(0.0)

    def test_max_avg_move_distance_impact_matches_weight(self) -> None:
        """avg_move_distance=MAX_MOVE_DISTANCE → impact = FEATURE_WEIGHTS['avg_move_distance']."""
        vec = _make_vec(avg_move_distance=MAX_MOVE_DISTANCE)
        contribs = {c.name: c for c in _compute_geometry_contributions(vec)}
        assert contribs["Average move distance"].impact == pytest.approx(
            FEATURE_WEIGHTS["avg_move_distance"]
        )

    def test_hops_beyond_max_capped_at_weight(self) -> None:
        """path_length_max_hops >> MAX_HOPS_NORM → impact capped at weight."""
        vec_max = _make_vec(path_length_max_hops=float(MAX_HOPS_NORM))
        vec_over = _make_vec(path_length_max_hops=float(MAX_HOPS_NORM * 5))
        contribs_max = {c.name: c for c in _compute_geometry_contributions(vec_max)}
        contribs_over = {c.name: c for c in _compute_geometry_contributions(vec_over)}
        assert contribs_max["Path length (hops)"].impact == pytest.approx(
            contribs_over["Path length (hops)"].impact
        )


# ---------------------------------------------------------------------------
# TestRankFeatureContributions
# ---------------------------------------------------------------------------


class TestRankFeatureContributions:
    """Tests for _rank_feature_contributions() private helper."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty input must return empty list."""
        assert _rank_feature_contributions([]) == []

    def test_returns_at_most_top_n(self) -> None:
        """Must return at most top_n contributions."""
        contribs = [
            FeatureContribution(
                name=f"F{i}", value=float(i), impact=float(i), description="d"
            )
            for i in range(8)
        ]
        ranked = _rank_feature_contributions(contribs, top_n=3)
        assert len(ranked) == 3

    def test_sorted_by_abs_impact_descending(self) -> None:
        """Contributions must be sorted by abs(impact) descending."""
        contribs = [
            FeatureContribution(name="A", value=0.1, impact=0.1, description="d"),
            FeatureContribution(name="B", value=0.5, impact=-0.5, description="d"),
            FeatureContribution(name="C", value=0.3, impact=0.3, description="d"),
        ]
        ranked = _rank_feature_contributions(contribs, top_n=3)
        assert ranked[0].name == "B"
        assert ranked[1].name == "C"
        assert ranked[2].name == "A"

    def test_fewer_than_top_n_returns_all(self) -> None:
        """If fewer than top_n contributions, return all without error."""
        contribs = [
            FeatureContribution(name="A", value=0.5, impact=0.5, description="d"),
        ]
        ranked = _rank_feature_contributions(contribs, top_n=5)
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# TestBuildHoldHighlights
# ---------------------------------------------------------------------------


class TestBuildHoldHighlights:
    """Tests for _build_hold_highlights() private helper."""

    def test_all_zeros_returns_empty(self) -> None:
        """All-zero ratios must return an empty list."""
        vec = _make_vec()
        assert _build_hold_highlights(vec) == []

    def test_single_hold_type_formatted_correctly(self) -> None:
        """Single hold type must be formatted as 'type (pct%)'."""
        vec = _make_vec(crimp_ratio=0.4)
        highlights = _build_hold_highlights(vec)
        assert highlights == ["crimps (40%)"]

    def test_top_n_defaults_to_three(self) -> None:
        """At most 3 hold types must be returned by default."""
        vec = _make_vec(
            crimp_ratio=0.4, sloper_ratio=0.3, jug_ratio=0.2, pinch_ratio=0.1
        )
        highlights = _build_hold_highlights(vec)
        assert len(highlights) <= 3

    def test_highest_ratio_first(self) -> None:
        """Hold types must be ranked by ratio descending."""
        vec = _make_vec(crimp_ratio=0.4, sloper_ratio=0.6)
        highlights = _build_hold_highlights(vec)
        assert highlights[0].startswith("slopers")


# ---------------------------------------------------------------------------
# TestBuildSummary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    """Tests for _build_summary() private helper."""

    def test_grade_appears_in_summary(self) -> None:
        """The grade must appear in the generated summary."""
        fc = FeatureContribution(
            name="Crimp ratio", value=0.4, impact=0.14, description="d"
        )
        summary = _build_summary("V5", "confident", ["crimps (40%)"], [fc])
        assert "V5" in summary

    def test_qualifier_appears_in_summary(self) -> None:
        """The qualifier must appear in the generated summary."""
        fc = FeatureContribution(
            name="Crimp ratio", value=0.4, impact=0.14, description="d"
        )
        summary = _build_summary("V3", "very confident", [], [fc])
        assert "very confident" in summary

    def test_top_feature_name_appears_in_summary(self) -> None:
        """The top feature name must appear in the generated summary."""
        fc = FeatureContribution(
            name="Average move distance", value=0.5, impact=0.25, description="d"
        )
        summary = _build_summary("V3", "confident", [], [fc])
        assert "Average move distance" in summary

    def test_empty_features_uses_fallback(self) -> None:
        """Empty top_features must use fallback text in the summary."""
        summary = _build_summary("V0", "uncertain", [], [])
        assert "overall route features" in summary


# ---------------------------------------------------------------------------
# TestGetEstimatorType
# ---------------------------------------------------------------------------


class TestGetEstimatorType:
    """Tests for _get_estimator_type() private helper."""

    def test_ml_result_returns_ml(self) -> None:
        """MLGradeResult must return 'ml'."""
        result = _make_ml_result()
        assert _get_estimator_type(result) == "ml"

    def test_heuristic_result_returns_heuristic(self) -> None:
        """HeuristicGradeResult must return 'heuristic'."""
        result = _make_heuristic_result()
        assert _get_estimator_type(result) == "heuristic"


# ---------------------------------------------------------------------------
# TestGenerateHoldDescription
# ---------------------------------------------------------------------------


class TestGenerateHoldDescription:
    """Tests for _generate_hold_description() private helper."""

    def test_positive_impact_mentions_adding_difficulty(self) -> None:
        """Positive impact must mention 'adding' difficulty."""
        desc = _generate_hold_description("Crimp ratio", 0.4, 0.14)
        assert "adding" in desc

    def test_negative_impact_mentions_reducing_difficulty(self) -> None:
        """Negative impact must mention 'reducing' difficulty."""
        desc = _generate_hold_description("Jug ratio", 0.5, -0.15)
        assert "reducing" in desc

    def test_zero_impact_mentions_neutral(self) -> None:
        """Zero impact must mention 'neutral'."""
        desc = _generate_hold_description("Volume ratio", 0.0, 0.0)
        assert "neutral" in desc

    def test_percentage_formatted_correctly(self) -> None:
        """value=0.4 must produce '40%' in the description."""
        desc = _generate_hold_description("Crimp ratio", 0.4, 0.14)
        assert "40%" in desc

    def test_pinch_pluralized_correctly(self) -> None:
        """'Pinch ratio' must produce 'pinches', not 'pinchs'."""
        desc = _generate_hold_description("Pinch ratio", 0.3, 0.06)
        assert "pinches" in desc
        assert "pinchs" not in desc


# ---------------------------------------------------------------------------
# TestGenerateGeometryDescription
# ---------------------------------------------------------------------------


class TestGenerateGeometryDescription:
    """Tests for _generate_geometry_description() private helper."""

    def test_positive_impact_mentions_contributing(self) -> None:
        """Positive impact must mention 'contributing'."""
        desc = _generate_geometry_description("Average move distance", 0.5, 0.25)
        assert "contributing" in desc

    def test_negative_impact_mentions_reducing(self) -> None:
        """Negative impact must mention 'reducing'."""
        desc = _generate_geometry_description("Some feature", 0.5, -0.1)
        assert "reducing" in desc

    def test_value_formatted_in_description(self) -> None:
        """The raw value must appear formatted in the description."""
        desc = _generate_geometry_description("Path length (hops)", 5.0, 0.1)
        assert "5.00" in desc

    def test_zero_impact_mentions_neutral(self) -> None:
        """Zero impact must mention 'neutral'."""
        desc = _generate_geometry_description("Average move distance", 0.0, 0.0)
        assert "neutral" in desc


# ---------------------------------------------------------------------------
# TestGenerateExplanation (integration)
# ---------------------------------------------------------------------------


class TestGenerateExplanation:
    """Integration tests for generate_explanation()."""

    def test_returns_explanation_result(self) -> None:
        """generate_explanation must return an ExplanationResult."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        assert isinstance(result, ExplanationResult)

    def test_grade_matches_prediction(self) -> None:
        """result.grade must match prediction.grade."""
        rf = _make_route_features()
        pred = _make_heuristic_result(grade="V3", grade_index=3)
        result = generate_explanation(rf, pred)
        assert result.grade == "V3"

    def test_heuristic_estimator_type(self) -> None:
        """Heuristic prediction must yield estimator_type='heuristic'."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        assert result.estimator_type == "heuristic"

    def test_ml_estimator_type(self) -> None:
        """ML prediction must yield estimator_type='ml'."""
        rf = _make_route_features()
        pred = _make_ml_result()
        result = generate_explanation(rf, pred)
        assert result.estimator_type == "ml"

    def test_confidence_qualifier_very_confident(self) -> None:
        """High confidence must yield 'very confident'."""
        rf = _make_route_features()
        pred = _make_heuristic_result(confidence=0.9)
        result = generate_explanation(rf, pred)
        assert result.confidence_qualifier == "very confident"

    def test_confidence_qualifier_uncertain_for_low_confidence(self) -> None:
        """Low confidence must yield 'uncertain'."""
        rf = _make_route_features()
        pred = _make_heuristic_result(confidence=0.55)
        result = generate_explanation(rf, pred)
        assert result.confidence_qualifier == "uncertain"

    def test_top_features_at_most_five(self) -> None:
        """top_features must contain at most 5 contributions."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        assert len(result.top_features) <= 5

    def test_top_features_sorted_by_abs_impact(self) -> None:
        """top_features must be sorted by abs(impact) descending."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        impacts = [abs(f.impact) for f in result.top_features]
        assert impacts == sorted(impacts, reverse=True)

    def test_summary_contains_grade(self) -> None:
        """summary must contain the predicted grade."""
        rf = _make_route_features()
        pred = _make_heuristic_result(grade="V3", grade_index=3)
        result = generate_explanation(rf, pred)
        assert "V3" in result.summary

    def test_hold_highlights_excluded_for_zero_ratios(self) -> None:
        """Hold types with zero ratio must not appear in hold_highlights."""
        holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.35, y_center=0.5, hold_type="crimp"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.65, y_center=0.5, hold_type="crimp"
            ),
        ]
        rf = _make_route_features(holds=holds)
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        # Zero-ratio types (jugs, slopers, pinches, volumes) must not appear
        zero_types = ["jugs", "slopers", "pinches", "volumes"]
        for highlight in result.hold_highlights:
            for zero_type in zero_types:
                assert zero_type not in highlight

    def test_deterministic_same_input(self) -> None:
        """Same inputs must produce identical ExplanationResult."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        r1 = generate_explanation(rf, pred)
        r2 = generate_explanation(rf, pred)
        assert r1.summary == r2.summary
        assert r1.grade == r2.grade

    def test_feature_extraction_error_wrapped_as_explanation_error(self) -> None:
        """FeatureExtractionError from to_vector() must be wrapped as ExplanationError."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        original = FeatureExtractionError("simulated taxonomy drift")
        with patch.object(type(rf), "to_vector", side_effect=original):
            with pytest.raises(ExplanationError) as exc_info:
                generate_explanation(rf, pred)
        assert exc_info.value.__cause__ is original

    def test_wrapped_error_message_contains_context(self) -> None:
        """Wrapped ExplanationError.message must include the original message."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        original = FeatureExtractionError("simulated taxonomy drift")
        with patch.object(type(rf), "to_vector", side_effect=original):
            with pytest.raises(ExplanationError) as exc_info:
                generate_explanation(rf, pred)
        assert "simulated taxonomy drift" in exc_info.value.message

    def test_crimp_heavy_route_has_crimp_in_highlights(self) -> None:
        """Route with all crimps must list crimps in hold_highlights."""
        holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.35, y_center=0.5, hold_type="crimp"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.65, y_center=0.5, hold_type="crimp"
            ),
        ]
        rf = _make_route_features(holds=holds)
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        assert any("crimp" in h for h in result.hold_highlights)

    def test_result_is_frozen(self) -> None:
        """ExplanationResult must be immutable (frozen=True)."""
        rf = _make_route_features()
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        with pytest.raises(ValidationError):
            result.grade = "V0"  # type: ignore[misc]

    def test_multi_hold_type_route_succeeds(self) -> None:
        """Route with varied hold types must produce a valid explanation."""
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
                hold_id=4, x_center=0.9, y_center=0.9, hold_type="volume"
            ),
        ]
        rf = _make_route_features(holds=holds, start_ids=[0], finish_id=4)
        pred = _make_heuristic_result()
        result = generate_explanation(rf, pred)
        assert isinstance(result, ExplanationResult)

    def test_ml_prediction_confidence_qualifier(self) -> None:
        """ML prediction at 0.73 confidence must yield 'confident'."""
        rf = _make_route_features()
        pred = _make_ml_result(confidence=0.73)
        result = generate_explanation(rf, pred)
        assert result.confidence_qualifier == "confident"
