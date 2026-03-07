"""Tests for src.features.holds module.

Covers:
- src/features/holds.py  — HoldFeatures model, extract_hold_features(),
                           and all private helpers: _count_by_type,
                           _compute_size_stats, _compute_soft_distribution
"""

import pydantic
import pytest

from src.constants import MAX_HOLD_COUNT
from src.features.exceptions import FeatureExtractionError
from src.features.holds import (
    HoldFeatures,
    _compute_size_stats,
    _compute_soft_distribution,
    _count_by_type,
    extract_hold_features,
)
from src.graph.types import ClassifiedHold
from src.training.classification_dataset import HOLD_CLASSES
from tests.conftest import make_classified_hold_for_tests as _make_classified_hold


# ---------------------------------------------------------------------------
# TestHoldFeatures — Pydantic model structure
# ---------------------------------------------------------------------------


class TestHoldFeatures:
    """Tests for HoldFeatures Pydantic model structure and constraints."""

    def _make_valid_features(self) -> HoldFeatures:
        """Return a fully-populated valid HoldFeatures instance."""
        return HoldFeatures(
            total_count=3,
            jug_count=2,
            crimp_count=1,
            sloper_count=0,
            pinch_count=0,
            volume_count=0,
            unknown_count=0,
            jug_ratio=2 / 3,
            crimp_ratio=1 / 3,
            sloper_ratio=0.0,
            pinch_ratio=0.0,
            volume_ratio=0.0,
            unknown_ratio=0.0,
            avg_hold_size=0.01,
            max_hold_size=0.02,
            min_hold_size=0.005,
            std_hold_size=0.005,
            jug_soft_ratio=0.6,
            crimp_soft_ratio=0.1,
            sloper_soft_ratio=0.1,
            pinch_soft_ratio=0.1,
            volume_soft_ratio=0.05,
            unknown_soft_ratio=0.05,
        )

    def test_all_23_fields_exist(self) -> None:
        """HoldFeatures must expose exactly 23 fields."""
        expected_fields = {
            "total_count",
            "jug_count",
            "crimp_count",
            "sloper_count",
            "pinch_count",
            "volume_count",
            "unknown_count",
            "jug_ratio",
            "crimp_ratio",
            "sloper_ratio",
            "pinch_ratio",
            "volume_ratio",
            "unknown_ratio",
            "avg_hold_size",
            "max_hold_size",
            "min_hold_size",
            "std_hold_size",
            "jug_soft_ratio",
            "crimp_soft_ratio",
            "sloper_soft_ratio",
            "pinch_soft_ratio",
            "volume_soft_ratio",
            "unknown_soft_ratio",
        }
        assert set(HoldFeatures.model_fields.keys()) == expected_fields
        assert len(HoldFeatures.model_fields) == 23

    def test_count_fields_are_int(self) -> None:
        """All count fields must be int instances."""
        features = self._make_valid_features()
        int_fields = [
            "total_count",
            "jug_count",
            "crimp_count",
            "sloper_count",
            "pinch_count",
            "volume_count",
            "unknown_count",
        ]
        for field in int_fields:
            assert isinstance(getattr(features, field), int), f"{field} not int"

    def test_ratio_fields_bounded_le_one(self) -> None:
        """Ratio and soft-ratio float fields must be le=1.0 (Pydantic constraint)."""
        ratio_fields = [
            "jug_ratio",
            "crimp_ratio",
            "sloper_ratio",
            "pinch_ratio",
            "volume_ratio",
            "unknown_ratio",
            "jug_soft_ratio",
            "crimp_soft_ratio",
            "sloper_soft_ratio",
            "pinch_soft_ratio",
            "volume_soft_ratio",
            "unknown_soft_ratio",
        ]
        for field in ratio_fields:
            with pytest.raises((pydantic.ValidationError, ValueError)):
                HoldFeatures(**{**self._make_valid_features().model_dump(), field: 1.1})

    def test_zero_values_accepted(self) -> None:
        """All fields set to zero must be accepted (no constraint violations)."""
        features = HoldFeatures(
            total_count=0,
            jug_count=0,
            crimp_count=0,
            sloper_count=0,
            pinch_count=0,
            volume_count=0,
            unknown_count=0,
            jug_ratio=0.0,
            crimp_ratio=0.0,
            sloper_ratio=0.0,
            pinch_ratio=0.0,
            volume_ratio=0.0,
            unknown_ratio=0.0,
            avg_hold_size=0.0,
            max_hold_size=0.0,
            min_hold_size=0.0,
            std_hold_size=0.0,
            jug_soft_ratio=0.0,
            crimp_soft_ratio=0.0,
            sloper_soft_ratio=0.0,
            pinch_soft_ratio=0.0,
            volume_soft_ratio=0.0,
            unknown_soft_ratio=0.0,
        )
        assert features.total_count == 0

    def test_negative_ratio_rejected(self) -> None:
        """Negative ratio float fields must be rejected by Pydantic (ge=0.0)."""
        with pytest.raises((pydantic.ValidationError, ValueError)):
            HoldFeatures(
                **{**self._make_valid_features().model_dump(), "jug_ratio": -0.1}
            )

    def test_negative_count_rejected(self) -> None:
        """Negative count fields must be rejected by Pydantic."""
        with pytest.raises((pydantic.ValidationError, ValueError)):
            HoldFeatures(
                **{**self._make_valid_features().model_dump(), "total_count": -1}
            )


# ---------------------------------------------------------------------------
# TestCountByType — private helper
# ---------------------------------------------------------------------------


class TestCountByType:
    """Tests for _count_by_type private helper."""

    def test_empty_list_returns_all_zeros(self) -> None:
        """Empty hold list must return a dict with zero count for all classes."""
        result = _count_by_type([])
        assert set(result.keys()) == set(HOLD_CLASSES)
        assert all(v == 0 for v in result.values())

    def test_single_hold_increments_its_type(self) -> None:
        """Single jug hold must set jug_count=1, all others 0."""
        holds = [_make_classified_hold(hold_id=0, hold_type="jug")]
        result = _count_by_type(holds)
        assert result["jug"] == 1
        assert result["crimp"] == 0
        assert result["sloper"] == 0
        assert result["pinch"] == 0
        assert result["volume"] == 0
        assert result["unknown"] == 0

    def test_all_same_type_counted(self) -> None:
        """Three crimp holds must set crimp_count=3, all others 0."""
        holds = [_make_classified_hold(hold_id=i, hold_type="crimp") for i in range(3)]
        result = _count_by_type(holds)
        assert result["crimp"] == 3
        assert result["jug"] == 0

    def test_mixed_types_counted(self) -> None:
        """Mixed hold list must count each type independently."""
        holds = [
            _make_classified_hold(hold_id=0, hold_type="jug"),
            _make_classified_hold(hold_id=1, hold_type="jug"),
            _make_classified_hold(hold_id=2, hold_type="crimp"),
            _make_classified_hold(hold_id=3, hold_type="sloper"),
        ]
        result = _count_by_type(holds)
        assert result["jug"] == 2
        assert result["crimp"] == 1
        assert result["sloper"] == 1
        assert result["pinch"] == 0
        assert result["volume"] == 0
        assert result["unknown"] == 0

    def test_unknown_type_counted(self) -> None:
        """Unknown hold type must be counted in the 'unknown' key."""
        holds = [_make_classified_hold(hold_id=0, hold_type="unknown")]
        result = _count_by_type(holds)
        assert result["unknown"] == 1
        assert result["jug"] == 0

    def test_all_six_types_present_in_result(self) -> None:
        """Result dict must always contain exactly the 6 HOLD_CLASSES keys."""
        holds = [_make_classified_hold(hold_id=0, hold_type="volume")]
        result = _count_by_type(holds)
        assert set(result.keys()) == set(HOLD_CLASSES)
        assert len(result) == 6

    def test_unexpected_hold_type_raises_via_model_construct(self) -> None:
        """Defensive guard must raise FeatureExtractionError for unknown hold_type.

        Uses model_construct() to bypass ClassifiedHold.validate_hold_type,
        simulating a hold that somehow bypasses normal construction.
        """
        bad_hold = ClassifiedHold.model_construct(hold_type="pocket")
        with pytest.raises(FeatureExtractionError, match="Unexpected hold_type"):
            _count_by_type([bad_hold])


# ---------------------------------------------------------------------------
# TestComputeSizeStats — private helper
# ---------------------------------------------------------------------------


class TestComputeSizeStats:
    """Tests for _compute_size_stats private helper."""

    def test_single_hold_std_is_zero(self) -> None:
        """Single hold must return std=0.0 and avg=max=min=area."""
        holds = [_make_classified_hold(hold_id=0, width=0.2, height=0.3)]
        avg, mx, mn, std = _compute_size_stats(holds)
        expected_area = 0.2 * 0.3
        assert avg == pytest.approx(expected_area)
        assert mx == pytest.approx(expected_area)
        assert mn == pytest.approx(expected_area)
        assert std == pytest.approx(0.0)

    def test_two_holds_avg_correct(self) -> None:
        """Two holds with different sizes must compute correct average area."""
        holds = [
            _make_classified_hold(hold_id=0, width=0.1, height=0.1),  # area=0.01
            _make_classified_hold(hold_id=1, width=0.3, height=0.3),  # area=0.09
        ]
        avg, mx, mn, std = _compute_size_stats(holds)
        assert avg == pytest.approx(0.05)
        assert mx == pytest.approx(0.09)
        assert mn == pytest.approx(0.01)

    def test_all_same_size_std_is_zero(self) -> None:
        """Holds with identical areas must produce zero standard deviation."""
        holds = [
            _make_classified_hold(hold_id=i, width=0.1, height=0.2) for i in range(4)
        ]
        avg, mx, mn, std = _compute_size_stats(holds)
        assert std == pytest.approx(0.0)
        assert avg == pytest.approx(0.02)
        assert mx == pytest.approx(0.02)
        assert mn == pytest.approx(0.02)

    def test_varying_sizes_std_correct(self) -> None:
        """Holds with different areas must produce the correct population std.

        areas = [0.01, 0.04, 0.09], mean = 14/300
        variance = sum((a - mean)^2) / 3
        """
        import math as _math

        holds = [
            _make_classified_hold(hold_id=0, width=0.1, height=0.1),  # area=0.01
            _make_classified_hold(hold_id=1, width=0.2, height=0.2),  # area=0.04
            _make_classified_hold(hold_id=2, width=0.3, height=0.3),  # area=0.09
        ]
        avg, mx, mn, std = _compute_size_stats(holds)
        areas = [0.01, 0.04, 0.09]
        mean = sum(areas) / 3
        expected_std = _math.sqrt(sum((a - mean) ** 2 for a in areas) / 3)
        assert std > 0.0
        assert std == pytest.approx(expected_std, rel=1e-6)

    def test_max_min_correct_for_known_values(self) -> None:
        """Max and min must match the extremes of the area distribution."""
        holds = [
            _make_classified_hold(hold_id=0, width=0.1, height=0.1),  # area=0.01
            _make_classified_hold(hold_id=1, width=0.5, height=0.5),  # area=0.25
            _make_classified_hold(hold_id=2, width=0.2, height=0.2),  # area=0.04
        ]
        avg, mx, mn, std = _compute_size_stats(holds)
        assert mn == pytest.approx(0.01)
        assert mx == pytest.approx(0.25)

    def test_empty_returns_zeros(self) -> None:
        """Empty hold list must return (0.0, 0.0, 0.0, 0.0)."""
        result = _compute_size_stats([])
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_zero_area_hold_is_handled(self) -> None:
        """Hold with width=0 produces area=0; must not crash and min must be 0."""
        holds = [
            _make_classified_hold(hold_id=0, width=0.0, height=0.2),  # area=0.0
            _make_classified_hold(hold_id=1, width=0.2, height=0.2),  # area=0.04
        ]
        avg, mx, mn, std = _compute_size_stats(holds)
        assert mn == pytest.approx(0.0)
        assert mx == pytest.approx(0.04)
        assert avg == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# TestComputeSoftDistribution — private helper
# ---------------------------------------------------------------------------


class TestComputeSoftDistribution:
    """Tests for _compute_soft_distribution private helper."""

    def test_single_hold_returns_its_probabilities(self) -> None:
        """Single hold must return its own type_probabilities unchanged."""
        hold = _make_classified_hold(hold_id=0, hold_type="jug", type_confidence=0.8)
        result = _compute_soft_distribution([hold])
        for cls in HOLD_CLASSES:
            assert result[cls] == pytest.approx(hold.type_probabilities[cls])

    def test_result_keys_equal_hold_classes(self) -> None:
        """Result dict keys must equal exactly the HOLD_CLASSES tuple members."""
        holds = [_make_classified_hold(hold_id=0)]
        result = _compute_soft_distribution(holds)
        assert set(result.keys()) == set(HOLD_CLASSES)

    def test_values_sum_to_one(self) -> None:
        """Soft distribution values must sum to approximately 1.0."""
        holds = [
            _make_classified_hold(hold_id=0, hold_type="jug"),
            _make_classified_hold(hold_id=1, hold_type="crimp"),
            _make_classified_hold(hold_id=2, hold_type="sloper"),
        ]
        result = _compute_soft_distribution(holds)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_mixed_holds_averages_probabilities(self) -> None:
        """Mixed holds must return the mean probability per class."""
        # Two holds: both have type_confidence=1.0 for their type
        # → probabilities are one-hot effectively, so soft = average
        hold_a = _make_classified_hold(hold_id=0, hold_type="jug", type_confidence=1.0)
        hold_b = _make_classified_hold(
            hold_id=1, hold_type="crimp", type_confidence=1.0
        )
        result = _compute_soft_distribution([hold_a, hold_b])
        expected_jug = (
            hold_a.type_probabilities["jug"] + hold_b.type_probabilities["jug"]
        ) / 2
        expected_crimp = (
            hold_a.type_probabilities["crimp"] + hold_b.type_probabilities["crimp"]
        ) / 2
        assert result["jug"] == pytest.approx(expected_jug)
        assert result["crimp"] == pytest.approx(expected_crimp)

    def test_uniform_probabilities_returned_correctly(self) -> None:
        """Hold with uniform probabilities must produce a uniform distribution."""
        # Build a hold with exactly uniform probabilities
        uniform_prob = 1.0 / len(HOLD_CLASSES)
        probs = {cls: uniform_prob for cls in HOLD_CLASSES}
        hold = _make_classified_hold(hold_id=0)
        # Override via model_copy
        hold_uniform = hold.model_copy(update={"type_probabilities": probs})
        result = _compute_soft_distribution([hold_uniform])
        for cls in HOLD_CLASSES:
            assert result[cls] == pytest.approx(uniform_prob)


# ---------------------------------------------------------------------------
# TestExtractHoldFeaturesValidation — error paths
# ---------------------------------------------------------------------------


class TestExtractHoldFeaturesValidation:
    """Tests for extract_hold_features() error handling."""

    def test_empty_list_raises_feature_extraction_error(self) -> None:
        """Empty holds list must raise FeatureExtractionError."""
        with pytest.raises(FeatureExtractionError):
            extract_hold_features([])

    def test_error_is_value_error_subclass(self) -> None:
        """FeatureExtractionError raised on empty input must be a ValueError."""
        with pytest.raises(ValueError):
            extract_hold_features([])

    def test_error_message_is_descriptive(self) -> None:
        """Error message must mention 'empty' or 'holds'."""
        with pytest.raises(FeatureExtractionError) as exc_info:
            extract_hold_features([])
        msg = exc_info.value.message.lower()
        assert "empty" in msg or "holds" in msg

    def test_exceeding_max_hold_count_raises(self) -> None:
        """List exceeding MAX_HOLD_COUNT must raise FeatureExtractionError."""
        holds = [_make_classified_hold(hold_id=i) for i in range(MAX_HOLD_COUNT + 1)]
        with pytest.raises(FeatureExtractionError) as exc_info:
            extract_hold_features(holds)
        assert str(MAX_HOLD_COUNT) in exc_info.value.message

    def test_exactly_max_hold_count_is_accepted(self) -> None:
        """List of exactly MAX_HOLD_COUNT holds must not raise."""
        holds = [_make_classified_hold(hold_id=i) for i in range(MAX_HOLD_COUNT)]
        result = extract_hold_features(holds)
        assert result.total_count == MAX_HOLD_COUNT


# ---------------------------------------------------------------------------
# TestExtractHoldFeatures — end-to-end
# ---------------------------------------------------------------------------


class TestExtractHoldFeatures:
    """End-to-end tests for extract_hold_features()."""

    def test_single_jug_returns_hold_features_instance(self) -> None:
        """Single jug hold must return a HoldFeatures instance."""
        holds = [_make_classified_hold(hold_id=0, hold_type="jug")]
        result = extract_hold_features(holds)
        assert isinstance(result, HoldFeatures)

    def test_all_jugs_count_equals_total_count(self) -> None:
        """Three jug holds must have jug_count == total_count == 3."""
        holds = [_make_classified_hold(hold_id=i, hold_type="jug") for i in range(3)]
        result = extract_hold_features(holds)
        assert result.total_count == 3
        assert result.jug_count == 3
        assert result.crimp_count == 0
        assert result.jug_ratio == pytest.approx(1.0)

    def test_mixed_types_ratios_sum_to_one(self) -> None:
        """Hard ratios must sum to exactly 1.0 for any valid hold list."""
        holds = [
            _make_classified_hold(hold_id=0, hold_type="jug"),
            _make_classified_hold(hold_id=1, hold_type="crimp"),
            _make_classified_hold(hold_id=2, hold_type="sloper"),
            _make_classified_hold(hold_id=3, hold_type="pinch"),
        ]
        result = extract_hold_features(holds)
        ratio_sum = (
            result.jug_ratio
            + result.crimp_ratio
            + result.sloper_ratio
            + result.pinch_ratio
            + result.volume_ratio
            + result.unknown_ratio
        )
        assert ratio_sum == pytest.approx(1.0, abs=1e-9)

    def test_soft_ratios_sum_to_one(self) -> None:
        """Soft ratios must sum to approximately 1.0."""
        holds = [
            _make_classified_hold(hold_id=0, hold_type="jug"),
            _make_classified_hold(hold_id=1, hold_type="crimp"),
        ]
        result = extract_hold_features(holds)
        soft_sum = (
            result.jug_soft_ratio
            + result.crimp_soft_ratio
            + result.sloper_soft_ratio
            + result.pinch_soft_ratio
            + result.volume_soft_ratio
            + result.unknown_soft_ratio
        )
        assert soft_sum == pytest.approx(1.0, abs=1e-6)

    def test_size_stats_single_hold(self) -> None:
        """Single hold must produce avg==max==min==area and std==0."""
        holds = [_make_classified_hold(hold_id=0, width=0.1, height=0.2)]
        result = extract_hold_features(holds)
        area = 0.1 * 0.2
        assert result.avg_hold_size == pytest.approx(area)
        assert result.max_hold_size == pytest.approx(area)
        assert result.min_hold_size == pytest.approx(area)
        assert result.std_hold_size == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_total_count_matches_input_length(self, n: int) -> None:
        """total_count must equal len(holds) for any non-empty input."""
        holds = [_make_classified_hold(hold_id=i) for i in range(n)]
        result = extract_hold_features(holds)
        assert result.total_count == n

    def test_unknown_holds_counted_correctly(self) -> None:
        """Unknown-type holds must be tallied in unknown_count / unknown_ratio."""
        holds = [
            _make_classified_hold(hold_id=0, hold_type="unknown"),
            _make_classified_hold(hold_id=1, hold_type="unknown"),
            _make_classified_hold(hold_id=2, hold_type="jug"),
        ]
        result = extract_hold_features(holds)
        assert result.unknown_count == 2
        assert result.unknown_ratio == pytest.approx(2 / 3)
        assert result.jug_count == 1

    def test_single_type_full_confidence_soft_ratios(self) -> None:
        """All holds same type with type_confidence=1.0 must give one-hot soft distribution.

        With type_confidence=1.0, make_classified_hold_for_tests sets the dominant
        class probability to 1.0 and all others to 0.0 exactly.
        """
        holds = [
            _make_classified_hold(hold_id=i, hold_type="jug", type_confidence=1.0)
            for i in range(3)
        ]
        result = extract_hold_features(holds)
        assert result.jug_soft_ratio == pytest.approx(1.0)
        assert result.crimp_soft_ratio == pytest.approx(0.0)
        assert result.sloper_soft_ratio == pytest.approx(0.0)
        assert result.pinch_soft_ratio == pytest.approx(0.0)
        assert result.volume_soft_ratio == pytest.approx(0.0)
        assert result.unknown_soft_ratio == pytest.approx(0.0)

    def test_confidence_weighted_distribution_is_averaged(self) -> None:
        """Soft ratios must reflect the mean probability across all holds."""
        hold_a = _make_classified_hold(hold_id=0, hold_type="jug", type_confidence=0.9)
        hold_b = _make_classified_hold(
            hold_id=1, hold_type="crimp", type_confidence=0.7
        )
        result = extract_hold_features([hold_a, hold_b])
        expected_jug_soft = (
            hold_a.type_probabilities["jug"] + hold_b.type_probabilities["jug"]
        ) / 2
        expected_crimp_soft = (
            hold_a.type_probabilities["crimp"] + hold_b.type_probabilities["crimp"]
        ) / 2
        assert result.jug_soft_ratio == pytest.approx(expected_jug_soft)
        assert result.crimp_soft_ratio == pytest.approx(expected_crimp_soft)

    def test_output_is_round_trippable(self) -> None:
        """HoldFeatures must survive model_dump() → HoldFeatures(**...) round-trip.

        Acts as a canary: if any computed field violates its ge/le constraints,
        the round-trip construction raises ValidationError.
        """
        holds = [
            _make_classified_hold(hold_id=0, hold_type="jug"),
            _make_classified_hold(hold_id=1, hold_type="crimp"),
            _make_classified_hold(hold_id=2, hold_type="jug"),
        ]
        result = extract_hold_features(holds)
        reconstructed = HoldFeatures(**result.model_dump())
        assert reconstructed == result
