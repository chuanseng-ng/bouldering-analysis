"""Tests for src.grading._utils module.

Covers _clamp() and _normalize_vector().
"""

import pytest

from src.grading._utils import _clamp, _normalize_vector


class TestClamp:
    """Tests for _clamp()."""

    def test_value_below_lo_returns_lo(self) -> None:
        """Value below lo must be clamped to lo."""
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_value_above_hi_returns_hi(self) -> None:
        """Value above hi must be clamped to hi."""
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_value_within_range_unchanged(self) -> None:
        """Value within [lo, hi] must be returned unchanged."""
        assert _clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_value_equal_lo_returns_lo(self) -> None:
        """Value exactly equal to lo must be returned as-is."""
        assert _clamp(0.0, 0.0, 1.0) == 0.0

    def test_value_equal_hi_returns_hi(self) -> None:
        """Value exactly equal to hi must be returned as-is."""
        assert _clamp(1.0, 0.0, 1.0) == 1.0

    def test_negative_range(self) -> None:
        """Value above negative range must be clamped to hi."""
        assert _clamp(0.0, -2.0, -1.0) == -1.0

    def test_asymmetric_range(self) -> None:
        """Value within an asymmetric range must be returned unchanged."""
        assert _clamp(5.0, 3.0, 7.0) == pytest.approx(5.0)


class TestNormalizeVector:
    """Tests for _normalize_vector()."""

    def test_basic_zscore(self) -> None:
        """Standard z-score: (3 - 1) / 2 == 1.0."""
        result = _normalize_vector({"x": 3.0}, {"x": 1.0}, {"x": 2.0})
        assert result["x"] == pytest.approx(1.0)

    def test_zero_mean(self) -> None:
        """Zero mean: (2 - 0) / 2 == 1.0."""
        result = _normalize_vector({"x": 2.0}, {"x": 0.0}, {"x": 2.0})
        assert result["x"] == pytest.approx(1.0)

    def test_zero_variance_fallback(self) -> None:
        """std=0.0 must fall back to std=1.0, giving (x - mean) / 1.0."""
        result = _normalize_vector({"x": 3.0}, {"x": 1.0}, {"x": 0.0})
        assert result["x"] == pytest.approx(2.0)

    def test_zero_variance_no_division_error(self) -> None:
        """Zero-variance features must not raise ZeroDivisionError."""
        result = _normalize_vector(
            {"a": 5.0, "b": 5.0}, {"a": 5.0, "b": 3.0}, {"a": 0.0, "b": 0.0}
        )
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(2.0)

    def test_multiple_features(self) -> None:
        """All features in the vector must be normalized independently."""
        result = _normalize_vector(
            {"x": 2.0, "y": 4.0},
            {"x": 0.0, "y": 2.0},
            {"x": 2.0, "y": 2.0},
        )
        assert result["x"] == pytest.approx(1.0)
        assert result["y"] == pytest.approx(1.0)

    def test_keys_preserved(self) -> None:
        """Output dict must contain exactly the same keys as the input vector."""
        result = _normalize_vector(
            {"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}
        )
        assert set(result.keys()) == {"a", "b"}

    def test_negative_zscore(self) -> None:
        """Value below mean must produce a negative normalized result."""
        result = _normalize_vector({"x": -1.0}, {"x": 1.0}, {"x": 2.0})
        assert result["x"] == pytest.approx(-1.0)

    def test_missing_mean_key_raises(self) -> None:
        """Missing key in mean must raise KeyError."""
        with pytest.raises(KeyError):
            _normalize_vector({"x": 1.0}, {}, {"x": 1.0})

    def test_missing_std_key_raises(self) -> None:
        """Missing key in std must raise KeyError."""
        with pytest.raises(KeyError):
            _normalize_vector({"x": 1.0}, {"x": 0.0}, {})

    def test_extra_key_in_mean_raises(self) -> None:
        """Extra key in mean (not in vector) must raise KeyError."""
        with pytest.raises(KeyError):
            _normalize_vector({"x": 1.0}, {"x": 0.0, "y": 0.0}, {"x": 1.0})

    def test_extra_key_in_std_raises(self) -> None:
        """Extra key in std (not in vector) must raise KeyError."""
        with pytest.raises(KeyError):
            _normalize_vector({"x": 1.0}, {"x": 0.0}, {"x": 1.0, "y": 1.0})
