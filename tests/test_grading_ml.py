"""Tests for src.grading.ml_estimator module.

Covers:
- MLGradeResult model validation
- _compute_confidence() — normalised entropy
- _compute_difficulty_score() — probability-weighted mean
- estimate_grade_ml() — full pipeline, caching, error handling
- _clear_model_cache()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError
from xgboost import XGBClassifier

from src.features.assembler import assemble_features
from src.features.exceptions import FeatureExtractionError
from src.graph.constraints import apply_route_constraints
from src.graph.route_graph import build_route_graph
from src.grading.constants import V_GRADES
from src.grading.exceptions import GradeEstimationError
from src.grading.ml_estimator import (
    MLGradeResult,
    _MODEL_CACHE,
    _clear_model_cache,
    _compute_confidence,
    _compute_difficulty_score,
    _predict_grade,
    estimate_grade_ml,
)
from src.training.train_grade_estimator import (
    generate_synthetic_training_data,
    train_grade_estimator,
)
from tests.conftest import make_classified_hold_for_tests as _make_hold

_N_GRADES = 18


# ---------------------------------------------------------------------------
# Session-scoped fixture: train a small real model once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a small XGBoost model for inference tests.

    Returns:
        Path to the versioned model directory.
    """
    samples = generate_synthetic_training_data(n_samples=200, seed=0)
    features, labels = zip(*samples)
    output_dir = tmp_path_factory.mktemp("models") / "grading"
    result = train_grade_estimator(list(features), list(labels), output_dir)
    return result.model_path.parent


@pytest.fixture(autouse=True)
def _clear_cache_after_each() -> Any:
    """Clear model cache before and after each test."""
    _clear_model_cache()
    yield
    _clear_model_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_probs() -> list[float]:
    """Return a uniform probability distribution over all 18 grades."""
    return [1.0 / _N_GRADES] * _N_GRADES


def _certain_probs(idx: int) -> list[float]:
    """Return a distribution with all probability mass at grade index *idx*."""
    probs = [0.0] * _N_GRADES
    probs[idx] = 1.0
    return probs


def _make_route_features() -> Any:
    """Build a minimal valid RouteFeatures for inference tests."""
    holds = [
        _make_hold(0, x_center=0.25, y_center=0.5, hold_type="jug"),
        _make_hold(1, x_center=0.50, y_center=0.5, hold_type="crimp"),
        _make_hold(2, x_center=0.75, y_center=0.5, hold_type="sloper"),
    ]
    rg = build_route_graph(holds, wall_angle=0.0)
    constrained = apply_route_constraints(rg, [0], 2)
    return assemble_features(constrained)


# ---------------------------------------------------------------------------
# TestMLGradeResult
# ---------------------------------------------------------------------------


class TestMLGradeResult:
    """Tests for MLGradeResult Pydantic model."""

    def _valid_kwargs(self) -> dict[str, Any]:
        """Return a dict of valid MLGradeResult constructor arguments."""
        return {
            "grade": "V0",
            "grade_index": 0,
            "confidence": 0.8,
            "difficulty_score": 0.1,
            "grade_probabilities": {g: (1.0 if g == "V0" else 0.0) for g in V_GRADES},
        }

    def test_valid_construction(self) -> None:
        """Valid kwargs must produce a correctly populated MLGradeResult."""
        r = MLGradeResult(**self._valid_kwargs())
        assert r.grade == "V0"
        assert r.grade_index == 0

    def test_frozen_model(self) -> None:
        """Assigning to a frozen model field must raise an error."""
        r = MLGradeResult(**self._valid_kwargs())
        with pytest.raises((ValidationError, TypeError)):
            r.grade = "V1"  # type: ignore[misc]

    def test_grade_index_below_zero_rejected(self) -> None:
        """grade_index < 0 must be rejected by the validator."""
        kw = self._valid_kwargs()
        kw["grade_index"] = -1
        with pytest.raises(ValidationError):
            MLGradeResult(**kw)

    def test_grade_index_above_17_rejected(self) -> None:
        """grade_index > 17 must be rejected by the validator."""
        kw = self._valid_kwargs()
        kw["grade_index"] = 18
        with pytest.raises(ValidationError):
            MLGradeResult(**kw)

    def test_confidence_below_zero_rejected(self) -> None:
        """confidence < 0 must be rejected by the validator."""
        kw = self._valid_kwargs()
        kw["confidence"] = -0.1
        with pytest.raises(ValidationError):
            MLGradeResult(**kw)

    def test_confidence_above_one_rejected(self) -> None:
        """confidence > 1 must be rejected by the validator."""
        kw = self._valid_kwargs()
        kw["confidence"] = 1.1
        with pytest.raises(ValidationError):
            MLGradeResult(**kw)

    def test_difficulty_score_in_range(self) -> None:
        """difficulty_score must lie within [0.0, 1.0]."""
        r = MLGradeResult(**self._valid_kwargs())
        assert 0.0 <= r.difficulty_score <= 1.0

    def test_grade_index_grade_consistency(self) -> None:
        """Mismatching grade/grade_index must raise ValidationError."""
        kw = self._valid_kwargs()
        kw["grade"] = "V3"  # mismatches grade_index=0
        with pytest.raises(ValidationError):
            MLGradeResult(**kw)

    def test_grade_probabilities_wrong_keys_rejected(self) -> None:
        """grade_probabilities with missing grade keys must raise ValidationError."""
        kw = self._valid_kwargs()
        kw["grade_probabilities"] = {"V0": 1.0}  # missing other grades
        with pytest.raises(ValidationError):
            MLGradeResult(**kw)

    def test_all_v_grades_valid(self) -> None:
        """All 18 V-grades must construct a valid MLGradeResult."""
        for idx, grade in enumerate(V_GRADES):
            probs = {g: (1.0 if g == grade else 0.0) for g in V_GRADES}
            r = MLGradeResult(
                grade=grade,
                grade_index=idx,
                confidence=1.0,
                difficulty_score=idx / 17.0,
                grade_probabilities=probs,
            )
            assert r.grade == grade
            assert r.grade_index == idx


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Tests for _compute_confidence()."""

    def test_certain_distribution_returns_one(self) -> None:
        """A certain distribution (all mass at one grade) must return 1.0."""
        assert _compute_confidence(_certain_probs(0)) == pytest.approx(1.0)

    def test_uniform_distribution_returns_zero(self) -> None:
        """A uniform distribution (maximum entropy) must return ≈0.0."""
        result = _compute_confidence(_uniform_probs())
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_concentrated_distribution_high_confidence(self) -> None:
        """A concentrated distribution must return confidence > 0.5."""
        probs = [0.01] * _N_GRADES
        probs[5] = 1.0 - 0.01 * 17
        result = _compute_confidence(probs)
        assert result > 0.5

    def test_result_in_zero_one(self) -> None:
        """Confidence must lie within [0.0, 1.0] for any certain distribution."""
        for idx in range(_N_GRADES):
            assert 0.0 <= _compute_confidence(_certain_probs(idx)) <= 1.0

    def test_zeros_in_probs_no_exception(self) -> None:
        """A distribution with many zeros must not raise (log(0) guard must hold)."""
        probs = _certain_probs(3)
        result = _compute_confidence(probs)
        assert result == pytest.approx(1.0)

    def test_mixed_zeros_no_exception(self) -> None:
        """Partial distribution with zeros must produce a valid confidence value."""
        probs = [0.0] * _N_GRADES
        probs[0] = 0.6
        probs[1] = 0.4
        result = _compute_confidence(probs)
        assert 0.0 <= result <= 1.0

    def test_more_concentrated_higher_confidence(self) -> None:
        """A narrower peak must yield higher confidence than a wider peak."""
        wide = [0.05] * _N_GRADES
        wide[0] = 1 - 0.05 * 17
        narrow = [0.0] * _N_GRADES
        narrow[0] = 0.95
        narrow[1] = 0.05
        assert _compute_confidence(narrow) > _compute_confidence(wide)


# ---------------------------------------------------------------------------
# TestComputeDifficultyScore
# ---------------------------------------------------------------------------


class TestComputeDifficultyScore:
    """Tests for _compute_difficulty_score()."""

    def test_v0_certain_returns_zero(self) -> None:
        """Certain V0 distribution must yield difficulty_score ≈ 0.0."""
        assert _compute_difficulty_score(_certain_probs(0)) == pytest.approx(0.0)

    def test_v17_certain_returns_one(self) -> None:
        """Certain V17 distribution must yield difficulty_score ≈ 1.0."""
        assert _compute_difficulty_score(_certain_probs(17)) == pytest.approx(1.0)

    def test_uniform_returns_half(self) -> None:
        """Uniform distribution must yield difficulty_score ≈ 0.5 (mean of [0..17]/17)."""
        result = _compute_difficulty_score(_uniform_probs())
        assert result == pytest.approx(0.5)

    def test_low_grade_peak_low_score(self) -> None:
        """Distribution peaked at low grades must produce difficulty_score < 0.2."""
        probs = [0.0] * _N_GRADES
        probs[0] = 0.8
        probs[1] = 0.2
        assert _compute_difficulty_score(probs) < 0.2

    def test_high_grade_peak_high_score(self) -> None:
        """Distribution peaked at high grades must produce difficulty_score > 0.8."""
        probs = [0.0] * _N_GRADES
        probs[16] = 0.8
        probs[17] = 0.2
        assert _compute_difficulty_score(probs) > 0.8

    def test_result_in_zero_one(self) -> None:
        """difficulty_score must lie within [0.0, 1.0] for every grade."""
        for idx in range(_N_GRADES):
            score = _compute_difficulty_score(_certain_probs(idx))
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestEstimateGradeML
# ---------------------------------------------------------------------------


class TestEstimateGradeML:
    """Integration tests for estimate_grade_ml()."""

    def test_returns_ml_grade_result(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """estimate_grade_ml must return an MLGradeResult instance."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        assert isinstance(result, MLGradeResult)

    def test_grade_in_v_grades(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """Predicted grade label must be one of the 18 V-grades."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        assert result.grade in V_GRADES

    def test_grade_index_in_range(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """grade_index must be within [0, 17]."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        assert 0 <= result.grade_index <= 17

    def test_confidence_in_range(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """confidence must be within [0.0, 1.0]."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        assert 0.0 <= result.confidence <= 1.0

    def test_difficulty_score_in_range(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """difficulty_score must be within [0.0, 1.0]."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        assert 0.0 <= result.difficulty_score <= 1.0

    def test_grade_probabilities_sum_to_one(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """All grade probabilities must sum to ≈1.0."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        total = sum(result.grade_probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_grade_probabilities_has_all_v_grades(
        self,
        trained_model_dir: Path,  # pylint: disable=redefined-outer-name
    ) -> None:
        """grade_probabilities must contain exactly the 18 V-grade keys."""
        features = _make_route_features()
        result = estimate_grade_ml(features, trained_model_dir)
        assert set(result.grade_probabilities.keys()) == set(V_GRADES)

    def test_caching_returns_same_model_object(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """A second call with the same path must reuse the cached model object."""
        features = _make_route_features()
        # First call loads the model
        estimate_grade_ml(features, trained_model_dir)
        cache_key = str(trained_model_dir.resolve())
        first_model, _ = _MODEL_CACHE[cache_key]
        # Second call should reuse the cached object
        estimate_grade_ml(features, trained_model_dir)
        second_model, _ = _MODEL_CACHE[cache_key]
        assert first_model is second_model

    def test_missing_model_file_raises(self, tmp_path: Path) -> None:
        """A directory without model.pkl must raise GradeEstimationError."""
        meta = {
            "feature_names": [],
            "normalization_mean": {},
            "normalization_std": {},
            "n_classes": 18,
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        features = _make_route_features()
        with pytest.raises(GradeEstimationError, match="Model file not found"):
            estimate_grade_ml(features, tmp_path)

    def test_missing_metadata_file_raises(self, tmp_path: Path) -> None:
        """A directory without metadata.json must raise GradeEstimationError."""
        (tmp_path / "model.pkl").write_bytes(b"dummy")
        features = _make_route_features()
        with pytest.raises(GradeEstimationError, match="Metadata file not found"):
            estimate_grade_ml(features, tmp_path)

    def test_wrong_model_type_raises(self, tmp_path: Path) -> None:
        """A non-XGBClassifier object loaded from pkl must raise GradeEstimationError."""
        (tmp_path / "model.pkl").write_bytes(b"placeholder")
        meta = {
            "feature_names": [],
            "normalization_mean": {},
            "normalization_std": {},
            "n_classes": 18,
            "classes": [0],
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        features = _make_route_features()
        with patch("src.grading.ml_estimator.joblib.load", return_value="not_a_model"):
            with pytest.raises(GradeEstimationError, match="Expected XGBClassifier"):
                estimate_grade_ml(features, tmp_path)

    def test_missing_metadata_field_raises(self, tmp_path: Path) -> None:
        """Metadata missing required fields must raise GradeEstimationError."""
        (tmp_path / "model.pkl").write_bytes(b"placeholder")
        (tmp_path / "metadata.json").write_text(json.dumps({"feature_names": []}))
        features = _make_route_features()
        with patch(
            "src.grading.ml_estimator.joblib.load",
            return_value=MagicMock(spec=XGBClassifier),
        ):
            with pytest.raises(GradeEstimationError, match="missing required field"):
                estimate_grade_ml(features, tmp_path)

    def test_wrong_feature_count_raises(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """A feature vector with wrong keys must raise GradeEstimationError."""
        features = _make_route_features()
        wrong_vector = {"wrong_key": 0.0}
        with patch.object(type(features), "to_vector", return_value=wrong_vector):
            with pytest.raises(GradeEstimationError, match="Feature vector"):
                estimate_grade_ml(features, trained_model_dir)

    def test_invalid_classes_in_metadata_raises(self, tmp_path: Path) -> None:
        """Out-of-range class indices in metadata must raise GradeEstimationError."""
        (tmp_path / "model.pkl").write_bytes(b"placeholder")
        meta = {
            "feature_names": [],
            "normalization_mean": {},
            "normalization_std": {},
            "n_classes": 18,
            "classes": [0, 99],  # 99 is out of range
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        features = _make_route_features()
        with patch(
            "src.grading.ml_estimator.joblib.load",
            return_value=MagicMock(spec=XGBClassifier),
        ):
            with pytest.raises(GradeEstimationError, match="out-of-range"):
                estimate_grade_ml(features, tmp_path)

    def test_feature_extraction_error_wrapped(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """A FeatureExtractionError from to_vector must be wrapped in GradeEstimationError."""
        features = _make_route_features()
        with patch.object(
            type(features), "to_vector", side_effect=FeatureExtractionError("bad")
        ):
            with pytest.raises(GradeEstimationError):
                estimate_grade_ml(features, trained_model_dir)

    def test_reproducible_results(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """Two calls with identical inputs must return the same grade and confidence."""
        features = _make_route_features()
        r1 = estimate_grade_ml(features, trained_model_dir)
        r2 = estimate_grade_ml(features, trained_model_dir)
        assert r1.grade == r2.grade
        assert r1.confidence == r2.confidence

    def test_string_path_accepted(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """estimate_grade_ml must accept a string path as well as a Path object."""
        features = _make_route_features()
        result = estimate_grade_ml(features, str(trained_model_dir))
        assert isinstance(result, MLGradeResult)


# ---------------------------------------------------------------------------
# TestPredictGrade
# ---------------------------------------------------------------------------


class TestPredictGrade:
    """Tests for _predict_grade() error paths."""

    def test_column_count_mismatch_raises(self) -> None:
        """predict_proba columns differing from trained_classes length must raise."""
        mock_clf = MagicMock()
        # predict_proba returns 3 columns but trained_classes has 2 entries
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.4, 0.3]])
        feature_vec = np.zeros((1, 1), dtype=np.float32)
        with pytest.raises(GradeEstimationError, match="probability columns"):
            _predict_grade(mock_clf, feature_vec, [0, 1])

    def test_matching_columns_no_error(self) -> None:
        """predict_proba columns matching trained_classes length must not raise."""
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.5, 0.5]])
        feature_vec = np.zeros((1, 1), dtype=np.float32)
        # _predict_grade returns (grade_index, full_18_probs_list)
        grade_index, probs = _predict_grade(mock_clf, feature_vec, [0, 1])
        assert isinstance(grade_index, int)
        assert len(probs) == _N_GRADES


# ---------------------------------------------------------------------------
# TestClearModelCache
# ---------------------------------------------------------------------------


class TestClearModelCache:
    """Tests for _clear_model_cache()."""

    def test_clears_cache(self, trained_model_dir: Path) -> None:  # pylint: disable=redefined-outer-name
        """After clearing, the model cache must be empty."""
        features = _make_route_features()
        estimate_grade_ml(features, trained_model_dir)
        assert len(_MODEL_CACHE) > 0
        _clear_model_cache()
        assert len(_MODEL_CACHE) == 0

    def test_clear_on_empty_cache_no_error(self) -> None:
        """Calling _clear_model_cache on an already-empty cache must not raise."""
        _clear_model_cache()
        _clear_model_cache()  # second call should not raise
