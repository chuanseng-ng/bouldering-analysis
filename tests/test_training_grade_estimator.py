"""Tests for src.training.train_grade_estimator module.

Covers:
- generate_synthetic_training_data()
- train_grade_estimator()
- GradeTrainingMetrics and GradeTrainingResult models
- _compute_normalization_stats()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.features.assembler import RouteFeatures
from src.grading.exceptions import GradeEstimationError
from src.training.train_grade_estimator import (
    GradeTrainingMetrics,
    GradeTrainingResult,
    _compute_normalization_stats,
    generate_synthetic_training_data,
    train_grade_estimator,
)


# ---------------------------------------------------------------------------
# TestGradeTrainingMetrics
# ---------------------------------------------------------------------------


class TestGradeTrainingMetrics:
    """Tests for GradeTrainingMetrics Pydantic model."""

    def test_valid_construction(self) -> None:
        """Valid accuracy and MAE values must construct GradeTrainingMetrics."""
        m = GradeTrainingMetrics(
            train_accuracy=0.9, val_accuracy=0.8, mean_absolute_error=0.5
        )
        assert m.train_accuracy == pytest.approx(0.9)

    def test_train_accuracy_above_one_rejected(self) -> None:
        """train_accuracy > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            GradeTrainingMetrics(
                train_accuracy=1.1, val_accuracy=0.8, mean_absolute_error=0.5
            )

    def test_val_accuracy_below_zero_rejected(self) -> None:
        """val_accuracy < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            GradeTrainingMetrics(
                train_accuracy=0.9, val_accuracy=-0.1, mean_absolute_error=0.5
            )

    def test_mae_below_zero_rejected(self) -> None:
        """mean_absolute_error < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            GradeTrainingMetrics(
                train_accuracy=0.9, val_accuracy=0.8, mean_absolute_error=-0.1
            )

    def test_zero_accuracy_accepted(self) -> None:
        """Zero accuracies and zero MAE must be valid boundary values."""
        m = GradeTrainingMetrics(
            train_accuracy=0.0, val_accuracy=0.0, mean_absolute_error=0.0
        )
        assert m.val_accuracy == 0.0


# ---------------------------------------------------------------------------
# TestGradeTrainingResult
# ---------------------------------------------------------------------------


class TestGradeTrainingResult:
    """Tests for GradeTrainingResult Pydantic model."""

    def _valid_kwargs(self, tmp_path: Path) -> dict[str, Any]:
        """Return valid kwargs for constructing a GradeTrainingResult."""
        model_p = tmp_path / "model.pkl"
        meta_p = tmp_path / "metadata.json"
        model_p.touch()
        meta_p.touch()
        return {
            "version": "v20260310_120000",
            "model_path": model_p,
            "metadata_path": meta_p,
            "metrics": GradeTrainingMetrics(
                train_accuracy=0.9, val_accuracy=0.8, mean_absolute_error=0.5
            ),
            "n_samples": 100,
            "feature_names": [f"f{i}" for i in range(40)],
            "data_source": "synthetic",
            "git_commit": "abc1234",
            "trained_at": "2026-03-10T12:00:00+00:00",
            "hyperparameters": {"n_estimators": 200},
        }

    def test_valid_construction(self, tmp_path: Path) -> None:
        """Valid kwargs must produce a correctly populated GradeTrainingResult."""
        r = GradeTrainingResult(**self._valid_kwargs(tmp_path))
        assert r.version == "v20260310_120000"

    def test_n_samples_zero_rejected(self, tmp_path: Path) -> None:
        """n_samples == 0 must raise ValidationError."""
        kw = self._valid_kwargs(tmp_path)
        kw["n_samples"] = 0
        with pytest.raises(ValidationError):
            GradeTrainingResult(**kw)

    def test_data_source_invalid_rejected(self, tmp_path: Path) -> None:
        """An unknown data_source value must raise ValidationError."""
        kw = self._valid_kwargs(tmp_path)
        kw["data_source"] = "unknown"  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            GradeTrainingResult(**kw)

    def test_git_commit_none_accepted(self, tmp_path: Path) -> None:
        """git_commit=None must be accepted as an optional field."""
        kw = self._valid_kwargs(tmp_path)
        kw["git_commit"] = None
        r = GradeTrainingResult(**kw)
        assert r.git_commit is None


# ---------------------------------------------------------------------------
# TestComputeNormalizationStats
# ---------------------------------------------------------------------------


class TestComputeNormalizationStats:
    """Tests for _compute_normalization_stats()."""

    def test_basic_mean_std(self) -> None:
        """Mean of [1, 3] must be 2.0 and population std must be 1.0."""
        vectors = [{"x": 1.0}, {"x": 3.0}]
        mean, std = _compute_normalization_stats(vectors)
        assert mean["x"] == pytest.approx(2.0)
        assert std["x"] == pytest.approx(1.0)

    def test_zero_variance_gives_zero_std(self) -> None:
        """Constant feature must yield std == 0.0."""
        vectors = [{"x": 5.0}, {"x": 5.0}, {"x": 5.0}]
        mean, std = _compute_normalization_stats(vectors)
        assert mean["x"] == pytest.approx(5.0)
        assert std["x"] == pytest.approx(0.0)

    def test_single_sample(self) -> None:
        """A single sample must yield std == 0.0 (population std)."""
        vectors = [{"x": 3.0}]
        mean, std = _compute_normalization_stats(vectors)
        assert mean["x"] == pytest.approx(3.0)
        assert std["x"] == pytest.approx(0.0)

    def test_multiple_features(self) -> None:
        """All features must be normalized independently."""
        vectors = [{"a": 0.0, "b": 10.0}, {"a": 2.0, "b": 20.0}]
        mean, std = _compute_normalization_stats(vectors)
        assert mean["a"] == pytest.approx(1.0)
        assert mean["b"] == pytest.approx(15.0)
        assert std["b"] == pytest.approx(5.0)

    def test_preserves_key_order(self) -> None:
        """Output dicts must preserve the key insertion order of the input vectors."""
        vectors = [{"c": 1.0, "a": 2.0, "b": 3.0}] * 3
        mean, _ = _compute_normalization_stats(vectors)
        assert list(mean.keys()) == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# TestGenerateSyntheticTrainingData
# ---------------------------------------------------------------------------


class TestGenerateSyntheticTrainingData:
    """Tests for generate_synthetic_training_data()."""

    def test_returns_correct_length(self) -> None:
        """The returned list must contain exactly n_samples entries."""
        samples = generate_synthetic_training_data(n_samples=20, seed=0)
        assert len(samples) == 20

    def test_items_are_tuples_of_correct_types(self) -> None:
        """Each sample must be a (RouteFeatures, int) tuple."""
        samples = generate_synthetic_training_data(n_samples=5, seed=0)
        for features, label in samples:
            assert isinstance(features, RouteFeatures)
            assert isinstance(label, int)

    def test_labels_in_valid_range(self) -> None:
        """All labels must be valid V-grade indices in [0, 17]."""
        samples = generate_synthetic_training_data(n_samples=50, seed=0)
        for _, label in samples:
            assert 0 <= label <= 17

    def test_reproducible_with_same_seed(self) -> None:
        """The same seed must produce identical label sequences."""
        s1 = generate_synthetic_training_data(n_samples=10, seed=99)
        s2 = generate_synthetic_training_data(n_samples=10, seed=99)
        assert [lbl for _, lbl in s1] == [lbl for _, lbl in s2]

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds must (very likely) produce different label sequences."""
        s1 = generate_synthetic_training_data(n_samples=20, seed=1)
        s2 = generate_synthetic_training_data(n_samples=20, seed=2)
        labels1 = [lbl for _, lbl in s1]
        labels2 = [lbl for _, lbl in s2]
        # Very unlikely to be identical with 20 samples
        assert labels1 != labels2

    def test_label_diversity_not_all_same(self) -> None:
        """With 100 samples the labels must not all be identical."""
        samples = generate_synthetic_training_data(n_samples=100, seed=0)
        labels = {lbl for _, lbl in samples}
        assert len(labels) > 1

    def test_graphs_have_edges(self) -> None:
        """At least one sample from the fixed grid must have edges."""
        samples = generate_synthetic_training_data(n_samples=10, seed=0)
        edge_counts = [f.geometry.edge_count for f, _ in samples]
        assert any(ec > 0 for ec in edge_counts)

    def test_feature_vector_has_40_keys(self) -> None:
        """Every sample's feature vector must contain exactly the 40 expected keys."""
        from src.training.classification_dataset import HOLD_CLASSES

        expected_geometry_keys = {
            "avg_move_distance",
            "max_move_distance",
            "min_move_distance",
            "std_move_distance",
            "path_length_min_distance",
            "path_length_max_distance",
            "path_length_min_hops",
            "path_length_max_hops",
            "hold_density",
            "node_count",
            "edge_count",
        }
        expected_hold_keys = (
            {f"{cls}_count" for cls in HOLD_CLASSES}
            | {f"{cls}_ratio" for cls in HOLD_CLASSES}
            | {f"{cls}_soft_ratio" for cls in HOLD_CLASSES}
            | {
                "total_count",
                "avg_hold_size",
                "max_hold_size",
                "min_hold_size",
                "std_hold_size",
            }
        )
        expected_keys = expected_geometry_keys | expected_hold_keys

        samples = generate_synthetic_training_data(n_samples=5, seed=0)
        for features, _ in samples:
            vec_keys = set(features.to_vector().keys())
            assert vec_keys == expected_keys
            assert len(vec_keys) == 40
            assert "volume_ratio" not in vec_keys


# ---------------------------------------------------------------------------
# TestTrainGradeEstimator
# ---------------------------------------------------------------------------


class TestTrainGradeEstimator:
    """Tests for train_grade_estimator()."""

    @pytest.fixture()
    def small_training_data(self) -> tuple[list[Any], list[int]]:
        """50 synthetic samples — fast enough for unit tests."""
        samples = generate_synthetic_training_data(n_samples=50, seed=42)
        features, labels = zip(*samples)
        return list(features), list(labels)

    def test_returns_grade_training_result(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """train_grade_estimator must return a GradeTrainingResult instance."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        assert isinstance(result, GradeTrainingResult)

    def test_model_pkl_created(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """model.pkl must exist in the versioned output directory."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        assert result.model_path.exists()

    def test_metadata_json_created(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """metadata.json must exist in the versioned output directory."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        assert result.metadata_path.exists()

    def test_version_format(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """Version string must match the vYYYYMMDD_HHMMSS pattern."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        assert re.fullmatch(r"v\d{8}_\d{6}", result.version)

    def test_feature_names_length(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """feature_names must contain exactly 40 entries."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        assert len(result.feature_names) == 40

    def test_metrics_in_range(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """All reported metrics must be in valid ranges."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        assert 0.0 <= result.metrics.train_accuracy <= 1.0
        assert 0.0 <= result.metrics.val_accuracy <= 1.0
        assert result.metrics.mean_absolute_error >= 0.0

    def test_metadata_json_has_required_fields(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """metadata.json must contain all required fields."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        with result.metadata_path.open() as fh:
            meta = json.load(fh)
        for field in (
            "feature_names",
            "normalization_mean",
            "normalization_std",
            "n_classes",
            "data_source",
        ):
            assert field in meta

    def test_data_source_synthetic_in_metadata(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """data_source must be 'synthetic' in metadata when using default source."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        with result.metadata_path.open() as fh:
            meta = json.load(fh)
        assert meta["data_source"] == "synthetic"

    def test_data_source_real_stored(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """data_source='real' must be persisted to both the result and metadata.json."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path, data_source="real")
        assert result.data_source == "real"
        with result.metadata_path.open() as fh:
            meta = json.load(fh)
        assert meta["data_source"] == "real"

    def test_normalization_stats_in_metadata(
        self, small_training_data: tuple[list[Any], list[int]], tmp_path: Path
    ) -> None:
        """normalization_mean and normalization_std must each have 40 entries."""
        features, labels = small_training_data
        result = train_grade_estimator(features, labels, tmp_path)
        with result.metadata_path.open() as fh:
            meta = json.load(fh)
        assert len(meta["normalization_mean"]) == 40
        assert len(meta["normalization_std"]) == 40

    def test_zero_variance_feature_std_zero_in_metadata(self, tmp_path: Path) -> None:
        """If a feature is constant across all samples, std=0.0 must be stored."""
        samples = generate_synthetic_training_data(n_samples=20, seed=0)
        features, labels = zip(*samples)

        # Identify features that are constant across all samples (e.g. node_count,
        # which is always 12 on the fixed synthetic grid).
        all_vectors = [f.to_vector() for f in features]
        constant_keys = [
            k for k in all_vectors[0] if len({v[k] for v in all_vectors}) == 1
        ]
        assert constant_keys, "Expected at least one constant feature in synthetic data"

        result = train_grade_estimator(list(features), list(labels), tmp_path)
        with result.metadata_path.open() as fh:
            meta = json.load(fh)
        for key in constant_keys:
            assert meta["normalization_std"][key] == pytest.approx(0.0), (
                f"Expected std=0.0 for constant feature '{key}'"
            )

    # ------------------------------------------------------------------
    # Input validation errors
    # ------------------------------------------------------------------

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        """Mismatched features/labels lengths must raise GradeEstimationError."""
        samples = generate_synthetic_training_data(n_samples=20, seed=0)
        features, labels = zip(*samples)
        with pytest.raises(GradeEstimationError, match="same length"):
            train_grade_estimator(list(features), list(labels)[:-1], tmp_path)

    def test_too_few_samples_raises(self, tmp_path: Path) -> None:
        """Fewer than _MIN_TRAINING_SAMPLES samples must raise GradeEstimationError."""
        samples = generate_synthetic_training_data(n_samples=5, seed=0)
        features, labels = zip(*samples)
        with pytest.raises(GradeEstimationError, match="At least"):
            train_grade_estimator(list(features), list(labels), tmp_path)

    def test_single_class_raises(self, tmp_path: Path) -> None:
        """Only one unique label class must raise GradeEstimationError."""
        samples = generate_synthetic_training_data(n_samples=20, seed=0)
        features, _ = zip(*samples)
        all_same_labels = [0] * 20  # only V0
        with pytest.raises(GradeEstimationError, match="unique grade classes"):
            train_grade_estimator(list(features), all_same_labels, tmp_path)

    def test_out_of_range_label_raises(self, tmp_path: Path) -> None:
        """A label > 17 must raise GradeEstimationError."""
        samples = generate_synthetic_training_data(n_samples=20, seed=0)
        features, labels = zip(*samples)
        bad_labels = list(labels)
        bad_labels[0] = 99
        with pytest.raises(GradeEstimationError, match="out-of-range"):
            train_grade_estimator(list(features), bad_labels, tmp_path)

    def test_negative_label_raises(self, tmp_path: Path) -> None:
        """A label < 0 must raise GradeEstimationError."""
        samples = generate_synthetic_training_data(n_samples=20, seed=0)
        features, labels = zip(*samples)
        bad_labels = list(labels)
        bad_labels[0] = -1
        with pytest.raises(GradeEstimationError, match="out-of-range"):
            train_grade_estimator(list(features), bad_labels, tmp_path)
