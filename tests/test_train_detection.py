"""Tests for the hold detection training loop.

This module tests the training pipeline for YOLOv8 hold detection models,
including result models, artifact saving, and the main train_hold_detector
function.

Tests follow TDD: written before implementation.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.training.detection_model import DetectionHyperparameters
from src.training.exceptions import (
    DatasetNotFoundError,
    ModelArtifactError,
    TrainingRunError,
)
from src.training.train_detection import (
    TrainingMetrics,
    TrainingResult,
    _build_metadata,
    _extract_metrics,
    _generate_version,
    _get_git_commit_hash,
    _resolve_data_yaml,
    _run_yolo_training,
    _save_artifacts,
    train_hold_detector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_yolo_model(tmp_path: Path) -> MagicMock:
    """Fixture providing a mock YOLO model with realistic training output."""
    model = MagicMock()

    weights_dir = tmp_path / "yolo_run" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"fake_best_weights")
    (weights_dir / "last.pt").write_bytes(b"fake_last_weights")

    results = MagicMock()
    results.save_dir = tmp_path / "yolo_run"
    results.results_dict = {
        "metrics/mAP50(B)": 0.87,
        "metrics/mAP50-95(B)": 0.64,
        "metrics/precision(B)": 0.89,
        "metrics/recall(B)": 0.85,
    }
    results.best_epoch = 87
    model.train.return_value = results
    return model


@pytest.fixture
def valid_dataset_root(tmp_path: Path) -> Path:
    """Fixture providing a minimal valid dataset directory with data.yaml."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 2\n"
        "names: ['hold', 'volume']\n"
        "dataset_version: v1.0\n",
        encoding="utf-8",
    )

    (dataset_dir / "train" / "images").mkdir(parents=True)
    (dataset_dir / "valid" / "images").mkdir(parents=True)

    return dataset_dir


@pytest.fixture
def sample_dataset(valid_dataset_root: Path) -> dict[str, Any]:
    """Fixture providing a pre-loaded dataset dict."""
    return {
        "train": valid_dataset_root / "train",
        "val": valid_dataset_root / "valid",
        "test": None,
        "nc": 2,
        "names": ["hold", "volume"],
        "train_image_count": 100,
        "val_image_count": 20,
        "test_image_count": 0,
        "version": "v1.0",
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# TestTrainingMetrics
# ---------------------------------------------------------------------------


class TestTrainingMetrics:
    """Tests for TrainingMetrics Pydantic model."""

    def test_valid_metrics_construction(self) -> None:
        """TrainingMetrics accepts all-valid float fields."""
        m = TrainingMetrics(
            map50=0.87,
            map50_95=0.64,
            precision=0.89,
            recall=0.85,
            best_epoch=87,
        )
        assert m.map50 == pytest.approx(0.87)
        assert m.map50_95 == pytest.approx(0.64)
        assert m.best_epoch == 87

    def test_metrics_bounds_zero(self) -> None:
        """TrainingMetrics accepts 0.0 for float fields."""
        m = TrainingMetrics(
            map50=0.0, map50_95=0.0, precision=0.0, recall=0.0, best_epoch=0
        )
        assert m.map50 == 0.0

    def test_metrics_bounds_one(self) -> None:
        """TrainingMetrics accepts 1.0 for float fields."""
        m = TrainingMetrics(
            map50=1.0, map50_95=1.0, precision=1.0, recall=1.0, best_epoch=99
        )
        assert m.map50 == 1.0

    def test_metrics_rejects_out_of_bounds(self) -> None:
        """TrainingMetrics rejects values outside [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            TrainingMetrics(
                map50=1.5,
                map50_95=0.5,
                precision=0.5,
                recall=0.5,
                best_epoch=0,
            )

    def test_metrics_rejects_negative_epoch(self) -> None:
        """TrainingMetrics rejects negative best_epoch."""
        with pytest.raises(ValidationError):
            TrainingMetrics(
                map50=0.5,
                map50_95=0.5,
                precision=0.5,
                recall=0.5,
                best_epoch=-1,
            )


# ---------------------------------------------------------------------------
# TestTrainingResult
# ---------------------------------------------------------------------------


class TestTrainingResult:
    """Tests for TrainingResult Pydantic model."""

    def test_training_result_construction(self, tmp_path: Path) -> None:
        """TrainingResult can be constructed with all required fields."""
        metrics = TrainingMetrics(
            map50=0.87, map50_95=0.64, precision=0.89, recall=0.85, best_epoch=87
        )
        result = TrainingResult(
            version="v20260220_143022",
            model_size="yolov8m",
            best_weights_path=tmp_path / "best.pt",
            last_weights_path=tmp_path / "last.pt",
            metadata_path=tmp_path / "metadata.json",
            metrics=metrics,
            dataset_version="v1.0",
            git_commit="abc1234",
            trained_at="2026-02-20T14:30:22Z",
            hyperparameters={"epochs": 100, "batch": 16},
        )
        assert result.version == "v20260220_143022"
        assert result.model_size == "yolov8m"
        assert result.git_commit == "abc1234"

    def test_training_result_optional_fields(self, tmp_path: Path) -> None:
        """TrainingResult allows None for optional fields."""
        metrics = TrainingMetrics(
            map50=0.5, map50_95=0.5, precision=0.5, recall=0.5, best_epoch=0
        )
        result = TrainingResult(
            version="v20260220_000000",
            model_size="yolov8n",
            best_weights_path=tmp_path / "best.pt",
            last_weights_path=tmp_path / "last.pt",
            metadata_path=tmp_path / "metadata.json",
            metrics=metrics,
            dataset_version=None,
            git_commit=None,
            trained_at="2026-02-20T00:00:00Z",
            hyperparameters={},
        )
        assert result.dataset_version is None
        assert result.git_commit is None


# ---------------------------------------------------------------------------
# TestGenerateVersion
# ---------------------------------------------------------------------------


class TestGenerateVersion:
    """Tests for _generate_version helper."""

    def test_version_format(self) -> None:
        """_generate_version returns a string matching v\\d{8}_\\d{6}."""
        version = _generate_version()
        assert re.match(r"^v\d{8}_\d{6}$", version), f"Bad format: {version!r}"

    def test_version_starts_with_v(self) -> None:
        """_generate_version always starts with 'v'."""
        assert _generate_version().startswith("v")

    def test_versions_are_unique(self) -> None:
        """Two calls separated in time produce different versions."""
        v1 = _generate_version()
        time.sleep(1.1)
        v2 = _generate_version()
        assert v1 != v2


# ---------------------------------------------------------------------------
# TestGetGitCommitHash
# ---------------------------------------------------------------------------


class TestGetGitCommitHash:
    """Tests for _get_git_commit_hash helper."""

    def test_returns_string_in_git_repo(self) -> None:
        """_get_git_commit_hash returns a non-empty string in a git repo."""
        result = _get_git_commit_hash()
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_none_on_failure(self) -> None:
        """_get_git_commit_hash returns None when subprocess fails."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_git_commit_hash()
        assert result is None

    def test_returns_none_on_nonzero_returncode(self) -> None:
        """_get_git_commit_hash returns None when git command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = _get_git_commit_hash()
        assert result is None


# ---------------------------------------------------------------------------
# TestResolveDatYaml
# ---------------------------------------------------------------------------


class TestResolveDatYaml:
    """Tests for _resolve_data_yaml helper."""

    def test_returns_path_when_yaml_exists(self, tmp_path: Path) -> None:
        """_resolve_data_yaml returns Path when data.yaml exists."""
        (tmp_path / "data.yaml").write_text("nc: 2\n", encoding="utf-8")
        result = _resolve_data_yaml(tmp_path)
        assert result == tmp_path / "data.yaml"

    def test_raises_when_yaml_missing(self, tmp_path: Path) -> None:
        """_resolve_data_yaml raises DatasetNotFoundError when data.yaml missing."""
        with pytest.raises(DatasetNotFoundError):
            _resolve_data_yaml(tmp_path)


# ---------------------------------------------------------------------------
# TestRunYoloTraining
# ---------------------------------------------------------------------------


class TestRunYoloTraining:
    """Tests for _run_yolo_training helper."""

    def test_calls_model_train(
        self, mock_yolo_model: MagicMock, tmp_path: Path
    ) -> None:
        """_run_yolo_training calls model.train() with correct arguments."""
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\n", encoding="utf-8")

        hyperparams = DetectionHyperparameters(epochs=5)
        _run_yolo_training(
            model=mock_yolo_model,
            data_yaml_path=data_yaml,
            hyperparameters=hyperparams,
            project_dir=tmp_path / "runs",
            run_name="test_run",
        )

        mock_yolo_model.train.assert_called_once()
        call_kwargs = mock_yolo_model.train.call_args[1]
        assert "data" in call_kwargs
        assert "project" in call_kwargs
        assert "name" in call_kwargs

    def test_wraps_exception_in_training_run_error(
        self, mock_yolo_model: MagicMock, tmp_path: Path
    ) -> None:
        """_run_yolo_training wraps exceptions in TrainingRunError."""
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\n", encoding="utf-8")

        mock_yolo_model.train.side_effect = RuntimeError("CUDA out of memory")
        hyperparams = DetectionHyperparameters(epochs=5)

        with pytest.raises(TrainingRunError) as exc_info:
            _run_yolo_training(
                model=mock_yolo_model,
                data_yaml_path=data_yaml,
                hyperparameters=hyperparams,
                project_dir=tmp_path / "runs",
                run_name="test_run",
            )
        assert "CUDA out of memory" in exc_info.value.message

    def test_returns_results_and_save_dir(
        self, mock_yolo_model: MagicMock, tmp_path: Path
    ) -> None:
        """_run_yolo_training returns (results, save_dir) tuple."""
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("nc: 2\n", encoding="utf-8")

        hyperparams = DetectionHyperparameters(epochs=5)
        results, save_dir = _run_yolo_training(
            model=mock_yolo_model,
            data_yaml_path=data_yaml,
            hyperparameters=hyperparams,
            project_dir=tmp_path / "runs",
            run_name="test_run",
        )
        assert results is not None
        assert isinstance(save_dir, Path)


# ---------------------------------------------------------------------------
# TestExtractMetrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    """Tests for _extract_metrics helper."""

    def test_extracts_from_results_dict(self) -> None:
        """_extract_metrics parses results_dict correctly."""
        yolo_results = MagicMock()
        yolo_results.results_dict = {
            "metrics/mAP50(B)": 0.87,
            "metrics/mAP50-95(B)": 0.64,
            "metrics/precision(B)": 0.89,
            "metrics/recall(B)": 0.85,
        }
        yolo_results.best_epoch = 42

        hyperparams = DetectionHyperparameters(epochs=100)
        metrics = _extract_metrics(yolo_results, hyperparams)

        assert metrics.map50 == pytest.approx(0.87)
        assert metrics.map50_95 == pytest.approx(0.64)
        assert metrics.precision == pytest.approx(0.89)
        assert metrics.recall == pytest.approx(0.85)
        assert metrics.best_epoch == 42

    def test_defaults_to_zero_for_missing_keys(self) -> None:
        """_extract_metrics uses 0.0 for missing metric keys."""
        yolo_results = MagicMock()
        yolo_results.results_dict = {}
        yolo_results.best_epoch = 0

        hyperparams = DetectionHyperparameters(epochs=50)
        metrics = _extract_metrics(yolo_results, hyperparams)

        assert metrics.map50 == 0.0
        assert metrics.precision == 0.0

    def test_uses_epochs_when_best_epoch_missing(self) -> None:
        """_extract_metrics falls back to hyperparams.epochs when best_epoch absent."""
        yolo_results = MagicMock(spec=[])  # no attributes
        hyperparams = DetectionHyperparameters(epochs=50)
        metrics = _extract_metrics(yolo_results, hyperparams)

        assert metrics.best_epoch == 50


# ---------------------------------------------------------------------------
# TestBuildMetadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Tests for _build_metadata helper."""

    def test_assembles_correct_shape(self) -> None:
        """_build_metadata returns dict with all required keys."""
        metrics = TrainingMetrics(
            map50=0.87, map50_95=0.64, precision=0.89, recall=0.85, best_epoch=87
        )
        hyperparams = DetectionHyperparameters(epochs=100)

        meta = _build_metadata(
            version="v20260220_143022",
            model_size="yolov8m",
            trained_at="2026-02-20T14:30:22Z",
            git_commit="abc1234",
            dataset_version="v1.0",
            dataset_train_image_count=100,
            dataset_val_image_count=20,
            hyperparameters=hyperparams,
            metrics=metrics,
        )

        assert meta["version"] == "v20260220_143022"
        assert meta["model_size"] == "yolov8m"
        assert meta["trained_at"] == "2026-02-20T14:30:22Z"
        assert meta["git_commit"] == "abc1234"
        assert meta["dataset_version"] == "v1.0"
        assert meta["dataset_train_image_count"] == 100
        assert meta["dataset_val_image_count"] == 20
        assert isinstance(meta["hyperparameters"], dict)
        assert isinstance(meta["metrics"], dict)

    def test_metadata_is_json_serializable(self) -> None:
        """_build_metadata returns a JSON-serializable dict."""
        metrics = TrainingMetrics(
            map50=0.5, map50_95=0.5, precision=0.5, recall=0.5, best_epoch=0
        )
        hyperparams = DetectionHyperparameters(epochs=10)

        meta = _build_metadata(
            version="v20260220_000000",
            model_size="yolov8n",
            trained_at="2026-02-20T00:00:00Z",
            git_commit=None,
            dataset_version=None,
            dataset_train_image_count=10,
            dataset_val_image_count=5,
            hyperparameters=hyperparams,
            metrics=metrics,
        )

        json_str = json.dumps(meta)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# TestSaveArtifacts
# ---------------------------------------------------------------------------


class TestSaveArtifacts:
    """Tests for _save_artifacts helper."""

    def test_creates_versioned_directory(self, tmp_path: Path) -> None:
        """_save_artifacts creates models/detection/<version>/weights/ structure."""
        yolo_save_dir = tmp_path / "yolo_run"
        weights_dir = yolo_save_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake_best")
        (weights_dir / "last.pt").write_bytes(b"fake_last")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        metrics = TrainingMetrics(
            map50=0.87, map50_95=0.64, precision=0.89, recall=0.85, best_epoch=87
        )
        hyperparams = DetectionHyperparameters(epochs=100)
        meta = {
            "version": "v20260220_143022",
            "hyperparameters": hyperparams.to_dict(),
            "metrics": metrics.model_dump(),
        }

        best_path, last_path, meta_path = _save_artifacts(
            yolo_save_dir=yolo_save_dir,
            version="v20260220_143022",
            output_dir=output_dir,
            metadata=meta,
        )

        assert best_path.exists()
        assert last_path.exists()
        assert meta_path.exists()
        assert best_path.name == "best.pt"
        assert last_path.name == "last.pt"
        assert meta_path.name == "metadata.json"

    def test_metadata_json_content(self, tmp_path: Path) -> None:
        """_save_artifacts writes valid JSON to metadata.json."""
        yolo_save_dir = tmp_path / "yolo_run"
        weights_dir = yolo_save_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"fake")
        (weights_dir / "last.pt").write_bytes(b"fake")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        meta = {"version": "v20260220_000000"}

        _, _, meta_path = _save_artifacts(
            yolo_save_dir=yolo_save_dir,
            version="v20260220_000000",
            output_dir=output_dir,
            metadata=meta,
        )

        with open(meta_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["version"] == "v20260220_000000"

    def test_raises_model_artifact_error_on_missing_weights(
        self, tmp_path: Path
    ) -> None:
        """_save_artifacts raises ModelArtifactError when weight files missing."""
        yolo_save_dir = tmp_path / "yolo_run"
        yolo_save_dir.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(ModelArtifactError):
            _save_artifacts(
                yolo_save_dir=yolo_save_dir,
                version="v20260220_000000",
                output_dir=output_dir,
                metadata={"version": "v20260220_000000"},
            )


# ---------------------------------------------------------------------------
# TestTrainHoldDetector (integration)
# ---------------------------------------------------------------------------


class TestTrainHoldDetector:
    """Integration tests for train_hold_detector."""

    @patch("src.training.train_detection.build_hold_detector")
    def test_returns_training_result(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        valid_dataset_root: Path,
        tmp_path: Path,
    ) -> None:
        """train_hold_detector returns a TrainingResult on success."""
        mock_build.return_value = mock_yolo_model

        result = train_hold_detector(
            dataset=sample_dataset,
            dataset_root=valid_dataset_root,
            output_dir=tmp_path / "models",
        )

        assert isinstance(result, TrainingResult)
        assert result.best_weights_path.exists()
        assert result.metadata_path.exists()

    @patch("src.training.train_detection.build_hold_detector")
    def test_uses_custom_hyperparameters(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        valid_dataset_root: Path,
        tmp_path: Path,
    ) -> None:
        """train_hold_detector passes custom hyperparameters to model.train()."""
        mock_build.return_value = mock_yolo_model

        hyperparams = DetectionHyperparameters(epochs=5, batch=8)
        train_hold_detector(
            dataset=sample_dataset,
            dataset_root=valid_dataset_root,
            hyperparameters=hyperparams,
            output_dir=tmp_path / "models",
        )

        call_kwargs = mock_yolo_model.train.call_args[1]
        assert call_kwargs.get("epochs") == 5

    @patch("src.training.train_detection.build_hold_detector")
    def test_raises_dataset_not_found_error_for_missing_yaml(
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """train_hold_detector raises DatasetNotFoundError for missing data.yaml."""
        mock_build.return_value = mock_yolo_model
        missing_root = tmp_path / "no_such_dir"

        with pytest.raises(DatasetNotFoundError):
            train_hold_detector(
                dataset=sample_dataset,
                dataset_root=missing_root,
                output_dir=tmp_path / "models",
            )

    @patch("src.training.train_detection.build_hold_detector")
    def test_raises_training_run_error_when_yolo_fails(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        valid_dataset_root: Path,
        tmp_path: Path,
    ) -> None:
        """train_hold_detector raises TrainingRunError when YOLO training fails."""
        mock_yolo_model.train.side_effect = RuntimeError("CUDA out of memory")
        mock_build.return_value = mock_yolo_model

        with pytest.raises(TrainingRunError):
            train_hold_detector(
                dataset=sample_dataset,
                dataset_root=valid_dataset_root,
                output_dir=tmp_path / "models",
            )

    @patch("src.training.train_detection.build_hold_detector")
    def test_version_in_result_matches_format(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        valid_dataset_root: Path,
        tmp_path: Path,
    ) -> None:
        """train_hold_detector produces a version string in the correct format."""
        mock_build.return_value = mock_yolo_model

        result = train_hold_detector(
            dataset=sample_dataset,
            dataset_root=valid_dataset_root,
            output_dir=tmp_path / "models",
        )
        assert re.match(r"^v\d{8}_\d{6}$", result.version)

    @patch("src.training.train_detection.build_hold_detector")
    def test_uses_default_hyperparameters_when_none(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        valid_dataset_root: Path,
        tmp_path: Path,
    ) -> None:
        """train_hold_detector uses default hyperparameters when None provided."""
        mock_build.return_value = mock_yolo_model

        result = train_hold_detector(
            dataset=sample_dataset,
            dataset_root=valid_dataset_root,
            output_dir=tmp_path / "models",
        )
        assert "epochs" in result.hyperparameters

    @patch("src.training.train_detection.build_hold_detector")
    def test_metrics_populated_from_yolo_results(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_build: MagicMock,
        mock_yolo_model: MagicMock,
        sample_dataset: dict[str, Any],
        valid_dataset_root: Path,
        tmp_path: Path,
    ) -> None:
        """train_hold_detector populates metrics from YOLO results."""
        mock_build.return_value = mock_yolo_model

        result = train_hold_detector(
            dataset=sample_dataset,
            dataset_root=valid_dataset_root,
            output_dir=tmp_path / "models",
        )
        assert result.metrics.map50 == pytest.approx(0.87)
        assert result.metrics.best_epoch == 87
