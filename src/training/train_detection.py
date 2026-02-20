"""Training loop for hold detection YOLOv8 models.

This module provides the main training orchestration for bouldering hold
detection. It wraps the Ultralytics YOLOv8 training API with structured
result models, artifact saving, and reproducibility metadata.

Example:
    >>> from src.training.datasets import load_hold_detection_dataset
    >>> from src.training.train_detection import train_hold_detector
    >>> dataset = load_hold_detection_dataset("data/climbing_holds")
    >>> result = train_hold_detector(dataset, "data/climbing_holds")
    >>> print(result.metrics.map50)
    0.87
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.logging_config import get_logger
from src.training.detection_model import (
    DEFAULT_MODEL_SIZE,
    DetectionHyperparameters,
    build_hold_detector,
)
from src.training.exceptions import (
    DatasetNotFoundError,
    ModelArtifactError,
    TrainingRunError,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_BASE_DIR: Path = Path("models/detection")
METADATA_FILENAME: str = "metadata.json"
VERSION_FORMAT: str = "v%Y%m%d_%H%M%S"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class TrainingMetrics(BaseModel):
    """Metrics captured at the end of a training run.

    Attributes:
        map50: Mean Average Precision at IoU threshold 0.50 (mAP50).
        map50_95: Mean Average Precision averaged over IoU 0.50:0.95 (mAP50-95).
        precision: Precision at best F1 threshold.
        recall: Recall at best F1 threshold.
        best_epoch: Epoch index at which the best checkpoint was saved.
    """

    map50: float = Field(ge=0.0, le=1.0)
    map50_95: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    best_epoch: int = Field(ge=0)


class TrainingResult(BaseModel):
    """Full output of a completed training run.

    Attributes:
        version: Unique version string in the format v%Y%m%d_%H%M%S.
        model_size: YOLOv8 variant used (e.g. 'yolov8m').
        best_weights_path: Absolute path to best.pt checkpoint.
        last_weights_path: Absolute path to last.pt checkpoint.
        metadata_path: Absolute path to metadata.json artifact.
        metrics: Training metrics from the best epoch.
        dataset_version: Version string from data.yaml, or None.
        git_commit: Short git commit hash at training time, or None.
        trained_at: ISO-8601 UTC timestamp when training completed.
        hyperparameters: Dictionary of hyperparameter values used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    version: str
    model_size: str
    best_weights_path: Path
    last_weights_path: Path
    metadata_path: Path
    metrics: TrainingMetrics
    dataset_version: str | None
    git_commit: str | None
    trained_at: str
    hyperparameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_version() -> str:
    """Generate a unique version string based on the current UTC time.

    Returns:
        Version string in the format v%Y%m%d_%H%M%S (e.g. 'v20260220_143022').
    """
    return datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)


def _get_git_commit_hash() -> str | None:
    """Get the short git commit hash of the current HEAD.

    Returns:
        Short commit hash string (e.g. 'abc1234'), or None if git is unavailable
        or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return None


def _resolve_data_yaml(dataset_root: Path) -> Path:
    """Resolve the absolute path to data.yaml inside dataset_root.

    Args:
        dataset_root: Root directory of the YOLOv8 dataset.

    Returns:
        Absolute path to data.yaml.

    Raises:
        DatasetNotFoundError: If data.yaml does not exist in dataset_root.
    """
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise DatasetNotFoundError(f"data.yaml not found in: {dataset_root}")
    return yaml_path


def _run_yolo_training(
    model: Any,
    data_yaml_path: Path,
    hyperparameters: DetectionHyperparameters,
    project_dir: Path,
    run_name: str,
) -> tuple[Any, Path]:
    """Call model.train() and return the results and save directory.

    Args:
        model: Ultralytics YOLO model instance.
        data_yaml_path: Path to the data.yaml config file.
        hyperparameters: DetectionHyperparameters to pass to model.train().
        project_dir: Parent directory for YOLO run output.
        run_name: Name for this training run (sub-directory inside project_dir).

    Returns:
        Tuple of (yolo_results, save_dir) where save_dir is a Path.

    Raises:
        TrainingRunError: If model.train() raises any exception.
    """
    try:
        yolo_results = model.train(
            data=str(data_yaml_path),
            project=str(project_dir),
            name=run_name,
            **hyperparameters.to_dict(),
        )
        save_dir = Path(yolo_results.save_dir)
        return yolo_results, save_dir
    except Exception as exc:
        raise TrainingRunError(f"YOLO training failed: {exc}") from exc


def _extract_metrics(
    yolo_results: Any,
    hyperparameters: DetectionHyperparameters,
) -> TrainingMetrics:
    """Extract training metrics from YOLO results object.

    Uses getattr with defaults to handle missing attributes gracefully.

    Args:
        yolo_results: Object returned by model.train(); expected to have
            results_dict and best_epoch attributes.
        hyperparameters: Used to fall back on epochs if best_epoch is absent.

    Returns:
        TrainingMetrics populated from the YOLO results.
    """
    results_dict: dict[str, float] = getattr(yolo_results, "results_dict", {})
    best_epoch: int = getattr(yolo_results, "best_epoch", hyperparameters.epochs)

    return TrainingMetrics(
        map50=results_dict.get("metrics/mAP50(B)", 0.0),
        map50_95=results_dict.get("metrics/mAP50-95(B)", 0.0),
        precision=results_dict.get("metrics/precision(B)", 0.0),
        recall=results_dict.get("metrics/recall(B)", 0.0),
        best_epoch=best_epoch,
    )


def _build_metadata(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    version: str,
    model_size: str,
    trained_at: str,
    git_commit: str | None,
    dataset_version: str | None,
    dataset_train_image_count: int,
    dataset_val_image_count: int,
    hyperparameters: DetectionHyperparameters,
    metrics: TrainingMetrics,
) -> dict[str, Any]:
    """Assemble the metadata dictionary to be saved as metadata.json.

    Args:
        version: Run version string.
        model_size: YOLOv8 variant name.
        trained_at: ISO-8601 UTC timestamp string.
        git_commit: Short git commit hash or None.
        dataset_version: Dataset version string or None.
        dataset_train_image_count: Number of training images.
        dataset_val_image_count: Number of validation images.
        hyperparameters: Hyperparameters used for training.
        metrics: Final training metrics.

    Returns:
        JSON-serializable dictionary with all required metadata fields.
    """
    return {
        "version": version,
        "model_size": model_size,
        "trained_at": trained_at,
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "dataset_train_image_count": dataset_train_image_count,
        "dataset_val_image_count": dataset_val_image_count,
        "hyperparameters": hyperparameters.to_dict(),
        "metrics": metrics.model_dump(),
    }


def _save_artifacts(  # pylint: disable=too-many-arguments
    yolo_save_dir: Path,
    version: str,
    output_dir: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Copy training artifacts to the versioned output directory.

    Creates the following layout:
        <output_dir>/<version>/
            weights/
                best.pt
                last.pt
            metadata.json

    Args:
        yolo_save_dir: YOLO run save directory containing weights/.
        version: Version string (used as sub-directory name).
        output_dir: Base output directory for all model versions.
        metadata: Dictionary to serialize as metadata.json.

    Returns:
        Tuple of (best_weights_path, last_weights_path, metadata_path).

    Raises:
        ModelArtifactError: If weight files are missing or cannot be copied.
    """
    version_dir = output_dir / version
    weights_out = version_dir / "weights"

    src_weights = yolo_save_dir / "weights"
    best_src = src_weights / "best.pt"
    last_src = src_weights / "last.pt"

    if not best_src.exists() or not last_src.exists():
        raise ModelArtifactError(f"Weight files not found in: {src_weights}")

    try:
        weights_out.mkdir(parents=True, exist_ok=True)
        best_dst = weights_out / "best.pt"
        last_dst = weights_out / "last.pt"
        shutil.copy2(best_src, best_dst)
        shutil.copy2(last_src, last_dst)

        meta_path = version_dir / METADATA_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return best_dst, last_dst, meta_path

    except IOError as exc:
        raise ModelArtifactError(f"Failed to save training artifacts: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_hold_detector(  # pylint: disable=too-many-arguments,too-many-locals
    dataset: dict[str, Any],
    dataset_root: Path | str,
    hyperparameters: DetectionHyperparameters | None = None,
    output_dir: Path | str | None = None,
    model_size: str = DEFAULT_MODEL_SIZE,
) -> TrainingResult:
    """Train a YOLOv8 hold detection model.

    Orchestrates the full training pipeline:
    1. Validate dataset root has data.yaml
    2. Build the YOLO model
    3. Run training via model.train()
    4. Extract metrics from results
    5. Save artifacts (weights + metadata.json) to versioned directory

    Args:
        dataset: Dataset configuration dict from load_hold_detection_dataset().
            Used to read image counts and dataset version.
        dataset_root: Path to dataset root directory containing data.yaml.
        hyperparameters: Training hyperparameters. Uses defaults if None.
        output_dir: Base directory to save model artifacts. Defaults to
            MODELS_BASE_DIR ('models/detection').
        model_size: YOLOv8 variant to train (default: DEFAULT_MODEL_SIZE).

    Returns:
        TrainingResult with paths to saved artifacts and training metrics.

    Raises:
        DatasetNotFoundError: If data.yaml is not found in dataset_root.
        TrainingRunError: If model.train() raises an exception.
        ModelArtifactError: If saving artifacts fails after training.

    Example:
        >>> from src.training.datasets import load_hold_detection_dataset
        >>> from src.training.train_detection import train_hold_detector
        >>> dataset = load_hold_detection_dataset("data/climbing_holds")
        >>> result = train_hold_detector(dataset, "data/climbing_holds")
        >>> print(result.metrics.map50)
        0.87
    """
    if hyperparameters is None:
        hyperparameters = DetectionHyperparameters()

    resolved_root = Path(dataset_root).resolve()
    resolved_output = Path(output_dir) if output_dir is not None else MODELS_BASE_DIR

    # Validate data.yaml exists
    data_yaml_path = _resolve_data_yaml(resolved_root)

    # Generate version and capture git commit
    version = _generate_version()
    git_commit = _get_git_commit_hash()
    trained_at = datetime.now(tz=timezone.utc).isoformat()

    logger.info(
        "Starting training run %s with model=%s, epochs=%d",
        version,
        model_size,
        hyperparameters.epochs,
    )

    # Build model and run training
    model = build_hold_detector(model_size=model_size)
    project_dir = resolved_output / "_runs"
    yolo_results, yolo_save_dir = _run_yolo_training(
        model=model,
        data_yaml_path=data_yaml_path,
        hyperparameters=hyperparameters,
        project_dir=project_dir,
        run_name=version,
    )

    # Extract metrics
    metrics = _extract_metrics(yolo_results, hyperparameters)

    # Build metadata
    metadata = _build_metadata(
        version=version,
        model_size=model_size,
        trained_at=trained_at,
        git_commit=git_commit,
        dataset_version=dataset.get("version"),
        dataset_train_image_count=dataset.get("train_image_count", 0),
        dataset_val_image_count=dataset.get("val_image_count", 0),
        hyperparameters=hyperparameters,
        metrics=metrics,
    )

    # Save artifacts
    best_path, last_path, meta_path = _save_artifacts(
        yolo_save_dir=yolo_save_dir,
        version=version,
        output_dir=resolved_output,
        metadata=metadata,
    )

    logger.info(
        "Training complete. mAP50=%.3f, best_epoch=%d, artifacts=%s",
        metrics.map50,
        metrics.best_epoch,
        best_path.parent.parent,
    )

    return TrainingResult(
        version=version,
        model_size=model_size,
        best_weights_path=best_path,
        last_weights_path=last_path,
        metadata_path=meta_path,
        metrics=metrics,
        dataset_version=dataset.get("version"),
        git_commit=git_commit,
        trained_at=trained_at,
        hyperparameters=hyperparameters.to_dict(),
    )
