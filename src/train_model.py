# pylint: disable=duplicate-code
"""
YOLOv8 Fine-tuning Pipeline for Bouldering Hold Detection and Classification.

This script provides a comprehensive training pipeline for fine-tuning YOLOv8 models
on custom bouldering hold datasets. It includes dataset validation, model training,
versioning, and database integration.

Supported Model Types:
    - hold_detection: Object detection model that finds holds and classifies their types
    - hold_classification: Image classification model that classifies cropped hold images

Usage Examples:
    Basic hold detection training with default parameters:
        python src/train_model.py --model-name v1.0

    Hold classification training:
        python src/train_model.py \\
            --model-name v1.0 \\
            --model-type hold_classification \\
            --data-path data/hold_classification/

    Custom training configuration:
        python src/train_model.py \\
            --model-name v1.1 \\
            --epochs 100 \\
            --batch-size 16 \\
            --learning-rate 0.01

    Training with automatic model activation:
        python src/train_model.py \\
            --model-name v2.0 \\
            --epochs 150 \\
            --activate

    Using custom dataset:
        python src/train_model.py \\
            --model-name custom_v1 \\
            --data-yaml data/custom_dataset/data.yaml \\
            --base-weights models/hold_detection/v1.0.pt
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

import yaml
from flask import Flask

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from src.config import (  # noqa: E402
    get_project_root,
    resolve_path,
    get_config_value,
    get_model_path,
    get_data_path,
    ConfigurationError,
)
from src.models import db, ModelVersion  # noqa: E402

# pylint: enable=wrong-import-position


# Configure logging
# First ensure the log directory exists
log_file_path = Path("logs/training.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Set up handlers with fallback for FileHandler failures
handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
try:
    handlers.append(logging.FileHandler(str(log_file_path), mode="a"))
except (OSError, IOError) as e:
    # If FileHandler creation fails, log to stdout only
    print(f"Warning: Could not create log file {log_file_path}: {e}", file=sys.stderr)
    print("Logging will continue to stdout only.", file=sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when there are issues during the training process."""


# Model type constants
MODEL_TYPE_HOLD_DETECTION = "hold_detection"
MODEL_TYPE_HOLD_CLASSIFICATION = "hold_classification"
VALID_MODEL_TYPES = [MODEL_TYPE_HOLD_DETECTION, MODEL_TYPE_HOLD_CLASSIFICATION]

# Default hold classes for classification
DEFAULT_HOLD_CLASSES = [
    "crimp",
    "jug",
    "sloper",
    "pinch",
    "pocket",
    "foot-hold",
    "start-hold",
    "top-out-hold",
]


def validate_classification_dataset(
    data_path: Path,
) -> dict[str, Any]:
    """
    Validate the YOLO classification dataset directory structure.

    YOLOv8 classification datasets use folder-based structure where each class
    has its own subdirectory containing images of that class.

    Expected structure:
        data_path/
        ├── train/
        │   ├── crimp/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   ├── jug/
        │   └── ...
        └── val/
            ├── crimp/
            ├── jug/
            └── ...

    Args:
        data_path: Path to the classification dataset root directory.

    Returns:
        Dict containing dataset configuration with keys:
            - train: Path to training directory
            - val: Path to validation directory
            - nc: Number of classes
            - names: List of class names

    Raises:
        TrainingError: If the dataset is invalid or improperly formatted.
    """
    logger.info("Validating classification dataset: %s", data_path)

    if not data_path.exists():
        raise TrainingError(
            f"Classification dataset directory not found: {data_path}\n"
            f"Please create a directory with train/ and val/ subdirectories."
        )

    # Validate train and val directories exist
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists():
        raise TrainingError(
            f"Training directory not found: {train_dir}\n"
            f"Expected structure: {data_path}/train/<class_name>/images"
        )

    if not val_dir.exists():
        raise TrainingError(
            f"Validation directory not found: {val_dir}\n"
            f"Expected structure: {data_path}/val/<class_name>/images"
        )

    # Get class names from training directory (subdirectories = classes)
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    if not train_classes:
        raise TrainingError(
            f"No class subdirectories found in {train_dir}\n"
            f"Expected structure: train/<class_name>/images"
        )

    # Warn if train and val have different classes
    if set(train_classes) != set(val_classes):
        missing_in_val = set(train_classes) - set(val_classes)
        missing_in_train = set(val_classes) - set(train_classes)
        logger.warning(
            "Class mismatch between train and val sets. "
            "Missing in val: %s, Missing in train: %s",
            missing_in_val,
            missing_in_train,
        )

    # Validate each class directory has images
    total_train_images = 0
    total_val_images = 0

    for class_name in train_classes:
        class_dir = train_dir / class_name
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        if not images:
            raise TrainingError(
                f"No images found in training class directory: {class_dir}\n"
                f"Supported formats: .jpg, .jpeg, .png"
            )
        total_train_images += len(images)
        logger.info("  Train class '%s': %d images", class_name, len(images))

    for class_name in val_classes:
        class_dir = val_dir / class_name
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        if not images:
            logger.warning(
                "No images found in validation class directory: %s", class_dir
            )
        total_val_images += len(images)
        logger.info("  Val class '%s': %d images", class_name, len(images))

    logger.info(
        "Classification dataset validated: %d classes, %d train images, %d val images",
        len(train_classes),
        total_train_images,
        total_val_images,
    )

    return {
        "train": str(train_dir),
        "val": str(val_dir),
        "nc": len(train_classes),
        "names": train_classes,
    }


def validate_dataset(
    data_yaml_path: Path,
) -> dict[str, Any]:  # pylint: disable=too-many-locals,too-many-branches
    """
    Validate the YOLO dataset configuration file and directory structure.

    Args:
        data_yaml_path: Path to the YOLO dataset configuration YAML file.

    Returns:
        Dict containing the parsed dataset configuration.

    Raises:
        TrainingError: If the dataset is invalid or improperly formatted.
    """
    logger.info("Validating dataset configuration: %s", data_yaml_path)

    # Check if data.yaml exists
    if not data_yaml_path.exists():
        raise TrainingError(
            f"Dataset configuration file not found: {data_yaml_path}\n"
            f"Please create a data.yaml file with YOLO format specifications."
        )

    # Parse the YAML file
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise TrainingError(f"Error parsing data.yaml file: {e}") from e

    # Validate required keys
    required_keys = ["train", "val", "nc", "names"]
    missing_keys = [key for key in required_keys if key not in data_config]
    if missing_keys:
        raise TrainingError(
            f"Missing required keys in data.yaml: {missing_keys}\n"
            f"Required keys: {required_keys}"
        )

    # Get dataset root directory (parent of data.yaml)
    dataset_root = data_yaml_path.parent

    # Validate train and val paths
    for split in ["train", "val"]:
        split_path = dataset_root / data_config[split]
        if not split_path.exists():
            raise TrainingError(
                f"Dataset {split} directory not found: {split_path}\n"
                f"Expected structure: {split}/images/ and {split}/labels/"
            )

        # Check for images and labels subdirectories
        # If images subdirectory exists, labels subdirectory must also exist
        has_images_subdir = (split_path / "images").exists()
        has_labels_subdir = (split_path / "labels").exists()

        if has_images_subdir:
            images_dir = split_path / "images"
            labels_dir = split_path / "labels"

            if not has_labels_subdir:
                raise TrainingError(f"Labels directory not found: {labels_dir}")
        else:
            # Images and labels are directly in split_path
            images_dir = split_path
            labels_dir = split_path

        if not images_dir.exists():
            raise TrainingError(  # pragma: no cover
                f"Images directory not found: {images_dir}"
            )

        if not labels_dir.exists():
            raise TrainingError(  # pragma: no cover
                f"Labels directory not found: {labels_dir}"
            )

        # Check if there are any files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            raise TrainingError(
                f"No image files found in {images_dir}\nSupported formats: .jpg, .png"
            )

        logger.info("Found %d images in %s set", len(image_files), split)

    # Validate number of classes
    num_classes = data_config["nc"]
    class_names = data_config["names"]

    if not isinstance(num_classes, int) or num_classes <= 0:
        raise TrainingError(f"Invalid number of classes: {num_classes}")

    if len(class_names) != num_classes:
        raise TrainingError(
            f"Mismatch between nc ({num_classes}) and number of class names ({len(class_names)})"
        )

    logger.info(
        "Dataset validated successfully: %d classes - %s", num_classes, class_names
    )
    return dict(data_config)


def validate_base_weights(base_weights_path: Path) -> None:
    """
    Validate that the base weights file exists and is accessible.

    Args:
        base_weights_path: Path to the base model weights file.

    Raises:
        TrainingError: If the weights file is invalid or not found.
    """
    if not base_weights_path.exists():
        raise TrainingError(
            f"Base weights file not found: {base_weights_path}\n"
            f"Please ensure the base model weights are available."
        )

    if base_weights_path.suffix != ".pt":
        raise TrainingError(
            f"Invalid weights file format: {base_weights_path.suffix}\n"
            f"Expected .pt file (PyTorch weights)"
        )

    logger.info("Base weights validated: %s", base_weights_path)


def setup_training_directories(model_name: str) -> dict[str, Path]:
    """
    Create necessary directories for training and model storage.

    Args:
        model_name: Name/version identifier for the model.

    Returns:
        Dict containing paths for model storage, logs, and training outputs.
    """
    project_root = get_project_root()

    # Create directories
    dirs = {
        "models": project_root / "models" / "hold_detection",
        "logs": project_root / "logs",
        "runs": project_root / "runs" / "detect" / model_name,
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: %s", dir_path)

    return dirs


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def train_yolov8(
    model_name: str,
    data_yaml: Path,
    base_weights: Path,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Train a YOLOv8 model on the specified dataset.

    Args:
        model_name: Name/version identifier for the model.
        data_yaml: Path to YOLO dataset configuration file.
        base_weights: Path to initial model weights.
        epochs: Number of training epochs.
        batch_size: Batch size for training (use -1 for auto-detect).
        img_size: Input image size.
        learning_rate: Initial learning rate.
        device: Device to use for training (e.g., "0", "cpu", "cuda:0").
            If None, automatically detects GPU availability:
            - Uses "0" if CUDA is available
            - Uses "cpu" otherwise

    Returns:
        Dict containing training results and metrics.

    Raises:
        TrainingError: If training fails.
    """
    # Determine device to use for training
    if device is None:
        import torch  # pylint: disable=import-outside-toplevel

        device = "0" if torch.cuda.is_available() else "cpu"
        logger.info("Auto-detected device: %s", device)

    logger.info("Starting YOLOv8 fine-tuning for model: %s", model_name)
    logger.info("Training configuration:")
    logger.info("  - Data: %s", data_yaml)
    logger.info("  - Base weights: %s", base_weights)
    logger.info("  - Epochs: %d", epochs)
    logger.info("  - Batch size: %d", batch_size)
    logger.info("  - Image size: %d", img_size)
    logger.info("  - Learning rate: %f", learning_rate)
    logger.info("  - Device: %s", device)

    try:
        # Import YOLO here to handle dynamic loading
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

        # Load the base model
        model = YOLO(str(base_weights))
        logger.info("Base model loaded successfully")

        # Configure training parameters
        training_args = {
            "data": str(data_yaml),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "lr0": learning_rate,
            "project": str(get_project_root() / "runs" / "detect"),
            "name": model_name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "Adam",
            "verbose": True,
            "save": True,
            "save_period": 10,  # Save checkpoint every 10 epochs
            "patience": 50,  # Early stopping patience
            "plots": True,  # Generate training plots
            "device": device,  # Use computed device
        }

        # Start training
        logger.info("Initiating training process...")
        results = model.train(**training_args)

        # Extract metrics from training results safely
        # Access attributes safely as ultralytics returns custom objects
        results_dict = getattr(results, "results_dict", {})
        metrics = {
            "final_mAP50": float(results_dict.get("metrics/mAP50(B)", 0.0)),
            "final_mAP50-95": float(results_dict.get("metrics/mAP50-95(B)", 0.0)),
            "final_precision": float(results_dict.get("metrics/precision(B)", 0.0)),
            "final_recall": float(results_dict.get("metrics/recall(B)", 0.0)),
            "best_epoch": int(getattr(results, "best_epoch", epochs)),
        }

        logger.info("Training completed successfully!")
        logger.info("Training metrics:")
        logger.info("  - mAP@0.5: %.4f", metrics["final_mAP50"])
        logger.info("  - mAP@0.5:0.95: %.4f", metrics["final_mAP50-95"])
        logger.info("  - Precision: %.4f", metrics["final_precision"])
        logger.info("  - Recall: %.4f", metrics["final_recall"])

        # Get save directory safely
        save_dir = getattr(
            results, "save_dir", get_project_root() / "runs" / "detect" / model_name
        )
        best_model_path = Path(save_dir) / "weights" / "best.pt"

        return {
            "results": results,
            "metrics": metrics,
            "best_model_path": best_model_path,
        }

    except Exception as e:
        raise TrainingError(f"Training failed: {str(e)}") from e


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def train_yolov8_classification(
    model_name: str,
    data_path: Path,
    base_weights: Path,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 224,
    learning_rate: float = 0.01,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Train a YOLOv8 classification model on the specified dataset.

    Uses YOLOv8's classification mode to train an image classifier for hold types.
    The dataset should be in folder-based format where each class has its own
    subdirectory containing images.

    Args:
        model_name: Name/version identifier for the model.
        data_path: Path to classification dataset root directory.
        base_weights: Path to initial model weights (e.g., yolov8n-cls.pt).
        epochs: Number of training epochs.
        batch_size: Batch size for training (use -1 for auto-detect).
        img_size: Input image size (default: 224 for classification).
        learning_rate: Initial learning rate.
        device: Device to use for training (e.g., "0", "cpu", "cuda:0").
            If None, automatically detects GPU availability.

    Returns:
        Dict containing training results and metrics.

    Raises:
        TrainingError: If training fails.
    """
    # Determine device to use for training
    if device is None:
        import torch  # pylint: disable=import-outside-toplevel

        device = "0" if torch.cuda.is_available() else "cpu"
        logger.info("Auto-detected device: %s", device)

    logger.info("Starting YOLOv8 classification training for model: %s", model_name)
    logger.info("Training configuration:")
    logger.info("  - Data path: %s", data_path)
    logger.info("  - Base weights: %s", base_weights)
    logger.info("  - Epochs: %d", epochs)
    logger.info("  - Batch size: %d", batch_size)
    logger.info("  - Image size: %d", img_size)
    logger.info("  - Learning rate: %f", learning_rate)
    logger.info("  - Device: %s", device)

    try:
        # Import YOLO here to handle dynamic loading
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

        # Load the base classification model
        model = YOLO(str(base_weights))
        logger.info("Base classification model loaded successfully")

        # Configure training parameters for classification
        training_args = {
            "data": str(data_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "lr0": learning_rate,
            "project": str(get_project_root() / "runs" / "classify"),
            "name": model_name,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "Adam",
            "verbose": True,
            "save": True,
            "save_period": 10,  # Save checkpoint every 10 epochs
            "patience": 50,  # Early stopping patience
            "plots": True,  # Generate training plots
            "device": device,  # Use computed device
        }

        # Start training
        logger.info("Initiating classification training process...")
        results = model.train(**training_args)

        # Extract metrics from training results safely
        # Classification uses different metrics than detection
        results_dict = getattr(results, "results_dict", {})
        metrics = {
            "top1_accuracy": float(results_dict.get("metrics/accuracy_top1", 0.0)),
            "top5_accuracy": float(results_dict.get("metrics/accuracy_top5", 0.0)),
            "best_epoch": int(getattr(results, "best_epoch", epochs)),
        }

        logger.info("Classification training completed successfully!")
        logger.info("Training metrics:")
        logger.info("  - Top-1 Accuracy: %.4f", metrics["top1_accuracy"])
        logger.info("  - Top-5 Accuracy: %.4f", metrics["top5_accuracy"])

        # Get save directory safely
        save_dir = getattr(
            results, "save_dir", get_project_root() / "runs" / "classify" / model_name
        )
        best_model_path = Path(save_dir) / "weights" / "best.pt"

        return {
            "results": results,
            "metrics": metrics,
            "best_model_path": best_model_path,
        }

    except Exception as e:
        raise TrainingError(f"Classification training failed: {str(e)}") from e


def setup_classification_directories(model_name: str) -> dict[str, Path]:
    """
    Create necessary directories for classification training and model storage.

    Args:
        model_name: Name/version identifier for the model.

    Returns:
        Dict containing paths for model storage, logs, and training outputs.
    """
    project_root = get_project_root()

    # Create directories
    dirs = {
        "models": project_root / "models" / "hold_classification",
        "logs": project_root / "logs",
        "runs": project_root / "runs" / "classify" / model_name,
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: %s", dir_path)

    return dirs


def save_model_version(
    model_name: str,
    trained_model_path: Path,
    metrics: dict[str, float],
    training_config: dict[str, Any],
    activate: bool = False,
    model_type: str = MODEL_TYPE_HOLD_DETECTION,
) -> ModelVersion:
    """
    Save the trained model and create a database entry for version tracking.

    Args:
        model_name: Name/version identifier for the model.
        trained_model_path: Path to the trained model weights.
        metrics: Dictionary containing training metrics.
        training_config: Dictionary containing training hyperparameters.
        activate: Whether to immediately activate this model version.
        model_type: Type of model being saved ('hold_detection' or 'hold_classification').

    Returns:
        ModelVersion: The created database record.

    Raises:
        TrainingError: If model saving or database operations fail.
    """
    logger.info("Saving model version: %s (type: %s)", model_name, model_type)

    app = None
    try:
        # Setup model storage directory based on model type
        model_storage_dir = get_project_root() / "models" / model_type
        model_storage_dir.mkdir(parents=True, exist_ok=True)

        # Copy trained model to storage location
        final_model_path = model_storage_dir / f"{model_name}.pt"
        shutil.copy2(trained_model_path, final_model_path)
        logger.info("Model saved to: %s", final_model_path)

        # Save training metadata
        metadata_path = model_storage_dir / f"{model_name}_metadata.yaml"
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "training_config": training_config,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        logger.info("Metadata saved to: %s", metadata_path)

        # Create database entry
        app = create_flask_app()
        with app.app_context():
            # Check if version already exists
            existing_version = ModelVersion.query.filter_by(
                model_type=model_type, version=model_name
            ).first()

            # Determine accuracy metric based on model type
            if model_type == MODEL_TYPE_HOLD_CLASSIFICATION:
                accuracy_value = metrics.get("top1_accuracy", 0.0)
            else:
                accuracy_value = metrics.get("final_mAP50-95", 0.0)

            if existing_version:
                logger.warning(
                    "Model version %s already exists. Updating existing record.",
                    model_name,
                )
                existing_version.model_path = str(final_model_path)
                existing_version.accuracy = accuracy_value
                existing_version.created_at = datetime.now(timezone.utc)

                # Handle activation state
                if activate:
                    # Deactivate other versions of same model type if this one should be active
                    ModelVersion.query.filter_by(
                        model_type=model_type, is_active=True
                    ).update({"is_active": False})
                    logger.info("Deactivated previous active %s models", model_type)
                    existing_version.is_active = True
                else:
                    existing_version.is_active = False

                model_version = existing_version
            else:
                # Deactivate other versions of same model type if this one should be active
                if activate:
                    ModelVersion.query.filter_by(
                        model_type=model_type, is_active=True
                    ).update({"is_active": False})
                    logger.info("Deactivated previous active %s models", model_type)

                # Create new model version entry
                # SQLAlchemy accepts column names as kwargs
                model_version = ModelVersion(
                    model_type=model_type,
                    version=model_name,
                    model_path=str(final_model_path),
                    accuracy=accuracy_value,
                    is_active=activate,
                )
                db.session.add(model_version)

            db.session.commit()
            logger.info("Database entry created: %s", model_version.to_dict())

        return model_version  # type: ignore[no-any-return]

    except Exception as e:
        raise TrainingError(f"Failed to save model version: {str(e)}") from e
    finally:
        # Dispose of database engine to close connections
        if app:
            try:
                db.engine.dispose()
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Ignore errors during cleanup


def create_flask_app() -> Flask:
    """
    Create a Flask application instance for database operations.

    Returns:
        Flask: Configured Flask application.
    """
    import os  # pylint: disable=import-outside-toplevel

    app = Flask(__name__)

    # Configure database - try environment variable, then config, then default
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        try:
            database_url = get_config_value("database.url")
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    if not database_url:
        database_url = "sqlite:///bouldering_analysis.db"

    # For SQLite relative paths, resolve against project root
    if database_url.startswith("sqlite:///") and not database_url.startswith(
        "sqlite:////"
    ):
        # Extract the path after sqlite:///
        db_path = database_url[10:]  # Remove "sqlite:///"
        path_obj = Path(db_path)

        # If path is relative, resolve against project root
        if not path_obj.is_absolute():
            resolved_path = get_project_root() / db_path
            database_url = "sqlite:///" + str(resolved_path)

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Initialize database
    db.init_app(app)

    return app


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements,too-many-branches
def main(
    model_name: str,
    model_type: str = MODEL_TYPE_HOLD_DETECTION,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    data_yaml: Optional[str] = None,
    data_path: Optional[str] = None,
    base_weights: Optional[str] = None,
    img_size: Optional[int] = None,
    learning_rate: float = 0.01,
    activate: bool = False,
) -> None:
    """
    Main training pipeline function supporting both detection and classification.

    Args:
        model_name: Name/version identifier for the new model (e.g., "v1.0").
        model_type: Type of model to train ('hold_detection' or 'hold_classification').
        epochs: Number of training epochs (default from config or 100).
        batch_size: Batch size for training (default from config or 16).
        data_yaml: Path to YOLO dataset configuration file (for detection).
        data_path: Path to classification dataset root directory (for classification).
        base_weights: Path to initial weights.
        img_size: Input image size (default: 640 for detection, 224 for classification).
        learning_rate: Initial learning rate (default: 0.01).
        activate: Whether to immediately activate the model after training.

    Raises:
        TrainingError: If any step of the training pipeline fails.
    """
    # Validate model type
    if model_type not in VALID_MODEL_TYPES:
        raise TrainingError(
            f"Invalid model type: {model_type}. Valid types: {VALID_MODEL_TYPES}"
        )

    is_classification = model_type == MODEL_TYPE_HOLD_CLASSIFICATION

    try:
        logger.info("=" * 80)
        logger.info(
            "YOLOv8 %s Pipeline - Model: %s",
            "Classification" if is_classification else "Detection",
            model_name,
        )
        logger.info("=" * 80)

        # Load configuration and set defaults based on model type
        try:
            if epochs is None:
                epochs = get_config_value("training.epochs", 100)
            if batch_size is None:
                batch_size = get_config_value(  # pragma: no cover
                    "training.batch_size", 16
                )

            if is_classification:
                # Classification-specific defaults
                if img_size is None:
                    img_size = get_config_value("training.classification_img_size", 224)
                if data_path is None:
                    data_path = str(  # pragma: no cover
                        get_data_path("hold_classification_dataset")
                    )
                if base_weights is None:
                    base_weights = get_config_value(  # pragma: no cover
                        "model_paths.base_yolov8_cls", "yolov8n-cls.pt"
                    )
            else:
                # Detection-specific defaults
                if img_size is None:
                    img_size = 640
                if data_yaml is None:
                    data_yaml = str(  # pragma: no cover
                        get_data_path("hold_dataset") / "data.yaml"
                    )
                if base_weights is None:
                    base_weights = str(
                        get_model_path("base_yolov8")
                    )  # pragma: no cover

        except ConfigurationError as e:
            logger.warning("Configuration loading warning: %s", e)
            # Use hardcoded defaults if config fails
            epochs = epochs or 100
            batch_size = batch_size or 16
            if is_classification:
                img_size = img_size or 224
                data_path = data_path or "data/hold_classification/"
                base_weights = base_weights or "yolov8n-cls.pt"
            else:
                img_size = img_size or 640
                data_yaml = data_yaml or "data/sample_hold/data.yaml"
                base_weights = base_weights or "yolov8n.pt"

        # Resolve paths
        base_weights_path = resolve_path(base_weights)

        # Ensure epochs, batch_size, and img_size are not None at this point
        final_epochs = epochs if epochs is not None else 100
        final_batch_size = batch_size if batch_size is not None else 16
        final_img_size = (
            img_size if img_size is not None else (224 if is_classification else 640)
        )

        if is_classification:
            # Classification training pipeline
            resolved_data_path = resolve_path(data_path) if data_path else None
            if resolved_data_path is None:
                raise TrainingError(
                    "Data path is required for classification training. "
                    "Use --data-path to specify the dataset directory."
                )

            # Step 1: Validate classification dataset
            logger.info("\n[Step 1/5] Validating classification dataset...")
            dataset_config = validate_classification_dataset(resolved_data_path)

            # Step 2: Validate base weights
            logger.info("\n[Step 2/5] Validating base weights...")
            validate_base_weights(base_weights_path)

            # Step 3: Setup directories
            logger.info("\n[Step 3/5] Setting up training directories...")
            setup_classification_directories(model_name)

            # Step 4: Train classification model
            logger.info("\n[Step 4/5] Training YOLOv8 classification model...")
            training_results = train_yolov8_classification(
                model_name=model_name,
                data_path=resolved_data_path,
                base_weights=base_weights_path,
                epochs=final_epochs,
                batch_size=final_batch_size,
                img_size=final_img_size,
                learning_rate=learning_rate,
            )

            # Step 5: Save model version
            logger.info("\n[Step 5/5] Saving model version and metadata...")
            training_config = {
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": final_img_size,
                "learning_rate": learning_rate,
                "base_weights": str(base_weights_path),
                "data_path": str(resolved_data_path),
                "num_classes": dataset_config["nc"],
                "class_names": dataset_config["names"],
            }

            model_version = save_model_version(
                model_name=model_name,
                trained_model_path=training_results["best_model_path"],
                metrics=training_results["metrics"],
                training_config=training_config,
                activate=activate,
                model_type=model_type,
            )

            # Training complete
            logger.info("=" * 80)
            logger.info("Classification Training Pipeline Completed Successfully!")
            logger.info("=" * 80)
            logger.info("Model Name: %s", model_name)
            logger.info("Model Type: %s", model_type)
            logger.info("Model Path: %s", model_version.model_path)
            logger.info("Top-1 Accuracy: %.4f", model_version.accuracy)
            logger.info("Active: %s", model_version.is_active)
            logger.info(
                "Training results saved to: %s", training_results["results"].save_dir
            )
            logger.info("=" * 80)

        else:
            # Detection training pipeline (original behavior)
            data_yaml_path = resolve_path(data_yaml) if data_yaml else None
            if data_yaml_path is None:
                raise TrainingError(
                    "Data YAML is required for detection training. "
                    "Use --data-yaml to specify the dataset configuration file."
                )

            # Step 1: Validate dataset
            logger.info("\n[Step 1/5] Validating dataset...")
            dataset_config = validate_dataset(data_yaml_path)

            # Step 2: Validate base weights
            logger.info("\n[Step 2/5] Validating base weights...")
            validate_base_weights(base_weights_path)

            # Step 3: Setup directories
            logger.info("\n[Step 3/5] Setting up training directories...")
            setup_training_directories(model_name)

            # Step 4: Train model
            logger.info("\n[Step 4/5] Training YOLOv8 model...")
            training_results = train_yolov8(
                model_name=model_name,
                data_yaml=data_yaml_path,
                base_weights=base_weights_path,
                epochs=final_epochs,
                batch_size=final_batch_size,
                img_size=final_img_size,
                learning_rate=learning_rate,
            )

            # Step 5: Save model version
            logger.info("\n[Step 5/5] Saving model version and metadata...")
            training_config = {
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": final_img_size,
                "learning_rate": learning_rate,
                "base_weights": str(base_weights_path),
                "data_yaml": str(data_yaml_path),
                "num_classes": dataset_config["nc"],
                "class_names": dataset_config["names"],
            }

            model_version = save_model_version(
                model_name=model_name,
                trained_model_path=training_results["best_model_path"],
                metrics=training_results["metrics"],
                training_config=training_config,
                activate=activate,
                model_type=model_type,
            )

            # Training complete
            logger.info("=" * 80)
            logger.info("Detection Training Pipeline Completed Successfully!")
            logger.info("=" * 80)
            logger.info("Model Name: %s", model_name)
            logger.info("Model Type: %s", model_type)
            logger.info("Model Path: %s", model_version.model_path)
            logger.info("mAP@0.5:0.95: %.4f", model_version.accuracy)
            logger.info("Active: %s", model_version.is_active)
            logger.info(
                "Training results saved to: %s", training_results["results"].save_dir
            )
            logger.info("=" * 80)

        if not activate:
            logger.info("\nNote: Model is not activated. To activate this model, use:")
            logger.info(
                "  UPDATE model_versions SET is_active=1 WHERE version='%s';",
                model_name,
            )

    except TrainingError as e:
        logger.error("Training pipeline failed: %s", e)  # pragma: no cover
        sys.exit(1)  # pragma: no cover
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected error during training: %s", e, exc_info=True)
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="YOLOv8 Fine-tuning Pipeline for Bouldering Hold Detection and Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic hold detection training with default parameters
  python src/train_model.py --model-name v1.0

  # Hold classification training
  python src/train_model.py --model-name v1.0 \\
      --model-type hold_classification \\
      --data-path data/hold_classification/

  # Custom detection training configuration
  python src/train_model.py --model-name v1.1 --epochs 100 --batch-size 16

  # Training with automatic activation
  python src/train_model.py --model-name v2.0 --epochs 150 --activate

  # Using custom detection dataset
  python src/train_model.py --model-name custom_v1 \\
      --data-yaml data/custom/data.yaml --base-weights yolov8n.pt

  # Classification with custom weights
  python src/train_model.py --model-name cls_v1 \\
      --model-type hold_classification \\
      --data-path data/holds/ \\
      --base-weights yolov8n-cls.pt \\
      --img-size 224
        """,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name/version identifier for the model (e.g., 'v1.0', 'v2.0-exp1')",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default=MODEL_TYPE_HOLD_DETECTION,
        choices=VALID_MODEL_TYPES,
        help=(
            f"Type of model to train. Options: {VALID_MODEL_TYPES}. "
            f"Default: {MODEL_TYPE_HOLD_DETECTION}"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config or 100)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: from config or 16, use -1 for auto)",
    )

    parser.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Path to YOLO dataset configuration file for detection (default: from config)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to classification dataset directory for hold_classification. "
            "Expected structure: data_path/train/<class>/ and data_path/val/<class>/"
        ),
    )

    parser.add_argument(
        "--base-weights",
        type=str,
        default=None,
        help=(
            "Path to initial model weights. "
            "Default: 'yolov8n.pt' for detection, 'yolov8n-cls.pt' for classification"
        ),
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input image size for training (default: 640 for detection, 224 for classification)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--activate",
        action="store_true",
        help="Immediately activate the model after training (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(
        model_name=args.model_name,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_yaml=args.data_yaml,
        data_path=args.data_path,
        base_weights=args.base_weights,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        activate=args.activate,
    )
