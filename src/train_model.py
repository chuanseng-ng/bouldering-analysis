# pylint: disable=duplicate-code
"""
YOLOv8 Fine-tuning Pipeline for Bouldering Hold Detection.

This script provides a comprehensive training pipeline for fine-tuning YOLOv8 models
on custom bouldering hold datasets. It includes dataset validation, model training,
versioning, and database integration.

Usage Examples:
    Basic training with default parameters:
        python src/train_model.py --model-name v1.0

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

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import yaml
from flask import Flask

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from src.config import (
    get_project_root,
    resolve_path,
    get_config_value,
    get_model_path,
    get_data_path,
    ConfigurationError,
)
from src.models import db, ModelVersion

# pylint: enable=wrong-import-position


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Raised when there are issues during the training process."""


def validate_dataset(data_yaml_path: Path) -> Dict[str, Any]:
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
        images_dir = (
            split_path / "images" if (split_path / "images").exists() else split_path
        )
        labels_dir = (
            split_path / "labels" if (split_path / "labels").exists() else split_path
        )

        if not images_dir.exists():
            raise TrainingError(f"Images directory not found: {images_dir}")

        if not labels_dir.exists():
            raise TrainingError(f"Labels directory not found: {labels_dir}")

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


def setup_training_directories(model_name: str) -> Dict[str, Path]:
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
) -> Dict[str, Any]:
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

    Returns:
        Dict containing training results and metrics.

    Raises:
        TrainingError: If training fails.
    """
    logger.info("Starting YOLOv8 fine-tuning for model: %s", model_name)
    logger.info("Training configuration:")
    logger.info("  - Data: %s", data_yaml)
    logger.info("  - Base weights: %s", base_weights)
    logger.info("  - Epochs: %d", epochs)
    logger.info("  - Batch size: %d", batch_size)
    logger.info("  - Image size: %d", img_size)
    logger.info("  - Learning rate: %f", learning_rate)

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
            "device": "0",  # Use GPU if available, else CPU
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


def save_model_version(
    model_name: str,
    trained_model_path: Path,
    metrics: Dict[str, float],
    training_config: Dict[str, Any],
    activate: bool = False,
) -> ModelVersion:
    """
    Save the trained model and create a database entry for version tracking.

    Args:
        model_name: Name/version identifier for the model.
        trained_model_path: Path to the trained model weights.
        metrics: Dictionary containing training metrics.
        training_config: Dictionary containing training hyperparameters.
        activate: Whether to immediately activate this model version.

    Returns:
        ModelVersion: The created database record.

    Raises:
        TrainingError: If model saving or database operations fail.
    """
    logger.info("Saving model version: %s", model_name)

    try:
        # Setup model storage directory
        model_storage_dir = get_project_root() / "models" / "hold_detection"
        model_storage_dir.mkdir(parents=True, exist_ok=True)

        # Copy trained model to storage location
        final_model_path = model_storage_dir / f"{model_name}.pt"
        shutil.copy2(trained_model_path, final_model_path)
        logger.info("Model saved to: %s", final_model_path)

        # Save training metadata
        metadata_path = model_storage_dir / f"{model_name}_metadata.yaml"
        metadata = {
            "model_name": model_name,
            "model_type": "hold_detection",
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
                model_type="hold_detection", version=model_name
            ).first()

            if existing_version:
                logger.warning(
                    "Model version %s already exists. Updating existing record.",
                    model_name,
                )
                existing_version.model_path = str(final_model_path)
                existing_version.accuracy = metrics.get("final_mAP50-95", 0.0)
                existing_version.created_at = datetime.now(timezone.utc)

                # Handle activation state
                if activate:
                    # Deactivate other versions if this one should be active
                    ModelVersion.query.filter_by(
                        model_type="hold_detection", is_active=True
                    ).update({"is_active": False})
                    logger.info("Deactivated previous active models")
                    existing_version.is_active = True
                else:
                    existing_version.is_active = False

                model_version = existing_version
            else:
                # Deactivate other versions if this one should be active
                if activate:
                    ModelVersion.query.filter_by(
                        model_type="hold_detection", is_active=True
                    ).update({"is_active": False})
                    logger.info("Deactivated previous active models")

                # Create new model version entry
                # SQLAlchemy accepts column names as kwargs
                model_version = ModelVersion(
                    model_type="hold_detection",
                    version=model_name,
                    model_path=str(final_model_path),
                    accuracy=metrics.get("final_mAP50-95", 0.0),
                    is_active=activate,
                )
                db.session.add(model_version)

            db.session.commit()
            logger.info("Database entry created: %s", model_version.to_dict())

        return model_version  # type: ignore[no-any-return]

    except Exception as e:
        raise TrainingError(f"Failed to save model version: {str(e)}") from e


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
        except (
            ConfigurationError,
            Exception,
        ):  # pylint: disable=broad-exception-caught
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
def main(
    model_name: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    data_yaml: Optional[str] = None,
    base_weights: Optional[str] = None,
    img_size: int = 640,
    learning_rate: float = 0.01,
    activate: bool = False,
) -> None:
    """
    Main training pipeline function.

    Args:
        model_name: Name/version identifier for the new model (e.g., "v1.0").
        epochs: Number of training epochs (default from config or 100).
        batch_size: Batch size for training (default from config or 16).
        data_yaml: Path to YOLO dataset configuration file.
        base_weights: Path to initial weights.
        img_size: Input image size (default: 640).
        learning_rate: Initial learning rate (default: 0.01).
        activate: Whether to immediately activate the model after training.

    Raises:
        TrainingError: If any step of the training pipeline fails.
    """
    try:
        logger.info("=" * 80)
        logger.info("YOLOv8 Fine-tuning Pipeline - Model: %s", model_name)
        logger.info("=" * 80)

        # Load configuration and set defaults
        try:
            if epochs is None:
                epochs = get_config_value("training.epochs", 100)
            if batch_size is None:
                batch_size = get_config_value("training.batch_size", 16)
            if data_yaml is None:
                data_yaml = str(get_data_path("hold_dataset") / "data.yaml")
            if base_weights is None:
                base_weights = str(get_model_path("base_yolov8"))
        except ConfigurationError as e:
            logger.warning("Configuration loading warning: %s", e)
            # Use hardcoded defaults if config fails
            epochs = epochs or 100
            batch_size = batch_size or 16
            data_yaml = data_yaml or "data/sample_hold/data.yaml"
            base_weights = base_weights or "yolov8n.pt"

        # Resolve paths
        data_yaml_path = resolve_path(data_yaml)
        base_weights_path = resolve_path(base_weights)

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
        # Ensure epochs and batch_size are not None at this point
        final_epochs = epochs if epochs is not None else 100
        final_batch_size = batch_size if batch_size is not None else 16

        training_results = train_yolov8(
            model_name=model_name,
            data_yaml=data_yaml_path,
            base_weights=base_weights_path,
            epochs=final_epochs,
            batch_size=final_batch_size,
            img_size=img_size,
            learning_rate=learning_rate,
        )

        # Step 5: Save model version
        logger.info("\n[Step 5/5] Saving model version and metadata...")
        training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
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
        )

        # Training complete
        logger.info("=" * 80)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info("Model Name: %s", model_name)
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
        logger.error("Training pipeline failed: %s", e)
        sys.exit(1)
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
        description="YOLOv8 Fine-tuning Pipeline for Bouldering Hold Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default parameters
  python src/train_model.py --model-name v1.0

  # Custom training configuration
  python src/train_model.py --model-name v1.1 --epochs 100 --batch-size 16

  # Training with automatic activation
  python src/train_model.py --model-name v2.0 --epochs 150 --activate

  # Using custom dataset
  python src/train_model.py --model-name custom_v1 \\
      --data-yaml data/custom/data.yaml --base-weights yolov8n.pt
        """,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name/version identifier for the model (e.g., 'v1.0', 'v2.0-exp1')",
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
        help="Path to YOLO dataset configuration file (default: from config)",
    )

    parser.add_argument(
        "--base-weights",
        type=str,
        default=None,
        help="Path to initial model weights (default: from config or 'yolov8n.pt')",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size for training (default: 640)",
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_yaml=args.data_yaml,
        base_weights=args.base_weights,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        activate=args.activate,
    )
