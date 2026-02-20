"""Training module for hold detection and classification models.

This package provides dataset loading, model training, and related utilities
for the bouldering route analysis perception models.

Modules:
    datasets: Dataset loading and validation for YOLOv8 format
    detection_model: YOLOv8 model definition and hyperparameters
    exceptions: Custom exception classes for training errors
    train_detection: Training loop and artifact management
"""

from src.training.datasets import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_CLASSES,
    count_dataset_images,
    load_hold_detection_dataset,
    validate_data_yaml,
    validate_directory_structure,
)
from src.training.detection_model import (
    DEFAULT_MODEL_SIZE,
    INPUT_RESOLUTION,
    VALID_MODEL_SIZES,
    DetectionHyperparameters,
    build_hold_detector,
    get_default_hyperparameters,
    load_hyperparameters_from_file,
)
from src.training.exceptions import (
    ClassTaxonomyError,
    DatasetNotFoundError,
    DatasetValidationError,
    ModelArtifactError,
    TrainingError,
    TrainingRunError,
)
from src.training.train_detection import (
    TrainingMetrics,
    TrainingResult,
    train_hold_detector,
)

__all__ = [
    # Dataset functions
    "load_hold_detection_dataset",
    "validate_data_yaml",
    "validate_directory_structure",
    "count_dataset_images",
    # Dataset constants
    "EXPECTED_CLASSES",
    "EXPECTED_CLASS_COUNT",
    # Model building
    "build_hold_detector",
    "get_default_hyperparameters",
    "load_hyperparameters_from_file",
    # Model configuration
    "DetectionHyperparameters",
    "DEFAULT_MODEL_SIZE",
    "INPUT_RESOLUTION",
    "VALID_MODEL_SIZES",
    # Training loop
    "train_hold_detector",
    "TrainingResult",
    "TrainingMetrics",
    # Exceptions
    "TrainingError",
    "DatasetNotFoundError",
    "DatasetValidationError",
    "ClassTaxonomyError",
    "TrainingRunError",
    "ModelArtifactError",
]
