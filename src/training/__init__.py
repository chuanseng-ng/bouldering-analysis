"""Training module for hold detection and classification models.

This package provides dataset loading, model training, and related utilities
for the bouldering route analysis perception models.

Modules:
    classification_dataset: Dataset loading for folder-per-class format
    classification_model: ResNet-18/MobileNetV3 model definition and hyperparameters
    datasets: Dataset loading and validation for YOLOv8 format
    detection_model: YOLOv8 model definition and hyperparameters
    exceptions: Custom exception classes for training errors
    train_detection: Training loop and artifact management
"""

from src.training.classification_dataset import (
    HOLD_CLASS_COUNT,
    HOLD_CLASSES,
    IMAGE_EXTENSIONS,
    ClassificationDatasetConfig,
    compute_class_weights,
    count_images_per_class,
    load_hold_classification_dataset,
    validate_classification_structure,
)
from src.training.classification_model import (
    DEFAULT_ARCHITECTURE,
    INPUT_SIZE,
    VALID_ARCHITECTURES,
    ClassifierConfig,
    ClassifierHyperparameters,
    build_hold_classifier,
    get_default_hyperparameters as get_default_classifier_hyperparameters,
    load_hyperparameters_from_file as load_classifier_hyperparameters_from_file,
)
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
    # Classification dataset
    "ClassificationDatasetConfig",
    "load_hold_classification_dataset",
    "validate_classification_structure",
    "compute_class_weights",
    "count_images_per_class",
    "HOLD_CLASSES",
    "HOLD_CLASS_COUNT",
    "IMAGE_EXTENSIONS",
    # Detection dataset functions
    "load_hold_detection_dataset",
    "validate_data_yaml",
    "validate_directory_structure",
    "count_dataset_images",
    # Detection dataset constants
    "EXPECTED_CLASSES",
    "EXPECTED_CLASS_COUNT",
    # Classification model building
    "build_hold_classifier",
    "get_default_classifier_hyperparameters",
    "load_classifier_hyperparameters_from_file",
    # Classification model configuration
    "ClassifierConfig",
    "ClassifierHyperparameters",
    "DEFAULT_ARCHITECTURE",
    "INPUT_SIZE",
    "VALID_ARCHITECTURES",
    # Detection model building
    # NOTE: use get_default_classifier_hyperparameters for the classifier;
    # get_default_hyperparameters refers to the detector (DetectionHyperparameters).
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
