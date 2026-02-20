"""Custom exceptions for the training module.

This module defines exception classes for dataset loading and validation errors
that may occur during hold detection model training.
"""


class TrainingError(Exception):
    """Base exception for training module errors.

    All training-related exceptions inherit from this class to allow
    for broad exception catching when needed.

    Attributes:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str) -> None:
        """Initialize TrainingError with a message.

        Args:
            message: Description of the error that occurred.
        """
        self.message = message
        super().__init__(self.message)


class DatasetNotFoundError(TrainingError):
    """Raised when a dataset or required files cannot be found.

    This exception is raised when:
    - The dataset root directory does not exist
    - The data.yaml configuration file is missing
    - Required subdirectories are missing

    Example:
        >>> raise DatasetNotFoundError("Dataset not found at: /path/to/dataset")
    """


class DatasetValidationError(TrainingError):
    """Raised when dataset structure or format is invalid.

    This exception is raised when:
    - data.yaml has invalid YAML syntax
    - Required keys are missing from data.yaml
    - Directory structure doesn't match expected format
    - Label files have invalid format

    Example:
        >>> raise DatasetValidationError("Missing required key: 'train'")
    """


class ClassTaxonomyError(TrainingError):
    """Raised when class configuration doesn't match expected taxonomy.

    For hold detection, we expect exactly 2 classes:
    - Class 0: hold
    - Class 1: volume

    This exception is raised when:
    - Number of classes (nc) is not 2
    - Class names don't match ['hold', 'volume']
    - Class IDs are not correctly mapped

    Example:
        >>> raise ClassTaxonomyError("Expected 2 classes, found 5")
    """


class TrainingRunError(TrainingError):
    """Raised when YOLO model.train() itself fails at runtime.

    This exception wraps errors that occur during the actual training
    execution, such as CUDA out-of-memory errors, data loading failures,
    or unexpected exceptions from the Ultralytics training loop.

    Example:
        >>> raise TrainingRunError("YOLO training failed: CUDA out of memory")
    """


class ModelArtifactError(TrainingError):
    """Raised when saving weights/metadata.json fails after training.

    This exception is raised when post-training artifact saving fails,
    such as when the output directory cannot be created or weight files
    cannot be copied.

    Example:
        >>> raise ModelArtifactError("Failed to copy best.pt to output dir")
    """


class InferenceError(Exception):
    """Raised when hold detection inference fails.

    This is a sibling of TrainingError (not a subclass) as it represents
    a separate operational context: real-time inference rather than training.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> raise InferenceError("Model weights not found: /path/to/model.pt")
    """

    def __init__(self, message: str) -> None:
        """Initialize InferenceError with a message.

        Args:
            message: Description of the inference error that occurred.
        """
        self.message = message
        super().__init__(self.message)
