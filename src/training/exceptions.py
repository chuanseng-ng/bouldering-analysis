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
