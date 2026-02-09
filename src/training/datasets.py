"""Dataset loading and validation for hold detection training.

This module provides functions to load and validate YOLOv8 format datasets
for training hold/volume detection models. It supports Roboflow exports
and enforces the simplified two-class taxonomy (hold, volume).

The main function `load_hold_detection_dataset()` validates the dataset
structure and returns a configuration dictionary ready for training.

Example:
    >>> from src.training.datasets import load_hold_detection_dataset
    >>> config = load_hold_detection_dataset("data/climbing_holds")
    >>> print(config["train_image_count"])
    500
"""

import warnings
from pathlib import Path
from typing import Any

import yaml

from src.logging_config import get_logger
from src.training.exceptions import (
    ClassTaxonomyError,
    DatasetNotFoundError,
    DatasetValidationError,
)

logger = get_logger(__name__)

# Expected class configuration for hold detection
EXPECTED_CLASSES = ["hold", "volume"]
EXPECTED_CLASS_COUNT = 2

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Required keys in data.yaml
REQUIRED_YAML_KEYS = ["train", "val", "nc", "names"]


def load_hold_detection_dataset(
    dataset_root: Path | str,
    strict: bool = True,
) -> dict[str, Any]:
    """Load and validate a hold detection dataset in YOLOv8 format.

    This function loads datasets exported from Roboflow or other sources
    in YOLOv8 object detection format. It validates the directory structure,
    configuration file, and class taxonomy.

    Args:
        dataset_root: Path to the dataset directory containing data.yaml.
        strict: If True, raise errors for validation failures.
            If False, log warnings and continue where possible.

    Returns:
        Dictionary containing:
            - train: Absolute path to training directory
            - val: Absolute path to validation directory
            - test: Absolute path to test directory (or None)
            - nc: Number of classes (always 2)
            - names: List of class names ["hold", "volume"]
            - train_image_count: Number of training images
            - val_image_count: Number of validation images
            - test_image_count: Number of test images (or 0)
            - version: Dataset version string (if available)
            - metadata: Additional metadata from data.yaml

    Raises:
        DatasetNotFoundError: If dataset_root or data.yaml doesn't exist.
        DatasetValidationError: If dataset structure or config is invalid.
        ClassTaxonomyError: If classes don't match expected [hold, volume].

    Example:
        >>> config = load_hold_detection_dataset("data/climbing_holds")
        >>> print(config["train_image_count"])
        500
        >>> print(config["names"])
        ['hold', 'volume']
    """
    # Convert to Path if string
    root = Path(dataset_root).resolve()

    logger.info("Loading hold detection dataset from: %s", root)

    # Check if dataset root exists
    if not root.exists():
        raise DatasetNotFoundError(f"Dataset not found at: {root}")

    if not root.is_dir():
        raise DatasetNotFoundError(f"Dataset path is not a directory: {root}")

    # Validate and parse data.yaml
    yaml_path = root / "data.yaml"
    config = validate_data_yaml(yaml_path, strict=strict)

    # Validate directory structure
    validate_directory_structure(root, config, strict=strict)

    # Build result dictionary
    train_path = root / config["train"]
    val_path = root / config["val"]
    test_path = root / config["test"] if config.get("test") else None

    # Count images in each split
    train_count = count_dataset_images(train_path)
    val_count = count_dataset_images(val_path)
    test_count = count_dataset_images(test_path) if test_path else 0

    logger.info(
        "Dataset loaded: train=%d, val=%d, test=%d images",
        train_count,
        val_count,
        test_count,
    )

    # Extract metadata (optional fields from data.yaml)
    metadata = {
        k: v
        for k, v in config.items()
        if k not in REQUIRED_YAML_KEYS and k not in ["test"]
    }

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": EXPECTED_CLASS_COUNT,
        "names": EXPECTED_CLASSES.copy(),
        "train_image_count": train_count,
        "val_image_count": val_count,
        "test_image_count": test_count,
        "version": config.get("dataset_version"),
        "metadata": metadata,
    }


def validate_data_yaml(yaml_path: Path, strict: bool = True) -> dict[str, Any]:
    """Validate and parse data.yaml configuration file.

    Args:
        yaml_path: Path to data.yaml file.
        strict: If True, raise errors for validation failures.

    Returns:
        Parsed YAML content as dictionary.

    Raises:
        DatasetNotFoundError: If file doesn't exist.
        DatasetValidationError: If YAML is invalid or missing required keys.
        ClassTaxonomyError: If class configuration is incorrect.
    """
    logger.debug("Validating data.yaml at: %s", yaml_path)

    if not yaml_path.exists():
        raise DatasetNotFoundError(f"data.yaml not found in: {yaml_path.parent}")

    # Parse YAML file
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DatasetValidationError(f"Failed to parse data.yaml: {e}") from e

    if config is None:
        raise DatasetValidationError("data.yaml is empty")

    if not isinstance(config, dict):
        raise DatasetValidationError("data.yaml must contain a YAML mapping")

    # Check required keys
    for key in REQUIRED_YAML_KEYS:
        if key not in config:
            raise DatasetValidationError(f"Missing required key in data.yaml: '{key}'")

    # Validate class configuration
    _validate_class_taxonomy(config, _strict=strict)

    logger.debug("data.yaml validation passed")
    return config


def _validate_class_taxonomy(
    config: dict[str, Any],
    _strict: bool = True,  # noqa: ARG001 - Reserved for future non-strict mode
) -> None:
    """Validate that class configuration matches expected taxonomy.

    Args:
        config: Parsed data.yaml configuration.
        _strict: Reserved for future use. Currently always raises on mismatch.

    Raises:
        ClassTaxonomyError: If class configuration is incorrect.

    Note:
        The _strict parameter is currently unused but reserved for future
        non-strict validation mode where mismatches would emit warnings
        instead of raising exceptions.
    """
    nc = config.get("nc")
    names = config.get("names")

    # Validate number of classes
    if nc != EXPECTED_CLASS_COUNT:
        raise ClassTaxonomyError(f"Expected {EXPECTED_CLASS_COUNT} classes, found {nc}")

    # Convert names to list if it's a dict (YOLO format allows both)
    if isinstance(names, dict):
        # Convert dict like {0: "hold", 1: "volume"} to list
        try:
            names_list = [names[i] for i in range(len(names))]
        except KeyError as e:
            raise ClassTaxonomyError(
                f"Class names dict must have sequential integer keys starting from 0: {e}"
            ) from e
    elif isinstance(names, list):
        names_list = names
    else:
        raise ClassTaxonomyError(
            f"'names' must be a list or dict, got {type(names).__name__}"
        )

    # Validate class names match expected
    if len(names_list) != EXPECTED_CLASS_COUNT:
        raise ClassTaxonomyError(
            f"Expected {EXPECTED_CLASS_COUNT} class names, found {len(names_list)}"
        )

    if names_list != EXPECTED_CLASSES:
        raise ClassTaxonomyError(
            f"Expected classes {EXPECTED_CLASSES}, got {names_list}"
        )

    logger.debug("Class taxonomy validation passed: %s", names_list)


def validate_directory_structure(
    dataset_root: Path,
    config: dict[str, Any],
    strict: bool = True,
) -> None:
    """Validate that required directories exist.

    Args:
        dataset_root: Root directory of the dataset.
        config: Parsed data.yaml configuration.
        strict: If True, raise errors for missing directories.

    Raises:
        DatasetValidationError: If required directories are missing.
    """
    logger.debug("Validating directory structure")

    # Validate train directory
    train_path = dataset_root / config["train"]
    _validate_split_directory(train_path, "train", _strict=strict)

    # Validate val directory
    val_path = dataset_root / config["val"]
    _validate_split_directory(val_path, "val", _strict=strict)

    # Validate test directory (optional)
    if config.get("test"):
        test_path = dataset_root / config["test"]
        if test_path.exists():
            _validate_split_directory(test_path, "test", _strict=strict)
        else:
            logger.warning("Test directory specified but not found: %s", test_path)
            if strict:
                raise DatasetValidationError(f"Test directory not found: {test_path}")

    logger.debug("Directory structure validation passed")


def _validate_split_directory(
    split_path: Path,
    split_name: str,
    _strict: bool,
) -> None:
    """Validate a single split directory (train/val/test).

    Args:
        split_path: Path to the split directory.
        split_name: Name of the split for error messages.
        _strict: If True, raise errors for validation failures.
            If False, emit warnings instead.

    Raises:
        DatasetValidationError: If directory structure is invalid (strict mode only).
    """
    if not split_path.exists():
        msg = f"{split_name.capitalize()} directory not found: {split_path}"
        if _strict:
            raise DatasetValidationError(msg)
        warnings.warn(msg, stacklevel=2)
        return

    if not split_path.is_dir():
        msg = f"{split_name.capitalize()} path is not a directory: {split_path}"
        if _strict:
            raise DatasetValidationError(msg)
        warnings.warn(msg, stacklevel=2)
        return

    # Check for images subdirectory
    images_path = split_path / "images"
    if not images_path.exists():
        msg = f"Images directory not found in {split_name}: {images_path}"
        if _strict:
            raise DatasetValidationError(msg)
        warnings.warn(msg, stacklevel=2)
        return

    if not images_path.is_dir():
        msg = f"Images path is not a directory: {images_path}"
        if _strict:
            raise DatasetValidationError(msg)
        warnings.warn(msg, stacklevel=2)


def count_dataset_images(split_path: Path | None) -> int:
    """Count the number of images in a dataset split.

    Args:
        split_path: Path to split directory (e.g., train/).
            If None, returns 0.

    Returns:
        Number of image files (.jpg, .jpeg, .png) in images/ subdirectory.
    """
    if split_path is None:
        return 0

    images_path = split_path / "images"
    if not images_path.exists():
        return 0

    count = 0
    for file_path in images_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    return count
