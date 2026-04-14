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

import random
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

# Expected class configuration for hold detection (8-class fine-grained taxonomy)
EXPECTED_CLASSES = [
    "Crimp",
    "Edges",
    "Foothold",
    "Hand-holds",
    "Jug",
    "Pinch",
    "Pocket",
    "Sloper",
]
EXPECTED_CLASS_COUNT = len(EXPECTED_CLASSES)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Required keys in data.yaml
REQUIRED_YAML_KEYS = ["train", "val", "nc", "names"]


def _sample_dataset_images(
    images_path: Path,
    max_images: int,
    seed: int = 42,
) -> list[Path]:
    """Sample up to ``max_images`` image file paths from an images directory.

    Args:
        images_path: Path to the ``images/`` subdirectory of a split.
        max_images: Maximum number of images to return.  Must be ≥ 1.
        seed: Random seed for reproducibility (default: 42).

    Returns:
        List of sampled :class:`~pathlib.Path` objects.  If the directory
        contains fewer than ``max_images`` files, all files are returned.

    Example:
        >>> paths = _sample_dataset_images(Path("data/train/images"), 100)
        >>> len(paths) <= 100
        True
    """
    all_files = [
        f
        for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(all_files) <= max_images:
        return all_files
    rng = random.Random(seed)
    return rng.sample(all_files, max_images)


def load_hold_detection_dataset(
    dataset_root: Path | str,
    strict: bool = True,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Load and validate a hold detection dataset in YOLOv8 format.

    This function loads datasets exported from Roboflow or other sources
    in YOLOv8 object detection format. It validates the directory structure,
    configuration file, and class taxonomy.

    Args:
        dataset_root: Path to the dataset directory containing data.yaml.
        strict: If True, raise errors for validation failures.
            If False, log warnings and continue where possible.
        max_images: When set, at most this many training images are used.
            Sampling is random but reproducible (seed 42).  The returned
            ``metadata["sampled_train_files"]`` contains the sampled file
            paths.  Val and test splits are not affected.

    Returns:
        Dictionary containing:
            - train: Absolute path to training directory
            - val: Absolute path to validation directory
            - test: Absolute path to test directory (or None)
            - nc: Number of classes
            - names: List of class names
            - train_image_count: Number of training images (after any sampling)
            - val_image_count: Number of validation images
            - test_image_count: Number of test images (or 0)
            - version: Dataset version string (if available)
            - metadata: Additional metadata from data.yaml; includes
              ``sampled_train_files`` key (list of Path) when ``max_images``
              is set.

    Raises:
        DatasetNotFoundError: If dataset_root or data.yaml doesn't exist.
        DatasetValidationError: If dataset structure or config is invalid.
        ClassTaxonomyError: If classes don't match expected taxonomy.

    Example:
        >>> config = load_hold_detection_dataset("data/climbing_holds")
        >>> print(config["train_image_count"])
        500
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

    # Volume control: sample training images when a cap is requested
    sampled_train_files: list[Path] | None = None
    if max_images is not None:
        images_path = train_path / "images"
        if images_path.is_dir():
            sampled_train_files = _sample_dataset_images(images_path, max_images)
            train_count = len(sampled_train_files)
        else:
            train_count = 0
    else:
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
    metadata: dict[str, Any] = {
        k: v
        for k, v in config.items()
        if k not in REQUIRED_YAML_KEYS and k not in ["test"]
    }
    if sampled_train_files is not None:
        metadata["sampled_train_files"] = sampled_train_files

    nc = config.get("nc", EXPECTED_CLASS_COUNT)
    names_raw = config.get("names", EXPECTED_CLASSES.copy())
    if isinstance(names_raw, dict):
        names = [names_raw[i] for i in range(len(names_raw))]
    else:
        names = list(names_raw)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": nc,
        "names": names,
        "train_image_count": train_count,
        "val_image_count": val_count,
        "test_image_count": test_count,
        "version": config.get("dataset_version"),
        "metadata": metadata,
    }


def validate_data_yaml(
    yaml_path: Path,
    strict: bool = True,
    flexible_nc: bool = False,
) -> dict[str, Any]:
    """Validate and parse data.yaml configuration file.

    Args:
        yaml_path: Path to data.yaml file.
        strict: If True, raise errors for validation failures.
        flexible_nc: If True, skip the ``nc == EXPECTED_CLASS_COUNT`` check
            and instead accept any ``nc >= 1``.  Use this for detection
            datasets with a different class count (e.g. 2-class hold/volume
            datasets) without disabling all validation via ``strict=False``.

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
    _validate_class_taxonomy(config, _strict=strict, flexible_nc=flexible_nc)

    logger.debug("data.yaml validation passed")
    return config


def _validate_class_taxonomy(
    config: dict[str, Any],
    _strict: bool = True,  # noqa: ARG001 - Reserved for future non-strict mode
    flexible_nc: bool = False,
) -> None:
    """Validate that class configuration matches expected taxonomy.

    Args:
        config: Parsed data.yaml configuration.
        _strict: Reserved for future use. Currently always raises on mismatch.
        flexible_nc: If True, skip the exact class-count check and accept any
            ``nc >= 1``.  When True, class-name validation is also skipped so
            datasets with an arbitrary taxonomy are accepted.

    Raises:
        ClassTaxonomyError: If class configuration is incorrect.

    Note:
        The _strict parameter is currently unused but reserved for future
        non-strict validation mode where mismatches would emit warnings
        instead of raising exceptions.
    """
    nc = config.get("nc")
    names = config.get("names")

    if flexible_nc:
        # Accept any positive class count; only validate that names is a
        # non-empty list (or dict) of strings.
        if not isinstance(nc, int) or nc < 1:
            raise ClassTaxonomyError(
                f"nc must be a positive integer when flexible_nc=True, got {nc!r}"
            )
        if isinstance(names, dict):
            names_list: list = list(names.values())
        elif isinstance(names, list):
            names_list = names
        else:
            raise ClassTaxonomyError(
                f"'names' must be a list or dict, got {type(names).__name__}"
            )
        if not names_list:
            raise ClassTaxonomyError("'names' must not be empty")
        for i, n in enumerate(names_list):
            if not isinstance(n, str):
                raise ClassTaxonomyError(
                    f"Class name at index {i} is not a string: {n!r}"
                )
        logger.debug("Class taxonomy validation passed (flexible_nc): %d classes", nc)
        return

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

    # Validate each element is a string before lowering (guards non-string YAML entries).
    for i, n in enumerate(names_list):
        if not isinstance(n, str):
            raise ClassTaxonomyError(
                f"Class name at index {i} is not a string: {n!r} (type {type(n).__name__})"
            )

    # Case-insensitive comparison: Roboflow exports may normalise capitalisation.
    if [n.lower() for n in names_list] != [e.lower() for e in EXPECTED_CLASSES]:
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
