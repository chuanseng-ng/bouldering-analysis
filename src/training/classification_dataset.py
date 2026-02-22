"""Dataset loading and validation for hold classification training.

This module provides functions to load and validate folder-per-class
classification datasets for training hold type classification models.
It supports Roboflow classification exports and torchvision ImageFolder
format with the 6-class hold taxonomy.

The main function ``load_hold_classification_dataset()`` validates the
dataset structure and returns a configuration dictionary ready for training.

Example:
    >>> from src.training.classification_dataset import load_hold_classification_dataset
    >>> config = load_hold_classification_dataset("data/hold_classification")
    >>> print(config["train_image_count"])
    500
"""

import warnings
from pathlib import Path
from typing import Any, TypedDict

from src.logging_config import get_logger
from src.training.exceptions import (
    ClassTaxonomyError,
    DatasetNotFoundError,
    DatasetValidationError,
)

logger = get_logger(__name__)

# Expected 6-class hold taxonomy (immutable tuple)
HOLD_CLASSES: tuple[str, ...] = ("jug", "crimp", "sloper", "pinch", "volume", "unknown")
HOLD_CLASS_COUNT: int = len(HOLD_CLASSES)

# Supported image extensions (immutable)
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


class ClassificationDatasetConfig(TypedDict):
    """Typed return value for :func:`load_hold_classification_dataset`.

    Attributes:
        train: Absolute path to the training split directory.
        val: Absolute path to the validation split directory.
        test: Absolute path to the test split directory, or None.
        nc: Number of classes (always 6).
        names: List of class name strings in ``HOLD_CLASSES`` order.
        train_image_count: Total number of training images.
        val_image_count: Total number of validation images.
        test_image_count: Total number of test images (0 if no test split).
        class_counts: Per-class image counts from the training split.
        class_weights: Inverse-frequency weights (length 6, HOLD_CLASSES order).
        version: Always ``None`` (no version source in folder-per-class format).
        metadata: Extensible metadata dict (empty by default).
    """

    train: Path
    val: Path
    test: Path | None
    nc: int
    names: list[str]
    train_image_count: int
    val_image_count: int
    test_image_count: int
    class_counts: dict[str, int]
    class_weights: list[float]
    version: None
    metadata: dict[str, Any]


def compute_class_weights(class_counts: dict[str, int]) -> list[float]:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Uses the sklearn convention: ``total / (n_classes * count_per_class)``.
    The returned list is in ``HOLD_CLASSES`` order, ready to be passed to
    ``torch.tensor()`` and then to ``torch.nn.CrossEntropyLoss(weight=...)``.

    Args:
        class_counts: Mapping of class name to image count.
            Must contain exactly the keys from ``HOLD_CLASSES`` — no more,
            no fewer.  Every count must be ≥1; zero-count classes are always
            rejected regardless of any ``strict`` flag on the caller, because
            ``torch.nn.CrossEntropyLoss(weight=...)`` requires a weight for
            every class and silently assigning weight 0 would cause that class
            to be skipped during training — a correctness hazard, not just a
            validation warning.

    Returns:
        List of float weights in ``HOLD_CLASSES`` order.

    Raises:
        DatasetValidationError: If any class has zero images, if required
            class keys are missing, or if unexpected keys are present.

    Example:
        >>> counts = {"jug": 10, "crimp": 10, "sloper": 10,
        ...           "pinch": 10, "volume": 10, "unknown": 10}
        >>> compute_class_weights(counts)
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    hold_classes_set = set(HOLD_CLASSES)
    missing = [cls for cls in HOLD_CLASSES if cls not in class_counts]
    if missing:
        raise DatasetValidationError(f"Missing class keys in counts: {missing}")

    unexpected = [k for k in class_counts if k not in hold_classes_set]
    if unexpected:
        raise DatasetValidationError(
            f"Unexpected class keys in counts (not in HOLD_CLASSES): {unexpected}"
        )

    non_positive_classes = [cls for cls in HOLD_CLASSES if class_counts[cls] < 1]
    if non_positive_classes:
        raise DatasetValidationError(
            f"Classes with non-positive image counts cannot be weighted: {non_positive_classes}"
        )

    total = sum(class_counts[cls] for cls in HOLD_CLASSES)
    return [total / (HOLD_CLASS_COUNT * class_counts[cls]) for cls in HOLD_CLASSES]


def count_images_per_class(split_path: Path | str) -> dict[str, int]:
    """Count images in each class subfolder of a split directory.

    Only files with extensions in ``IMAGE_EXTENSIONS`` (case-insensitive)
    are counted. Subdirectories inside class folders are ignored.
    Only folders matching ``HOLD_CLASSES`` names are counted.

    Args:
        split_path: Path to a split directory (e.g., ``train/``).

    Returns:
        Dictionary mapping class name to image count for each class
        found in ``HOLD_CLASSES``.

    Raises:
        DatasetNotFoundError: If ``split_path`` does not exist.

    Example:
        >>> counts = count_images_per_class(Path("data/hold_classification/train"))
        >>> print(counts["jug"])
        42
    """
    split_path = Path(split_path).resolve()

    if not split_path.is_dir():
        raise DatasetNotFoundError(f"Split directory not found: {split_path}")

    counts: dict[str, int] = {}
    for cls in HOLD_CLASSES:
        cls_dir = split_path / cls
        if not cls_dir.is_dir():
            counts[cls] = 0
            continue
        count = sum(
            1
            for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[cls] = count

    return counts


def _validate_class_taxonomy_structure(
    split_path: Path,
    split_name: str,
    strict: bool,
) -> None:
    """Check that a split directory has exactly the expected class subfolders.

    Args:
        split_path: Path to the split directory.
        split_name: Name of the split for error messages (e.g., ``"train"``).
        strict: If True, raise on taxonomy mismatch. If False, warn.

    Raises:
        ClassTaxonomyError: If class folders are missing or unexpected
            (strict mode only).
    """
    existing = {d.name for d in split_path.iterdir() if d.is_dir()}
    expected = set(HOLD_CLASSES)

    missing = expected - existing
    extra = existing - expected

    if missing:
        msg = f"Missing class folders in {split_name}: {sorted(missing)}"
        if strict:
            raise ClassTaxonomyError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    if extra:
        msg = f"Unexpected class folders in {split_name}: {sorted(extra)}"
        if strict:
            raise ClassTaxonomyError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)


def validate_classification_structure(
    dataset_root: Path | str,
    strict: bool = True,
) -> None:
    """Validate that a classification dataset has the expected structure.

    Checks that ``train/`` and ``val/`` split directories exist and each
    contains the 6 expected class subfolders. The ``test/`` split is
    optional and validated only if present.

    Args:
        dataset_root: Path to the dataset root directory.
        strict: If True, raise :class:`~src.training.exceptions.ClassTaxonomyError`
            for any taxonomy mismatch (missing or extra class folders).
            If False, emit :class:`UserWarning` instead.

        Note:
            ``strict=False`` is useful when the dataset contains *extra*
            class folders (e.g., an unexpected ``pocket/`` folder from a
            Roboflow export) — the extra folder is simply ignored and the
            pipeline completes normally.  However, *missing* class folders
            still cause the load to fail downstream: the missing class gets a
            count of 0 from ``count_images_per_class``, and
            ``compute_class_weights`` unconditionally raises
            :class:`~src.training.exceptions.DatasetValidationError` for any
            zero-count class.  This is by design — a missing training class
            cannot be given a valid weight for
            ``torch.nn.CrossEntropyLoss(weight=...)``.

    Raises:
        DatasetValidationError: If required split directories (``train/``,
            ``val/``) are missing.  Always raised regardless of ``strict``.
        ClassTaxonomyError: If class folders are missing or unexpected
            (strict mode only).

    Example:
        >>> validate_classification_structure("data/hold_classification")
    """
    root = Path(dataset_root).resolve()

    logger.debug("Validating classification dataset structure at: %s", root)

    # Validate required splits
    for split_name in ("train", "val"):
        split_path = root / split_name
        if not split_path.exists() or not split_path.is_dir():
            raise DatasetValidationError(
                f"Required split directory not found: {split_name} "
                f"(expected at {split_path})"
            )
        _validate_class_taxonomy_structure(split_path, split_name, strict)

    # Validate optional test split
    test_path = root / "test"
    if test_path.is_dir():
        _validate_class_taxonomy_structure(test_path, "test", strict)

    logger.debug("Classification dataset structure validation passed")


def load_hold_classification_dataset(
    dataset_root: Path | str,
    strict: bool = True,
) -> ClassificationDatasetConfig:
    """Load and validate a hold classification dataset.

    This function loads folder-per-class datasets (Roboflow classification
    export / torchvision ImageFolder format). It validates the directory
    structure, counts images per class, and computes class weights.

    Args:
        dataset_root: Path to the dataset directory containing
            ``train/``, ``val/``, and optionally ``test/`` splits.
        strict: If True, raise errors for taxonomy mismatches (missing or
            extra class folders).  If False, emit :class:`UserWarning` and
            continue.

        Note:
            ``strict=False`` tolerates *extra* class folders without
            interrupting the pipeline.  It does **not** allow the pipeline to
            complete when class folders are *missing* — a missing folder
            produces a zero image count, and weight computation always fails
            on zero-count classes.  Use ``strict=False`` only when the dataset
            may contain unexpected extra folders (e.g., unreleased hold types
            added to a Roboflow export).

    Returns:
        :class:`ClassificationDatasetConfig` typed dict containing:
            - train: Absolute path to training split
            - val: Absolute path to validation split
            - test: Absolute path to test split (or None)
            - nc: Number of classes (always 6)
            - names: List of class names
            - train_image_count: Total training images
            - val_image_count: Total validation images
            - test_image_count: Total test images (0 if no test split)
            - class_counts: Per-class image counts from train split
            - class_weights: Inverse-frequency weights (len=6)
            - version: Always None (no version source in this format)
            - metadata: Empty dict (extensible for future use)

    Raises:
        DatasetNotFoundError: If dataset_root doesn't exist or is not
            a directory.
        DatasetValidationError: If dataset structure is invalid.
        ClassTaxonomyError: If class folders don't match expected taxonomy.

    Example:
        >>> config = load_hold_classification_dataset("data/hold_classification")
        >>> print(config["nc"])
        6
        >>> print(config["names"])
        ['jug', 'crimp', 'sloper', 'pinch', 'volume', 'unknown']
    """
    root = Path(dataset_root).resolve()

    logger.info("Loading hold classification dataset from: %s", root)

    # Validate root exists and is a directory
    if not root.exists():
        raise DatasetNotFoundError(f"Dataset not found at: {root}")
    if not root.is_dir():
        raise DatasetNotFoundError(f"Dataset path is not a directory: {root}")

    # Validate structure
    validate_classification_structure(root, strict=strict)

    # Resolve split paths
    train_path = (root / "train").resolve()
    val_path = (root / "val").resolve()
    test_path_candidate = root / "test"
    test_path = test_path_candidate.resolve() if test_path_candidate.is_dir() else None

    # Count images
    train_counts = count_images_per_class(train_path)
    val_counts = count_images_per_class(val_path)
    test_counts = count_images_per_class(test_path) if test_path else {}

    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    test_total = sum(test_counts.values()) if test_counts else 0

    # Compute class weights from training distribution
    class_weights = compute_class_weights(train_counts)

    logger.info(
        "Dataset loaded: train=%d, val=%d, test=%d images",
        train_total,
        val_total,
        test_total,
    )

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": HOLD_CLASS_COUNT,
        "names": list(HOLD_CLASSES),
        "train_image_count": train_total,
        "val_image_count": val_total,
        "test_image_count": test_total,
        "class_counts": train_counts,
        "class_weights": class_weights,
        "version": None,
        "metadata": {},
    }
