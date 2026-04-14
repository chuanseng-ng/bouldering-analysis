"""Dataset loading and validation for hold classification training.

This module provides functions to load and validate folder-per-class
classification datasets for training hold type classification models.
It supports Roboflow classification exports and torchvision ImageFolder
format with the 7-class hold taxonomy.

The main function ``load_hold_classification_dataset()`` validates the
dataset structure and returns a configuration dictionary ready for training.

Example:
    >>> from src.training.classification_dataset import load_hold_classification_dataset
    >>> config = load_hold_classification_dataset("data/hold_classification")
    >>> print(config["train_image_count"])
    500
"""

import random
import warnings
from pathlib import Path
from typing import Any, Final, TypedDict

from src.logging_config import get_logger
from src.training.exceptions import (
    ClassTaxonomyError,
    DatasetNotFoundError,
    DatasetValidationError,
)

logger = get_logger(__name__)

# 7-class canonical hold taxonomy.
# Edges have been aliased to crimp (identical difficulty profile).
# Mapping from dataset labels (capitalised) to normalised lowercase names:
#   Crimp      → crimp   |  Edges    → crimp    |  Foothold → foothold
#   Hand-holds → unknown |  Jug      → jug      |  Pinch    → pinch
#   Pocket     → pocket  |  Sloper   → sloper
HOLD_CLASSES: tuple[str, ...] = (
    "jug",
    "crimp",
    "sloper",
    "pinch",
    "pocket",
    "foothold",
    "unknown",
)
HOLD_CLASS_COUNT: int = len(HOLD_CLASSES)

# Alias map: Roboflow folder names (lowercased) → canonical HOLD_CLASSES entry.
# Keys cover all known Roboflow export folder names; any unrecognised folder
# triggers a UserWarning in count_images_per_class and
# _validate_class_taxonomy_structure.
LABEL_ALIASES: Final[dict[str, str]] = {
    "crimp": "crimp",
    "edges": "crimp",  # edge holds share the crimp difficulty profile
    "foothold": "foothold",
    "hand-holds": "unknown",
    "jug": "jug",
    "pinch": "pinch",
    "pocket": "pocket",
    "sloper": "sloper",
    "unknown": "unknown",
}

# Supported image extensions (immutable)
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


class ClassificationDatasetConfig(TypedDict):
    """Typed return value for :func:`load_hold_classification_dataset`.

    Attributes:
        train: Absolute path to the training split directory.
        val: Absolute path to the validation split directory.
        test: Absolute path to the test split directory, or None.
        nc: Number of classes (always 7).
        names: List of class name strings in ``HOLD_CLASSES`` order.
        train_image_count: Total number of training images (after any sampling).
        val_image_count: Total number of validation images.
        test_image_count: Total number of test images (0 if no test split).
        class_counts: Per-class image counts from the training split
            (after any sampling).
        class_weights: Inverse-frequency weights (length HOLD_CLASS_COUNT,
            HOLD_CLASSES order).  Zero-weight entries indicate masked classes
            (no training samples).
        active_classes: Canonical class names that have at least one training
            image.  A subset of ``HOLD_CLASSES`` when partial-class datasets
            are used.
        class_mask: Boolean list aligned with ``HOLD_CLASSES``; ``True`` means
            the class has training samples, ``False`` means it is masked.
        sampled_train_files: Mapping from canonical class name to the list of
            sampled file :class:`~pathlib.Path` objects used for training, when
            ``max_samples_per_class`` was set.  ``None`` when no volume cap is
            applied (all files are used).
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
    class_weights: list[float]  # length == HOLD_CLASS_COUNT, HOLD_CLASSES order
    active_classes: list[str]
    class_mask: list[bool]
    sampled_train_files: dict[str, list[Path]] | None
    version: None
    metadata: dict[str, Any]


def compute_class_weights(
    class_counts: dict[str, int],
    allow_missing: bool = False,
) -> list[float]:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Uses the sklearn convention: ``total / (n_classes * count_per_class)``.
    The returned list is in ``HOLD_CLASSES`` order, ready to be passed to
    ``torch.tensor()`` and then to ``torch.nn.CrossEntropyLoss(weight=...)``.

    Args:
        class_counts: Mapping of class name to image count.
            Must contain exactly the keys from ``HOLD_CLASSES`` — no more,
            no fewer.
        allow_missing: If ``False`` (default), raises
            :class:`~src.training.exceptions.DatasetValidationError` when any
            class has zero images — the original behaviour.  If ``True``,
            zero-count classes are assigned a weight of ``0.0`` instead.
            Weight computation uses only the active (non-zero) classes so the
            remaining weights are still properly normalised.  Use this flag
            only for partial-class datasets paired with a fine-tuning
            checkpoint; training from scratch with masked classes produces
            uninitialised output heads.

    Returns:
        List of float weights in ``HOLD_CLASSES`` order.  Zero-weight entries
        indicate masked classes that will be ignored by
        ``torch.nn.CrossEntropyLoss(weight=...)``.

    Raises:
        DatasetValidationError: If required class keys are missing, unexpected
            keys are present, or (when ``allow_missing=False``) any class has
            zero images.  Also raised when ``allow_missing=True`` but *all*
            classes have zero images.

    Example:
        >>> counts = {"jug": 10, "crimp": 10, "sloper": 10,
        ...           "pinch": 10, "pocket": 10, "foothold": 10, "unknown": 10}
        >>> compute_class_weights(counts)
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        >>> partial = {"jug": 10, "crimp": 20, "sloper": 0,
        ...            "pinch": 0, "pocket": 0, "foothold": 0, "unknown": 0}
        >>> weights = compute_class_weights(partial, allow_missing=True)
        >>> weights[0] > 0  # jug has weight
        True
        >>> weights[2] == 0.0  # sloper is masked
        True
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

    if allow_missing:
        active_counts = {cls: c for cls, c in class_counts.items() if c > 0}
        if not active_counts:
            raise DatasetValidationError(
                "No classes with positive image counts — cannot compute weights."
            )
        total_active = sum(active_counts.values())
        n_active = len(active_counts)
        return [
            (
                total_active / (n_active * class_counts[cls])
                if class_counts[cls] > 0
                else 0.0
            )
            for cls in HOLD_CLASSES
        ]

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
    Folder names are lowercased and resolved through ``LABEL_ALIASES``
    before accumulating counts, so aliased folders (e.g. ``Edges/``)
    are merged into their canonical target class (e.g. ``crimp``).

    Args:
        split_path: Path to a split directory (e.g., ``train/``).

    Returns:
        Dictionary mapping canonical class name to image count for each
        class in ``HOLD_CLASSES``.  All keys are always present.

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

    counts: dict[str, int] = {cls: 0 for cls in HOLD_CLASSES}
    for item in split_path.iterdir():
        if not item.is_dir():
            continue
        normalized = item.name.lower()
        target = LABEL_ALIASES.get(normalized)
        if target is None:
            warnings.warn(
                f"Unknown class folder {item.name!r} in {split_path} — skipping",
                UserWarning,
                stacklevel=2,
            )
            continue
        counts[target] += sum(
            1
            for f in item.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

    return counts


def _sample_class_images(
    split_path: Path,
    max_per_class: int,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Sample up to ``max_per_class`` image paths per class from a split directory.

    Iterates over class sub-folders (applying ``LABEL_ALIASES``), collects all
    image files, and randomly samples up to ``max_per_class`` per class.  When
    a class has fewer than ``max_per_class`` images all of its files are kept.

    Args:
        split_path: Path to the split directory (e.g. ``train/``).
        max_per_class: Maximum number of files to retain per canonical class.
            Must be ≥ 1.
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Dict mapping canonical class name to a list of sampled
        :class:`~pathlib.Path` objects.  All seven ``HOLD_CLASSES`` keys are
        always present; classes with no files have an empty list.

    Raises:
        DatasetNotFoundError: If ``split_path`` does not exist.

    Example:
        >>> sampled = _sample_class_images(Path("data/crops/train"), max_per_class=50)
        >>> len(sampled["jug"]) <= 50
        True
    """
    split_path = Path(split_path).resolve()
    if not split_path.is_dir():
        from src.training.exceptions import (
            DatasetNotFoundError,
        )  # local to avoid circular

        raise DatasetNotFoundError(f"Split directory not found: {split_path}")

    rng = random.Random(seed)
    result: dict[str, list[Path]] = {cls: [] for cls in HOLD_CLASSES}

    for item in split_path.iterdir():
        if not item.is_dir():
            continue
        normalized = item.name.lower()
        target = LABEL_ALIASES.get(normalized)
        if target is None:
            continue

        all_files = [
            f
            for f in item.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(all_files) <= max_per_class:
            result[target].extend(all_files)
        else:
            result[target].extend(rng.sample(all_files, max_per_class))

    return result


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
    # Resolve each existing folder via LABEL_ALIASES (case-insensitive).
    # Multiple source folders may map to the same target (e.g. Crimp + Edges → crimp).
    existing_dirs = [d for d in split_path.iterdir() if d.is_dir()]
    resolved: set[str] = set()
    unrecognised: list[str] = []
    for d in existing_dirs:
        target = LABEL_ALIASES.get(d.name.lower())
        if target is None:
            unrecognised.append(d.name)
        else:
            resolved.add(target)

    expected = set(HOLD_CLASSES)
    missing = expected - resolved
    extra = unrecognised  # folders with no alias are the only "unexpected" ones

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
    contains folders that resolve to all 7 expected canonical classes via
    ``LABEL_ALIASES``. The ``test/`` split is optional and validated only if
    present.

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
    max_samples_per_class: int | None = None,
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
        max_samples_per_class: When set, at most this many images are used
            per class during training.  Sampling is random but reproducible
            (seed 42).  Original files are **never deleted** — the cap is
            applied only at load time.  The returned ``sampled_train_files``
            field stores the selected file paths for use by the data loader.
            When ``None`` (default) all available images are used.

        Note:
            ``strict=False`` tolerates *extra* class folders without
            interrupting the pipeline.  A *missing* class folder produces a
            zero count.  When ``max_samples_per_class`` is provided a
            missing class simply contributes zero samples; when the parameter
            is ``None`` the standard behaviour raises immediately.

    Returns:
        :class:`ClassificationDatasetConfig` typed dict containing:
            - train: Absolute path to training split
            - val: Absolute path to validation split
            - test: Absolute path to test split (or None)
            - nc: Number of classes (always 7)
            - names: List of class names
            - train_image_count: Total training images (after any sampling)
            - val_image_count: Total validation images
            - test_image_count: Total test images (0 if no test split)
            - class_counts: Per-class image counts (after any sampling)
            - class_weights: Inverse-frequency weights (len=HOLD_CLASS_COUNT);
              zero for masked classes
            - active_classes: Classes with ≥1 training image
            - class_mask: Boolean list aligned with HOLD_CLASSES
            - sampled_train_files: Sampled file paths when max_samples_per_class
              is set, else None
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
        7
        >>> print(config["names"])
        ['jug', 'crimp', 'sloper', 'pinch', 'pocket', 'foothold', 'unknown']
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

    # Volume control: sample train images when a cap is requested
    sampled_train_files: dict[str, list[Path]] | None = None
    if max_samples_per_class is not None:
        sampled_train_files = _sample_class_images(train_path, max_samples_per_class)
        train_counts: dict[str, int] = {
            cls: len(paths) for cls, paths in sampled_train_files.items()
        }
    else:
        train_counts = count_images_per_class(train_path)

    val_counts = count_images_per_class(val_path)
    test_counts = count_images_per_class(test_path) if test_path else {}

    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    test_total = sum(test_counts.values()) if test_counts else 0

    # Determine active classes and mask
    active_classes = [cls for cls in HOLD_CLASSES if train_counts.get(cls, 0) > 0]
    class_mask = [train_counts.get(cls, 0) > 0 for cls in HOLD_CLASSES]

    # Compute class weights — allow zero-count classes when any are missing
    has_missing_class = any(count == 0 for count in train_counts.values())
    class_weights = compute_class_weights(
        train_counts,
        allow_missing=has_missing_class,
    )

    logger.info(
        "Dataset loaded: train=%d, val=%d, test=%d images (active_classes=%d/%d)",
        train_total,
        val_total,
        test_total,
        len(active_classes),
        HOLD_CLASS_COUNT,
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
        "active_classes": active_classes,
        "class_mask": class_mask,
        "sampled_train_files": sampled_train_files,
        "version": None,
        "metadata": {},
    }
