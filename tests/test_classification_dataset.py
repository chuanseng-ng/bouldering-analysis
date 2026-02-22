"""Tests for the classification dataset loader module.

This module provides comprehensive tests for loading and validating
folder-per-class classification datasets used for hold type classification.
"""

from pathlib import Path
from collections.abc import Sequence

import pytest

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

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
from src.training.exceptions import (
    ClassTaxonomyError,
    DatasetNotFoundError,
    DatasetValidationError,
)


# ============================================================================
# Fixtures
# ============================================================================


def _create_class_folders(
    split_path: Path, classes: Sequence[str], image_count: int
) -> None:
    """Create class subfolders with dummy images inside a split directory.

    Args:
        split_path: Path to the split directory (e.g., train/).
        classes: List of class folder names to create.
        image_count: Number of dummy .jpg images per class folder.
    """
    for cls in classes:
        cls_dir = split_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(image_count):
            (cls_dir / f"img_{i}.jpg").touch()


@pytest.fixture
def valid_classification_dataset(tmp_path: Path) -> Path:
    """Create a valid classification dataset with train and val splits.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the dataset root directory.
    """
    classes = ["jug", "crimp", "sloper", "pinch", "volume", "unknown"]
    _create_class_folders(tmp_path / "train", classes, image_count=5)
    _create_class_folders(tmp_path / "val", classes, image_count=3)
    return tmp_path


@pytest.fixture
def valid_classification_dataset_with_test(
    valid_classification_dataset: Path,
) -> Path:
    """Create a valid classification dataset with train, val, and test splits.

    Args:
        valid_classification_dataset: Base valid dataset fixture.

    Returns:
        Path to the dataset with test split.
    """
    classes = ["jug", "crimp", "sloper", "pinch", "volume", "unknown"]
    _create_class_folders(valid_classification_dataset / "test", classes, image_count=2)
    return valid_classification_dataset


@pytest.fixture
def dataset_with_extra_class(tmp_path: Path) -> Path:
    """Create a dataset with an extra class folder (pocket) in train split.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the dataset root directory.
    """
    classes = ["jug", "crimp", "sloper", "pinch", "volume", "unknown", "pocket"]
    _create_class_folders(tmp_path / "train", classes, image_count=2)
    classes_val = ["jug", "crimp", "sloper", "pinch", "volume", "unknown"]
    _create_class_folders(tmp_path / "val", classes_val, image_count=2)
    return tmp_path


@pytest.fixture
def dataset_with_missing_class(tmp_path: Path) -> Path:
    """Create a dataset missing one class folder (unknown) in train split.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the dataset root directory.
    """
    classes = ["jug", "crimp", "sloper", "pinch", "volume"]  # missing "unknown"
    _create_class_folders(tmp_path / "train", classes, image_count=2)
    all_classes = ["jug", "crimp", "sloper", "pinch", "volume", "unknown"]
    _create_class_folders(tmp_path / "val", all_classes, image_count=2)
    return tmp_path


# ============================================================================
# TestConstants
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_hold_classes_value(self) -> None:
        """HOLD_CLASSES should contain the 6 hold types in expected order."""
        assert HOLD_CLASSES == ("jug", "crimp", "sloper", "pinch", "volume", "unknown")

    def test_hold_classes_is_tuple(self) -> None:
        """HOLD_CLASSES should be a tuple (immutable constant)."""
        assert isinstance(HOLD_CLASSES, tuple)

    def test_hold_class_count_value(self) -> None:
        """HOLD_CLASS_COUNT should be 6."""
        assert HOLD_CLASS_COUNT == 6

    def test_hold_class_count_matches_classes(self) -> None:
        """HOLD_CLASS_COUNT should match length of HOLD_CLASSES."""
        assert HOLD_CLASS_COUNT == len(HOLD_CLASSES)

    def test_image_extensions_value(self) -> None:
        """IMAGE_EXTENSIONS should contain .jpg, .jpeg, .png."""
        assert IMAGE_EXTENSIONS == frozenset({".jpg", ".jpeg", ".png"})

    def test_image_extensions_is_frozenset(self) -> None:
        """IMAGE_EXTENSIONS should be a frozenset (immutable)."""
        assert isinstance(IMAGE_EXTENSIONS, frozenset)


# ============================================================================
# TestComputeClassWeights
# ============================================================================


class TestComputeClassWeights:
    """Tests for the compute_class_weights function."""

    def test_balanced_classes_all_ones(self) -> None:
        """Balanced classes should produce weights all equal to 1.0."""
        counts = {
            "jug": 10,
            "crimp": 10,
            "sloper": 10,
            "pinch": 10,
            "volume": 10,
            "unknown": 10,
        }
        weights = compute_class_weights(counts)

        assert len(weights) == HOLD_CLASS_COUNT
        for w in weights:
            assert w == pytest.approx(1.0)

    def test_imbalanced_classes_correct_weights(self) -> None:
        """Imbalanced classes should produce inverse-frequency weights."""
        counts = {
            "jug": 60,
            "crimp": 30,
            "sloper": 20,
            "pinch": 10,
            "volume": 5,
            "unknown": 5,
        }
        total = 130
        n_classes = 6
        weights = compute_class_weights(counts)

        # Weight for jug: 130 / (6 * 60) = 0.3611...
        assert weights[0] == pytest.approx(total / (n_classes * 60))
        # Weight for unknown: 130 / (6 * 5) = 4.3333...
        assert weights[5] == pytest.approx(total / (n_classes * 5))

    def test_weights_order_matches_hold_classes(self) -> None:
        """Weights should be in HOLD_CLASSES order."""
        counts = {
            "jug": 10,
            "crimp": 20,
            "sloper": 30,
            "pinch": 40,
            "volume": 50,
            "unknown": 60,
        }
        weights = compute_class_weights(counts)

        # jug has smallest count → largest weight
        assert (
            weights[0] > weights[1] > weights[2] > weights[3] > weights[4] > weights[5]
        )

    def test_returns_list_of_floats(self) -> None:
        """Return value should be a list of floats."""
        counts = {
            "jug": 5,
            "crimp": 5,
            "sloper": 5,
            "pinch": 5,
            "volume": 5,
            "unknown": 5,
        }
        weights = compute_class_weights(counts)

        assert isinstance(weights, list)
        for w in weights:
            assert isinstance(w, float)

    def test_zero_count_raises_validation_error(self) -> None:
        """Zero count for any class should raise DatasetValidationError."""
        counts = {
            "jug": 10,
            "crimp": 0,
            "sloper": 10,
            "pinch": 10,
            "volume": 10,
            "unknown": 10,
        }

        with pytest.raises(DatasetValidationError, match="zero images"):
            compute_class_weights(counts)

    def test_missing_class_key_raises_validation_error(self) -> None:
        """Missing class key should raise DatasetValidationError."""
        counts = {
            "jug": 10,
            "crimp": 10,
            "sloper": 10,
        }  # missing pinch, volume, unknown

        with pytest.raises(DatasetValidationError, match="Missing class"):
            compute_class_weights(counts)

    def test_unexpected_class_key_raises_validation_error(self) -> None:
        """Unexpected class key should raise DatasetValidationError."""
        counts = {
            "jug": 10,
            "crimp": 10,
            "sloper": 10,
            "pinch": 10,
            "volume": 10,
            "unknown": 10,
            "pocket": 5,  # unexpected
        }

        with pytest.raises(DatasetValidationError, match="Unexpected class"):
            compute_class_weights(counts)


# ============================================================================
# TestCountImagesPerClass
# ============================================================================


class TestCountImagesPerClass:
    """Tests for the count_images_per_class function."""

    def test_correct_counts(self, valid_classification_dataset: Path) -> None:
        """Return correct image counts per class folder."""
        counts = count_images_per_class(valid_classification_dataset / "train")

        for cls in HOLD_CLASSES:
            assert counts[cls] == 5

    def test_mixed_extensions(self, tmp_path: Path) -> None:
        """Count images with .jpg, .jpeg, .png, and uppercase extensions."""
        split_path = tmp_path / "train"
        jug_dir = split_path / "jug"
        jug_dir.mkdir(parents=True)
        (jug_dir / "a.jpg").touch()
        (jug_dir / "b.jpeg").touch()
        (jug_dir / "c.png").touch()
        (jug_dir / "d.JPG").touch()

        counts = count_images_per_class(split_path)

        assert counts["jug"] == 4

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        """Non-image files should not be counted."""
        split_path = tmp_path / "train"
        jug_dir = split_path / "jug"
        jug_dir.mkdir(parents=True)
        (jug_dir / "a.jpg").touch()
        (jug_dir / "readme.txt").touch()
        (jug_dir / "data.json").touch()

        counts = count_images_per_class(split_path)

        assert counts["jug"] == 1

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        """Subdirectories inside class folders should not be counted."""
        split_path = tmp_path / "train"
        jug_dir = split_path / "jug"
        jug_dir.mkdir(parents=True)
        (jug_dir / "a.jpg").touch()
        sub = jug_dir / "nested"
        sub.mkdir()
        (sub / "b.jpg").touch()

        counts = count_images_per_class(split_path)

        assert counts["jug"] == 1

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        """Nonexistent split path should raise DatasetNotFoundError."""
        with pytest.raises(DatasetNotFoundError, match="Split directory not found"):
            count_images_per_class(tmp_path / "nonexistent")

    def test_empty_class_folders(self, tmp_path: Path) -> None:
        """Empty class folders should return count of 0."""
        split_path = tmp_path / "train"
        for cls in HOLD_CLASSES:
            (split_path / cls).mkdir(parents=True)

        counts = count_images_per_class(split_path)

        for cls in HOLD_CLASSES:
            assert counts[cls] == 0

    def test_only_counts_known_classes(self, tmp_path: Path) -> None:
        """Only count images in folders matching HOLD_CLASSES names."""
        split_path = tmp_path / "train"
        for cls in HOLD_CLASSES:
            (split_path / cls).mkdir(parents=True)
        extra = split_path / "pocket"
        extra.mkdir()
        (extra / "img.jpg").touch()

        counts = count_images_per_class(split_path)

        assert "pocket" not in counts


# ============================================================================
# TestValidateClassificationStructure
# ============================================================================


class TestValidateClassificationStructure:
    """Tests for the validate_classification_structure function."""

    def test_valid_structure_passes(self, valid_classification_dataset: Path) -> None:
        """Valid dataset with train and val splits should pass validation."""
        validate_classification_structure(valid_classification_dataset)

    def test_valid_structure_with_test(
        self, valid_classification_dataset_with_test: Path
    ) -> None:
        """Valid dataset with train, val, and test splits should pass."""
        validate_classification_structure(valid_classification_dataset_with_test)

    def test_missing_train_raises(self, tmp_path: Path) -> None:
        """Missing train directory should raise DatasetValidationError."""
        _create_class_folders(tmp_path / "val", HOLD_CLASSES, image_count=2)

        with pytest.raises(DatasetValidationError, match="train"):
            validate_classification_structure(tmp_path)

    def test_missing_val_raises(self, tmp_path: Path) -> None:
        """Missing val directory should raise DatasetValidationError."""
        _create_class_folders(tmp_path / "train", HOLD_CLASSES, image_count=2)

        with pytest.raises(DatasetValidationError, match="val"):
            validate_classification_structure(tmp_path)

    def test_extra_class_strict_raises(self, dataset_with_extra_class: Path) -> None:
        """Extra class folder in strict mode should raise ClassTaxonomyError."""
        with pytest.raises(ClassTaxonomyError, match="Unexpected class"):
            validate_classification_structure(dataset_with_extra_class, strict=True)

    def test_extra_class_non_strict_warns(self, dataset_with_extra_class: Path) -> None:
        """Extra class folder in non-strict mode should warn, not raise."""
        with pytest.warns(UserWarning, match="Unexpected class"):
            validate_classification_structure(dataset_with_extra_class, strict=False)

    def test_missing_class_strict_raises(
        self, dataset_with_missing_class: Path
    ) -> None:
        """Missing class folder in strict mode should raise ClassTaxonomyError."""
        with pytest.raises(ClassTaxonomyError, match="Missing class"):
            validate_classification_structure(dataset_with_missing_class, strict=True)

    def test_missing_class_non_strict_warns(
        self, dataset_with_missing_class: Path
    ) -> None:
        """Missing class folder in non-strict mode should warn, not raise."""
        with pytest.warns(UserWarning, match="Missing class"):
            validate_classification_structure(dataset_with_missing_class, strict=False)


# ============================================================================
# TestLoadHoldClassificationDataset
# ============================================================================


class TestLoadHoldClassificationDataset:
    """Tests for the main load_hold_classification_dataset function."""

    def test_load_valid_dataset(self, valid_classification_dataset: Path) -> None:
        """Load valid dataset and verify all result keys."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        assert result["train"] == (valid_classification_dataset / "train").resolve()
        assert result["val"] == (valid_classification_dataset / "val").resolve()
        assert result["test"] is None
        assert result["nc"] == 6
        assert result["names"] == list(HOLD_CLASSES)
        assert result["train_image_count"] == 30  # 6 classes * 5 images
        assert result["val_image_count"] == 18  # 6 classes * 3 images
        assert result["test_image_count"] == 0

    def test_load_with_test_split(
        self, valid_classification_dataset_with_test: Path
    ) -> None:
        """Load dataset with test split present."""
        result = load_hold_classification_dataset(
            valid_classification_dataset_with_test
        )

        assert result["test"] is not None
        assert result["test_image_count"] == 12  # 6 classes * 2 images

    def test_string_path_accepted(self, valid_classification_dataset: Path) -> None:
        """String path should be accepted and converted."""
        result = load_hold_classification_dataset(str(valid_classification_dataset))

        assert result["nc"] == 6
        assert isinstance(result["train"], Path)

    def test_nonexistent_root_raises(self, tmp_path: Path) -> None:
        """Nonexistent root directory should raise DatasetNotFoundError."""
        with pytest.raises(DatasetNotFoundError, match="not found"):
            load_hold_classification_dataset(tmp_path / "nonexistent")

    def test_file_as_root_raises(self, tmp_path: Path) -> None:
        """File path as root should raise DatasetNotFoundError."""
        file_path = tmp_path / "file.txt"
        file_path.touch()

        with pytest.raises(DatasetNotFoundError, match="not a directory"):
            load_hold_classification_dataset(file_path)

    def test_result_contains_class_counts(
        self, valid_classification_dataset: Path
    ) -> None:
        """Result should contain class_counts dict from train split."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        assert "class_counts" in result
        assert isinstance(result["class_counts"], dict)
        for cls in HOLD_CLASSES:
            assert result["class_counts"][cls] == 5

    def test_result_contains_class_weights(
        self, valid_classification_dataset: Path
    ) -> None:
        """Result should contain class_weights list."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        assert "class_weights" in result
        assert isinstance(result["class_weights"], list)
        assert len(result["class_weights"]) == HOLD_CLASS_COUNT

    def test_balanced_weights_all_ones(
        self, valid_classification_dataset: Path
    ) -> None:
        """Balanced dataset should produce weights all equal to 1.0."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        for w in result["class_weights"]:
            assert w == pytest.approx(1.0)

    def test_version_is_none(self, valid_classification_dataset: Path) -> None:
        """Version should always be None for classification datasets."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        assert result["version"] is None

    def test_metadata_is_empty_dict(self, valid_classification_dataset: Path) -> None:
        """Metadata should be an empty dict."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        assert isinstance(result["metadata"], dict)
        assert not result["metadata"]

    def test_result_has_all_required_keys(
        self, valid_classification_dataset: Path
    ) -> None:
        """Result should contain all ClassificationDatasetConfig keys."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        expected_keys = set(ClassificationDatasetConfig.__annotations__)
        assert set(result.keys()) == expected_keys

    def test_result_paths_are_absolute(
        self, valid_classification_dataset: Path
    ) -> None:
        """Train and val paths in result should be absolute."""
        result = load_hold_classification_dataset(valid_classification_dataset)

        assert result["train"].is_absolute()
        assert result["val"].is_absolute()

    def test_non_strict_mode_warns_not_raises(
        self, dataset_with_extra_class: Path
    ) -> None:
        """Non-strict mode should warn, not raise, for taxonomy mismatches."""
        with pytest.warns(UserWarning):
            result = load_hold_classification_dataset(
                dataset_with_extra_class, strict=False
            )

        assert result["nc"] == 6
        assert result["names"] == list(HOLD_CLASSES)
