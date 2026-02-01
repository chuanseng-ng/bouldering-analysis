"""Tests for the training datasets module.

This module provides comprehensive tests for the hold detection dataset
loading and validation functionality.
"""

from pathlib import Path

import pytest
import yaml

from src.training import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_CLASSES,
    ClassTaxonomyError,
    DatasetNotFoundError,
    DatasetValidationError,
    TrainingError,
    count_dataset_images,
    load_hold_detection_dataset,
    validate_data_yaml,
    validate_directory_structure,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """Create a valid YOLOv8 detection dataset structure.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the created dataset root directory.
    """
    # Create directories
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "val" / "labels").mkdir(parents=True)

    # Create data.yaml
    data_yaml = {
        "train": "train",
        "val": "val",
        "nc": 2,
        "names": ["hold", "volume"],
    }
    with open(tmp_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    # Create sample images
    for i in range(5):
        (tmp_path / "train" / "images" / f"img_{i}.jpg").touch()
        (tmp_path / "val" / "images" / f"img_{i}.jpg").touch()

    return tmp_path


@pytest.fixture
def valid_dataset_with_test(valid_dataset: Path) -> Path:
    """Create a valid dataset with optional test split.

    Args:
        valid_dataset: Base valid dataset fixture.

    Returns:
        Path to the dataset with test split.
    """
    # Add test split directories
    (valid_dataset / "test" / "images").mkdir(parents=True)
    (valid_dataset / "test" / "labels").mkdir(parents=True)

    # Create sample test images
    for i in range(3):
        (valid_dataset / "test" / "images" / f"test_img_{i}.jpg").touch()

    # Update data.yaml to include test
    data_yaml = {
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": 2,
        "names": ["hold", "volume"],
    }
    with open(valid_dataset / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    return valid_dataset


@pytest.fixture
def valid_dataset_with_metadata(valid_dataset: Path) -> Path:
    """Create a valid dataset with full metadata.

    Args:
        valid_dataset: Base valid dataset fixture.

    Returns:
        Path to the dataset with metadata.
    """
    data_yaml = {
        "train": "train",
        "val": "val",
        "nc": 2,
        "names": ["hold", "volume"],
        "dataset_version": "rf-climbing-holds-v1",
        "export_format": "yolov8",
        "export_date": "2026-01-28",
        "notes": "Test dataset for unit tests",
    }
    with open(valid_dataset / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    return valid_dataset


@pytest.fixture
def dataset_with_dict_names(valid_dataset: Path) -> Path:
    """Create a dataset with names as dict format.

    Args:
        valid_dataset: Base valid dataset fixture.

    Returns:
        Path to the dataset with dict names.
    """
    data_yaml = {
        "train": "train",
        "val": "val",
        "nc": 2,
        "names": {0: "hold", 1: "volume"},
    }
    with open(valid_dataset / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)

    return valid_dataset


# ============================================================================
# TestLoadHoldDetectionDataset - Main function tests
# ============================================================================


class TestLoadHoldDetectionDataset:
    """Tests for the main load_hold_detection_dataset function."""

    def test_load_valid_dataset_train_val_only(self, valid_dataset: Path) -> None:
        """Load valid dataset with train/val only."""
        result = load_hold_detection_dataset(valid_dataset)

        assert result["train"] == valid_dataset / "train"
        assert result["val"] == valid_dataset / "val"
        assert result["test"] is None
        assert result["nc"] == 2
        assert result["names"] == ["hold", "volume"]
        assert result["train_image_count"] == 5
        assert result["val_image_count"] == 5
        assert result["test_image_count"] == 0

    def test_load_valid_dataset_with_test(self, valid_dataset_with_test: Path) -> None:
        """Load valid dataset with train/val/test."""
        result = load_hold_detection_dataset(valid_dataset_with_test)

        assert result["train"] == valid_dataset_with_test / "train"
        assert result["val"] == valid_dataset_with_test / "val"
        assert result["test"] == valid_dataset_with_test / "test"
        assert result["nc"] == 2
        assert result["names"] == ["hold", "volume"]
        assert result["train_image_count"] == 5
        assert result["val_image_count"] == 5
        assert result["test_image_count"] == 3

    def test_load_dataset_with_full_metadata(
        self, valid_dataset_with_metadata: Path
    ) -> None:
        """Load dataset with full metadata."""
        result = load_hold_detection_dataset(valid_dataset_with_metadata)

        assert result["version"] == "rf-climbing-holds-v1"
        assert result["metadata"]["export_format"] == "yolov8"
        assert result["metadata"]["export_date"] == "2026-01-28"
        assert result["metadata"]["notes"] == "Test dataset for unit tests"

    def test_load_dataset_with_dict_names(self, dataset_with_dict_names: Path) -> None:
        """Load dataset with names in dict format."""
        result = load_hold_detection_dataset(dataset_with_dict_names)

        assert result["names"] == ["hold", "volume"]
        assert result["nc"] == 2

    def test_load_dataset_returns_absolute_paths(self, valid_dataset: Path) -> None:
        """Loaded paths should be absolute Path objects."""
        result = load_hold_detection_dataset(valid_dataset)

        assert isinstance(result["train"], Path)
        assert isinstance(result["val"], Path)
        assert result["train"].is_absolute()
        assert result["val"].is_absolute()

    def test_load_dataset_string_path(self, valid_dataset: Path) -> None:
        """Load dataset using string path instead of Path object."""
        result = load_hold_detection_dataset(str(valid_dataset))

        assert result["train"] == valid_dataset / "train"
        assert result["nc"] == 2

    def test_load_nonexistent_dataset(self, tmp_path: Path) -> None:
        """Raise DatasetNotFoundError for non-existent path."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_hold_detection_dataset(nonexistent)

        assert "Dataset not found at" in str(exc_info.value)

    def test_load_dataset_file_not_directory(self, tmp_path: Path) -> None:
        """Raise DatasetNotFoundError when path is a file, not directory."""
        file_path = tmp_path / "file.txt"
        file_path.touch()

        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_hold_detection_dataset(file_path)

        assert "not a directory" in str(exc_info.value)

    def test_load_dataset_missing_data_yaml(self, tmp_path: Path) -> None:
        """Raise DatasetNotFoundError when data.yaml is missing."""
        # Create directories but no data.yaml
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)

        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_hold_detection_dataset(tmp_path)

        assert "data.yaml not found" in str(exc_info.value)

    def test_load_dataset_no_version_returns_none(self, valid_dataset: Path) -> None:
        """Version should be None if not specified in data.yaml."""
        result = load_hold_detection_dataset(valid_dataset)

        assert result["version"] is None

    def test_load_dataset_counts_mixed_image_formats(self, valid_dataset: Path) -> None:
        """Count images with various extensions (.jpg, .jpeg, .png)."""
        # Add images with different extensions
        (valid_dataset / "train" / "images" / "extra.jpeg").touch()
        (valid_dataset / "train" / "images" / "extra.png").touch()

        result = load_hold_detection_dataset(valid_dataset)

        # 5 original .jpg + 1 .jpeg + 1 .png = 7
        assert result["train_image_count"] == 7

    def test_load_dataset_ignores_non_image_files(self, valid_dataset: Path) -> None:
        """Non-image files should not be counted."""
        # Add non-image files
        (valid_dataset / "train" / "images" / "readme.txt").touch()
        (valid_dataset / "train" / "images" / "data.json").touch()

        result = load_hold_detection_dataset(valid_dataset)

        # Only the 5 original .jpg files
        assert result["train_image_count"] == 5


# ============================================================================
# TestValidateDataYaml - YAML validation tests
# ============================================================================


class TestValidateDataYaml:
    """Tests for the validate_data_yaml function."""

    def test_validate_valid_yaml(self, valid_dataset: Path) -> None:
        """Valid data.yaml should pass validation."""
        yaml_path = valid_dataset / "data.yaml"
        result = validate_data_yaml(yaml_path)

        assert result["train"] == "train"
        assert result["val"] == "val"
        assert result["nc"] == 2

    def test_validate_missing_yaml(self, tmp_path: Path) -> None:
        """Raise DatasetNotFoundError for missing data.yaml."""
        yaml_path = tmp_path / "data.yaml"

        with pytest.raises(DatasetNotFoundError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "data.yaml not found" in str(exc_info.value)

    def test_validate_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError for invalid YAML syntax."""
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text("invalid: yaml: content: [\n", encoding="utf-8")

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "Failed to parse data.yaml" in str(exc_info.value)

    def test_validate_empty_yaml(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError for empty data.yaml."""
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text("", encoding="utf-8")

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "empty" in str(exc_info.value)

    def test_validate_yaml_not_mapping(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError when YAML is not a mapping."""
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "must contain a YAML mapping" in str(exc_info.value)

    def test_validate_missing_train_key(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError when 'train' key is missing."""
        yaml_path = tmp_path / "data.yaml"
        data = {"val": "val", "nc": 2, "names": ["hold", "volume"]}
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "Missing required key" in str(exc_info.value)
        assert "'train'" in str(exc_info.value)

    def test_validate_missing_val_key(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError when 'val' key is missing."""
        yaml_path = tmp_path / "data.yaml"
        data = {"train": "train", "nc": 2, "names": ["hold", "volume"]}
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "Missing required key" in str(exc_info.value)
        assert "'val'" in str(exc_info.value)

    def test_validate_missing_nc_key(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError when 'nc' key is missing."""
        yaml_path = tmp_path / "data.yaml"
        data = {"train": "train", "val": "val", "names": ["hold", "volume"]}
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "Missing required key" in str(exc_info.value)
        assert "'nc'" in str(exc_info.value)

    def test_validate_missing_names_key(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError when 'names' key is missing."""
        yaml_path = tmp_path / "data.yaml"
        data = {"train": "train", "val": "val", "nc": 2}
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_data_yaml(yaml_path)

        assert "Missing required key" in str(exc_info.value)
        assert "'names'" in str(exc_info.value)


# ============================================================================
# TestClassTaxonomy - Class validation tests
# ============================================================================


class TestClassTaxonomy:
    """Tests for class taxonomy validation."""

    def test_valid_class_count(self, valid_dataset: Path) -> None:
        """Valid dataset has exactly 2 classes."""
        result = load_hold_detection_dataset(valid_dataset)

        assert result["nc"] == EXPECTED_CLASS_COUNT
        assert len(result["names"]) == EXPECTED_CLASS_COUNT

    def test_valid_class_names(self, valid_dataset: Path) -> None:
        """Valid dataset has expected class names."""
        result = load_hold_detection_dataset(valid_dataset)

        assert result["names"] == EXPECTED_CLASSES

    def test_wrong_class_count(self, tmp_path: Path) -> None:
        """Raise ClassTaxonomyError when nc != 2."""
        # Create directories
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)

        # Create data.yaml with wrong class count
        data_yaml = {
            "train": "train",
            "val": "val",
            "nc": 5,
            "names": ["crimp", "jug", "sloper", "pinch", "pocket"],
        }
        with open(tmp_path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

        with pytest.raises(ClassTaxonomyError) as exc_info:
            load_hold_detection_dataset(tmp_path)

        assert "Expected 2 classes, found 5" in str(exc_info.value)

    def test_wrong_class_names(self, tmp_path: Path) -> None:
        """Raise ClassTaxonomyError when class names don't match."""
        # Create directories
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)

        # Create data.yaml with wrong class names
        data_yaml = {
            "train": "train",
            "val": "val",
            "nc": 2,
            "names": ["crimp", "jug"],
        }
        with open(tmp_path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

        with pytest.raises(ClassTaxonomyError) as exc_info:
            load_hold_detection_dataset(tmp_path)

        assert "Expected classes" in str(exc_info.value)
        assert "['hold', 'volume']" in str(exc_info.value)

    def test_class_names_wrong_order(self, tmp_path: Path) -> None:
        """Raise ClassTaxonomyError when class order is wrong."""
        # Create directories
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)

        # Create data.yaml with classes in wrong order
        data_yaml = {
            "train": "train",
            "val": "val",
            "nc": 2,
            "names": ["volume", "hold"],  # Wrong order
        }
        with open(tmp_path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

        with pytest.raises(ClassTaxonomyError) as exc_info:
            load_hold_detection_dataset(tmp_path)

        assert "Expected classes" in str(exc_info.value)

    def test_dict_names_valid(self, dataset_with_dict_names: Path) -> None:
        """Accept names in dict format with correct mapping."""
        result = load_hold_detection_dataset(dataset_with_dict_names)

        assert result["names"] == ["hold", "volume"]

    def test_dict_names_missing_key(self, tmp_path: Path) -> None:
        """Raise ClassTaxonomyError for dict names with missing keys."""
        # Create directories
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)

        # Create data.yaml with dict names missing key 0
        data_yaml = {
            "train": "train",
            "val": "val",
            "nc": 2,
            "names": {1: "hold", 2: "volume"},  # Missing key 0
        }
        with open(tmp_path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

        with pytest.raises(ClassTaxonomyError) as exc_info:
            load_hold_detection_dataset(tmp_path)

        assert "sequential integer keys" in str(exc_info.value)

    def test_names_wrong_type(self, tmp_path: Path) -> None:
        """Raise ClassTaxonomyError when names is neither list nor dict."""
        # Create directories
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)

        # Create data.yaml with names as string
        data_yaml = {
            "train": "train",
            "val": "val",
            "nc": 2,
            "names": "hold, volume",
        }
        with open(tmp_path / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

        with pytest.raises(ClassTaxonomyError) as exc_info:
            load_hold_detection_dataset(tmp_path)

        assert "must be a list or dict" in str(exc_info.value)


# ============================================================================
# TestValidateDirectoryStructure - Directory validation tests
# ============================================================================


class TestValidateDirectoryStructure:
    """Tests for the validate_directory_structure function."""

    def test_validate_valid_structure(self, valid_dataset: Path) -> None:
        """Valid directory structure passes validation."""
        config = {"train": "train", "val": "val"}

        # Should not raise
        validate_directory_structure(valid_dataset, config)

    def test_missing_train_directory(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError for missing train directory."""
        # Create only val
        (tmp_path / "val" / "images").mkdir(parents=True)

        config = {"train": "train", "val": "val"}

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_directory_structure(tmp_path, config)

        assert "Train directory not found" in str(exc_info.value)

    def test_missing_val_directory(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError for missing val directory."""
        # Create only train
        (tmp_path / "train" / "images").mkdir(parents=True)

        config = {"train": "train", "val": "val"}

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_directory_structure(tmp_path, config)

        assert "Val directory not found" in str(exc_info.value)

    def test_missing_images_subdirectory(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError for missing images subdirectory."""
        # Create train and val without images subdirs
        (tmp_path / "train").mkdir()
        (tmp_path / "val" / "images").mkdir(parents=True)

        config = {"train": "train", "val": "val"}

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_directory_structure(tmp_path, config)

        assert "Images directory not found" in str(exc_info.value)

    def test_split_path_is_file(self, tmp_path: Path) -> None:
        """Raise DatasetValidationError when split path is a file."""
        # Create train as file, val as directory
        (tmp_path / "train").touch()
        (tmp_path / "val" / "images").mkdir(parents=True)

        config = {"train": "train", "val": "val"}

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_directory_structure(tmp_path, config)

        assert "not a directory" in str(exc_info.value)

    def test_optional_test_missing_in_non_strict(self, valid_dataset: Path) -> None:
        """Missing test directory should log warning in non-strict mode."""
        config = {"train": "train", "val": "val", "test": "test"}

        # Should not raise (warning only)
        validate_directory_structure(valid_dataset, config, strict=False)

    def test_optional_test_missing_in_strict(self, valid_dataset: Path) -> None:
        """Missing test directory should raise in strict mode."""
        config = {"train": "train", "val": "val", "test": "test"}

        with pytest.raises(DatasetValidationError) as exc_info:
            validate_directory_structure(valid_dataset, config, strict=True)

        assert "Test directory not found" in str(exc_info.value)


# ============================================================================
# TestCountDatasetImages - Image counting tests
# ============================================================================


class TestCountDatasetImages:
    """Tests for the count_dataset_images function."""

    def test_count_images_valid_split(self, valid_dataset: Path) -> None:
        """Count images in a valid split directory."""
        count = count_dataset_images(valid_dataset / "train")

        assert count == 5

    def test_count_images_none_path(self) -> None:
        """Return 0 for None path."""
        count = count_dataset_images(None)

        assert count == 0

    def test_count_images_missing_images_dir(self, tmp_path: Path) -> None:
        """Return 0 when images directory doesn't exist."""
        split_path = tmp_path / "train"
        split_path.mkdir()

        count = count_dataset_images(split_path)

        assert count == 0

    def test_count_images_empty_directory(self, tmp_path: Path) -> None:
        """Return 0 for empty images directory."""
        (tmp_path / "train" / "images").mkdir(parents=True)

        count = count_dataset_images(tmp_path / "train")

        assert count == 0

    def test_count_images_mixed_extensions(self, tmp_path: Path) -> None:
        """Count images with various valid extensions."""
        images_path = tmp_path / "train" / "images"
        images_path.mkdir(parents=True)

        # Create images with different extensions
        (images_path / "img1.jpg").touch()
        (images_path / "img2.jpeg").touch()
        (images_path / "img3.png").touch()
        (images_path / "img4.JPG").touch()  # Uppercase

        count = count_dataset_images(tmp_path / "train")

        assert count == 4

    def test_count_images_ignores_subdirectories(self, tmp_path: Path) -> None:
        """Subdirectories in images should not be counted."""
        images_path = tmp_path / "train" / "images"
        images_path.mkdir(parents=True)

        # Create images and subdirectory
        (images_path / "img1.jpg").touch()
        (images_path / "subdir").mkdir()
        (images_path / "subdir" / "img2.jpg").touch()

        count = count_dataset_images(tmp_path / "train")

        # Only count top-level images
        assert count == 1


# ============================================================================
# TestExceptions - Exception class tests
# ============================================================================


class TestExceptions:
    """Tests for custom exception classes."""

    def test_training_error_message(self) -> None:
        """TrainingError stores and returns message correctly."""
        error = TrainingError("Test error message")

        assert error.message == "Test error message"
        assert str(error) == "Test error message"

    def test_dataset_not_found_error_inheritance(self) -> None:
        """DatasetNotFoundError inherits from TrainingError."""
        error = DatasetNotFoundError("Path not found")

        assert isinstance(error, TrainingError)
        assert error.message == "Path not found"

    def test_dataset_validation_error_inheritance(self) -> None:
        """DatasetValidationError inherits from TrainingError."""
        error = DatasetValidationError("Validation failed")

        assert isinstance(error, TrainingError)
        assert error.message == "Validation failed"

    def test_class_taxonomy_error_inheritance(self) -> None:
        """ClassTaxonomyError inherits from TrainingError."""
        error = ClassTaxonomyError("Wrong classes")

        assert isinstance(error, TrainingError)
        assert error.message == "Wrong classes"

    def test_can_catch_all_with_training_error(self) -> None:
        """All custom exceptions can be caught with TrainingError."""
        exceptions = [
            DatasetNotFoundError("test"),
            DatasetValidationError("test"),
            ClassTaxonomyError("test"),
        ]

        for exc in exceptions:
            with pytest.raises(TrainingError):
                raise exc


# ============================================================================
# TestConstants - Module constants tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_expected_classes_value(self) -> None:
        """EXPECTED_CLASSES should be ['hold', 'volume']."""
        assert EXPECTED_CLASSES == ["hold", "volume"]

    def test_expected_class_count_value(self) -> None:
        """EXPECTED_CLASS_COUNT should be 2."""
        assert EXPECTED_CLASS_COUNT == 2

    def test_expected_classes_is_list(self) -> None:
        """EXPECTED_CLASSES should be a list."""
        assert isinstance(EXPECTED_CLASSES, list)

    def test_expected_class_count_matches_classes(self) -> None:
        """EXPECTED_CLASS_COUNT should match length of EXPECTED_CLASSES."""
        assert EXPECTED_CLASS_COUNT == len(EXPECTED_CLASSES)
