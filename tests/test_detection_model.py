"""Tests for the training detection_model module.

This module provides comprehensive tests for the YOLOv8 detection model
building and hyperparameter configuration functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from src.training import (
    DEFAULT_MODEL_SIZE,
    INPUT_RESOLUTION,
    VALID_MODEL_SIZES,
    DetectionHyperparameters,
    build_hold_detector,
    get_default_hyperparameters,
    load_hyperparameters_from_file,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_hyperparams_yaml(tmp_path: Path) -> Path:
    """Create a valid hyperparameters YAML file.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the created YAML file.
    """
    config = {
        "epochs": 150,
        "batch_size": 32,
        "image_size": 640,
        "lr0": 0.001,
        "optimizer": "Adam",
    }
    yaml_path = tmp_path / "hyperparams.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return yaml_path


@pytest.fixture
def invalid_hyperparams_yaml(tmp_path: Path) -> Path:
    """Create an invalid hyperparameters YAML file.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to the created YAML file with invalid values.
    """
    config = {
        "epochs": -10,  # Invalid: must be >= 1
        "optimizer": "InvalidOptimizer",  # Invalid: not in allowed list
        "image_size": 100,  # Invalid: not multiple of 32
    }
    yaml_path = tmp_path / "invalid_hyperparams.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return yaml_path


# ============================================================================
# DetectionHyperparameters Tests
# ============================================================================


class TestDetectionHyperparameters:
    """Tests for the DetectionHyperparameters Pydantic model."""

    def test_default_hyperparameters(self) -> None:
        """Default hyperparameters should be valid and sensible."""
        hyperparams = DetectionHyperparameters()

        assert hyperparams.epochs == 100
        assert hyperparams.batch_size == 16
        assert hyperparams.image_size == INPUT_RESOLUTION
        assert hyperparams.learning_rate == 0.01
        assert hyperparams.optimizer == "AdamW"
        assert hyperparams.pretrained is True
        assert hyperparams.augment is True

    def test_custom_hyperparameters(self) -> None:
        """Custom hyperparameters should override defaults."""
        hyperparams = DetectionHyperparameters(  # type: ignore[call-arg]
            epochs=200,
            batch_size=32,
            lr0=0.001,  # Use alias for learning_rate
            optimizer="Adam",
        )

        assert hyperparams.epochs == 200
        assert hyperparams.batch_size == 32
        assert hyperparams.learning_rate == 0.001
        assert hyperparams.optimizer == "Adam"

    def test_learning_rate_alias(self) -> None:
        """Learning rate should be accessible via lr0 alias."""
        hyperparams = DetectionHyperparameters(lr0=0.005)
        assert hyperparams.learning_rate == 0.005

    def test_epochs_validation_positive(self) -> None:
        """Epochs must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(epochs=0)

        assert "epochs" in str(exc_info.value)

    def test_epochs_validation_maximum(self) -> None:
        """Epochs must not exceed maximum."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(epochs=1001)

        assert "epochs" in str(exc_info.value)

    def test_learning_rate_validation_positive(self) -> None:
        """Learning rate must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(lr0=0.0)

        assert "lr0" in str(exc_info.value)

    def test_learning_rate_validation_maximum(self) -> None:
        """Learning rate must not exceed 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(lr0=1.5)

        assert "lr0" in str(exc_info.value)

    def test_optimizer_validation_invalid(self) -> None:
        """Optimizer must be one of the valid options."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(optimizer="InvalidOptimizer")

        assert "optimizer" in str(exc_info.value)

    @pytest.mark.parametrize(
        "optimizer", ["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]
    )
    def test_optimizer_validation_valid(self, optimizer: str) -> None:
        """All valid optimizer names should be accepted."""
        hyperparams = DetectionHyperparameters(optimizer=optimizer)
        assert hyperparams.optimizer == optimizer

    def test_image_size_validation_multiple_of_32(self) -> None:
        """Image size must be a multiple of 32."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(image_size=100)  # type: ignore[call-arg]

        assert "image_size" in str(exc_info.value)

    @pytest.mark.parametrize("size", [32, 64, 128, 256, 512, 640, 1024])
    def test_image_size_validation_valid_sizes(self, size: int) -> None:
        """Valid image sizes (multiples of 32) should be accepted."""
        hyperparams = DetectionHyperparameters(image_size=size)  # type: ignore[call-arg]
        assert hyperparams.image_size == size

    def test_augmentation_parameters_range(self) -> None:
        """Augmentation parameters should be within valid ranges."""
        hyperparams = DetectionHyperparameters(
            hsv_h=0.5,
            hsv_s=0.8,
            hsv_v=0.6,
            degrees=45.0,
            translate=0.2,
            scale=0.8,
            shear=10.0,
            perspective=0.0005,
            flipud=0.3,
            fliplr=0.7,
            mosaic=0.9,
            mixup=0.1,
        )

        assert hyperparams.hsv_h == 0.5
        assert hyperparams.hsv_s == 0.8
        assert hyperparams.hsv_v == 0.6
        assert hyperparams.degrees == 45.0
        assert hyperparams.translate == 0.2
        assert hyperparams.scale == 0.8
        assert hyperparams.shear == 10.0
        assert hyperparams.perspective == 0.0005
        assert hyperparams.flipud == 0.3
        assert hyperparams.fliplr == 0.7
        assert hyperparams.mosaic == 0.9
        assert hyperparams.mixup == 0.1

    def test_to_dict_conversion(self) -> None:
        """to_dict should return dictionary with aliases."""
        hyperparams = DetectionHyperparameters(
            epochs=150,
            lr0=0.005,  # Use alias
            optimizer="Adam",
        )

        config_dict = hyperparams.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["epochs"] == 150
        assert config_dict["lr0"] == 0.005  # Should use alias
        assert config_dict["optimizer"] == "Adam"
        assert config_dict["imgsz"] == INPUT_RESOLUTION  # Should use alias

    def test_batch_size_auto(self) -> None:
        """Batch size -1 should be allowed for auto-batch."""
        hyperparams = DetectionHyperparameters(batch_size=-1)  # type: ignore[call-arg]
        assert hyperparams.batch_size == -1

    def test_batch_size_validation_zero(self) -> None:
        """Batch size 0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DetectionHyperparameters(batch_size=0)  # type: ignore[call-arg]

        assert "batch_size" in str(exc_info.value)

    def test_yolo_compatible_aliases(self) -> None:
        """Should accept YOLO-compatible field aliases."""
        # Test using aliases
        hyperparams = DetectionHyperparameters(
            batch=32,  # alias for batch_size
            imgsz=640,  # alias for image_size
            lr0=0.005,  # alias for learning_rate
        )

        assert hyperparams.batch_size == 32
        assert hyperparams.image_size == 640
        assert hyperparams.learning_rate == 0.005

    def test_populate_by_name(self) -> None:
        """Should accept both field names and aliases."""
        # Using field names
        hyperparams1 = DetectionHyperparameters(  # type: ignore[call-arg]
            batch_size=16, image_size=640, learning_rate=0.01
        )
        assert hyperparams1.batch_size == 16

        # Using aliases
        hyperparams2 = DetectionHyperparameters(batch=32, imgsz=320, lr0=0.005)
        assert hyperparams2.batch_size == 32

    def test_to_dict_uses_aliases(self) -> None:
        """to_dict should use YOLO-compatible aliases."""
        hyperparams = DetectionHyperparameters(  # type: ignore[call-arg]
            batch_size=32, image_size=320
        )
        config_dict = hyperparams.to_dict()

        # Should use aliases in output
        assert config_dict["batch"] == 32
        assert config_dict["imgsz"] == 320

    def test_device_configuration(self) -> None:
        """Device parameter should accept various formats."""
        # Empty string (auto-detect)
        hyperparams1 = DetectionHyperparameters(device="")
        assert hyperparams1.device == ""

        # CUDA device
        hyperparams2 = DetectionHyperparameters(device="cuda")
        assert hyperparams2.device == "cuda"

        # CPU device
        hyperparams3 = DetectionHyperparameters(device="cpu")
        assert hyperparams3.device == "cpu"

        # Device ID
        hyperparams4 = DetectionHyperparameters(device=0)
        assert hyperparams4.device == 0


# ============================================================================
# build_hold_detector Tests
# ============================================================================


class TestBuildHoldDetector:
    """Tests for the build_hold_detector function."""

    @patch("src.training.detection_model.YOLO")
    def test_build_default_detector(self, mock_yolo_class: MagicMock) -> None:
        """Should build detector with default configuration."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        result = build_hold_detector()

        # Verify YOLO was called with correct model path
        mock_yolo_class.assert_called_once_with("yolov8m.pt")
        assert result == mock_model

    @pytest.mark.parametrize(
        "size", ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    )
    @patch("src.training.detection_model.YOLO")
    def test_build_detector_different_sizes(
        self, mock_yolo_class: MagicMock, size: str
    ) -> None:
        """Should support different model sizes."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        build_hold_detector(model_size=size)
        mock_yolo_class.assert_called_once_with(f"{size}.pt")

    @patch("src.training.detection_model.YOLO")
    def test_build_detector_without_pretrained(
        self, mock_yolo_class: MagicMock
    ) -> None:
        """Should load YAML config when pretrained=False."""
        mock_model = MagicMock()
        mock_model.model.yaml = {}
        mock_yolo_class.return_value = mock_model

        _ = build_hold_detector(pretrained=False)

        # Verify YOLO was called with YAML path
        mock_yolo_class.assert_called_once_with("yolov8m.yaml")

        # Verify model config was updated
        assert mock_model.model.yaml["nc"] == 2
        assert mock_model.model.yaml["names"] == ["hold", "volume"]

    @patch("src.training.detection_model.YOLO")
    def test_build_detector_invalid_model_size(
        self, mock_yolo_class: MagicMock
    ) -> None:
        """Should raise ValueError for invalid model size."""
        with pytest.raises(ValueError) as exc_info:
            build_hold_detector(model_size="invalid_model")

        assert "model_size must be one of" in str(exc_info.value)
        mock_yolo_class.assert_not_called()

    @patch("src.training.detection_model.YOLO")
    def test_build_detector_invalid_num_classes(
        self, mock_yolo_class: MagicMock
    ) -> None:
        """Should raise ValueError for incorrect number of classes."""
        with pytest.raises(ValueError) as exc_info:
            build_hold_detector(num_classes=5)

        assert "num_classes must be 2" in str(exc_info.value)
        mock_yolo_class.assert_not_called()

    @patch("src.training.detection_model.YOLO")
    def test_build_detector_large_model(self, mock_yolo_class: MagicMock) -> None:
        """Should build large model variant."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        result = build_hold_detector(model_size="yolov8l", pretrained=True)

        mock_yolo_class.assert_called_once_with("yolov8l.pt")
        assert result == mock_model

    @patch("src.training.detection_model.YOLO")
    def test_build_detector_nano_model(self, mock_yolo_class: MagicMock) -> None:
        """Should build nano model variant for fast inference."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        result = build_hold_detector(model_size="yolov8n", pretrained=True)

        mock_yolo_class.assert_called_once_with("yolov8n.pt")
        assert result == mock_model


# ============================================================================
# get_default_hyperparameters Tests
# ============================================================================


class TestGetDefaultHyperparameters:
    """Tests for the get_default_hyperparameters function."""

    def test_returns_hyperparameters_instance(self) -> None:
        """Should return DetectionHyperparameters instance."""
        result = get_default_hyperparameters()
        assert isinstance(result, DetectionHyperparameters)

    def test_default_values_match(self) -> None:
        """Returned instance should have expected default values."""
        result = get_default_hyperparameters()

        assert result.epochs == 100
        assert result.batch_size == 16
        assert result.image_size == INPUT_RESOLUTION
        assert result.optimizer == "AdamW"
        assert result.pretrained is True


# ============================================================================
# load_hyperparameters_from_file Tests
# ============================================================================


class TestLoadHyperparametersFromFile:
    """Tests for the load_hyperparameters_from_file function."""

    def test_load_valid_hyperparameters(self, valid_hyperparams_yaml: Path) -> None:
        """Should load hyperparameters from valid YAML file."""
        result = load_hyperparameters_from_file(valid_hyperparams_yaml)

        assert isinstance(result, DetectionHyperparameters)
        assert result.epochs == 150
        assert result.batch_size == 32
        assert result.image_size == 640
        assert result.learning_rate == 0.001
        assert result.optimizer == "Adam"

    def test_load_from_nonexistent_file(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_hyperparameters_from_file(nonexistent)

        assert "Config file not found" in str(exc_info.value)

    def test_load_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        """Empty YAML should return hyperparameters with defaults."""
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")

        result = load_hyperparameters_from_file(empty_yaml)

        # Should have default values
        assert result.epochs == 100
        assert result.batch_size == 16

    def test_load_partial_config(self, tmp_path: Path) -> None:
        """Partial config should merge with defaults."""
        config = {"epochs": 200}
        yaml_path = tmp_path / "partial.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        result = load_hyperparameters_from_file(yaml_path)

        # Custom value
        assert result.epochs == 200
        # Default values
        assert result.batch_size == 16
        assert result.optimizer == "AdamW"

    def test_load_invalid_hyperparameters(self, invalid_hyperparams_yaml: Path) -> None:
        """Should raise ValidationError for invalid values."""
        with pytest.raises(ValidationError):
            load_hyperparameters_from_file(invalid_hyperparams_yaml)

    def test_load_accepts_string_path(self, valid_hyperparams_yaml: Path) -> None:
        """Should accept string path as well as Path object."""
        result = load_hyperparameters_from_file(str(valid_hyperparams_yaml))

        assert isinstance(result, DetectionHyperparameters)
        assert result.epochs == 150

    def test_load_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        """Should raise ValueError for malformed YAML."""
        malformed_yaml = tmp_path / "malformed.yaml"
        # Write invalid YAML syntax
        malformed_yaml.write_text("key: [unclosed list", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_hyperparameters_from_file(malformed_yaml)

        assert "Failed to parse YAML config file" in str(exc_info.value)
        assert str(malformed_yaml) in str(exc_info.value)


# ============================================================================
# Module Constants Tests
# ============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_model_size(self) -> None:
        """DEFAULT_MODEL_SIZE should be yolov8m."""
        assert DEFAULT_MODEL_SIZE == "yolov8m"

    def test_input_resolution(self) -> None:
        """INPUT_RESOLUTION should be 640."""
        assert INPUT_RESOLUTION == 640
        assert INPUT_RESOLUTION % 32 == 0  # Must be multiple of 32

    def test_valid_model_sizes(self) -> None:
        """VALID_MODEL_SIZES should contain all YOLOv8 variants."""
        expected_sizes = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        assert VALID_MODEL_SIZES == expected_sizes
        assert DEFAULT_MODEL_SIZE in VALID_MODEL_SIZES
