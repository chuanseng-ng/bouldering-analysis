"""Tests for the hold type classification inference module.

This module tests the classification inference using trained ResNet/MobileNetV3
weights, including model caching, input validation, transform pipeline,
logit-to-result conversion, and the public classify_hold / classify_holds API.

Tests follow TDD: written before implementation.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import PIL.Image as PILImage
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from src.inference.classification import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    ClassificationInferenceError,
    HoldTypeResult,
    _INPUT_SIZE_CACHE,
    VAL_RESIZE_RATIO,
    _build_model_from_metadata,
    _clear_model_cache,
    _get_inference_transform,
    _load_metadata,
    _load_model_cached,
    _logits_to_result,
    _to_pil_image,
    _validate_crop_input,
    classify_hold,
    classify_holds,
)
from src.inference.crop_extractor import HoldCrop
from src.inference.detection import DetectedHold


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cache() -> Iterator[None]:
    """Clear the model cache before and after every test."""
    _clear_model_cache()
    yield
    _clear_model_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_weights(tmp_path: Path) -> Path:
    """A non-empty .pt file in the expected artifact directory layout.

    Creates:
        tmp_path/v20260222_120000/weights/best.pt
    """
    version_dir = tmp_path / "v20260222_120000"
    weights_dir = version_dir / "weights"
    weights_dir.mkdir(parents=True)
    weights_path = weights_dir / "best.pt"
    weights_path.write_bytes(b"fake_weights_data")
    return weights_path


@pytest.fixture
def fake_metadata(fake_weights: Path) -> dict[str, Any]:
    """Create metadata.json and return the dict.

    The metadata uses dropout_rate=0.0 so the state_dict key paths are
    the same as the default model (no Dropout wrapper in the saved keys).
    """
    metadata: dict[str, Any] = {
        "version": "v20260222_120000",
        "architecture": "resnet18",
        "trained_at": "2026-02-22T12:00:00+00:00",
        "git_commit": None,
        "dataset_root": "/data/hold_classification",
        "hyperparameters": {
            "architecture": "resnet18",
            "pretrained": False,
            "num_classes": 6,
            "input_size": 224,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "scheduler": "StepLR",
            "label_smoothing": 0.1,
            "dropout_rate": 0.0,
        },
        "metrics": {
            "top1_accuracy": 0.9,
            "val_loss": 0.1,
            "ece": 0.05,
            "best_epoch": 0,
        },
    }
    meta_path = fake_weights.parent.parent / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return metadata


@pytest.fixture
def rgb_crop() -> PILImage.Image:
    """A 224×224 RGB PIL image for input tests."""
    return PILImage.new("RGB", (224, 224), color=(128, 64, 32))


@pytest.fixture
def hold_crop(rgb_crop: PILImage.Image) -> HoldCrop:
    """A HoldCrop wrapping rgb_crop with a DetectedHold at the center."""
    hold = DetectedHold(
        x_center=0.5,
        y_center=0.5,
        width=0.4,
        height=0.4,
        class_id=0,
        class_name="hold",
        confidence=0.9,
    )
    return HoldCrop(
        crop=rgb_crop,
        hold=hold,
        pixel_box=(56, 56, 168, 168),
    )


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock nn.Module whose forward returns batch-size-aware logits.

    The first class ("jug") always has the highest logit (2.0), so
    result.predicted_class == "jug" for any input batch size.
    """
    model = MagicMock(spec=nn.Module)

    def _forward(batch: torch.Tensor) -> torch.Tensor:
        n = batch.shape[0]
        base = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        return base.unsqueeze(0).expand(n, -1).clone()

    model.side_effect = _forward
    return model


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants have expected values."""

    def test_input_size_is_224(self) -> None:
        """INPUT_SIZE should be 224 to match the standard ImageNet resolution."""
        assert INPUT_SIZE == 224

    def test_imagenet_mean_is_tuple_of_three(self) -> None:
        """IMAGENET_MEAN should be a 3-element tuple."""
        assert isinstance(IMAGENET_MEAN, tuple)
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_is_tuple_of_three(self) -> None:
        """IMAGENET_STD should be a 3-element tuple."""
        assert isinstance(IMAGENET_STD, tuple)
        assert len(IMAGENET_STD) == 3

    def test_imagenet_mean_values(self) -> None:
        """IMAGENET_MEAN should match standard ImageNet statistics."""
        assert IMAGENET_MEAN == pytest.approx((0.485, 0.456, 0.406))

    def test_imagenet_std_values(self) -> None:
        """IMAGENET_STD should match standard ImageNet statistics."""
        assert IMAGENET_STD == pytest.approx((0.229, 0.224, 0.225))


# ---------------------------------------------------------------------------
# TestHoldTypeResult
# ---------------------------------------------------------------------------


class TestHoldTypeResult:
    """Tests for the HoldTypeResult Pydantic model."""

    def test_valid_construction(self) -> None:
        """HoldTypeResult should construct with all required fields."""
        result = HoldTypeResult(
            predicted_class="jug",
            confidence=0.92,
            probabilities={
                "jug": 0.92,
                "crimp": 0.05,
                "sloper": 0.01,
                "pinch": 0.01,
                "volume": 0.005,
                "unknown": 0.005,
            },
        )
        assert result.predicted_class == "jug"
        assert result.confidence == pytest.approx(0.92)
        assert result.source_crop is None

    def test_confidence_bounds_lower(self) -> None:
        """Confidence below 0.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            HoldTypeResult(
                predicted_class="crimp",
                confidence=-0.1,
                probabilities={},
            )

    def test_confidence_bounds_upper(self) -> None:
        """Confidence above 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            HoldTypeResult(
                predicted_class="crimp",
                confidence=1.1,
                probabilities={},
            )

    def test_source_crop_stored(self, hold_crop: HoldCrop) -> None:
        """source_crop should store the HoldCrop when provided."""
        result = HoldTypeResult(
            predicted_class="jug",
            confidence=0.9,
            probabilities={},
            source_crop=hold_crop,
        )
        assert result.source_crop is hold_crop

    def test_source_crop_defaults_none(self) -> None:
        """source_crop should default to None when not provided."""
        result = HoldTypeResult(
            predicted_class="sloper",
            confidence=0.5,
            probabilities={},
        )
        assert result.source_crop is None

    def test_probabilities_dict(self) -> None:
        """probabilities should store a dict of class name to float."""
        probs = {"jug": 0.8, "crimp": 0.2}
        result = HoldTypeResult(
            predicted_class="jug", confidence=0.8, probabilities=probs
        )
        assert result.probabilities["jug"] == pytest.approx(0.8)
        assert result.probabilities["crimp"] == pytest.approx(0.2)

    def test_confidence_zero_is_valid(self) -> None:
        """Confidence of exactly 0.0 should be valid."""
        result = HoldTypeResult(
            predicted_class="unknown", confidence=0.0, probabilities={}
        )
        assert result.confidence == pytest.approx(0.0)

    def test_confidence_one_is_valid(self) -> None:
        """Confidence of exactly 1.0 should be valid."""
        result = HoldTypeResult(predicted_class="jug", confidence=1.0, probabilities={})
        assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestValidateCropInput
# ---------------------------------------------------------------------------


class TestValidateCropInput:
    """Tests for _validate_crop_input."""

    def test_accepts_hold_crop(self, hold_crop: HoldCrop) -> None:
        """HoldCrop should pass validation without raising."""
        _validate_crop_input(hold_crop)  # should not raise

    def test_accepts_pil_image(self, rgb_crop: PILImage.Image) -> None:
        """PIL.Image.Image should pass validation without raising."""
        _validate_crop_input(rgb_crop)  # should not raise

    def test_rejects_ndarray(self) -> None:
        """numpy ndarray should raise TypeError."""
        import numpy as np  # pylint: disable=import-outside-toplevel

        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(TypeError, match="HoldCrop or PIL.Image.Image"):
            _validate_crop_input(arr)

    def test_rejects_string_path(self) -> None:
        """A string path should raise TypeError."""
        with pytest.raises(TypeError, match="HoldCrop or PIL.Image.Image"):
            _validate_crop_input("route.jpg")

    def test_rejects_none(self) -> None:
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="HoldCrop or PIL.Image.Image"):
            _validate_crop_input(None)

    def test_rejects_integer(self) -> None:
        """An integer should raise TypeError."""
        with pytest.raises(TypeError, match="HoldCrop or PIL.Image.Image"):
            _validate_crop_input(42)

    def test_error_message_includes_type_name(self) -> None:
        """TypeError message should include the actual type name."""
        with pytest.raises(TypeError, match="int"):
            _validate_crop_input(123)


# ---------------------------------------------------------------------------
# TestToPilImage
# ---------------------------------------------------------------------------


class TestToPilImage:
    """Tests for _to_pil_image."""

    def test_hold_crop_returns_crop_attribute(self, hold_crop: HoldCrop) -> None:
        """_to_pil_image should return the crop.crop PIL image from a HoldCrop."""
        result = _to_pil_image(hold_crop)
        assert result is hold_crop.crop

    def test_pil_image_passthrough(self, rgb_crop: PILImage.Image) -> None:
        """_to_pil_image should return the PIL image unchanged."""
        result = _to_pil_image(rgb_crop)
        assert result is rgb_crop

    def test_grayscale_converted_to_rgb(self) -> None:
        """A grayscale PIL image should be converted to RGB."""
        gray = PILImage.new("L", (224, 224), color=128)
        result = _to_pil_image(gray)
        assert result.mode == "RGB"

    def test_rgba_converted_to_rgb(self) -> None:
        """An RGBA PIL image should be converted to RGB."""
        rgba = PILImage.new("RGBA", (224, 224), color=(128, 128, 128, 255))
        result = _to_pil_image(rgba)
        assert result.mode == "RGB"

    def test_rgb_image_not_converted(self, rgb_crop: PILImage.Image) -> None:
        """An already-RGB image should not be converted (mode stays RGB)."""
        result = _to_pil_image(rgb_crop)
        assert result.mode == "RGB"

    def test_hold_crop_grayscale_converted(self) -> None:
        """HoldCrop with a grayscale crop should produce an RGB result."""
        gray_img = PILImage.new("L", (224, 224), color=100)
        hold = DetectedHold(
            x_center=0.5,
            y_center=0.5,
            width=0.4,
            height=0.4,
            class_id=0,
            class_name="hold",
            confidence=0.8,
        )
        crop = HoldCrop(crop=gray_img, hold=hold, pixel_box=(56, 56, 168, 168))
        result = _to_pil_image(crop)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# TestGetInferenceTransform
# ---------------------------------------------------------------------------


class TestGetInferenceTransform:
    """Tests for _get_inference_transform."""

    def test_returns_compose(self) -> None:
        """_get_inference_transform should return a transforms.Compose."""
        from torchvision import transforms  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

        result = _get_inference_transform()
        assert isinstance(result, transforms.Compose)

    def test_rgb_224_input_produces_correct_tensor_shape(
        self, rgb_crop: PILImage.Image
    ) -> None:
        """A 224×224 RGB image should produce a (3, 224, 224) tensor."""
        transform = _get_inference_transform()
        tensor = transform(rgb_crop)
        assert tensor.shape == (3, 224, 224)

    def test_larger_image_produces_correct_tensor_shape(self) -> None:
        """A larger image should be resized/cropped to (3, 224, 224)."""
        large = PILImage.new("RGB", (512, 512), color=(100, 150, 200))
        transform = _get_inference_transform()
        tensor = transform(large)
        assert tensor.shape == (3, 224, 224)

    def test_output_is_float_tensor(self, rgb_crop: PILImage.Image) -> None:
        """The transform output should be a float tensor."""
        transform = _get_inference_transform()
        tensor = transform(rgb_crop)
        assert tensor.dtype == torch.float32

    def test_output_is_normalized(self, rgb_crop: PILImage.Image) -> None:
        """After normalization, values can be negative (unlike raw pixels)."""
        transform = _get_inference_transform()
        # After ImageNet normalization a constant 128/255 ≈ 0.502 image produces
        # values near (0.502 - mean) / std; the blue channel mean is 0.406 so its
        # normalized value is positive, but the red channel mean is 0.485 which
        # is close to 0.502, however the green channel mean is 0.456 giving
        # (0.502-0.456)/0.224 ≈ +0.2 — the key property is that *some* channel
        # values go negative (red: (0.502-0.485)/0.229 ≈ +0.07 is positive, but
        # a slightly different constant image would produce negatives).  A simpler
        # and robust check: use a dark pixel (0/255 = 0.0); after normalization,
        # (0.0 - mean) / std is negative for all ImageNet channel means.
        dark = PILImage.new("RGB", rgb_crop.size, color=(0, 0, 0))
        dark_tensor = transform(dark)
        assert (
            dark_tensor.min().item() < 0
        )  # ImageNet-normalized zero pixel is negative

    def test_custom_input_size_produces_correct_tensor_shape(self) -> None:
        """Custom input_size should produce a (3, input_size, input_size) tensor."""
        custom_size = 192
        large = PILImage.new("RGB", (512, 512), color=(100, 150, 200))
        transform = _get_inference_transform(custom_size)
        tensor = transform(large)
        assert tensor.shape == (3, custom_size, custom_size)

    def test_resize_size_uses_val_resize_ratio(self) -> None:
        """Resize step should apply the standard 256/224 ratio to input_size."""
        expected_resize = int(224 * VAL_RESIZE_RATIO)
        assert expected_resize == 256  # sanity-check against legacy constant


# ---------------------------------------------------------------------------
# TestLogitsToResult
# ---------------------------------------------------------------------------


class TestLogitsToResult:
    """Tests for _logits_to_result."""

    def test_correct_argmax_prediction(self) -> None:
        """The class with the highest logit should be predicted."""
        # HOLD_CLASSES = ("jug", "crimp", "sloper", "pinch", "volume", "unknown")
        logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, None)
        assert result.predicted_class == "jug"  # index 0

    def test_probabilities_sum_to_one(self) -> None:
        """All probabilities should sum to approximately 1.0."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, None)
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_all_six_classes_in_probabilities(self) -> None:
        """probabilities dict should have exactly 6 entries."""
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = _logits_to_result(logits, None)
        from src.training.classification_dataset import HOLD_CLASSES  # pylint: disable=import-outside-toplevel

        assert set(result.probabilities.keys()) == set(HOLD_CLASSES)

    def test_confidence_matches_predicted_class_probability(self) -> None:
        """confidence should equal probabilities[predicted_class]."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, None)
        assert result.confidence == pytest.approx(
            result.probabilities[result.predicted_class], abs=1e-6
        )

    def test_source_crop_propagated_for_hold_crop(self, hold_crop: HoldCrop) -> None:
        """source_crop should be stored when input is a HoldCrop."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, hold_crop)
        assert result.source_crop is hold_crop

    def test_source_crop_none_for_pil_input(self, rgb_crop: PILImage.Image) -> None:
        """source_crop should be None when input is a raw PIL image."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, rgb_crop)
        assert result.source_crop is None

    def test_source_crop_none_when_none_passed(self) -> None:
        """source_crop should be None when None is passed explicitly."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, None)
        assert result.source_crop is None

    def test_second_class_predicted_correctly(self) -> None:
        """The second class (crimp) should be predicted when its logit is highest."""
        # Crimp is index 1
        logits = torch.tensor([0.1, 5.0, 0.5, 0.3, 0.2, 0.1])
        result = _logits_to_result(logits, None)
        assert result.predicted_class == "crimp"

    def test_last_class_predicted_correctly(self) -> None:
        """The last class (unknown) should be predicted when its logit is highest."""
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 10.0])
        result = _logits_to_result(logits, None)
        assert result.predicted_class == "unknown"


# ---------------------------------------------------------------------------
# TestLoadMetadata
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    """Tests for _load_metadata (reads metadata.json adjacent to weights)."""

    def test_happy_path_returns_dict(
        self,
        fake_weights: Path,
        fake_metadata: dict[str, Any],  # pylint: disable=unused-argument  # fixture used for side-effects (writes metadata.json)
    ) -> None:
        """_load_metadata should return the parsed dict when file exists."""
        result = _load_metadata(fake_weights)
        assert result["version"] == "v20260222_120000"
        assert "hyperparameters" in result

    def test_missing_metadata_raises_inference_error(self, tmp_path: Path) -> None:
        """_load_metadata should raise when metadata.json is absent."""
        version_dir = tmp_path / "v20260222_120000"
        weights_dir = version_dir / "weights"
        weights_dir.mkdir(parents=True)
        weights_path = weights_dir / "best.pt"
        weights_path.write_bytes(b"fake")
        # No metadata.json created
        with pytest.raises(
            ClassificationInferenceError, match="metadata.json not found"
        ):
            _load_metadata(weights_path)

    def test_invalid_json_raises_inference_error(self, fake_weights: Path) -> None:
        """_load_metadata should raise when metadata.json contains invalid JSON."""
        meta_path = fake_weights.parent.parent / "metadata.json"
        meta_path.write_text("{ invalid json }", encoding="utf-8")
        with pytest.raises(
            ClassificationInferenceError, match="Failed to parse metadata.json"
        ):
            _load_metadata(fake_weights)

    def test_resolves_correct_path(
        self,
        fake_weights: Path,
        fake_metadata: dict[str, Any],  # pylint: disable=unused-argument  # fixture used for side-effects (writes metadata.json)
    ) -> None:
        """metadata.json should be resolved as weights_path.parent.parent / 'metadata.json'."""
        expected_meta_path = fake_weights.parent.parent / "metadata.json"
        assert expected_meta_path.exists()
        result = _load_metadata(fake_weights)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestBuildModelFromMetadata
# ---------------------------------------------------------------------------


class TestBuildModelFromMetadata:
    """Tests for _build_model_from_metadata (reconstructs model from metadata)."""

    def test_missing_weights_raises_inference_error(
        self, fake_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """_build_model_from_metadata should raise when weights file is absent."""
        missing = tmp_path / "nonexistent.pt"
        with pytest.raises(
            ClassificationInferenceError, match="Model weights not found"
        ):
            _build_model_from_metadata(fake_metadata, missing)

    def test_missing_hyperparameters_key_raises_inference_error(
        self, fake_weights: Path
    ) -> None:
        """_build_model_from_metadata should raise when 'hyperparameters' key is absent."""
        no_hp_metadata: dict[str, Any] = {"version": "v1", "architecture": "resnet18"}
        with pytest.raises(
            ClassificationInferenceError,
            match="missing the required 'hyperparameters' key",
        ):
            _build_model_from_metadata(no_hp_metadata, fake_weights)

    def test_invalid_hyperparameters_raises_inference_error(
        self, fake_weights: Path
    ) -> None:
        """_build_model_from_metadata should raise for malformed hyperparameters."""
        bad_metadata: dict[str, Any] = {
            "hyperparameters": {"architecture": "invalid_arch"}
        }
        with pytest.raises(
            ClassificationInferenceError, match="Invalid hyperparameters"
        ):
            _build_model_from_metadata(bad_metadata, fake_weights)

    @patch("src.inference.classification.torch.load")
    def test_torch_load_failure_raises_inference_error(
        self,
        mock_torch_load: MagicMock,
        fake_weights: Path,
        fake_metadata: dict[str, Any],
    ) -> None:
        """_build_model_from_metadata should raise when torch.load fails."""
        mock_torch_load.side_effect = RuntimeError("corrupt file")
        with pytest.raises(
            ClassificationInferenceError, match="Failed to load state_dict"
        ):
            _build_model_from_metadata(fake_metadata, fake_weights)


# ---------------------------------------------------------------------------
# TestLoadModelCached
# ---------------------------------------------------------------------------


class TestLoadModelCached:
    """Tests for _load_model_cached caching behaviour."""

    @patch("src.inference.classification._build_model_from_metadata")
    @patch("src.inference.classification._load_metadata")
    def test_first_call_loads_model(
        self,
        mock_meta: MagicMock,
        mock_build: MagicMock,
        fake_weights: Path,
    ) -> None:
        """First call should invoke _build_model_from_metadata exactly once."""
        mock_meta.return_value = {"hyperparameters": {}}
        mock_model = MagicMock(spec=nn.Module)
        mock_build.return_value = mock_model

        result = _load_model_cached(fake_weights)

        assert result is mock_model
        mock_build.assert_called_once()

    @patch("src.inference.classification._build_model_from_metadata")
    @patch("src.inference.classification._load_metadata")
    def test_second_call_returns_same_instance(
        self,
        mock_meta: MagicMock,
        mock_build: MagicMock,
        fake_weights: Path,
    ) -> None:
        """Repeated calls with the same path should return the cached model."""
        mock_meta.return_value = {"hyperparameters": {}}
        mock_model = MagicMock(spec=nn.Module)
        mock_build.return_value = mock_model

        result1 = _load_model_cached(fake_weights)
        result2 = _load_model_cached(fake_weights)

        assert result1 is result2
        # _build_model_from_metadata should only be called once
        mock_build.assert_called_once()

    def test_missing_weights_raises_inference_error(self, tmp_path: Path) -> None:
        """A path to a non-existent weights file should raise ClassificationInferenceError."""
        missing = tmp_path / "nonexistent" / "weights" / "best.pt"
        with pytest.raises(ClassificationInferenceError):
            _load_model_cached(missing)

    @patch("src.inference.classification._build_model_from_metadata")
    @patch("src.inference.classification._load_metadata")
    def test_str_and_path_coercion_share_cache(
        self,
        mock_meta: MagicMock,
        mock_build: MagicMock,
        fake_weights: Path,
    ) -> None:
        """str and Path inputs for the same weights file should share the same cached entry."""
        mock_meta.return_value = {"hyperparameters": {}}
        mock_model = MagicMock(spec=nn.Module)
        mock_build.return_value = mock_model

        result1 = _load_model_cached(str(fake_weights))
        result2 = _load_model_cached(fake_weights)

        assert result1 is result2
        mock_build.assert_called_once()

    @patch("src.inference.classification._build_model_from_metadata")
    @patch("src.inference.classification._load_metadata")
    def test_concurrent_loads_call_loader_only_once(
        self,
        mock_load_meta: MagicMock,
        mock_build: MagicMock,
        fake_weights: Path,
        fake_metadata: dict,
    ) -> None:
        """Concurrent calls to _load_model_cached load the model exactly once."""
        import concurrent.futures

        mock_load_meta.return_value = fake_metadata
        mock_model = MagicMock(spec=nn.Module)
        mock_model.eval.return_value = mock_model
        mock_build.return_value = mock_model

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(_load_model_cached, fake_weights) for _ in range(10)
            ]
            models = [f.result() for f in concurrent.futures.as_completed(futures)]

        mock_build.assert_called_once()
        first = models[0]
        assert all(m is first for m in models)


# ---------------------------------------------------------------------------
# TestClearModelCache
# ---------------------------------------------------------------------------


class TestClearModelCache:
    """Tests for _clear_model_cache."""

    @patch("src.inference.classification._build_model_from_metadata")
    @patch("src.inference.classification._load_metadata")
    def test_clear_forces_reload(
        self,
        mock_meta: MagicMock,
        mock_build: MagicMock,
        fake_weights: Path,
    ) -> None:
        """After clearing, _load_model_cached should call _build_model_from_metadata again."""
        mock_meta.return_value = {"hyperparameters": {}}
        mock_model1 = MagicMock(spec=nn.Module)
        mock_model2 = MagicMock(spec=nn.Module)
        mock_build.side_effect = [mock_model1, mock_model2]

        result1 = _load_model_cached(fake_weights)
        _clear_model_cache()
        result2 = _load_model_cached(fake_weights)

        assert result1 is mock_model1
        assert result2 is mock_model2
        assert mock_build.call_count == 2

    @patch("src.inference.classification._build_model_from_metadata")
    @patch("src.inference.classification._load_metadata")
    def test_clear_also_removes_input_size_cache(
        self,
        mock_meta: MagicMock,
        mock_build: MagicMock,
        fake_weights: Path,
    ) -> None:
        """Clearing the model cache should also clear _INPUT_SIZE_CACHE."""
        mock_meta.return_value = {"hyperparameters": {"input_size": 224}}
        mock_build.return_value = MagicMock(spec=nn.Module)

        _load_model_cached(fake_weights)
        resolved = str(fake_weights.resolve())
        assert resolved in _INPUT_SIZE_CACHE

        _clear_model_cache()
        assert resolved not in _INPUT_SIZE_CACHE

    def test_clear_empty_cache_is_safe(self) -> None:
        """Clearing an already-empty cache should not raise."""
        _clear_model_cache()  # should not raise
        _clear_model_cache()  # second clear also safe


# ---------------------------------------------------------------------------
# TestClassifyHold
# ---------------------------------------------------------------------------


class TestClassifyHold:
    """Tests for the public classify_hold function."""

    @patch("src.inference.classification._load_model_cached")
    def test_happy_path_hold_crop(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        hold_crop: HoldCrop,
        fake_weights: Path,
    ) -> None:
        """classify_hold with a HoldCrop should return correct prediction."""
        mock_load.return_value = mock_model

        result = classify_hold(hold_crop, fake_weights)

        assert result.predicted_class == "jug"
        assert 0.0 <= result.confidence <= 1.0
        assert result.source_crop is hold_crop

    @patch("src.inference.classification._load_model_cached")
    def test_happy_path_pil_image(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        rgb_crop: PILImage.Image,
        fake_weights: Path,
    ) -> None:
        """classify_hold with a PIL image should return None source_crop."""
        mock_load.return_value = mock_model

        result = classify_hold(rgb_crop, fake_weights)

        assert result.predicted_class == "jug"
        assert result.source_crop is None

    def test_wrong_input_type_raises_type_error(self, fake_weights: Path) -> None:
        """Passing a non-supported type should raise TypeError."""
        with pytest.raises(TypeError, match="HoldCrop or PIL.Image.Image"):
            classify_hold("not_a_crop", fake_weights)  # type: ignore[arg-type]

    def test_missing_weights_raises_inference_error(
        self, hold_crop: HoldCrop, tmp_path: Path
    ) -> None:
        """Missing weights file should raise ClassificationInferenceError."""
        missing = tmp_path / "nonexistent" / "weights" / "best.pt"
        with pytest.raises(ClassificationInferenceError):
            classify_hold(hold_crop, missing)

    @patch("src.inference.classification._load_model_cached")
    def test_probabilities_have_all_six_classes(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        hold_crop: HoldCrop,
        fake_weights: Path,
    ) -> None:
        """probabilities should contain all 6 hold class keys."""
        from src.training.classification_dataset import HOLD_CLASSES  # pylint: disable=import-outside-toplevel

        mock_load.return_value = mock_model
        result = classify_hold(hold_crop, fake_weights)

        assert set(result.probabilities.keys()) == set(HOLD_CLASSES)

    @patch("src.inference.classification._load_model_cached")
    def test_probabilities_sum_to_one(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        rgb_crop: PILImage.Image,
        fake_weights: Path,
    ) -> None:
        """Probabilities should sum to approximately 1.0."""
        mock_load.return_value = mock_model
        result = classify_hold(rgb_crop, fake_weights)

        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    @patch("src.inference.classification._load_model_cached")
    def test_model_exception_wrapped_as_inference_error(
        self,
        mock_load: MagicMock,
        fake_weights: Path,
        hold_crop: HoldCrop,
    ) -> None:
        """Exceptions from the model forward pass should become ClassificationInferenceError."""
        model = MagicMock(spec=nn.Module)
        model.side_effect = RuntimeError("CUDA out of memory")
        mock_load.return_value = model

        with pytest.raises(
            ClassificationInferenceError, match="Hold classification failed"
        ):
            classify_hold(hold_crop, fake_weights)

    @patch("src.inference.classification._to_pil_image")
    @patch("src.inference.classification._load_model_cached")
    def test_type_error_inside_try_is_reraised(
        self,
        mock_load: MagicMock,
        mock_pil: MagicMock,
        mock_model: MagicMock,
        fake_weights: Path,
        hold_crop: HoldCrop,
    ) -> None:
        """TypeError raised inside the try block should propagate as TypeError (not wrapped)."""
        mock_load.return_value = mock_model
        mock_pil.side_effect = TypeError("unexpected type")

        with pytest.raises(TypeError, match="unexpected type"):
            classify_hold(hold_crop, fake_weights)


# ---------------------------------------------------------------------------
# TestClassifyHolds
# ---------------------------------------------------------------------------


class TestClassifyHolds:
    """Tests for the public classify_holds batch function."""

    @patch("src.inference.classification._load_model_cached")
    def test_happy_path_batch(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        hold_crop: HoldCrop,
        rgb_crop: PILImage.Image,
        fake_weights: Path,
    ) -> None:
        """classify_holds with 2 crops should return 2 results in order."""
        mock_load.return_value = mock_model
        crops: list[HoldCrop | PILImage.Image] = [hold_crop, rgb_crop]

        results = classify_holds(crops, fake_weights)

        assert len(results) == 2
        assert results[0].predicted_class == "jug"
        assert results[1].predicted_class == "jug"

    def test_empty_list_raises_value_error(self, fake_weights: Path) -> None:
        """Empty crops list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            classify_holds([], fake_weights)

    def test_wrong_input_type_raises_type_error(self, fake_weights: Path) -> None:
        """A non-supported crop type in the list should raise TypeError."""
        with pytest.raises(TypeError, match="HoldCrop or PIL.Image.Image"):
            classify_holds(["not_a_crop"], fake_weights)  # type: ignore[list-item]

    @patch("src.inference.classification._load_model_cached")
    def test_order_preserved(
        self,
        mock_load: MagicMock,
        fake_weights: Path,
    ) -> None:
        """Results should match input order: different logits per position."""
        model = MagicMock(spec=nn.Module)

        def _order_sensitive(batch: torch.Tensor) -> torch.Tensor:
            n = batch.shape[0]
            # Each position gets a unique argmax: 0,1,2,...
            out = torch.zeros(n, 6)
            for i in range(n):
                # Set class i as the highest for sample i
                out[i, i % 6] = 10.0
            return out

        model.side_effect = _order_sensitive
        mock_load.return_value = model

        from src.training.classification_dataset import HOLD_CLASSES  # pylint: disable=import-outside-toplevel

        crops: list[HoldCrop | PILImage.Image] = [
            PILImage.new("RGB", (224, 224)) for _ in range(6)
        ]
        results = classify_holds(crops, fake_weights)

        assert len(results) == 6
        for i, result in enumerate(results):
            assert result.predicted_class == HOLD_CLASSES[i]

    @patch("src.inference.classification._load_model_cached")
    def test_source_crop_per_result(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        hold_crop: HoldCrop,
        rgb_crop: PILImage.Image,
        fake_weights: Path,
    ) -> None:
        """HoldCrop inputs should have source_crop set; PIL inputs should have None."""
        mock_load.return_value = mock_model
        results = classify_holds([hold_crop, rgb_crop], fake_weights)

        assert results[0].source_crop is hold_crop
        assert results[1].source_crop is None

    def test_missing_weights_raises_inference_error(
        self, rgb_crop: PILImage.Image, tmp_path: Path
    ) -> None:
        """Missing weights file should raise ClassificationInferenceError."""
        missing = tmp_path / "nonexistent" / "weights" / "best.pt"
        with pytest.raises(ClassificationInferenceError):
            classify_holds([rgb_crop], missing)

    @patch("src.inference.classification._to_pil_image")
    @patch("src.inference.classification._load_model_cached")
    def test_type_error_inside_try_is_reraised(
        self,
        mock_load: MagicMock,
        mock_pil: MagicMock,
        mock_model: MagicMock,
        rgb_crop: PILImage.Image,
        fake_weights: Path,
    ) -> None:
        """TypeError raised inside the try block should propagate as TypeError (not wrapped)."""
        mock_load.return_value = mock_model
        mock_pil.side_effect = TypeError("unexpected type in batch")

        with pytest.raises(TypeError, match="unexpected type in batch"):
            classify_holds([rgb_crop], fake_weights)

    @patch("src.inference.classification._load_model_cached")
    def test_batch_exception_wrapped_as_inference_error(
        self,
        mock_load: MagicMock,
        rgb_crop: PILImage.Image,
        fake_weights: Path,
    ) -> None:
        """General exceptions from batch forward pass become ClassificationInferenceError."""
        model = MagicMock(spec=nn.Module)
        model.side_effect = RuntimeError("batch inference failed")
        mock_load.return_value = model

        with pytest.raises(
            ClassificationInferenceError, match="Hold classification failed"
        ):
            classify_holds([rgb_crop], fake_weights)

    @patch("src.inference.classification._load_model_cached")
    def test_chunk_size_produces_same_results(
        self,
        mock_load: MagicMock,
        mock_model: MagicMock,
        fake_weights: Path,
    ) -> None:
        """classify_holds with chunk_size should return same results as without."""
        mock_load.return_value = mock_model
        crops: list[HoldCrop | PILImage.Image] = [
            PILImage.new("RGB", (224, 224)) for _ in range(6)
        ]

        results_no_chunk = classify_holds(crops, fake_weights)
        results_chunked = classify_holds(crops, fake_weights, chunk_size=2)

        assert len(results_no_chunk) == len(results_chunked) == 6
        for r1, r2 in zip(results_no_chunk, results_chunked):
            assert r1.predicted_class == r2.predicted_class

    def test_chunk_size_less_than_one_raises_value_error(
        self, fake_weights: Path
    ) -> None:
        """chunk_size=0 should raise ValueError."""
        crops: list[HoldCrop | PILImage.Image] = [PILImage.new("RGB", (224, 224))]
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            classify_holds(crops, fake_weights, chunk_size=0)
