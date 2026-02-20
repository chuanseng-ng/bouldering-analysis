"""Tests for the hold detection inference module.

This module tests real-time inference using trained YOLOv8 weights,
including model caching, input validation, result parsing, and the
public detect_holds / detect_holds_batch API.

Tests follow TDD: written before implementation.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image as PILImage
import pytest
import torch
from pydantic import ValidationError

from src.inference.detection import (
    CLASS_NAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DetectedHold,
    InferenceError,
    _clear_model_cache,
    _load_model_cached,
    _parse_yolo_results,
    _validate_image_input,
    detect_holds,
    detect_holds_batch,
)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cache() -> object:
    """Clear the model cache before and after every test."""
    _clear_model_cache()
    yield
    _clear_model_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_weights(tmp_path: Path) -> Path:
    """A non-empty .pt file that satisfies path-existence checks."""
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake_weights")
    return weights


@pytest.fixture
def fake_image_file(tmp_path: Path) -> Path:
    """A minimal JPEG file for path-based input tests."""
    img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
    path = tmp_path / "route.jpg"
    img.save(path)
    return path


@pytest.fixture
def mock_yolo_predict() -> MagicMock:
    """Mock YOLO model whose predict() returns one detection."""
    results = MagicMock()
    boxes = MagicMock()
    boxes.xywhn = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    boxes.cls = torch.tensor([0.0])
    boxes.conf = torch.tensor([0.87])
    results.boxes = boxes

    model = MagicMock()
    model.predict.return_value = [results]
    return model


@pytest.fixture
def mock_yolo_no_detections() -> MagicMock:
    """Mock YOLO model whose predict() returns zero detections."""
    results = MagicMock()
    boxes = MagicMock()
    boxes.xywhn = torch.zeros((0, 4))
    boxes.cls = torch.zeros(0)
    boxes.conf = torch.zeros(0)
    results.boxes = boxes

    model = MagicMock()
    model.predict.return_value = [results]
    return model


# ---------------------------------------------------------------------------
# TestDetectedHold
# ---------------------------------------------------------------------------


class TestDetectedHold:
    """Tests for DetectedHold Pydantic model."""

    def test_valid_hold_construction(self) -> None:
        """DetectedHold accepts all valid fields."""
        hold = DetectedHold(
            x_center=0.5,
            y_center=0.5,
            width=0.2,
            height=0.3,
            class_id=0,
            class_name="hold",
            confidence=0.87,
        )
        assert hold.x_center == pytest.approx(0.5)
        assert hold.class_name == "hold"
        assert hold.confidence == pytest.approx(0.87)

    def test_valid_volume_construction(self) -> None:
        """DetectedHold accepts class_id=1 and class_name='volume'."""
        hold = DetectedHold(
            x_center=0.3,
            y_center=0.7,
            width=0.4,
            height=0.4,
            class_id=1,
            class_name="volume",
            confidence=0.65,
        )
        assert hold.class_name == "volume"
        assert hold.class_id == 1

    def test_rejects_coordinates_out_of_bounds(self) -> None:
        """DetectedHold rejects coordinates outside [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            DetectedHold(
                x_center=1.5,
                y_center=0.5,
                width=0.2,
                height=0.3,
                class_id=0,
                class_name="hold",
                confidence=0.87,
            )

    def test_rejects_confidence_out_of_bounds(self) -> None:
        """DetectedHold rejects confidence outside [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            DetectedHold(
                x_center=0.5,
                y_center=0.5,
                width=0.2,
                height=0.3,
                class_id=0,
                class_name="hold",
                confidence=1.1,
            )

    def test_rejects_invalid_class_name(self) -> None:
        """DetectedHold rejects class_name not in ('hold', 'volume')."""
        with pytest.raises(ValidationError):
            DetectedHold(
                x_center=0.5,
                y_center=0.5,
                width=0.2,
                height=0.3,
                class_id=0,
                class_name="crimp",  # type: ignore[arg-type]
                confidence=0.5,
            )

    def test_rejects_invalid_class_id(self) -> None:
        """DetectedHold rejects class_id outside [0, 1]."""
        with pytest.raises(ValidationError):
            DetectedHold(
                x_center=0.5,
                y_center=0.5,
                width=0.2,
                height=0.3,
                class_id=5,
                class_name="hold",
                confidence=0.5,
            )


# ---------------------------------------------------------------------------
# TestLoadModelCached
# ---------------------------------------------------------------------------


class TestLoadModelCached:
    """Tests for _load_model_cached helper."""

    @patch("src.inference.detection.YOLO")
    def test_loads_model_on_first_call(
        self, mock_yolo_cls: MagicMock, fake_weights: Path
    ) -> None:
        """_load_model_cached calls YOLO() on the first invocation."""
        _load_model_cached(fake_weights)
        mock_yolo_cls.assert_called_once()

    @patch("src.inference.detection.YOLO")
    def test_returns_cached_model_on_second_call(
        self, mock_yolo_cls: MagicMock, fake_weights: Path
    ) -> None:
        """_load_model_cached calls YOLO() only once across multiple invocations."""
        m1 = _load_model_cached(fake_weights)
        m2 = _load_model_cached(fake_weights)
        mock_yolo_cls.assert_called_once()
        assert m1 is m2

    def test_raises_inference_error_for_missing_file(self, tmp_path: Path) -> None:
        """_load_model_cached raises InferenceError when weights file is absent."""
        missing = tmp_path / "no_model.pt"
        with pytest.raises(InferenceError):
            _load_model_cached(missing)


# ---------------------------------------------------------------------------
# TestClearModelCache
# ---------------------------------------------------------------------------


class TestClearModelCache:
    """Tests for _clear_model_cache helper."""

    @patch("src.inference.detection.YOLO")
    def test_reload_after_clear(
        self, mock_yolo_cls: MagicMock, fake_weights: Path
    ) -> None:
        """After _clear_model_cache(), the next call to _load_model_cached reloads."""
        _load_model_cached(fake_weights)
        _clear_model_cache()
        _load_model_cached(fake_weights)
        assert mock_yolo_cls.call_count == 2


# ---------------------------------------------------------------------------
# TestValidateImageInput
# ---------------------------------------------------------------------------


class TestValidateImageInput:
    """Tests for _validate_image_input helper."""

    def test_accepts_path(self, fake_image_file: Path) -> None:
        """_validate_image_input accepts a Path to an existing file."""
        _validate_image_input(fake_image_file)  # should not raise

    def test_accepts_str_path(self, fake_image_file: Path) -> None:
        """_validate_image_input accepts a str path to an existing file."""
        _validate_image_input(str(fake_image_file))  # should not raise

    def test_accepts_pil_image(self) -> None:
        """_validate_image_input accepts a PIL.Image.Image."""
        img = PILImage.new("RGB", (32, 32))
        _validate_image_input(img)  # should not raise

    def test_accepts_numpy_array(self) -> None:
        """_validate_image_input accepts an np.ndarray."""
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        _validate_image_input(arr)  # should not raise

    def test_raises_type_error_for_invalid_type(self) -> None:
        """_validate_image_input raises TypeError for unsupported types."""
        with pytest.raises(TypeError):
            _validate_image_input(12345)  # type: ignore[arg-type]

    def test_raises_inference_error_for_nonexistent_path(self, tmp_path: Path) -> None:
        """_validate_image_input raises InferenceError for a missing file path."""
        with pytest.raises(InferenceError):
            _validate_image_input(tmp_path / "no_image.jpg")

    def test_raises_inference_error_for_nonexistent_str_path(
        self, tmp_path: Path
    ) -> None:
        """_validate_image_input raises InferenceError for a missing str path."""
        with pytest.raises(InferenceError):
            _validate_image_input(str(tmp_path / "no_image.jpg"))


# ---------------------------------------------------------------------------
# TestParseYoloResults
# ---------------------------------------------------------------------------


class TestParseYoloResults:
    """Tests for _parse_yolo_results helper."""

    def test_extracts_single_detection(self) -> None:
        """_parse_yolo_results returns one DetectedHold from one box."""
        results = MagicMock()
        boxes = MagicMock()
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        boxes.cls = torch.tensor([0.0])
        boxes.conf = torch.tensor([0.87])
        results.boxes = boxes

        holds = _parse_yolo_results([results], conf_threshold=0.25)
        assert len(holds) == 1
        assert holds[0].x_center == pytest.approx(0.5)
        assert holds[0].class_name == "hold"
        assert holds[0].confidence == pytest.approx(0.87)

    def test_filters_below_confidence_threshold(self) -> None:
        """_parse_yolo_results excludes detections below conf_threshold."""
        results = MagicMock()
        boxes = MagicMock()
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        boxes.cls = torch.tensor([0.0])
        boxes.conf = torch.tensor([0.15])  # below default 0.25
        results.boxes = boxes

        holds = _parse_yolo_results([results], conf_threshold=0.25)
        assert len(holds) == 0

    def test_sorts_by_confidence_descending(self) -> None:
        """_parse_yolo_results returns detections sorted by confidence descending."""
        results = MagicMock()
        boxes = MagicMock()
        boxes.xywhn = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]])
        boxes.cls = torch.tensor([0.0, 1.0])
        boxes.conf = torch.tensor([0.60, 0.90])
        results.boxes = boxes

        holds = _parse_yolo_results([results], conf_threshold=0.25)
        assert len(holds) == 2
        assert holds[0].confidence > holds[1].confidence

    def test_returns_empty_list_for_no_detections(self) -> None:
        """_parse_yolo_results returns [] when there are no boxes."""
        results = MagicMock()
        boxes = MagicMock()
        boxes.xywhn = torch.zeros((0, 4))
        boxes.cls = torch.zeros(0)
        boxes.conf = torch.zeros(0)
        results.boxes = boxes

        holds = _parse_yolo_results([results], conf_threshold=0.25)
        assert holds == []

    def test_maps_class_id_to_class_name(self) -> None:
        """_parse_yolo_results correctly maps class_id 1 to 'volume'."""
        results = MagicMock()
        boxes = MagicMock()
        boxes.xywhn = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        boxes.cls = torch.tensor([1.0])
        boxes.conf = torch.tensor([0.75])
        results.boxes = boxes

        holds = _parse_yolo_results([results], conf_threshold=0.25)
        assert holds[0].class_name == "volume"
        assert holds[0].class_id == 1


# ---------------------------------------------------------------------------
# TestDetectHolds
# ---------------------------------------------------------------------------


class TestDetectHolds:
    """Tests for detect_holds public function."""

    @patch("src.inference.detection._load_model_cached")
    def test_returns_list_of_detected_holds(
        self,
        mock_load: MagicMock,
        mock_yolo_predict: MagicMock,
        fake_image_file: Path,
        fake_weights: Path,
    ) -> None:
        """detect_holds returns a list of DetectedHold on success."""
        mock_load.return_value = mock_yolo_predict

        holds = detect_holds(fake_image_file, fake_weights)

        assert isinstance(holds, list)
        assert len(holds) == 1
        assert isinstance(holds[0], DetectedHold)

    @patch("src.inference.detection._load_model_cached")
    def test_returns_empty_list_for_no_detections(
        self,
        mock_load: MagicMock,
        mock_yolo_no_detections: MagicMock,
        fake_image_file: Path,
        fake_weights: Path,
    ) -> None:
        """detect_holds returns [] when the model finds no detections."""
        mock_load.return_value = mock_yolo_no_detections

        holds = detect_holds(fake_image_file, fake_weights)

        assert holds == []

    def test_raises_inference_error_for_missing_weights(
        self, fake_image_file: Path, tmp_path: Path
    ) -> None:
        """detect_holds raises InferenceError when weights file does not exist."""
        missing_weights = tmp_path / "no_model.pt"
        with pytest.raises(InferenceError):
            detect_holds(fake_image_file, missing_weights)

    @patch("src.inference.detection._load_model_cached")
    def test_passes_thresholds_to_predict(
        self,
        mock_load: MagicMock,
        mock_yolo_predict: MagicMock,
        fake_image_file: Path,
        fake_weights: Path,
    ) -> None:
        """detect_holds forwards conf and iou thresholds to model.predict()."""
        mock_load.return_value = mock_yolo_predict

        detect_holds(
            fake_image_file, fake_weights, conf_threshold=0.5, iou_threshold=0.6
        )

        call_kwargs = mock_yolo_predict.predict.call_args[1]
        assert call_kwargs["conf"] == pytest.approx(0.5)
        assert call_kwargs["iou"] == pytest.approx(0.6)

    @patch("src.inference.detection._load_model_cached")
    def test_accepts_pil_image_input(
        self,
        mock_load: MagicMock,
        mock_yolo_predict: MagicMock,
        fake_weights: Path,
    ) -> None:
        """detect_holds accepts a PIL.Image as image input."""
        mock_load.return_value = mock_yolo_predict
        img = PILImage.new("RGB", (64, 64))

        holds = detect_holds(img, fake_weights)
        assert isinstance(holds, list)

    @patch("src.inference.detection._load_model_cached")
    def test_accepts_numpy_array_input(
        self,
        mock_load: MagicMock,
        mock_yolo_predict: MagicMock,
        fake_weights: Path,
    ) -> None:
        """detect_holds accepts an np.ndarray as image input."""
        mock_load.return_value = mock_yolo_predict
        arr = np.zeros((64, 64, 3), dtype=np.uint8)

        holds = detect_holds(arr, fake_weights)
        assert isinstance(holds, list)


# ---------------------------------------------------------------------------
# TestDetectHoldsBatch
# ---------------------------------------------------------------------------


class TestDetectHoldsBatch:
    """Tests for detect_holds_batch public function."""

    @patch("src.inference.detection._load_model_cached")
    def test_result_count_matches_input(
        self,
        mock_load: MagicMock,
        mock_yolo_predict: MagicMock,
        fake_image_file: Path,
        fake_weights: Path,
    ) -> None:
        """detect_holds_batch returns one list per input image."""
        mock_load.return_value = mock_yolo_predict

        results = detect_holds_batch(
            [fake_image_file, fake_image_file, fake_image_file], fake_weights
        )

        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)

    def test_raises_value_error_for_empty_list(self, fake_weights: Path) -> None:
        """detect_holds_batch raises ValueError when images list is empty."""
        with pytest.raises(ValueError, match="empty"):
            detect_holds_batch([], fake_weights)

    @patch("src.inference.detection._load_model_cached")
    def test_each_result_is_list_of_detected_holds(
        self,
        mock_load: MagicMock,
        mock_yolo_predict: MagicMock,
        fake_image_file: Path,
        fake_weights: Path,
    ) -> None:
        """detect_holds_batch: each sub-list contains DetectedHold instances."""
        mock_load.return_value = mock_yolo_predict

        results = detect_holds_batch([fake_image_file], fake_weights)

        assert len(results[0]) == 1
        assert isinstance(results[0][0], DetectedHold)


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_class_names(self) -> None:
        """CLASS_NAMES contains exactly ['hold', 'volume']."""
        assert CLASS_NAMES == ["hold", "volume"]

    def test_default_conf_threshold(self) -> None:
        """DEFAULT_CONF_THRESHOLD is between 0.0 and 1.0."""
        assert 0.0 < DEFAULT_CONF_THRESHOLD < 1.0

    def test_default_iou_threshold(self) -> None:
        """DEFAULT_IOU_THRESHOLD is between 0.0 and 1.0."""
        assert 0.0 < DEFAULT_IOU_THRESHOLD < 1.0
