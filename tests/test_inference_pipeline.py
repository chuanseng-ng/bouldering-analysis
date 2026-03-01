"""Integration tests for the detect -> crop -> classify inference pipeline.

These tests verify the three inference modules work correctly together,
exercising real coordinate transformations and type handoffs between modules.
Models are mocked so no actual .pt weights are required.

Tested pipeline:
    detect_holds(image, weights) -> [DetectedHold, ...]
    extract_hold_crops(image, holds) -> [HoldCrop, ...]
    classify_holds(crops, weights) -> [HoldTypeResult, ...]
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import PIL.Image as PILImage
import pytest
import torch

from src.inference.classification import (
    INPUT_SIZE,
    classify_holds,
    reset_classification_model_cache,
)
from src.inference.crop_extractor import TARGET_SIZE, HoldCrop, extract_hold_crops
from src.inference.detection import (
    detect_holds,
    reset_detection_model_cache,
)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_caches() -> Iterator[None]:
    """Clear both inference model caches before and after every test."""
    reset_detection_model_cache()
    reset_classification_model_cache()
    yield
    reset_detection_model_cache()
    reset_classification_model_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rgb_image() -> PILImage.Image:
    """A 400x300 RGB test image."""
    return PILImage.new("RGB", (400, 300), color=(100, 150, 200))


@pytest.fixture
def detection_weights(tmp_path: Path) -> Path:
    """Fake detection weights file."""
    p = tmp_path / "detection.pt"
    p.write_bytes(b"fake_detection_weights")
    return p


@pytest.fixture
def classification_weights(tmp_path: Path) -> Path:
    """Fake classification weights in expected artifact layout."""
    version_dir = tmp_path / "v20260224_120000"
    weights_dir = version_dir / "weights"
    weights_dir.mkdir(parents=True)
    p = weights_dir / "best.pt"
    p.write_bytes(b"fake_classification_weights")
    return p


@pytest.fixture
def mock_yolo_model() -> MagicMock:
    """Mock YOLO model returning two detections: one hold and one volume."""
    results = MagicMock()
    boxes = MagicMock()
    # Two detections centred well inside a 400x300 image
    boxes.xywhn = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.2, 0.3, 0.2, 0.3]])
    boxes.cls = torch.tensor([0.0, 1.0])
    boxes.conf = torch.tensor([0.90, 0.75])
    results.boxes = boxes

    model = MagicMock()
    model.predict.return_value = [results]
    return model


@pytest.fixture
def mock_classifier_model() -> MagicMock:
    """Mock classifier returning 'jug' (index 0) for every crop."""
    model = MagicMock()

    def _predict(batch: torch.Tensor) -> torch.Tensor:
        n = batch.shape[0]
        logits = torch.zeros(n, 6)
        logits[:, 0] = 10.0  # jug wins for all
        return logits

    model.side_effect = _predict
    return model


# ---------------------------------------------------------------------------
# TestDetectToCrop
# ---------------------------------------------------------------------------


class TestDetectToCrop:
    """Tests for the detect_holds -> extract_hold_crops handoff."""

    @patch("src.inference.detection._load_model_cached")
    def test_detected_holds_produce_valid_crops(
        self,
        mock_load: MagicMock,
        mock_yolo_model: MagicMock,
        rgb_image: PILImage.Image,
        detection_weights: Path,
    ) -> None:
        """Holds from detect_holds() should produce 224x224 crops via extract_hold_crops."""
        mock_load.return_value = mock_yolo_model

        holds = detect_holds(rgb_image, detection_weights)
        assert len(holds) == 2

        crops = extract_hold_crops(rgb_image, holds)
        assert len(crops) == 2
        for crop in crops:
            assert isinstance(crop, HoldCrop)
            assert crop.crop.size == TARGET_SIZE
            assert crop.crop.mode == "RGB"

    @patch("src.inference.detection._load_model_cached")
    def test_crop_references_source_hold(
        self,
        mock_load: MagicMock,
        mock_yolo_model: MagicMock,
        rgb_image: PILImage.Image,
        detection_weights: Path,
    ) -> None:
        """Each HoldCrop.hold should be the corresponding DetectedHold."""
        mock_load.return_value = mock_yolo_model

        holds = detect_holds(rgb_image, detection_weights)
        crops = extract_hold_crops(rgb_image, holds)

        for hold, crop in zip(holds, crops):
            assert crop.hold is hold

    @patch("src.inference.detection._load_model_cached")
    def test_empty_detections_produce_no_crops(
        self,
        mock_load: MagicMock,
        rgb_image: PILImage.Image,
        detection_weights: Path,
    ) -> None:
        """When no holds are detected, extract_hold_crops returns an empty list."""
        empty_model = MagicMock()
        empty_results = MagicMock()
        empty_boxes = MagicMock()
        empty_boxes.xywhn = torch.zeros((0, 4))
        empty_boxes.cls = torch.zeros(0)
        empty_boxes.conf = torch.zeros(0)
        empty_results.boxes = empty_boxes
        empty_model.predict.return_value = [empty_results]
        mock_load.return_value = empty_model

        holds = detect_holds(rgb_image, detection_weights)
        assert holds == []

        crops = extract_hold_crops(rgb_image, holds)
        assert crops == []


# ---------------------------------------------------------------------------
# TestCropToClassify
# ---------------------------------------------------------------------------


class TestCropToClassify:
    """Tests for the extract_hold_crops -> classify_holds handoff."""

    @patch("src.inference.detection._load_model_cached")
    @patch("src.inference.classification._load_model_cached")
    def test_full_pipeline_returns_one_result_per_hold(
        self,
        mock_cls_load: MagicMock,
        mock_det_load: MagicMock,
        mock_yolo_model: MagicMock,
        mock_classifier_model: MagicMock,
        rgb_image: PILImage.Image,
        detection_weights: Path,
        classification_weights: Path,
    ) -> None:
        """End-to-end pipeline returns exactly one HoldTypeResult per detected hold."""
        mock_det_load.return_value = mock_yolo_model
        mock_cls_load.return_value = (mock_classifier_model, INPUT_SIZE)

        holds = detect_holds(rgb_image, detection_weights)
        crops = extract_hold_crops(rgb_image, holds)
        results = classify_holds(crops, classification_weights)

        assert len(results) == len(holds) == 2

    @patch("src.inference.detection._load_model_cached")
    @patch("src.inference.classification._load_model_cached")
    def test_source_crop_links_back_to_hold(
        self,
        mock_cls_load: MagicMock,
        mock_det_load: MagicMock,
        mock_yolo_model: MagicMock,
        mock_classifier_model: MagicMock,
        rgb_image: PILImage.Image,
        detection_weights: Path,
        classification_weights: Path,
    ) -> None:
        """HoldTypeResult.source_crop.hold should link back to the original DetectedHold."""
        mock_det_load.return_value = mock_yolo_model
        mock_cls_load.return_value = (mock_classifier_model, INPUT_SIZE)

        holds = detect_holds(rgb_image, detection_weights)
        crops = extract_hold_crops(rgb_image, holds)
        results = classify_holds(crops, classification_weights)

        for result, hold in zip(results, holds):
            assert result.source_crop is not None
            assert result.source_crop.hold is hold

    @patch("src.inference.detection._load_model_cached")
    @patch("src.inference.classification._load_model_cached")
    def test_results_have_valid_probabilities(
        self,
        mock_cls_load: MagicMock,
        mock_det_load: MagicMock,
        mock_yolo_model: MagicMock,
        mock_classifier_model: MagicMock,
        rgb_image: PILImage.Image,
        detection_weights: Path,
        classification_weights: Path,
    ) -> None:
        """Each result should have probabilities summing to ~1.0 over 6 classes."""
        mock_det_load.return_value = mock_yolo_model
        mock_cls_load.return_value = (mock_classifier_model, INPUT_SIZE)

        holds = detect_holds(rgb_image, detection_weights)
        crops = extract_hold_crops(rgb_image, holds)
        results = classify_holds(crops, classification_weights)

        for result in results:
            assert len(result.probabilities) == 6
            total = sum(result.probabilities.values())
            assert total == pytest.approx(1.0, abs=1e-4)
