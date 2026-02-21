"""Tests for src/inference/crop_extractor.py.

Covers:
- HoldCrop Pydantic model validation
- _normalized_to_pixel_box coordinate conversion
- _load_as_pil image loading from multiple input types
- extract_hold_crops public API — happy paths and edge cases
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
from pathlib import Path
from typing import Literal

import numpy as np
import PIL.Image as PILImage
import pytest
from pydantic import ValidationError

from src.inference.crop_extractor import (
    TARGET_SIZE,
    CropExtractorError,
    HoldCrop,
    _load_as_pil,
    _normalized_to_pixel_box,
    extract_hold_crops,
)
from src.inference.detection import DetectedHold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hold(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.4,
    height: float = 0.4,
    class_id: int = 0,
    class_name: Literal["hold", "volume"] = "hold",
    confidence: float = 0.9,
) -> DetectedHold:
    """Factory for DetectedHold test objects.

    Convenience wrapper around DetectedHold that supplies sensible defaults
    so individual tests only need to override the fields they care about.

    Args:
        x_center: Horizontal centre of the bounding box (0–1). Defaults to 0.5.
        y_center: Vertical centre of the bounding box (0–1). Defaults to 0.5.
        width: Bounding-box width as a fraction of image width (0–1). Defaults to 0.4.
        height: Bounding-box height as a fraction of image height (0–1). Defaults to 0.4.
        class_id: Class index (0 = hold, 1 = volume). Defaults to 0.
        class_name: Human-readable class label. Defaults to "hold".
        confidence: Detection confidence score (0–1). Defaults to 0.9.

    Returns:
        DetectedHold: A new DetectedHold instance with the given field values.

    Example::

        >>> h = _make_hold(x_center=0.3, y_center=0.7, confidence=0.8)
        >>> h.class_name
        'hold'
    """
    return DetectedHold(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rgb_image() -> PILImage.Image:
    """400×300 solid-colour RGB PIL image.

    Returns:
        PILImage.Image: A 400×300 RGB image filled with a solid blue-grey colour,
            suitable as a generic source image in crop extraction tests.

    Example::

        >>> img = PILImage.new("RGB", (400, 300), color=(100, 150, 200))
        >>> img.size
        (400, 300)
        >>> img.mode
        'RGB'
    """
    img = PILImage.new("RGB", (400, 300), color=(100, 150, 200))
    return img


@pytest.fixture
def rgb_image_path(tmp_path: Path, rgb_image: PILImage.Image) -> Path:
    """Saved JPEG version of rgb_image for path-based input tests.

    Args:
        tmp_path: Pytest built-in fixture providing an isolated temporary directory.
        rgb_image: The in-memory RGB image fixture to persist to disk.

    Returns:
        Path: Absolute path to the saved JPEG file inside tmp_path.

    Example::

        >>> path = tmp_path / "route.jpg"
        >>> path.suffix
        '.jpg'
    """
    path = tmp_path / "route.jpg"
    rgb_image.save(str(path), format="JPEG")
    return path


@pytest.fixture
def center_hold() -> DetectedHold:
    """Hold centred in the image, clearly within bounds (40% × 40%).

    Returns:
        DetectedHold: A hold at (x_center=0.5, y_center=0.5, width=0.4, height=0.4)
            that maps to pixel box (120, 90, 280, 210) on a 400×300 image with
            no clamping required.

    Example::

        >>> hold = _make_hold(x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        >>> hold.x_center, hold.y_center
        (0.5, 0.5)
    """
    return _make_hold(x_center=0.5, y_center=0.5, width=0.4, height=0.4)


@pytest.fixture
def edge_hold() -> DetectedHold:
    """Hold at the top-left corner — box extends outside the image boundary.

    Returns:
        DetectedHold: A hold at (x_center=0.0, y_center=0.0, width=0.2, height=0.2)
            whose raw bounding box extends into negative pixel coordinates;
            the crop extractor clamps the box to (0, 0, 40, 30) on a 400×300 image.

    Example::

        >>> hold = _make_hold(x_center=0.0, y_center=0.0, width=0.2, height=0.2)
        >>> hold.x_center, hold.y_center
        (0.0, 0.0)
    """
    return _make_hold(x_center=0.0, y_center=0.0, width=0.2, height=0.2)


@pytest.fixture
def tiny_hold() -> DetectedHold:
    """Very small hold (1% of image dimensions).

    Returns:
        DetectedHold: A hold at (x_center=0.5, y_center=0.5, width=0.01, height=0.01)
            that maps to a very small pixel region; the crop extractor must
            upsample the result to the target size.

    Example::

        >>> hold = _make_hold(x_center=0.5, y_center=0.5, width=0.01, height=0.01)
        >>> hold.width, hold.height
        (0.01, 0.01)
    """
    return _make_hold(x_center=0.5, y_center=0.5, width=0.01, height=0.01)


@pytest.fixture
def volume_hold() -> DetectedHold:
    """Volume-class hold (class_id=1).

    Returns:
        DetectedHold: A hold at (x_center=0.3, y_center=0.3, width=0.3, height=0.3)
            with class_id=1 and class_name="volume", used to verify that
            volume-class detections are processed identically to regular holds.

    Example::

        >>> hold = _make_hold(x_center=0.3, y_center=0.3, width=0.3, height=0.3,
        ...                   class_id=1, class_name="volume")
        >>> hold.class_name
        'volume'
    """
    return _make_hold(
        x_center=0.3,
        y_center=0.3,
        width=0.3,
        height=0.3,
        class_id=1,
        class_name="volume",
    )


# ---------------------------------------------------------------------------
# TestHoldCrop
# ---------------------------------------------------------------------------


class TestHoldCrop:
    """Pydantic model validation for HoldCrop."""

    def test_accepts_valid_pil_image_and_hold(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """HoldCrop should accept a PIL.Image and DetectedHold."""
        hc = HoldCrop(
            crop=rgb_image,
            hold=center_hold,
            pixel_box=(10, 20, 80, 90),
        )
        assert hc.crop is rgb_image
        assert hc.hold is center_hold
        assert hc.pixel_box == (10, 20, 80, 90)

    def test_rejects_non_image_crop(self, center_hold: DetectedHold) -> None:
        """HoldCrop should reject a non-PIL crop value."""
        with pytest.raises(ValidationError):
            HoldCrop(
                crop="not an image",  # type: ignore[arg-type]
                hold=center_hold,
                pixel_box=(0, 0, 10, 10),
            )


# ---------------------------------------------------------------------------
# TestNormalizedToPixelBox
# ---------------------------------------------------------------------------


class TestNormalizedToPixelBox:
    """Unit tests for _normalized_to_pixel_box coordinate conversion."""

    def test_center_hold_correct_coords(self, center_hold: DetectedHold) -> None:
        """Center hold (50%, 50%, 40%×40%) on 400×300 → correct pixels."""
        # cx=200, cy=150, half_w=80, half_h=60
        # x1=120, y1=90, x2=280, y2=210
        box = _normalized_to_pixel_box(center_hold, img_w=400, img_h=300)
        assert box == (120, 90, 280, 210)

    def test_top_left_corner_clamps_at_zero(self) -> None:
        """Hold at top-left corner should clamp negative coords to 0."""
        # centre at (0,0), half_w=40, half_h=30 → raw: (-40,-30,40,30)
        hold = _make_hold(x_center=0.0, y_center=0.0, width=0.2, height=0.2)
        box = _normalized_to_pixel_box(hold, img_w=400, img_h=300)
        x1, y1, x2, y2 = box
        assert x1 == 0
        assert y1 == 0
        assert x2 > 0
        assert y2 > 0

    def test_bottom_right_corner_clamps_at_dimensions(self) -> None:
        """Hold at bottom-right corner should clamp to image size."""
        hold = _make_hold(x_center=1.0, y_center=1.0, width=0.2, height=0.2)
        box = _normalized_to_pixel_box(hold, img_w=400, img_h=300)
        x1, y1, x2, y2 = box
        assert x2 == 400
        assert y2 == 300
        assert x1 < 400
        assert y1 < 300

    def test_full_image_hold(self) -> None:
        """Hold spanning the full image should equal image dimensions."""
        hold = _make_hold(x_center=0.5, y_center=0.5, width=1.0, height=1.0)
        box = _normalized_to_pixel_box(hold, img_w=400, img_h=300)
        assert box == (0, 0, 400, 300)

    def test_returns_int_coordinates(self, center_hold: DetectedHold) -> None:
        """All coordinates in the returned tuple should be integers."""
        box = _normalized_to_pixel_box(center_hold, img_w=400, img_h=300)
        assert all(isinstance(v, int) for v in box)


# ---------------------------------------------------------------------------
# TestLoadAsPil
# ---------------------------------------------------------------------------


class TestLoadAsPil:
    """Tests for _load_as_pil image loading."""

    def test_accepts_pil_image(self, rgb_image: PILImage.Image) -> None:
        """_load_as_pil should accept a PIL.Image and return it as RGB."""
        result = _load_as_pil(rgb_image)
        assert isinstance(result, PILImage.Image)
        assert result.mode == "RGB"

    def test_accepts_numpy_array(self) -> None:
        """_load_as_pil should accept an HxWxC uint8 numpy array."""
        arr = np.zeros((100, 150, 3), dtype=np.uint8)
        result = _load_as_pil(arr)
        assert isinstance(result, PILImage.Image)
        assert result.mode == "RGB"

    def test_accepts_path_object(self, rgb_image_path: Path) -> None:
        """_load_as_pil should accept a Path and load the image."""
        result = _load_as_pil(rgb_image_path)
        assert isinstance(result, PILImage.Image)
        assert result.mode == "RGB"

    def test_accepts_str_path(self, rgb_image_path: Path) -> None:
        """_load_as_pil should accept a str path and load the image."""
        result = _load_as_pil(str(rgb_image_path))
        assert isinstance(result, PILImage.Image)
        assert result.mode == "RGB"

    def test_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        """_load_as_pil should raise CropExtractorError for missing file."""
        with pytest.raises(CropExtractorError, match="Image file not found"):
            _load_as_pil(tmp_path / "no_such_file.jpg")

    def test_raises_for_unsupported_type(self) -> None:
        """_load_as_pil should raise CropExtractorError for unsupported types."""
        with pytest.raises(CropExtractorError, match="Unsupported image type"):
            _load_as_pil(42)  # type: ignore[arg-type]

    def test_converts_grayscale_to_rgb(self) -> None:
        """_load_as_pil should convert grayscale images to RGB."""
        gray = PILImage.new("L", (100, 100), color=128)
        result = _load_as_pil(gray)
        assert result.mode == "RGB"

    def test_converts_rgba_to_rgb(self) -> None:
        """_load_as_pil should convert RGBA images to RGB."""
        rgba = PILImage.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        result = _load_as_pil(rgba)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# TestExtractHoldCrops
# ---------------------------------------------------------------------------


class TestExtractHoldCrops:
    """Tests for the extract_hold_crops public API."""

    def test_returns_empty_list_for_empty_boxes(
        self, rgb_image: PILImage.Image
    ) -> None:
        """extract_hold_crops should return [] immediately when boxes is empty."""
        result = extract_hold_crops(rgb_image, [])
        assert not result

    def test_returns_one_crop_per_hold(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """Output list length should equal len(boxes)."""
        holds = [center_hold, _make_hold(x_center=0.3, y_center=0.3)]
        result = extract_hold_crops(rgb_image, holds)
        assert len(result) == 2

    def test_output_crops_are_224x224(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """All output crops should be TARGET_SIZE (224×224)."""
        result = extract_hold_crops(rgb_image, [center_hold])
        assert result[0].crop.size == TARGET_SIZE

    def test_output_crops_are_rgb(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """All output crops should be in RGB mode."""
        result = extract_hold_crops(rgb_image, [center_hold])
        assert result[0].crop.mode == "RGB"

    def test_center_hold_correct_crop_dimensions(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """Center hold pixel_box should match _normalized_to_pixel_box output."""
        result = extract_hold_crops(rgb_image, [center_hold])
        crop = result[0]
        expected_box = _normalized_to_pixel_box(center_hold, img_w=400, img_h=300)
        assert crop.pixel_box == expected_box

    def test_edge_hold_clamped_still_224x224(
        self, rgb_image: PILImage.Image, edge_hold: DetectedHold
    ) -> None:
        """Partially out-of-bounds hold should still produce 224×224 crop."""
        result = extract_hold_crops(rgb_image, [edge_hold])
        assert result[0].crop.size == TARGET_SIZE

    def test_tiny_hold_resized_to_224x224(
        self, rgb_image: PILImage.Image, tiny_hold: DetectedHold
    ) -> None:
        """Very small hold should be upsampled to 224×224."""
        result = extract_hold_crops(rgb_image, [tiny_hold])
        assert result[0].crop.size == TARGET_SIZE

    def test_volume_class_hold_works(
        self, rgb_image: PILImage.Image, volume_hold: DetectedHold
    ) -> None:
        """Volume-class holds should be processed identically to regular holds."""
        result = extract_hold_crops(rgb_image, [volume_hold])
        assert len(result) == 1
        assert result[0].crop.size == TARGET_SIZE
        assert result[0].hold.class_name == "volume"

    def test_multiple_holds_preserve_order(self, rgb_image: PILImage.Image) -> None:
        """Output list must preserve the same order as the input boxes list."""
        hold_a = _make_hold(x_center=0.2, y_center=0.2, confidence=0.9)
        hold_b = _make_hold(x_center=0.5, y_center=0.5, confidence=0.7)
        hold_c = _make_hold(x_center=0.8, y_center=0.8, confidence=0.5)
        holds = [hold_a, hold_b, hold_c]

        result = extract_hold_crops(rgb_image, holds)

        assert len(result) == 3
        assert result[0].hold is hold_a
        assert result[1].hold is hold_b
        assert result[2].hold is hold_c

    def test_accepts_path_input(
        self, rgb_image_path: Path, center_hold: DetectedHold
    ) -> None:
        """extract_hold_crops should accept a Path for the image argument."""
        result = extract_hold_crops(rgb_image_path, [center_hold])
        assert len(result) == 1
        assert result[0].crop.size == TARGET_SIZE

    def test_accepts_str_path_input(
        self, rgb_image_path: Path, center_hold: DetectedHold
    ) -> None:
        """extract_hold_crops should accept a str path for the image argument."""
        result = extract_hold_crops(str(rgb_image_path), [center_hold])
        assert len(result) == 1
        assert result[0].crop.size == TARGET_SIZE

    def test_accepts_numpy_input(self, center_hold: DetectedHold) -> None:
        """extract_hold_crops should accept a numpy array for the image argument."""
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        result = extract_hold_crops(arr, [center_hold])
        assert len(result) == 1
        assert result[0].crop.size == TARGET_SIZE

    def test_raises_for_degenerate_box_fully_outside(self) -> None:
        """Hold fully outside image bounds should raise CropExtractorError."""
        # 400×300 image; hold is at far right corner with zero visible area
        # x_center=1.0, width=0.0 → raw x1=400, x2=400 → degenerate after clamp
        image = PILImage.new("RGB", (400, 300), color=(0, 0, 0))
        # Zero-width hold at edge — will produce x1==x2 after clamping
        degenerate_hold = _make_hold(x_center=1.0, y_center=0.5, width=0.0, height=0.4)
        with pytest.raises(CropExtractorError, match="zero area"):
            extract_hold_crops(image, [degenerate_hold])

    def test_pixel_box_reflects_clamped_coordinates(
        self, rgb_image: PILImage.Image, edge_hold: DetectedHold
    ) -> None:
        """pixel_box should store clamped coordinates, never negative values."""
        result = extract_hold_crops(rgb_image, [edge_hold])
        x1, y1, x2, y2 = result[0].pixel_box
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 400
        assert y2 <= 300

    def test_hold_reference_stored_in_crop(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """The hold field of HoldCrop must be the exact source DetectedHold."""
        result = extract_hold_crops(rgb_image, [center_hold])
        assert result[0].hold is center_hold

    def test_grayscale_image_produces_rgb_crops(
        self, center_hold: DetectedHold
    ) -> None:
        """Grayscale source image should still produce RGB crops."""
        gray_img = PILImage.new("L", (400, 300), color=128)
        result = extract_hold_crops(gray_img, [center_hold])
        assert result[0].crop.mode == "RGB"

    def test_custom_target_size(
        self, rgb_image: PILImage.Image, center_hold: DetectedHold
    ) -> None:
        """extract_hold_crops should honour a non-default target_size."""
        result = extract_hold_crops(rgb_image, [center_hold], target_size=(128, 128))
        assert result[0].crop.size == (128, 128)

    def test_raises_for_missing_image_path(
        self, tmp_path: Path, center_hold: DetectedHold
    ) -> None:
        """extract_hold_crops should raise CropExtractorError for missing path."""
        with pytest.raises(CropExtractorError, match="Image file not found"):
            extract_hold_crops(tmp_path / "no_such_file.jpg", [center_hold])
