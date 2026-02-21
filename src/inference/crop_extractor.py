"""Hold crop extraction from detected bounding boxes.

This module converts normalized bounding boxes produced by the hold detection
model into fixed-size 224×224 RGB PIL image crops, ready to be fed into the
hold classification model (PR-4.5).

Each ``DetectedHold`` stores coordinates in YOLOv8 xywhn format (centre_x,
centre_y, width, height, all in [0, 1]).  ``extract_hold_crops`` converts
those to pixel coordinates, clamps them to the image boundaries, crops the
region, and resizes to ``TARGET_SIZE`` using high-quality LANCZOS resampling.

Example::

    >>> from PIL import Image
    >>> from src.inference.crop_extractor import extract_hold_crops
    >>> image = Image.open("route.jpg")
    >>> crops = extract_hold_crops(image, detected_holds)
    >>> print(crops[0].crop.size)   # (224, 224)
    (224, 224)
"""

from pathlib import Path
from typing import Final

import numpy as np
import PIL.Image as PILImage
from pydantic import BaseModel, ConfigDict

from src.inference.detection import DetectedHold, ImageInput
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SIZE: Final[tuple[int, int]] = (224, 224)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class CropExtractorError(Exception):
    """Raised when hold crop extraction fails.

    This is a sibling of ``InferenceError`` — same operational context but
    distinct responsibility: image cropping rather than model inference.

    Attributes:
        message: Human-readable description of the error.

    Example::

        >>> raise CropExtractorError("Degenerate box after clamping: area = 0")
    """

    def __init__(self, message: str) -> None:
        """Initialize CropExtractorError with a message.

        Args:
            message: Description of the crop extraction error.
        """
        self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class HoldCrop(BaseModel):
    """A 224×224 RGB crop of a single detected hold.

    Attributes:
        crop: 224×224 RGB PIL image cut from the original route image.
        hold: The source ``DetectedHold`` detection (immutable reference).
        pixel_box: ``(x1, y1, x2, y2)`` pixel coordinates of the cropped
            region **after clamping** to the image boundaries.

    Example::

        >>> hc = HoldCrop(crop=img_224, hold=detected_hold, pixel_box=(10, 20, 80, 90))
        >>> hc.crop.size
        (224, 224)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    crop: PILImage.Image
    hold: DetectedHold
    pixel_box: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_as_pil(image: ImageInput) -> PILImage.Image:
    """Load an image from various input types and return an RGB PIL Image.

    Accepts all types in the ``ImageInput`` union alias:
    ``Path``, ``str``, ``np.ndarray`` (H×W×C uint8), or
    ``PIL.Image.Image``.

    Args:
        image: Input image in one of the supported formats.

    Returns:
        RGB ``PIL.Image.Image``.

    Raises:
        CropExtractorError: If a ``Path``/``str`` does not point to an
            existing file, or if the type is not supported.

    Example::

        >>> pil = _load_as_pil(Path("route.jpg"))
        >>> pil.mode
        'RGB'
    """
    pil: PILImage.Image
    if isinstance(image, (Path, str)):
        path = Path(image)
        if not path.exists():
            raise CropExtractorError(f"Image file not found: {image}")
        pil = PILImage.open(path)
    elif isinstance(image, np.ndarray):
        pil = PILImage.fromarray(image)
    elif isinstance(image, PILImage.Image):
        pil = image
    else:
        raise CropExtractorError(
            f"Unsupported image type: {type(image).__name__}. "
            "Expected Path, str, np.ndarray, or PIL.Image.Image."
        )

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def _normalized_to_pixel_box(
    hold: DetectedHold,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Convert normalized xywhn coordinates to clamped pixel (x1, y1, x2, y2).

    YOLOv8 xywhn format stores centre coordinates and half-dimensions as
    fractions of the image dimensions.  This function converts to top-left /
    bottom-right corner coordinates in pixel space and clamps them to the
    valid image region.

    Args:
        hold: Source detection with normalized bounding box.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` integer pixel coordinates clamped to
        ``[0, img_w]`` (x-axis) and ``[0, img_h]`` (y-axis).

    Example::

        >>> h = DetectedHold(x_center=0.5, y_center=0.5, width=0.4, height=0.4,
        ...                  class_id=0, class_name="hold", confidence=0.9)
        >>> _normalized_to_pixel_box(h, 400, 300)
        (120, 90, 280, 210)
    """
    cx = hold.x_center * img_w
    cy = hold.y_center * img_h
    half_w = (hold.width * img_w) / 2.0
    half_h = (hold.height * img_h) / 2.0

    x1 = int(max(0, cx - half_w))
    y1 = int(max(0, cy - half_h))
    x2 = int(min(img_w, cx + half_w))
    y2 = int(min(img_h, cy + half_h))

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_hold_crops(
    image: ImageInput,
    boxes: list[DetectedHold],
    target_size: tuple[int, int] = TARGET_SIZE,
) -> list[HoldCrop]:
    """Extract fixed-size RGB image crops for each detected hold.

    For each ``DetectedHold`` in ``boxes``:

    1. Convert normalized xywhn coordinates to pixel (x1, y1, x2, y2).
    2. Clamp coordinates to image boundaries.
    3. Validate that the clamped crop has non-zero area.
    4. Crop the region from the source image.
    5. Resize to ``target_size`` using LANCZOS resampling.
    6. Ensure the crop is in RGB mode.

    Args:
        image: Source image.  Accepted types: ``Path``, ``str`` (file path),
            ``np.ndarray`` (H×W×C uint8), or ``PIL.Image.Image``.
        boxes: List of detected holds whose bounding boxes define crop regions.
            Processed in the same order as the input list.
        target_size: ``(width, height)`` tuple for the output crops.
            Defaults to ``(224, 224)``.

    Returns:
        List of ``HoldCrop`` objects, one per entry in ``boxes``, preserving
        input order.  Returns an empty list immediately if ``boxes`` is empty.

    Raises:
        CropExtractorError: If the image cannot be loaded, or if any hold
            produces a degenerate (zero-area) crop after coordinate clamping.

    Example::

        >>> crops = extract_hold_crops(image, holds)
        >>> assert all(c.crop.size == (224, 224) for c in crops)
        >>> assert all(c.crop.mode == "RGB" for c in crops)
    """
    if not boxes:
        return []

    try:
        pil_image = _load_as_pil(image)
    except CropExtractorError:
        raise
    except Exception as exc:
        raise CropExtractorError(f"Failed to load image: {exc}") from exc

    img_w, img_h = pil_image.size
    crops: list[HoldCrop] = []

    for hold in boxes:
        x1, y1, x2, y2 = _normalized_to_pixel_box(hold, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            raise CropExtractorError(
                f"Degenerate crop for hold at "
                f"({hold.x_center:.4f}, {hold.y_center:.4f}): "
                f"clamped box ({x1}, {y1}, {x2}, {y2}) has zero area."
            )

        try:
            crop_img = pil_image.crop((x1, y1, x2, y2))
            crop_img = crop_img.resize(target_size, PILImage.Resampling.LANCZOS)
            if crop_img.mode != "RGB":
                crop_img = crop_img.convert("RGB")
        except CropExtractorError:
            raise
        except Exception as exc:
            raise CropExtractorError(
                f"Crop/resize failed for hold at "
                f"({hold.x_center:.4f}, {hold.y_center:.4f}): {exc}"
            ) from exc

        crops.append(
            HoldCrop(
                crop=crop_img,
                hold=hold,
                pixel_box=(x1, y1, x2, y2),
            )
        )

        logger.debug(
            "Extracted crop %d/%d: box=(%d,%d,%d,%d) → %s",
            len(crops),
            len(boxes),
            x1,
            y1,
            x2,
            y2,
            target_size,
        )

    return crops
