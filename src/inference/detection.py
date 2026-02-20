"""Hold detection inference using trained YOLOv8 models.

This module provides functions to run hold/volume detection on bouldering
route images using pre-trained or fine-tuned YOLOv8 weights.

Detection results are returned as normalized bounding boxes (x_center,
y_center, width, height) in the range [0, 1], sorted by confidence
descending.

Example:
    >>> from src.inference.detection import detect_holds
    >>> holds = detect_holds("route.jpg", "models/detection/v1/weights/best.pt")
    >>> print(holds[0].class_name, holds[0].confidence)
    hold 0.87
"""

import threading
from pathlib import Path
from typing import Any, Final, Literal, Union, cast

import numpy as np
import PIL.Image as PILImage
from pydantic import BaseModel, Field
from ultralytics import YOLO

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class InferenceError(Exception):
    """Raised when hold detection inference fails.

    This is a sibling of TrainingError (not a subclass) as it represents
    a separate operational context: real-time inference rather than training.

    Attributes:
        message: Human-readable description of the error.

    Example:
        >>> raise InferenceError("Model weights not found: /path/to/model.pt")
    """

    def __init__(self, message: str) -> None:
        """Initialize InferenceError with a message.

        Args:
            message: Description of the inference error that occurred.
        """
        self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES: Final[tuple[str, ...]] = ("hold", "volume")
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45

# Module-level model cache: resolved path string → loaded YOLO model
_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_CACHE_LOCK: threading.Lock = threading.Lock()

# Type alias for accepted image inputs
ImageInput = Union[Path, str, np.ndarray, PILImage.Image]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DetectedHold(BaseModel):
    """A single detected climbing hold or volume.

    Coordinates are normalized to [0, 1] relative to image dimensions
    (YOLOv8 xywhn format: centre x/y, width, height).

    Attributes:
        x_center: Horizontal centre of the bounding box (0–1).
        y_center: Vertical centre of the bounding box (0–1).
        width: Bounding box width as a fraction of image width (0–1).
        height: Bounding box height as a fraction of image height (0–1).
        class_id: Class index (0 = hold, 1 = volume).
        class_name: Human-readable class label ('hold' or 'volume').
        confidence: Detection confidence score (0–1).
    """

    x_center: float = Field(ge=0.0, le=1.0)
    y_center: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0, le=1)
    class_name: Literal["hold", "volume"]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clear_model_cache() -> None:
    """Clear all cached YOLO model instances.

    Intended for testing and memory management. After calling this,
    the next call to _load_model_cached will reload from disk.
    """
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _validate_image_input(image: ImageInput) -> None:
    """Validate that image is a supported type and, if a path, that it exists.

    Args:
        image: The image to validate. Accepted types:
            - Path or str: checked for existence on disk.
            - np.ndarray or PIL.Image.Image: accepted as-is.

    Raises:
        TypeError: If image is not one of the accepted types.
        InferenceError: If image is a path/str pointing to a non-existent file.
    """
    if isinstance(image, (Path, str)):
        if not Path(image).exists():
            raise InferenceError(f"Image file not found: {image}")
    elif not isinstance(image, (np.ndarray, PILImage.Image)):
        raise TypeError(
            f"image must be Path, str, np.ndarray, or PIL.Image; "
            f"got {type(image).__name__}"
        )


def _load_model_cached(weights_path: Path | str) -> YOLO:
    """Load a YOLO model from disk, returning a cached instance on repeat calls.

    The cache key is the resolved absolute path string so that equivalent
    relative/absolute paths share the same cached model.

    Args:
        weights_path: Path to the .pt weights file.

    Returns:
        Loaded YOLO model instance.

    Raises:
        InferenceError: If the weights file does not exist.
    """
    resolved = str(Path(weights_path).resolve())
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]

    with _MODEL_CACHE_LOCK:
        if resolved in _MODEL_CACHE:
            return _MODEL_CACHE[resolved]

        if not Path(resolved).exists():
            raise InferenceError(f"Model weights not found: {weights_path}")

        logger.info("Loading model from: %s", resolved)
        model = YOLO(resolved)
        _MODEL_CACHE[resolved] = model
    return model


def _parse_yolo_results(
    results: list[Any],
    conf_threshold: float,
) -> list[DetectedHold]:
    """Parse YOLO predict() output into a sorted list of DetectedHold objects.

    Args:
        results: List of Ultralytics Results objects (one per image).
        conf_threshold: Minimum confidence to include a detection.

    Returns:
        List of DetectedHold objects filtered by conf_threshold and sorted
        by confidence descending.
    """
    holds: list[DetectedHold] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        xywhn: np.ndarray = boxes.xywhn.cpu().numpy()
        cls_arr: np.ndarray = boxes.cls.cpu().numpy()
        conf_arr: np.ndarray = boxes.conf.cpu().numpy()

        for j, conf in enumerate(conf_arr):
            if conf < conf_threshold:
                continue
            class_id = int(cls_arr[j])
            if not 0 <= class_id < len(CLASS_NAMES):
                logger.warning("Skipping detection with unknown class_id: %d", class_id)
                continue
            holds.append(
                DetectedHold(
                    x_center=float(xywhn[j][0]),
                    y_center=float(xywhn[j][1]),
                    width=float(xywhn[j][2]),
                    height=float(xywhn[j][3]),
                    class_id=class_id,
                    class_name=cast(Literal["hold", "volume"], CLASS_NAMES[class_id]),
                    confidence=float(conf),
                )
            )

    holds.sort(key=lambda h: h.confidence, reverse=True)
    return holds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_holds(
    image: ImageInput,
    weights_path: Path | str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[DetectedHold]:
    """Detect climbing holds and volumes in a single image.

    Args:
        image: The input image. Accepted types: Path, str (file path),
            np.ndarray (HxWxC, uint8), or PIL.Image.Image.
        weights_path: Path to a trained YOLOv8 .pt weights file.
        conf_threshold: Minimum detection confidence (default 0.25).
        iou_threshold: IoU threshold for NMS (default 0.45).

    Returns:
        List of DetectedHold objects sorted by confidence descending.
        Returns an empty list if no holds are detected above the threshold.

    Raises:
        TypeError: If image is not a supported input type.
        InferenceError: If the weights file is missing or prediction fails.

    Example:
        >>> holds = detect_holds("route.jpg", "models/best.pt")
        >>> print(len(holds), "holds detected")
        12 holds detected
    """
    _validate_image_input(image)
    model = _load_model_cached(weights_path)

    try:
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        return _parse_yolo_results(results, conf_threshold)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Hold detection failed: {exc}") from exc


def detect_holds_batch(
    images: list[ImageInput],
    weights_path: Path | str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[list[DetectedHold]]:
    """Detect climbing holds in a list of images.

    Args:
        images: Non-empty list of images (Path, str, np.ndarray, PIL.Image).
        weights_path: Path to a trained YOLOv8 .pt weights file.
        conf_threshold: Minimum detection confidence (default 0.25).
        iou_threshold: IoU threshold for NMS (default 0.45).

    Returns:
        List of detection lists, one per input image, in the same order.

    Raises:
        ValueError: If images is an empty list.
        TypeError: If any image is not a supported input type.
        InferenceError: If the weights file is missing or prediction fails.

    Example:
        >>> results = detect_holds_batch(["r1.jpg", "r2.jpg"], "models/best.pt")
        >>> print(len(results))
        2
    """
    if not images:
        raise ValueError("images list must not be empty")

    for image in images:
        _validate_image_input(image)

    model = _load_model_cached(weights_path)

    try:
        batch_results = model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        return [
            _parse_yolo_results([result], conf_threshold) for result in batch_results
        ]
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Hold detection failed: {exc}") from exc
