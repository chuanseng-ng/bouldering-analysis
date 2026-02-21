"""Inference module for hold detection and classification.

This package provides real-time inference using trained YOLOv8 models
to detect and classify climbing holds in bouldering route images.

Modules:
    detection: Hold detection inference using trained YOLOv8 weights
    crop_extractor: 224×224 RGB crop extraction from detected hold boxes
"""

from src.inference.crop_extractor import (
    TARGET_SIZE,
    CropExtractorError,
    HoldCrop,
    extract_hold_crops,
)
from src.inference.detection import (
    CLASS_NAMES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DetectedHold,
    InferenceError,
    detect_holds,
    detect_holds_batch,
)

__all__ = [
    "CLASS_NAMES",
    "CropExtractorError",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "DetectedHold",
    "HoldCrop",
    "InferenceError",
    "TARGET_SIZE",
    "detect_holds",
    "detect_holds_batch",
    "extract_hold_crops",
]
