"""Inference module for hold detection and classification.

This package provides real-time inference using trained YOLOv8 models
to detect and classify climbing holds in bouldering route images.

Modules:
    detection: Hold detection inference using trained YOLOv8 weights
"""

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
    "DetectedHold",
    "InferenceError",
    "CLASS_NAMES",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "detect_holds",
    "detect_holds_batch",
]
