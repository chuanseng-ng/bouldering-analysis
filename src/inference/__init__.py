"""Inference module for hold detection and classification.

This package provides real-time inference using trained models to detect
and classify climbing holds in bouldering route images.

Modules:
    detection: Hold detection inference using trained YOLOv8 weights
    crop_extractor: 224×224 RGB crop extraction from detected hold boxes
    classification: Hold type classification using trained ResNet/MobileNetV3 weights
"""

from src.inference.classification import (
    ClassificationInferenceError,
    HoldTypeResult,
    classify_hold,
    classify_holds,
    reset_classification_model_cache,
)
from src.inference.exceptions import InferencePipelineError
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
    reset_detection_model_cache,
)

__all__ = [
    "CLASS_NAMES",
    "ClassificationInferenceError",
    "CropExtractorError",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "DetectedHold",
    "HoldCrop",
    "HoldTypeResult",
    "InferencePipelineError",
    "InferenceError",
    "TARGET_SIZE",
    "classify_hold",
    "classify_holds",
    "detect_holds",
    "detect_holds_batch",
    "extract_hold_crops",
    "reset_classification_model_cache",
    "reset_detection_model_cache",
]
