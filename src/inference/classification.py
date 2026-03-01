"""Hold type classification inference using trained ResNet/MobileNetV3 weights.

This module provides functions to classify climbing hold types from 224×224 RGB
crop images produced by the crop_extractor module.  It reconstructs the identical
model architecture (including optional dropout wrapping) from the training
metadata.json, loads the saved state_dict, and applies the validation-time
transform pipeline before running the forward pass.

The module-level model cache ensures each weights file is loaded only once per
process, making repeated classify_hold calls cheap after the first load.

Example::

    >>> from src.inference.classification import classify_hold
    >>> result = classify_hold(hold_crop, "models/classification/v1/weights/best.pt")
    >>> print(result.predicted_class, result.confidence)
    jug 0.92
"""

import json
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import PIL.Image as PILImage
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torchvision import transforms  # type: ignore[import-untyped]

from src.inference.cache import _InferenceModelCache
from src.inference.crop_extractor import HoldCrop
from src.logging_config import get_logger
from src.inference.exceptions import InferencePipelineError
from src.training.classification_dataset import HOLD_CLASSES
from src.training.classification_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    VAL_RESIZE_RATIO,
    ClassifierHyperparameters,
    apply_classifier_dropout,
    build_hold_classifier,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Module-level model cache: resolved path string → loaded nn.Module
_MODEL_CACHE: _InferenceModelCache[nn.Module] = _InferenceModelCache()
# Input-size cache: resolved path string → input_size from training metadata
_INPUT_SIZE_CACHE: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ClassificationInferenceError(InferencePipelineError):
    """Raised when hold classification inference fails.

    This is a sibling of InferenceError (not a subclass) as it represents
    the classification context rather than hold detection.

    Attributes:
        message: Human-readable description of the error.

    Example::

        >>> raise ClassificationInferenceError("Model weights not found: /path/to/model.pt")
    """

    def __init__(self, message: str) -> None:
        """Initialize ClassificationInferenceError with a message.

        Args:
            message: Description of the classification inference error.
        """
        self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class HoldTypeResult(BaseModel):
    """Classification result for a single hold crop.

    Attributes:
        predicted_class: Predicted hold type; one of the 6 entries in
            HOLD_CLASSES (``"jug"``, ``"crimp"``, ``"sloper"``,
            ``"pinch"``, ``"volume"``, ``"unknown"``).
        confidence: Softmax probability of the predicted class (0–1).
        probabilities: Full probability distribution over all 6 hold classes,
            keyed by class name.
        source_crop: The HoldCrop input, if provided; None when input was a
            raw PIL.Image.Image rather than a HoldCrop.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    source_crop: HoldCrop | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clear_model_cache() -> None:
    """Clear all cached classifier model instances and input-size entries.

    Intended for testing and memory management.  After calling this,
    the next call to :func:`_load_model_cached` will reload from disk.
    """
    _MODEL_CACHE.clear()
    _INPUT_SIZE_CACHE.clear()


def reset_classification_model_cache() -> None:
    """Clear all cached classifier model instances (public API for testing and hot-reload).

    After calling this, the next call to classify_hold will reload the model
    from disk. Use reset_supabase_client_cache() pattern for test isolation.

    Example:
        >>> reset_classification_model_cache()
    """
    _clear_model_cache()


def _load_metadata(weights_path: Path) -> dict[str, Any]:
    """Load training metadata.json adjacent to the weights directory.

    The metadata.json is located at::

        weights_path.parent.parent / "metadata.json"

    This matches the artifact directory layout produced by
    :func:`~src.training.train_classification.train_hold_classifier`::

        models/classification/<version>/
            weights/
                best.pt    ← weights_path
                last.pt
            metadata.json  ← resolved here

    Args:
        weights_path: Path to the .pt weights file.

    Returns:
        Parsed metadata dictionary.

    Raises:
        ClassificationInferenceError: If metadata.json is not found or
            contains invalid JSON.
    """
    metadata_path = weights_path.parent.parent / "metadata.json"
    if not metadata_path.exists():
        raise ClassificationInferenceError(
            f"metadata.json not found at: {metadata_path}"
        )

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return dict(json.load(f))
    except json.JSONDecodeError as exc:
        raise ClassificationInferenceError(
            f"Failed to parse metadata.json at '{metadata_path}': {exc}"
        ) from exc


def _build_model_from_metadata(
    metadata: dict[str, Any],
    weights_path: Path,
) -> nn.Module:
    """Reconstruct model from metadata and load saved weights.

    Reads hyperparameters from the metadata dict, builds the classifier
    backbone via :func:`~src.training.classification_model.build_hold_classifier`,
    applies the same dropout wrapping used at training time via
    :func:`~src.training.classification_model.apply_classifier_dropout`,
    loads the saved state_dict with ``weights_only=True``, and sets the
    model to eval mode.

    Args:
        metadata: Parsed metadata.json dictionary containing a
            ``"hyperparameters"`` key.
        weights_path: Resolved path to the .pt weights file.

    Returns:
        nn.Module in eval mode with the saved weights loaded.

    Raises:
        ClassificationInferenceError: If the weights file is missing,
            hyperparameters are invalid, or the state_dict cannot be loaded.
    """
    if not weights_path.exists():
        raise ClassificationInferenceError(f"Model weights not found: {weights_path}")

    if "hyperparameters" not in metadata:
        raise ClassificationInferenceError(
            "metadata.json is missing the required 'hyperparameters' key"
        )

    try:
        hp = ClassifierHyperparameters(**metadata["hyperparameters"])
    except Exception as exc:
        raise ClassificationInferenceError(
            f"Invalid hyperparameters in metadata: {exc}"
        ) from exc

    config = build_hold_classifier(hp)
    model: nn.Module = config["model"]
    model = apply_classifier_dropout(model, hp.architecture, hp.dropout_rate)

    try:
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint)
    except Exception as exc:
        raise ClassificationInferenceError(
            f"Failed to load state_dict from '{weights_path}': {exc}"
        ) from exc

    model.eval()
    return model


def _load_model_cached(weights_path: Path | str) -> nn.Module:
    """Load a classifier model from disk, returning a cached instance on repeat calls.

    The cache key is the resolved absolute path string so that equivalent
    relative/absolute paths share the same cached model.

    Args:
        weights_path: Path to the .pt weights file.

    Returns:
        Loaded nn.Module in eval mode.

    Raises:
        ClassificationInferenceError: If the weights file is missing,
            metadata.json is absent, or model loading fails.
    """
    resolved_path = Path(weights_path).resolve()
    resolved = str(resolved_path)

    # Fast path: check if already cached
    cached = _MODEL_CACHE.get(resolved)
    if cached is not None:
        return cached

    # Load via cache (thread-safe double-checked locking)
    def _loader(p: Path) -> nn.Module:
        logger.info("Loading classification model from: %s", str(p))
        metadata = _load_metadata(p)
        model = _build_model_from_metadata(metadata, p)
        input_size = int(
            metadata.get("hyperparameters", {}).get("input_size", INPUT_SIZE)
        )
        _INPUT_SIZE_CACHE[str(p)] = input_size
        return model

    return _MODEL_CACHE.load_or_store(weights_path, _loader)


def _validate_crop_input(crop: Any) -> None:
    """Validate that the crop input is a supported type.

    Args:
        crop: The crop to validate.  Accepted types:
            :class:`~src.inference.crop_extractor.HoldCrop` or
            ``PIL.Image.Image``.

    Raises:
        TypeError: If crop is not HoldCrop or PIL.Image.Image.
    """
    if not isinstance(crop, (HoldCrop, PILImage.Image)):
        raise TypeError(
            f"crop must be HoldCrop or PIL.Image.Image; got {type(crop).__name__}"
        )


def _to_pil_image(crop: HoldCrop | PILImage.Image) -> PILImage.Image:
    """Extract or pass through a PIL Image, converting to RGB if needed.

    Args:
        crop: Either a :class:`~src.inference.crop_extractor.HoldCrop`
            (extracts ``crop.crop``) or a ``PIL.Image.Image`` (passed
            through directly).

    Returns:
        RGB ``PIL.Image.Image`` ready for the inference transform.
    """
    if isinstance(crop, HoldCrop):
        pil = crop.crop
    else:
        pil = crop

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


@lru_cache(maxsize=8)
def _get_inference_transform(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Build the validation-time inference transform pipeline.

    Mirrors the validation transform used in the classification training loop::

        Resize(input_size * 256/224) → CenterCrop(input_size) → ToTensor → Normalize(ImageNet)

    The resize size is derived from ``input_size`` using the standard
    ImageNet evaluation ratio (256/224), matching the training-time
    validation transform exactly regardless of the trained model's input
    resolution.

    Args:
        input_size: Square crop resolution expected by the model.  Defaults
            to ``INPUT_SIZE`` (224) which matches the standard ImageNet
            resolution used for all current trained checkpoints.

    Returns:
        ``transforms.Compose`` pipeline ready to apply to a PIL.Image.Image.
    """
    resize_size = int(input_size * VAL_RESIZE_RATIO)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(IMAGENET_MEAN),
                std=list(IMAGENET_STD),
            ),
        ]
    )


def _logits_to_result(
    logits: torch.Tensor,
    source_crop: HoldCrop | PILImage.Image | None,
) -> HoldTypeResult:
    """Convert raw model logits to a HoldTypeResult.

    Applies softmax, finds the argmax class, and builds the result object.

    Args:
        logits: Raw logits tensor of shape ``(num_classes,)``.
        source_crop: Original input crop for provenance tracking.  Only
            stored when it is a HoldCrop; raw PIL images are stored as None.

    Returns:
        :class:`HoldTypeResult` with predicted_class, confidence, and
        a full probabilities dict.
    """
    probs = torch.softmax(logits, dim=0)
    predicted_idx = int(probs.argmax().item())
    predicted_class = HOLD_CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx].item())

    probabilities = {cls: float(probs[i].item()) for i, cls in enumerate(HOLD_CLASSES)}
    hold_crop = source_crop if isinstance(source_crop, HoldCrop) else None

    return HoldTypeResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        source_crop=hold_crop,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_hold(
    crop: HoldCrop | PILImage.Image,
    weights_path: Path | str,
) -> HoldTypeResult:
    """Classify a single hold crop into one of 6 hold types.

    Applies the standard ImageNet validation transform
    (Resize→CenterCrop→ToTensor→Normalize) and runs a single forward
    pass through the loaded classifier.

    Args:
        crop: The input image.  Accepted types:
            :class:`~src.inference.crop_extractor.HoldCrop` (from
            crop_extractor) or ``PIL.Image.Image``.  numpy arrays and
            file paths are not supported.
        weights_path: Path to a trained classifier .pt weights file.

    Returns:
        :class:`HoldTypeResult` with predicted_class, confidence, and
        full probability distribution.

    Raises:
        TypeError: If crop is not a HoldCrop or PIL.Image.Image.
        ClassificationInferenceError: If the weights file is missing,
            metadata.json is absent, or inference fails.

    Example::

        >>> result = classify_hold(hold_crop, "models/classification/v1/weights/best.pt")
        >>> print(result.predicted_class, result.confidence)
        jug 0.92
    """
    _validate_crop_input(crop)
    resolved = str(Path(weights_path).resolve())
    model = _load_model_cached(weights_path)
    input_size = _INPUT_SIZE_CACHE.get(resolved, INPUT_SIZE)

    try:
        pil = _to_pil_image(crop)
        transform = _get_inference_transform(input_size)
        tensor = transform(pil).unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():
            logits = model(tensor)[0]  # (num_classes,)

        return _logits_to_result(logits, crop)
    except (ClassificationInferenceError, TypeError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ClassificationInferenceError(
            f"Hold classification failed: {exc}"
        ) from exc


def classify_holds(
    crops: Sequence[HoldCrop | PILImage.Image],
    weights_path: Path | str,
    chunk_size: int | None = None,
) -> list[HoldTypeResult]:
    """Classify a batch of hold crops in a single forward pass.

    Stacks all crops into a single batch tensor and runs one forward pass
    for efficiency, rather than calling :func:`classify_hold` in a loop.

    When ``chunk_size`` is specified, crops are split into chunks of at most
    ``chunk_size`` elements and each chunk is processed in a separate forward
    pass.  This bounds the peak memory usage for large batches while
    preserving result order.

    Args:
        crops: Non-empty sequence of hold crops
            (:class:`~src.inference.crop_extractor.HoldCrop` or
            ``PIL.Image.Image``).
        weights_path: Path to a trained classifier .pt weights file.
        chunk_size: Maximum number of crops per forward pass.  ``None``
            (the default) processes all crops in a single pass.  Must be
            >= 1 when specified.

    Returns:
        List of :class:`HoldTypeResult` objects, one per crop, in the
        same order as the input list.

    Raises:
        ValueError: If crops is an empty list or chunk_size < 1.
        TypeError: If any crop is not a HoldCrop or PIL.Image.Image.
        ClassificationInferenceError: If the weights file is missing,
            metadata.json is absent, or inference fails.

    Example::

        >>> results = classify_holds(hold_crops, "models/classification/v1/weights/best.pt")
        >>> print(len(results))
        5
        >>> # Memory-bounded variant
        >>> results = classify_holds(hold_crops, weights_path, chunk_size=16)
    """
    if not crops:
        raise ValueError("crops list must not be empty")

    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    for crop in crops:
        _validate_crop_input(crop)

    resolved = str(Path(weights_path).resolve())
    model = _load_model_cached(weights_path)
    input_size = _INPUT_SIZE_CACHE.get(resolved, INPUT_SIZE)

    effective_chunk = chunk_size if chunk_size is not None else len(crops)
    results: list[HoldTypeResult] = []

    try:
        transform = _get_inference_transform(input_size)
        for start in range(0, len(crops), effective_chunk):
            chunk = crops[start : start + effective_chunk]
            tensors = [transform(_to_pil_image(crop)) for crop in chunk]
            batch = torch.stack(tensors, dim=0)  # (chunk, C, H, W)

            with torch.no_grad():
                logits_batch = model(batch)  # (chunk, num_classes)

            results.extend(
                _logits_to_result(logits_batch[i], crop) for i, crop in enumerate(chunk)
            )
        return results
    except (ClassificationInferenceError, TypeError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise ClassificationInferenceError(
            f"Hold classification failed: {exc}"
        ) from exc
