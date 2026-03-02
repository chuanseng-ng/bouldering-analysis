"""Canonical hold type for route graph construction (Milestones 5–7).

This module defines :class:`ClassifiedHold`, a flat, validated Pydantic model
that combines position data from :class:`~src.inference.detection.DetectedHold`
with semantic type data from :class:`~src.inference.classification.HoldTypeResult`.

Use :func:`make_classified_hold` to create instances from pipeline outputs.
``ClassifiedHold`` is the primary node type for all downstream stages:
route graph construction (PR-5.x), feature extraction (PR-6.x), and grade
estimation (PR-7.x).

Example::

    >>> from src.graph.types import make_classified_hold
    >>> hold = make_classified_hold(hold_id=0, detection=det, classification=clf)
    >>> print(hold.x_center, hold.hold_type)
    0.42 jug
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.inference.classification import HoldTypeResult
from src.inference.detection import DetectedHold
from src.training.classification_dataset import HOLD_CLASSES


class ClassifiedHold(BaseModel):
    """A detected and classified climbing hold, ready for graph construction.

    This is the canonical node type for Milestones 5–7. It is constructed via
    :func:`make_classified_hold` from a :class:`~src.inference.detection.DetectedHold`
    and a :class:`~src.inference.classification.HoldTypeResult`.

    All spatial coordinates are normalized to [0, 1] (YOLOv8 xywhn format).

    Attributes:
        hold_id: Non-negative integer uniquely identifying this hold within
            a single route. Typically the 0-based index in the detection list.
        x_center: Horizontal centre of the bounding box, normalized [0, 1].
        y_center: Vertical centre of the bounding box, normalized [0, 1].
        width: Bounding box width as a fraction of image width [0, 1].
        height: Bounding box height as a fraction of image height [0, 1].
        detection_class: YOLO detection class: ``"hold"`` or ``"volume"``.
        detection_confidence: YOLO detection confidence score [0, 1].
        hold_type: Semantic hold type; one of the 6 entries in
            :data:`~src.training.classification_dataset.HOLD_CLASSES`.
            Validated at construction time.
        type_confidence: Classifier confidence for ``hold_type`` [0, 1].
        type_probabilities: Full softmax probability distribution over all
            6 hold classes, keyed by class name.

    Example::

        >>> hold = ClassifiedHold(
        ...     hold_id=0, x_center=0.5, y_center=0.5,
        ...     width=0.1, height=0.1,
        ...     detection_class="hold", detection_confidence=0.9,
        ...     hold_type="jug", type_confidence=0.85,
        ...     type_probabilities={"jug": 0.85, ...},
        ... )
    """

    hold_id: int = Field(ge=0)
    x_center: float = Field(ge=0.0, le=1.0)
    y_center: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)
    detection_class: Literal["hold", "volume"]
    detection_confidence: float = Field(ge=0.0, le=1.0)
    hold_type: str
    type_confidence: float = Field(ge=0.0, le=1.0)
    type_probabilities: dict[str, float]

    @field_validator("type_probabilities")
    @classmethod
    def validate_type_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that type_probabilities is a well-formed distribution.

        Checks that keys match ``HOLD_CLASSES`` exactly, all values lie in
        [0, 1], and the distribution sums to approximately 1.0 (tolerance 0.01
        to absorb floating-point rounding from softmax computations).

        Args:
            v: The probability distribution dict to validate.

        Returns:
            The validated distribution dict, unchanged.

        Raises:
            ValueError: If keys do not match HOLD_CLASSES, any value is outside
                [0, 1], or the sum deviates from 1.0 by more than 0.01.
        """
        expected = set(HOLD_CLASSES)
        if set(v.keys()) != expected:
            raise ValueError(
                f"type_probabilities keys must be exactly {expected}, "
                f"got {set(v.keys())}"
            )
        for key, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"probability for {key!r} must be in [0.0, 1.0], got {prob}"
                )
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"type_probabilities must sum to approximately 1.0 "
                f"(tolerance 0.01), got {total:.6f}"
            )
        return v

    @field_validator("hold_type")
    @classmethod
    def validate_hold_type(cls, v: str) -> str:
        """Validate that hold_type is one of the 6 canonical hold classes.

        Note:
            No cross-field check is performed to assert that ``hold_type``
            equals ``argmax(type_probabilities)``.  This is intentional:
            probability distributions may be recalibrated after inference,
            and ``hold_type`` may be overridden during reclassification
            without updating the stored distribution.  Argmax consistency
            is the responsibility of the upstream
            :class:`~src.inference.classification.HoldTypeResult`.

        Args:
            v: The hold_type string to validate.

        Returns:
            The validated hold_type string.

        Raises:
            ValueError: If v is not in HOLD_CLASSES.
        """
        if v not in HOLD_CLASSES:
            raise ValueError(f"hold_type must be one of {HOLD_CLASSES}, got {v!r}")
        return v


def make_classified_hold(
    hold_id: int,
    detection: DetectedHold,
    classification: HoldTypeResult,
) -> ClassifiedHold:
    """Create a :class:`ClassifiedHold` from detection and classification outputs.

    Combines spatial data from ``detection`` with semantic type data from
    ``classification``. The explicit ``detection`` parameter is authoritative
    for position — ``classification.source_crop.hold`` is NOT used, ensuring
    this function works correctly even when ``source_crop`` is ``None`` or
    contains a different detection (e.g., when reclassifying).

    Args:
        hold_id: Non-negative integer identifier for this hold within the route.
            Typically the 0-based index in the detection list.
        detection: The :class:`~src.inference.detection.DetectedHold` providing
            position and detection-class information.
        classification: The :class:`~src.inference.classification.HoldTypeResult`
            providing hold type and probability distribution.

    Returns:
        A validated :class:`ClassifiedHold` combining both inputs.

    Example::

        >>> hold = make_classified_hold(hold_id=0, detection=det, classification=clf)
        >>> print(hold.hold_type, hold.x_center)
        jug 0.42
    """
    return ClassifiedHold(
        hold_id=hold_id,
        x_center=detection.x_center,
        y_center=detection.y_center,
        width=detection.width,
        height=detection.height,
        detection_class=detection.class_name,
        detection_confidence=detection.confidence,
        hold_type=classification.predicted_class,
        type_confidence=classification.confidence,
        type_probabilities=classification.probabilities,
    )
