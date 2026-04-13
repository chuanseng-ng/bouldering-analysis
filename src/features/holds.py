"""Hold composition feature extraction from classified hold lists.

Computes interpretable hold-type metrics from a list of
:class:`~src.graph.types.ClassifiedHold` instances. The resulting
:class:`HoldFeatures` feeds directly into the feature assembler (PR-6.3)
and the grade estimator (Milestone 7).

Example::

    >>> from src.features.holds import extract_hold_features
    >>> hf = extract_hold_features(holds)
    >>> print(hf.total_count, hf.jug_ratio)
    5 0.4
"""

import math

from pydantic import BaseModel, Field

from src.constants import MAX_HOLD_COUNT
from src.features.exceptions import FeatureExtractionError
from src.graph.types import ClassifiedHold
from src.logging_config import get_logger
from src.training.classification_dataset import HOLD_CLASSES

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class HoldFeatures(BaseModel):
    """Hold composition features extracted from a list of classified holds.

    All fields are non-negative.  Ratio fields are additionally bounded
    to [0, 1].  Soft ratios are the confidence-weighted (mean probability)
    distribution across all holds for each class.

    Attributes:
        total_count: Total number of holds in the list.

        jug_count: Number of holds classified as jug.
        crimp_count: Number of holds classified as crimp.
        sloper_count: Number of holds classified as sloper.
        pinch_count: Number of holds classified as pinch.
        pocket_count: Number of holds classified as pocket.
        edges_count: Number of holds classified as edges.
        foothold_count: Number of holds classified as foothold.
        unknown_count: Number of holds classified as unknown.

        jug_ratio: Fraction of holds classified as jug (count / total_count).
            ``0.0`` when total_count is 0.
        crimp_ratio: Fraction of holds classified as crimp.
        sloper_ratio: Fraction of holds classified as sloper.
        pinch_ratio: Fraction of holds classified as pinch.
        pocket_ratio: Fraction of holds classified as pocket.
        edges_ratio: Fraction of holds classified as edges.
        foothold_ratio: Fraction of holds classified as foothold.
        unknown_ratio: Fraction of holds classified as unknown.

        avg_hold_size: Mean bounding-box area (width * height) across holds.
        max_hold_size: Maximum bounding-box area.
        min_hold_size: Minimum bounding-box area.
        std_hold_size: Population standard deviation of bounding-box areas.
            ``0.0`` when fewer than 2 holds.

        jug_soft_ratio: Mean ``type_probabilities["jug"]`` across all holds.
        crimp_soft_ratio: Mean ``type_probabilities["crimp"]`` across all holds.
        sloper_soft_ratio: Mean ``type_probabilities["sloper"]`` across all holds.
        pinch_soft_ratio: Mean ``type_probabilities["pinch"]`` across all holds.
        pocket_soft_ratio: Mean ``type_probabilities["pocket"]`` across all holds.
        edges_soft_ratio: Mean ``type_probabilities["edges"]`` across all holds.
        foothold_soft_ratio: Mean ``type_probabilities["foothold"]`` across all holds.
        unknown_soft_ratio: Mean ``type_probabilities["unknown"]`` across all holds.

    Example::

        >>> hf = extract_hold_features(holds)
        >>> print(hf.total_count, hf.jug_soft_ratio)
        3 0.65
    """

    # Totals
    total_count: int = Field(ge=0)

    # Hard counts per type
    jug_count: int = Field(ge=0)
    crimp_count: int = Field(ge=0)
    sloper_count: int = Field(ge=0)
    pinch_count: int = Field(ge=0)
    pocket_count: int = Field(ge=0)
    edges_count: int = Field(ge=0)
    foothold_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)

    # Hard ratios per type (count / total_count)
    jug_ratio: float = Field(ge=0.0, le=1.0)
    crimp_ratio: float = Field(ge=0.0, le=1.0)
    sloper_ratio: float = Field(ge=0.0, le=1.0)
    pinch_ratio: float = Field(ge=0.0, le=1.0)
    pocket_ratio: float = Field(ge=0.0, le=1.0)
    edges_ratio: float = Field(ge=0.0, le=1.0)
    foothold_ratio: float = Field(ge=0.0, le=1.0)
    unknown_ratio: float = Field(ge=0.0, le=1.0)

    # Bounding-box area stats
    avg_hold_size: float = Field(ge=0.0)
    max_hold_size: float = Field(ge=0.0)
    min_hold_size: float = Field(ge=0.0)
    std_hold_size: float = Field(ge=0.0)

    # Confidence-weighted soft distribution
    jug_soft_ratio: float = Field(ge=0.0, le=1.0)
    crimp_soft_ratio: float = Field(ge=0.0, le=1.0)
    sloper_soft_ratio: float = Field(ge=0.0, le=1.0)
    pinch_soft_ratio: float = Field(ge=0.0, le=1.0)
    pocket_soft_ratio: float = Field(ge=0.0, le=1.0)
    edges_soft_ratio: float = Field(ge=0.0, le=1.0)
    foothold_soft_ratio: float = Field(ge=0.0, le=1.0)
    unknown_soft_ratio: float = Field(ge=0.0, le=1.0)


# Guard: verify HoldFeatures field names match HOLD_CLASSES exactly.
# Catches both taxonomy growth (new class → missing fields) and renames
# (e.g. jug_count → jug_cnt keeps the count at 23 but breaks downstream
# dict lookups like counts["jug"] → jug_count).
# Uses an explicit raise rather than assert so it cannot be silenced by python -O.
_EXPECTED_HOLD_FEATURES_FIELDS: frozenset[str] = frozenset(
    {"total_count", "avg_hold_size", "max_hold_size", "min_hold_size", "std_hold_size"}
    | {f"{cls}_count" for cls in HOLD_CLASSES}
    | {f"{cls}_ratio" for cls in HOLD_CLASSES}
    | {f"{cls}_soft_ratio" for cls in HOLD_CLASSES}
)
_actual_hold_features_fields = frozenset(HoldFeatures.model_fields.keys())
if _actual_hold_features_fields != _EXPECTED_HOLD_FEATURES_FIELDS:
    _missing = sorted(_EXPECTED_HOLD_FEATURES_FIELDS - _actual_hold_features_fields)
    _extra = sorted(_actual_hold_features_fields - _EXPECTED_HOLD_FEATURES_FIELDS)
    raise RuntimeError(
        f"HoldFeatures fields out of sync with HOLD_CLASSES. "
        f"Missing: {_missing}. Extra: {_extra}. "
        "Add the new hold type's count, ratio, and soft_ratio fields."
    )
del _actual_hold_features_fields


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _count_by_type(holds: list[ClassifiedHold]) -> dict[str, int]:
    """Count holds by their classified type.

    Returns a dict keyed on every entry in :data:`~src.training.classification_dataset.HOLD_CLASSES`
    with missing types defaulting to 0.

    Args:
        holds: List of classified holds.  May be empty.

    Returns:
        Dict mapping each hold class to its count.  All 8 HOLD_CLASSES keys
        are always present.

    Raises:
        FeatureExtractionError: If any hold has a ``hold_type`` not in
            ``HOLD_CLASSES`` (defensive guard; normally prevented by
            :class:`~src.graph.types.ClassifiedHold` field validation).

    Example::

        >>> counts = _count_by_type(holds)
        >>> print(counts["jug"])
        2
    """
    counts: dict[str, int] = {cls: 0 for cls in HOLD_CLASSES}
    for hold in holds:
        if hold.hold_type not in counts:
            raise FeatureExtractionError(
                f"Unexpected hold_type {hold.hold_type!r}; expected one of {list(HOLD_CLASSES)}"
            )
        counts[hold.hold_type] += 1
    return counts


def _compute_size_stats(
    holds: list[ClassifiedHold],
) -> tuple[float, float, float, float]:
    """Compute bounding-box area statistics across holds.

    Area per hold is ``width * height``.  Uses ``math.fsum`` for compensated
    floating-point summation.

    Args:
        holds: List of classified holds.  May be empty.

    Returns:
        4-tuple ``(avg, max, min, std)`` of bounding-box areas.
        Returns ``(0.0, 0.0, 0.0, 0.0)`` for an empty list.
        Returns ``(area, area, area, 0.0)`` for a single-element list.
        Population standard deviation (``ddof=0``) for 2 or more holds.

    Example::

        >>> avg, mx, mn, std = _compute_size_stats(holds)
    """
    if not holds:
        return (0.0, 0.0, 0.0, 0.0)

    areas = [h.width * h.height for h in holds]
    n = len(areas)

    if n == 1:
        area = areas[0]
        return (area, area, area, 0.0)

    avg = math.fsum(areas) / n
    variance = math.fsum((a - avg) ** 2 for a in areas) / n
    std = math.sqrt(variance)
    return (avg, max(areas), min(areas), std)


def _compute_soft_distribution(holds: list[ClassifiedHold]) -> dict[str, float]:
    """Compute the mean probability per class across all holds.

    For each class ``c``, the soft ratio is
    ``mean(h.type_probabilities[c] for h in holds)``.

    Args:
        holds: List of classified holds.  May be empty; returns all-zero
            distribution in that case.

    Returns:
        Dict mapping each HOLD_CLASSES entry to its mean probability.
        Values sum to approximately 1.0 for non-empty input; all 0.0 for
        empty input.

    Example::

        >>> soft = _compute_soft_distribution(holds)
        >>> print(soft["jug"])
        0.65
    """
    if not holds:
        return {cls: 0.0 for cls in HOLD_CLASSES}
    n = len(holds)
    return {
        cls: math.fsum(h.type_probabilities[cls] for h in holds) / n
        for cls in HOLD_CLASSES
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_hold_features(holds: list[ClassifiedHold]) -> HoldFeatures:
    """Extract hold composition features from a list of classified holds.

    Computes hard counts and ratios per hold type, bounding-box area
    statistics, and a confidence-weighted soft distribution over hold types.

    Args:
        holds: Non-empty list of :class:`~src.graph.types.ClassifiedHold`
            instances, typically the full hold list for a single route.

    Returns:
        A :class:`HoldFeatures` instance with all 23 fields populated.

    Raises:
        FeatureExtractionError: If ``holds`` is empty, or if
            ``len(holds)`` exceeds :data:`~src.constants.MAX_HOLD_COUNT`.

    Example::

        >>> hf = extract_hold_features(holds)
        >>> assert hf.total_count == len(holds)
        >>> assert hf.jug_ratio + hf.crimp_ratio + ... == 1.0
    """
    if not holds:
        raise FeatureExtractionError(
            "Cannot extract hold features from empty holds list"
        )
    if len(holds) > MAX_HOLD_COUNT:
        raise FeatureExtractionError(
            f"hold count {len(holds)} exceeds maximum {MAX_HOLD_COUNT}"
        )

    logger.debug("Extracting hold features from %d holds", len(holds))

    total_count = len(holds)
    counts = _count_by_type(holds)
    avg_size, max_size, min_size, std_size = _compute_size_stats(holds)
    soft = _compute_soft_distribution(holds)

    return HoldFeatures(
        total_count=total_count,
        jug_count=counts["jug"],
        crimp_count=counts["crimp"],
        sloper_count=counts["sloper"],
        pinch_count=counts["pinch"],
        pocket_count=counts["pocket"],
        edges_count=counts["edges"],
        foothold_count=counts["foothold"],
        unknown_count=counts["unknown"],
        jug_ratio=counts["jug"] / total_count,
        crimp_ratio=counts["crimp"] / total_count,
        sloper_ratio=counts["sloper"] / total_count,
        pinch_ratio=counts["pinch"] / total_count,
        pocket_ratio=counts["pocket"] / total_count,
        edges_ratio=counts["edges"] / total_count,
        foothold_ratio=counts["foothold"] / total_count,
        unknown_ratio=counts["unknown"] / total_count,
        avg_hold_size=avg_size,
        max_hold_size=max_size,
        min_hold_size=min_size,
        std_hold_size=std_size,
        jug_soft_ratio=soft["jug"],
        crimp_soft_ratio=soft["crimp"],
        sloper_soft_ratio=soft["sloper"],
        pinch_soft_ratio=soft["pinch"],
        pocket_soft_ratio=soft["pocket"],
        edges_soft_ratio=soft["edges"],
        foothold_soft_ratio=soft["foothold"],
        unknown_soft_ratio=soft["unknown"],
    )
