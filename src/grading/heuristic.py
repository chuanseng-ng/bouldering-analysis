"""Heuristic grade estimator for bouldering routes.

Maps a :class:`~src.features.assembler.RouteFeatures` vector to a V-scale
grade (V0–V17) using a weighted combination of hold composition and geometry
features.

NOTE: Calibrated conservatively; tends to underestimate grades above V8.
PR-7.2 provides data-driven correction.

NOTE: Hold size features (avg/max/min/std_hold_size) are intentionally unused.
Accurate hold-size estimation requires dataset-level normalization (min/max
or mean/std) that is not available until PR-7.2.

Example::

    >>> from src.grading import estimate_grade_heuristic
    >>> result = estimate_grade_heuristic(route_features)
    >>> print(result.grade, result.confidence)
    V3 0.82
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features.assembler import RouteFeatures
from src.features.exceptions import FeatureExtractionError
from src.grading._utils import _clamp
from src.grading.constants import (
    FEATURE_WEIGHTS,
    GRADE_THRESHOLDS,
    MAX_HOPS_NORM,
    MAX_MOVE_DISTANCE,
    V_GRADES,
)
from src.grading.exceptions import GradeEstimationError
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level guard
# ---------------------------------------------------------------------------

if len(V_GRADES) != len(GRADE_THRESHOLDS):
    raise RuntimeError(
        "V_GRADES and GRADE_THRESHOLDS length mismatch — update constants.py"
    )

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

_N_GRADES = len(V_GRADES)  # 18


class HeuristicGradeResult(BaseModel):
    """Grade estimation result from the heuristic estimator.

    Attributes:
        grade: V-scale grade label, e.g. ``"V5"``.
        grade_index: Ordinal index into :data:`~src.grading.constants.V_GRADES`
            (0 = V0, 17 = V17).
        confidence: Confidence in the grade estimate.  ``1.0`` at the centre
            of a grade interval; ``0.5`` at the boundary between grades.
        difficulty_score: Raw difficulty score in ``[0, 1]`` before grade
            mapping.  Primary explainability handle for PR-8.

    Example::

        >>> result = HeuristicGradeResult(
        ...     grade="V3", grade_index=3, confidence=0.9, difficulty_score=0.19
        ... )
    """

    model_config = ConfigDict(frozen=True)

    grade: str
    grade_index: int = Field(ge=0, le=17)
    confidence: float = Field(ge=0.5, le=1.0)
    difficulty_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_grade_consistency(self) -> "HeuristicGradeResult":
        """Validate that grade and grade_index are mutually consistent.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If ``grade`` does not match ``V_GRADES[grade_index]``.
        """
        if self.grade != V_GRADES[self.grade_index]:
            raise ValueError(
                f"grade {self.grade!r} does not match "
                f"V_GRADES[{self.grade_index}] = {V_GRADES[self.grade_index]!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_hold_difficulty(vec: dict[str, float]) -> float:
    """Compute the hold-composition difficulty sub-score.

    Linearly combines hold-type ratios with signed weights from
    :data:`~src.grading.constants.FEATURE_WEIGHTS`.  Jugs reduce
    difficulty (negative weight); crimps, slopers, pinches, edges, and pockets
    increase it.  The result is clamped to ``[0, 1]``.

    Args:
        vec: Feature vector from :meth:`~src.features.assembler.RouteFeatures.to_vector`.

    Returns:
        Hold difficulty sub-score in ``[0, 1]``.

    Example::

        >>> score = _compute_hold_difficulty(vec)
        >>> assert 0.0 <= score <= 1.0
    """
    raw = (
        FEATURE_WEIGHTS["crimp_ratio"] * vec["crimp_ratio"]
        + FEATURE_WEIGHTS["sloper_ratio"] * vec["sloper_ratio"]
        + FEATURE_WEIGHTS["pinch_ratio"] * vec["pinch_ratio"]
        + FEATURE_WEIGHTS["edges_ratio"] * vec["edges_ratio"]
        + FEATURE_WEIGHTS["pocket_ratio"] * vec["pocket_ratio"]
        + FEATURE_WEIGHTS["jug_ratio"] * vec["jug_ratio"]
    )
    return _clamp(raw, 0.0, 1.0)


def _compute_geometry_difficulty(vec: dict[str, float]) -> float:
    """Compute the movement geometry difficulty sub-score.

    Normalises all three geometry inputs before weighting:

    * avg/max move distances are divided by :data:`~src.grading.constants.MAX_MOVE_DISTANCE`
      (``sqrt(2)`` — the maximum Euclidean distance in normalised [0,1]×[0,1]
      image coordinates), then capped at 1.0.
    * path hop count is divided by :data:`~src.grading.constants.MAX_HOPS_NORM`,
      then capped at 1.0.

    Weights sum to 1.0 so the raw value is naturally in ``[0, 1]``;
    it is clamped defensively.

    Args:
        vec: Feature vector from :meth:`~src.features.assembler.RouteFeatures.to_vector`.

    Returns:
        Geometry difficulty sub-score in ``[0, 1]``.

    Example::

        >>> score = _compute_geometry_difficulty(vec)
        >>> assert 0.0 <= score <= 1.0
    """
    norm_avg = min(vec["avg_move_distance"] / MAX_MOVE_DISTANCE, 1.0)
    norm_max = min(vec["max_move_distance"] / MAX_MOVE_DISTANCE, 1.0)
    norm_hops = min(vec["path_length_max_hops"] / MAX_HOPS_NORM, 1.0)
    raw = (
        FEATURE_WEIGHTS["avg_move_distance"] * norm_avg
        + FEATURE_WEIGHTS["max_move_distance"] * norm_max
        + FEATURE_WEIGHTS["path_length_max_hops"] * norm_hops
    )
    return _clamp(raw, 0.0, 1.0)


def _combine_scores(hold_score: float, geometry_score: float) -> float:
    """Combine hold and geometry sub-scores into the overall difficulty score.

    Uses a fixed weighted mix: 45 % hold composition, 55 % geometry.
    Result is clamped to ``[0, 1]``.

    Args:
        hold_score: Hold difficulty sub-score from :func:`_compute_hold_difficulty`.
        geometry_score: Geometry difficulty sub-score from
            :func:`_compute_geometry_difficulty`.

    Returns:
        Overall difficulty score in ``[0, 1]``.

    Example::

        >>> score = _combine_scores(0.3, 0.4)
        >>> assert 0.0 <= score <= 1.0
    """
    raw = (
        FEATURE_WEIGHTS["hold_weight"] * hold_score
        + FEATURE_WEIGHTS["geometry_weight"] * geometry_score
    )
    return _clamp(raw, 0.0, 1.0)


def _score_to_grade_index(score: float) -> int:
    """Map a difficulty score to a grade index.

    Partitions ``[0, 1)`` into 18 equal intervals; ``score == 1.0`` maps to
    index 17 (V17).

    Args:
        score: Overall difficulty score in ``[0, 1]``.

    Returns:
        Grade index in ``[0, 17]``.

    Example::

        >>> _score_to_grade_index(0.0)
        0
        >>> _score_to_grade_index(1.0)
        17
    """
    return min(int(score * _N_GRADES), _N_GRADES - 1)


def _compute_confidence(score: float, grade_index: int) -> float:
    """Compute confidence for a grade estimate.

    Confidence is ``1.0`` at the centre of the grade interval and ``0.5``
    at the boundary.  It is clamped to ``[0.5, 1.0]``.

    Args:
        score: Overall difficulty score in ``[0, 1]``.
        grade_index: Grade index returned by :func:`_score_to_grade_index`.

    Returns:
        Confidence in ``[0.5, 1.0]``.

    Example::

        >>> _compute_confidence(0.028, 0)   # near centre of V0 interval
        1.0
    """
    interval = 1.0 / _N_GRADES
    center = (grade_index + 0.5) * interval
    normalized_distance = abs(score - center) / (interval / 2)
    return _clamp(1.0 - 0.5 * normalized_distance, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_grade_heuristic(features: RouteFeatures) -> HeuristicGradeResult:
    """Estimate the V-scale grade of a bouldering route using heuristic rules.

    Converts a :class:`~src.features.assembler.RouteFeatures` instance into a
    difficulty score by combining a hold-composition sub-score with a movement
    geometry sub-score, then maps the score to the nearest V-grade.

    NOTE: Calibrated conservatively; tends to underestimate grades above V8.
    PR-7.2 provides data-driven correction.

    NOTE: Hold size features (avg/max/min/std_hold_size) are intentionally
    unused here — accurate normalization requires training-data statistics
    that are deferred to PR-7.2.

    Args:
        features: Assembled :class:`~src.features.assembler.RouteFeatures`
            from :func:`~src.features.assembler.assemble_features`.

    Returns:
        :class:`HeuristicGradeResult` with the estimated grade, grade index,
        confidence, and raw difficulty score.

    Raises:
        GradeEstimationError: If the feature vector cannot be assembled
            (wraps :class:`~src.features.exceptions.FeatureExtractionError`).

    Example::

        >>> result = estimate_grade_heuristic(route_features)
        >>> print(result.grade, result.confidence)
        V3 0.82
    """
    try:
        vec = features.to_vector()
    except FeatureExtractionError as exc:
        raise GradeEstimationError(
            f"Failed to build feature vector for grade estimation: {exc.message}"
        ) from exc

    logger.debug(
        "Estimating grade for route with node_count=%s",
        int(vec.get("node_count", 0)),
    )

    hold_score = _compute_hold_difficulty(vec)
    geometry_score = _compute_geometry_difficulty(vec)
    difficulty_score = _combine_scores(hold_score, geometry_score)
    grade_index = _score_to_grade_index(difficulty_score)
    confidence = _compute_confidence(difficulty_score, grade_index)
    grade = V_GRADES[grade_index]

    logger.debug(
        "Grade estimate: grade=%s score=%.4f confidence=%.4f",
        grade,
        difficulty_score,
        confidence,
    )

    return HeuristicGradeResult(
        grade=grade,
        grade_index=grade_index,
        confidence=confidence,
        difficulty_score=difficulty_score,
    )
