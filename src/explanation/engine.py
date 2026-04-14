"""Explanation engine for bouldering route grade estimates.

Produces a structured :class:`~src.explanation.types.ExplanationResult`
with natural language narrative, ranked feature contributions, and a
confidence qualifier from a :class:`~src.features.assembler.RouteFeatures`
and a prediction result.

No new third-party dependencies — pure Python + Pydantic.

Example::

    >>> from src.explanation import generate_explanation
    >>> result = generate_explanation(route_features, heuristic_result)
    >>> print(result.summary)
    This route is estimated at V3. The model is confident, primarily ...
"""

from typing import Literal

from src.explanation.exceptions import ExplanationError
from src.explanation.types import ExplanationResult, FeatureContribution
from src.features.assembler import RouteFeatures
from src.features.exceptions import FeatureExtractionError
from src.grading.constants import (
    FEATURE_WEIGHTS,
    MAX_HOPS_NORM,
    MAX_MOVE_DISTANCE,
)
from src.grading.heuristic import HeuristicGradeResult
from src.grading.ml_estimator import MLGradeResult
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Display name mapping for features used in explanations
# ---------------------------------------------------------------------------

_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "crimp_ratio": "Crimp ratio",
    "sloper_ratio": "Sloper ratio",
    "pinch_ratio": "Pinch ratio",
    "pocket_ratio": "Pocket ratio",
    "jug_ratio": "Jug ratio",
    "avg_move_distance": "Average move distance",
    "max_move_distance": "Maximum move distance",
    "path_length_max_hops": "Path length (hops)",
}

# Module-level guard: all FEATURE_WEIGHTS hold-feature keys must be mapped.
# Uses RuntimeError (not assert) so the check survives python -O mode.
_missing_display_keys = [
    k
    for k in FEATURE_WEIGHTS
    if k not in ("hold_weight", "geometry_weight") and k not in _FEATURE_DISPLAY_NAMES
]
if _missing_display_keys:
    raise RuntimeError(
        f"FEATURE_WEIGHTS keys missing from _FEATURE_DISPLAY_NAMES: "
        f"{_missing_display_keys} — update _FEATURE_DISPLAY_NAMES in engine.py"
    )

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_confidence_qualifier(
    confidence: float,
) -> Literal["very confident", "confident", "uncertain"]:
    """Map a numeric confidence score to a human-readable qualifier.

    Args:
        confidence: Confidence value from a grade prediction result.

    Returns:
        ``"very confident"`` if confidence >= 0.85,
        ``"confident"`` if confidence >= 0.65,
        ``"uncertain"`` otherwise.

    Example::

        >>> _get_confidence_qualifier(0.90)
        'very confident'
        >>> _get_confidence_qualifier(0.70)
        'confident'
        >>> _get_confidence_qualifier(0.50)
        'uncertain'
    """
    if confidence >= 0.85:
        return "very confident"
    if confidence >= 0.65:
        return "confident"
    return "uncertain"


def _generate_hold_description(name: str, value: float, impact: float) -> str:
    """Generate a one-sentence description for a hold-type contribution.

    Args:
        name: Human-readable feature name, e.g. ``"Crimp ratio"``.
        value: Raw ratio value (0.0–1.0).
        impact: Signed impact of this feature on difficulty.

    Returns:
        A single descriptive sentence.

    Example::

        >>> _generate_hold_description("Crimp ratio", 0.4, 0.14)
        '40% of holds are crimps, adding significant difficulty.'
    """
    _hold_plurals: dict[str, str] = {
        "Crimp ratio": "crimps",
        "Sloper ratio": "slopers",
        "Pinch ratio": "pinches",
        "Pocket ratio": "pockets",
        "Jug ratio": "jugs",
    }
    pct = round(value * 100)
    hold_type = _hold_plurals.get(name, name.split()[0].lower() + "s")
    if impact > 0:
        direction = "adding significant difficulty"
    elif impact < 0:
        direction = "reducing overall difficulty"
    else:
        direction = "having a neutral effect on difficulty"
    return f"{pct}% of holds are {hold_type}, {direction}."


def _generate_geometry_description(name: str, value: float, impact: float) -> str:
    """Generate a one-sentence description for a geometry contribution.

    Args:
        name: Human-readable feature name, e.g. ``"Average move distance"``.
        value: Raw feature value.
        impact: Signed impact of this feature on difficulty.

    Returns:
        A single descriptive sentence.

    Example::

        >>> _generate_geometry_description("Average move distance", 0.5, 0.25)
        'The average move distance is 0.50, contributing to route difficulty.'
    """
    if impact > 0:
        direction = "contributing to route difficulty"
    elif impact < 0:
        direction = "reducing route difficulty"
    else:
        direction = "having a neutral effect on difficulty"
    return f"The {name.lower()} is {value:.2f}, {direction}."


def _compute_hold_contributions(vec: dict[str, float]) -> list[FeatureContribution]:
    """Compute feature contributions for all hold-type ratios.

    Mirrors the normalization logic from
    :func:`~src.grading.heuristic._compute_hold_difficulty`: ratios are
    already in ``[0, 1]`` so the normalized value equals the raw value.
    Impact = ``weight * value`` (signed per :data:`~src.grading.constants.FEATURE_WEIGHTS`).

    Args:
        vec: Feature vector from :meth:`~src.features.assembler.RouteFeatures.to_vector`.

    Returns:
        List of 5 :class:`~src.explanation.types.FeatureContribution` instances,
        one per hold type (crimp, sloper, pinch, pocket, jug).
        ``foothold_ratio`` and ``unknown_ratio`` are intentionally excluded:
        foot placements and unclassified holds do not appear in
        :data:`~src.grading.constants.FEATURE_WEIGHTS` because they do not
        influence hand-move difficulty.  All 7 types are still shown in
        :func:`_build_hold_highlights` for route composition reporting.

    Example::

        >>> contribs = _compute_hold_contributions(vec)
        >>> len(contribs)
        5
    """
    hold_keys = [
        "crimp_ratio",
        "sloper_ratio",
        "pinch_ratio",
        "pocket_ratio",
        "jug_ratio",
    ]
    contributions: list[FeatureContribution] = []
    for key in hold_keys:
        value = vec.get(key, 0.0)
        weight = FEATURE_WEIGHTS[key]
        impact = weight * value
        name = _FEATURE_DISPLAY_NAMES[key]
        description = _generate_hold_description(name, value, impact)
        contributions.append(
            FeatureContribution(
                name=name, value=value, impact=impact, description=description
            )
        )
    return contributions


def _compute_geometry_contributions(vec: dict[str, float]) -> list[FeatureContribution]:
    """Compute feature contributions for geometry features.

    Applies the same normalization as
    :func:`~src.grading.heuristic._compute_geometry_difficulty`:

    * avg/max move distances are divided by :data:`~src.grading.constants.MAX_MOVE_DISTANCE`
      and capped at 1.0.
    * path hop count is divided by :data:`~src.grading.constants.MAX_HOPS_NORM`
      and capped at 1.0.

    Args:
        vec: Feature vector from :meth:`~src.features.assembler.RouteFeatures.to_vector`.

    Returns:
        List of 3 :class:`~src.explanation.types.FeatureContribution` instances
        (avg_move, max_move, path_hops).

    Example::

        >>> contribs = _compute_geometry_contributions(vec)
        >>> len(contribs)
        3
    """
    geo_keys = [
        ("avg_move_distance", MAX_MOVE_DISTANCE),
        ("max_move_distance", MAX_MOVE_DISTANCE),
        ("path_length_max_hops", float(MAX_HOPS_NORM)),
    ]
    contributions: list[FeatureContribution] = []
    for key, normalizer in geo_keys:
        raw_value = vec.get(key, 0.0)
        norm_value = min(raw_value / normalizer, 1.0)
        weight = FEATURE_WEIGHTS[key]
        impact = weight * norm_value
        name = _FEATURE_DISPLAY_NAMES[key]
        description = _generate_geometry_description(name, raw_value, impact)
        contributions.append(
            FeatureContribution(
                name=name, value=raw_value, impact=impact, description=description
            )
        )
    return contributions


def _rank_feature_contributions(
    contributions: list[FeatureContribution],
    top_n: int = 5,
) -> list[FeatureContribution]:
    """Rank feature contributions by absolute impact and return the top N.

    Args:
        contributions: All computed :class:`~src.explanation.types.FeatureContribution`
            instances.
        top_n: Maximum number of contributions to return (default 5).
            If fewer than *top_n* contributions are provided, all are returned.

    Returns:
        Up to *top_n* contributions sorted by ``abs(impact)`` descending.

    Example::

        >>> ranked = _rank_feature_contributions(contribs, top_n=3)
        >>> len(ranked) <= 3
        True
    """
    sorted_contribs = sorted(contributions, key=lambda c: abs(c.impact), reverse=True)
    return sorted_contribs[:top_n]


def _build_hold_highlights(vec: dict[str, float], top_n: int = 3) -> list[str]:
    """Build a list of the most prominent hold types for the narrative.

    Formats the top-N hold types by ratio as ``"crimps (40%)"``-style strings.
    Zero-ratio hold types are excluded.

    Args:
        vec: Feature vector from :meth:`~src.features.assembler.RouteFeatures.to_vector`.
        top_n: Maximum number of hold types to include (default 3).

    Returns:
        List of formatted strings, e.g. ``["crimps (40%)", "slopers (25%)"]``.
        Empty list if all ratios are zero.

    Example::

        >>> _build_hold_highlights(vec)
        ['crimps (40%)', 'slopers (25%)']
    """
    hold_keys = {
        "crimp_ratio": "crimps",
        "sloper_ratio": "slopers",
        "pinch_ratio": "pinches",
        "jug_ratio": "jugs",
        "pocket_ratio": "pockets",
        "foothold_ratio": "footholds",
        "unknown_ratio": "unknowns",
    }
    ranked = sorted(
        ((k, label) for k, label in hold_keys.items() if vec.get(k, 0.0) > 0.0),
        key=lambda pair: vec.get(pair[0], 0.0),
        reverse=True,
    )
    return [f"{label} ({round(vec[key] * 100)}%)" for key, label in ranked[:top_n]]


def _build_summary(
    grade: str,
    qualifier: str,
    highlights: list[str],
    top_features: list[FeatureContribution],
) -> str:
    """Build a 1–2 sentence natural language grade summary.

    Template::

        "This route is estimated at {grade}. The model is {qualifier},
        primarily driven by {top_feature} ({highlights})."

    Args:
        grade: V-scale grade label, e.g. ``"V3"``.
        qualifier: Confidence qualifier from :func:`_get_confidence_qualifier`.
        highlights: Hold type highlights from :func:`_build_hold_highlights`.
        top_features: Ranked feature contributions from
            :func:`_rank_feature_contributions`.

    Returns:
        A natural language summary string.

    Example::

        >>> _build_summary("V3", "confident", ["crimps (40%)"], top_features)
        'This route is estimated at V3. ...'
    """
    top_feature_name = (
        top_features[0].name if top_features else "overall route features"
    )
    highlight_str = ", ".join(highlights) if highlights else "varied hold types"
    return (
        f"This route is estimated at {grade}. "
        f"The model is {qualifier}, primarily driven by "
        f"{top_feature_name} ({highlight_str})."
    )


def _get_estimator_type(
    prediction: HeuristicGradeResult | MLGradeResult,
) -> Literal["heuristic", "ml"]:
    """Determine the estimator type string from a prediction result.

    Args:
        prediction: Either a :class:`~src.grading.heuristic.HeuristicGradeResult`
            or :class:`~src.grading.ml_estimator.MLGradeResult`.

    Returns:
        ``"ml"`` for :class:`~src.grading.ml_estimator.MLGradeResult`,
        ``"heuristic"`` for all other prediction types.

    Example::

        >>> _get_estimator_type(ml_result)
        'ml'
        >>> _get_estimator_type(heuristic_result)
        'heuristic'
    """
    if isinstance(prediction, MLGradeResult):
        return "ml"
    return "heuristic"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_explanation(
    features: RouteFeatures,
    prediction: HeuristicGradeResult | MLGradeResult,
) -> ExplanationResult:
    """Generate a structured explanation for a bouldering route grade estimate.

    Produces natural language narrative, ranked feature contributions, and a
    confidence qualifier from a :class:`~src.features.assembler.RouteFeatures`
    instance and a prediction result from either the heuristic or ML estimator.

    Args:
        features: Assembled :class:`~src.features.assembler.RouteFeatures`
            from :func:`~src.features.assembler.assemble_features`.
        prediction: Grade prediction from either
            :func:`~src.grading.heuristic.estimate_grade_heuristic` or
            :func:`~src.grading.ml_estimator.estimate_grade_ml`.

    Returns:
        :class:`~src.explanation.types.ExplanationResult` with grade,
        estimator type, confidence qualifier, top feature contributions,
        summary, and hold highlights.

    Raises:
        ExplanationError: If the feature vector cannot be assembled
            (wraps :class:`~src.features.exceptions.FeatureExtractionError`).

    Example::

        >>> result = generate_explanation(route_features, heuristic_result)
        >>> print(result.summary)
        This route is estimated at V3. The model is confident, ...
    """
    try:
        vec = features.to_vector()
    except FeatureExtractionError as exc:
        raise ExplanationError(
            f"Failed to build feature vector for explanation: {exc.message}"
        ) from exc

    logger.debug(
        "Generating explanation for grade=%s estimator=%s",
        prediction.grade,
        _get_estimator_type(prediction),
    )

    hold_contribs = _compute_hold_contributions(vec)
    geo_contribs = _compute_geometry_contributions(vec)
    top_features = _rank_feature_contributions(hold_contribs + geo_contribs)
    highlights = _build_hold_highlights(vec)
    qualifier = _get_confidence_qualifier(prediction.confidence)
    summary = _build_summary(prediction.grade, qualifier, highlights, top_features)
    estimator_type = _get_estimator_type(prediction)

    return ExplanationResult(
        grade=prediction.grade,
        estimator_type=estimator_type,
        confidence_qualifier=qualifier,
        top_features=top_features,
        summary=summary,
        hold_highlights=highlights,
    )
