"""Pydantic models for the explanation engine output.

Defines the structured data types returned by
:func:`~src.explanation.engine.generate_explanation`.

Example::

    >>> from src.explanation import FeatureContribution, ExplanationResult
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

_ConfidenceQualifier = Literal["very confident", "confident", "uncertain"]


class FeatureContribution(BaseModel):
    """A single feature's contribution to the grade estimate.

    Attributes:
        name: Human-readable feature name, e.g. ``"Crimp ratio"``.
        value: Raw feature value (ratio, distance, etc.).
        impact: Signed contribution: ``weight * normalized_value``.
            Positive values increase difficulty; negative values reduce it.
        description: One-sentence natural language description of the
            feature's effect on difficulty.

    Example::

        >>> fc = FeatureContribution(
        ...     name="Crimp ratio",
        ...     value=0.4,
        ...     impact=0.14,
        ...     description="40% of holds are crimps, adding significant difficulty.",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    impact: float
    description: str


class ExplanationResult(BaseModel):
    """Structured explanation for a bouldering route grade estimate.

    Attributes:
        grade: V-scale grade label, e.g. ``"V5"``.
        estimator_type: Which estimator produced the prediction
            (``"heuristic"`` or ``"ml"``).
        confidence_qualifier: Human-readable confidence level:
            ``"very confident"``, ``"confident"``, or ``"uncertain"``.
        top_features: Up to 5 :class:`FeatureContribution` instances
            ranked by absolute impact (highest first).
        summary: 1–2 sentence natural language grade summary.
        hold_highlights: Top hold types by ratio, formatted as
            ``"crimps (40%)"``; zero-ratio types are excluded.

    Example::

        >>> result = ExplanationResult(
        ...     grade="V3",
        ...     estimator_type="heuristic",
        ...     confidence_qualifier="confident",
        ...     top_features=[],
        ...     summary="This route is estimated at V3.",
        ...     hold_highlights=["crimps (40%)", "slopers (25%)"],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    grade: str
    estimator_type: Literal["heuristic", "ml"]
    confidence_qualifier: _ConfidenceQualifier
    top_features: list[FeatureContribution]
    summary: str
    hold_highlights: list[str]
