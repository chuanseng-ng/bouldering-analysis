"""Explanation engine for bouldering route grade estimates.

Public API::

    from src.explanation import (
        ExplanationError,
        FeatureContribution,
        ExplanationResult,
        generate_explanation,
    )

Example::

    >>> from src.explanation import generate_explanation
    >>> result = generate_explanation(route_features, prediction)
    >>> print(result.summary)
"""

from src.explanation.engine import generate_explanation
from src.explanation.exceptions import ExplanationError
from src.explanation.types import ExplanationResult, FeatureContribution

__all__ = [
    "ExplanationError",
    "FeatureContribution",
    "ExplanationResult",
    "generate_explanation",
]
