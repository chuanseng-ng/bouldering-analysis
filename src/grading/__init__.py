"""Grade estimation package for bouldering route analysis.

Public API::

    from src.grading import (
        GradeEstimationError,
        HeuristicGradeResult,
        MLGradeResult,
        estimate_grade_heuristic,
        estimate_grade_ml,
    )

Note:
    :mod:`src.grading.constants` is intentionally not re-exported here.
    Import it directly when needed for explainability (e.g. PR-8)::

        from src.grading.constants import V_GRADES, FEATURE_WEIGHTS
"""

from src.grading.exceptions import GradeEstimationError
from src.grading.heuristic import HeuristicGradeResult, estimate_grade_heuristic
from src.grading.ml_estimator import MLGradeResult, estimate_grade_ml

__all__ = [
    "GradeEstimationError",
    "HeuristicGradeResult",
    "MLGradeResult",
    "estimate_grade_heuristic",
    "estimate_grade_ml",
]
