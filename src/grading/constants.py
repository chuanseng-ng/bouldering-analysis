"""Grade estimation constants for the heuristic estimator.

Defines the V-scale grade taxonomy, threshold mapping, and feature weights
used by :func:`~src.grading.heuristic.estimate_grade_heuristic`.
"""

from typing import Final

V_GRADES: Final[tuple[str, ...]] = (
    "V0",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
)

_N: Final[int] = len(V_GRADES)  # 18

GRADE_THRESHOLDS: Final[tuple[float, ...]] = tuple(i / _N for i in range(_N))
# (0.0, 0.0556, 0.1111, ..., 0.9444)

MAX_HOPS_NORM: Final[int] = 20  # empirical upper bound for path hop count

FEATURE_WEIGHTS: Final[dict[str, float]] = {
    # Hold difficulty sub-score (positive = harder, negative = easier)
    "crimp_ratio": 0.35,
    "sloper_ratio": 0.25,
    "pinch_ratio": 0.20,
    "jug_ratio": -0.30,
    "volume_ratio": 0.10,
    # Geometry difficulty sub-score (all in [0,1] after normalization)
    "avg_move_distance": 0.50,
    "max_move_distance": 0.30,
    "path_length_max_hops": 0.20,  # weight sum = 1.0 → sub-score in [0,1]
    # Mixing
    "hold_weight": 0.45,
    "geometry_weight": 0.55,
}
