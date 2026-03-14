"""Shared internal utilities for the grading package.

Not part of the public API — import directly where needed within
``src.grading``.  Do **not** re-export from ``src.grading.__init__``.
"""

from __future__ import annotations


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive range [lo, hi].

    Args:
        value: The value to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Clamped float value.

    Example::

        >>> _clamp(-0.5, 0.0, 1.0)
        0.0
        >>> _clamp(1.5, 0.0, 1.0)
        1.0
    """
    return max(lo, min(hi, value))


def _normalize_vector(
    vector: dict[str, float],
    mean: dict[str, float],
    std: dict[str, float],
) -> dict[str, float]:
    """Apply z-score normalization to a feature vector.

    For each key *k* in *vector*, computes ``(vector[k] - mean[k]) / std[k]``.
    If ``std[k] == 0.0``, a fallback of ``1.0`` is used to avoid
    division by zero (zero-variance feature → normalized value = vector[k] − mean[k]).

    All three dicts must share the same keys.

    Args:
        vector: Raw feature vector to normalize.
        mean: Per-feature training means, keyed by feature name.
        std: Per-feature training standard deviations, keyed by feature name.

    Returns:
        New ``dict[str, float]`` with z-scored values, same keys as *vector*.

    Raises:
        KeyError: If the key sets of *vector*, *mean*, and *std* are not identical
            (extra or missing keys in any dict).

    Example::

        >>> _normalize_vector({"x": 3.0}, {"x": 1.0}, {"x": 2.0})
        {'x': 1.0}
        >>> _normalize_vector({"x": 3.0}, {"x": 1.0}, {"x": 0.0})  # zero-variance
        {'x': 2.0}
    """
    v_keys, m_keys, s_keys = set(vector), set(mean), set(std)
    if v_keys != m_keys or v_keys != s_keys:
        raise KeyError(
            f"Feature key mismatch: vector has {v_keys}, "
            f"mean has {m_keys}, std has {s_keys}"
        )
    return {
        k: (vector[k] - mean[k]) / (std[k] if std[k] != 0.0 else 1.0) for k in vector
    }
