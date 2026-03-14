"""XGBoost ML grade estimator for bouldering routes.

Loads a pre-trained :class:`~xgboost.XGBClassifier` from a versioned model
directory and predicts the V-scale grade of a route from its
:class:`~src.features.assembler.RouteFeatures`.

Model directories are expected to contain:

* ``model.pkl`` — serialised ``XGBClassifier`` (via :mod:`joblib`)
* ``metadata.json`` — training metadata including ``feature_names``,
  ``normalization_mean``, ``normalization_std``, ``n_classes``, and
  ``data_source``.

The loaded model is cached in-process by resolved path string so that
repeated calls with the same path do not reload from disk.  Call
:func:`_clear_model_cache` in test teardown to prevent cross-test
contamination.

Example::

    >>> from src.grading import estimate_grade_ml
    >>> result = estimate_grade_ml(route_features, model_path="models/grading/v20260310_120000")
    >>> print(result.grade, result.confidence)
    V4 0.73
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from xgboost import XGBClassifier

from src.features.assembler import RouteFeatures
from src.features.exceptions import FeatureExtractionError
from src.grading._utils import _clamp, _normalize_vector
from src.grading.constants import V_GRADES
from src.grading.exceptions import GradeEstimationError
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache: resolved_path → (classifier, metadata)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}

_N_GRADES = len(V_GRADES)  # 18


def _clear_model_cache() -> None:
    """Clear the in-process model cache.

    Intended for use in test teardown fixtures to prevent cross-test
    cache contamination when different tests write models to the same path.

    Example::

        >>> _clear_model_cache()
    """
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class MLGradeResult(BaseModel):
    """Grade estimation result from the ML estimator.

    Attributes:
        grade: V-scale grade label, e.g. ``"V5"``.
        grade_index: Ordinal index into :data:`~src.grading.constants.V_GRADES`
            (0 = V0, 17 = V17).
        confidence: Normalised entropy confidence: ``1 − H(p) / log(18)``.
            ``1.0`` means the model is certain; ``0.0`` means the
            distribution is perfectly uniform.
        difficulty_score: Probability-weighted mean grade index divided by 17.
            Smooth, soft estimate in ``[0, 1]``.
        grade_probabilities: Full softmax probability distribution over all
            18 V-grades, keyed by grade label (e.g. ``"V0"``–``"V17"``).
            Values sum to approximately 1.0.

    Example::

        >>> result = MLGradeResult(
        ...     grade="V4", grade_index=4, confidence=0.73,
        ...     difficulty_score=0.24,
        ...     grade_probabilities={"V0": 0.01, ..., "V17": 0.0},
        ... )
    """

    model_config = ConfigDict(frozen=True)

    grade: str
    grade_index: int = Field(ge=0, le=17)
    confidence: float = Field(ge=0.0, le=1.0)
    difficulty_score: float = Field(ge=0.0, le=1.0)
    grade_probabilities: dict[str, float]

    @model_validator(mode="after")
    def _validate_grade_consistency(self) -> "MLGradeResult":
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

    @model_validator(mode="after")
    def _validate_grade_probabilities(self) -> "MLGradeResult":
        """Validate that grade_probabilities contains exactly V_GRADES keys.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If the keys do not match V_GRADES exactly.
        """
        expected = set(V_GRADES)
        if set(self.grade_probabilities.keys()) != expected:
            raise ValueError(
                f"grade_probabilities must have exactly these keys: {expected}"
            )
        return self


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_confidence(probs: list[float]) -> float:
    """Compute normalised-entropy confidence from a probability distribution.

    Uses ``1 − H(p) / log(N)`` where ``H(p) = −Σ p_i log(p_i)`` and
    ``N = 18``.  The convention ``0 * log(0) = 0`` is enforced by skipping
    zero-probability entries.  Result is clamped to ``[0, 1]``.

    Args:
        probs: List of 18 non-negative probabilities summing to ~1.0.

    Returns:
        Confidence in ``[0, 1]``.  ``1.0`` when certain; ``0.0`` when uniform.

    Example::

        >>> _compute_confidence([1.0] + [0.0] * 17)
        1.0
        >>> _compute_confidence([1/18] * 18)
        0.0
    """
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    max_entropy = math.log(_N_GRADES)
    return _clamp(1.0 - entropy / max_entropy, 0.0, 1.0)


def _compute_difficulty_score(probs: list[float]) -> float:
    """Compute probability-weighted mean grade index as difficulty score.

    Args:
        probs: List of 18 non-negative probabilities, one per V-grade.

    Returns:
        Weighted mean grade index divided by 17, in ``[0, 1]``.

    Example::

        >>> _compute_difficulty_score([1.0] + [0.0] * 17)  # V0
        0.0
        >>> _compute_difficulty_score([0.0] * 17 + [1.0])  # V17
        1.0
    """
    weighted_mean = sum(i * p for i, p in enumerate(probs))
    return _clamp(weighted_mean / (_N_GRADES - 1), 0.0, 1.0)


def _load_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load model and metadata from *model_dir*, using the in-process cache.

    .. warning::
        ``model.pkl`` is deserialized with :func:`joblib.load`, which uses
        pickle internally.  **Never** pass a path that originates from
        untrusted user input or an untrusted network location.  Only load
        models that were produced by :func:`~src.training.train_grade_estimator.train_grade_estimator`
        from a trusted source.

    Args:
        model_dir: Path to a versioned model directory containing
            ``model.pkl`` and ``metadata.json``.  Must point to a trusted
            location — never derived from user-controlled input.

    Returns:
        Tuple of ``(classifier, metadata_dict)``.

    Raises:
        GradeEstimationError: If the model or metadata file is missing,
            cannot be parsed, or metadata contains invalid values.
    """
    cache_key = str(model_dir.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise GradeEstimationError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise GradeEstimationError(f"Metadata file not found: {metadata_path}")

    try:
        classifier = joblib.load(model_path)
    except Exception as exc:
        raise GradeEstimationError(
            f"Failed to load model from {model_path}: {exc}"
        ) from exc

    if not isinstance(classifier, XGBClassifier):
        raise GradeEstimationError(
            f"Expected XGBClassifier in {model_path}, "
            f"got {type(classifier).__name__}. Model file may be corrupt or tampered."
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata: dict[str, Any] = json.load(fh)
    except Exception as exc:
        raise GradeEstimationError(
            f"Failed to parse metadata from {metadata_path}: {exc}"
        ) from exc

    for required in (
        "feature_names",
        "normalization_mean",
        "normalization_std",
        "n_classes",
        "classes",
    ):
        if required not in metadata:
            raise GradeEstimationError(
                f"metadata.json is missing required field: {required!r}"
            )

    # Validate that stored class indices are in the legal V-grade range.
    classes: list[int] = metadata.get("classes", [])
    invalid = [c for c in classes if not (0 <= c <= 17)]
    if invalid:
        raise GradeEstimationError(
            f"metadata.json 'classes' contains out-of-range grade indices: {invalid}. "
            "Valid range is [0, 17]."
        )

    _MODEL_CACHE[cache_key] = (classifier, metadata)
    return classifier, metadata


def _predict_grade(
    classifier: Any,
    feature_vec: "np.ndarray[Any, Any]",
    trained_classes: list[int],
) -> tuple[int, list[float]]:
    """Run inference and map trained-class probabilities to full 18-grade distribution.

    Args:
        classifier: Fitted XGBClassifier instance.
        feature_vec: Float32 array of shape (1, n_features).
        trained_classes: Grade indices seen during training (e.g. [1, 2, 3, 4]).

    Returns:
        Tuple of ``(grade_index, full_probs)`` where *grade_index* is the
        best V-grade index (0–17) and *full_probs* is a 18-entry list with
        0.0 for grades not seen during training.
    """
    proba_matrix = classifier.predict_proba(feature_vec)
    trained_probs: list[float] = proba_matrix[0].tolist()
    n_trained = len(trained_classes)
    if len(trained_probs) != n_trained:
        raise GradeEstimationError(
            f"predict_proba returned {len(trained_probs)} probability columns, "
            f"expected {n_trained} (trained classes: {trained_classes}). "
            "Model and metadata may be out of sync."
        )

    # Map trained-class probabilities back to the full 18-grade distribution.
    # Grades not seen during training receive probability 0.0.
    full_probs = [0.0] * _N_GRADES
    for local_idx, grade_idx in enumerate(trained_classes):
        full_probs[grade_idx] = trained_probs[local_idx]

    best_local_idx = int(np.argmax(proba_matrix[0]))
    grade_index = trained_classes[best_local_idx]
    return grade_index, full_probs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_grade_ml(
    features: RouteFeatures,
    model_path: Path | str,
) -> MLGradeResult:
    """Estimate the V-scale grade of a bouldering route using the ML model.

    Loads (or retrieves from cache) the XGBClassifier at *model_path*,
    normalises the 34-field feature vector using training statistics stored
    in ``metadata.json``, and returns a full probability distribution over
    V0–V17.

    Args:
        features: Assembled :class:`~src.features.assembler.RouteFeatures`
            from :func:`~src.features.assembler.assemble_features`.
        model_path: Path to the versioned model directory (contains
            ``model.pkl`` and ``metadata.json``).

    Returns:
        :class:`MLGradeResult` with the estimated grade, grade index,
        normalised-entropy confidence, probability-weighted difficulty score,
        and full grade probability distribution.

    Raises:
        GradeEstimationError: If feature extraction fails, the model cannot
            be loaded, feature names do not match the trained model, or any
            other estimation error occurs.

    Example::

        >>> result = estimate_grade_ml(route_features, "models/grading/v20260310_120000")
        >>> print(result.grade, result.confidence)
        V4 0.73
    """
    model_dir = Path(model_path)

    try:
        vector = features.to_vector()
    except FeatureExtractionError as exc:
        raise GradeEstimationError(
            f"Failed to build feature vector for ML grade estimation: {exc.message}"
        ) from exc

    classifier, metadata = _load_model(model_dir)

    # Validate feature contract against training metadata
    expected_names: list[str] = metadata["feature_names"]
    actual_names = list(vector.keys())
    if actual_names != expected_names:
        raise GradeEstimationError(
            f"Feature vector has {len(actual_names)} features with keys "
            f"{actual_names!r}, but model was trained on {len(expected_names)} "
            f"features {expected_names!r}. RouteFeatures schema may have drifted."
        )

    # trained_classes: grade indices seen during training (may be a subset of 0–17)
    trained_classes: list[int] = metadata["classes"]

    # Warn when running a model trained on synthetic data
    if metadata.get("data_source") == "synthetic":
        logger.warning(
            "ML grade estimator is using a model trained on synthetic data "
            "(labels from heuristic estimator). Predictions may be biased. "
            "Replace with a real-data model for production use."
        )

    # Normalise using training statistics (z-score, std=1 fallback)
    norm_mean: dict[str, float] = metadata["normalization_mean"]
    norm_std: dict[str, float] = metadata["normalization_std"]
    normalised = _normalize_vector(vector, norm_mean, norm_std)

    # Build numpy array in training feature order and predict
    feature_vec = np.array([[normalised[k] for k in expected_names]], dtype=np.float32)
    grade_index, full_probs = _predict_grade(classifier, feature_vec, trained_classes)
    grade = V_GRADES[grade_index]
    confidence = _compute_confidence(full_probs)
    difficulty_score = _compute_difficulty_score(full_probs)
    grade_probabilities = {V_GRADES[i]: full_probs[i] for i in range(_N_GRADES)}

    logger.debug(
        "ML grade estimate: grade=%s confidence=%.4f difficulty=%.4f",
        grade,
        confidence,
        difficulty_score,
    )

    return MLGradeResult(
        grade=grade,
        grade_index=grade_index,
        confidence=confidence,
        difficulty_score=difficulty_score,
        grade_probabilities=grade_probabilities,
    )
