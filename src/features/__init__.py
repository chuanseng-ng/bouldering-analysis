"""Feature extraction package for bouldering route analysis.

Public API::

    from src.features import (
        FeatureExtractionError,
        GeometryFeatures,
        extract_geometry_features,
        HoldFeatures,
        extract_hold_features,
    )
"""

from src.features.exceptions import FeatureExtractionError
from src.features.geometry import GeometryFeatures, extract_geometry_features
from src.features.holds import HoldFeatures, extract_hold_features

__all__ = [
    "FeatureExtractionError",
    "GeometryFeatures",
    "extract_geometry_features",
    "HoldFeatures",
    "extract_hold_features",
]
