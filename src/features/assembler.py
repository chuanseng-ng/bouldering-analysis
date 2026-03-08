"""Feature vector assembly for bouldering route grade estimation.

Combines :class:`~src.features.geometry.GeometryFeatures` and
:class:`~src.features.holds.HoldFeatures` into a single
:class:`RouteFeatures` model.

NOTE: Feature normalization is intentionally absent. Normalization
requires training-data statistics (min/max or mean/std) that are
not available at this stage. It is deferred to PR-7, where each
estimator can apply normalization appropriate to its algorithm.
See plans/MIGRATION_PLAN.md PR-6.3 task 3 for context.

Example::

    >>> from src.features import assemble_features
    >>> rf = assemble_features(constrained_graph)
    >>> vec = rf.to_vector()
    >>> len(vec)
    34
"""

from pydantic import BaseModel, ConfigDict

from src.features.exceptions import FeatureExtractionError
from src.features.geometry import GeometryFeatures, extract_geometry_features
from src.features.holds import HoldFeatures, extract_hold_features
from src.graph.route_graph import RouteGraph
from src.logging_config import get_logger

logger = get_logger(__name__)


class RouteFeatures(BaseModel):
    """Assembled feature set for a bouldering route.

    Composes :class:`GeometryFeatures` (11 fields) and
    :class:`HoldFeatures` (23 fields) into a single immutable model.
    Use :meth:`to_vector` to obtain a flat ``dict[str, float]``
    suitable for ML estimators or JSON/JSONB serialisation.

    Attributes:
        geometry: Geometry features from the constrained RouteGraph.
        holds: Hold composition features from the classified holds.
    """

    model_config = ConfigDict(frozen=True)

    geometry: GeometryFeatures
    holds: HoldFeatures

    def to_vector(self) -> dict[str, float]:
        """Flatten into a dict suitable for ML model input.

        Merges geometry (11 keys) and holds (23 keys) into a single
        34-key dictionary.  All values are cast to :class:`float` for
        consistency with ML frameworks that expect uniform numeric types.

        NOTE: Values are raw (un-normalised). Normalisation is the
        responsibility of the consuming estimator in PR-7.

        Returns:
            ``dict[str, float]`` with one entry per feature field.
            Key names match the Pydantic field names of
            :class:`GeometryFeatures` and :class:`HoldFeatures`.

        Raises:
            FeatureExtractionError: If a sub-model field holds a
                non-numeric value (indicates model taxonomy drift).

        Example::

            >>> vec = rf.to_vector()
            >>> vec["avg_move_distance"]
            0.35
        """
        vec: dict[str, float] = {}
        for source in (self.geometry.model_dump(), self.holds.model_dump()):
            for k, v in source.items():
                if not isinstance(v, (int, float)):
                    raise FeatureExtractionError(
                        f"Feature field {k!r} has non-numeric value {v!r}; "
                        "sub-model taxonomy may have drifted"
                    )
                vec[k] = float(v)
        return vec


def assemble_features(graph: RouteGraph) -> RouteFeatures:
    """Extract and assemble geometry and hold features for a route.

    Calls :func:`~src.features.geometry.extract_geometry_features` and
    :func:`~src.features.holds.extract_hold_features` using the holds
    embedded in *graph* (``graph.holds``).  Any
    :class:`~src.features.exceptions.FeatureExtractionError` raised by
    the sub-extractors propagates unchanged to the caller.

    NOTE: *graph* must have been processed by
    :func:`~src.graph.constraints.apply_route_constraints` before
    calling this function.

    NOTE: Normalisation of the resulting feature vector is intentionally
    deferred to PR-7 (requires training-data statistics).

    Args:
        graph: Constrained :class:`~src.graph.route_graph.RouteGraph`
            with start/finish node attributes set.

    Returns:
        :class:`RouteFeatures` combining geometry and hold sub-models.

    Raises:
        FeatureExtractionError: If geometry or hold extraction fails.

    Example::

        >>> rf = assemble_features(constrained_graph)
        >>> rf.geometry.node_count
        8
        >>> rf.holds.total_count
        8
    """
    logger.debug("Assembling route features from graph with %d nodes", graph.node_count)
    geometry = extract_geometry_features(graph)
    hold_features = extract_hold_features(graph.holds)
    return RouteFeatures(geometry=geometry, holds=hold_features)
