"""Geometry feature extraction from constrained route graphs.

Computes interpretable geometry metrics from a constrained
:class:`~src.graph.route_graph.RouteGraph`. The graph must have been processed
by :func:`~src.graph.constraints.apply_route_constraints` before calling
:func:`extract_geometry_features`; absence of start/finish node attributes
raises :class:`~src.features.exceptions.FeatureExtractionError`.

The resulting :class:`GeometryFeatures` feeds directly into the grade estimator
(Milestone 7) and the explainability engine (Milestone 8).

Example::

    >>> from src.graph import build_route_graph, apply_route_constraints
    >>> from src.features import extract_geometry_features
    >>> rg = build_route_graph(holds, wall_angle=0.0)
    >>> crg = apply_route_constraints(rg, start_ids=[0], finish_id=2)
    >>> gf = extract_geometry_features(crg)
    >>> print(gf.node_count, gf.path_length_min_distance)
    3 0.4
"""

import math

import networkx as nx
from pydantic import BaseModel, Field

from src.features.exceptions import FeatureExtractionError
from src.graph import NODE_ATTR_IS_FINISH, NODE_ATTR_IS_START, RouteGraph
from src.graph.types import ClassifiedHold
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class GeometryFeatures(BaseModel):
    """Geometry features extracted from a constrained RouteGraph.

    All fields are non-negative.  Path statistics are computed over Dijkstra
    shortest paths from each start hold to the single finish hold.  Edge
    statistics describe the distribution of individual move distances over the
    entire graph.

    Attributes:
        avg_move_distance: Mean edge weight across all graph edges.
            ``0.0`` when the graph has no edges.
        max_move_distance: Maximum edge weight.  ``0.0`` when no edges.
        min_move_distance: Minimum edge weight.  ``0.0`` when no edges.
        std_move_distance: Population standard deviation of edge weights.
            ``0.0`` when fewer than 2 edges.
        path_length_min_distance: Weighted shortest-path distance from the
            nearest start hold to the finish hold.
        path_length_min_hops: Number of moves along the minimum-distance path.
        path_length_max_distance: Weighted shortest-path distance from the
            farthest start hold to the finish hold.
        path_length_max_hops: Number of moves along the maximum-distance path.
        hold_density: ``node_count / bounding_box_area`` in holds per
            normalised image unit².  ``0.0`` when the bounding box area is
            zero (fewer than 2 holds or all holds collinear along one axis).
        node_count: Number of hold nodes in the constrained graph.
        edge_count: Number of move edges in the constrained graph.

    Example::

        >>> gf = extract_geometry_features(constrained_rg)
        >>> print(gf.avg_move_distance, gf.node_count)
        0.2 3
    """

    avg_move_distance: float = Field(ge=0.0)
    max_move_distance: float = Field(ge=0.0)
    min_move_distance: float = Field(ge=0.0)
    std_move_distance: float = Field(ge=0.0)
    path_length_min_distance: float = Field(ge=0.0)
    path_length_min_hops: int = Field(ge=0)
    path_length_max_distance: float = Field(ge=0.0)
    path_length_max_hops: int = Field(ge=0)
    hold_density: float = Field(ge=0.0)
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _edge_weights(graph: nx.Graph) -> list[float]:
    """Extract all edge weights from the graph as a list.

    Materialises edge weights as a list because multiple aggregates (avg,
    max, min, std) require multiple passes over the data.

    Args:
        graph: NetworkX graph whose edges carry a ``weight`` attribute.

    Returns:
        List of float weights, one per edge.  Empty when the graph has
        no edges.
    """
    return [data["weight"] for _, _, data in graph.edges(data=True)]


def _compute_edge_stats(weights: list[float]) -> tuple[float, float, float, float]:
    """Compute (avg, max, min, population_std) from a list of edge weights.

    Uses ``math.fsum`` for compensated floating-point summation and
    ``math.sqrt`` for the standard deviation — no NumPy dependency.

    Args:
        weights: List of non-negative float edge weights.

    Returns:
        4-tuple ``(avg, max, min, std)`` where all values are ``float``.
        Returns ``(0.0, 0.0, 0.0, 0.0)`` for an empty list.
        Returns ``(w, w, w, 0.0)`` for a single-element list.
        Population standard deviation (``ddof=0``) for 2 or more weights.
    """
    if not weights:
        return (0.0, 0.0, 0.0, 0.0)
    n = len(weights)
    if n == 1:
        w = weights[0]
        return (w, w, w, 0.0)
    avg = math.fsum(weights) / n
    variance = math.fsum((w - avg) ** 2 for w in weights) / n
    std = math.sqrt(variance)
    return (avg, max(weights), min(weights), std)


def _find_constraints(graph: nx.Graph) -> tuple[set[int], int]:
    """Extract validated start and finish node IDs from graph attributes.

    Reads :data:`~src.graph.NODE_ATTR_IS_START` and
    :data:`~src.graph.NODE_ATTR_IS_FINISH` boolean node attributes written by
    :func:`~src.graph.constraints.apply_route_constraints`.

    Args:
        graph: NetworkX graph with ``is_start`` / ``is_finish`` attributes
            on every node.

    Returns:
        2-tuple ``(start_ids, finish_id)`` where ``start_ids`` is a
        non-empty set of integer node IDs and ``finish_id`` is a single
        integer node ID.

    Raises:
        FeatureExtractionError: If no nodes are marked as start, if no
            nodes are marked as finish, or if more than one node is
            marked as finish.
    """
    start_ids: set[int] = {
        node for node in graph.nodes if graph.nodes[node].get(NODE_ATTR_IS_START, False)
    }
    finish_nodes: list[int] = [
        node
        for node in graph.nodes
        if graph.nodes[node].get(NODE_ATTR_IS_FINISH, False)
    ]

    if not start_ids:
        raise FeatureExtractionError(
            "Graph has no start nodes; apply_route_constraints must be called first"
        )
    if len(finish_nodes) == 0:
        raise FeatureExtractionError(
            "Graph has no finish node; apply_route_constraints must be called first"
        )
    if len(finish_nodes) > 1:
        raise FeatureExtractionError(
            f"Expected exactly 1 finish node, found {len(finish_nodes)}"
        )

    return (start_ids, finish_nodes[0])


def _compute_path_stats(
    graph: nx.Graph,
    start_ids: set[int],
    finish_id: int,
) -> tuple[float, int, float, int]:
    """Compute min/max weighted shortest-path statistics from starts to finish.

    Issues one ``nx.single_source_dijkstra`` call per start hold, which
    returns both the weighted distance and the node path in a single
    graph traversal.

    If no path exists from a particular start to the finish, that start
    is skipped and a DEBUG message is emitted.  If all starts fail,
    ``(0.0, 0, 0.0, 0)`` is returned with a WARNING.

    Args:
        graph: NetworkX graph with ``weight`` edge attributes.
        start_ids: Set of node IDs designating start holds.  All values
            are assumed to be present as nodes in ``graph``.
        finish_id: Node ID of the finish hold.  Assumed to be present.

    Returns:
        4-tuple ``(min_distance, min_hops, max_distance, max_hops)`` where
        min/max are keyed on weighted path distance.
    """
    path_results: list[tuple[float, int]] = []

    for start in start_ids:
        try:
            distance, path = nx.single_source_dijkstra(
                graph, start, finish_id, weight="weight"
            )
            hops = len(path) - 1
            path_results.append((distance, hops))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.debug("No path from start %d to finish %d", start, finish_id)

    if not path_results:
        logger.warning(
            "No path found from any of %d start holds to finish; returning zeros",
            len(start_ids),
        )
        return (0.0, 0, 0.0, 0)

    min_result = min(path_results, key=lambda x: x[0])
    max_result = max(path_results, key=lambda x: x[0])
    return (min_result[0], min_result[1], max_result[0], max_result[1])


def _compute_hold_density(holds: list[ClassifiedHold]) -> float:
    """Compute hold density as ``node_count / bounding_box_area``.

    Uses 4 generator expressions for ``min/max(x/y_center)``.  Returns
    ``0.0`` when the bounding box area is zero or non-positive, which
    occurs when fewer than 2 holds are present or when all holds share the
    same x-coordinate or y-coordinate (collinear case).

    Args:
        holds: List of classified holds with normalised ``x_center`` and
            ``y_center`` coordinates in ``[0, 1]``.

    Returns:
        Density in holds per normalised image unit².  ``0.0`` for degenerate
        inputs (fewer than 2 holds, or zero-area bounding box).
    """
    if len(holds) < 2:
        return 0.0

    min_x = min(h.x_center for h in holds)
    max_x = max(h.x_center for h in holds)
    min_y = min(h.y_center for h in holds)
    max_y = max(h.y_center for h in holds)

    bbox_area = (max_x - min_x) * (max_y - min_y)
    if bbox_area <= 0.0:
        return 0.0

    return len(holds) / bbox_area


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_geometry_features(graph: RouteGraph) -> GeometryFeatures:
    """Extract geometry features from a constrained RouteGraph.

    The graph must have been processed by
    :func:`~src.graph.constraints.apply_route_constraints` so that every node
    carries :data:`~src.graph.NODE_ATTR_IS_START` and
    :data:`~src.graph.NODE_ATTR_IS_FINISH` boolean attributes.  Calling this
    function on an unconstrained graph raises
    :class:`~src.features.exceptions.FeatureExtractionError`.

    Args:
        graph: A :class:`~src.graph.route_graph.RouteGraph` produced by
            :func:`~src.graph.route_graph.build_route_graph` and constrained
            by :func:`~src.graph.constraints.apply_route_constraints`.

    Returns:
        A :class:`GeometryFeatures` instance with all fields populated.

    Raises:
        FeatureExtractionError: If ``graph.holds`` is empty, or if the graph
            has no start nodes, no finish node, or more than one finish node.

    Example::

        >>> crg = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        >>> gf = extract_geometry_features(crg)
        >>> assert gf.node_count == 3
        >>> assert gf.path_length_min_distance > 0
    """
    if not graph.holds:
        raise FeatureExtractionError("Cannot extract features from empty graph")

    start_ids, finish_id = _find_constraints(graph.graph)

    weights = _edge_weights(graph.graph)
    avg, mx, mn, std = _compute_edge_stats(weights)
    min_dist, min_hops, max_dist, max_hops = _compute_path_stats(
        graph.graph, start_ids, finish_id
    )
    density = _compute_hold_density(graph.holds)

    return GeometryFeatures(
        avg_move_distance=avg,
        max_move_distance=mx,
        min_move_distance=mn,
        std_move_distance=std,
        path_length_min_distance=min_dist,
        path_length_min_hops=min_hops,
        path_length_max_distance=max_dist,
        path_length_max_hops=max_hops,
        hold_density=density,
        node_count=graph.node_count,
        edge_count=graph.edge_count,
    )
