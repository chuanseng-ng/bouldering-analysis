"""Route graph construction from classified holds.

This module builds a weighted undirected graph where nodes are classified
climbing holds and edges represent feasible moves between holds, based on
spatial reachability heuristics.

The graph produced here is consumed by:

- **PR-5.2** ``apply_route_constraints`` — prunes unreachable nodes and
  returns a new :class:`RouteGraph` keeping ``holds`` and ``graph.nodes``
  in sync.
- **PR-6.x** Feature Extraction — computes path statistics and hold metrics.
- **PR-7.x** Grade Estimation — predicts V-scale difficulty.

Example::

    >>> from src.graph import build_route_graph, make_classified_hold
    >>> holds = [make_classified_hold(i, det, clf) for i, (det, clf) in enumerate(pairs)]
    >>> rg = build_route_graph(holds, wall_angle=15.0)
    >>> print(rg.node_count, rg.edge_count)
    12 28
"""

import math
from typing import Final

import networkx as nx
from pydantic import BaseModel, ConfigDict, model_validator

from src.constants import MAX_HOLD_COUNT
from src.graph.exceptions import RouteGraphError
from src.graph.types import ClassifiedHold
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tunable reachability constants.
# BASE_REACH_RADIUS: empirical estimate of average climbing reach in
# normalized image coordinates. Assumes the route image captures roughly
# 2–3 metres of wall width; 0.35 ≈ 0.7–1.0 m reach. Calibrate with
# labelled route data once available.
BASE_REACH_RADIUS: Final[float] = 0.35

# WALL_ANGLE_REACH_SCALE: fractional change in reach per unit of
# sin(wall_angle). 0.2 means a full slab (90°) adds 20% reach vs vertical.
# Convention: -15° = overhang (reduced reach), 0° = vertical, 90° = slab
# (increased reach, gravity-assisted footwork).
WALL_ANGLE_REACH_SCALE: Final[float] = 0.2

WALL_ANGLE_MIN: Final[float] = -15.0  # steep overhang
WALL_ANGLE_MAX: Final[float] = 90.0  # full slab

_LARGE_HOLD_COUNT_THRESHOLD: Final[int] = 100

# Hard upper limit — imported from src.constants so all modules share the cap.


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class RouteGraph(BaseModel):
    """A weighted undirected graph of classified climbing holds.

    Produced by :func:`build_route_graph`. Nodes are keyed by
    ``hold.hold_id`` and carry all :class:`~src.graph.types.ClassifiedHold`
    fields (except ``hold_id`` itself) as node attributes. Edges carry a
    ``weight`` equal to the Euclidean distance between hold centres.

    Note:
        The wrapped ``graph`` (``nx.Graph``) is mutable. Do not modify it
        after construction — :func:`~src.graph.route_graph.build_route_graph`
        is the sole authorised constructor. PR-5.2 (``apply_route_constraints``)
        will produce a *new* ``RouteGraph`` when pruning nodes, keeping
        ``holds`` and ``graph.nodes`` consistent.

    Attributes:
        graph: NetworkX undirected graph with hold nodes and move edges.
        holds: Snapshot of the input :class:`~src.graph.types.ClassifiedHold`
            list, in the same order as passed to :func:`build_route_graph`.
            Always consistent with ``graph.nodes`` at construction time.
        wall_angle: Wall inclination in degrees used to build this graph.
            -15 = steep overhang, 0 = vertical, 90 = full slab.

    Example:
        >>> from src.graph.types import make_classified_hold
        >>> from src.graph.route_graph import build_route_graph
        >>> from src.inference.detection import DetectedHold
        >>> from src.inference.classification import HoldTypeResult
        >>> hold = make_classified_hold(0, detection, classification)  # ClassifiedHold
        >>> route = build_route_graph([hold], wall_angle=0.0)
        >>> route.wall_angle
        0.0
        >>> route.holds[0].hold_type
        'jug'
        >>> route.graph.number_of_nodes()
        1
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: nx.Graph
    holds: list[ClassifiedHold]
    wall_angle: float

    @model_validator(mode="after")
    def validate_graph_holds_consistency(self) -> "RouteGraph":
        """Enforce that graph nodes and holds are consistent at construction time.

        Verifies:
        1. ``graph`` is an ``nx.Graph`` instance (guards against direct
           construction bypassing :func:`build_route_graph`).
        2. The set of node IDs in ``graph`` exactly equals
           ``{h.hold_id for h in holds}`` — stronger than a count check,
           this catches the case where counts match but IDs differ
           (e.g. graph nodes {0, 2} vs hold_ids {0, 1}).

        Returns:
            The validated ``RouteGraph`` instance.

        Raises:
            ValueError: If the graph type is wrong, holds contains duplicate
                hold_id values, or graph node IDs do not match hold_id values.
        """
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("graph must be an nx.Graph instance")
        hold_ids = {h.hold_id for h in self.holds}
        if len(hold_ids) != len(self.holds):
            raise ValueError(
                f"holds contains duplicate hold_id values; all {len(self.holds)} "
                f"entries in RouteGraph.holds must have a unique h.hold_id"
            )
        graph_node_ids = set(self.graph.nodes())
        if graph_node_ids != hold_ids:
            raise ValueError(
                f"graph node IDs {sorted(graph_node_ids)} must match "
                f"hold_id values {sorted(hold_ids)}"
            )
        return self

    @property
    def node_count(self) -> int:
        """Return the number of hold nodes in the graph.

        Returns:
            Number of nodes in the underlying NetworkX graph.
        """
        return int(self.graph.number_of_nodes())

    @property
    def edge_count(self) -> int:
        """Return the number of move edges in the graph.

        Returns:
            Number of edges in the underlying NetworkX graph.
        """
        return int(self.graph.number_of_edges())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_effective_reach(wall_angle: float) -> float:
    """Compute the effective reachability radius for a given wall angle.

    The formula scales ``BASE_REACH_RADIUS`` by a sine-based factor derived
    from the wall inclination.  On a slab (positive angle) gravity assists
    footwork, allowing the climber to reach higher; on an overhang (negative
    angle) reach is slightly reduced.

    Args:
        wall_angle: Wall inclination in degrees.  Positive = slab,
            negative = overhang, 0 = vertical.  Expected range [-15, 90].

    Returns:
        Effective reach radius in normalised image coordinates.
    """
    return BASE_REACH_RADIUS * (
        1.0 + WALL_ANGLE_REACH_SCALE * math.sin(math.radians(wall_angle))
    )


def _euclidean_distance(h1: ClassifiedHold, h2: ClassifiedHold) -> float:
    """Return the Euclidean distance between two hold centres.

    Coordinates are in normalised image space [0, 1].

    Args:
        h1: First classified hold.
        h2: Second classified hold.

    Returns:
        Euclidean distance between hold centres.
    """
    dx = h1.x_center - h2.x_center
    dy = h1.y_center - h2.y_center
    return math.sqrt(dx * dx + dy * dy)


def _add_graph_nodes(graph: nx.Graph, holds: list[ClassifiedHold]) -> None:
    """Add classified holds as nodes in the NetworkX graph.

    Each hold becomes a node keyed by ``hold.hold_id``.  All
    :class:`~src.graph.types.ClassifiedHold` fields except ``hold_id``
    are stored as node attributes via ``model_dump``, so adding new fields
    to ``ClassifiedHold`` automatically propagates here without code changes.

    Args:
        graph: The NetworkX graph to populate.
        holds: Classified holds to add as nodes.
    """
    for h in holds:
        graph.add_node(h.hold_id, **h.model_dump(exclude={"hold_id"}))


def _add_graph_edges(
    graph: nx.Graph,
    holds: list[ClassifiedHold],
    reach_sq: float,
) -> None:
    """Add reachability edges between holds within the reach threshold.

    Uses squared-distance comparison in the inner loop to avoid a ``sqrt``
    call per pair (``sqrt`` is only called when an edge is confirmed).

    Complexity: O(n²) where n = ``len(holds)``.  Designed for n ≤ 40;
    acceptable up to ~100.  A warning is emitted by :func:`build_route_graph`
    when n exceeds :data:`_LARGE_HOLD_COUNT_THRESHOLD`.

    Args:
        graph: The NetworkX graph to populate with edges.
        holds: Classified holds; nodes must already exist in ``graph``.
        reach_sq: Square of the effective reach radius.  Pairs within this
            squared distance receive an edge.
    """
    n = len(holds)
    for i in range(n):
        for j in range(i + 1, n):
            dx = holds[i].x_center - holds[j].x_center
            dy = holds[i].y_center - holds[j].y_center
            dist_sq = dx * dx + dy * dy
            # Exclude coincident holds (dist_sq == 0): a zero-weight edge has
            # no physical meaning and can cause division-by-zero in downstream
            # path algorithms (PR-6.x).
            if 0.0 < dist_sq <= reach_sq:
                graph.add_edge(
                    holds[i].hold_id,
                    holds[j].hold_id,
                    weight=math.sqrt(dist_sq),
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_route_graph(
    holds: list[ClassifiedHold],
    wall_angle: float = 0.0,
) -> RouteGraph:
    """Build a movement graph from classified climbing holds.

    Creates an undirected weighted graph where each node is a classified hold
    and each edge represents a physically feasible move between two holds.
    Edge weight is the Euclidean distance between hold centres in normalised
    image coordinates.  Difficulty modifiers based on hold type are deferred
    to PR-6.x Feature Extraction and are not encoded here.

    Complexity: O(n²) for edge construction, where n = ``len(holds)``.
    Designed for n ≤ 40 holds per route; logs a warning if n > 100.

    Args:
        holds: Non-empty list of :class:`~src.graph.types.ClassifiedHold`
            objects, typically produced by
            :func:`~src.graph.types.make_classified_hold`.
        wall_angle: Wall inclination in degrees.
            ``-15`` = steep overhang, ``0`` = vertical (default),
            ``90`` = full slab.  Affects the effective reach radius used
            to determine which hold pairs are connected.

    Returns:
        A :class:`RouteGraph` with:
        - One node per hold, keyed by ``hold_id`` with all hold fields as
          node attributes.
        - One edge between each pair of holds within effective reach,
          weighted by Euclidean distance.
        - ``holds`` matching the input list (a shallow copy).
        - ``wall_angle`` matching the provided value.

    Raises:
        RouteGraphError: If ``holds`` is empty or ``wall_angle`` is outside
            ``[WALL_ANGLE_MIN, WALL_ANGLE_MAX]``.

    Example::

        >>> rg = build_route_graph(classified_holds, wall_angle=10.0)
        >>> print(rg.node_count, rg.edge_count)
        15 31
    """
    if not holds:
        raise RouteGraphError("holds must not be empty")

    if len(holds) > MAX_HOLD_COUNT:
        raise RouteGraphError(
            f"hold count {len(holds)} exceeds the maximum {MAX_HOLD_COUNT}; "
            "pass a smaller hold list or increase MAX_HOLD_COUNT in src/constants.py"
        )

    if wall_angle < WALL_ANGLE_MIN or wall_angle > WALL_ANGLE_MAX:
        raise RouteGraphError(
            f"wall_angle must be in [{WALL_ANGLE_MIN}, {WALL_ANGLE_MAX}], "
            f"got {wall_angle}"
        )

    hold_ids = [h.hold_id for h in holds]
    if len(set(hold_ids)) != len(hold_ids):
        raise RouteGraphError("hold_id values must be unique across all holds")

    if len(holds) > _LARGE_HOLD_COUNT_THRESHOLD:
        logger.warning(
            "build_route_graph received %d holds (> %d threshold); "
            "edge construction is O(n^2) and may be slow for large inputs.",
            len(holds),
            _LARGE_HOLD_COUNT_THRESHOLD,
        )

    effective_reach = _compute_effective_reach(wall_angle)
    reach_sq = effective_reach * effective_reach

    graph: nx.Graph = nx.Graph()
    _add_graph_nodes(graph, holds)
    _add_graph_edges(graph, holds, reach_sq)

    return RouteGraph(
        graph=graph,
        holds=list(holds),
        wall_angle=wall_angle,
    )
