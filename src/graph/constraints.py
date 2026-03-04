"""Start/finish constraint application for route graphs.

Applies start and finish hold constraints to a :class:`~src.graph.route_graph.RouteGraph`,
pruning holds in connected components that cannot participate in any start→finish
traversal. Surviving nodes are annotated with :data:`NODE_ATTR_IS_START` and
:data:`NODE_ATTR_IS_FINISH` boolean attributes on the NetworkX graph.

The graph is undirected (``nx.Graph``). A hold is retained if and only if it
belongs to a connected component that contains at least one ``start_id`` **and**
the ``finish_id``. All other holds (and their incident edges) are discarded.

Example::

    >>> from src.graph import build_route_graph, apply_route_constraints
    >>> rg = build_route_graph(holds, wall_angle=10.0)
    >>> constrained = apply_route_constraints(rg, start_ids=[0, 2], finish_id=7)
    >>> print(constrained.node_count, constrained.edge_count)
    5 8
"""

from typing import Final

import networkx as nx

from src.graph.exceptions import RouteGraphError
from src.graph.route_graph import RouteGraph
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

NODE_ATTR_IS_START: Final[str] = "is_start"
"""NetworkX node attribute key set to ``True`` on start hold nodes."""

NODE_ATTR_IS_FINISH: Final[str] = "is_finish"
"""NetworkX node attribute key set to ``True`` on finish hold nodes."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_constraint_ids(
    rg: RouteGraph,
    start_ids: list[int],
    start_id_set: set[int],
    finish_id: int,
) -> None:
    """Validate start_ids and finish_id against the graph before pruning.

    Checks in order:
    1. ``start_ids`` is non-empty.
    2. ``len(start_ids)`` does not exceed the graph node count.
    3. ``start_ids`` contains no duplicate values.
    4. Every value in ``start_ids`` is a node in the graph.
    5. ``finish_id`` is a node in the graph.

    The full node ID list is emitted at DEBUG level for server-side diagnostics
    when checks 4 or 5 fail; error messages expose only the node count to
    avoid leaking internal graph structure to callers.

    Args:
        rg: The route graph whose nodes are checked.
        start_ids: List of hold IDs designating start holds.
        start_id_set: Pre-computed ``set(start_ids)`` from the caller —
            avoids constructing the set a second time.
        finish_id: Hold ID designating the finish hold.

    Raises:
        RouteGraphError: If any of the five conditions above is violated.
    """
    if not start_ids:
        raise RouteGraphError("start_ids must not be empty")

    if len(start_ids) > rg.node_count:
        raise RouteGraphError(
            f"start_ids length ({len(start_ids)}) exceeds graph node count ({rg.node_count})"
        )

    if len(start_id_set) != len(start_ids):
        raise RouteGraphError("start_ids must not contain duplicate hold_id values")

    graph_nodes = rg.graph.nodes
    for sid in start_ids:
        if sid not in graph_nodes:
            logger.debug(
                "start_id %d not found; graph nodes: %s", sid, sorted(graph_nodes)
            )
            raise RouteGraphError(
                f"start_id {sid} not found in graph ({len(graph_nodes)} nodes)"
            )

    if finish_id not in graph_nodes:
        logger.debug(
            "finish_id %d not found; graph nodes: %s", finish_id, sorted(graph_nodes)
        )
        raise RouteGraphError(
            f"finish_id {finish_id} not found in graph ({len(graph_nodes)} nodes)"
        )


def _compute_keep_ids(
    rg: RouteGraph,
    start_id_set: set[int],
    finish_id: int,
) -> set[int]:
    """Identify nodes in components that contain both a start and the finish.

    Iterates over all connected components of the undirected graph. A component
    is retained when it contains at least one value from ``start_id_set`` **and**
    ``finish_id``.

    Args:
        rg: The route graph to analyse.
        start_id_set: Set of hold IDs designating start holds.
        finish_id: Hold ID designating the finish hold.

    Returns:
        Set of node IDs that survive pruning (may be empty if no valid
        component exists).
    """
    keep_ids: set[int] = set()
    for component in nx.connected_components(rg.graph):
        if any(sid in component for sid in start_id_set) and finish_id in component:
            keep_ids |= component
    return keep_ids


def _mark_start_finish(
    graph: nx.Graph,
    start_id_set: set[int],
    finish_id: int,
) -> None:
    """Set is_start / is_finish boolean attributes on all nodes in graph.

    Every node receives both attributes; their values are determined by whether
    the node ID is in ``start_id_set`` or equals ``finish_id``.

    Args:
        graph: The (pruned, mutable) NetworkX graph to annotate.
        start_id_set: Set of node IDs that are start holds.
        finish_id: Node ID of the finish hold.
    """
    for node in graph.nodes:
        graph.nodes[node][NODE_ATTR_IS_START] = node in start_id_set
        graph.nodes[node][NODE_ATTR_IS_FINISH] = node == finish_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_route_constraints(
    rg: RouteGraph,
    start_ids: list[int],
    finish_id: int,
) -> RouteGraph:
    """Prune a RouteGraph to holds reachable between designated start and finish.

    A hold survives iff it belongs to a connected component that contains at
    least one ``start_id`` **and** ``finish_id``. The returned :class:`RouteGraph`
    is a new object; the input ``rg`` is never mutated.

    Node attributes :data:`NODE_ATTR_IS_START` and :data:`NODE_ATTR_IS_FINISH`
    (both ``bool``) are written onto every node in the returned graph.

    Note:
        ``start_ids`` may include ``finish_id`` — a degenerate single-hold
        route where start and finish coincide is valid.

    Args:
        rg: Source :class:`RouteGraph` produced by
            :func:`~src.graph.route_graph.build_route_graph`.
        start_ids: Non-empty list of hold IDs for start holds.  All values
            must be present as graph nodes and must be unique.  Length must
            not exceed ``rg.node_count``.
        finish_id: Hold ID for the finish hold. Must be present as a graph
            node.

    Returns:
        A new :class:`RouteGraph` containing only the holds in components
        that include at least one start and the finish.  ``holds`` and
        ``graph.nodes`` are kept in sync.  All surviving edges from ``rg``
        are preserved with their original weights.

    Raises:
        RouteGraphError: If ``start_ids`` is empty, its length exceeds the
            graph node count, it contains duplicates, any ``start_id`` or
            ``finish_id`` is absent from the graph, or no connected component
            contains both a start and the finish.

    Example::

        >>> rg = build_route_graph(holds, wall_angle=10.0)
        >>> constrained = apply_route_constraints(rg, start_ids=[0], finish_id=7)
        >>> constrained.graph.nodes[0]["is_start"]
        True
    """
    start_id_set = set(start_ids)
    _validate_constraint_ids(rg, start_ids, start_id_set, finish_id)

    keep_ids = _compute_keep_ids(rg, start_id_set, finish_id)
    if not keep_ids:
        raise RouteGraphError(
            f"no path exists from any start hold {sorted(start_ids)} "
            f"to finish hold {finish_id} in the route graph"
        )

    pruned_graph: nx.Graph = rg.graph.subgraph(keep_ids).copy()
    _mark_start_finish(pruned_graph, start_id_set, finish_id)

    pruned_holds = [h for h in rg.holds if h.hold_id in keep_ids]

    logger.info(
        "apply_route_constraints: kept %d/%d holds (start_ids=%s finish_id=%d)",
        len(pruned_holds),
        len(rg.holds),
        sorted(start_ids),
        finish_id,
    )

    return RouteGraph(graph=pruned_graph, holds=pruned_holds, wall_angle=rg.wall_angle)
