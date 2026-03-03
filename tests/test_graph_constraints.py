"""Tests for src.graph.constraints module.

Covers:
- src/graph/constraints.py — apply_route_constraints, NODE_ATTR_IS_START,
  NODE_ATTR_IS_FINISH, and all validation / pruning / marking logic.
"""

import logging

import networkx as nx
import pytest

from src.graph.constraints import (
    NODE_ATTR_IS_FINISH,
    NODE_ATTR_IS_START,
    apply_route_constraints,
)
from src.graph.exceptions import RouteGraphError
from src.graph.route_graph import RouteGraph, build_route_graph
from tests.conftest import make_classified_hold_for_tests as _make_classified_hold


def _make_line_graph(n: int) -> RouteGraph:
    """Build a fully-connected line graph with n holds spaced 0.1 apart.

    Holds are placed at (0.1 * i, 0.5) for i in range(n).
    Spacing 0.1 < BASE_REACH_RADIUS (0.35), so every adjacent pair is
    connected, and the graph is a single connected component.

    Args:
        n: Number of holds (must be >= 1).

    Returns:
        A RouteGraph with n nodes in one connected component.
    """
    holds = [
        _make_classified_hold(hold_id=i, x_center=0.1 * i, y_center=0.5)
        for i in range(n)
    ]
    return build_route_graph(holds, wall_angle=0.0)


def _make_disconnected_graph() -> RouteGraph:
    """Build a disconnected graph with two separate components.

    Layout (wall_angle=0, reach=0.35):
    - Component A: holds 0,1 at x=0.10, 0.20  (gap 0.10 < 0.35 → connected)
    - Component B: holds 2,3 at x=0.70, 0.80  (gap 0.10 < 0.35 → connected)
    - Gap between components: |0.70 - 0.20| = 0.50 > 0.35 → disconnected

    Returns:
        A RouteGraph with components {0, 1} and {2, 3}.
    """
    holds = [
        _make_classified_hold(hold_id=0, x_center=0.10, y_center=0.5),
        _make_classified_hold(hold_id=1, x_center=0.20, y_center=0.5),
        _make_classified_hold(hold_id=2, x_center=0.70, y_center=0.5),
        _make_classified_hold(hold_id=3, x_center=0.80, y_center=0.5),
    ]
    return build_route_graph(holds, wall_angle=0.0)


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsConstants
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsConstants:
    """Tests for NODE_ATTR_IS_START and NODE_ATTR_IS_FINISH constants."""

    def test_node_attr_is_start_is_string(self) -> None:
        """NODE_ATTR_IS_START must be a str."""
        assert isinstance(NODE_ATTR_IS_START, str)

    def test_node_attr_is_finish_is_string(self) -> None:
        """NODE_ATTR_IS_FINISH must be a str."""
        assert isinstance(NODE_ATTR_IS_FINISH, str)

    def test_node_attr_constants_are_distinct(self) -> None:
        """NODE_ATTR_IS_START and NODE_ATTR_IS_FINISH must be different strings."""
        assert NODE_ATTR_IS_START != NODE_ATTR_IS_FINISH


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsValidation
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsValidation:
    """Tests for apply_route_constraints input validation (all error branches)."""

    def test_empty_start_ids_raises_route_graph_error(self) -> None:
        """Empty start_ids list raises RouteGraphError."""
        rg = _make_line_graph(3)
        with pytest.raises(RouteGraphError):
            apply_route_constraints(rg, start_ids=[], finish_id=2)

    def test_empty_start_ids_error_message_mentions_empty(self) -> None:
        """Empty start_ids error message mentions 'empty'."""
        rg = _make_line_graph(3)
        with pytest.raises(RouteGraphError, match="empty"):
            apply_route_constraints(rg, start_ids=[], finish_id=2)

    def test_duplicate_start_ids_raises_route_graph_error(self) -> None:
        """Duplicate values in start_ids raise RouteGraphError."""
        rg = _make_line_graph(3)
        with pytest.raises(RouteGraphError):
            apply_route_constraints(rg, start_ids=[0, 0], finish_id=2)

    def test_duplicate_start_ids_error_message_mentions_duplicate(self) -> None:
        """Duplicate start_ids error message mentions 'duplicate'."""
        rg = _make_line_graph(3)
        with pytest.raises(RouteGraphError, match="duplicate"):
            apply_route_constraints(rg, start_ids=[0, 0], finish_id=2)

    def test_start_ids_length_exceeds_node_count_raises(self) -> None:
        """start_ids longer than graph node count raises RouteGraphError."""
        rg = _make_line_graph(3)  # nodes: 0, 1, 2 (node_count=3)
        with pytest.raises(RouteGraphError, match="exceeds"):
            apply_route_constraints(rg, start_ids=[0, 1, 2, 99], finish_id=2)

    def test_start_ids_length_at_node_count_boundary_accepted(self) -> None:
        """start_ids with len == node_count is accepted (boundary)."""
        rg = _make_line_graph(3)  # nodes: 0, 1, 2 (node_count=3)
        result = apply_route_constraints(rg, start_ids=[0, 1, 2], finish_id=2)
        assert isinstance(result, RouteGraph)

    def test_start_id_not_in_graph_raises_route_graph_error(self) -> None:
        """start_id absent from graph nodes raises RouteGraphError."""
        rg = _make_line_graph(3)  # nodes: 0, 1, 2
        with pytest.raises(RouteGraphError):
            apply_route_constraints(rg, start_ids=[99], finish_id=2)

    def test_start_id_not_in_graph_error_message_mentions_id(self) -> None:
        """Missing start_id error message mentions the missing id."""
        rg = _make_line_graph(3)
        with pytest.raises(RouteGraphError, match="99"):
            apply_route_constraints(rg, start_ids=[99], finish_id=2)

    def test_finish_id_not_in_graph_raises_route_graph_error(self) -> None:
        """finish_id absent from graph nodes raises RouteGraphError."""
        rg = _make_line_graph(3)  # nodes: 0, 1, 2
        with pytest.raises(RouteGraphError):
            apply_route_constraints(rg, start_ids=[0], finish_id=99)

    def test_finish_id_not_in_graph_error_message_mentions_id(self) -> None:
        """Missing finish_id error message mentions the missing id."""
        rg = _make_line_graph(3)
        with pytest.raises(RouteGraphError, match="99"):
            apply_route_constraints(rg, start_ids=[0], finish_id=99)

    def test_no_path_between_start_and_finish_raises_route_graph_error(self) -> None:
        """No path from start to finish in disconnected graph raises RouteGraphError."""
        rg = _make_disconnected_graph()  # {0,1} and {2,3}
        with pytest.raises(RouteGraphError):
            apply_route_constraints(rg, start_ids=[0], finish_id=2)

    def test_no_path_error_message_mentions_start_and_finish_ids(self) -> None:
        """No-path error message mentions start_ids and finish_id."""
        rg = _make_disconnected_graph()
        with pytest.raises(RouteGraphError, match="2"):
            apply_route_constraints(rg, start_ids=[0], finish_id=2)


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsReturnType
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsReturnType:
    """Tests for apply_route_constraints return type and field preservation."""

    def test_returns_route_graph_instance(self) -> None:
        """apply_route_constraints returns a RouteGraph instance."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert isinstance(result, RouteGraph)

    def test_returns_new_object_not_same_as_input(self) -> None:
        """The returned RouteGraph is a new object (not the input)."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result is not rg

    def test_wall_angle_preserved(self) -> None:
        """wall_angle in the returned RouteGraph matches the input RouteGraph."""
        holds = [
            _make_classified_hold(hold_id=i, x_center=0.1 * i, y_center=0.5)
            for i in range(3)
        ]
        rg = build_route_graph(holds, wall_angle=30.0)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result.wall_angle == pytest.approx(30.0)

    def test_holds_are_subset_of_input_holds(self) -> None:
        """Holds in the result are a subset of input RouteGraph.holds."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        result_ids = {h.hold_id for h in result.holds}
        input_ids = {h.hold_id for h in rg.holds}
        assert result_ids <= input_ids

    def test_graph_is_nx_graph_instance(self) -> None:
        """The returned RouteGraph.graph is an nx.Graph."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert isinstance(result.graph, nx.Graph)


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsPruning
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsPruning:
    """Tests for pruning logic: which nodes and edges survive."""

    def test_connected_component_with_start_and_finish_is_kept(self) -> None:
        """Component containing both a start and the finish is kept."""
        rg = _make_disconnected_graph()  # {0,1} and {2,3}
        result = apply_route_constraints(rg, start_ids=[0], finish_id=1)
        assert 0 in result.graph.nodes
        assert 1 in result.graph.nodes

    def test_disconnected_component_without_finish_is_pruned(self) -> None:
        """Component that does not contain finish_id is pruned."""
        rg = _make_disconnected_graph()  # {0,1} and {2,3}
        # start in {0,1}, finish in {2,3} → but wait, no path, so we need
        # a case where start covers both components but finish is in only one.
        # start_ids=[0,2], finish_id=3 → {2,3} kept; {0,1} pruned
        result = apply_route_constraints(rg, start_ids=[0, 2], finish_id=3)
        assert 0 not in result.graph.nodes
        assert 1 not in result.graph.nodes

    def test_disconnected_component_without_start_is_pruned(self) -> None:
        """Component that does not contain any start_id is pruned."""
        rg = _make_disconnected_graph()  # {0,1} and {2,3}
        # start in {0,1}, finish in {0,1} → {2,3} has no start and no finish → pruned
        result = apply_route_constraints(rg, start_ids=[0], finish_id=1)
        assert 2 not in result.graph.nodes
        assert 3 not in result.graph.nodes

    def test_holds_and_nodes_remain_in_sync_after_pruning(self) -> None:
        """After pruning: result.holds hold_ids match result.graph.nodes exactly."""
        rg = _make_disconnected_graph()
        result = apply_route_constraints(rg, start_ids=[0, 2], finish_id=3)
        hold_ids = {h.hold_id for h in result.holds}
        assert hold_ids == set(result.graph.nodes)

    def test_edges_between_kept_nodes_are_preserved(self) -> None:
        """Edges between kept nodes from input graph are present in result."""
        rg = _make_line_graph(4)  # 0-1-2-3 all connected
        result = apply_route_constraints(rg, start_ids=[0], finish_id=3)
        # All nodes kept; original edges must survive
        assert result.edge_count == rg.edge_count

    def test_input_route_graph_not_mutated(self) -> None:
        """apply_route_constraints does not modify the input RouteGraph.holds."""
        rg = _make_disconnected_graph()
        original_node_count = rg.node_count
        original_hold_count = len(rg.holds)
        apply_route_constraints(rg, start_ids=[0, 2], finish_id=3)
        assert rg.node_count == original_node_count
        assert len(rg.holds) == original_hold_count

    def test_input_graph_nodes_unchanged_after_call(self) -> None:
        """apply_route_constraints does not remove nodes from the input graph."""
        rg = _make_disconnected_graph()
        original_nodes = set(rg.graph.nodes)
        apply_route_constraints(rg, start_ids=[0, 2], finish_id=3)
        assert set(rg.graph.nodes) == original_nodes

    def test_all_nodes_kept_when_graph_is_fully_connected(self) -> None:
        """When all nodes are reachable, no nodes are pruned."""
        rg = _make_line_graph(4)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=3)
        assert result.node_count == rg.node_count

    def test_pruned_holds_preserve_original_order(self) -> None:
        """Surviving holds appear in the same relative order as in input holds."""
        rg = _make_disconnected_graph()
        result = apply_route_constraints(rg, start_ids=[0], finish_id=1)
        # Only {0, 1} survive; order must be preserved
        assert [h.hold_id for h in result.holds] == [0, 1]

    def test_edge_weights_preserved_after_pruning(self) -> None:
        """Edge weights in the pruned graph match those in the original graph.

        apply_route_constraints docstring promises: "All surviving edges from rg
        are preserved with their original weights."
        """
        rg = _make_line_graph(4)  # 0-1-2-3 all connected
        result = apply_route_constraints(rg, start_ids=[0], finish_id=3)
        for u, v, data in result.graph.edges(data=True):
            assert data["weight"] == pytest.approx(rg.graph[u][v]["weight"])


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsStartFinishMarking
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsStartFinishMarking:
    """Tests for is_start / is_finish node attribute marking."""

    def test_start_node_has_is_start_true(self) -> None:
        """The start node carries is_start=True as a node attribute."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result.graph.nodes[0][NODE_ATTR_IS_START] is True

    def test_finish_node_has_is_finish_true(self) -> None:
        """The finish node carries is_finish=True as a node attribute."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result.graph.nodes[2][NODE_ATTR_IS_FINISH] is True

    def test_non_start_nodes_have_is_start_false(self) -> None:
        """Non-start nodes carry is_start=False."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result.graph.nodes[1][NODE_ATTR_IS_START] is False
        assert result.graph.nodes[2][NODE_ATTR_IS_START] is False

    def test_non_finish_nodes_have_is_finish_false(self) -> None:
        """Non-finish nodes carry is_finish=False."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result.graph.nodes[0][NODE_ATTR_IS_FINISH] is False
        assert result.graph.nodes[1][NODE_ATTR_IS_FINISH] is False

    def test_is_start_attribute_is_bool(self) -> None:
        """The is_start node attribute is a Python bool."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert isinstance(result.graph.nodes[0][NODE_ATTR_IS_START], bool)

    def test_is_finish_attribute_is_bool(self) -> None:
        """The is_finish node attribute is a Python bool."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert isinstance(result.graph.nodes[2][NODE_ATTR_IS_FINISH], bool)

    def test_multiple_start_ids_all_marked_as_start(self) -> None:
        """When multiple start_ids are given, all are marked is_start=True."""
        rg = _make_line_graph(4)
        result = apply_route_constraints(rg, start_ids=[0, 1], finish_id=3)
        assert result.graph.nodes[0][NODE_ATTR_IS_START] is True
        assert result.graph.nodes[1][NODE_ATTR_IS_START] is True

    def test_input_graph_nodes_not_contaminated_with_start_finish_attrs(self) -> None:
        """Calling apply_route_constraints does not add is_start/is_finish to input graph nodes."""
        rg = _make_line_graph(3)
        apply_route_constraints(rg, start_ids=[0], finish_id=2)
        for node in rg.graph.nodes:
            assert NODE_ATTR_IS_START not in rg.graph.nodes[node]
            assert NODE_ATTR_IS_FINISH not in rg.graph.nodes[node]

    def test_marking_does_not_clobber_existing_node_attrs(self) -> None:
        """Marking is_start/is_finish does not overwrite pre-existing hold attributes."""
        rg = _make_line_graph(3)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        for node_id in result.graph.nodes:
            attrs = result.graph.nodes[node_id]
            assert "x_center" in attrs, f"node {node_id} missing x_center"
            assert "y_center" in attrs, f"node {node_id} missing y_center"
            assert "hold_type" in attrs, f"node {node_id} missing hold_type"
            assert "detection_confidence" in attrs, (
                f"node {node_id} missing detection_confidence"
            )


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsEdgeCases
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsEdgeCases:
    """Edge cases: single hold, start==finish, order, edge count."""

    def test_single_hold_graph_start_equals_finish_is_valid(self) -> None:
        """A single-hold route where start_id == finish_id is accepted."""
        rg = _make_line_graph(1)  # node: 0
        result = apply_route_constraints(rg, start_ids=[0], finish_id=0)
        assert isinstance(result, RouteGraph)
        assert result.node_count == 1

    def test_start_id_equals_finish_id_is_not_an_error(self) -> None:
        """start_ids containing finish_id is valid (degenerate single-hold route)."""
        rg = _make_line_graph(3)
        # start_ids=[2], finish_id=2: degenerate — node 2 is both start and finish
        result = apply_route_constraints(rg, start_ids=[2], finish_id=2)
        assert isinstance(result, RouteGraph)

    def test_start_and_finish_same_node_has_both_attrs_true(self) -> None:
        """When start == finish, that node has is_start=True AND is_finish=True."""
        rg = _make_line_graph(1)
        result = apply_route_constraints(rg, start_ids=[0], finish_id=0)
        assert result.graph.nodes[0][NODE_ATTR_IS_START] is True
        assert result.graph.nodes[0][NODE_ATTR_IS_FINISH] is True

    def test_order_of_surviving_holds_matches_original_index_order(self) -> None:
        """After pruning, surviving holds are in the same index order as input."""
        rg = _make_line_graph(5)  # nodes: 0,1,2,3,4
        result = apply_route_constraints(rg, start_ids=[0], finish_id=4)
        ids = [h.hold_id for h in result.holds]
        assert ids == sorted(ids)  # original order is already ascending

    def test_edge_count_after_pruning_is_correct(self) -> None:
        """After pruning one component, only intra-component edges survive."""
        rg = _make_disconnected_graph()  # {0,1} and {2,3}; each has 1 edge
        result = apply_route_constraints(rg, start_ids=[0], finish_id=1)
        # Only component {0,1} kept; 1 edge between them
        assert result.edge_count == 1

    def test_multiple_start_ids_only_reachable_component_kept(self) -> None:
        """With start_ids spanning components, only the component reaching finish is kept."""
        rg = _make_disconnected_graph()  # {0,1} and {2,3}
        # start_ids=[0, 2]; finish_id=3: component {2,3} matches; {0,1} does not
        result = apply_route_constraints(rg, start_ids=[0, 2], finish_id=3)
        assert set(result.graph.nodes) == {2, 3}

    def test_no_path_when_start_and_finish_in_different_components(self) -> None:
        """start_ids=[0] and finish_id=2 in disconnected graph → RouteGraphError."""
        rg = _make_disconnected_graph()
        with pytest.raises(RouteGraphError):
            apply_route_constraints(rg, start_ids=[0], finish_id=2)


# ---------------------------------------------------------------------------
# TestApplyRouteConstraintsIntegration
# ---------------------------------------------------------------------------


class TestApplyRouteConstraintsIntegration:
    """Integration tests: full pipeline and idempotent chaining."""

    def test_full_pipeline_build_route_graph_then_apply_constraints(self) -> None:
        """build_route_graph → apply_route_constraints works end to end."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=3, x_center=0.8, y_center=0.5),  # isolated
        ]
        rg = build_route_graph(holds, wall_angle=0.0)
        # {0,1,2} are connected; {3} is isolated
        result = apply_route_constraints(rg, start_ids=[0], finish_id=2)
        assert result.node_count == 3
        assert 3 not in result.graph.nodes
        assert result.graph.nodes[0][NODE_ATTR_IS_START] is True
        assert result.graph.nodes[2][NODE_ATTR_IS_FINISH] is True

    def test_idempotent_chaining_produces_same_result(self) -> None:
        """Applying constraints twice with the same parameters yields the same graph."""
        rg = _make_line_graph(4)
        first = apply_route_constraints(rg, start_ids=[0], finish_id=3)
        second = apply_route_constraints(first, start_ids=[0], finish_id=3)
        assert set(second.graph.nodes) == set(first.graph.nodes)
        assert second.node_count == first.node_count
        assert second.edge_count == first.edge_count
        # Marking should be consistent
        assert second.graph.nodes[0][NODE_ATTR_IS_START] is True
        assert second.graph.nodes[3][NODE_ATTR_IS_FINISH] is True

    def test_apply_route_constraints_logs_held_hold_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """apply_route_constraints emits an INFO log mentioning the kept hold count."""
        rg = _make_disconnected_graph()  # 4 holds; pruning to {0,1} keeps 2
        with caplog.at_level(logging.INFO, logger="src.graph.constraints"):
            apply_route_constraints(rg, start_ids=[0], finish_id=1)
        assert any("2" in msg and "4" in msg for msg in caplog.messages), (
            f"Expected INFO log mentioning 2 kept from 4. Messages: {caplog.messages}"
        )
