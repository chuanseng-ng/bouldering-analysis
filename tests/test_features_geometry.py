"""Tests for src.features.geometry module.

Covers:
- src/features/exceptions.py  — FeatureExtractionError
- src/features/geometry.py    — GeometryFeatures model, extract_geometry_features(),
                                 and all private helpers
"""

import math

import networkx as nx
import pytest

from src.features.exceptions import FeatureExtractionError
from src.features.geometry import (
    GeometryFeatures,
    _compute_edge_stats,
    _compute_hold_density,
    _compute_path_stats,
    _edge_weights,
    _find_constraints,
    extract_geometry_features,
)
from src.graph.constraints import (
    NODE_ATTR_IS_FINISH,
    NODE_ATTR_IS_START,
    apply_route_constraints,
)
from src.graph.route_graph import RouteGraph, build_route_graph
from src.graph.types import ClassifiedHold
from tests.conftest import make_classified_hold_for_tests as _make_classified_hold


# ---------------------------------------------------------------------------
# Local test helpers
# ---------------------------------------------------------------------------


def _make_constrained_graph(
    holds: list[ClassifiedHold],
    start_ids: list[int],
    finish_id: int,
    wall_angle: float = 0.0,
) -> RouteGraph:
    """Build a constrained RouteGraph for geometry tests.

    Args:
        holds: List of ClassifiedHold instances.
        start_ids: List of hold IDs designating start holds.
        finish_id: Hold ID designating the finish hold.
        wall_angle: Wall inclination in degrees (default 0.0).

    Returns:
        A constrained RouteGraph with start/finish attributes set.
    """
    rg = build_route_graph(holds, wall_angle)
    return apply_route_constraints(rg, list(start_ids), finish_id)


def _make_annotated_nx_graph(
    nodes: list[int],
    edges: list[tuple[int, int, float]],
    start_ids: set[int],
    finish_ids: set[int],
) -> nx.Graph:
    """Build a NetworkX graph with is_start/is_finish attributes set directly.

    Args:
        nodes: List of node IDs to add.
        edges: List of (u, v, weight) tuples.
        start_ids: Set of node IDs to mark as start.
        finish_ids: Set of node IDs to mark as finish.

    Returns:
        A NetworkX graph with the given attributes.
    """
    g = nx.Graph()
    for node in nodes:
        g.add_node(
            node,
            **{
                NODE_ATTR_IS_START: node in start_ids,
                NODE_ATTR_IS_FINISH: node in finish_ids,
            },
        )
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)
    return g


# ---------------------------------------------------------------------------
# TestFeatureExtractionError
# ---------------------------------------------------------------------------


class TestFeatureExtractionError:
    """Tests for FeatureExtractionError exception."""

    def test_is_value_error_subclass(self) -> None:
        """FeatureExtractionError must subclass ValueError."""
        assert issubclass(FeatureExtractionError, ValueError)

    def test_message_attribute_set(self) -> None:
        """FeatureExtractionError must expose a message attribute."""
        err = FeatureExtractionError("some error")
        assert err.message == "some error"

    def test_str_representation(self) -> None:
        """str(err) must equal the message."""
        err = FeatureExtractionError("test message")
        assert str(err) == "test message"

    def test_can_be_caught_as_value_error(self) -> None:
        """FeatureExtractionError must be catchable as ValueError."""
        with pytest.raises(ValueError, match="catch me"):
            raise FeatureExtractionError("catch me")

    def test_can_be_caught_directly(self) -> None:
        """FeatureExtractionError must be directly catchable."""
        with pytest.raises(FeatureExtractionError, match="direct"):
            raise FeatureExtractionError("direct")


# ---------------------------------------------------------------------------
# TestGeometryFeaturesModel
# ---------------------------------------------------------------------------


class TestGeometryFeaturesModel:
    """Tests for GeometryFeatures Pydantic model."""

    def _make_valid_features(self) -> GeometryFeatures:
        """Return a valid GeometryFeatures instance for structural tests."""
        return GeometryFeatures(
            avg_move_distance=0.2,
            max_move_distance=0.3,
            min_move_distance=0.1,
            std_move_distance=0.05,
            path_length_min_distance=0.5,
            path_length_min_hops=2,
            path_length_max_distance=0.8,
            path_length_max_hops=3,
            hold_density=5.0,
            node_count=5,
            edge_count=4,
        )

    def test_all_fields_present(self) -> None:
        """All expected fields must be present on the model."""
        expected_fields = {
            "avg_move_distance",
            "max_move_distance",
            "min_move_distance",
            "std_move_distance",
            "path_length_min_distance",
            "path_length_min_hops",
            "path_length_max_distance",
            "path_length_max_hops",
            "hold_density",
            "node_count",
            "edge_count",
        }
        assert set(GeometryFeatures.model_fields.keys()) == expected_fields

    def test_float_fields_are_float(self) -> None:
        """Float fields must be float instances."""
        features = self._make_valid_features()
        float_fields = [
            "avg_move_distance",
            "max_move_distance",
            "min_move_distance",
            "std_move_distance",
            "path_length_min_distance",
            "path_length_max_distance",
            "hold_density",
        ]
        for field in float_fields:
            assert isinstance(getattr(features, field), float), f"{field} not float"

    def test_int_fields_are_int(self) -> None:
        """Integer fields must be int instances."""
        features = self._make_valid_features()
        int_fields = [
            "path_length_min_hops",
            "path_length_max_hops",
            "node_count",
            "edge_count",
        ]
        for field in int_fields:
            assert isinstance(getattr(features, field), int), f"{field} not int"

    def test_zero_values_accepted(self) -> None:
        """All fields at zero must be accepted (no non-negative constraint violations)."""
        features = GeometryFeatures(
            avg_move_distance=0.0,
            max_move_distance=0.0,
            min_move_distance=0.0,
            std_move_distance=0.0,
            path_length_min_distance=0.0,
            path_length_min_hops=0,
            path_length_max_distance=0.0,
            path_length_max_hops=0,
            hold_density=0.0,
            node_count=0,
            edge_count=0,
        )
        assert features.node_count == 0


# ---------------------------------------------------------------------------
# TestEdgeWeights
# ---------------------------------------------------------------------------


class TestEdgeWeights:
    """Tests for _edge_weights private helper."""

    def test_zero_edge_graph_returns_empty_list(self) -> None:
        """_edge_weights on a graph with no edges must return []."""
        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        assert _edge_weights(g) == []

    def test_single_edge_returns_single_weight(self) -> None:
        """_edge_weights on a 1-edge graph must return a list with one weight."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.25)
        result = _edge_weights(g)
        assert result == pytest.approx([0.25])

    def test_multi_edge_returns_all_weights(self) -> None:
        """_edge_weights on a multi-edge graph must return all edge weights."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.1)
        g.add_edge(1, 2, weight=0.3)
        g.add_edge(0, 2, weight=0.2)
        result = sorted(_edge_weights(g))
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_result_is_list(self) -> None:
        """_edge_weights must return a list."""
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.5)
        assert isinstance(_edge_weights(g), list)


# ---------------------------------------------------------------------------
# TestEdgeStats
# ---------------------------------------------------------------------------


class TestEdgeStats:
    """Tests for _compute_edge_stats private helper."""

    def test_empty_list_returns_zeros(self) -> None:
        """Empty weight list must return (0.0, 0.0, 0.0, 0.0)."""
        result = _compute_edge_stats([])
        assert result == (0.0, 0.0, 0.0, 0.0)

    def test_single_weight_std_is_zero(self) -> None:
        """Single weight must give identical avg/max/min and zero std."""
        avg, mx, mn, std = _compute_edge_stats([0.3])
        assert avg == pytest.approx(0.3)
        assert mx == pytest.approx(0.3)
        assert mn == pytest.approx(0.3)
        assert std == pytest.approx(0.0)

    def test_two_equal_weights_std_is_zero(self) -> None:
        """Two equal weights must produce zero population std."""
        avg, mx, mn, std = _compute_edge_stats([0.2, 0.2])
        assert avg == pytest.approx(0.2)
        assert mx == pytest.approx(0.2)
        assert mn == pytest.approx(0.2)
        assert std == pytest.approx(0.0)

    def test_two_different_weights_correct_std(self) -> None:
        """Two different weights [0.1, 0.3] must yield population std=0.1."""
        # avg = 0.2, variance = ((0.1-0.2)^2 + (0.3-0.2)^2) / 2 = 0.01 → std = 0.1
        avg, mx, mn, std = _compute_edge_stats([0.1, 0.3])
        assert avg == pytest.approx(0.2)
        assert mx == pytest.approx(0.3)
        assert mn == pytest.approx(0.1)
        assert std == pytest.approx(0.1)

    def test_three_weights_avg_correct(self) -> None:
        """Three weights must compute correct average."""
        avg, mx, mn, std = _compute_edge_stats([0.1, 0.2, 0.3])
        assert avg == pytest.approx(0.2)
        assert mx == pytest.approx(0.3)
        assert mn == pytest.approx(0.1)

    def test_three_weights_std_correct(self) -> None:
        """Three weights [0.1, 0.2, 0.3] must compute correct population std."""
        # avg=0.2, variance = (0.01 + 0.0 + 0.01)/3 = 0.02/3
        _, _, _, std = _compute_edge_stats([0.1, 0.2, 0.3])
        expected_std = math.sqrt((0.01 + 0.0 + 0.01) / 3)
        assert std == pytest.approx(expected_std)

    def test_returns_tuple_of_four(self) -> None:
        """_compute_edge_stats must always return a 4-tuple."""
        result = _compute_edge_stats([0.5])
        assert isinstance(result, tuple)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# TestFindConstraints
# ---------------------------------------------------------------------------


class TestFindConstraints:
    """Tests for _find_constraints private helper."""

    def test_no_start_nodes_raises(self) -> None:
        """Graph with no is_start=True nodes must raise FeatureExtractionError."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1],
            edges=[(0, 1, 0.2)],
            start_ids=set(),
            finish_ids={1},
        )
        with pytest.raises(FeatureExtractionError) as exc_info:
            _find_constraints(g)
        assert "no start nodes" in exc_info.value.message
        assert "apply_route_constraints" in exc_info.value.message

    def test_no_finish_nodes_raises(self) -> None:
        """Graph with no is_finish=True nodes must raise FeatureExtractionError."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1],
            edges=[(0, 1, 0.2)],
            start_ids={0},
            finish_ids=set(),
        )
        with pytest.raises(FeatureExtractionError) as exc_info:
            _find_constraints(g)
        assert "no finish node" in exc_info.value.message
        assert "apply_route_constraints" in exc_info.value.message

    def test_more_than_one_finish_raises(self) -> None:
        """Graph with >1 is_finish=True nodes must raise FeatureExtractionError."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1, 2],
            edges=[(0, 1, 0.2), (1, 2, 0.2)],
            start_ids={0},
            finish_ids={1, 2},
        )
        with pytest.raises(FeatureExtractionError) as exc_info:
            _find_constraints(g)
        assert "2" in exc_info.value.message

    def test_one_start_one_finish_returns_correctly(self) -> None:
        """Graph with 1 start and 1 finish must return (start_set, finish_id)."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1, 2],
            edges=[(0, 1, 0.2), (1, 2, 0.2)],
            start_ids={0},
            finish_ids={2},
        )
        start_ids, finish_id = _find_constraints(g)
        assert start_ids == {0}
        assert finish_id == 2

    def test_two_starts_one_finish_returns_correctly(self) -> None:
        """Graph with 2 starts and 1 finish must return both starts in set."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1, 2, 3],
            edges=[(0, 2, 0.2), (1, 2, 0.2), (2, 3, 0.2)],
            start_ids={0, 1},
            finish_ids={3},
        )
        start_ids, finish_id = _find_constraints(g)
        assert start_ids == {0, 1}
        assert finish_id == 3

    def test_returns_set_and_int(self) -> None:
        """_find_constraints must return a (set, int) tuple."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1],
            edges=[(0, 1, 0.2)],
            start_ids={0},
            finish_ids={1},
        )
        start_ids, finish_id = _find_constraints(g)
        assert isinstance(start_ids, set)
        assert isinstance(finish_id, int)


# ---------------------------------------------------------------------------
# TestPathStats
# ---------------------------------------------------------------------------


class TestPathStats:
    """Tests for _compute_path_stats private helper."""

    def test_single_start_direct_path(self) -> None:
        """Single start with direct edge to finish returns correct distance/hops."""
        g = _make_annotated_nx_graph(
            nodes=[0, 1],
            edges=[(0, 1, 0.25)],
            start_ids={0},
            finish_ids={1},
        )
        min_dist, min_hops, max_dist, max_hops = _compute_path_stats(g, {0}, 1)
        assert min_dist == pytest.approx(0.25)
        assert min_hops == 1
        assert max_dist == pytest.approx(0.25)
        assert max_hops == 1

    def test_single_start_two_hop_path(self) -> None:
        """Single start via intermediate node returns correct 2-hop stats."""
        # 0 → 1 → 2, weights 0.2 and 0.2, total = 0.4, hops = 2
        g = nx.Graph()
        g.add_edge(0, 1, weight=0.2)
        g.add_edge(1, 2, weight=0.2)
        min_dist, min_hops, max_dist, max_hops = _compute_path_stats(g, {0}, 2)
        assert min_dist == pytest.approx(0.4)
        assert min_hops == 2
        assert max_dist == pytest.approx(0.4)
        assert max_hops == 2

    def test_multi_start_min_less_than_max(self) -> None:
        """Multi-start must yield min_distance < max_distance at known positions.

        Layout (wall_angle=0, reach=0.35):
        - hold 0 (start_A): x=0.2, y=0.5  → distance to finish = 0.2
        - hold 1 (finish): x=0.4, y=0.5
        - hold 2 (start_B): x=0.7, y=0.5  → distance to finish = 0.3

        Paths:
        - 0→1: direct, distance=0.2, hops=1
        - 2→1: direct, distance=0.3, hops=1
        """
        g = _make_annotated_nx_graph(
            nodes=[0, 1, 2],
            edges=[(0, 1, 0.2), (1, 2, 0.3)],
            start_ids={0, 2},
            finish_ids={1},
        )
        min_dist, min_hops, max_dist, max_hops = _compute_path_stats(g, {0, 2}, 1)
        assert min_dist == pytest.approx(0.2)
        assert min_hops == 1
        assert max_dist == pytest.approx(0.3)
        assert max_hops == 1
        assert min_dist < max_dist

    def test_all_starts_no_path_returns_zeros(self) -> None:
        """When no start can reach finish, must return (0.0, 0, 0.0, 0)."""
        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        # No edges — no path from 0 to 1
        result = _compute_path_stats(g, {0}, 1)
        assert result == (0.0, 0, 0.0, 0)

    def test_source_equals_target(self) -> None:
        """Start same as finish must return zero distance and zero hops."""
        g = nx.Graph()
        g.add_node(0)
        min_dist, min_hops, max_dist, max_hops = _compute_path_stats(g, {0}, 0)
        assert min_dist == pytest.approx(0.0)
        assert min_hops == 0
        assert max_dist == pytest.approx(0.0)
        assert max_hops == 0


# ---------------------------------------------------------------------------
# TestHoldDensity
# ---------------------------------------------------------------------------


class TestHoldDensity:
    """Tests for _compute_hold_density private helper."""

    def test_one_hold_returns_zero(self) -> None:
        """Single hold has no bounding box area, density must be 0.0."""
        holds = [_make_classified_hold(hold_id=0, x_center=0.5, y_center=0.5)]
        assert _compute_hold_density(holds) == pytest.approx(0.0)

    def test_two_collinear_x_returns_zero(self) -> None:
        """Two holds with same y_center have zero bbox height, density must be 0.0."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.8, y_center=0.5),
        ]
        assert _compute_hold_density(holds) == pytest.approx(0.0)

    def test_two_collinear_y_returns_zero(self) -> None:
        """Two holds with same x_center have zero bbox width, density must be 0.0."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.5, y_center=0.2),
            _make_classified_hold(hold_id=1, x_center=0.5, y_center=0.8),
        ]
        assert _compute_hold_density(holds) == pytest.approx(0.0)

    def test_four_holds_known_bbox(self) -> None:
        """Four corner holds must compute correct density.

        bbox: (0.1, 0.1) to (0.9, 0.9) → area = 0.8 * 0.8 = 0.64
        density = 4 / 0.64 = 6.25
        """
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.1),
            _make_classified_hold(hold_id=1, x_center=0.9, y_center=0.1),
            _make_classified_hold(hold_id=2, x_center=0.1, y_center=0.9),
            _make_classified_hold(hold_id=3, x_center=0.9, y_center=0.9),
        ]
        density = _compute_hold_density(holds)
        assert density == pytest.approx(4.0 / 0.64)

    def test_empty_list_returns_zero(self) -> None:
        """Empty hold list must return 0.0 (len < 2)."""
        assert _compute_hold_density([]) == pytest.approx(0.0)

    def test_density_is_positive_for_spread_holds(self) -> None:
        """Valid spread holds must produce a positive density."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.1),
            _make_classified_hold(hold_id=1, x_center=0.9, y_center=0.9),
        ]
        # bbox_area = 0.8 * 0.8 = 0.64; density = 2/0.64 ≈ 3.125
        assert _compute_hold_density(holds) > 0.0


# ---------------------------------------------------------------------------
# TestExtractGeometryFeaturesValidation
# ---------------------------------------------------------------------------


class TestExtractGeometryFeaturesValidation:
    """Tests for extract_geometry_features() error paths."""

    def _make_two_hold_line(self) -> RouteGraph:
        """Build a minimal 2-hold RouteGraph (not yet constrained)."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
        ]
        return build_route_graph(holds, wall_angle=0.0)

    def test_unconstrained_graph_raises(self) -> None:
        """Passing a RouteGraph without applied constraints must raise.

        Note: build_route_graph already prevents empty graphs, so the
        defensive empty-holds guard cannot be triggered via the public API.
        This test verifies that an unconstrained graph (no is_start/is_finish
        attributes set) correctly raises FeatureExtractionError.
        """
        rg = self._make_two_hold_line()
        # No constraints applied — is_start/is_finish not set on any node
        with pytest.raises(FeatureExtractionError) as exc_info:
            extract_geometry_features(rg)
        assert "no start nodes" in exc_info.value.message

    def test_no_start_raises_with_message(self) -> None:
        """Graph without start constraints must raise with correct message."""
        rg = self._make_two_hold_line()
        with pytest.raises(FeatureExtractionError) as exc_info:
            extract_geometry_features(rg)
        assert "apply_route_constraints" in exc_info.value.message

    def test_no_finish_raises_with_message(self) -> None:
        """Graph with start but no finish must raise FeatureExtractionError.

        We manually annotate a graph to have starts but no finishes,
        then wrap it in a RouteGraph to call extract_geometry_features.
        """
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
        ]
        rg = build_route_graph(holds, wall_angle=0.0)
        # Manually annotate: start but no finish
        rg.graph.nodes[0][NODE_ATTR_IS_START] = True
        rg.graph.nodes[1][NODE_ATTR_IS_START] = False
        rg.graph.nodes[0][NODE_ATTR_IS_FINISH] = False
        rg.graph.nodes[1][NODE_ATTR_IS_FINISH] = False
        with pytest.raises(FeatureExtractionError) as exc_info:
            extract_geometry_features(rg)
        assert "no finish node" in exc_info.value.message

    def test_multiple_finish_raises_with_count(self) -> None:
        """Graph with >1 finish nodes must raise with finish count in message."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.6, y_center=0.5),
        ]
        rg = build_route_graph(holds, wall_angle=0.0)
        rg.graph.nodes[0][NODE_ATTR_IS_START] = True
        rg.graph.nodes[1][NODE_ATTR_IS_START] = False
        rg.graph.nodes[2][NODE_ATTR_IS_START] = False
        rg.graph.nodes[0][NODE_ATTR_IS_FINISH] = False
        rg.graph.nodes[1][NODE_ATTR_IS_FINISH] = True
        rg.graph.nodes[2][NODE_ATTR_IS_FINISH] = True
        with pytest.raises(FeatureExtractionError) as exc_info:
            extract_geometry_features(rg)
        assert "2" in exc_info.value.message


# ---------------------------------------------------------------------------
# TestExtractGeometryFeatures
# ---------------------------------------------------------------------------


class TestExtractGeometryFeatures:
    """End-to-end tests for extract_geometry_features()."""

    def test_single_start_returns_geometry_features_instance(self) -> None:
        """extract_geometry_features must return a GeometryFeatures instance."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        gf = extract_geometry_features(crg)
        assert isinstance(gf, GeometryFeatures)

    def test_node_count_matches_constrained_graph(self) -> None:
        """node_count must equal rg.node_count of the constrained graph."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        gf = extract_geometry_features(crg)
        assert gf.node_count == crg.node_count

    def test_edge_count_matches_constrained_graph(self) -> None:
        """edge_count must equal rg.edge_count of the constrained graph."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        gf = extract_geometry_features(crg)
        assert gf.edge_count == crg.edge_count

    def test_single_start_path_distance_correct(self) -> None:
        """Single start with known 2-hop path must compute correct path distance.

        Layout (wall_angle=0, reach=0.35):
        - hold 0 (start): x=0.1, y=0.5
        - hold 1 (intermediate): x=0.3, y=0.5  → edge 0-1: 0.2
        - hold 2 (finish): x=0.5, y=0.5        → edge 1-2: 0.2
        - edge 0-2: distance 0.4 > 0.35 (no direct edge)

        Expected path: [0, 1, 2], distance=0.4, hops=2
        """
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        gf = extract_geometry_features(crg)
        assert gf.path_length_min_distance == pytest.approx(0.4, abs=1e-6)
        assert gf.path_length_max_distance == pytest.approx(0.4, abs=1e-6)
        assert gf.path_length_min_hops == 2
        assert gf.path_length_max_hops == 2

    def test_single_start_min_equals_max(self) -> None:
        """Single start must have min_distance == max_distance."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        gf = extract_geometry_features(crg)
        assert gf.path_length_min_distance == pytest.approx(gf.path_length_max_distance)

    def test_multi_start_min_less_than_max(self) -> None:
        """Multi-start must yield path_length_min_distance < path_length_max_distance.

        Layout (wall_angle=0, reach=0.35):
        - hold 0 (start_A): x=0.2, y=0.5  → direct edge to finish (dist=0.2)
        - hold 1 (finish):  x=0.4, y=0.5
        - hold 2 (start_B): x=0.7, y=0.5  → direct edge to finish (dist=0.3)

        Edges: 0-1 (0.2), 1-2 (0.3), no 0-2 edge (0.5 > 0.35)
        Paths:
        - 0→1: distance=0.2, hops=1  (min)
        - 2→1: distance=0.3, hops=1  (max)
        """
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.7, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0, 2], finish_id=1)
        gf = extract_geometry_features(crg)
        assert gf.path_length_min_distance == pytest.approx(0.2, abs=1e-6)
        assert gf.path_length_max_distance == pytest.approx(0.3, abs=1e-6)
        assert gf.path_length_min_distance < gf.path_length_max_distance
        assert gf.path_length_min_hops == 1
        assert gf.path_length_max_hops == 1

    def test_edge_stats_on_line_graph(self) -> None:
        """Line graph with equal spacing must have correct avg/min/max/std.

        Layout: 3 holds at x=0.1, 0.3, 0.5 (spacing 0.2 each)
        Edges: 0-1 (0.2), 1-2 (0.2) → avg=0.2, min=0.2, max=0.2, std=0.0
        """
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        gf = extract_geometry_features(crg)
        assert gf.avg_move_distance == pytest.approx(0.2, abs=1e-6)
        assert gf.min_move_distance == pytest.approx(0.2, abs=1e-6)
        assert gf.max_move_distance == pytest.approx(0.2, abs=1e-6)
        assert gf.std_move_distance == pytest.approx(0.0, abs=1e-6)

    def test_edge_count_zero_degenerate_route(self) -> None:
        """Degenerate route (start==finish, 1 hold, 0 edges) returns zero distances.

        This is the edge-count=0 valid graph scenario.
        """
        holds = [_make_classified_hold(hold_id=0, x_center=0.5, y_center=0.5)]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=0)
        gf = extract_geometry_features(crg)
        assert gf.edge_count == 0
        assert gf.node_count == 1
        assert gf.avg_move_distance == pytest.approx(0.0)
        assert gf.max_move_distance == pytest.approx(0.0)
        assert gf.min_move_distance == pytest.approx(0.0)
        assert gf.std_move_distance == pytest.approx(0.0)
        assert gf.path_length_min_distance == pytest.approx(0.0)
        assert gf.path_length_max_distance == pytest.approx(0.0)
        assert gf.path_length_min_hops == 0
        assert gf.path_length_max_hops == 0
        assert gf.hold_density == pytest.approx(0.0)

    def test_hold_density_known_value(self) -> None:
        """Known 4-hold layout must produce correct hold_density.

        Layout: holds at corners (0.1,0.1), (0.9,0.1), (0.1,0.9), (0.9,0.9)
        bbox_area = 0.8 * 0.8 = 0.64 → density = 4 / 0.64 = 6.25

        To ensure a valid constrained graph, hold 0 → intermediate → finish path
        is used. Here we arrange a line for constraints but use 4 corner holds
        which are NOT within reach of each other — so we'll test density separately
        by verifying the math with holds that form a valid constrained graph.

        Instead: place holds so they're connected and span a known box.
        """
        # Holds at (0.1, 0.1) and (0.4, 0.4): bbox = 0.3*0.3 = 0.09, density=2/0.09
        # But 0.3*sqrt(2) = 0.424 > 0.35 — no edge, apply_route_constraints fails.

        # Use holds close enough for edges but with a non-zero bbox:
        # hold 0 (start): x=0.1, y=0.1
        # hold 1 (finish): x=0.3, y=0.3  → distance = 0.2*sqrt(2) ≈ 0.283 < 0.35 ✓
        # bbox = (0.3-0.1) * (0.3-0.1) = 0.04, density = 2 / 0.04 = 50.0
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.1),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.3),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=1)
        gf = extract_geometry_features(crg)
        expected_density = 2.0 / ((0.3 - 0.1) * (0.3 - 0.1))
        assert gf.hold_density == pytest.approx(expected_density, rel=1e-4)
