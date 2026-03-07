"""Tests for src.features.assembler module.

Covers:
- src/features/assembler.py  — RouteFeatures model, assemble_features()
"""

import pytest
from pydantic import ValidationError

from src.features.assembler import RouteFeatures, assemble_features
from src.features.exceptions import FeatureExtractionError
from src.features.geometry import GeometryFeatures
from src.features.holds import HoldFeatures
from src.graph.constraints import apply_route_constraints
from src.graph.route_graph import build_route_graph
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
):
    """Build a constrained RouteGraph for assembler tests.

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


def _make_valid_geometry_features() -> GeometryFeatures:
    """Return a valid GeometryFeatures instance for unit tests."""
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


def _make_valid_hold_features() -> HoldFeatures:
    """Return a valid HoldFeatures instance for unit tests."""
    holds = [
        _make_classified_hold(hold_id=0, hold_type="jug"),
        _make_classified_hold(hold_id=1, hold_type="crimp"),
    ]
    from src.features.holds import extract_hold_features

    return extract_hold_features(holds)


# ---------------------------------------------------------------------------
# TestRouteFeatures
# ---------------------------------------------------------------------------


class TestRouteFeatures:
    """Tests for RouteFeatures Pydantic model structure."""

    def test_valid_composition(self) -> None:
        """RouteFeatures accepts valid GeometryFeatures + HoldFeatures."""
        gf = _make_valid_geometry_features()
        hf = _make_valid_hold_features()
        rf = RouteFeatures(geometry=gf, holds=hf)
        assert isinstance(rf, RouteFeatures)

    def test_geometry_accessible(self) -> None:
        """rf.geometry.avg_move_distance must return the correct value."""
        gf = _make_valid_geometry_features()
        hf = _make_valid_hold_features()
        rf = RouteFeatures(geometry=gf, holds=hf)
        assert rf.geometry.avg_move_distance == pytest.approx(0.2)

    def test_holds_accessible(self) -> None:
        """rf.holds.total_count must return the correct value."""
        gf = _make_valid_geometry_features()
        hf = _make_valid_hold_features()
        rf = RouteFeatures(geometry=gf, holds=hf)
        assert rf.holds.total_count == 2

    def test_immutable_geometry(self) -> None:
        """Assigning rf.geometry must raise ValidationError (frozen=True)."""
        gf = _make_valid_geometry_features()
        hf = _make_valid_hold_features()
        rf = RouteFeatures(geometry=gf, holds=hf)
        with pytest.raises(ValidationError):
            rf.geometry = _make_valid_geometry_features()  # type: ignore[misc]

    def test_immutable_holds(self) -> None:
        """Assigning rf.holds must raise ValidationError (frozen=True)."""
        gf = _make_valid_geometry_features()
        hf = _make_valid_hold_features()
        rf = RouteFeatures(geometry=gf, holds=hf)
        with pytest.raises(ValidationError):
            rf.holds = _make_valid_hold_features()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestToVector
# ---------------------------------------------------------------------------


class TestToVector:
    """Tests for RouteFeatures.to_vector() method."""

    def _make_route_features(self) -> RouteFeatures:
        """Build a RouteFeatures instance for to_vector tests."""
        return RouteFeatures(
            geometry=_make_valid_geometry_features(),
            holds=_make_valid_hold_features(),
        )

    def test_returns_dict(self) -> None:
        """to_vector() must return a dict."""
        rf = self._make_route_features()
        assert isinstance(rf.to_vector(), dict)

    def test_key_set_matches_sub_models(self) -> None:
        """Key set must equal union of GeometryFeatures and HoldFeatures field names."""
        rf = self._make_route_features()
        expected = set(GeometryFeatures.model_fields) | set(HoldFeatures.model_fields)
        assert set(rf.to_vector().keys()) == expected

    def test_all_values_float(self) -> None:
        """Every value in the vector must be a float instance."""
        rf = self._make_route_features()
        for k, v in rf.to_vector().items():
            assert isinstance(v, float), f"Key {k!r} has non-float value {v!r}"

    def test_no_key_collision(self) -> None:
        """GeometryFeatures and HoldFeatures field names must be disjoint."""
        geo_keys = set(GeometryFeatures.model_fields)
        hold_keys = set(HoldFeatures.model_fields)
        assert geo_keys.isdisjoint(hold_keys), f"Colliding keys: {geo_keys & hold_keys}"

    def test_geometry_keys_present(self) -> None:
        """avg_move_distance must be present in the vector."""
        rf = self._make_route_features()
        assert "avg_move_distance" in rf.to_vector()

    def test_holds_keys_present(self) -> None:
        """total_count must be present in the vector."""
        rf = self._make_route_features()
        assert "total_count" in rf.to_vector()

    def test_integer_fields_cast_to_float(self) -> None:
        """Integer fields node_count, path_length_min_hops, path_length_max_hops must be float."""
        rf = self._make_route_features()
        vec = rf.to_vector()
        for field in ("node_count", "path_length_min_hops", "path_length_max_hops"):
            assert isinstance(vec[field], float), f"{field} not cast to float"

    def test_round_trip_geometry_value(self) -> None:
        """vec['avg_move_distance'] must equal float(rf.geometry.avg_move_distance)."""
        rf = self._make_route_features()
        vec = rf.to_vector()
        assert vec["avg_move_distance"] == float(rf.geometry.avg_move_distance)

    def test_round_trip_holds_value(self) -> None:
        """vec['total_count'] must equal float(rf.holds.total_count)."""
        rf = self._make_route_features()
        vec = rf.to_vector()
        assert vec["total_count"] == float(rf.holds.total_count)


# ---------------------------------------------------------------------------
# TestAssembleFeatures
# ---------------------------------------------------------------------------


class TestAssembleFeatures:
    """Integration tests for assemble_features() — happy path."""

    def test_returns_route_features(self) -> None:
        """assemble_features must return a RouteFeatures instance."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=1)
        rf = assemble_features(crg)
        assert isinstance(rf, RouteFeatures)

    def test_geometry_sub_model_populated(self) -> None:
        """All 11 geometry fields must be non-negative."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=1)
        rf = assemble_features(crg)
        for field in GeometryFeatures.model_fields:
            val = getattr(rf.geometry, field)
            assert val >= 0, f"geometry.{field} is negative: {val}"

    def test_holds_sub_model_populated(self) -> None:
        """All 23 hold fields must be non-negative."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=1)
        rf = assemble_features(crg)
        for field in HoldFeatures.model_fields:
            val = getattr(rf.holds, field)
            assert val >= 0, f"holds.{field} is negative: {val}"

    def test_single_hold_route(self) -> None:
        """Minimal valid constrained graph (1 hold as start+finish) must succeed."""
        holds = [_make_classified_hold(hold_id=0, x_center=0.5, y_center=0.5)]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=0)
        rf = assemble_features(crg)
        assert isinstance(rf, RouteFeatures)
        assert rf.geometry.node_count == 1
        assert rf.holds.total_count == 1

    def test_multi_hold_route(self) -> None:
        """Richer graph must populate sub-models with correct field values."""
        holds = [
            _make_classified_hold(
                hold_id=0, x_center=0.1, y_center=0.5, hold_type="jug"
            ),
            _make_classified_hold(
                hold_id=1, x_center=0.3, y_center=0.5, hold_type="crimp"
            ),
            _make_classified_hold(
                hold_id=2, x_center=0.5, y_center=0.5, hold_type="sloper"
            ),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=2)
        rf = assemble_features(crg)
        assert rf.geometry.node_count == 3
        assert rf.holds.total_count == 3

    def test_multiple_starts(self) -> None:
        """Graph with 2 start holds must assemble without error."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.7, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0, 2], finish_id=1)
        rf = assemble_features(crg)
        assert isinstance(rf, RouteFeatures)
        assert rf.geometry.node_count == 3

    def test_hold_count_matches_node_count(self) -> None:
        """rf.holds.total_count must equal rf.geometry.node_count (cross-PR contract)."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.3, y_center=0.5),
            _make_classified_hold(hold_id=2, x_center=0.5, y_center=0.5),
            _make_classified_hold(hold_id=3, x_center=0.7, y_center=0.5),
        ]
        crg = _make_constrained_graph(holds, start_ids=[0], finish_id=3)
        rf = assemble_features(crg)
        assert rf.holds.total_count == rf.geometry.node_count


# ---------------------------------------------------------------------------
# TestAssembleFeaturesErrors
# ---------------------------------------------------------------------------


class TestAssembleFeaturesErrors:
    """Tests for assemble_features() error propagation."""

    def test_unconstrained_graph_raises(self) -> None:
        """Graph without start/finish attributes must raise FeatureExtractionError."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
        ]
        rg = build_route_graph(holds, wall_angle=0.0)
        with pytest.raises(FeatureExtractionError):
            assemble_features(rg)

    def test_error_message_contains_context(self) -> None:
        """FeatureExtractionError.message must contain a diagnostic string."""
        holds = [
            _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.5),
            _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5),
        ]
        rg = build_route_graph(holds, wall_angle=0.0)
        with pytest.raises(FeatureExtractionError) as exc_info:
            assemble_features(rg)
        # The sub-extractor message references apply_route_constraints
        assert "apply_route_constraints" in exc_info.value.message
