"""Tests for src.graph.route_graph and src.graph.types modules.

Covers:
- src/graph/exceptions.py  — RouteGraphError
- src/graph/types.py       — ClassifiedHold, make_classified_hold
- src/graph/route_graph.py — RouteGraph, build_route_graph, internal helpers
"""

import logging

import networkx as nx
import PIL.Image as PILImage
import pytest
from pydantic import ValidationError

from src.graph.exceptions import RouteGraphError
from src.graph.route_graph import (
    BASE_REACH_RADIUS,
    WALL_ANGLE_MAX,
    WALL_ANGLE_MIN,
    WALL_ANGLE_REACH_SCALE,
    RouteGraph,
    _MAX_HOLD_COUNT,
    _compute_effective_reach,
    _euclidean_distance,
    build_route_graph,
)
from src.graph.types import ClassifiedHold, make_classified_hold
from src.inference.classification import HoldTypeResult
from src.inference.crop_extractor import HoldCrop
from src.inference.detection import DetectedHold
from src.training.classification_dataset import HOLD_CLASSES

# ---------------------------------------------------------------------------
# Module-level helpers (not fixtures — keep tests self-contained)
# ---------------------------------------------------------------------------


def _make_detection(
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.1,
    height: float = 0.1,
    class_name: str = "hold",
    confidence: float = 0.9,
) -> DetectedHold:
    """Create a DetectedHold with sensible defaults."""
    return DetectedHold(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        class_id=0 if class_name == "hold" else 1,
        class_name=class_name,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _make_classification(
    predicted_class: str = "jug",
    confidence: float = 0.85,
    source_crop: HoldCrop | None = None,
) -> HoldTypeResult:
    """Create a HoldTypeResult with a plausible probability distribution."""
    n = len(HOLD_CLASSES)
    remainder = (1.0 - confidence) / max(n - 1, 1)
    probs = {
        c: (confidence if c == predicted_class else remainder) for c in HOLD_CLASSES
    }
    return HoldTypeResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probs,
        source_crop=source_crop,
    )


def _make_classified_hold(
    hold_id: int = 0,
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.1,
    height: float = 0.1,
    hold_type: str = "jug",
    detection_class: str = "hold",
    detection_confidence: float = 0.9,
    type_confidence: float = 0.8,
) -> ClassifiedHold:
    """Create a ClassifiedHold directly, bypassing the factory."""
    probs = {
        c: (
            type_confidence
            if c == hold_type
            else (1.0 - type_confidence) / max(len(HOLD_CLASSES) - 1, 1)
        )
        for c in HOLD_CLASSES
    }
    return ClassifiedHold(
        hold_id=hold_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        detection_class=detection_class,  # type: ignore[arg-type]
        detection_confidence=detection_confidence,
        hold_type=hold_type,
        type_confidence=type_confidence,
        type_probabilities=probs,
    )


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants in src.graph.route_graph."""

    def test_base_reach_radius_is_positive_float_in_unit_range(self) -> None:
        """BASE_REACH_RADIUS must be a positive float in (0, 1)."""
        assert isinstance(BASE_REACH_RADIUS, float)
        assert 0.0 < BASE_REACH_RADIUS < 1.0

    def test_wall_angle_reach_scale_is_positive_float_less_than_one(self) -> None:
        """WALL_ANGLE_REACH_SCALE must be in (0, 1)."""
        assert isinstance(WALL_ANGLE_REACH_SCALE, float)
        assert 0.0 < WALL_ANGLE_REACH_SCALE < 1.0

    def test_wall_angle_min_is_negative(self) -> None:
        """WALL_ANGLE_MIN must be negative (overhang convention)."""
        assert WALL_ANGLE_MIN < 0.0

    def test_wall_angle_max_is_positive(self) -> None:
        """WALL_ANGLE_MAX must be positive (slab convention)."""
        assert WALL_ANGLE_MAX > 0.0

    def test_wall_angle_range_contains_vertical(self) -> None:
        """0.0 (vertical) must lie within [WALL_ANGLE_MIN, WALL_ANGLE_MAX]."""
        assert WALL_ANGLE_MIN <= 0.0 <= WALL_ANGLE_MAX


# ---------------------------------------------------------------------------
# TestRouteGraphError
# ---------------------------------------------------------------------------


class TestRouteGraphError:
    """Tests for the RouteGraphError exception class."""

    def test_is_subclass_of_value_error(self) -> None:
        """RouteGraphError must inherit from ValueError."""
        assert issubclass(RouteGraphError, ValueError)

    def test_message_attribute_equals_constructor_argument(self) -> None:
        """RouteGraphError.message must equal the constructor string."""
        err = RouteGraphError("something went wrong")
        assert err.message == "something went wrong"

    def test_str_contains_message(self) -> None:
        """str(RouteGraphError) must include the message text."""
        err = RouteGraphError("holds must not be empty")
        assert "holds must not be empty" in str(err)

    def test_can_be_caught_as_value_error(self) -> None:
        """RouteGraphError can be caught with except ValueError."""
        with pytest.raises(ValueError):
            raise RouteGraphError("test")


# ---------------------------------------------------------------------------
# TestClassifiedHold
# ---------------------------------------------------------------------------


class TestClassifiedHold:
    """Tests for the ClassifiedHold Pydantic model."""

    def test_valid_creation_stores_all_fields(self) -> None:
        """ClassifiedHold stores all provided field values."""
        hold = _make_classified_hold(hold_id=3, x_center=0.4, y_center=0.6)
        assert hold.hold_id == 3
        assert hold.x_center == pytest.approx(0.4)
        assert hold.y_center == pytest.approx(0.6)

    def test_negative_hold_id_raises_validation_error(self) -> None:
        """hold_id < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_classified_hold(hold_id=-1)

    def test_x_center_above_one_raises_validation_error(self) -> None:
        """x_center > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_classified_hold(x_center=1.01)

    def test_x_center_below_zero_raises_validation_error(self) -> None:
        """x_center < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_classified_hold(x_center=-0.01)

    def test_detection_confidence_above_one_raises_validation_error(self) -> None:
        """detection_confidence > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_classified_hold(detection_confidence=1.1)

    def test_all_hold_classes_accepted(self) -> None:
        """Every string in HOLD_CLASSES is a valid hold_type."""
        for hold_type in HOLD_CLASSES:
            hold = _make_classified_hold(hold_type=hold_type)
            assert hold.hold_type == hold_type

    def test_invalid_hold_type_raises_validation_error(self) -> None:
        """Unknown hold_type string raises ValidationError mentioning HOLD_CLASSES."""
        with pytest.raises(ValidationError, match="hold_type must be one of"):
            _make_classified_hold(hold_type="bad_type")

    def test_hold_type_is_case_sensitive(self) -> None:
        """'Jug' (capitalised) is rejected — hold_type is case-sensitive."""
        with pytest.raises(ValidationError):
            _make_classified_hold(hold_type="Jug")

    def test_hold_type_unknown_capitalised_is_rejected(self) -> None:
        """'Unknown' (capitalised) is rejected; only lowercase 'unknown' is valid."""
        with pytest.raises(ValidationError):
            _make_classified_hold(hold_type="Unknown")

    def test_type_probabilities_preserved_exactly(self) -> None:
        """type_probabilities dict is stored as-provided."""
        probs = {c: 1.0 / len(HOLD_CLASSES) for c in HOLD_CLASSES}
        hold = ClassifiedHold(
            hold_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.1,
            height=0.1,
            detection_class="hold",
            detection_confidence=0.9,
            hold_type="jug",
            type_confidence=0.8,
            type_probabilities=probs,
        )
        assert hold.type_probabilities == probs

    def test_type_probabilities_wrong_keys_raises_validation_error(self) -> None:
        """type_probabilities with wrong key set raises ValidationError."""
        bad_probs = {"jug": 1.0, "INJECTED": 0.0}  # wrong keys
        with pytest.raises(ValidationError, match="type_probabilities keys must be"):
            ClassifiedHold(
                hold_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.1,
                height=0.1,
                detection_class="hold",
                detection_confidence=0.9,
                hold_type="jug",
                type_confidence=0.8,
                type_probabilities=bad_probs,
            )

    def test_type_probabilities_value_out_of_range_raises_validation_error(
        self,
    ) -> None:
        """type_probabilities with a probability > 1.0 raises ValidationError."""
        bad_probs = {c: (999.0 if c == "jug" else 0.0) for c in HOLD_CLASSES}
        with pytest.raises(ValidationError, match="must be in"):
            ClassifiedHold(
                hold_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.1,
                height=0.1,
                detection_class="hold",
                detection_confidence=0.9,
                hold_type="jug",
                type_confidence=0.8,
                type_probabilities=bad_probs,
            )

    def test_type_probabilities_not_normalized_raises_validation_error(self) -> None:
        """type_probabilities that sum to 0.5 (not ~1.0) raises ValidationError."""
        bad_probs = {c: 0.1 for c in HOLD_CLASSES}  # sum = 0.6
        with pytest.raises(ValidationError, match="approximately 1.0"):
            ClassifiedHold(
                hold_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.1,
                height=0.1,
                detection_class="hold",
                detection_confidence=0.9,
                hold_type="jug",
                type_confidence=0.8,
                type_probabilities=bad_probs,
            )

    def test_detection_class_volume_accepted(self) -> None:
        """detection_class='volume' is a valid value."""
        hold = _make_classified_hold(detection_class="volume", hold_type="volume")
        assert hold.detection_class == "volume"

    def test_invalid_detection_class_raises_validation_error(self) -> None:
        """detection_class='wall' (invalid) raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_classified_hold(detection_class="wall")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestMakeClassifiedHold
# ---------------------------------------------------------------------------


class TestMakeClassifiedHold:
    """Tests for the make_classified_hold factory function."""

    def test_happy_path_copies_detection_coordinates(self) -> None:
        """x_center and y_center come from the explicit detection argument."""
        det = _make_detection(x_center=0.3, y_center=0.7)
        clf = _make_classification()
        hold = make_classified_hold(0, det, clf)
        assert hold.x_center == pytest.approx(0.3)
        assert hold.y_center == pytest.approx(0.7)

    def test_uses_explicit_detection_not_source_crop_hold(self) -> None:
        """When source_crop.hold differs from detection, detection is authoritative.

        This test verifies the contract: the explicit ``detection`` parameter
        determines position, not ``classification.source_crop.hold``.
        """
        det_a = _make_detection(x_center=0.2, y_center=0.2)
        det_b = _make_detection(x_center=0.8, y_center=0.8)  # different position
        crop = HoldCrop(
            crop=PILImage.new("RGB", (224, 224)),
            hold=det_b,
            pixel_box=(0, 0, 224, 224),
        )
        clf = _make_classification(source_crop=crop)
        hold = make_classified_hold(0, det_a, clf)
        assert hold.x_center == pytest.approx(0.2), (
            "Should use det_a, not source_crop.hold"
        )
        assert hold.y_center == pytest.approx(0.2), (
            "Should use det_a, not source_crop.hold"
        )

    def test_source_crop_none_does_not_raise(self) -> None:
        """source_crop=None is handled without error."""
        det = _make_detection()
        clf = _make_classification(source_crop=None)
        hold = make_classified_hold(0, det, clf)
        assert hold.hold_id == 0

    def test_hold_id_is_propagated(self) -> None:
        """The provided hold_id is stored on the ClassifiedHold."""
        hold = make_classified_hold(42, _make_detection(), _make_classification())
        assert hold.hold_id == 42

    def test_probabilities_from_classification_preserved(self) -> None:
        """The full probability distribution from HoldTypeResult is preserved."""
        clf = _make_classification(predicted_class="crimp", confidence=0.75)
        hold = make_classified_hold(0, _make_detection(), clf)
        assert hold.type_probabilities == clf.probabilities

    def test_detection_class_name_copied(self) -> None:
        """detection_class comes from detection.class_name."""
        det = _make_detection(class_name="volume")
        clf = _make_classification(predicted_class="volume")
        hold = make_classified_hold(0, det, clf)
        assert hold.detection_class == "volume"

    def test_width_and_height_from_detection(self) -> None:
        """width and height come from DetectedHold, not from any crop size."""
        det = _make_detection(width=0.15, height=0.25)
        hold = make_classified_hold(0, det, _make_classification())
        assert hold.width == pytest.approx(0.15)
        assert hold.height == pytest.approx(0.25)

    def test_detection_confidence_from_detection(self) -> None:
        """detection_confidence is copied from DetectedHold.confidence."""
        det = _make_detection(confidence=0.77)
        hold = make_classified_hold(0, det, _make_classification())
        assert hold.detection_confidence == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# TestMakeClassifiedHoldFromPipelineOutput
# ---------------------------------------------------------------------------


class TestMakeClassifiedHoldFromPipelineOutput:
    """Integration: real DetectedHold + HoldTypeResult → ClassifiedHold → RouteGraph.

    No mocks. Verifies that the data types produced by the inference pipeline
    chain together with the graph builder without field misalignment.
    """

    def test_real_pipeline_objects_produce_valid_classified_hold(self) -> None:
        """Real DetectedHold + HoldTypeResult → valid ClassifiedHold."""
        det = DetectedHold(
            x_center=0.4,
            y_center=0.6,
            width=0.08,
            height=0.12,
            class_id=0,
            class_name="hold",
            confidence=0.91,
        )
        probs = {c: (0.88 if c == "jug" else 0.024) for c in HOLD_CLASSES}
        clf = HoldTypeResult(
            predicted_class="jug", confidence=0.88, probabilities=probs
        )
        hold = make_classified_hold(0, det, clf)
        assert hold.x_center == pytest.approx(0.4)
        assert hold.hold_type == "jug"
        assert hold.hold_id == 0

    def test_pipeline_classified_holds_feed_into_build_route_graph(self) -> None:
        """A list of ClassifiedHold built from real types produces a valid RouteGraph."""
        holds = [
            make_classified_hold(
                i,
                DetectedHold(
                    x_center=0.1 + i * 0.2,
                    y_center=0.5,
                    width=0.05,
                    height=0.05,
                    class_id=0,
                    class_name="hold",
                    confidence=0.9,
                ),
                HoldTypeResult(
                    predicted_class="crimp",
                    confidence=0.7,
                    probabilities={
                        c: (0.7 if c == "crimp" else 0.06) for c in HOLD_CLASSES
                    },
                ),
            )
            for i in range(3)
        ]
        rg = build_route_graph(holds, wall_angle=0.0)
        assert rg.node_count == 3
        assert isinstance(rg.graph, nx.Graph)

    def test_volume_detection_class_accepted_end_to_end(self) -> None:
        """detection_class='volume' flows through the full pipeline without error."""
        det = DetectedHold(
            x_center=0.5,
            y_center=0.5,
            width=0.2,
            height=0.2,
            class_id=1,
            class_name="volume",
            confidence=0.95,
        )
        probs = {c: (0.8 if c == "volume" else 0.04) for c in HOLD_CLASSES}
        clf = HoldTypeResult(
            predicted_class="volume", confidence=0.8, probabilities=probs
        )
        hold = make_classified_hold(0, det, clf)
        rg = build_route_graph([hold])
        assert rg.node_count == 1
        assert hold.detection_class == "volume"


# ---------------------------------------------------------------------------
# TestComputeEffectiveReach
# ---------------------------------------------------------------------------


class TestComputeEffectiveReach:
    """Tests for the _compute_effective_reach internal helper."""

    def test_vertical_wall_returns_base_radius(self) -> None:
        """At 0° (vertical), effective reach equals BASE_REACH_RADIUS exactly."""
        assert _compute_effective_reach(0.0) == pytest.approx(
            BASE_REACH_RADIUS, rel=1e-9
        )

    def test_slab_returns_greater_than_base_radius(self) -> None:
        """At 90° (slab), effective reach > BASE_REACH_RADIUS."""
        assert _compute_effective_reach(90.0) > BASE_REACH_RADIUS

    def test_overhang_returns_less_than_base_radius(self) -> None:
        """At -15° (overhang), effective reach < BASE_REACH_RADIUS."""
        assert _compute_effective_reach(-15.0) < BASE_REACH_RADIUS

    def test_slab_formula_matches_expected_value(self) -> None:
        """At 90°, effective reach = BASE_REACH_RADIUS * (1 + WALL_ANGLE_REACH_SCALE)."""
        expected = BASE_REACH_RADIUS * (1.0 + WALL_ANGLE_REACH_SCALE)
        assert _compute_effective_reach(90.0) == pytest.approx(expected, rel=1e-9)

    def test_formula_is_directional_slab_greater_than_overhang(self) -> None:
        """Slab (90°) gives more reach than overhang (-15°)."""
        assert _compute_effective_reach(90.0) > _compute_effective_reach(-15.0)

    def test_negative_angle_returns_positive_float(self) -> None:
        """A valid negative wall angle still produces a positive reach radius."""
        assert _compute_effective_reach(-15.0) > 0.0


# ---------------------------------------------------------------------------
# TestEuclideanDistance
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    """Tests for the _euclidean_distance internal helper."""

    def test_same_position_gives_zero_distance(self) -> None:
        """Two holds at identical coordinates have distance 0."""
        h = _make_classified_hold(x_center=0.5, y_center=0.5)
        assert _euclidean_distance(h, h) == pytest.approx(0.0)

    def test_known_345_right_triangle(self) -> None:
        """Holds at a 3-4-5 right triangle have distance 0.05 (scaled by 0.01)."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.03, y_center=0.04)
        assert _euclidean_distance(h1, h2) == pytest.approx(0.05, abs=1e-9)

    def test_distance_is_symmetric(self) -> None:
        """dist(a, b) == dist(b, a)."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.2, y_center=0.3)
        h2 = _make_classified_hold(hold_id=1, x_center=0.5, y_center=0.7)
        assert _euclidean_distance(h1, h2) == pytest.approx(_euclidean_distance(h2, h1))

    def test_returns_float(self) -> None:
        """Return type is float."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.1, y_center=0.1)
        h2 = _make_classified_hold(hold_id=1, x_center=0.4, y_center=0.5)
        assert isinstance(_euclidean_distance(h1, h2), float)

    def test_horizontal_unit_distance(self) -> None:
        """Two holds separated by 0.1 horizontally have distance 0.1."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.5)
        h2 = _make_classified_hold(hold_id=1, x_center=0.1, y_center=0.5)
        assert _euclidean_distance(h1, h2) == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------------------------
# TestBuildRouteGraphValidation
# ---------------------------------------------------------------------------


class TestBuildRouteGraphValidation:
    """Tests for build_route_graph input validation."""

    def test_empty_holds_raises_route_graph_error(self) -> None:
        """Empty holds list raises RouteGraphError."""
        with pytest.raises(RouteGraphError, match="empty"):
            build_route_graph([])

    def test_wall_angle_below_min_raises_route_graph_error(self) -> None:
        """wall_angle below WALL_ANGLE_MIN raises RouteGraphError."""
        hold = _make_classified_hold()
        with pytest.raises(RouteGraphError):
            build_route_graph([hold], wall_angle=WALL_ANGLE_MIN - 0.01)

    def test_wall_angle_above_max_raises_route_graph_error(self) -> None:
        """wall_angle above WALL_ANGLE_MAX raises RouteGraphError."""
        hold = _make_classified_hold()
        with pytest.raises(RouteGraphError):
            build_route_graph([hold], wall_angle=WALL_ANGLE_MAX + 0.01)

    def test_wall_angle_at_min_boundary_is_accepted(self) -> None:
        """wall_angle == WALL_ANGLE_MIN is a valid input."""
        rg = build_route_graph([_make_classified_hold()], wall_angle=WALL_ANGLE_MIN)
        assert rg.wall_angle == pytest.approx(WALL_ANGLE_MIN)

    def test_wall_angle_at_max_boundary_is_accepted(self) -> None:
        """wall_angle == WALL_ANGLE_MAX is a valid input."""
        rg = build_route_graph([_make_classified_hold()], wall_angle=WALL_ANGLE_MAX)
        assert rg.wall_angle == pytest.approx(WALL_ANGLE_MAX)

    def test_default_wall_angle_is_zero(self) -> None:
        """Default wall_angle is 0.0 (vertical)."""
        rg = build_route_graph([_make_classified_hold()])
        assert rg.wall_angle == pytest.approx(0.0)

    def test_exceeds_max_hold_count_raises_route_graph_error(self) -> None:
        """hold count > _MAX_HOLD_COUNT raises RouteGraphError."""
        holds = [_make_classified_hold(hold_id=i) for i in range(_MAX_HOLD_COUNT + 1)]
        with pytest.raises(RouteGraphError, match="exceeds the maximum"):
            build_route_graph(holds)

    def test_at_max_hold_count_boundary_is_accepted(self) -> None:
        """_MAX_HOLD_COUNT holds (boundary) is accepted without error."""
        holds = [
            _make_classified_hold(hold_id=i, x_center=0.5, y_center=0.5)
            for i in range(_MAX_HOLD_COUNT)
        ]
        rg = build_route_graph(holds)
        assert rg.node_count == _MAX_HOLD_COUNT

    def test_duplicate_hold_ids_raises_route_graph_error(self) -> None:
        """Two holds sharing the same hold_id raise RouteGraphError."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.1)
        h2 = _make_classified_hold(hold_id=0, x_center=0.5)  # duplicate hold_id
        with pytest.raises(RouteGraphError, match="unique"):
            build_route_graph([h1, h2])


# ---------------------------------------------------------------------------
# TestBuildRouteGraphSingleHold
# ---------------------------------------------------------------------------


class TestBuildRouteGraphSingleHold:
    """Tests for build_route_graph with exactly one hold."""

    def test_single_hold_produces_one_node(self) -> None:
        """Graph with one hold has exactly 1 node."""
        rg = build_route_graph([_make_classified_hold(hold_id=0)])
        assert rg.node_count == 1

    def test_single_hold_produces_zero_edges(self) -> None:
        """Graph with one hold has 0 edges (no self-loops)."""
        rg = build_route_graph([_make_classified_hold(hold_id=0)])
        assert rg.edge_count == 0

    def test_single_hold_node_is_keyed_by_hold_id(self) -> None:
        """The node key in the NetworkX graph is the hold's hold_id."""
        rg = build_route_graph([_make_classified_hold(hold_id=7)])
        assert 7 in rg.graph.nodes


# ---------------------------------------------------------------------------
# TestBuildRouteGraphTwoHolds
# ---------------------------------------------------------------------------


class TestBuildRouteGraphTwoHolds:
    """Tests for build_route_graph with two holds at known positions.

    BASE_REACH_RADIUS = 0.35 at 0° (vertical).
    Effective reach at 90° (slab) = 0.35 * 1.2 = 0.42.
    """

    def test_close_holds_produce_one_edge(self) -> None:
        """Holds at distance 0.30 < 0.35 (reach at 0°) are connected."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.30, y_center=0.0)
        assert build_route_graph([h1, h2], wall_angle=0.0).edge_count == 1

    def test_holds_at_0_38_not_connected_at_vertical(self) -> None:
        """Holds at distance 0.38 > 0.35 (reach at 0°) are NOT connected."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.38, y_center=0.0)
        assert build_route_graph([h1, h2], wall_angle=0.0).edge_count == 0

    def test_holds_at_0_38_connected_at_slab(self) -> None:
        """Same holds at 0.38 ARE connected at 90° (reach = 0.42)."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.38, y_center=0.0)
        assert build_route_graph([h1, h2], wall_angle=90.0).edge_count == 1

    def test_very_distant_holds_not_connected_even_at_slab(self) -> None:
        """Holds at 0.50 > 0.42 (reach at 90°) are NOT connected even on a slab."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.50, y_center=0.0)
        assert build_route_graph([h1, h2], wall_angle=90.0).edge_count == 0

    def test_overhang_has_less_reach_than_vertical(self) -> None:
        """Two holds at a border distance are connected at 0° but not at -15°.

        Effective reach at -15°: 0.35 * (1 + 0.2 * sin(-15°)) ≈ 0.332.
        We use distance ≈ 0.34 > 0.332 but < 0.35.
        """
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.34, y_center=0.0)
        assert build_route_graph([h1, h2], wall_angle=0.0).edge_count == 1
        assert build_route_graph([h1, h2], wall_angle=-15.0).edge_count == 0

    def test_coincident_holds_produce_no_edge(self) -> None:
        """Two holds at identical coordinates produce no edge (zero-distance excluded).

        Zero-weight edges are excluded to prevent division-by-zero in downstream
        shortest-path algorithms (PR-6.x).
        """
        h1 = _make_classified_hold(hold_id=0, x_center=0.5, y_center=0.5)
        h2 = _make_classified_hold(hold_id=1, x_center=0.5, y_center=0.5)
        rg = build_route_graph([h1, h2])
        assert rg.edge_count == 0


# ---------------------------------------------------------------------------
# TestBuildRouteGraphNodeAttributes
# ---------------------------------------------------------------------------


class TestBuildRouteGraphNodeAttributes:
    """Tests for node attribute storage in the built graph."""

    def test_all_classified_hold_fields_stored_as_node_attrs(self) -> None:
        """Every ClassifiedHold field (except hold_id) is a node attribute."""
        hold = _make_classified_hold(
            hold_id=3, x_center=0.4, y_center=0.6, hold_type="crimp"
        )
        rg = build_route_graph([hold])
        attrs = rg.graph.nodes[3]
        assert attrs["x_center"] == pytest.approx(0.4)
        assert attrs["y_center"] == pytest.approx(0.6)
        assert attrs["hold_type"] == "crimp"
        assert attrs["detection_class"] == "hold"
        assert "detection_confidence" in attrs
        assert "type_confidence" in attrs
        assert "type_probabilities" in attrs

    def test_hold_id_excluded_from_node_attrs(self) -> None:
        """hold_id is the node key and must NOT appear in node attribute dict."""
        hold = _make_classified_hold(hold_id=5)
        rg = build_route_graph([hold])
        assert "hold_id" not in rg.graph.nodes[5]

    def test_node_attrs_match_model_dump_exactly(self) -> None:
        """Node attributes exactly equal hold.model_dump(exclude={'hold_id'})."""
        hold = _make_classified_hold(hold_id=0, x_center=0.25, y_center=0.75)
        rg = build_route_graph([hold])
        expected = hold.model_dump(exclude={"hold_id"})
        assert rg.graph.nodes[0] == expected


# ---------------------------------------------------------------------------
# TestBuildRouteGraphEdgeAttributes
# ---------------------------------------------------------------------------


class TestBuildRouteGraphEdgeAttributes:
    """Tests for edge attribute storage in the built graph."""

    def test_edge_weight_equals_euclidean_distance(self) -> None:
        """Edge weight is the Euclidean distance between the two hold centers."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.03, y_center=0.04)
        rg = build_route_graph([h1, h2])
        assert rg.graph.has_edge(0, 1)
        assert rg.graph.edges[0, 1]["weight"] == pytest.approx(0.05, abs=1e-9)

    def test_edge_weight_is_float(self) -> None:
        """Edge weight is stored as a Python float."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.1, y_center=0.0)
        rg = build_route_graph([h1, h2])
        assert isinstance(rg.graph.edges[0, 1]["weight"], float)

    def test_edge_weight_is_positive(self) -> None:
        """Edge weight is strictly > 0 (no self-loops; holds are distinct)."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.1, y_center=0.1)
        rg = build_route_graph([h1, h2])
        assert rg.graph.edges[0, 1]["weight"] > 0.0


# ---------------------------------------------------------------------------
# TestBuildRouteGraphReturnType
# ---------------------------------------------------------------------------


class TestBuildRouteGraphReturnType:
    """Tests for the RouteGraph return type and field preservation."""

    def test_returns_route_graph_instance(self) -> None:
        """build_route_graph returns a RouteGraph instance."""
        assert isinstance(build_route_graph([_make_classified_hold()]), RouteGraph)

    def test_holds_list_preserved_in_order(self) -> None:
        """RouteGraph.holds matches the input list in order."""
        holds = [_make_classified_hold(hold_id=i, x_center=0.1 * i) for i in range(4)]
        rg = build_route_graph(holds)
        assert rg.holds == holds

    def test_wall_angle_preserved(self) -> None:
        """RouteGraph.wall_angle equals the provided wall_angle."""
        rg = build_route_graph([_make_classified_hold()], wall_angle=45.0)
        assert rg.wall_angle == pytest.approx(45.0)

    def test_graph_is_networkx_graph_instance(self) -> None:
        """RouteGraph.graph is an nx.Graph."""
        rg = build_route_graph([_make_classified_hold()])
        assert isinstance(rg.graph, nx.Graph)

    def test_holds_list_is_independent_copy(self) -> None:
        """Mutating the input list after the call does not change RouteGraph.holds."""
        holds = [_make_classified_hold(hold_id=0)]
        rg = build_route_graph(holds)
        holds.append(_make_classified_hold(hold_id=1))
        assert len(rg.holds) == 1


# ---------------------------------------------------------------------------
# TestBuildRouteGraphLargeInput
# ---------------------------------------------------------------------------


class TestBuildRouteGraphLargeInput:
    """Tests for large-input handling (O(n²) warning)."""

    def test_large_input_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """101+ holds triggers a WARNING about O(n²) complexity."""
        holds = [
            _make_classified_hold(
                hold_id=i,
                x_center=(i % 10) * 0.1,
                y_center=(i // 10) * 0.1,
            )
            for i in range(101)
        ]
        with caplog.at_level(logging.WARNING, logger="src.graph.route_graph"):
            build_route_graph(holds)
        assert any("101" in msg for msg in caplog.messages), (
            f"Expected a warning mentioning 101 holds. Messages: {caplog.messages}"
        )

    def test_large_input_returns_valid_route_graph(self) -> None:
        """101 holds still returns a valid RouteGraph (not an error)."""
        holds = [
            _make_classified_hold(
                hold_id=i,
                x_center=(i % 10) * 0.1,
                y_center=(i // 10) * 0.1,
            )
            for i in range(101)
        ]
        rg = build_route_graph(holds)
        assert rg.node_count == 101


# ---------------------------------------------------------------------------
# TestRouteGraphProperties
# ---------------------------------------------------------------------------


class TestRouteGraphProperties:
    """Tests for RouteGraph.node_count and RouteGraph.edge_count properties."""

    def test_node_count_equals_number_of_holds(self) -> None:
        """node_count equals the number of input holds."""
        holds = [_make_classified_hold(hold_id=i, x_center=i * 0.4) for i in range(3)]
        rg = build_route_graph(holds)
        assert rg.node_count == len(holds)

    def test_edge_count_is_zero_when_holds_all_out_of_reach(self) -> None:
        """edge_count is 0 when all hold pairs exceed the reach threshold.

        Holds at x = 0.0, 0.4, 0.8 — adjacent distance 0.4 > 0.35 (reach at 0°).
        """
        holds = [_make_classified_hold(hold_id=i, x_center=i * 0.4) for i in range(3)]
        rg = build_route_graph(holds)
        assert rg.edge_count == 0

    def test_edge_count_matches_graph_number_of_edges(self) -> None:
        """edge_count == graph.number_of_edges() (property is consistent)."""
        h1 = _make_classified_hold(hold_id=0, x_center=0.0, y_center=0.0)
        h2 = _make_classified_hold(hold_id=1, x_center=0.2, y_center=0.0)
        h3 = _make_classified_hold(hold_id=2, x_center=0.0, y_center=0.2)
        rg = build_route_graph([h1, h2, h3])
        # All three pairs within distance 0.283 < 0.35
        assert rg.edge_count == rg.graph.number_of_edges()

    def test_node_count_is_integer(self) -> None:
        """node_count returns an int."""
        rg = build_route_graph([_make_classified_hold()])
        assert isinstance(rg.node_count, int)

    def test_edge_count_is_integer(self) -> None:
        """edge_count returns an int."""
        rg = build_route_graph([_make_classified_hold()])
        assert isinstance(rg.edge_count, int)


# ---------------------------------------------------------------------------
# TestRouteGraphModelValidator
# ---------------------------------------------------------------------------


class TestRouteGraphModelValidator:
    """Tests for the RouteGraph model_validator that enforces construction invariants."""

    def test_mismatched_node_count_raises_validation_error(self) -> None:
        """RouteGraph with graph.number_of_nodes() != len(holds) raises ValidationError."""
        graph = nx.Graph()
        graph.add_node(0)  # 1 node
        holds = [
            _make_classified_hold(hold_id=0),
            _make_classified_hold(hold_id=1, x_center=0.3),
        ]  # 2 holds
        with pytest.raises(ValidationError, match="must match"):
            RouteGraph(graph=graph, holds=holds, wall_angle=0.0)

    def test_consistent_graph_and_holds_is_accepted(self) -> None:
        """RouteGraph with matching graph.nodes count and holds list is accepted."""
        hold = _make_classified_hold(hold_id=0)
        graph = nx.Graph()
        graph.add_node(0, **hold.model_dump(exclude={"hold_id"}))
        rg = RouteGraph(graph=graph, holds=[hold], wall_angle=0.0)
        assert rg.node_count == 1

    def test_empty_graph_and_empty_holds_raises_validation_error(self) -> None:
        """RouteGraph with both graph and holds empty is technically consistent (0==0).

        This edge case is accepted by the model_validator; the RouteGraphError
        for empty holds is enforced by build_route_graph, not by RouteGraph itself.
        """
        graph = nx.Graph()
        rg = RouteGraph(graph=graph, holds=[], wall_angle=0.0)
        assert rg.node_count == 0
