"""Tests for src.graph.types module.

Covers:
- src/graph/types.py — ClassifiedHold, make_classified_hold
"""

import networkx as nx
import PIL.Image as PILImage
import pytest
from pydantic import ValidationError

from src.graph.route_graph import build_route_graph
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
        """type_probabilities that do not sum to ~1.0 raises ValidationError."""
        bad_probs = {c: 0.1 for c in HOLD_CLASSES}  # sum = len(HOLD_CLASSES) * 0.1
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
        confidence = 0.88
        remainder = (1.0 - confidence) / max(len(HOLD_CLASSES) - 1, 1)
        probs = {c: (confidence if c == "jug" else remainder) for c in HOLD_CLASSES}
        clf = HoldTypeResult(
            predicted_class="jug", confidence=confidence, probabilities=probs
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
                        c: (
                            0.7
                            if c == "crimp"
                            else (1.0 - 0.7) / max(len(HOLD_CLASSES) - 1, 1)
                        )
                        for c in HOLD_CLASSES
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
        probs = {
            c: (0.8 if c == "volume" else (1.0 - 0.8) / max(len(HOLD_CLASSES) - 1, 1))
            for c in HOLD_CLASSES
        }
        clf = HoldTypeResult(
            predicted_class="volume", confidence=0.8, probabilities=probs
        )
        hold = make_classified_hold(0, det, clf)
        rg = build_route_graph([hold])
        assert rg.node_count == 1
        assert hold.detection_class == "volume"
