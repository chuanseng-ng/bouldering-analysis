"""
End-to-end tests for Phase 1a MVP grade prediction.

These tests verify the complete flow from image upload through grade prediction,
including wall_incline parameter handling and score breakdown validation.
"""

from __future__ import annotations

import io
import os
import time
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.models import Analysis, WallInclineType
from src.models import db


def create_mock_yolo_result(num_boxes: int = 5) -> Mock:
    """Create a mock YOLO detection result with specified number of boxes."""
    mock_result = Mock()
    boxes = []

    for i in range(num_boxes):
        box = Mock()
        box.xyxy = [Mock()]
        box.xyxy[0].cpu.return_value.numpy.return_value = [
            i * 50,
            i * 100,
            i * 50 + 40,
            i * 100 + 40,
        ]
        box.conf = [Mock()]
        box.conf[0].cpu.return_value.numpy.return_value = 0.85
        box.cls = [Mock()]
        # Alternate between crimp (0) and jug (1)
        box.cls[0].cpu.return_value.numpy.return_value = i % 2
        boxes.append(box)

    mock_result.boxes = boxes
    return mock_result


def create_test_image() -> io.BytesIO:
    """Create a test image in memory."""
    img = Image.new("RGB", (800, 1200), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestWallInclineValidation:
    """Test wall_incline validation in API endpoints."""

    def test_analyze_with_valid_wall_inclines(self, test_client):
        """Test /analyze accepts all valid wall_incline values."""
        valid_inclines = [
            "slab",
            "vertical",
            "slight_overhang",
            "moderate_overhang",
            "steep_overhang",
        ]

        for incline in valid_inclines:
            with patch("src.main.hold_detection_model") as mock_model:
                with patch(
                    "src.main.get_hold_types", return_value={0: "crimp", 1: "jug"}
                ):
                    mock_model.return_value = [create_mock_yolo_result()]

                    img = create_test_image()
                    response = test_client.post(
                        "/analyze",
                        data={"file": (img, "test.jpg"), "wall_incline": incline},
                        content_type="multipart/form-data",
                    )

                    assert response.status_code == 200, (
                        f"Failed for wall_incline={incline}"
                    )
                    json_data = response.get_json()
                    assert json_data["score_breakdown"]["wall_angle"] == incline

    def test_analyze_with_invalid_wall_incline(self, test_client):
        """Test /analyze rejects invalid wall_incline values."""
        img = create_test_image()

        response = test_client.post(
            "/analyze",
            data={"file": (img, "test.jpg"), "wall_incline": "invalid_angle"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        json_data = response.get_json()
        assert "Invalid wall incline" in json_data["error"]

    def test_analyze_defaults_to_vertical(self, test_client):
        """Test /analyze defaults to 'vertical' when wall_incline not provided."""
        with patch("src.main.hold_detection_model") as mock_model:
            with patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"}):
                mock_model.return_value = [create_mock_yolo_result()]

                img = create_test_image()
                response = test_client.post(
                    "/analyze",
                    data={"file": (img, "test.jpg")},  # No wall_incline
                    content_type="multipart/form-data",
                )

                assert response.status_code == 200
                json_data = response.get_json()
                assert json_data["score_breakdown"]["wall_angle"] == "vertical"

    def test_index_post_with_invalid_wall_incline(self, test_client):
        """Test index POST rejects invalid wall_incline values."""
        img = create_test_image()

        response = test_client.post(
            "/",
            data={"file": (img, "test.jpg"), "wall_incline": "super_steep"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200  # Returns HTML page
        assert b"Invalid wall incline" in response.data


class TestWallInclineType:
    """Test WallInclineType enum validation."""

    def test_is_valid_accepts_valid_values(self):
        """Test is_valid returns True for valid wall incline values."""
        valid_values = [
            "slab",
            "vertical",
            "slight_overhang",
            "moderate_overhang",
            "steep_overhang",
        ]

        for value in valid_values:
            assert WallInclineType.is_valid(value) is True, f"Failed for {value}"

    def test_is_valid_rejects_invalid_values(self):
        """Test is_valid returns False for invalid wall incline values."""
        invalid_values = [
            "invalid",
            "VERTICAL",  # Case sensitive
            "Slab",
            "overhang",
            "",
            "45_degrees",
        ]

        for value in invalid_values:
            assert WallInclineType.is_valid(value) is False, f"Should reject {value}"


class TestGradeVariesWithWallAngle:
    """Test that predicted grades increase with steeper wall angles."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_steeper_angles_produce_higher_scores(
        self, mock_model, _mock_get_hold_types, test_client
    ):
        """Test that wall incline scores increase: slab < vertical < steep_overhang."""
        mock_model.return_value = [create_mock_yolo_result(num_boxes=8)]

        scores = {}
        for incline in ["slab", "vertical", "steep_overhang"]:
            img = create_test_image()
            response = test_client.post(
                "/analyze",
                data={"file": (img, "test.jpg"), "wall_incline": incline},
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            json_data = response.get_json()
            scores[incline] = json_data["score_breakdown"]["final_score"]

        # Verify ordering: slab < vertical < steep_overhang
        assert scores["slab"] < scores["vertical"], (
            f"Slab ({scores['slab']}) should be less than vertical ({scores['vertical']})"
        )
        assert scores["vertical"] < scores["steep_overhang"], (
            f"Vertical ({scores['vertical']}) should be less than "
            f"steep_overhang ({scores['steep_overhang']})"
        )

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_factor4_wall_incline_component_monotonicity(
        self, mock_model, _mock_get_hold_types, test_client
    ):
        """Test that Factor 4 (wall_incline) component score increases monotonically.

        This test verifies the wall_incline component score from the score breakdown
        (i.e., score_breakdown["wall_incline"]), NOT the final combined grade.
        The Factor 4 component represents the isolated wall angle difficulty
        contribution before weighting and combination with other factors.

        Expected ordering: slab (3.0) < vertical (6.0) < ... < steep_overhang (11.0)

        Note: Tests relative ordering rather than exact values to allow for
        Phase 1b calibration adjustments without breaking tests.
        """
        inclines_ordered = [
            "slab",
            "vertical",
            "slight_overhang",
            "moderate_overhang",
            "steep_overhang",
        ]

        mock_model.return_value = [create_mock_yolo_result()]

        scores = []
        for incline in inclines_ordered:
            img = create_test_image()
            response = test_client.post(
                "/analyze",
                data={"file": (img, "test.jpg"), "wall_incline": incline},
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            json_data = response.get_json()
            scores.append(json_data["score_breakdown"]["wall_incline"])

        # Verify monotonic increase: slab < vertical < ... < steep_overhang
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Wall incline scores should increase monotonically: "
                f"{inclines_ordered[i]} ({scores[i]}) should be less than "
                f"{inclines_ordered[i + 1]} ({scores[i + 1]})"
            )


class TestDatabasePersistence:
    """Test that wall_incline is correctly persisted in database."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_wall_incline_stored_in_analysis(
        self, mock_model, _mock_get_hold_types, test_client, test_app
    ):
        """Test that wall_incline is stored in Analysis record."""
        mock_model.return_value = [create_mock_yolo_result()]

        with test_app.app_context():
            img = create_test_image()
            response = test_client.post(
                "/analyze",
                data={"file": (img, "test.jpg"), "wall_incline": "moderate_overhang"},
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            json_data = response.get_json()

            # Verify database record
            analysis = db.session.get(Analysis, json_data["analysis_id"])
            assert analysis is not None
            assert analysis.wall_incline == "moderate_overhang"

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_all_wall_inclines_stored_correctly(
        self, mock_model, _mock_get_hold_types, test_client, test_app
    ):
        """Test all wall_incline values are stored correctly."""
        mock_model.return_value = [create_mock_yolo_result()]

        inclines = [
            "slab",
            "vertical",
            "slight_overhang",
            "moderate_overhang",
            "steep_overhang",
        ]

        with test_app.app_context():
            for incline in inclines:
                img = create_test_image()
                response = test_client.post(
                    "/analyze",
                    data={"file": (img, "test.jpg"), "wall_incline": incline},
                    content_type="multipart/form-data",
                )

                assert response.status_code == 200
                json_data = response.get_json()

                analysis = db.session.get(Analysis, json_data["analysis_id"])
                assert analysis is not None, "Analysis record not found"
                assert analysis.wall_incline == incline, (
                    f"Expected {incline}, got {analysis.wall_incline}"
                )


class TestScoreBreakdownValidation:
    """Test score breakdown contains all expected fields and valid values."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_breakdown_has_all_required_fields(
        self, mock_model, _mock_get_hold_types, test_client
    ):
        """Test breakdown includes all 4 factors and metadata."""
        mock_model.return_value = [create_mock_yolo_result()]

        img = create_test_image()
        response = test_client.post(
            "/analyze",
            data={"file": (img, "test.jpg"), "wall_incline": "vertical"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        breakdown = response.get_json()["score_breakdown"]

        # Required factor fields
        required_fields = [
            "hold_difficulty",
            "hold_density",
            "distance",
            "wall_incline",
            "base_score",
            "final_score",
            "handhold_count",
            "foothold_count",
            "wall_angle",
            "algorithm_version",
            "weights",
        ]

        for field in required_fields:
            assert field in breakdown, f"Missing required field: {field}"

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_weighted_sum_calculation(
        self, mock_model, _mock_get_hold_types, test_client
    ):
        """Test that final_score equals weighted sum of factors."""
        mock_model.return_value = [create_mock_yolo_result()]

        img = create_test_image()
        response = test_client.post(
            "/analyze",
            data={"file": (img, "test.jpg"), "wall_incline": "vertical"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        breakdown = response.get_json()["score_breakdown"]

        # Calculate expected weighted sum
        weights = breakdown["weights"]
        expected_score = (
            breakdown["hold_difficulty"] * weights["hold_difficulty"]
            + breakdown["hold_density"] * weights["hold_density"]
            + breakdown["distance"] * weights["distance"]
            + breakdown["wall_incline"] * weights["wall_incline"]
        )

        # Use pytest.approx for cleaner float comparison
        assert breakdown["final_score"] == pytest.approx(expected_score, abs=0.01), (
            f"Final score {breakdown['final_score']} doesn't match "
            f"weighted sum {expected_score}"
        )


class TestPerformance:
    """Test prediction performance requirements."""

    @pytest.mark.skipif(
        os.environ.get("GITHUB_ACTIONS") == "true",
        reason="Performance tests are flaky on CI runners due to variable load",
    )
    @pytest.mark.slow
    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_prediction_under_500ms(
        self, mock_model, _mock_get_hold_types, test_client
    ):
        """Test that full E2E request completes within 500ms.

        The 500ms threshold accounts for:
        - Flask test client HTTP overhead
        - File upload processing
        - JSON serialization/deserialization
        - Test framework overhead

        The underlying prediction algorithm targets <100ms, but this E2E test
        measures the complete request cycle with mocked YOLO detection.

        Note: This test is skipped on CI due to variable runner performance.
        Run locally with: pytest -m slow
        """
        # Use a simple mock that returns immediately
        mock_model.return_value = [create_mock_yolo_result()]

        img = create_test_image()

        start = time.perf_counter()
        response = test_client.post(
            "/analyze",
            data={"file": (img, "test.jpg"), "wall_incline": "vertical"},
            content_type="multipart/form-data",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        # 500ms threshold for full E2E request (prediction algorithm itself is <100ms)
        assert elapsed_ms < 500, f"E2E request took {elapsed_ms:.2f}ms (>500ms limit)"


class TestEdgeCases:
    """Test edge cases in E2E flow."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_no_holds_detected(self, mock_model, _mock_get_hold_types, test_client):
        """Test handling when no holds are detected."""
        # Return empty detection result
        mock_result = Mock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        img = create_test_image()
        response = test_client.post(
            "/analyze",
            data={"file": (img, "test.jpg"), "wall_incline": "vertical"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["features"]["total_holds"] == 0
        assert "predicted_grade" in json_data  # Should still produce a grade

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_many_holds_detected(self, mock_model, _mock_get_hold_types, test_client):
        """Test handling when many holds are detected."""
        mock_model.return_value = [create_mock_yolo_result(num_boxes=50)]

        img = create_test_image()
        response = test_client.post(
            "/analyze",
            data={"file": (img, "test.jpg"), "wall_incline": "vertical"},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["features"]["total_holds"] == 50
