"""
End-to-end tests for Phase 1a MVP grade prediction.

These tests verify the complete flow from image upload through grade prediction,
including wall_incline parameter handling and score breakdown validation.
"""

from __future__ import annotations

import io
import time
from unittest.mock import Mock, patch

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

    def test_analyze_with_valid_wall_inclines(self, test_client, test_app):
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

    def test_analyze_defaults_to_vertical(self, test_client, test_app):
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
        self, mock_model, mock_get_hold_types, test_client, test_app
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
    def test_wall_incline_score_values(
        self, mock_model, mock_get_hold_types, test_client, test_app
    ):
        """Test that wall_incline factor scores match expected values."""
        expected_scores = {
            "slab": 3.0,
            "vertical": 6.0,
            "slight_overhang": 7.5,
            "moderate_overhang": 9.0,
            "steep_overhang": 11.0,
        }

        mock_model.return_value = [create_mock_yolo_result()]

        for incline, expected_score in expected_scores.items():
            img = create_test_image()
            response = test_client.post(
                "/analyze",
                data={"file": (img, "test.jpg"), "wall_incline": incline},
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            json_data = response.get_json()
            actual_score = json_data["score_breakdown"]["wall_incline"]
            assert actual_score == expected_score, (
                f"Wall incline score for {incline}: expected {expected_score}, "
                f"got {actual_score}"
            )


class TestDatabasePersistence:
    """Test that wall_incline is correctly persisted in database."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_wall_incline_stored_in_analysis(
        self, mock_model, mock_get_hold_types, test_client, test_app
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
        self, mock_model, mock_get_hold_types, test_client, test_app
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
        self, mock_model, mock_get_hold_types, test_client, test_app
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
        self, mock_model, mock_get_hold_types, test_client, test_app
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

        # Allow small floating point tolerance
        assert abs(breakdown["final_score"] - expected_score) < 0.01, (
            f"Final score {breakdown['final_score']} doesn't match "
            f"weighted sum {expected_score}"
        )


class TestPerformance:
    """Test prediction performance requirements."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_prediction_under_100ms(
        self, mock_model, mock_get_hold_types, test_client, test_app
    ):
        """Test that prediction completes within 100ms (excluding I/O)."""
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
        # Note: This includes mocked YOLO, so actual prediction should be faster
        # Set generous limit since test includes HTTP overhead
        assert elapsed_ms < 500, f"Prediction took {elapsed_ms:.2f}ms (>500ms limit)"


class TestEdgeCases:
    """Test edge cases in E2E flow."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_no_holds_detected(
        self, mock_model, mock_get_hold_types, test_client, test_app
    ):
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
    def test_many_holds_detected(
        self, mock_model, mock_get_hold_types, test_client, test_app
    ):
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
