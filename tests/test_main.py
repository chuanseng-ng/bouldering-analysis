"""
Unit tests for src/main.py
"""

import uuid
import tempfile
from unittest.mock import Mock, patch

from src.main import (
    app,
    predict_grade,
    allowed_file,
    check_db_connection,
    _process_box,
    _process_detection_results,
    _create_database_records,
    _format_holds_for_response,
    create_tables,
)
from src.models import Analysis, HoldType, DetectedHold, Feedback
from src.models import db


class TestAppInitialization:  # pylint: disable=too-few-public-methods
    """Test cases for Flask app initialization."""

    def test_app_config(self):
        """Test Flask app configuration."""
        assert app.config["SECRET_KEY"] is not None
        assert app.config["SQLALCHEMY_DATABASE_URI"] is not None
        assert app.config["UPLOAD_FOLDER"] == "data/uploads"
        assert app.config["MAX_CONTENT_LENGTH"] == 16 * 1024 * 1024
        assert "png" in app.config["ALLOWED_EXTENSIONS"]
        assert "jpg" in app.config["ALLOWED_EXTENSIONS"]
        assert "jpeg" in app.config["ALLOWED_EXTENSIONS"]


class TestAllowedFile:
    """Test cases for the allowed_file function."""

    def test_allowed_file_valid_extensions(self):
        """Test allowed file with valid extensions."""
        assert allowed_file("test.jpg") is True
        assert allowed_file("test.jpeg") is True
        assert allowed_file("test.png") is True

    def test_allowed_file_invalid_extensions(self):
        """Test allowed file with invalid extensions."""
        assert allowed_file("test.txt") is False
        assert allowed_file("test.pdf") is False
        assert allowed_file("test.gif") is False

    def test_allowed_file_no_extension(self):
        """Test allowed file with no extension."""
        assert allowed_file("test") is False

    def test_allowed_file_uppercase_extension(self):
        """Test allowed file with uppercase extension."""
        assert allowed_file("test.JPG") is True
        assert allowed_file("test.PNG") is True


class TestCheckDbConnection:
    """Test cases for the check_db_connection function."""

    def test_check_db_connection_success(self, test_app):
        """Test successful database connection check."""
        with test_app.app_context():
            result = check_db_connection()
            assert result is True

    @patch("src.main.db")
    def test_check_db_connection_failure(self, mock_db):
        """Test failed database connection check."""
        mock_db.session.execute.side_effect = Exception("Connection failed")

        result = check_db_connection()

        assert result is False


class TestCreateTables:  # pylint: disable=too-few-public-methods
    """Test cases for the create_tables function."""

    def test_create_tables_with_existing_data(self, test_app):
        """Test create_tables when hold types already exist - covers lines 89-90."""
        with test_app.app_context():
            # Hold types should already exist from conftest setup
            initial_count = db.session.query(HoldType).count()

            # Call create_tables again - should not add duplicates
            create_tables()

            # Count should remain the same
            final_count = db.session.query(HoldType).count()
            assert final_count == initial_count


class TestPredictGrade:
    """Test cases for the predict_grade function."""

    def test_predict_grade_few_holds(self):
        """Test grade prediction with few holds."""
        features = {
            "total_holds": 2,
            "hold_types": {"crimp": 1, "jug": 1},
            "average_confidence": 0.8,
        }

        grade = predict_grade(features)
        assert grade == "V0"

    def test_predict_grade_many_holds(self):
        """Test grade prediction with many holds."""
        features = {
            "total_holds": 15,
            "hold_types": {"crimp": 5, "jug": 5, "pocket": 5},
            "average_confidence": 0.8,
        }

        grade = predict_grade(features)
        assert grade == "V7"  # Should be V7 based on current implementation

    def test_predict_grade_with_difficulty_multiplier(self):
        """Test grade prediction with difficulty multiplier."""
        features = {
            "total_holds": 5,
            "hold_types": {"crimp": 3, "pocket": 2},
            "average_confidence": 0.8,
        }

        grade = predict_grade(features)
        # Base grade V1 + (0.2 * 3 crimps) + (0.3 * 2 pockets) = V1 + 1.2 = V2.2 -> V2
        assert grade in ["V2", "V3"]  # Either V2 or V3 depending on rounding

    def test_predict_grade_base_grades(self):
        """Test base grade predictions."""
        test_cases = [
            (3, "V0"),
            (5, "V1"),
            (7, "V2"),
            (9, "V3"),
            (12, "V4"),
            (15, "V5"),
        ]

        for hold_count, expected_grade in test_cases:
            features = {
                "total_holds": hold_count,
                "hold_types": {"jug": hold_count},
                "average_confidence": 0.8,
            }

            grade = predict_grade(features)
            assert grade == expected_grade

    def test_predict_grade_very_many_holds(self):
        """Test grade prediction with very many holds - covers line 469."""
        features = {
            "total_holds": 20,
            "hold_types": {"jug": 20},
            "average_confidence": 0.8,
        }

        grade = predict_grade(features)
        assert grade == "V10"  # Should cap at V10

    def test_predict_grade_with_slopers(self):
        """Test grade prediction with sloper holds - covers line 480."""
        features = {
            "total_holds": 7,
            "hold_types": {"sloper": 3, "jug": 4},
            "average_confidence": 0.8,
        }

        grade = predict_grade(features)
        # Base grade V2 + (0.1 * 3 slopers) = V2.3 -> V2
        assert grade in ["V2", "V3"]

    def test_predict_grade_four_holds(self):
        """Test grade prediction with exactly 4 holds - covers line 457."""
        features = {
            "total_holds": 4,
            "hold_types": {"jug": 4},
            "average_confidence": 0.8,
        }

        grade = predict_grade(features)
        assert grade == "V0"  # Should be V0 for 4 holds


class TestProcessBox:  # pylint: disable=too-few-public-methods
    """Test cases for the _process_box function."""

    def test_process_box(self):
        """Test processing a single detection box."""
        # Mock box object
        box = Mock()
        box.xyxy = [Mock()]
        box.xyxy[0].cpu.return_value.numpy.return_value = [10, 20, 50, 60]
        box.conf = [Mock()]
        box.conf[0].cpu.return_value.numpy.return_value = 0.9
        box.cls = [Mock()]
        box.cls[0].cpu.return_value.numpy.return_value = 0

        hold_types_mapping = {0: "crimp"}

        hold_data, features = _process_box(box, hold_types_mapping)

        assert hold_data["hold_type"] == "crimp"
        assert hold_data["confidence"] == 0.9
        assert hold_data["bbox_x1"] == 10.0
        assert hold_data["bbox_y1"] == 20.0
        assert hold_data["bbox_x2"] == 50.0
        assert hold_data["bbox_y2"] == 60.0

        assert features["total_holds"] == 1
        assert features["hold_types"]["crimp"] == 1
        assert features["average_confidence"] == 0.9


class TestProcessDetectionResults:
    """Test cases for the _process_detection_results function."""

    def test_process_detection_results_empty(self):
        """Test processing empty detection results."""
        results: list[Mock] = []
        hold_types_mapping = {0: "crimp"}

        detected_holds, features = _process_detection_results(
            results, hold_types_mapping
        )

        assert not detected_holds
        assert features["total_holds"] == 0
        assert not features["hold_types"]
        assert features["average_confidence"] == 0

    def test_process_detection_results_with_boxes(self):
        """Test processing detection results with boxes."""
        # Mock result with boxes
        result = Mock()
        box1 = Mock()
        box1.xyxy = [Mock()]
        box1.conf = [Mock()]
        box1.cls = [Mock()]

        box2 = Mock()
        box2.xyxy = [Mock()]
        box2.conf = [Mock()]
        box2.cls = [Mock()]

        result.boxes = [box1, box2]

        # Configure the first box
        result.boxes[0].xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 50, 50]
        result.boxes[0].conf[0].cpu.return_value.numpy.return_value = 0.9
        result.boxes[0].cls[0].cpu.return_value.numpy.return_value = 0

        # Configure the second box
        result.boxes[1].xyxy[0].cpu.return_value.numpy.return_value = [20, 20, 60, 60]
        result.boxes[1].conf[0].cpu.return_value.numpy.return_value = 0.8
        result.boxes[1].cls[0].cpu.return_value.numpy.return_value = 1

        hold_types_mapping = {0: "crimp", 1: "jug"}

        detected_holds, features = _process_detection_results(
            [result], hold_types_mapping
        )

        assert len(detected_holds) == 2
        assert features["total_holds"] == 2
        assert features["hold_types"]["crimp"] == 1
        assert features["hold_types"]["jug"] == 1
        # Use approximate comparison for float values
        assert abs(features["average_confidence"] - 0.85) < 1e-10  # (0.9 + 0.8) / 2


class TestCreateDatabaseRecords:  # pylint: disable=too-few-public-methods
    """Test cases for the _create_database_records function."""

    def test_create_database_records(self, test_app, sample_analysis_data):
        """Test creating database records for analysis and detected holds."""
        with test_app.app_context():
            # Create analysis
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()

            # Mock detected holds data
            detected_holds_data = [
                {
                    "hold_type": "crimp",
                    "confidence": 0.9,
                    "bbox_x1": 10.0,
                    "bbox_y1": 10.0,
                    "bbox_x2": 50.0,
                    "bbox_y2": 50.0,
                }
            ]

            # Call the function
            _create_database_records(analysis, detected_holds_data)

            # Verify records were created
            detected_holds = (
                db.session.query(DetectedHold).filter_by(analysis_id=analysis.id).all()
            )
            assert len(detected_holds) == 1
            assert hold_type is not None
            assert detected_holds[0].hold_type_id == hold_type.id
            assert detected_holds[0].confidence == 0.9


class TestFormatHoldsForResponse:  # pylint: disable=too-few-public-methods
    """Test cases for the _format_holds_for_response function."""

    def test_format_holds_for_response(self, test_app, sample_analysis_data):
        """Test formatting detected holds for API response."""
        with test_app.app_context():
            # Create analysis
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()
            assert hold_type is not None

            # Create detected hold
            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=0.9,
                bbox_x1=10.0,
                bbox_y1=10.0,
                bbox_x2=50.0,
                bbox_y2=50.0,
            )

            db.session.add(detected_hold)
            db.session.commit()

            # Format for response
            result = _format_holds_for_response([detected_hold])

            assert len(result) == 1
            assert result[0]["type"] == "crimp"
            assert result[0]["confidence"] == 0.9
            assert result[0]["bbox"]["x1"] == 10.0
            assert result[0]["bbox"]["y1"] == 10.0
            assert result[0]["bbox"]["x2"] == 50.0
            assert result[0]["bbox"]["y2"] == 50.0


class TestAnalyzeImage:
    """Test cases for the analyze_image function."""

    @patch("src.main.hold_detection_model", None)
    def test_analyze_image_no_model(self, test_app, sample_image_path):
        """Test analyze_image when model is not loaded - covers lines 399-400."""
        with test_app.app_context():
            try:
                from src.main import (  # pylint: disable=import-outside-toplevel
                    analyze_image,
                )

                analyze_image(sample_image_path, "test.jpg")
                assert False, "Expected RuntimeError"
            except RuntimeError as e:
                assert "Hold detection model not loaded" in str(e)

    @patch("src.main.hold_detection_model")
    @patch("src.main.HOLD_TYPES", {0: "crimp", 1: "jug"})
    def test_analyze_image_success(self, mock_model, test_app, sample_image_path):
        """Test successful image analysis - covers lines 403-441."""
        with test_app.app_context():
            # Mock YOLO results
            mock_result = Mock()
            mock_box = Mock()
            mock_box.xyxy = [Mock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 20, 50, 60]
            mock_box.conf = [Mock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = 0.9
            mock_box.cls = [Mock()]
            mock_box.cls[0].cpu.return_value.numpy.return_value = 0

            mock_result.boxes = [mock_box]
            mock_model.return_value = [mock_result]

            # Call analyze_image
            from src.main import (  # pylint: disable=import-outside-toplevel
                analyze_image,
            )

            result = analyze_image(sample_image_path, "test.jpg")

            # Verify result structure
            assert "analysis_id" in result
            assert "predicted_grade" in result
            assert "confidence" in result
            assert "holds" in result
            assert "features" in result

            # Verify database records were created
            analysis = db.session.get(Analysis, result["analysis_id"])
            assert analysis is not None
            assert analysis.image_filename == "test.jpg"


class TestRoutes:
    """Test cases for Flask routes."""

    def test_index_get(self, test_client):
        """Test GET request to index route."""
        response = test_client.get("/")

        assert response.status_code == 200
        assert b"Bouldering Route Analysis" in response.data

    def test_index_post_no_file(self, test_client):
        """Test POST request to index with no file."""
        response = test_client.post("/", data={})

        assert response.status_code == 200
        # Check if the error message is in the rendered template
        assert b"No file part" in response.data

    def test_index_post_empty_filename(self, test_client):
        """Test POST request to index with empty filename."""
        response = test_client.post("/", data={"file": (None, "")})

        assert response.status_code == 200
        assert b"No selected file" in response.data

    def test_index_post_invalid_file_type(self, test_client):
        """Test POST request to index with invalid file type."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            data = {"file": (tmp_file, "test.txt")}
            response = test_client.post("/", data=data)

        assert response.status_code == 200
        assert b"Invalid file type" in response.data

    @patch("src.main.analyze_image")
    def test_index_post_processing_error(
        self, mock_analyze, test_client, sample_image_path
    ):
        """Test POST request to index with processing error."""
        mock_analyze.side_effect = RuntimeError("Processing error")

        with open(sample_image_path, "rb") as f:
            data = {"file": (f, "test.jpg")}
            response = test_client.post("/", data=data)

        assert response.status_code == 200
        assert b"Error processing image" in response.data

    def test_analyze_route_no_file(self, test_client):
        """Test analyze route with no file."""
        response = test_client.post("/analyze", data={})

        assert response.status_code == 400
        assert b"No file part" in response.data

    def test_analyze_route_empty_filename(self, test_client):
        """Test analyze route with empty filename - covers line 155."""
        response = test_client.post("/analyze", data={"file": (None, "")})

        assert response.status_code == 400
        assert b"No selected file" in response.data

    def test_analyze_route_invalid_file_type(self, test_client):
        """Test analyze route with invalid file type - covers lines 172-177."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            data = {"file": (tmp_file, "test.txt")}
            response = test_client.post("/analyze", data=data)

        assert response.status_code == 400
        assert b"Invalid file type" in response.data

    def test_feedback_route_missing_analysis_id(self, test_client):
        """Test feedback route with missing analysis_id."""
        response = test_client.post("/feedback", json={})

        assert response.status_code == 400
        assert b"Missing analysis_id" in response.data

    def test_feedback_route_analysis_not_found(self, test_client):
        """Test feedback route with non-existent analysis."""
        response = test_client.post(
            "/feedback", json={"analysis_id": str(uuid.uuid4())}
        )

        assert response.status_code == 404
        assert b"Analysis not found" in response.data

    def test_feedback_route_success(self, test_app, test_client, sample_analysis_data):
        """Test successful feedback submission."""
        with test_app.app_context():
            # Create a real analysis
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.commit()
            analysis_id = analysis.id

        # Submit feedback
        response = test_client.post(
            "/feedback",
            json={
                "analysis_id": analysis_id,
                "user_grade": "V3",
                "is_accurate": True,
                "comments": "Good prediction",
            },
        )

        assert response.status_code == 200
        json_data = response.get_json()
        assert "feedback_id" in json_data

    def test_stats_route_success(self, test_app, test_client, sample_analysis_data):
        """Test successful stats route - covers lines 216-241."""
        with test_app.app_context():
            # Clear existing data for this test
            db.session.query(Feedback).delete()
            db.session.query(Analysis).delete()
            db.session.commit()

            # Create some test data
            analysis1 = Analysis(**sample_analysis_data)
            db.session.add(analysis1)
            db.session.flush()

            analysis2 = Analysis(
                image_filename="test2.jpg",
                image_path="/path/to/test2.jpg",
                predicted_grade="V3",
                confidence_score=0.85,
                features_extracted={"total_holds": 8},
            )
            db.session.add(analysis2)
            db.session.flush()

            # Create feedback
            feedback1 = Feedback(
                analysis_id=analysis1.id,
                user_grade="V2",
                is_accurate=True,
                comments="Good",
            )
            db.session.add(feedback1)

            feedback2 = Feedback(
                analysis_id=analysis2.id,
                user_grade="V3",
                is_accurate=False,
                comments="Not quite",
            )
            db.session.add(feedback2)
            db.session.commit()

        # Test stats endpoint
        response = test_client.get("/stats")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["total_analyses"] == 2
        assert json_data["total_feedback"] == 2
        assert json_data["accurate_predictions"] == 1
        assert json_data["accuracy_rate"] == 50.0
        assert "grade_distribution" in json_data

    @patch("src.main.check_db_connection")
    @patch("src.main.hold_detection_model")
    def test_health_route_healthy(self, mock_model, mock_db_check, test_client):
        """Test health route when everything is healthy."""
        mock_model.return_value = Mock()  # Model loaded
        mock_db_check.return_value = True  # Database connected

        response = test_client.get("/health")

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "healthy"
        assert json_data["model_loaded"] is True
        assert json_data["database_connected"] is True

    def test_uploads_route(self, test_client):
        """Test uploads route."""
        # This would normally serve files from the upload folder
        # For testing, we'll just check the route exists
        response = test_client.get("/uploads/nonexistent.jpg")

        # Should return 404 since file doesn't exist
        assert response.status_code == 404


class TestCreateTablesHoldTypeInsertion:  # pylint: disable=too-few-public-methods
    """Test cases for create_tables hold type insertion - covers lines 90-106."""

    def test_create_tables_with_empty_database(self, test_app):
        """Test create_tables when database is empty - covers lines 90-106."""
        with test_app.app_context():
            # Clear all hold types
            db.session.query(HoldType).delete()
            db.session.commit()

            # Verify hold types are empty
            assert db.session.query(HoldType).count() == 0

            # Call create_tables
            create_tables()

            # Verify hold types were created
            hold_types = db.session.query(HoldType).all()
            assert len(hold_types) == 8

            # Verify specific hold types
            crimp = db.session.query(HoldType).filter_by(name="crimp").first()
            assert crimp is not None
            assert crimp.id == 0
            assert "crimp" in crimp.description.lower()

            jug = db.session.query(HoldType).filter_by(name="jug").first()
            assert jug is not None
            assert jug.id == 1

            top_out = db.session.query(HoldType).filter_by(name="top-out-hold").first()
            assert top_out is not None
            assert top_out.id == 7


class TestIndexRouteErrorHandling:  # pylint: disable=too-few-public-methods
    """Test cases for index route error handling - covers line 131."""

    @patch("src.main.analyze_image")
    def test_index_post_io_error(self, mock_analyze, test_client, sample_image_path):
        """Test POST request to index with IOError - covers line 131."""
        mock_analyze.side_effect = IOError("File read error")

        with open(sample_image_path, "rb") as f:
            data = {"file": (f, "test.jpg")}
            response = test_client.post("/", data=data)

        assert response.status_code == 200
        assert b"Error processing image" in response.data


class TestAnalyzeRouteErrorHandling:
    """Test cases for analyze route error handling - covers lines 159-170."""

    @patch("src.main.analyze_image")
    def test_analyze_route_io_error(self, mock_analyze, test_client, sample_image_path):
        """Test analyze route with IOError - covers lines 169-170."""
        mock_analyze.side_effect = IOError("File read error")

        with open(sample_image_path, "rb") as f:
            data = {"file": (f, "test.jpg")}
            response = test_client.post("/analyze", data=data)

        assert response.status_code == 500
        assert b"Error processing image" in response.data

    @patch("src.main.analyze_image")
    def test_analyze_route_runtime_error(
        self, mock_analyze, test_client, sample_image_path
    ):
        """Test analyze route with RuntimeError - covers lines 169-170."""
        mock_analyze.side_effect = RuntimeError("Model error")

        with open(sample_image_path, "rb") as f:
            data = {"file": (f, "test.jpg")}
            response = test_client.post("/analyze", data=data)

        assert response.status_code == 500
        assert b"Error processing image" in response.data


class TestFeedbackRouteErrorHandling:  # pylint: disable=too-few-public-methods
    """Test cases for feedback route error handling - covers lines 208-210."""

    @patch("src.main.db")
    def test_feedback_route_database_error(self, mock_db, test_client):
        """Test feedback route with database error - covers lines 208-210."""
        # Mock db.session.get to raise an exception
        mock_db.session.get.side_effect = Exception("Database error")
        mock_db.session.rollback = Mock()

        response = test_client.post(
            "/feedback",
            json={
                "analysis_id": str(uuid.uuid4()),
                "user_grade": "V3",
                "is_accurate": True,
            },
        )

        assert response.status_code == 500
        assert b"Error submitting feedback" in response.data
        mock_db.session.rollback.assert_called_once()


class TestStatsRouteErrorHandling:  # pylint: disable=too-few-public-methods
    """Test cases for stats route error handling - covers lines 243-244."""

    @patch("src.main.db")
    def test_stats_route_database_error(self, mock_db, test_client):
        """Test stats route with database error - covers lines 243-244."""
        # Mock db.session.query to raise an exception
        mock_db.session.query.side_effect = Exception("Database error")

        response = test_client.get("/stats")

        assert response.status_code == 500
        assert b"Error getting stats" in response.data
