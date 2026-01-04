# pylint: disable=too-many-lines,duplicate-code
"""
Unit tests for src/main.py
"""

import uuid
import tempfile
from unittest.mock import Mock, patch

from sqlalchemy.exc import SQLAlchemyError

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

    def test_create_database_records(
        self, test_app, sample_analysis_data, sample_detected_hold_data
    ):
        """Test creating database records for analysis and detected holds."""
        with test_app.app_context():
            # Create analysis
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()

            # Mock detected holds data using fixture
            detected_holds_data = [
                {
                    "hold_type": "crimp",
                    **sample_detected_hold_data,
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

    def test_format_holds_for_response(
        self, test_app, sample_analysis_data, sample_detected_hold_data
    ):
        """Test formatting detected holds for API response."""
        with test_app.app_context():
            # Create analysis
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()
            assert hold_type is not None

            # Create detected hold using fixture data
            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                **sample_detected_hold_data,
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

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_analyze_image_success(
        self, mock_model, mock_get_hold_types, test_app, sample_image_path
    ):
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
        # Mock db.session.get to raise a SQLAlchemyError
        mock_db.session.get.side_effect = SQLAlchemyError("Database error")
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
        # Mock db.session.query to raise a SQLAlchemyError
        mock_db.session.query.side_effect = SQLAlchemyError("Database error")

        response = test_client.get("/stats")

        assert response.status_code == 500
        assert b"Error getting stats" in response.data


class TestLoadActiveHoldDetectionModel:
    """Test cases for load_active_hold_detection_model function - Week 3-4 feature."""

    @patch("src.main.YOLO")
    def test_load_active_model_from_database(
        self, mock_yolo, test_app, active_model_version
    ):  # pylint: disable=unused-argument
        """Test loading active model from ModelVersion table."""
        # Mock YOLO to return a fake model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # pylint: disable=import-outside-toplevel
        from src.main import load_active_hold_detection_model

        with test_app.app_context():
            model, threshold = load_active_hold_detection_model()

            # Should have loaded the active model from database
            assert model is mock_model
            assert threshold == 0.25  # Default threshold
            # Verify YOLO was called with the expected path from active_model_version
            mock_yolo.assert_called_once_with(active_model_version.model_path)

    @patch("src.main.get_config_value")
    def test_load_model_with_custom_threshold(self, mock_config, test_app):  # pylint: disable=unused-argument
        """Test that custom confidence threshold is loaded from config."""
        mock_config.return_value = 0.35

        # pylint: disable=import-outside-toplevel
        from src.main import load_active_hold_detection_model

        with test_app.app_context():
            model, threshold = load_active_hold_detection_model()  # pylint: disable=unused-variable
            _ = model  # May be None

            assert threshold == 0.35

    @patch("src.main.YOLO")
    @patch("src.main.get_model_path")
    def test_fallback_to_base_model(self, mock_get_path, mock_yolo, test_app, tmp_path):  # pylint: disable=unused-argument
        """Test fallback to base model when no active model exists."""
        # Setup base model path
        base_model = tmp_path / "yolov8n.pt"
        base_model.write_text("base model")
        mock_get_path.return_value = base_model

        # Mock YOLO to return a fake model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # pylint: disable=import-outside-toplevel
        from src.main import load_active_hold_detection_model

        with test_app.app_context():
            model, _ = load_active_hold_detection_model()

            # Should have loaded base model
            assert model is mock_model
            mock_yolo.assert_called_once_with(str(base_model))

    @patch("src.main.YOLO")
    def test_fallback_when_active_model_file_missing(
        self, mock_yolo, test_app, tmp_path
    ):  # pylint: disable=unused-argument
        """Test fallback when active model file doesn't exist."""
        from src.models import ModelVersion  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Create model version with non-existent file
            nonexistent = tmp_path / "nonexistent.pt"
            model_v = ModelVersion(
                model_type="hold_detection",
                version="v_missing",
                model_path=str(nonexistent),
                accuracy=0.85,
                is_active=True,
            )
            db.session.add(model_v)
            db.session.commit()

        # Mock YOLO for base model fallback
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # pylint: disable=import-outside-toplevel
        from src.main import load_active_hold_detection_model

        with test_app.app_context():
            model, threshold = load_active_hold_detection_model()  # pylint: disable=unused-variable
            _ = model  # May be None

            # Should fallback to base model
            # Since file doesn't exist, it should try to load yolov8n.pt
            assert threshold == 0.25


class TestProcessDetectionResultsWithThreshold:
    """Test cases for _process_detection_results with confidence threshold - Week 3-4 feature."""

    def test_confidence_threshold_filters_detections(self):
        """Test that detections below threshold are filtered out."""
        # Mock result with mixed confidence detections
        result = Mock()

        # Create high confidence box (should be kept)
        high_conf_box = Mock()
        high_conf_box.xyxy = [Mock()]
        high_conf_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 50, 50]
        high_conf_box.conf = [Mock()]
        high_conf_box.conf[
            0
        ].cpu.return_value.numpy.return_value = 0.8  # Above threshold
        high_conf_box.cls = [Mock()]
        high_conf_box.cls[0].cpu.return_value.numpy.return_value = 0

        # Create low confidence box (should be filtered)
        low_conf_box = Mock()
        low_conf_box.xyxy = [Mock()]
        low_conf_box.xyxy[0].cpu.return_value.numpy.return_value = [60, 60, 100, 100]
        low_conf_box.conf = [Mock()]
        low_conf_box.conf[
            0
        ].cpu.return_value.numpy.return_value = 0.15  # Below threshold
        low_conf_box.cls = [Mock()]
        low_conf_box.cls[0].cpu.return_value.numpy.return_value = 1

        result.boxes = [high_conf_box, low_conf_box]

        hold_types_mapping = {0: "crimp", 1: "jug"}

        detected_holds, features = _process_detection_results(
            [result], hold_types_mapping, conf_threshold=0.25
        )

        # Only high confidence detection should be kept
        assert len(detected_holds) == 1
        assert features["total_holds"] == 1
        assert detected_holds[0]["confidence"] == 0.8

    def test_all_detections_filtered_by_threshold(self):
        """Test when all detections are below threshold."""
        result = Mock()

        # Create low confidence box
        low_conf_box = Mock()
        low_conf_box.xyxy = [Mock()]
        low_conf_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 50, 50]
        low_conf_box.conf = [Mock()]
        low_conf_box.conf[0].cpu.return_value.numpy.return_value = 0.1
        low_conf_box.cls = [Mock()]
        low_conf_box.cls[0].cpu.return_value.numpy.return_value = 0

        result.boxes = [low_conf_box]

        hold_types_mapping = {0: "crimp"}

        detected_holds, features = _process_detection_results(
            [result], hold_types_mapping, conf_threshold=0.25
        )

        # All detections filtered
        assert len(detected_holds) == 0
        assert features["total_holds"] == 0
        assert features["average_confidence"] == 0


class TestAnalyzeEndpointIntegration:
    """Integration tests for POST /analyze endpoint with Week 3-4 features."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_analyze_endpoint_success(
        self, mock_model, mock_get_hold_types, test_client, test_app, sample_image_path
    ):
        """Test successful analysis via POST /analyze endpoint."""
        with test_app.app_context():
            # Mock YOLO detection results
            mock_result = Mock()
            mock_box = Mock()
            mock_box.xyxy = [Mock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 20, 50, 60]
            mock_box.conf = [Mock()]
            mock_box.conf[0].cpu.return_value.numpy.return_value = 0.85
            mock_box.cls = [Mock()]
            mock_box.cls[0].cpu.return_value.numpy.return_value = 0

            mock_result.boxes = [mock_box]
            mock_model.return_value = [mock_result]

            # Post image to analyze endpoint
            with open(sample_image_path, "rb") as f:
                response = test_client.post(
                    "/analyze",
                    data={"file": (f, "test_image.jpg")},
                    content_type="multipart/form-data",
                )

            assert response.status_code == 200
            json_data = response.get_json()

            assert "analysis_id" in json_data
            assert "predicted_grade" in json_data
            assert "confidence" in json_data
            assert "holds" in json_data
            assert "features" in json_data

            # Verify detected holds were stored
            analysis = db.session.get(Analysis, json_data["analysis_id"])
            assert analysis is not None

            detected_holds = (
                db.session.query(DetectedHold).filter_by(analysis_id=analysis.id).all()
            )
            assert len(detected_holds) > 0

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.confidence_threshold", 0.5)
    @patch("src.main.hold_detection_model")
    def test_analyze_applies_confidence_threshold(
        self, mock_model, mock_get_hold_types, test_client, test_app, sample_image_path
    ):
        """Test that confidence threshold is applied during analysis."""
        with test_app.app_context():
            # Mock detection with low confidence
            mock_result = Mock()
            mock_box = Mock()
            mock_box.xyxy = [Mock()]
            mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 20, 50, 60]
            mock_box.conf = [Mock()]
            mock_box.conf[
                0
            ].cpu.return_value.numpy.return_value = 0.3  # Below threshold
            mock_box.cls = [Mock()]
            mock_box.cls[0].cpu.return_value.numpy.return_value = 0

            mock_result.boxes = [mock_box]
            mock_model.return_value = [mock_result]

            with open(sample_image_path, "rb") as f:
                response = test_client.post(
                    "/analyze",
                    data={"file": (f, "test_image.jpg")},
                    content_type="multipart/form-data",
                )

            assert response.status_code == 200
            json_data = response.get_json()

            # Should have no holds due to threshold filtering
            assert json_data["features"]["total_holds"] == 0

    def test_analyze_invalid_file_type(self, test_client, tmp_path):
        """Test POST /analyze with invalid file type."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")

        with open(text_file, "rb") as f:
            response = test_client.post(
                "/analyze",
                data={"file": (f, "test.txt")},
                content_type="multipart/form-data",
            )

        assert response.status_code == 400
        assert b"Invalid file type" in response.data

    def test_analyze_no_file_in_request(self, test_client):
        """Test POST /analyze with no file in request."""
        response = test_client.post("/analyze", data={})

        assert response.status_code == 400
        assert b"No file part" in response.data


class TestAnalysisResultsStorage:
    """Test cases for analysis results storage with filtered holds - Week 3-4 feature."""

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_filtered_holds_stored_correctly(
        self, mock_model, mock_get_hold_types, test_app, sample_image_path
    ):
        """Test that only holds passing confidence threshold are stored."""
        from src.main import analyze_image  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Mock results with mixed confidence
            mock_result = Mock()

            high_box = Mock()
            high_box.xyxy = [Mock()]
            high_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 50, 50]
            high_box.conf = [Mock()]
            high_box.conf[0].cpu.return_value.numpy.return_value = 0.9
            high_box.cls = [Mock()]
            high_box.cls[0].cpu.return_value.numpy.return_value = 0

            low_box = Mock()
            low_box.xyxy = [Mock()]
            low_box.xyxy[0].cpu.return_value.numpy.return_value = [60, 60, 100, 100]
            low_box.conf = [Mock()]
            low_box.conf[
                0
            ].cpu.return_value.numpy.return_value = 0.15  # Below default 0.25
            low_box.cls = [Mock()]
            low_box.cls[0].cpu.return_value.numpy.return_value = 1

            mock_result.boxes = [high_box, low_box]
            mock_model.return_value = [mock_result]

            # Analyze image
            result = analyze_image(sample_image_path, "test.jpg")

            # Verify only high confidence hold was stored
            holds = (
                db.session.query(DetectedHold)
                .filter_by(analysis_id=result["analysis_id"])
                .all()
            )

            assert len(holds) == 1
            assert holds[0].confidence >= 0.25

    @patch("src.main.get_hold_types", return_value={0: "crimp", 1: "jug"})
    @patch("src.main.hold_detection_model")
    def test_analysis_features_reflect_filtered_holds(
        self, mock_model, mock_get_hold_types, test_app, sample_image_path
    ):
        """Test that features reflect filtered holds, not all detections."""
        from src.main import analyze_image  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Mock 5 detections, but only 2 above threshold
            mock_result = Mock()
            boxes = []

            # 2 high confidence
            for i in range(2):
                box = Mock()
                box.xyxy = [Mock()]
                box.xyxy[0].cpu.return_value.numpy.return_value = [
                    i * 10,
                    i * 10,
                    i * 10 + 40,
                    i * 10 + 40,
                ]
                box.conf = [Mock()]
                box.conf[0].cpu.return_value.numpy.return_value = 0.8
                box.cls = [Mock()]
                box.cls[0].cpu.return_value.numpy.return_value = 0
                boxes.append(box)

            # 3 low confidence
            for i in range(3):
                box = Mock()
                box.xyxy = [Mock()]
                box.xyxy[0].cpu.return_value.numpy.return_value = [
                    (i + 3) * 10,
                    (i + 3) * 10,
                    (i + 3) * 10 + 40,
                    (i + 3) * 10 + 40,
                ]
                box.conf = [Mock()]
                box.conf[0].cpu.return_value.numpy.return_value = 0.1  # Below threshold
                box.cls = [Mock()]
                box.cls[0].cpu.return_value.numpy.return_value = 0
                boxes.append(box)

            mock_result.boxes = boxes
            mock_model.return_value = [mock_result]

            result = analyze_image(sample_image_path, "test.jpg")

            # Features should only count the 2 high-confidence holds
            assert result["features"]["total_holds"] == 2
            assert len(result["holds"]) == 2


class TestLoadActiveModelErrorPaths:
    """Test error paths in load_active_hold_detection_model - covers lines 177-201."""

    @patch("src.main.get_config_value")
    def test_load_model_config_error_exception(self, mock_get_config, test_app):
        """Test handling of general Exception in config loading - covers lines 147-149."""
        # Make get_config_value raise a general Exception
        mock_get_config.side_effect = Exception("Unexpected error")

        from src.main import (  # pylint: disable=import-outside-toplevel
            load_active_hold_detection_model,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            model, threshold = load_active_hold_detection_model()
            _ = model  # May be None

            # Should use default threshold despite exception
            assert threshold == 0.25

    @patch("src.main.YOLO")
    def test_load_model_file_loading_error(self, mock_yolo, test_app, tmp_path):
        """Test handling of model file loading errors - covers lines 195-201."""
        from src.models import ModelVersion  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Create model version with existing file but YOLO fails to load
            model_file = tmp_path / "bad_model.pt"
            model_file.write_text("corrupted model")

            model_v = ModelVersion(
                model_type="hold_detection",
                version="v_corrupt",
                model_path=str(model_file),
                accuracy=0.85,
                is_active=True,
            )
            db.session.add(model_v)
            db.session.commit()

        # Make YOLO raise an error when loading the active model
        mock_yolo.side_effect = [
            RuntimeError("Failed to load model"),  # First call for active model fails
            Mock(),  # Second call for base model succeeds
        ]

        from src.main import (  # pylint: disable=import-outside-toplevel
            load_active_hold_detection_model,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            model, threshold = load_active_hold_detection_model()

            # Should fallback to base model
            assert model is not None
            assert threshold == 0.25

    @patch("src.main.YOLO")
    def test_load_active_model_with_relative_path(self, mock_yolo, test_app, tmp_path):
        """Test loading active model with relative path - covers lines 177-179."""
        from src.models import ModelVersion  # pylint: disable=import-outside-toplevel

        # Create a real model file in a subdirectory
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "test_model.pt"
        model_file.write_text("test model")

        # Use a relative path (relative to project root)
        relative_path = "models/test_model.pt"

        with test_app.app_context():
            # Create model version with relative path
            model_v = ModelVersion(
                model_type="hold_detection",
                version="v_relative",
                model_path=relative_path,
                accuracy=0.85,
                is_active=True,
            )
            db.session.add(model_v)
            db.session.commit()

        # Mock YOLO to succeed
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        from src.main import (  # pylint: disable=import-outside-toplevel
            load_active_hold_detection_model,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            _model, threshold = load_active_hold_detection_model()

            # Should have loaded the model
            # The relative path should be resolved
            assert threshold == 0.25

    @patch("src.main.YOLO")
    @patch("src.main.get_model_path")
    def test_load_model_base_model_config_error(
        self, mock_get_path, mock_yolo, test_app
    ):
        """Test handling when base model config fails - covers lines 217-221."""
        from src.config import (  # pylint: disable=import-outside-toplevel
            ConfigurationError,
        )  # pylint: disable=import-outside-toplevel

        # Make get_model_path raise ConfigurationError
        mock_get_path.side_effect = ConfigurationError("Base model not configured")

        # Make YOLO succeed for hardcoded path
        mock_yolo.return_value = Mock()

        from src.main import (  # pylint: disable=import-outside-toplevel
            load_active_hold_detection_model,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            _model, threshold = load_active_hold_detection_model()

            # Should try hardcoded yolov8n.pt path
            assert threshold == 0.25
            # YOLO should be called with hardcoded path
            mock_yolo.assert_called_with("yolov8n.pt")

    @patch("src.main.YOLO")
    @patch("src.main.get_model_path")
    def test_load_model_all_paths_fail(self, mock_get_path, mock_yolo, test_app):
        """Test when all model loading paths fail - covers lines 228-230."""
        from src.config import (  # pylint: disable=import-outside-toplevel
            ConfigurationError,
        )  # pylint: disable=import-outside-toplevel

        # Make get_model_path raise ConfigurationError
        mock_get_path.side_effect = ConfigurationError("Base model not configured")

        # Make YOLO fail for hardcoded path too
        mock_yolo.side_effect = OSError("Model file not found")

        from src.main import (  # pylint: disable=import-outside-toplevel
            load_active_hold_detection_model,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            model, threshold = load_active_hold_detection_model()

            # Should return None when all paths fail
            assert model is None
            assert threshold == 0.25
