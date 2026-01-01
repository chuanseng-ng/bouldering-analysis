"""
Unit tests for src/models.py
"""

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from src.models import (
    db,
    Base,
    Analysis,
    Feedback,
    HoldType,
    DetectedHold,
    ModelVersion,
    UserSession,
)


class TestBase:
    """Test cases for the Base model class."""

    def test_base_to_dict(self, test_app):
        """Test the Base class to_dict method using HoldType without override."""
        with test_app.app_context():
            # Create a simple model instance that can use Base.to_dict()
            # We'll use HoldType and call the parent's to_dict explicitly
            hold_type = HoldType(id=1, name="test_hold", description="Test description")

            # Call the Base class to_dict method directly
            base_dict = Base.to_dict(hold_type)

            # Verify it includes all columns
            assert "id" in base_dict
            assert "name" in base_dict
            assert "description" in base_dict
            assert base_dict["id"] == 1
            assert base_dict["name"] == "test_hold"
            assert base_dict["description"] == "Test description"

    def test_base_repr(self, test_app):
        """Test the Base class __repr__ method."""
        with test_app.app_context():
            hold_type = HoldType(id=1, name="test_hold")
            # The repr should show the class name
            repr_str = Base.__repr__(hold_type)
            assert repr_str == "<HoldType>"


class TestHoldType:
    """Test cases for the HoldType model."""

    def test_create_hold_type(self, test_app):
        """Test creating a HoldType instance."""
        with test_app.app_context():
            # Use a unique name that won't conflict
            unique_name = f"test_hold_{uuid4().hex[:8]}"
            hold_type = HoldType(
                name=unique_name,
                description="Test hold for testing purposes",
            )

            db.session.add(hold_type)
            db.session.commit()

            # Verify the hold type was created
            retrieved = db.session.query(HoldType).filter_by(name=unique_name).first()
            assert retrieved is not None
            assert retrieved.name == unique_name
            assert retrieved.description == "Test hold for testing purposes"

    def test_hold_type_to_dict(self, test_app):
        """Test the to_dict method of HoldType."""
        with test_app.app_context():
            hold_type = HoldType(
                id=1, name="jug", description="Large, easy-to-hold jug"
            )

            result = hold_type.to_dict()

            assert result == {
                "id": 1,
                "name": "jug",
                "description": "Large, easy-to-hold jug",
            }

    def test_hold_type_repr(self, test_app):
        """Test the __repr__ method of HoldType."""
        with test_app.app_context():
            hold_type = HoldType(id=2, name="sloper", description="Round hold")
            repr_str = repr(hold_type)

            assert "<HoldType 2: sloper>" in repr_str


class TestAnalysis:
    """Test cases for the Analysis model."""

    def test_create_analysis(self, test_app, sample_analysis_data):
        """Test creating an Analysis instance."""
        with test_app.app_context():
            analysis = Analysis(**sample_analysis_data)

            db.session.add(analysis)
            db.session.commit()

            # Verify the analysis was created
            retrieved = db.session.query(Analysis).filter_by(id=analysis.id).first()
            assert retrieved is not None
            assert retrieved.image_filename == "test.jpg"
            assert retrieved.predicted_grade == "V2"
            assert retrieved.confidence_score == 0.85

    def test_analysis_to_dict(self, test_app, sample_analysis_data):
        """Test the to_dict method of Analysis."""
        with test_app.app_context():
            analysis = Analysis(**sample_analysis_data)

            result = analysis.to_dict()

            assert result["id"] is not None
            assert result["image_filename"] == "test.jpg"
            assert result["predicted_grade"] == "V2"
            assert result["confidence_score"] == 0.85
            assert (
                result["features_extracted"]
                == sample_analysis_data["features_extracted"]
            )
            assert "created_at" in result
            assert "updated_at" in result

    def test_analysis_repr(self, test_app, sample_analysis_data):
        """Test the __repr__ method of Analysis."""
        with test_app.app_context():
            analysis = Analysis(**sample_analysis_data)
            repr_str = repr(analysis)

            assert "<Analysis" in repr_str
            assert "test.jpg" in repr_str
            assert "V2" in repr_str

    def test_analysis_relationships(self, test_app, sample_analysis_data):
        """Test Analysis relationships with other models."""
        with test_app.app_context():
            analysis = Analysis(**sample_analysis_data)

            # Create feedback
            feedback = Feedback(
                analysis_id=analysis.id,
                user_grade="V3",
                is_accurate=False,
                comments="Test feedback",
            )

            # Create detected holds
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()
            assert hold_type is not None
            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=0.9,
                bbox_x1=10.0,
                bbox_y1=10.0,
                bbox_x2=50.0,
                bbox_y2=50.0,
            )

            db.session.add_all([analysis, feedback, detected_hold])
            db.session.commit()

            # Test relationships
            assert analysis.feedback is not None
            assert analysis.feedback.user_grade == "V3"
            assert len(analysis.detected_holds) == 1
            assert analysis.detected_holds[0].hold_type.name == "crimp"  # type: ignore[index]


class TestFeedback:
    """Test cases for the Feedback model."""

    def test_create_feedback(
        self, test_app, sample_analysis_data, sample_feedback_data
    ):
        """Test creating a Feedback instance."""
        with test_app.app_context():
            # Create analysis first
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()  # Get the ID

            feedback = Feedback(analysis_id=analysis.id, **sample_feedback_data)

            db.session.add(feedback)
            db.session.commit()

            # Verify the feedback was created
            retrieved = db.session.query(Feedback).filter_by(id=feedback.id).first()
            assert retrieved is not None
            assert retrieved.analysis_id == analysis.id
            assert retrieved.user_grade == "V3"
            assert retrieved.is_accurate is False
            assert retrieved.comments == "The grade seems a bit high"

    def test_feedback_to_dict(
        self, test_app, sample_analysis_data, sample_feedback_data
    ):
        """Test the to_dict method of Feedback."""
        with test_app.app_context():
            # Create analysis first
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            feedback = Feedback(analysis_id=analysis.id, **sample_feedback_data)

            result = feedback.to_dict()

            assert result["id"] is not None
            assert result["analysis_id"] == analysis.id
            assert result["user_grade"] == "V3"
            assert result["is_accurate"] is False
            assert result["comments"] == "The grade seems a bit high"
            assert "created_at" in result

    def test_feedback_repr(self, test_app, sample_analysis_data, sample_feedback_data):
        """Test the __repr__ method of Feedback."""
        with test_app.app_context():
            # Create analysis first
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            feedback = Feedback(analysis_id=analysis.id, **sample_feedback_data)
            repr_str = repr(feedback)

            assert "<Feedback" in repr_str
            assert str(analysis.id) in repr_str
            assert "Accurate False" in repr_str


class TestDetectedHold:
    """Test cases for the DetectedHold model."""

    def test_create_detected_hold(
        self, test_app, sample_analysis_data, sample_detected_hold_data
    ):
        """Test creating a DetectedHold instance."""
        with test_app.app_context():
            # Create analysis first
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()
            assert hold_type is not None

            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=sample_detected_hold_data["confidence"],
                bbox_x1=sample_detected_hold_data["bbox_x1"],
                bbox_y1=sample_detected_hold_data["bbox_y1"],
                bbox_x2=sample_detected_hold_data["bbox_x2"],
                bbox_y2=sample_detected_hold_data["bbox_y2"],
            )

            db.session.add(detected_hold)
            db.session.commit()

            # Verify the detected hold was created
            retrieved = (
                db.session.query(DetectedHold).filter_by(id=detected_hold.id).first()
            )
            assert retrieved is not None
            assert retrieved.analysis_id == analysis.id
            assert retrieved.hold_type_id == hold_type.id
            assert retrieved.confidence == 0.9
            assert retrieved.bbox_x1 == 10.0
            assert retrieved.bbox_y1 == 10.0
            assert retrieved.bbox_x2 == 50.0
            assert retrieved.bbox_y2 == 50.0

    def test_detected_hold_to_dict(
        self, test_app, sample_analysis_data, sample_detected_hold_data
    ):
        """Test the to_dict method of DetectedHold."""
        with test_app.app_context():
            # Create analysis first
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()
            assert hold_type is not None

            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=sample_detected_hold_data["confidence"],
                bbox_x1=sample_detected_hold_data["bbox_x1"],
                bbox_y1=sample_detected_hold_data["bbox_y1"],
                bbox_x2=sample_detected_hold_data["bbox_x2"],
                bbox_y2=sample_detected_hold_data["bbox_y2"],
            )

            db.session.add(detected_hold)
            db.session.flush()

            result = detected_hold.to_dict()

            assert result["id"] is not None
            assert result["analysis_id"] == analysis.id
            assert result["hold_type_id"] == hold_type.id
            assert result["confidence"] == 0.9
            assert result["bbox"] == {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}
            assert "created_at" in result

    def test_detected_hold_repr(
        self, test_app, sample_analysis_data, sample_detected_hold_data
    ):
        """Test the __repr__ method of DetectedHold."""
        with test_app.app_context():
            # Create analysis first
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = db.session.query(HoldType).filter_by(name="crimp").first()
            assert hold_type is not None

            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=sample_detected_hold_data["confidence"],
                bbox_x1=sample_detected_hold_data["bbox_x1"],
                bbox_y1=sample_detected_hold_data["bbox_y1"],
                bbox_x2=sample_detected_hold_data["bbox_x2"],
                bbox_y2=sample_detected_hold_data["bbox_y2"],
            )
            repr_str = repr(detected_hold)

            assert "<DetectedHold" in repr_str
            assert str(analysis.id) in repr_str
            assert str(hold_type.id) in repr_str


class TestModelVersion:
    """Test cases for the ModelVersion model."""

    def test_create_model_version(self, test_app):
        """Test creating a ModelVersion instance."""
        with test_app.app_context():
            model_version = ModelVersion(
                model_type="hold_detection",
                version="1.0.0",
                model_path="/path/to/model.pt",
                accuracy=0.95,
            )

            db.session.add(model_version)
            db.session.commit()

            # Verify the model version was created
            retrieved = (
                db.session.query(ModelVersion).filter_by(id=model_version.id).first()
            )
            assert retrieved is not None
            assert retrieved.model_type == "hold_detection"
            assert retrieved.version == "1.0.0"
            assert retrieved.model_path == "/path/to/model.pt"
            assert retrieved.accuracy == 0.95
            assert retrieved.is_active is True

    def test_model_version_to_dict(self, test_app):
        """Test the to_dict method of ModelVersion."""
        with test_app.app_context():
            model_version = ModelVersion(
                model_type="route_grading",
                version="2.1.0",
                model_path="/path/to/grading_model.pt",
                accuracy=0.88,
            )

            db.session.add(model_version)
            db.session.flush()

            result = model_version.to_dict()

            assert result["id"] is not None
            assert result["model_type"] == "route_grading"
            assert result["version"] == "2.1.0"
            assert result["model_path"] == "/path/to/grading_model.pt"
            assert result["accuracy"] == 0.88
            assert result["is_active"] is True
            assert "created_at" in result

    def test_model_version_repr(self, test_app):
        """Test the __repr__ method of ModelVersion."""
        with test_app.app_context():
            model_version = ModelVersion(
                model_type="hold_detection",
                version="1.0.0",
                model_path="/path/to/model.pt",
            )
            repr_str = repr(model_version)

            assert "<ModelVersion" in repr_str
            assert "hold_detection" in repr_str
            assert "v1.0.0" in repr_str
            assert "Active True" in repr_str

    def test_model_version_unique_constraint(self, test_app):
        """Test the unique constraint on model_type and version."""
        with test_app.app_context():
            model_version = ModelVersion(
                model_type="hold_detection",
                version="1.0.0",
                model_path="/path/to/model.pt",
            )

            db.session.add(model_version)
            db.session.commit()

            # Try to create another with same type and version
            duplicate = ModelVersion(
                model_type="hold_detection",
                version="1.0.0",
                model_path="/path/to/another.pt",
            )

            db.session.add(duplicate)

            with pytest.raises(IntegrityError):  # Should raise integrity error
                db.session.flush()
            db.session.rollback()  # Clean up the failed transaction


class TestUserSession:
    """Test cases for the UserSession model."""

    def test_create_user_session(self, test_app):
        """Test creating a UserSession instance."""
        with test_app.app_context():
            session_id = str(uuid4())
            user_session = UserSession(
                session_id=session_id, ip_address="127.0.0.1", user_agent="Mozilla/5.0"
            )

            db.session.add(user_session)
            db.session.commit()

            # Verify the user session was created
            retrieved = (
                db.session.query(UserSession).filter_by(session_id=session_id).first()
            )
            assert retrieved is not None
            assert retrieved.session_id == session_id
            assert retrieved.ip_address == "127.0.0.1"
            assert retrieved.user_agent == "Mozilla/5.0"

    def test_user_session_to_dict(self, test_app):
        """Test the to_dict method of UserSession."""
        with test_app.app_context():
            session_id = str(uuid4())
            user_session = UserSession(
                session_id=session_id, ip_address="192.168.1.1", user_agent="Test Agent"
            )

            result = user_session.to_dict()

            assert result["id"] is not None
            assert result["session_id"] == session_id
            assert result["ip_address"] == "192.168.1.1"
            assert result["user_agent"] == "Test Agent"
            assert "created_at" in result
            assert "last_activity" in result

    def test_user_session_repr(self, test_app):
        """Test the __repr__ method of UserSession."""
        with test_app.app_context():
            session_id = str(uuid4())
            user_session = UserSession(session_id=session_id, ip_address="127.0.0.1")
            repr_str = repr(user_session)

            assert "<UserSession" in repr_str
            assert session_id in repr_str
