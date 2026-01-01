"""
Pytest configuration and fixtures for bouldering analysis tests.
"""

import os
import tempfile
import pytest
from PIL import Image
from src.main import app as flask_app
from src.models import db, HoldType


@pytest.fixture
def test_app():
    """Create and configure a test Flask application instance."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()

    # Configure the app for testing
    flask_app.config["TESTING"] = True
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    flask_app.config["SERVER_NAME"] = "localhost.localdomain"

    # Create application context
    with flask_app.app_context():
        # Create all database tables
        db.create_all()

        # Initialize hold types
        hold_type_data = [
            (0, "crimp", "Small, narrow hold requiring crimping fingers"),
            (1, "jug", "Large, easy-to-hold jug"),
            (2, "sloper", "Round, sloping hold that requires open-handed grip"),
            (3, "pinch", "Hold that requires pinching between thumb and fingers"),
            (4, "pocket", "Small hole that fingers fit into"),
            (5, "foot-hold", "Hold specifically for feet"),
            (6, "start-hold", "Starting hold for the route"),
            (7, "top-out-hold", "Hold used to complete the route"),
        ]

        for hold_id, name, description in hold_type_data:
            hold_type = HoldType(id=hold_id, name=name, description=description)
            db.session.add(hold_type)

        db.session.commit()

    yield flask_app

    # Cleanup
    with flask_app.app_context():
        db.session.remove()
        db.drop_all()

    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def test_client(test_app):
    """Create a test client for the Flask application."""
    return test_app.test_client()


@pytest.fixture
def sample_analysis_data():
    """Provide sample data for creating an Analysis instance."""
    return {
        "image_filename": "test.jpg",
        "image_path": "/path/to/test.jpg",
        "predicted_grade": "V2",
        "confidence_score": 0.85,
        "features_extracted": {
            "total_holds": 5,
            "hold_types": {"crimp": 2, "jug": 3},
            "average_confidence": 0.85,
        },
    }


@pytest.fixture
def sample_feedback_data():
    """Provide sample data for creating a Feedback instance."""
    return {
        "user_grade": "V3",
        "is_accurate": False,
        "comments": "The grade seems a bit high",
    }


@pytest.fixture
def sample_detected_hold_data():
    """Provide sample data for creating a DetectedHold instance."""
    return {
        "confidence": 0.9,
        "bbox_x1": 10.0,
        "bbox_y1": 10.0,
        "bbox_x2": 50.0,
        "bbox_y2": 50.0,
    }


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image and return its path."""
    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return str(img_path)
