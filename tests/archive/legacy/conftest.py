"""
Pytest configuration and fixtures for bouldering analysis tests.
"""

import os
import tempfile
import pytest
import yaml
from PIL import Image
from src.main import app as flask_app, clear_hold_types_cache
from src.models import db, HoldType, ModelVersion
from src.constants import HOLD_TYPES


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear hold types cache before and after each test."""
    clear_hold_types_cache()
    yield
    clear_hold_types_cache()


@pytest.fixture
def test_app():
    """Create and configure a test Flask application instance."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()
    os.close(db_fd)  # Close immediately, SQLAlchemy will open it

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
        # Use shared HOLD_TYPES constant
        hold_type_data = HOLD_TYPES

        # Only add hold types if they don't already exist
        existing_types = {ht.id for ht in db.session.query(HoldType).all()}

        for hold_id, name, description in hold_type_data:
            if hold_id not in existing_types:
                # Use merge to avoid integrity errors
                hold_type = HoldType()
                hold_type.id = hold_id
                hold_type.name = name
                hold_type.description = description
                db.session.merge(hold_type)

        db.session.commit()

    try:
        yield flask_app
    finally:
        # Cleanup
        with flask_app.app_context():
            db.session.remove()
            db.drop_all()

        os.unlink(db_path)


@pytest.fixture
def test_client(test_app):  # pylint: disable=redefined-outer-name
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
        "wall_incline": "vertical",
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


@pytest.fixture
def sample_model_version_data():
    """Provide sample data for creating ModelVersion instances."""
    return {
        "model_type": "hold_detection",
        "version": "v1.0",
        "model_path": "models/hold_detection/v1.0.pt",
        "accuracy": 0.85,
        "is_active": False,
    }


@pytest.fixture
def active_model_version(test_app, sample_model_version_data, tmp_path):  # pylint: disable=redefined-outer-name,unused-argument
    """Create an active ModelVersion entry in the test database with a mock model file."""
    with test_app.app_context():
        # Create a temporary model file
        model_dir = tmp_path / "models" / "hold_detection"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "v1.0.pt"
        model_file.write_text("mock model data")

        # Create ModelVersion with absolute path to the temp file
        model_version = ModelVersion(
            model_type="hold_detection",
            version="v1.0",
            model_path=str(model_file),
            accuracy=0.85,
            is_active=True,
        )
        db.session.add(model_version)
        db.session.commit()

        yield model_version

        # Cleanup is handled by test_app fixture


@pytest.fixture
def inactive_model_version(test_app, tmp_path):  # pylint: disable=redefined-outer-name
    """Create an inactive ModelVersion entry in the test database."""
    with test_app.app_context():
        # Create a temporary model file
        model_dir = tmp_path / "models" / "hold_detection"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "v2.0.pt"
        model_file.write_text("mock model data v2")

        model_version = ModelVersion(
            model_type="hold_detection",
            version="v2.0",
            model_path=str(model_file),
            accuracy=0.88,
            is_active=False,
        )
        db.session.add(model_version)
        db.session.commit()

        yield model_version

        # Cleanup is handled by test_app fixture


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary model file for testing."""
    model_file = tmp_path / "test_model.pt"
    model_file.write_text("mock yolo model weights")
    return model_file


@pytest.fixture
def test_config_yaml(tmp_path):
    """Create a temporary configuration YAML file for testing."""
    config = {
        "model_defaults": {
            "hold_detection_confidence_threshold": 0.25,
        },
        "model_paths": {
            "base_yolov8": "yolov8n.pt",
            "fine_tuned_models": "models/hold_detection/",
        },
        "data_paths": {
            "hold_dataset": "data/sample_hold/",
            "uploads": "data/uploads/",
        },
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return config_file


@pytest.fixture
def invalid_config_yaml(tmp_path):
    """Create an invalid configuration YAML file for testing."""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("{ invalid yaml content: [")
    return config_file


@pytest.fixture
def empty_config_yaml(tmp_path):
    """Create an empty configuration YAML file for testing."""
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("")
    return config_file


@pytest.fixture
def sample_yolo_dataset(tmp_path):
    """Create a sample YOLO dataset structure for testing."""
    dataset_dir = tmp_path / "sample_dataset"

    # Create directory structure
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    val_images = dataset_dir / "val" / "images"
    val_labels = dataset_dir / "val" / "labels"

    for directory in [train_images, train_labels, val_images, val_labels]:
        directory.mkdir(parents=True, exist_ok=True)

    # Create sample images
    for i in range(3):
        img = Image.new("RGB", (640, 640), color="blue")
        img.save(train_images / f"image_{i}.jpg")

        # Create corresponding label file
        with open(train_labels / f"image_{i}.txt", "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

    # Create validation images
    for i in range(2):
        img = Image.new("RGB", (640, 640), color="green")
        img.save(val_images / f"val_{i}.jpg")

        with open(val_labels / f"val_{i}.txt", "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")

    # Create data.yaml
    data_yaml = dataset_dir / "data.yaml"
    dataset_config = {
        "train": "train/images",
        "val": "val/images",
        "nc": 2,
        "names": ["crimp", "jug"],
    }

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f)

    return dataset_dir


@pytest.fixture
def create_analysis_with_hold_type(test_app, sample_analysis_data):  # pylint: disable=redefined-outer-name
    """Helper fixture to create an analysis and return hold type."""

    def _create(hold_type_name="crimp"):
        """Create analysis and return it with the specified hold type."""
        from src.models import Analysis  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Create analysis
            analysis = Analysis(**sample_analysis_data)
            db.session.add(analysis)
            db.session.flush()

            # Get hold type
            hold_type = (
                db.session.query(HoldType).filter_by(name=hold_type_name).first()
            )
            assert hold_type is not None

            return analysis, hold_type

    return _create


@pytest.fixture
def create_detected_hold_for_analysis(test_app):  # pylint: disable=redefined-outer-name
    """Helper fixture to create a DetectedHold for an analysis."""

    def _create(analysis, hold_type, **kwargs):
        """Create and return a DetectedHold with the given parameters."""
        from src.models import DetectedHold  # pylint: disable=import-outside-toplevel

        # Default values for bounding box parameters
        defaults = {
            "confidence": 0.9,
            "bbox_x1": 10.0,
            "bbox_y1": 10.0,
            "bbox_x2": 50.0,
            "bbox_y2": 50.0,
        }
        # Merge defaults with provided kwargs
        params = {**defaults, **kwargs}

        with test_app.app_context():
            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=params["confidence"],
                bbox_x1=params["bbox_x1"],
                bbox_y1=params["bbox_y1"],
                bbox_x2=params["bbox_x2"],
                bbox_y2=params["bbox_y2"],
            )
            db.session.add(detected_hold)
            db.session.commit()

            return detected_hold

    return _create
