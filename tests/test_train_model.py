# pylint: disable=duplicate-code
"""
Unit tests for src/train_model.py - Training pipeline for YOLOv8 models.

Tests cover:
- Dataset validation
- Training metadata saving
- Model version creation
- CLI argument parsing
- Error handling for invalid parameters
"""

from unittest.mock import patch, MagicMock

import pytest

from src.train_model import (
    validate_dataset,
    validate_base_weights,
    setup_training_directories,
    save_model_version,
    create_flask_app,
    TrainingError,
    parse_arguments,
)
from src.models import db, ModelVersion


class TestValidateDataset:
    """Test cases for validate_dataset function."""

    def test_validate_dataset_success(self, sample_yolo_dataset):
        """Test successful dataset validation."""
        data_yaml = sample_yolo_dataset / "data.yaml"

        config = validate_dataset(data_yaml)

        assert isinstance(config, dict)
        assert config["nc"] == 2
        assert config["names"] == ["crimp", "jug"]
        assert "train" in config
        assert "val" in config

    def test_validate_dataset_file_not_found(self, tmp_path):
        """Test validation fails when data.yaml doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(TrainingError, match="Dataset configuration file not found"):
            validate_dataset(nonexistent_file)

    def test_validate_dataset_invalid_yaml(self, tmp_path):
        """Test validation fails with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("{ invalid yaml [")

        with pytest.raises(TrainingError, match="Error parsing data.yaml"):
            validate_dataset(invalid_yaml)

    def test_validate_dataset_missing_required_keys(self, tmp_path):
        """Test validation fails when required keys are missing."""
        import yaml  # pylint: disable=import-outside-toplevel

        # Create config with missing keys
        incomplete_config = tmp_path / "incomplete.yaml"
        with open(incomplete_config, "w", encoding="utf-8") as f:
            yaml.dump({"train": "train/", "val": "val/"}, f)

        with pytest.raises(TrainingError, match="Missing required keys"):
            validate_dataset(incomplete_config)

    def test_validate_dataset_missing_train_directory(self, tmp_path):
        """Test validation fails when train directory doesn't exist."""
        import yaml  # pylint: disable=import-outside-toplevel

        data_yaml = tmp_path / "data.yaml"
        config = {
            "train": "train/",
            "val": "val/",
            "nc": 1,
            "names": ["hold"],
        }

        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        with pytest.raises(TrainingError, match="train directory not found"):
            validate_dataset(data_yaml)

    def test_validate_dataset_missing_images(self, tmp_path):
        """Test validation fails when no images are found."""
        import yaml  # pylint: disable=import-outside-toplevel

        # Create structure but no images
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train" / "images"
        train_dir.mkdir(parents=True)

        data_yaml = dataset_dir / "data.yaml"
        config = {
            "train": "train",
            "val": "train",  # Use same for simplicity
            "nc": 1,
            "names": ["hold"],
        }

        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        with pytest.raises(TrainingError, match="No image files found"):
            validate_dataset(data_yaml)

    def test_validate_dataset_class_count_mismatch(
        self, tmp_path, sample_yolo_dataset
    ):  # pylint: disable=unused-argument
        """Test validation fails when class count doesn't match names."""
        import yaml  # pylint: disable=import-outside-toplevel

        # Modify the data.yaml to have mismatched nc and names
        data_yaml = sample_yolo_dataset / "data.yaml"
        config = {
            "train": "train/images",
            "val": "val/images",
            "nc": 5,  # Mismatched with names
            "names": ["crimp", "jug"],
        }

        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        with pytest.raises(TrainingError, match="Mismatch between nc"):
            validate_dataset(data_yaml)


class TestValidateBaseWeights:
    """Test cases for validate_base_weights function."""

    def test_validate_base_weights_success(self, temp_model_file):
        """Test successful base weights validation."""
        # Should not raise an error
        validate_base_weights(temp_model_file)

    def test_validate_base_weights_file_not_found(self, tmp_path):
        """Test validation fails when weights file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.pt"

        with pytest.raises(TrainingError, match="Base weights file not found"):
            validate_base_weights(nonexistent)

    def test_validate_base_weights_wrong_extension(self, tmp_path):
        """Test validation fails with wrong file extension."""
        wrong_ext = tmp_path / "model.txt"
        wrong_ext.write_text("not a pt file")

        with pytest.raises(TrainingError, match="Invalid weights file format"):
            validate_base_weights(wrong_ext)


class TestSetupTrainingDirectories:  # pylint: disable=too-few-public-methods
    """Test cases for setup_training_directories function."""

    @patch("src.train_model.get_project_root")
    def test_setup_training_directories(self, mock_root, tmp_path):
        """Test creation of training directories."""
        mock_root.return_value = tmp_path

        dirs = setup_training_directories("test_model")

        assert "models" in dirs
        assert "logs" in dirs
        assert "runs" in dirs

        # Verify directories were created
        assert dirs["models"].exists()
        assert dirs["logs"].exists()
        assert dirs["runs"].exists()


class TestSaveModelVersion:
    """Test cases for save_model_version function."""

    @patch("src.train_model.create_flask_app")
    @patch("src.train_model.get_project_root")
    def test_save_model_version_new(
        self, mock_root, mock_create_app, tmp_path, test_app
    ):
        """Test saving a new model version."""
        mock_root.return_value = tmp_path
        mock_create_app.return_value = test_app

        # Create a temp model file
        trained_model = tmp_path / "trained.pt"
        trained_model.write_text("trained model weights")

        metrics = {
            "final_mAP50": 0.85,
            "final_mAP50-95": 0.75,
            "final_precision": 0.80,
            "final_recall": 0.78,
        }

        training_config = {
            "epochs": 100,
            "batch_size": 16,
            "img_size": 640,
            "learning_rate": 0.01,
        }

        with test_app.app_context():
            model_version = save_model_version(
                model_name="test_v1",
                trained_model_path=trained_model,
                metrics=metrics,
                training_config=training_config,
                activate=False,
            )

            assert model_version is not None
            assert model_version.version == "test_v1"
            assert model_version.model_type == "hold_detection"
            assert model_version.accuracy == 0.75  # Uses mAP50-95
            assert model_version.is_active is False

    @patch("src.train_model.create_flask_app")
    @patch("src.train_model.get_project_root")
    def test_save_model_version_with_activation(
        self, mock_root, mock_create_app, tmp_path, test_app
    ):
        """Test saving and immediately activating a model version."""
        mock_root.return_value = tmp_path
        mock_create_app.return_value = test_app

        # Create existing active model first
        with test_app.app_context():
            existing = ModelVersion(
                model_type="hold_detection",
                version="old_v1",
                model_path="old/path.pt",
                accuracy=0.70,
                is_active=True,
            )
            db.session.add(existing)
            db.session.commit()

        # Create a temp model file
        trained_model = tmp_path / "trained.pt"
        trained_model.write_text("trained model weights")

        metrics = {"final_mAP50-95": 0.85}
        training_config = {"epochs": 100}

        with test_app.app_context():
            model_version = save_model_version(
                model_name="new_v1",
                trained_model_path=trained_model,
                metrics=metrics,
                training_config=training_config,
                activate=True,
            )

            # Check new model is active
            assert model_version.is_active is True

            # Check old model is deactivated
            old_model = (
                db.session.query(ModelVersion).filter_by(version="old_v1").first()
            )
            assert old_model is not None
            assert old_model.is_active is False

    @patch("src.train_model.create_flask_app")
    @patch("src.train_model.get_project_root")
    def test_save_model_version_updates_existing(
        self, mock_root, mock_create_app, tmp_path, test_app
    ):
        """Test updating an existing model version."""
        mock_root.return_value = tmp_path
        mock_create_app.return_value = test_app

        # Create existing model
        with test_app.app_context():
            existing = ModelVersion(
                model_type="hold_detection",
                version="test_v1",
                model_path="old/path.pt",
                accuracy=0.70,
                is_active=False,
            )
            db.session.add(existing)
            db.session.commit()

        # Create a temp model file
        trained_model = tmp_path / "trained.pt"
        trained_model.write_text("new trained model weights")

        metrics = {"final_mAP50-95": 0.85}
        training_config = {"epochs": 100}

        with test_app.app_context():
            model_version = save_model_version(
                model_name="test_v1",
                trained_model_path=trained_model,
                metrics=metrics,
                training_config=training_config,
                activate=False,
            )

            # Should update the existing model
            assert model_version.accuracy == 0.85

            # Should only have one model with this version
            count = db.session.query(ModelVersion).filter_by(version="test_v1").count()
            assert count == 1

    @patch("src.train_model.create_flask_app")
    def test_save_model_version_error_handling(self, mock_create_app, tmp_path):
        """Test error handling when saving fails."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        # Mock app context to raise an exception
        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Save error"
        )

        trained_model = tmp_path / "trained.pt"
        trained_model.write_text("trained model")

        with pytest.raises(TrainingError, match="Failed to save model version"):
            save_model_version(
                model_name="test",
                trained_model_path=trained_model,
                metrics={},
                training_config={},
            )


class TestCreateFlaskApp:  # pylint: disable=too-few-public-methods
    """Test cases for create_flask_app function."""

    @patch("src.train_model.get_project_root")
    def test_create_flask_app(self, mock_root, tmp_path):
        """Test Flask app creation for database operations."""
        mock_root.return_value = tmp_path

        app = create_flask_app()

        assert app is not None
        assert "SQLALCHEMY_DATABASE_URI" in app.config
        assert app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] is False


class TestTrainYOLOv8Mock:
    """Test cases for train_yolov8 function with mocked YOLO."""

    @patch("src.train_model.YOLO")
    @patch("src.train_model.get_project_root")
    def test_train_yolov8_parameters(
        self, mock_root, mock_yolo_class, tmp_path, sample_yolo_dataset
    ):
        """Test that training parameters are passed correctly to YOLO."""
        mock_root.return_value = tmp_path

        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock training results
        mock_results = MagicMock()
        mock_results.results_dict = {
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.75,
            "metrics/precision(B)": 0.80,
            "metrics/recall(B)": 0.78,
        }
        mock_results.best_epoch = 50
        mock_results.save_dir = tmp_path / "runs"
        mock_model.train.return_value = mock_results

        # pylint: disable=import-outside-toplevel
        from src.train_model import train_yolov8

        data_yaml = sample_yolo_dataset / "data.yaml"
        base_weights = tmp_path / "base.pt"
        base_weights.write_text("base weights")

        result = train_yolov8(
            model_name="test_model",
            data_yaml=data_yaml,
            base_weights=base_weights,
            epochs=100,
            batch_size=16,
            img_size=640,
            learning_rate=0.01,
        )

        # Verify YOLO was called with correct parameters
        mock_yolo_class.assert_called_once()
        mock_model.train.assert_called_once()

        # Check training args
        train_call_args = mock_model.train.call_args
        assert train_call_args[1]["epochs"] == 100
        assert train_call_args[1]["batch"] == 16
        assert train_call_args[1]["imgsz"] == 640
        assert train_call_args[1]["lr0"] == 0.01

        # Verify return structure
        assert "results" in result
        assert "metrics" in result
        assert "best_model_path" in result

    @patch("src.train_model.YOLO")
    def test_train_yolov8_error_handling(
        self, mock_yolo_class, tmp_path, sample_yolo_dataset
    ):
        """Test error handling when training fails."""
        # Mock YOLO to raise an exception
        mock_yolo_class.side_effect = RuntimeError("Training failed")

        # pylint: disable=import-outside-toplevel
        from src.train_model import train_yolov8

        data_yaml = sample_yolo_dataset / "data.yaml"
        base_weights = tmp_path / "base.pt"
        base_weights.write_text("base weights")

        with pytest.raises(TrainingError, match="Training failed"):
            train_yolov8(
                model_name="test_model",
                data_yaml=data_yaml,
                base_weights=base_weights,
            )


class TestParseArguments:
    """Test cases for command-line argument parsing."""

    def test_parse_arguments_minimal(self):
        """Test parsing minimal required arguments."""
        with patch("sys.argv", ["train_model.py", "--model-name", "v1.0"]):
            args = parse_arguments()

            assert args.model_name == "v1.0"
            assert args.epochs is None  # Should use default
            assert args.batch_size is None
            assert args.activate is False

    def test_parse_arguments_full(self):
        """Test parsing all arguments."""
        with patch(
            "sys.argv",
            [
                "train_model.py",
                "--model-name",
                "v2.0",
                "--epochs",
                "150",
                "--batch-size",
                "32",
                "--img-size",
                "800",
                "--learning-rate",
                "0.001",
                "--data-yaml",
                "data/custom/data.yaml",
                "--base-weights",
                "models/custom.pt",
                "--activate",
            ],
        ):
            args = parse_arguments()

            assert args.model_name == "v2.0"
            assert args.epochs == 150
            assert args.batch_size == 32
            assert args.img_size == 800
            assert args.learning_rate == 0.001
            assert args.data_yaml == "data/custom/data.yaml"
            assert args.base_weights == "models/custom.pt"
            assert args.activate is True

    def test_parse_arguments_missing_required(self):
        """Test that missing required arguments raises error."""
        with patch("sys.argv", ["train_model.py"]):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestTrainingPipelineIntegration:  # pylint: disable=too-few-public-methods
    """Integration tests for the complete training pipeline."""

    @patch("src.train_model.YOLO")
    @patch("src.train_model.get_project_root")
    @patch("src.train_model.create_flask_app")
    def test_main_pipeline_success(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mock_create_app,
        mock_root,
        mock_yolo_class,
        tmp_path,
        sample_yolo_dataset,
        test_app,
    ):
        """Test the main training pipeline executes successfully."""
        mock_root.return_value = tmp_path
        mock_create_app.return_value = test_app

        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock training results
        mock_results = MagicMock()
        mock_results.results_dict = {
            "metrics/mAP50(B)": 0.85,
            "metrics/mAP50-95(B)": 0.75,
            "metrics/precision(B)": 0.80,
            "metrics/recall(B)": 0.78,
        }
        mock_results.best_epoch = 50
        mock_results.save_dir = tmp_path / "runs"
        (tmp_path / "runs" / "weights").mkdir(parents=True)
        best_pt = tmp_path / "runs" / "weights" / "best.pt"
        best_pt.write_text("best model")
        mock_model.train.return_value = mock_results

        from src.train_model import main  # pylint: disable=import-outside-toplevel

        data_yaml = str(sample_yolo_dataset / "data.yaml")
        base_weights = tmp_path / "base.pt"
        base_weights.write_text("base weights")

        # Should not raise an exception
        main(
            model_name="pipeline_test",
            epochs=10,
            batch_size=8,
            data_yaml=data_yaml,
            base_weights=str(base_weights),
            activate=False,
        )

        # Verify model was saved to database
        with test_app.app_context():
            saved_model = (
                db.session.query(ModelVersion)
                .filter_by(version="pipeline_test")
                .first()
            )
            assert saved_model is not None
            assert saved_model.model_type == "hold_detection"
