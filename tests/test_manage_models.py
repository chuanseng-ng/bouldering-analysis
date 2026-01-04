"""
Unit tests for src/manage_models.py - Model version management.

Tests cover:
- Model activation and deactivation
- Model listing and filtering
- Active model retrieval
- Error handling for non-existent models
- File validation for model activation
"""

from unittest.mock import patch, MagicMock
from src.manage_models import (
    activate_model,
    deactivate_model,
    list_models,
    get_active_model,
    get_models_data,
)
from src.models import db, ModelVersion


class TestActivateModel:
    """Test cases for activate_model function."""

    def test_activate_model_success(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test successful model activation."""
        with test_app.app_context():
            # Create another inactive model
            model2 = ModelVersion(
                model_type="hold_detection",
                version="v2.0",
                model_path=active_model_version.model_path,  # Use same path for testing
                accuracy=0.90,
                is_active=False,
            )
            db.session.add(model2)
            db.session.commit()
            model2_version = model2.version

        # Activate the second model
        success, message = activate_model("hold_detection", model2_version)

        assert success is True
        assert "Successfully activated" in message or "already active" in message

        # Verify in database
        with test_app.app_context():
            activated = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version=model2_version)
                .first()
            )
            assert activated is not None
            assert activated.is_active is True

    def test_activate_model_deactivates_others(
        self, test_app, active_model_version, inactive_model_version
    ):  # pylint: disable=unused-argument
        """Test that activating a model deactivates other models of the same type."""
        with test_app.app_context():
            # Verify initial state
            active = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version="v1.0")
                .first()
            )
            assert active is not None
            assert active.is_active is True

        # Activate the inactive model
        success, _ = activate_model("hold_detection", "v2.0")

        assert success is True

        # Verify v2.0 is now active and v1.0 is deactivated
        with test_app.app_context():
            v1 = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version="v1.0")
                .first()
            )
            v2 = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version="v2.0")
                .first()
            )

            assert v1 is not None
            assert v2 is not None
            assert v1.is_active is False
            assert v2.is_active is True

    def test_activate_model_not_found(self, test_app):  # pylint: disable=unused-argument
        """Test activating a non-existent model."""
        success, message = activate_model("hold_detection", "v99.0")

        assert success is False
        assert "Model not found" in message

    def test_activate_model_file_not_found(self, test_app, tmp_path):  # pylint: disable=unused-argument
        """Test activating a model when the file doesn't exist."""
        with test_app.app_context():
            # Create model with non-existent file
            nonexistent_path = tmp_path / "nonexistent.pt"
            model = ModelVersion(
                model_type="hold_detection",
                version="v3.0",
                model_path=str(nonexistent_path),
                accuracy=0.85,
                is_active=False,
            )
            db.session.add(model)
            db.session.commit()

        success, message = activate_model("hold_detection", "v3.0")

        assert success is False
        assert "Model file not found" in message

    def test_activate_already_active_model(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test activating a model that's already active."""
        success, message = activate_model("hold_detection", "v1.0")

        assert success is True
        assert "already active" in message

    @patch("src.manage_models._setup_flask_app")
    def test_activate_model_database_error(self, mock_setup):  # pylint: disable=unused-argument
        """Test error handling when database operation fails."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        # Mock app context to raise an exception
        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        success, message = activate_model("hold_detection", "v1.0")

        assert success is False
        assert "Error activating model" in message


class TestDeactivateModel:
    """Test cases for deactivate_model function."""

    def test_deactivate_model_success(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test successful model deactivation."""
        success, message = deactivate_model("hold_detection", "v1.0")

        assert success is True
        assert "Successfully deactivated" in message or "deactivated" in message

        # Verify in database
        with test_app.app_context():
            model = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version="v1.0")
                .first()
            )
            assert model is not None
            assert model.is_active is False

    def test_deactivate_inactive_model(self, test_app, inactive_model_version):  # pylint: disable=unused-argument
        """Test deactivating a model that's already inactive."""
        success, message = deactivate_model("hold_detection", "v2.0")

        assert success is True
        assert "already inactive" in message

    def test_deactivate_model_not_found(self, test_app):  # pylint: disable=unused-argument
        """Test deactivating a non-existent model."""
        success, message = deactivate_model("hold_detection", "v99.0")

        assert success is False
        assert "Model not found" in message

    @patch("src.manage_models._setup_flask_app")
    def test_deactivate_model_database_error(self, mock_setup):  # pylint: disable=unused-argument
        """Test error handling when database operation fails."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        success, message = deactivate_model("hold_detection", "v1.0")

        assert success is False
        assert "Error deactivating model" in message


class TestGetActiveModel:
    """Test cases for get_active_model function."""

    def test_get_active_model_exists(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test retrieving an active model."""
        model = get_active_model("hold_detection")

        assert model is not None
        assert model.model_type == "hold_detection"
        assert model.version == "v1.0"
        assert model.is_active is True

    def test_get_active_model_none(self, test_app, inactive_model_version):  # pylint: disable=unused-argument
        """Test retrieving active model when none exists."""
        # Deactivate the model first
        with test_app.app_context():
            model = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version="v2.0")
                .first()
            )
            assert model is not None
            model.is_active = False
            db.session.commit()

        result = get_active_model("hold_detection")

        assert result is None

    def test_get_active_model_different_type(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test retrieving active model for a different type."""
        model = get_active_model("route_grading")

        assert model is None

    @patch("src.manage_models._setup_flask_app")
    def test_get_active_model_database_error(self, mock_setup):  # pylint: disable=unused-argument
        """Test error handling when database query fails."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        result = get_active_model("hold_detection")

        assert result is None


class TestListModels:
    """Test cases for list_models function."""

    def test_list_models_all(
        self, test_app, active_model_version, inactive_model_version
    ):  # pylint: disable=unused-argument
        """Test listing all models."""
        result = list_models()

        assert isinstance(result, str)
        assert "v1.0" in result
        assert "v2.0" in result
        assert "hold_detection" in result
        assert "[ACTIVE]" in result

    def test_list_models_filtered_by_type(
        self, test_app, active_model_version, inactive_model_version
    ):  # pylint: disable=unused-argument
        """Test listing models filtered by type."""
        result = list_models(model_type="hold_detection")

        assert isinstance(result, str)
        assert "v1.0" in result
        assert "v2.0" in result
        assert "hold_detection" in result

    def test_list_models_no_models(self, test_app):  # pylint: disable=unused-argument
        """Test listing models when none exist."""
        result = list_models()

        assert isinstance(result, str)
        assert "No models found" in result

    def test_list_models_no_models_for_type(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test listing models for a type that doesn't exist."""
        result = list_models(model_type="route_grading")

        assert isinstance(result, str)
        assert "No models found" in result

    @patch("src.manage_models._setup_flask_app")
    def test_list_models_database_error(self, mock_setup):  # pylint: disable=unused-argument
        """Test error handling when database query fails."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        result = list_models()

        assert isinstance(result, str)
        assert "Error listing models" in result


class TestGetModelsData:
    """Test cases for get_models_data function."""

    def test_get_models_data_all(
        self, test_app, active_model_version, inactive_model_version
    ):  # pylint: disable=unused-argument
        """Test getting all models as data."""
        models = get_models_data()

        assert isinstance(models, list)
        assert len(models) == 2

        # Check structure
        for model in models:
            assert "id" in model
            assert "model_type" in model
            assert "version" in model
            assert "model_path" in model
            assert "accuracy" in model
            assert "is_active" in model
            assert "file_exists" in model

    def test_get_models_data_filtered(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test getting models filtered by type."""
        models = get_models_data(model_type="hold_detection")

        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(m["model_type"] == "hold_detection" for m in models)

    def test_get_models_data_empty(self, test_app):  # pylint: disable=unused-argument
        """Test getting models when none exist."""
        models = get_models_data()

        assert isinstance(models, list)
        assert len(models) == 0

    @patch("src.manage_models._setup_flask_app")
    def test_get_models_data_database_error(self, mock_setup):  # pylint: disable=unused-argument
        """Test error handling when database query fails."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        models = get_models_data()

        assert isinstance(models, list)
        assert len(models) == 0


class TestModelActivationValidation:
    """Test cases for model file validation during activation."""

    def test_activate_validates_model_file_exists(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test that activation validates model file existence."""
        # active_model_version has a real temp file, so this should succeed
        success, _ = activate_model("hold_detection", "v1.0")

        assert success is True

    def test_activate_rejects_missing_file(self, test_app, tmp_path):  # pylint: disable=unused-argument
        """Test that activation rejects models with missing files."""
        with test_app.app_context():
            # Create model with path that doesn't exist
            bad_path = tmp_path / "missing" / "model.pt"
            model = ModelVersion(
                model_type="hold_detection",
                version="v_bad",
                model_path=str(bad_path),
                accuracy=0.75,
                is_active=False,
            )
            db.session.add(model)
            db.session.commit()

        success, message = activate_model("hold_detection", "v_bad")

        assert success is False
        assert "Model file not found" in message


class TestModelTypeIsolation:  # pylint: disable=too-few-public-methods
    """Test that model activation is isolated by model_type."""

    def test_different_types_dont_interfere(
        self, test_app, active_model_version, tmp_path
    ):  # pylint: disable=unused-argument
        """Test that activating a model of one type doesn't affect other types."""
        with test_app.app_context():
            # Create a route_grading model
            model_file = tmp_path / "route_model.pt"
            model_file.write_text("mock route model")

            route_model = ModelVersion(
                model_type="route_grading",
                version="v1.0",
                model_path=str(model_file),
                accuracy=0.80,
                is_active=False,
            )
            db.session.add(route_model)
            db.session.commit()

        # Activate route grading model
        success, _ = activate_model("route_grading", "v1.0")
        assert success is True

        # Verify hold_detection model is still active
        with test_app.app_context():
            hold_model = (
                db.session.query(ModelVersion)
                .filter_by(model_type="hold_detection", version="v1.0")
                .first()
            )
            assert hold_model is not None
            assert hold_model.is_active is True


class TestSetupFlaskAppErrorPaths:
    """Test error paths in _setup_flask_app - covers lines 86-89."""

    @patch("flask.Flask")
    def test_setup_flask_app_import_error(self, mock_flask):
        """Test _setup_flask_app handles ImportError - covers lines 86-87."""
        # Make Flask raise ImportError
        mock_flask.side_effect = ImportError("Flask not installed")

        from src.manage_models import (  # pylint: disable=import-outside-toplevel
            _setup_flask_app,
        )  # pylint: disable=import-outside-toplevel

        try:
            _setup_flask_app()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Failed to import Flask dependencies" in str(e)

    @patch("src.models.db")
    def test_setup_flask_app_general_error(self, mock_db):
        """Test _setup_flask_app handles general Exception - covers lines 88-89."""
        # Make db.init_app raise a general exception
        mock_db.init_app.side_effect = Exception("Database initialization failed")

        from src.manage_models import (  # pylint: disable=import-outside-toplevel
            _setup_flask_app,
        )  # pylint: disable=import-outside-toplevel

        try:
            _setup_flask_app()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Failed to initialize Flask app" in str(e)


class TestActivateModelRollbackError:
    """Test rollback error handling in activate_model - covers lines 202-204."""

    def test_activate_model_rollback_on_error(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test that rollback is called on commit error - covers lines 202-204."""
        from sqlalchemy.exc import (  # pylint: disable=import-outside-toplevel
            SQLAlchemyError,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Patch commit to raise an error
            with patch("src.models.db.session.commit") as mock_commit:
                with patch("src.models.db.session.rollback") as mock_rollback:
                    mock_commit.side_effect = SQLAlchemyError("Commit failed")

                    success, message = activate_model("hold_detection", "v1.0")

                    assert success is False
                    assert "Error activating model" in message
                    # Verify rollback was called
                    mock_rollback.assert_called_once()

    @patch("src.models.db")
    @patch("src.manage_models._setup_flask_app")
    def test_activate_model_rollback_fails(self, mock_setup, mock_db):
        """Test when both activation and rollback fail - covers lines 205-208."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        # Make the main operation fail
        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Primary error"
        )

        # Make rollback also fail
        mock_db.session.rollback.side_effect = Exception("Rollback failed")

        success, message = activate_model("hold_detection", "v1.0")

        assert success is False
        assert "Error activating model" in message


class TestDeactivateModelRollbackError:
    """Test rollback error handling in deactivate_model - covers lines 282-288."""

    def test_deactivate_model_rollback_on_error(self, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test that rollback is called on commit error - covers lines 282-284."""
        from sqlalchemy.exc import (  # pylint: disable=import-outside-toplevel
            SQLAlchemyError,
        )  # pylint: disable=import-outside-toplevel

        with test_app.app_context():
            # Patch commit to raise an error
            with patch("src.models.db.session.commit") as mock_commit:
                with patch("src.models.db.session.rollback") as mock_rollback:
                    mock_commit.side_effect = SQLAlchemyError("Commit failed")

                    success, message = deactivate_model("hold_detection", "v1.0")

                    assert success is False
                    assert "Error deactivating model" in message
                    # Verify rollback was called
                    mock_rollback.assert_called_once()

    @patch("src.models.db")
    @patch("src.manage_models._setup_flask_app")
    def test_deactivate_model_rollback_fails(self, mock_setup, mock_db):
        """Test when both deactivation and rollback fail - covers lines 285-288."""
        mock_app = MagicMock()
        mock_setup.return_value = mock_app

        # Make the main operation fail
        mock_app.app_context.return_value.__enter__.side_effect = Exception(
            "Primary error"
        )

        # Make rollback also fail
        mock_db.session.rollback.side_effect = Exception("Rollback failed")

        success, message = deactivate_model("hold_detection", "v1.0")

        assert success is False
        assert "Error deactivating model" in message


class TestListModelsMultipleTypes:  # pylint: disable=too-few-public-methods
    """Test list_models with multiple model types - covers line 390."""

    def test_list_models_multiple_types_separator(self, test_app, tmp_path):
        """Test that separator is shown between different model types - covers line 390."""
        with test_app.app_context():
            # Create models of different types
            hold_file = tmp_path / "hold.pt"
            hold_file.write_text("hold model")

            route_file = tmp_path / "route.pt"
            route_file.write_text("route model")

            hold_model = ModelVersion(
                model_type="hold_detection",
                version="v1.0",
                model_path=str(hold_file),
                accuracy=0.85,
                is_active=True,
            )
            db.session.add(hold_model)

            route_model = ModelVersion(
                model_type="route_grading",
                version="v1.0",
                model_path=str(route_file),
                accuracy=0.80,
                is_active=True,
            )
            db.session.add(route_model)
            db.session.commit()

        result = list_models()

        # Should contain both model types and separator
        assert "hold_detection" in result
        assert "route_grading" in result
        # Check for separator between types (dashes)
        assert "---" in result


class TestCLIMain:
    """Test the main CLI function - covers lines 500-576."""

    @patch("sys.argv", ["manage_models.py", "list"])
    @patch("sys.exit")
    def test_main_list_command(self, mock_exit, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test main() with list command - covers lines 559-562."""
        from src.manage_models import main  # pylint: disable=import-outside-toplevel

        # Capture stdout
        from io import StringIO  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            main()
        finally:
            sys.stdout = sys.__stdout__

        # Should exit with 0
        mock_exit.assert_called_with(0)

        # Output should contain model info
        output = captured_output.getvalue()
        assert "v1.0" in output or "MODEL" in output

    @patch(
        "sys.argv",
        [
            "manage_models.py",
            "activate",
            "--model-type",
            "hold_detection",
            "--version",
            "v1.0",
        ],
    )
    @patch("sys.exit")
    def test_main_activate_command_success(
        self, mock_exit, test_app, active_model_version
    ):  # pylint: disable=unused-argument
        """Test main() with activate command - covers lines 564-567."""
        from src.manage_models import main  # pylint: disable=import-outside-toplevel

        main()

        # Should exit with 0 for success
        mock_exit.assert_called_with(0)

    @patch(
        "sys.argv",
        [
            "manage_models.py",
            "activate",
            "--model-type",
            "hold_detection",
            "--version",
            "v99.0",
        ],
    )
    @patch("sys.exit")
    def test_main_activate_command_failure(self, mock_exit, test_app):  # pylint: disable=unused-argument
        """Test main() with activate command that fails - covers lines 564-567."""
        from src.manage_models import main  # pylint: disable=import-outside-toplevel

        main()

        # Should exit with 1 for failure
        mock_exit.assert_called_with(1)

    @patch(
        "sys.argv",
        [
            "manage_models.py",
            "deactivate",
            "--model-type",
            "hold_detection",
            "--version",
            "v1.0",
        ],
    )
    @patch("sys.exit")
    def test_main_deactivate_command(self, mock_exit, test_app, active_model_version):  # pylint: disable=unused-argument
        """Test main() with deactivate command - covers lines 569-572."""
        from src.manage_models import main  # pylint: disable=import-outside-toplevel

        main()

        # Should exit with 0 for success
        mock_exit.assert_called_with(0)

    @patch("sys.argv", ["manage_models.py", "list", "--model-type", "hold_detection"])
    @patch("sys.exit")
    def test_main_list_filtered_command(
        self, mock_exit, test_app, active_model_version
    ):  # pylint: disable=unused-argument
        """Test main() with list command and filter - covers lines 559-562."""
        from src.manage_models import main  # pylint: disable=import-outside-toplevel

        # Capture stdout
        from io import StringIO  # pylint: disable=import-outside-toplevel
        import sys  # pylint: disable=import-outside-toplevel

        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            main()
        finally:
            sys.stdout = sys.__stdout__

        mock_exit.assert_called_with(0)

        output = captured_output.getvalue()
        assert "hold_detection" in output or "MODEL" in output
