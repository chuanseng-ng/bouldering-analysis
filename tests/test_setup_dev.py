"""
Unit tests for setup_dev.py
"""

import sys
import os
import subprocess
from unittest.mock import Mock, patch

# Add the project root to the path so we can import setup_dev
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pylint: disable=wrong-import-position # noqa: E402
from src.setup_dev import run_command, verify_installation, main


class TestRunCommand:
    """Test cases for the run_command function."""

    @patch("src.setup_dev.subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value = Mock(stdout="Success", stderr="")

        result = run_command(["echo", "test"], "Test command")

        assert result is True
        mock_run.assert_called_once_with(
            ["echo", "test"], shell=False, check=True, capture_output=True, text=True
        )

    @patch("src.setup_dev.subprocess.run")
    @patch("src.setup_dev.print")
    def test_run_command_failure(self, mock_print, mock_run):
        """Test failed command execution."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="test", stderr="Error message"
        )

        result = run_command(["failing_command"], "Failing command")

        assert result is False
        mock_print.assert_any_call("✗ Failing command failed")
        mock_print.assert_any_call("Error: Error message")

    @patch("src.setup_dev.subprocess.run")
    @patch("src.setup_dev.print")
    def test_run_command_with_stdout(self, mock_print, mock_run):
        """Test command execution with stdout output."""
        mock_run.return_value = Mock(stdout="Command output", stderr="")

        result = run_command(["echo", "test"], "Test with output")

        assert result is True
        mock_print.assert_any_call("Command output")


class TestVerifyInstallation:
    """Test cases for the verify_installation function."""

    @patch("ultralytics.YOLO")
    @patch("sqlalchemy.text")
    @patch("src.models.db")
    @patch("src.main.app")
    @patch("src.setup_dev.print")
    @patch("src.setup_dev.subprocess.run")
    # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    def test_verify_installation_success(
        self, mock_run, mock_print, mock_app, mock_db, mock_text, mock_yolo
    ):
        """Test successful installation verification."""
        # Mock the YOLO model loading
        mock_yolo.return_value = Mock()

        # Mock app context
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()

        # Mock db session execute
        mock_db.session.execute = Mock()

        result = verify_installation()

        assert result is True
        mock_print.assert_any_call("✓ All imports successful")
        mock_print.assert_any_call("✓ Database connection successful")
        mock_print.assert_any_call("✓ YOLO model loaded successfully")

    @patch("src.setup_dev.print")
    @patch("src.setup_dev.subprocess.run")
    # pylint: disable=unused-argument
    def test_verify_installation_import_error(self, mock_run, mock_print):
        """Test installation verification with import error.

        Simulate an import-time failure for `src.main` by inserting a
        sentinel into `sys.modules` so that importing `src.main` raises
        ImportError during `verify_installation()`.
        """
        with patch.dict(sys.modules, {"src.main": None}):
            result = verify_installation()

        assert result is False
        # Check that verification failed message is printed
        assert any("✗ Verification failed" in str(c) for c in mock_print.call_args_list)

    @patch("ultralytics.YOLO", side_effect=ImportError("Model not found"))
    @patch("sqlalchemy.text")
    @patch("src.models.db")
    @patch("src.main.app")
    @patch("src.setup_dev.print")
    @patch("src.setup_dev.subprocess.run")
    # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    def test_verify_installation_model_loading_error(
        self, mock_run, mock_print, mock_app, mock_db, mock_text, mock_yolo
    ):
        """Test installation verification with YOLO model loading error."""
        # Mock app context
        mock_app.app_context.return_value.__enter__ = Mock()
        mock_app.app_context.return_value.__exit__ = Mock()

        # Mock db session execute
        mock_db.session.execute = Mock()

        result = verify_installation()

        assert result is True  # Should still pass with warning
        mock_print.assert_any_call("⚠ YOLO model loading failed: Model not found")


class TestMain:
    """Test cases for the main function."""

    @patch("src.setup_dev.Path")
    @patch("src.setup_dev.print")
    def test_main_wrong_directory(self, mock_print, mock_path):
        """Test main function when run from wrong directory."""
        mock_path.return_value.exists.return_value = False

        result = main()

        assert result is False
        mock_print.assert_any_call(
            "✗ Please run this script from the project root directory"
        )

    @patch("src.setup_dev.Path")
    @patch("src.setup_dev.print")
    @patch("src.setup_dev.create_directories")
    @patch("src.setup_dev.setup_database")
    @patch("src.setup_dev.verify_installation")
    # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    def test_main_directory_check(
        self, mock_verify, mock_setup, mock_create, mock_print, mock_path
    ):
        """Test main function directory check."""
        # Mock src directory exists
        mock_path.return_value.exists.return_value = True
        mock_create.return_value = True
        mock_setup.return_value = True
        mock_verify.return_value = True

        result = main()

        assert result is True
        mock_create.assert_called_once()
        mock_setup.assert_called_once()
        mock_verify.assert_called_once()

    @patch("src.setup_dev.Path")
    @patch("src.setup_dev.print")
    @patch("src.setup_dev.create_directories")
    @patch("src.setup_dev.setup_database")
    @patch("src.setup_dev.verify_installation")
    # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    def test_main_step_failure(
        self, mock_verify, mock_setup, mock_create, mock_print, mock_path
    ):
        """Test main function when a step fails."""
        mock_path.return_value.exists.return_value = True
        mock_create.return_value = False

        result = main()

        assert result is False
        mock_print.assert_any_call("\n✗ Setup failed at: Creating directories")

    @patch("src.setup_dev.Path")
    @patch("src.setup_dev.print")
    @patch("src.setup_dev.create_directories")
    @patch("src.setup_dev.setup_database")
    @patch("src.setup_dev.verify_installation")
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_main_success(
        self, mock_verify, mock_setup, mock_create, mock_print, mock_path
    ):
        """Test main function successful execution."""
        mock_path.return_value.exists.return_value = True
        mock_create.return_value = True
        mock_setup.return_value = True
        mock_verify.return_value = True

        result = main()

        assert result is True
        mock_print.assert_any_call(
            "✓ Development environment setup completed successfully!"
        )

    @patch("src.setup_dev.Path")
    @patch("src.setup_dev.create_directories")
    @patch("src.setup_dev.setup_database")
    @patch("src.setup_dev.verify_installation")
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def test_main_returns_true_on_success(
        self, mock_verify, mock_setup, mock_create, mock_path
    ):
        """Test that main() returns True on successful setup."""
        # Mock all dependencies to ensure success
        mock_path.return_value.exists.return_value = True
        mock_create.return_value = True
        mock_setup.return_value = True
        mock_verify.return_value = True

        result = main()

        assert result is True

    @patch("src.setup_dev.Path")
    @patch("src.setup_dev.create_directories")
    # pylint: disable=unused-argument
    def test_main_returns_false_on_failure(self, mock_create, mock_path):
        """Test that main() returns False when setup fails."""
        # Mock dependencies to cause failure
        mock_path.return_value.exists.return_value = True
        mock_create.return_value = False  # Cause failure

        result = main()

        assert result is False
