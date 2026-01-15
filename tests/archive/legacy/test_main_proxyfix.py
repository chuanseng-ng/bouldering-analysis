"""
Tests for ProxyFix configuration in main.py.

These tests call configure_proxy_fix() to test the ProxyFix configuration
with different environment variables, allowing coverage tracking.
"""

from typing import Generator
import pytest
from werkzeug.middleware.proxy_fix import ProxyFix


@pytest.fixture
def clean_proxy_fix() -> Generator:
    """
    Fixture to clean and restore the ProxyFix configuration.

    This restores the original wsgi_app after each test to ensure
    a clean state for subsequent tests.
    """
    from src import main  # pylint: disable=import-outside-toplevel

    # Get the base wsgi_app without ProxyFix wrapper
    # If it's already wrapped, unwrap it; otherwise use the current one
    if isinstance(main.app.wsgi_app, ProxyFix):
        # ProxyFix stores the original app in its 'app' attribute
        original_wsgi_app = main.app.wsgi_app.app
    else:
        original_wsgi_app = main.app.wsgi_app

    yield

    # Restore original wsgi_app to remove ProxyFix
    main.app.wsgi_app = original_wsgi_app  # type: ignore[method-assign, assignment]


class TestProxyFixConfiguration:  # pylint: disable=redefined-outer-name
    """Test ProxyFix middleware configuration - covers lines 30-65."""

    def test_proxy_fix_enabled_with_valid_env_vars(
        self,
        clean_proxy_fix,
        monkeypatch,  # pylint: disable=unused-argument
    ):
        """Test ProxyFix configuration with valid environment variables - covers lines 44-56."""
        from src import main  # pylint: disable=import-outside-toplevel

        # Set environment variables before configuring
        monkeypatch.setenv("ENABLE_PROXY_FIX", "true")
        monkeypatch.setenv("PROXY_FIX_X_FOR", "2")
        monkeypatch.setenv("PROXY_FIX_X_PROTO", "2")
        monkeypatch.setenv("PROXY_FIX_X_HOST", "2")
        monkeypatch.setenv("PROXY_FIX_X_PORT", "2")

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        # Verify ProxyFix was applied
        assert isinstance(main.app.wsgi_app, ProxyFix), "ProxyFix should be enabled"

        # Verify the ProxyFix configuration
        assert main.app.wsgi_app.x_for == 2
        assert main.app.wsgi_app.x_proto == 2
        assert main.app.wsgi_app.x_host == 2
        assert main.app.wsgi_app.x_port == 2

    def test_proxy_fix_enabled_with_invalid_env_vars(
        self,
        clean_proxy_fix,
        monkeypatch,  # pylint: disable=unused-argument
    ):
        """Test ProxyFix with invalid env vars falls back to defaults - covers lines 49-51."""
        from src import main  # pylint: disable=import-outside-toplevel

        # Set environment variables with invalid values
        monkeypatch.setenv("ENABLE_PROXY_FIX", "true")
        monkeypatch.setenv("PROXY_FIX_X_FOR", "invalid_value")
        monkeypatch.setenv("PROXY_FIX_X_PROTO", "not_a_number")
        monkeypatch.setenv("PROXY_FIX_X_HOST", "abc")
        monkeypatch.setenv("PROXY_FIX_X_PORT", "xyz")

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        # Verify ProxyFix was applied with fallback values (1)
        assert isinstance(main.app.wsgi_app, ProxyFix), (
            "ProxyFix should be enabled with fallback"
        )
        assert main.app.wsgi_app.x_for == 1
        assert main.app.wsgi_app.x_proto == 1
        assert main.app.wsgi_app.x_host == 1
        assert main.app.wsgi_app.x_port == 1

    def test_proxy_fix_disabled_by_default(self, clean_proxy_fix, monkeypatch):  # pylint: disable=unused-argument
        """Test that ProxyFix is disabled by default - covers lines 37-42."""
        from src import main  # pylint: disable=import-outside-toplevel

        # Ensure ENABLE_PROXY_FIX is not set (explicitly remove it)
        monkeypatch.delenv("ENABLE_PROXY_FIX", raising=False)

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        # Verify ProxyFix was NOT applied
        assert not isinstance(main.app.wsgi_app, ProxyFix), (
            "ProxyFix should be disabled by default"
        )

    def test_proxy_fix_enabled_with_1_value(self, clean_proxy_fix, monkeypatch):  # pylint: disable=unused-argument
        """Test ProxyFix can be enabled with '1' - covers enable check."""
        from src import main  # pylint: disable=import-outside-toplevel

        monkeypatch.setenv("ENABLE_PROXY_FIX", "1")

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        assert isinstance(main.app.wsgi_app, ProxyFix), (
            "ProxyFix should be enabled with '1'"
        )

    def test_proxy_fix_enabled_with_yes_value(self, clean_proxy_fix, monkeypatch):  # pylint: disable=unused-argument
        """Test ProxyFix can be enabled with 'yes' - covers enable check."""
        from src import main  # pylint: disable=import-outside-toplevel

        monkeypatch.setenv("ENABLE_PROXY_FIX", "yes")

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        assert isinstance(main.app.wsgi_app, ProxyFix), (
            "ProxyFix should be enabled with 'yes'"
        )

    def test_proxy_fix_disabled_with_false_value(self, clean_proxy_fix, monkeypatch):  # pylint: disable=unused-argument
        """Test ProxyFix is disabled with 'false'."""
        from src import main  # pylint: disable=import-outside-toplevel

        monkeypatch.setenv("ENABLE_PROXY_FIX", "false")

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        assert not isinstance(main.app.wsgi_app, ProxyFix), (
            "ProxyFix should be disabled with 'false'"
        )

    def test_proxy_fix_with_partial_config(self, clean_proxy_fix, monkeypatch):  # pylint: disable=unused-argument
        """Test ProxyFix with only some env vars set uses defaults for others - covers lines 45-48."""
        from src import main  # pylint: disable=import-outside-toplevel

        monkeypatch.setenv("ENABLE_PROXY_FIX", "true")
        monkeypatch.setenv("PROXY_FIX_X_FOR", "3")
        # Only set X_FOR, others should default to 1

        # Call configure_proxy_fix to apply the configuration
        main.configure_proxy_fix()

        assert isinstance(main.app.wsgi_app, ProxyFix), "ProxyFix should be enabled"
        assert main.app.wsgi_app.x_for == 3
        # When env vars are not set, get() returns "1" as default in the code
        assert main.app.wsgi_app.x_proto == 1
        assert main.app.wsgi_app.x_host == 1
        assert main.app.wsgi_app.x_port == 1
