"""Tests for configuration management module."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings, get_settings_override


class TestSettings:
    """Tests for the Settings class."""

    def test_default_app_name(self) -> None:
        """Default app name should be 'bouldering-analysis'."""
        settings = Settings()
        assert settings.app_name == "bouldering-analysis"

    def test_default_app_version(self) -> None:
        """Default app version should be '0.1.0'."""
        settings = Settings()
        assert settings.app_version == "0.1.0"

    def test_default_debug_is_false(self) -> None:
        """Debug should default to False."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("BA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.debug is False

    def test_default_testing_is_false(self) -> None:
        """Testing should default to False."""
        settings = Settings()
        assert settings.testing is False

    def test_default_cors_origins(self) -> None:
        """CORS origins should default to wildcard."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("BA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.cors_origins == ["*"]

    def test_default_log_level(self) -> None:
        """Log level should default to INFO."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("BA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.log_level == "INFO"

    def test_default_supabase_url(self) -> None:
        """Supabase URL should default to empty string."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("BA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.supabase_url == ""

    def test_default_supabase_key(self) -> None:
        """Supabase key should default to empty string."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("BA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.supabase_key == ""

    def test_settings_from_env_vars(self) -> None:
        """Settings should load from environment variables."""
        with patch.dict(os.environ, {"BA_APP_NAME": "test-app"}):
            # Clear cache to get fresh settings
            get_settings.cache_clear()
            settings = Settings()
            assert settings.app_name == "test-app"

    def test_settings_debug_from_env(self) -> None:
        """Debug setting should load from BA_DEBUG env var."""
        with patch.dict(os.environ, {"BA_DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is True

    def test_settings_log_level_from_env(self) -> None:
        """Log level should load from BA_LOG_LEVEL env var."""
        with patch.dict(os.environ, {"BA_LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.log_level == "DEBUG"

    def test_settings_supabase_url_from_env(self) -> None:
        """Supabase URL should load from BA_SUPABASE_URL env var."""
        with patch.dict(os.environ, {"BA_SUPABASE_URL": "https://test.supabase.co"}):
            settings = Settings()
            assert settings.supabase_url == "https://test.supabase.co"

    def test_settings_supabase_key_from_env(self) -> None:
        """Supabase key should load from BA_SUPABASE_KEY env var."""
        with patch.dict(os.environ, {"BA_SUPABASE_KEY": "test-key-12345"}):
            settings = Settings()
            assert settings.supabase_key == "test-key-12345"


class TestLogLevelValidation:
    """Tests for log level validation."""

    def test_valid_log_levels(self) -> None:
        """All standard log levels should be valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level

    def test_log_level_case_insensitive(self) -> None:
        """Log level validation should be case insensitive."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_invalid_log_level_raises_error(self) -> None:
        """Invalid log level should raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")


class TestCorsOriginsValidation:
    """Tests for CORS origins parsing."""

    def test_cors_origins_from_list(self) -> None:
        """CORS origins should accept a list."""
        origins = ["http://localhost:3000", "http://example.com"]
        settings = Settings(cors_origins=origins)
        assert settings.cors_origins == origins

    def test_cors_origins_from_json_string_env(self) -> None:
        """CORS origins should parse JSON array string from env."""
        with patch.dict(os.environ, {"BA_CORS_ORIGINS": '["http://localhost:3000"]'}):
            settings = Settings()
            assert settings.cors_origins == ["http://localhost:3000"]

    def test_cors_origins_from_multiple_json_env(self) -> None:
        """CORS origins should parse JSON array with multiple origins from env."""
        with patch.dict(
            os.environ,
            {"BA_CORS_ORIGINS": '["http://localhost:3000", "http://example.com"]'},
        ):
            settings = Settings()
            assert settings.cors_origins == [
                "http://localhost:3000",
                "http://example.com",
            ]

    def test_cors_origins_validator_with_json_string(self) -> None:
        """Validator should parse JSON string when passed directly."""
        # Test the validator directly by calling it
        result = Settings.parse_cors_origins('["http://localhost:3000"]')
        assert result == ["http://localhost:3000"]

    def test_cors_origins_validator_with_comma_separated(self) -> None:
        """Validator should parse comma-separated string."""
        result = Settings.parse_cors_origins("http://a.com, http://b.com")
        assert result == ["http://a.com", "http://b.com"]

    def test_cors_origins_validator_with_non_list_value(self) -> None:
        """Validator should return empty list for unexpected types."""
        result = Settings.parse_cors_origins(123)
        assert result == []


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings_instance(self) -> None:
        """get_settings should return a Settings instance."""
        # Clear cache first
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self) -> None:
        """get_settings should return the same cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestGetSettingsOverride:
    """Tests for get_settings_override function."""

    def test_override_single_setting(self) -> None:
        """Should override a single setting."""
        settings = get_settings_override({"debug": True})
        assert settings.debug is True

    def test_override_multiple_settings(self) -> None:
        """Should override multiple settings."""
        settings = get_settings_override(
            {
                "debug": True,
                "testing": True,
                "app_version": "2.0.0",
            }
        )
        assert settings.debug is True
        assert settings.testing is True
        assert settings.app_version == "2.0.0"

    def test_override_preserves_defaults(self) -> None:
        """Non-overridden settings should keep defaults."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("BA_")}
        with patch.dict(os.environ, env, clear=True):
            settings = get_settings_override({"debug": True})
            assert settings.app_name == "bouldering-analysis"
            assert settings.log_level == "INFO"

    def test_override_creates_new_instance(self) -> None:
        """Override should create a new Settings instance each time."""
        settings1 = get_settings_override({"debug": True})
        settings2 = get_settings_override({"debug": True})
        assert settings1 is not settings2
