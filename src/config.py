"""Application configuration management.

This module provides centralized configuration using Pydantic Settings,
loading values from environment variables with sensible defaults.
"""

import json
import logging
from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables prefixed with BA_.
    For example, BA_DEBUG=true sets debug=True.

    Attributes:
        app_name: Application name for logging and identification.
        app_version: Semantic version string.
        debug: Enable debug mode (disable in production).
        testing: Enable testing mode.
        cors_origins: List of allowed CORS origins.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        supabase_url: Supabase project URL (required for database operations).
        supabase_key: Supabase API key (required for database operations).
        supabase_timeout_seconds: Supabase PostgREST request timeout in seconds.
        max_upload_size_mb: Maximum allowed file upload size in megabytes.
        storage_bucket: Name of the Supabase Storage bucket for route images.
        allowed_image_types: List of allowed MIME types for image uploads.
        health_check_table: Database table queried by the DB health endpoint.
        inference_timeout_seconds: Timeout in seconds for ML inference operations
            wrapped in asyncio.to_thread. Applies to detect_holds, classify_holds,
            and related pipeline calls.
        api_key: Optional API key required via the ``X-API-Key`` header on all
            non-health endpoints. Empty string disables authentication.
        rate_limit_upload: Maximum upload requests per IP per minute on
            ``POST /api/v1/routes/upload``. Set to 0 to disable rate limiting.
    """

    app_name: str = "bouldering-analysis"
    app_version: str = "0.1.0"
    debug: bool = False
    testing: bool = False
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_timeout_seconds: int = 10

    # Upload configuration
    max_upload_size_mb: int = 10
    storage_bucket: str = "route-images"
    allowed_image_types: list[str] = ["image/jpeg", "image/png"]

    # Health check configuration
    health_check_table: str = "routes"

    # Inference timeout configuration
    inference_timeout_seconds: int = 30

    # Security configuration
    api_key: str = ""  # empty = no authentication required

    # Rate limiting: max upload requests per minute per IP (0 = disabled)
    rate_limit_upload: int = 10

    model_config = SettingsConfigDict(
        env_prefix="BA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("supabase_timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is a positive integer.

        Warns at startup if the value exceeds 60 seconds, as a high timeout
        can cause long hangs when Supabase is unreachable.
        """
        if v <= 0:
            raise ValueError("supabase_timeout_seconds must be a positive integer")
        if v > 60:
            logging.getLogger(__name__).warning(
                "supabase_timeout_seconds is set to %d — this may cause long hangs "
                "if Supabase is unreachable. Consider a lower value (10–30s).",
                v,
            )
        return v

    @field_validator("inference_timeout_seconds")
    @classmethod
    def validate_inference_timeout(cls, v: int) -> int:
        """Validate inference timeout is a positive integer.

        Args:
            v: Timeout value in seconds.

        Returns:
            Validated timeout value.

        Raises:
            ValueError: If value is not positive.
        """
        if v <= 0:
            raise ValueError("inference_timeout_seconds must be a positive integer")
        return v

    @field_validator("api_key")
    @classmethod
    def strip_api_key(cls, v: str) -> str:
        """Strip surrounding whitespace from api_key.

        Environment variables read from .env files can carry trailing newlines
        or spaces, which would cause every authentication attempt to fail.
        An all-whitespace value is treated as disabled (empty string).

        Args:
            v: Raw api_key value from the environment.

        Returns:
            Stripped api_key; empty string means authentication is disabled.
        """
        return v.strip()

    @field_validator("rate_limit_upload")
    @classmethod
    def validate_rate_limit_upload(cls, v: int) -> int:
        """Validate rate limit is non-negative.

        Args:
            v: Rate limit value.

        Returns:
            Validated rate limit value.

        Raises:
            ValueError: If value is negative.
        """
        if v < 0:
            raise ValueError("rate_limit_upload must be >= 0 (0 = disabled)")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle JSON array string or comma-separated
            if v.startswith("["):
                try:
                    result: list[str] = json.loads(v)
                    return result
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON format for cors_origins: {e.msg}"
                    ) from e
            return [origin.strip() for origin in v.split(",")]
        if isinstance(v, list):
            return list(v)
        return []


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Settings are loaded once and cached for the lifetime of the application.
    Use this function to access settings throughout the codebase.

    Returns:
        Settings instance with values from environment or defaults.

    Example:
        >>> settings = get_settings()
        >>> print(settings.app_name)
        'bouldering-analysis'
    """
    return Settings()


def get_settings_override(overrides: dict[str, Any]) -> Settings:
    """Create settings with specific overrides.

    Used primarily for testing to create settings with custom values
    without affecting the cached settings. Disables .env file loading
    to ensure only the provided overrides and defaults are used.

    Args:
        overrides: Dictionary of setting names to override values.

    Returns:
        New Settings instance with overrides applied.

    Example:
        >>> test_settings = get_settings_override({"testing": True})
        >>> test_settings.testing
        True
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
