"""Pytest configuration and fixtures.

This module provides shared fixtures for testing the FastAPI application.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings, get_settings_override
from src.database.supabase_client import get_supabase_client


@pytest.fixture
def test_settings() -> dict[str, Any]:
    """Provide test-specific settings overrides.

    Returns:
        Dictionary of settings to override for testing.
    """
    return {
        "testing": True,
        "debug": True,
        "log_level": "DEBUG",
        "cors_origins": ["http://localhost:3000", "http://test"],
        "rate_limit_upload": 1000,  # effectively disabled in tests
    }


@pytest.fixture
def app(test_settings: dict[str, Any]) -> FastAPI:
    """Create test application instance.

    Args:
        test_settings: Test-specific settings overrides.

    Returns:
        Configured FastAPI application for testing.
    """
    return create_app(test_settings)


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Create synchronous test client.

    Args:
        app: FastAPI application instance.

    Yields:
        TestClient for making HTTP requests to the app.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def app_settings(app: FastAPI) -> Settings:
    """Get settings from the test application.

    Args:
        app: FastAPI application instance.

    Returns:
        Settings instance used by the application.
    """
    settings: Settings = app.state.settings
    return settings


@pytest.fixture(autouse=True)
def _clear_supabase_cache() -> Generator[None, None, None]:
    """Clear the Supabase client lru_cache before and after every test.

    Yields:
        None
    """
    get_supabase_client.cache_clear()
    yield
    get_supabase_client.cache_clear()


@pytest.fixture
def mock_supabase_client() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Provide mocked Supabase client with bucket for storage tests.

    This fixture reduces boilerplate by setting up the common mock pattern
    used across multiple Supabase storage tests.

    Yields:
        Tuple of (mock_client, mock_bucket) for test use.

    Example:
        >>> def test_storage(mock_supabase_client):
        ...     mock_client, mock_bucket = mock_supabase_client
        ...     mock_bucket.upload.return_value = None
        ...     # Test code here
    """
    get_supabase_client.cache_clear()

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.storage.from_.return_value = mock_bucket

    with (
        patch("src.database.supabase_client.get_settings") as mock_get_settings,
        patch("src.database.supabase_client.create_client") as mock_create_client,
    ):
        mock_get_settings.return_value = get_settings_override(
            {
                "supabase_url": "https://test.supabase.co",
                "supabase_key": "test-key",
            }
        )
        mock_create_client.return_value = mock_client

        yield mock_client, mock_bucket
