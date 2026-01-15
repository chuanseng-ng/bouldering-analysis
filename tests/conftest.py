"""Pytest configuration and fixtures.

This module provides shared fixtures for testing the FastAPI application.
"""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings


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
