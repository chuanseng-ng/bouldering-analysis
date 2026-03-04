"""Pytest configuration and fixtures.

This module provides shared fixtures and helper utilities for testing the
FastAPI application and graph modules.
"""

from collections.abc import Generator
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings, get_settings_override
from src.database.supabase_client import get_supabase_client
from src.graph.types import ClassifiedHold
from src.training.classification_dataset import HOLD_CLASSES


# ---------------------------------------------------------------------------
# Shared graph test helpers (module-level, not fixtures)
# ---------------------------------------------------------------------------


def make_classified_hold_for_tests(
    hold_id: int = 0,
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.1,
    height: float = 0.1,
    hold_type: str = "jug",
    detection_class: Literal["hold", "volume"] = "hold",
    detection_confidence: float = 0.9,
    type_confidence: float = 0.8,
) -> ClassifiedHold:
    """Create a ClassifiedHold directly for use in graph-module tests.

    Constructs a valid :class:`~src.graph.types.ClassifiedHold` without
    going through :func:`~src.graph.types.make_classified_hold`, so tests
    can control every field independently.

    Args:
        hold_id: Non-negative integer identifying the hold.
        x_center: Horizontal centre, normalised [0, 1].
        y_center: Vertical centre, normalised [0, 1].
        width: Bounding box width, normalised [0, 1].
        height: Bounding box height, normalised [0, 1].
        hold_type: One of the 6 canonical hold classes.
        detection_class: YOLO class — ``"hold"`` or ``"volume"``.
        detection_confidence: YOLO detection confidence [0, 1].
        type_confidence: Classifier confidence for ``hold_type`` [0, 1].

    Returns:
        A validated :class:`ClassifiedHold` instance.
    """
    probs = {
        c: (
            type_confidence
            if c == hold_type
            else (1.0 - type_confidence) / max(len(HOLD_CLASSES) - 1, 1)
        )
        for c in HOLD_CLASSES
    }
    return ClassifiedHold(
        hold_id=hold_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        detection_class=detection_class,
        detection_confidence=detection_confidence,
        hold_type=hold_type,
        type_confidence=type_confidence,
        type_probabilities=probs,
    )


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
