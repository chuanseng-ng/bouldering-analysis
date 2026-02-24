"""Tests for health check endpoint and response model."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import Settings
from src.routes.health import DbHealthResponse, HealthResponse


@pytest.fixture
def mock_supabase_client() -> MagicMock:
    """Provide a pre-configured mock Supabase client for DB health check tests.

    Configures the chained table query interface to succeed by default.
    Tests that simulate query failures can override ``execute.side_effect``.
    """
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock()
    return mock_client


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response_valid_healthy(self) -> None:
        """Should accept 'healthy' status."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
        )
        assert response.status == "healthy"

    def test_health_response_valid_degraded(self) -> None:
        """Should accept 'degraded' status."""
        response = HealthResponse(
            status="degraded",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
        )
        assert response.status == "degraded"

    def test_health_response_valid_unhealthy(self) -> None:
        """Should accept 'unhealthy' status."""
        response = HealthResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
        )
        assert response.status == "unhealthy"

    def test_health_response_invalid_status(self) -> None:
        """Should reject invalid status values."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="invalid",  # type: ignore[arg-type]
                version="1.0.0",
                timestamp=datetime.now(timezone.utc),
            )

    def test_health_response_requires_version(self) -> None:
        """Version field should be required."""
        with pytest.raises(ValidationError):
            HealthResponse(  # type: ignore[call-arg]
                status="healthy",
                timestamp=datetime.now(timezone.utc),
            )

    def test_health_response_requires_timestamp(self) -> None:
        """Timestamp field should be required."""
        with pytest.raises(ValidationError):
            HealthResponse(  # type: ignore[call-arg]
                status="healthy",
                version="1.0.0",
            )

    def test_health_response_json_serialization(self) -> None:
        """Response should serialize to JSON correctly."""
        now = datetime.now(timezone.utc)
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=now,
        )
        json_data = response.model_dump_json()
        assert "healthy" in json_data
        assert "1.0.0" in json_data

    def test_health_response_schema_example(self) -> None:
        """Model should have JSON schema example."""
        schema = HealthResponse.model_json_schema()
        assert "example" in schema
        assert schema["example"]["status"] == "healthy"


class TestHealthEndpoint:
    """Integration tests for health check endpoints."""

    def test_health_endpoint_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_response_structure(self, client: TestClient) -> None:
        """Health response should contain required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_endpoint_status_healthy(self, client: TestClient) -> None:
        """Health status should be 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_version(
        self, client: TestClient, app_settings: Settings
    ) -> None:
        """Health response should include version."""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == app_settings.app_version

    def test_health_endpoint_versioned_api(self, client: TestClient) -> None:
        """Versioned health endpoint should return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDbHealthResponse:
    """Tests for DbHealthResponse model."""

    def test_db_health_response_valid_healthy(self) -> None:
        """Should accept 'healthy' status."""
        response = DbHealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
        )
        assert response.status == "healthy"

    def test_db_health_response_valid_degraded(self) -> None:
        """Should accept 'degraded' status."""
        response = DbHealthResponse(
            status="degraded",
            version="1.0.0",
            timestamp=datetime.now(timezone.utc),
        )
        assert response.status == "degraded"

    def test_db_health_response_rejects_unhealthy(self) -> None:
        """DbHealthResponse should not accept 'unhealthy' (only healthy/degraded)."""
        with pytest.raises(ValidationError):
            DbHealthResponse(
                status="unhealthy",  # type: ignore[arg-type]
                version="1.0.0",
                timestamp=datetime.now(timezone.utc),
            )


class TestDbHealthEndpoint:
    """Integration tests for the database health check endpoint."""

    @patch("src.routes.health.get_supabase_client")
    def test_db_health_returns_healthy_when_supabase_reachable(
        self,
        mock_get_client: MagicMock,
        client: TestClient,
        mock_supabase_client: MagicMock,
    ) -> None:
        """DB health endpoint should return 'healthy' when Supabase responds."""
        mock_get_client.return_value = mock_supabase_client

        response = client.get("/api/v1/health/db")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    @patch("src.routes.health.get_supabase_client")
    def test_db_health_returns_degraded_when_supabase_unreachable(
        self, mock_get_client: MagicMock, client: TestClient
    ) -> None:
        """DB health endpoint should return 'degraded' when Supabase fails."""
        mock_get_client.side_effect = Exception("Connection refused")

        response = client.get("/api/v1/health/db")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    @patch("src.routes.health.get_supabase_client")
    def test_db_health_returns_degraded_when_query_fails(
        self,
        mock_get_client: MagicMock,
        client: TestClient,
        mock_supabase_client: MagicMock,
    ) -> None:
        """DB health endpoint should return 'degraded' when the DB query fails."""
        mock_supabase_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception(
            "Table not found"
        )
        mock_get_client.return_value = mock_supabase_client

        response = client.get("/api/v1/health/db")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    @patch("src.routes.health.get_settings")
    def test_db_health_returns_degraded_when_health_check_table_invalid(
        self, mock_get_settings: MagicMock, client: TestClient
    ) -> None:
        """DB health endpoint should return 'degraded' when health_check_table is unknown."""
        mock_settings = MagicMock()
        mock_settings.health_check_table = "unknown_table"
        mock_settings.app_version = "0.1.0"
        mock_get_settings.return_value = mock_settings

        response = client.get("/api/v1/health/db")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    @patch("src.routes.health.get_supabase_client")
    @patch("src.routes.health.get_settings")
    def test_db_health_uses_configured_table_name(
        self,
        mock_get_settings: MagicMock,
        mock_get_client: MagicMock,
        client: TestClient,
        mock_supabase_client: MagicMock,
    ) -> None:
        """DB health endpoint should query the table specified in health_check_table."""
        mock_settings = MagicMock()
        mock_settings.health_check_table = "routes"
        mock_settings.app_version = "0.1.0"
        mock_get_settings.return_value = mock_settings
        mock_get_client.return_value = mock_supabase_client

        response = client.get("/api/v1/health/db")

        assert response.status_code == 200
        mock_supabase_client.table.assert_called_once_with("routes")
