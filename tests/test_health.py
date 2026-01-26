"""Tests for health check endpoint and response model."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import Settings
from src.routes.health import HealthResponse


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
                status="invalid",
                version="1.0.0",
                timestamp=datetime.now(timezone.utc),
            )

    def test_health_response_requires_version(self) -> None:
        """Version field should be required."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="healthy",
                timestamp=datetime.now(timezone.utc),
            )

    def test_health_response_requires_timestamp(self) -> None:
        """Timestamp field should be required."""
        with pytest.raises(ValidationError):
            HealthResponse(
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
