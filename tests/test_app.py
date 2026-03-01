"""Tests for FastAPI application factory and core functionality."""

from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """Factory should return a FastAPI application instance."""
        app = create_app({"testing": True})
        assert isinstance(app, FastAPI)

    def test_create_app_with_config_override(self) -> None:
        """Config override should modify application settings."""
        custom_version = "1.2.3"
        app = create_app({"app_version": custom_version, "testing": True})
        assert app.state.settings.app_version == custom_version

    def test_create_app_stores_settings_in_state(self) -> None:
        """Settings should be accessible via app.state."""
        app = create_app({"testing": True})
        assert hasattr(app.state, "settings")
        assert isinstance(app.state.settings, Settings)

    def test_create_app_sets_title_from_settings(self) -> None:
        """App title should match app_name setting."""
        app = create_app({"app_name": "test-app", "testing": True})
        assert app.title == "test-app"

    def test_create_app_enables_docs_in_debug_mode(self) -> None:
        """Docs endpoints should be enabled in debug mode."""
        app = create_app({"debug": True, "testing": True})
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_create_app_enables_docs_in_testing_mode(self) -> None:
        """Docs endpoints should be enabled in testing mode."""
        app = create_app({"debug": False, "testing": True})
        assert app.docs_url == "/docs"

    def test_create_app_disables_docs_in_production(self) -> None:
        """Docs endpoints should be disabled when not in debug/testing."""
        app = create_app({"debug": False, "testing": False})
        assert app.docs_url is None
        assert app.redoc_url is None


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_endpoint_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_response_schema(self, client: TestClient) -> None:
        """Health response should contain required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data

    def test_health_endpoint_returns_healthy_status(self, client: TestClient) -> None:
        """Health endpoint should return 'healthy' status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_returns_correct_version(
        self, client: TestClient, app_settings: Settings
    ) -> None:
        """Health endpoint should return the configured app version."""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == app_settings.app_version

    def test_health_endpoint_returns_valid_timestamp(self, client: TestClient) -> None:
        """Health endpoint should return a valid ISO timestamp."""
        response = client.get("/health")
        data = response.json()

        # Should be parseable as datetime
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert timestamp.tzinfo is not None

        # Should be recent (within last minute)
        now = datetime.now(timezone.utc)
        delta = abs((now - timestamp).total_seconds())
        assert delta < 60


class TestVersionedHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_versioned_health_endpoint_returns_200(self, client: TestClient) -> None:
        """Versioned health endpoint should return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_versioned_health_same_as_root(self, client: TestClient) -> None:
        """Versioned health should return same schema as root."""
        root_response = client.get("/health")
        versioned_response = client.get("/api/v1/health")

        root_data = root_response.json()
        versioned_data = versioned_response.json()

        assert root_data["status"] == versioned_data["status"]
        assert root_data["version"] == versioned_data["version"]


class TestCorsMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_headers_present_on_options(self, client: TestClient) -> None:
        """CORS headers should be present on OPTIONS requests."""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_configured_origins(self, client: TestClient) -> None:
        """CORS should allow requests from configured origins."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.headers.get("access-control-allow-origin") in [
            "http://localhost:3000",
            "*",
        ]


class TestRequestIdMiddleware:
    """Tests for request ID middleware."""

    def test_request_id_header_generated(self, client: TestClient) -> None:
        """Request ID should be generated if not provided."""
        response = client.get("/health")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0

    def test_request_id_header_preserved(self, client: TestClient) -> None:
        """Provided request ID should be preserved in response."""
        custom_id = "test-request-id-12345"
        response = client.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        assert response.headers["x-request-id"] == custom_id

    def test_request_id_is_uuid_format(self, client: TestClient) -> None:
        """Generated request ID should be in UUID format."""
        response = client.get("/health")
        request_id = response.headers["x-request-id"]

        # UUID format: 8-4-4-4-12 hex characters
        parts = request_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12


class TestOpenAPISchema:
    """Tests for OpenAPI schema availability."""

    def test_openapi_schema_available(self, client: TestClient) -> None:
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_openapi_schema_contains_health_endpoint(self, client: TestClient) -> None:
        """OpenAPI schema should document the health endpoint."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]

    def test_docs_endpoint_available_in_test_mode(self, client: TestClient) -> None:
        """Swagger docs should be available in test mode."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestConfigSettings:
    """Tests for configuration and settings."""

    def test_settings_testing_mode_enabled(self, app_settings: Settings) -> None:
        """Testing mode should be enabled in test fixtures."""
        assert app_settings.testing is True

    def test_settings_debug_mode_enabled(self, app_settings: Settings) -> None:
        """Debug mode should be enabled in test fixtures."""
        assert app_settings.debug is True

    def test_settings_cors_origins_configured(self, app_settings: Settings) -> None:
        """CORS origins should be configured."""
        assert len(app_settings.cors_origins) > 0


class TestLogging:
    """Tests for logging configuration."""

    def test_logging_configured_on_app_creation(self) -> None:
        """Logging should be configured when app is created."""
        with patch("src.app.configure_logging") as mock_configure:
            create_app({"testing": True})
            mock_configure.assert_called_once()

    def test_logging_uses_settings_log_level(self) -> None:
        """Logging should use the configured log level."""
        with patch("src.app.configure_logging") as mock_configure:
            create_app({"testing": True, "log_level": "DEBUG"})
            mock_configure.assert_called_once()
            args, kwargs = mock_configure.call_args
            assert args[0] == "DEBUG" or kwargs.get("log_level") == "DEBUG"


class TestApiKeyMiddleware:
    """Tests for API key authentication middleware."""

    def test_health_endpoint_bypasses_api_key(self) -> None:
        """Health endpoints must remain accessible without an API key."""
        app = create_app({"testing": True, "api_key": "secret"})
        with TestClient(app) as c:
            assert c.get("/health").status_code == 200
            assert c.get("/api/v1/health").status_code == 200

    def test_missing_api_key_returns_401(self) -> None:
        """Request to a protected endpoint without X-API-Key returns 401."""
        app = create_app(
            {"testing": True, "api_key": "secret", "rate_limit_upload": 1000}
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get("/api/v1/routes/00000000-0000-0000-0000-000000000000")
            assert response.status_code == 401
            assert "API key" in response.json()["detail"]

    def test_wrong_api_key_returns_401(self) -> None:
        """Request with an incorrect key returns 401."""
        app = create_app(
            {"testing": True, "api_key": "secret", "rate_limit_upload": 1000}
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get(
                "/api/v1/routes/00000000-0000-0000-0000-000000000000",
                headers={"X-API-Key": "wrong"},
            )
            assert response.status_code == 401

    def test_correct_api_key_passes_through(self) -> None:
        """Request with the correct key reaches the route handler (not 401)."""
        app = create_app(
            {"testing": True, "api_key": "secret", "rate_limit_upload": 1000}
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get(
                "/api/v1/routes/00000000-0000-0000-0000-000000000000",
                headers={"X-API-Key": "secret"},
            )
            # 404 is fine — it means the route handler was reached, not blocked by auth
            assert response.status_code != 401

    def test_empty_api_key_disables_auth(self) -> None:
        """When api_key is empty all endpoints are open (no 401)."""
        app = create_app({"testing": True, "api_key": "", "rate_limit_upload": 1000})
        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get("/api/v1/routes/00000000-0000-0000-0000-000000000000")
            assert response.status_code != 401


class TestUploadRateLimitMiddleware:
    """Tests for the per-IP upload rate limiter."""

    def test_requests_within_limit_are_allowed(self) -> None:
        """Requests up to the configured limit should receive 2xx responses."""
        from unittest.mock import patch

        app = create_app({"testing": True, "api_key": "", "rate_limit_upload": 5})
        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("src.routes.upload.upload_to_storage") as mock_upload:
                mock_upload.return_value = "https://example.com/img.jpg"
                import io

                for _ in range(5):
                    png_bytes = (
                        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
                        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
                        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
                        b"\x00\x00\x00\x00IEND\xaeB`\x82"
                    )
                    response = c.post(
                        "/api/v1/routes/upload",
                        files={"file": ("img.png", io.BytesIO(png_bytes), "image/png")},
                    )
                    assert 200 <= response.status_code < 300

    def test_exceeding_limit_returns_429(self) -> None:
        """Requests beyond the configured limit should return 429; requests within the limit should be 2xx."""
        from unittest.mock import patch

        app = create_app({"testing": True, "api_key": "", "rate_limit_upload": 2})
        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("src.routes.upload.upload_to_storage") as mock_upload:
                mock_upload.return_value = "https://example.com/img.jpg"
                import io

                png_bytes = (
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
                    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
                    b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
                    b"\x00\x00\x00\x00IEND\xaeB`\x82"
                )
                responses = [
                    c.post(
                        "/api/v1/routes/upload",
                        files={"file": ("img.png", io.BytesIO(png_bytes), "image/png")},
                    )
                    for _ in range(3)
                ]
                # First two requests are within the limit
                assert 200 <= responses[0].status_code < 300
                assert 200 <= responses[1].status_code < 300
                # Third request exceeds the limit
                assert responses[2].status_code == 429

    def test_non_upload_endpoints_not_rate_limited(self) -> None:
        """Rate limiter must not apply to GET endpoints."""
        app = create_app({"testing": True, "api_key": "", "rate_limit_upload": 1})
        with TestClient(app, raise_server_exceptions=False) as c:
            for _ in range(5):
                assert c.get("/health").status_code != 429

    def test_rate_limit_zero_disables_limiter(self) -> None:
        """rate_limit_upload=0 should disable the rate limiter entirely."""
        from unittest.mock import patch

        app = create_app({"testing": True, "api_key": "", "rate_limit_upload": 0})
        with TestClient(app, raise_server_exceptions=False) as c:
            with patch("src.routes.upload.upload_to_storage") as mock_upload:
                mock_upload.return_value = "https://example.com/img.jpg"
                import io

                for _ in range(20):
                    png_bytes = (
                        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
                        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
                        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
                        b"\x00\x00\x00\x00IEND\xaeB`\x82"
                    )
                    response = c.post(
                        "/api/v1/routes/upload",
                        files={"file": ("img.png", io.BytesIO(png_bytes), "image/png")},
                    )
                    assert response.status_code != 429
