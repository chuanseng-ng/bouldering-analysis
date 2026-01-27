"""Tests for the route record management endpoints.

This module tests route creation and retrieval functionality including
validation, database integration, and error handling.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app import create_app
from src.database.supabase_client import SupabaseClientError, get_supabase_client


@pytest.fixture
def app_with_routes_settings() -> Any:
    """Create test application with routes-specific settings."""
    test_config = {
        "testing": True,
        "debug": True,
    }
    return create_app(test_config)


@pytest.fixture
def client_with_routes(app_with_routes_settings: Any) -> TestClient:
    """Create test client for routes testing."""
    return TestClient(app_with_routes_settings)


@pytest.fixture
def mock_supabase_table() -> MagicMock:
    """Create a mock for Supabase table operations."""
    mock_table = MagicMock()
    mock_table.insert.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    return mock_table


@pytest.fixture
def sample_route_record() -> dict[str, Any]:
    """Sample route record as returned from database."""
    return {
        "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "image_url": "https://example.supabase.co/storage/v1/object/public/route-images/2026/01/test.jpg",
        "wall_angle": 15.0,
        "created_at": "2026-01-27T12:00:00+00:00",
        "updated_at": "2026-01-27T12:00:00+00:00",
    }


class TestCreateRouteEndpoint:
    """Tests for POST /api/v1/routes endpoint."""

    @patch("src.routes.routes.insert_record")
    def test_create_route_with_image_url_only(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with only image_url should succeed."""
        # Set up mock record with the URL we're sending
        test_url = "https://example.com/image.jpg"
        record_no_angle = sample_route_record.copy()
        record_no_angle["wall_angle"] = None
        record_no_angle["image_url"] = test_url
        mock_insert.return_value = record_no_angle

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": test_url},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["image_url"] == test_url
        assert data["wall_angle"] is None
        assert "created_at" in data
        assert "updated_at" in data

    @patch("src.routes.routes.insert_record")
    def test_create_route_with_image_url_and_wall_angle(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with image_url and wall_angle should succeed."""
        mock_insert.return_value = sample_route_record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={
                "image_url": "https://example.supabase.co/storage/v1/object/public/route-images/2026/01/test.jpg",
                "wall_angle": 15.0,
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["wall_angle"] == 15.0

        # Verify insert was called with correct data
        mock_insert.assert_called_once()
        call_kwargs = mock_insert.call_args.kwargs
        assert call_kwargs["table"] == "routes"
        assert "image_url" in call_kwargs["data"]
        assert call_kwargs["data"]["wall_angle"] == 15.0

    @patch("src.routes.routes.insert_record")
    def test_create_route_with_null_wall_angle(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with explicit null wall_angle should succeed."""
        record_null_angle = sample_route_record.copy()
        record_null_angle["wall_angle"] = None
        mock_insert.return_value = record_null_angle

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": None},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["wall_angle"] is None

    @patch("src.routes.routes.insert_record")
    def test_create_route_wall_angle_boundary_min(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with wall_angle=-90 should succeed."""
        record = sample_route_record.copy()
        record["wall_angle"] = -90.0
        mock_insert.return_value = record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": -90.0},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["wall_angle"] == -90.0

    @patch("src.routes.routes.insert_record")
    def test_create_route_wall_angle_boundary_max(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with wall_angle=90 should succeed."""
        record = sample_route_record.copy()
        record["wall_angle"] = 90.0
        mock_insert.return_value = record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 90.0},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["wall_angle"] == 90.0

    @patch("src.routes.routes.insert_record")
    def test_create_route_wall_angle_zero(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with wall_angle=0 (vertical wall) should succeed."""
        record = sample_route_record.copy()
        record["wall_angle"] = 0.0
        mock_insert.return_value = record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 0},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["wall_angle"] == 0.0

    def test_create_route_invalid_image_url_not_https(
        self, client_with_routes: TestClient
    ) -> None:
        """Route creation with non-HTTPS URL should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "http://example.com/image.jpg"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "HTTPS" in str(data["detail"])

    def test_create_route_invalid_image_url_empty(
        self, client_with_routes: TestClient
    ) -> None:
        """Route creation with empty URL should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_wall_angle_too_low(
        self, client_with_routes: TestClient
    ) -> None:
        """Route creation with wall_angle < -90 should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": -91.0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_wall_angle_too_high(
        self, client_with_routes: TestClient
    ) -> None:
        """Route creation with wall_angle > 90 should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 91.0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_missing_image_url(
        self, client_with_routes: TestClient
    ) -> None:
        """Route creation without image_url should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"wall_angle": 15.0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_empty_body(self, client_with_routes: TestClient) -> None:
        """Route creation with empty body should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.insert_record")
    def test_create_route_database_error(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
    ) -> None:
        """Database error should return 500."""
        mock_insert.side_effect = SupabaseClientError("Database connection failed")

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg"},
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to create route record" in data["detail"]

    @patch("src.routes.routes.insert_record")
    def test_create_route_response_contains_all_fields(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Response should contain all expected fields."""
        mock_insert.return_value = sample_route_record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 15.0},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()

        # Verify all fields are present
        assert "id" in data
        assert "image_url" in data
        assert "wall_angle" in data
        assert "created_at" in data
        assert "updated_at" in data

        # Verify UUID format
        assert len(data["id"]) == 36
        assert data["id"].count("-") == 4

    @patch("src.routes.routes.insert_record")
    def test_create_route_wall_angle_precision(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Wall angle should be rounded to 1 decimal place."""
        record = sample_route_record.copy()
        record["wall_angle"] = 15.1
        mock_insert.return_value = record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 15.123},
        )

        assert response.status_code == status.HTTP_201_CREATED

        # Verify insert was called with rounded value
        call_kwargs = mock_insert.call_args.kwargs
        assert call_kwargs["data"]["wall_angle"] == 15.1


class TestGetRouteEndpoint:
    """Tests for GET /api/v1/routes/{route_id} endpoint."""

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_existing(
        self,
        mock_select: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Getting an existing route should return the route data."""
        mock_select.return_value = sample_route_record
        route_id = sample_route_record["id"]

        response = client_with_routes.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == route_id
        assert data["image_url"] == sample_route_record["image_url"]
        assert data["wall_angle"] == sample_route_record["wall_angle"]

        # Verify select was called correctly
        mock_select.assert_called_once_with(table="routes", record_id=route_id)

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_not_found(
        self,
        mock_select: MagicMock,
        client_with_routes: TestClient,
    ) -> None:
        """Getting a non-existent route should return 404."""
        mock_select.return_value = None
        route_id = "00000000-0000-0000-0000-000000000000"

        response = client_with_routes.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "Route not found" in data["detail"]

    def test_get_route_invalid_uuid_format(
        self, client_with_routes: TestClient
    ) -> None:
        """Getting a route with invalid UUID should return 422."""
        response = client_with_routes.get("/api/v1/routes/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "Invalid route ID format" in data["detail"]

    def test_get_route_uuid_too_short(self, client_with_routes: TestClient) -> None:
        """Getting a route with too short UUID should return 422."""
        response = client_with_routes.get("/api/v1/routes/a1b2c3d4")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_route_uuid_with_invalid_chars(
        self, client_with_routes: TestClient
    ) -> None:
        """Getting a route with invalid chars in UUID should return 422."""
        response = client_with_routes.get(
            "/api/v1/routes/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_database_error(
        self,
        mock_select: MagicMock,
        client_with_routes: TestClient,
    ) -> None:
        """Database error should return 500."""
        mock_select.side_effect = SupabaseClientError("Database query failed")
        route_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        response = client_with_routes.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve route" in data["detail"]

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_timestamps_format(
        self,
        mock_select: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Timestamps should be in ISO 8601 format ending with Z."""
        mock_select.return_value = sample_route_record
        route_id = sample_route_record["id"]

        response = client_with_routes.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["created_at"].endswith("Z")
        assert data["updated_at"].endswith("Z")

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_with_null_wall_angle(
        self,
        mock_select: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Getting a route with null wall_angle should return null."""
        record = sample_route_record.copy()
        record["wall_angle"] = None
        mock_select.return_value = record
        route_id = record["id"]

        response = client_with_routes.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["wall_angle"] is None


class TestRouteValidation:
    """Tests for input validation edge cases."""

    def test_image_url_max_length(self, client_with_routes: TestClient) -> None:
        """Image URL exceeding max length should be rejected."""
        # Create URL longer than 2048 characters
        long_url = "https://example.com/" + "a" * 2030

        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": long_url},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.insert_record")
    def test_extra_fields_ignored(
        self,
        mock_insert: MagicMock,
        client_with_routes: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Extra fields in request should be ignored."""
        mock_insert.return_value = sample_route_record

        response = client_with_routes.post(
            "/api/v1/routes",
            json={
                "image_url": "https://example.com/image.jpg",
                "wall_angle": 15.0,
                "extra_field": "should be ignored",
                "another_extra": 123,
            },
        )

        assert response.status_code == status.HTTP_201_CREATED

        # Verify extra fields were not passed to insert
        call_kwargs = mock_insert.call_args.kwargs
        assert "extra_field" not in call_kwargs["data"]
        assert "another_extra" not in call_kwargs["data"]

    def test_wall_angle_just_outside_min_boundary(
        self, client_with_routes: TestClient
    ) -> None:
        """Wall angle just below -90 should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": -90.1},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_wall_angle_just_outside_max_boundary(
        self, client_with_routes: TestClient
    ) -> None:
        """Wall angle just above 90 should be rejected."""
        response = client_with_routes.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 90.1},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestDatabaseOperations:
    """Tests for database function unit tests."""

    def test_insert_record_success(self) -> None:
        """Insert record should return complete record."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_result = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = mock_result
        mock_result.data = [
            {
                "id": "test-uuid",
                "image_url": "https://example.com/image.jpg",
                "wall_angle": 15.0,
                "created_at": "2026-01-27T12:00:00Z",
                "updated_at": "2026-01-27T12:00:00Z",
            }
        ]

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            from src.config import get_settings_override

            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            from src.database.supabase_client import insert_record

            result = insert_record(
                "routes", {"image_url": "https://example.com/image.jpg"}
            )

            assert result["id"] == "test-uuid"
            mock_client.table.assert_called_with("routes")
            mock_table.insert.assert_called_once()

    def test_insert_record_no_data_returned(self) -> None:
        """Insert record with no data returned should raise error."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_result = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = mock_result
        mock_result.data = []  # No data returned

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            from src.config import get_settings_override

            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            from src.database.supabase_client import insert_record

            with pytest.raises(SupabaseClientError, match="returned no data"):
                insert_record("routes", {"image_url": "https://example.com"})

    def test_insert_record_connection_error(self) -> None:
        """Insert record with connection error should raise wrapped error."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("Connection failed")

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            from src.config import get_settings_override

            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            from src.database.supabase_client import insert_record

            with pytest.raises(SupabaseClientError, match="Failed to insert record"):
                insert_record("routes", {"image_url": "https://example.com"})

    def test_select_record_by_id_success(self) -> None:
        """Select record should return found record."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_result = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = mock_result
        mock_result.data = [{"id": "test-uuid", "image_url": "https://example.com"}]

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            from src.config import get_settings_override

            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            from src.database.supabase_client import select_record_by_id

            result = select_record_by_id("routes", "test-uuid")

            assert result is not None
            assert result["id"] == "test-uuid"
            mock_table.eq.assert_called_with("id", "test-uuid")

    def test_select_record_by_id_not_found(self) -> None:
        """Select record for non-existent ID should return None."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_result = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.execute.return_value = mock_result
        mock_result.data = []  # No data found

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            from src.config import get_settings_override

            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            from src.database.supabase_client import select_record_by_id

            result = select_record_by_id("routes", "nonexistent-uuid")

            assert result is None

    def test_select_record_by_id_connection_error(self) -> None:
        """Select record with connection error should raise wrapped error."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("Connection failed")

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            from src.config import get_settings_override

            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            from src.database.supabase_client import select_record_by_id

            with pytest.raises(SupabaseClientError, match="Failed to select record"):
                select_record_by_id("routes", "test-uuid")


class TestRouteModels:
    """Tests for Pydantic model validation."""

    def test_route_create_model_valid(self) -> None:
        """RouteCreate model should accept valid data."""
        from src.routes.routes import RouteCreate

        data = RouteCreate(image_url="https://example.com/image.jpg", wall_angle=15.0)
        assert data.image_url == "https://example.com/image.jpg"
        assert data.wall_angle == 15.0

    def test_route_create_model_optional_wall_angle(self) -> None:
        """RouteCreate model should accept missing wall_angle."""
        from src.routes.routes import RouteCreate

        data = RouteCreate(image_url="https://example.com/image.jpg")
        assert data.image_url == "https://example.com/image.jpg"
        assert data.wall_angle is None

    def test_route_response_model_valid(self) -> None:
        """RouteResponse model should accept valid data."""
        from src.routes.routes import RouteResponse

        data = RouteResponse(
            id="test-uuid",
            image_url="https://example.com/image.jpg",
            wall_angle=15.0,
            created_at="2026-01-27T12:00:00Z",
            updated_at="2026-01-27T12:00:00Z",
        )
        assert data.id == "test-uuid"
        assert data.wall_angle == 15.0

    def test_route_response_model_null_wall_angle(self) -> None:
        """RouteResponse model should accept null wall_angle."""
        from src.routes.routes import RouteResponse

        data = RouteResponse(
            id="test-uuid",
            image_url="https://example.com/image.jpg",
            wall_angle=None,
            created_at="2026-01-27T12:00:00Z",
            updated_at="2026-01-27T12:00:00Z",
        )
        assert data.wall_angle is None


class TestTimestampFormatting:
    """Tests for timestamp formatting helper."""

    def test_format_timestamp_with_timezone(self) -> None:
        """Timestamp with timezone should be formatted correctly."""
        from src.routes.routes import _format_timestamp

        result = _format_timestamp("2026-01-27T12:00:00+00:00")
        assert result.endswith("Z")
        assert "+" not in result

    def test_format_timestamp_without_timezone(self) -> None:
        """Timestamp without timezone should get Z appended."""
        from src.routes.routes import _format_timestamp

        result = _format_timestamp("2026-01-27T12:00:00")
        assert result == "2026-01-27T12:00:00Z"

    def test_format_timestamp_already_has_z(self) -> None:
        """Timestamp already ending with Z should remain unchanged."""
        from src.routes.routes import _format_timestamp

        result = _format_timestamp("2026-01-27T12:00:00Z")
        assert result == "2026-01-27T12:00:00Z"

    def test_format_timestamp_none(self) -> None:
        """None timestamp should return empty string."""
        from src.routes.routes import _format_timestamp

        result = _format_timestamp(None)
        assert result == ""
