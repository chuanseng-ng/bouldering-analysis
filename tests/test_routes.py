"""Tests for the route record management endpoints.

This module tests route creation and retrieval functionality including
validation, database integration, and error handling.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.database.supabase_client import SupabaseClientError
from src.routes.routes import _format_timestamp


@pytest.fixture
def sample_route_record() -> dict[str, Any]:
    """Sample route record as returned from database."""
    return {
        "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "image_url": "https://example.supabase.co/storage/v1/object/public/route-images/2026/01/test.jpg",
        "wall_angle": 15.0,
        "created_at": "2026-01-27T12:00:00+00:00",
        "updated_at": "2026-01-27T12:00:00+00:00",
        "status": "pending",
    }


class TestCreateRouteEndpoint:
    """Tests for POST /api/v1/routes endpoint."""

    @patch("src.routes.routes.insert_record")
    def test_create_route_with_image_url_only(
        self,
        mock_insert: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with only image_url should succeed."""
        # Set up mock record with the URL we're sending
        test_url = "https://example.com/image.jpg"
        record_no_angle = sample_route_record.copy()
        record_no_angle["wall_angle"] = None
        record_no_angle["image_url"] = test_url
        mock_insert.return_value = record_no_angle

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with image_url and wall_angle should succeed."""
        mock_insert.return_value = sample_route_record

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with explicit null wall_angle should succeed."""
        record_null_angle = sample_route_record.copy()
        record_null_angle["wall_angle"] = None
        mock_insert.return_value = record_null_angle

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with wall_angle=-90 should succeed."""
        record = sample_route_record.copy()
        record["wall_angle"] = -90.0
        mock_insert.return_value = record

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with wall_angle=90 should succeed."""
        record = sample_route_record.copy()
        record["wall_angle"] = 90.0
        mock_insert.return_value = record

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Route creation with wall_angle=0 (vertical wall) should succeed."""
        record = sample_route_record.copy()
        record["wall_angle"] = 0.0
        mock_insert.return_value = record

        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 0},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["wall_angle"] == 0.0

    def test_create_route_invalid_image_url_not_https(self, client: TestClient) -> None:
        """Route creation with non-HTTPS URL should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"image_url": "http://example.com/image.jpg"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "HTTPS" in str(data["detail"])

    def test_create_route_invalid_image_url_empty(self, client: TestClient) -> None:
        """Route creation with empty URL should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"image_url": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_wall_angle_too_low(self, client: TestClient) -> None:
        """Route creation with wall_angle < -90 should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": -91.0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_wall_angle_too_high(self, client: TestClient) -> None:
        """Route creation with wall_angle > 90 should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 91.0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_missing_image_url(self, client: TestClient) -> None:
        """Route creation without image_url should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"wall_angle": 15.0},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_route_empty_body(self, client: TestClient) -> None:
        """Route creation with empty body should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.insert_record")
    def test_create_route_database_error(
        self,
        mock_insert: MagicMock,
        client: TestClient,
    ) -> None:
        """Database error should return 500."""
        mock_insert.side_effect = SupabaseClientError("Database connection failed")

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Response should contain all expected fields."""
        mock_insert.return_value = sample_route_record

        response = client.post(
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
        assert "status" in data

        # Verify UUID format
        assert len(data["id"]) == 36
        assert data["id"].count("-") == 4

    @patch("src.routes.routes.insert_record")
    def test_create_route_returns_500_when_record_missing_required_field(
        self,
        mock_insert: MagicMock,
        client: TestClient,
    ) -> None:
        """Database record missing 'id' should return 500, not leak a KeyError."""
        mock_insert.return_value = {
            # 'id' is intentionally absent to simulate a corrupt DB record
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00+00:00",
            "updated_at": "2026-01-27T12:00:00+00:00",
        }

        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg"},
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to create route record" in data["detail"]

    @patch("src.routes.routes.insert_record")
    def test_create_route_returns_500_when_created_at_missing(
        self,
        mock_insert: MagicMock,
        client: TestClient,
    ) -> None:
        """Database record missing 'created_at' should return 500 with field identified."""
        mock_insert.return_value = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            # 'created_at' is intentionally absent
            "updated_at": "2026-01-27T12:00:00+00:00",
        }

        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg"},
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to create route record" in data["detail"]

    @patch("src.routes.routes.insert_record")
    def test_create_route_wall_angle_precision(
        self,
        mock_insert: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Wall angle should be rounded to 1 decimal place."""
        record = sample_route_record.copy()
        record["wall_angle"] = 15.1
        mock_insert.return_value = record

        response = client.post(
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
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Getting an existing route should return the route data."""
        mock_select.return_value = sample_route_record
        route_id = sample_route_record["id"]

        response = client.get(f"/api/v1/routes/{route_id}")

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
        client: TestClient,
    ) -> None:
        """Getting a non-existent route should return 404."""
        mock_select.return_value = None
        route_id = "00000000-0000-0000-0000-000000000000"

        response = client.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "Route not found" in data["detail"]

    def test_get_route_invalid_uuid_format(self, client: TestClient) -> None:
        """Getting a route with invalid UUID should return 422."""
        response = client.get("/api/v1/routes/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_route_uuid_too_short(self, client: TestClient) -> None:
        """Getting a route with too short UUID should return 422."""
        response = client.get("/api/v1/routes/a1b2c3d4")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_route_uuid_with_invalid_chars(self, client: TestClient) -> None:
        """Getting a route with invalid chars in UUID should return 422."""
        response = client.get("/api/v1/routes/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_returns_500_when_record_missing_required_field(
        self,
        mock_select: MagicMock,
        client: TestClient,
    ) -> None:
        """Database record missing 'id' should return 500, not leak a KeyError."""
        mock_select.return_value = {
            # 'id' is intentionally absent to simulate a corrupt DB record
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00+00:00",
            "updated_at": "2026-01-27T12:00:00+00:00",
        }
        route_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        response = client.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve route" in data["detail"]

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_returns_500_when_updated_at_missing(
        self,
        mock_select: MagicMock,
        client: TestClient,
    ) -> None:
        """Database record missing 'updated_at' should return 500 with field identified."""
        mock_select.return_value = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00+00:00",
            # 'updated_at' is intentionally absent
        }
        route_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        response = client.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve route" in data["detail"]

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_database_error(
        self,
        mock_select: MagicMock,
        client: TestClient,
    ) -> None:
        """Database error should return 500."""
        mock_select.side_effect = SupabaseClientError("Database query failed")
        route_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        response = client.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to retrieve route" in data["detail"]

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_timestamps_format(
        self,
        mock_select: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Timestamps should be in ISO 8601 format ending with Z."""
        mock_select.return_value = sample_route_record
        route_id = sample_route_record["id"]

        response = client.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["created_at"].endswith("Z")
        assert data["updated_at"].endswith("Z")

    @patch("src.routes.routes.select_record_by_id")
    def test_get_route_with_null_wall_angle(
        self,
        mock_select: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Getting a route with null wall_angle should return null."""
        record = sample_route_record.copy()
        record["wall_angle"] = None
        mock_select.return_value = record
        route_id = record["id"]

        response = client.get(f"/api/v1/routes/{route_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["wall_angle"] is None


class TestRouteValidation:
    """Tests for input validation edge cases."""

    def test_image_url_max_length(self, client: TestClient) -> None:
        """Image URL exceeding max length should be rejected."""
        # Create URL longer than 2048 characters
        long_url = "https://example.com/" + "a" * 2030

        response = client.post(
            "/api/v1/routes",
            json={"image_url": long_url},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.insert_record")
    def test_extra_fields_ignored(
        self,
        mock_insert: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Extra fields in request should be ignored."""
        mock_insert.return_value = sample_route_record

        response = client.post(
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

    def test_wall_angle_just_outside_min_boundary(self, client: TestClient) -> None:
        """Wall angle just below -90 should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": -90.1},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_wall_angle_just_outside_max_boundary(self, client: TestClient) -> None:
        """Wall angle just above 90 should be rejected."""
        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg", "wall_angle": 90.1},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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

        # wall_angle has a default of None, so this is valid
        data = RouteCreate(image_url="https://example.com/image.jpg")  # type: ignore[call-arg]
        assert data.image_url == "https://example.com/image.jpg"
        assert data.wall_angle is None

    def test_route_response_model_valid(self) -> None:
        """RouteResponse model should accept valid data."""
        from src.routes.routes import RouteResponse

        data = RouteResponse(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            image_url="https://example.com/image.jpg",
            wall_angle=15.0,
            created_at="2026-01-27T12:00:00Z",
            updated_at="2026-01-27T12:00:00Z",
        )
        assert data.id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert data.wall_angle == 15.0

    def test_route_response_model_null_wall_angle(self) -> None:
        """RouteResponse model should accept null wall_angle."""
        from src.routes.routes import RouteResponse

        data = RouteResponse(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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
        result = _format_timestamp("2026-01-27T12:00:00+00:00")
        assert result.endswith("Z")
        assert "+" not in result

    def test_format_timestamp_without_timezone(self) -> None:
        """Timestamp without timezone should get Z appended."""
        result = _format_timestamp("2026-01-27T12:00:00")
        assert result == "2026-01-27T12:00:00Z"

    def test_format_timestamp_already_has_z(self) -> None:
        """Timestamp already ending with Z should remain unchanged."""
        result = _format_timestamp("2026-01-27T12:00:00Z")
        assert result == "2026-01-27T12:00:00Z"

    def test_format_timestamp_none_raises(self) -> None:
        """None timestamp should raise ValueError — required field missing from record."""
        with pytest.raises(ValueError, match="Timestamp cannot be None"):
            _format_timestamp(None)

    def test_format_timestamp_with_negative_offset(self) -> None:
        """Timestamp with negative UTC offset should be converted to UTC and Z appended."""
        result = _format_timestamp("2026-01-27T07:00:00-05:00")
        assert result == "2026-01-27T12:00:00Z"
        assert "-05:00" not in result

    def test_format_timestamp_with_positive_offset(self) -> None:
        """Timestamp with positive UTC offset should be converted to UTC and Z appended."""
        result = _format_timestamp("2026-01-27T17:00:00+05:00")
        assert result == "2026-01-27T12:00:00Z"
        assert "+05:00" not in result

    def test_format_timestamp_with_microseconds(self) -> None:
        """Microseconds should be preserved in the converted UTC timestamp."""
        result = _format_timestamp("2026-01-27T12:00:00.123456+00:00")
        assert result == "2026-01-27T12:00:00.123456Z"


class TestRecordToResponse:
    """Unit tests for _record_to_response conversion helper."""

    def test_record_to_response_valid(self) -> None:
        """Valid record should convert to RouteResponse with Z-normalised timestamps."""
        from src.routes.routes import _record_to_response

        record = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "wall_angle": 15.0,
            "created_at": "2026-01-27T12:00:00+00:00",
            "updated_at": "2026-01-27T12:00:00+00:00",
        }

        result = _record_to_response(record)

        assert result.id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert result.image_url == "https://example.com/image.jpg"
        assert result.wall_angle == 15.0
        assert result.created_at == "2026-01-27T12:00:00Z"
        assert result.updated_at == "2026-01-27T12:00:00Z"

    def test_record_to_response_null_wall_angle(self) -> None:
        """Record with null wall_angle should produce null in the response."""
        from src.routes.routes import _record_to_response

        record = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "wall_angle": None,
            "created_at": "2026-01-27T12:00:00Z",
            "updated_at": "2026-01-27T12:00:00Z",
        }

        result = _record_to_response(record)

        assert result.wall_angle is None

    def test_record_to_response_missing_id_raises_key_error(self) -> None:
        """Missing 'id' field should raise KeyError."""
        from src.routes.routes import _record_to_response

        record = {
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00Z",
            "updated_at": "2026-01-27T12:00:00Z",
        }

        with pytest.raises(KeyError):
            _record_to_response(record)

    def test_record_to_response_missing_created_at_raises_value_error(self) -> None:
        """Missing 'created_at' should raise ValueError naming the field."""
        from src.routes.routes import _record_to_response

        record = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "updated_at": "2026-01-27T12:00:00Z",
        }

        with pytest.raises(ValueError, match="created_at"):
            _record_to_response(record)

    def test_record_to_response_missing_updated_at_raises_value_error(self) -> None:
        """Missing 'updated_at' should raise ValueError naming the field."""
        from src.routes.routes import _record_to_response

        record = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00Z",
        }

        with pytest.raises(ValueError, match="updated_at"):
            _record_to_response(record)

    def test_record_to_response_status_defaults_to_pending(self) -> None:
        """Records without a status field should default to 'pending'."""
        from src.routes.routes import _record_to_response

        record = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00Z",
            "updated_at": "2026-01-27T12:00:00Z",
        }

        result = _record_to_response(record)
        assert result.status == "pending"

    def test_record_to_response_status_from_record(self) -> None:
        """Status should be read from the record when present."""
        from src.routes.routes import _record_to_response

        record = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00Z",
            "updated_at": "2026-01-27T12:00:00Z",
            "status": "done",
        }

        result = _record_to_response(record)
        assert result.status == "done"


class TestGetRouteStatusEndpoint:
    """Tests for GET /api/v1/routes/{route_id}/status endpoint."""

    @patch("src.routes.routes.select_record_by_id")
    def test_get_status_returns_pending(
        self,
        mock_select: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Status endpoint should return the route's current status."""
        mock_select.return_value = sample_route_record
        route_id = sample_route_record["id"]

        response = client.get(f"/api/v1/routes/{route_id}/status")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == route_id
        assert data["status"] == "pending"

    @patch("src.routes.routes.select_record_by_id")
    def test_get_status_returns_done(
        self,
        mock_select: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Status endpoint should reflect 'done' status."""
        record = sample_route_record.copy()
        record["status"] = "done"
        mock_select.return_value = record
        route_id = record["id"]

        response = client.get(f"/api/v1/routes/{route_id}/status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "done"

    @patch("src.routes.routes.select_record_by_id")
    def test_get_status_defaults_to_pending_when_missing(
        self,
        mock_select: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Status defaults to 'pending' when the DB record has no status field."""
        record = {k: v for k, v in sample_route_record.items() if k != "status"}
        mock_select.return_value = record

        response = client.get(f"/api/v1/routes/{sample_route_record['id']}/status")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "pending"

    @patch("src.routes.routes.select_record_by_id")
    def test_get_status_not_found(
        self,
        mock_select: MagicMock,
        client: TestClient,
    ) -> None:
        """Status endpoint should return 404 for unknown route IDs."""
        mock_select.return_value = None
        route_id = "00000000-0000-0000-0000-000000000000"

        response = client.get(f"/api/v1/routes/{route_id}/status")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_status_invalid_uuid(self, client: TestClient) -> None:
        """Invalid UUID format should return 422."""
        response = client.get("/api/v1/routes/not-a-uuid/status")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("src.routes.routes.select_record_by_id")
    def test_get_status_database_error(
        self,
        mock_select: MagicMock,
        client: TestClient,
    ) -> None:
        """Database error should return 500."""
        mock_select.side_effect = SupabaseClientError("DB failure")
        route_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        response = client.get(f"/api/v1/routes/{route_id}/status")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "route status" in data["detail"].lower()


class TestBackgroundTask:
    """Tests for the background processing task."""

    @patch("src.routes.routes.insert_record")
    def test_background_task_scheduled_on_create(
        self,
        mock_insert: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """Creating a route should schedule a background task (non-blocking)."""
        mock_insert.return_value = sample_route_record

        response = client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg"},
        )

        # Background task runs synchronously in TestClient; response still 201
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["status"] == "pending"

    @patch("src.routes.routes.insert_record")
    def test_create_route_inserts_status_pending(
        self,
        mock_insert: MagicMock,
        client: TestClient,
        sample_route_record: dict[str, Any],
    ) -> None:
        """The DB insert should include status='pending'."""
        mock_insert.return_value = sample_route_record

        client.post(
            "/api/v1/routes",
            json={"image_url": "https://example.com/image.jpg"},
        )

        call_kwargs = mock_insert.call_args.kwargs
        assert call_kwargs["data"]["status"] == "pending"
