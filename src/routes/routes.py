"""Route record management endpoints.

This module provides endpoints for creating and retrieving bouldering
route records that link uploaded images to route metadata.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, status
from pydantic import BaseModel, Field, field_validator

from src.database.supabase_client import (
    SupabaseClientError,
    insert_record,
    select_record_by_id,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["routes"])

# Constants
WALL_ANGLE_MIN = -90.0
WALL_ANGLE_MAX = 90.0
IMAGE_URL_MAX_LENGTH = 2048
_ROUTES_TABLE = "routes"


class RouteCreate(BaseModel):
    """Request model for creating a route.

    Attributes:
        image_url: Public URL of the uploaded route image.
        wall_angle: Optional wall angle in degrees (-90 to 90).
            Negative values indicate overhang, positive values indicate slab.
    """

    image_url: Annotated[
        str,
        Field(
            min_length=1,
            max_length=IMAGE_URL_MAX_LENGTH,
            description="Public URL of the uploaded route image",
            examples=[
                "https://example.supabase.co/storage/v1/object/public/route-images/2026/01/uuid.jpg"
            ],
        ),
    ]
    wall_angle: Annotated[
        float | None,
        Field(
            default=None,
            ge=WALL_ANGLE_MIN,
            le=WALL_ANGLE_MAX,
            description="Wall angle in degrees (-90 to 90). Negative=overhang, Positive=slab.",
            examples=[0.0, 15.0, -30.0],
        ),
    ]

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """Validate that image_url is a valid HTTPS URL.

        Args:
            v: The image URL to validate.

        Returns:
            The validated URL.

        Raises:
            ValueError: If URL is not a valid HTTPS URL.
        """
        if not v.startswith("https://"):
            raise ValueError("Image URL must use HTTPS scheme")
        return v


class RouteResponse(BaseModel):
    """Response model for route data.

    Attributes:
        id: Unique identifier for the route.
        image_url: Public URL of the route image.
        wall_angle: Wall angle in degrees, or None if unknown.
        created_at: ISO 8601 timestamp of creation.
        updated_at: ISO 8601 timestamp of last update.
    """

    id: str
    image_url: str
    wall_angle: float | None
    created_at: str
    updated_at: str


class ErrorResponse(BaseModel):
    """Response model for route errors.

    Attributes:
        detail: Human-readable error message.
    """

    detail: str


def _format_timestamp(value: str | None) -> str:
    """Format a timestamp value for response, converting to UTC.

    Args:
        value: Timestamp string from database. Must not be None.
            May include a UTC offset (e.g., +00:00, -05:00) or end with 'Z'.
            Naive timestamps are assumed to be UTC.

    Returns:
        UTC timestamp string in ISO 8601 format ending with 'Z'
        (e.g., '2026-01-27T12:00:00Z').

    Raises:
        ValueError: If value is None, indicating a required timestamp field is
            missing from the database record.
    """
    if value is None:
        raise ValueError("Timestamp cannot be None: required field missing from record")

    timestamp = str(value)

    # Already in UTC Z format — return as-is
    if timestamp.endswith("Z"):
        return timestamp

    # Parse and convert to UTC
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is not None:
        # Offset-aware: convert to UTC
        dt = dt.astimezone(timezone.utc)
    else:
        # Naive (no offset): assume UTC
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.isoformat().replace("+00:00", "Z")


def _record_to_response(record: dict[str, Any]) -> RouteResponse:
    """Convert a database record to a RouteResponse.

    Args:
        record: Database record dictionary.

    Returns:
        RouteResponse model instance.

    Raises:
        KeyError: If a required string field (``id``, ``image_url``) is absent.
        ValueError: If a required timestamp field (``created_at``, ``updated_at``)
            is absent or None, with the field name identified in the message.
    """
    for field in ("created_at", "updated_at"):
        if record.get(field) is None:
            raise ValueError(f"Record missing required timestamp field '{field}'")

    return RouteResponse(
        id=str(record["id"]),
        image_url=str(record["image_url"]),
        wall_angle=record.get("wall_angle"),
        created_at=_format_timestamp(record.get("created_at")),
        updated_at=_format_timestamp(record.get("updated_at")),
    )


@router.post(
    "/routes",
    response_model=RouteResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Validation error - invalid request body format",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during route creation",
        },
    },
)
async def create_route(route_data: RouteCreate) -> RouteResponse:
    """Create a new route record.

    This endpoint creates a new route record in the database, linking
    an uploaded image to route metadata like wall angle.

    Args:
        route_data: Route creation data with image_url and optional wall_angle.

    Returns:
        Created route record with generated ID and timestamps.

    Raises:
        HTTPException: 422 for validation errors, 500 for database errors.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/routes" \\
             -H "Content-Type: application/json" \\
             -d '{"image_url": "https://example.com/image.jpg", "wall_angle": 15.0}'
        ```
    """
    # Prepare data for insertion
    insert_data: dict[str, Any] = {
        "image_url": route_data.image_url,
    }

    # Only include wall_angle if provided (let DB handle NULL)
    if route_data.wall_angle is not None:
        insert_data["wall_angle"] = round(route_data.wall_angle, 1)

    try:
        # Insert record (run in thread to avoid blocking event loop)
        record = await asyncio.to_thread(
            insert_record,
            table=_ROUTES_TABLE,
            data=insert_data,
        )

        logger.info(
            "Route created successfully",
            extra={
                "route_id": record["id"],
                "image_url": route_data.image_url,
            },
        )

        return _record_to_response(record)

    except (SupabaseClientError, KeyError, ValueError) as e:
        logger.error(
            "Failed to create route record",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "image_url": route_data.image_url,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create route record",
        ) from e


@router.get(
    "/routes/{route_id}",
    response_model=RouteResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Route not found",
        },
        422: {
            "model": ErrorResponse,
            "description": "Invalid route ID format",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during retrieval",
        },
    },
)
async def get_route(
    route_id: Annotated[
        uuid.UUID,
        Path(
            description="UUID of the route to retrieve",
            examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
        ),
    ],
) -> RouteResponse:
    """Retrieve a route by ID.

    Args:
        route_id: UUID of the route to retrieve. FastAPI validates the format
            and returns 422 for malformed values before this handler is called.

    Returns:
        Route record with all fields.

    Raises:
        HTTPException: 404 if not found, 422 for invalid UUID, 500 for database errors.

    Example:
        ```bash
        curl "http://localhost:8000/api/v1/routes/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        ```
    """
    try:
        # Query record (run in thread to avoid blocking event loop)
        record = await asyncio.to_thread(
            select_record_by_id,
            table=_ROUTES_TABLE,
            record_id=str(route_id),
        )

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found",
            )

        return _record_to_response(record)

    except HTTPException:
        raise
    except (SupabaseClientError, KeyError, ValueError) as e:
        logger.error(
            "Failed to retrieve route",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "route_id": str(route_id),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve route",
        ) from e
