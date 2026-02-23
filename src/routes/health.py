"""Health check endpoint.

Provides health status information for load balancers,
monitoring systems, and orchestration platforms.
"""

import asyncio
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from src.config import get_settings
from src.database.supabase_client import get_supabase_client

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Current health status of the application.
        version: Application version string.
        timestamp: ISO 8601 timestamp of the health check.
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2026-01-14T12:00:00Z",
            }
        }
    )


class DbHealthResponse(BaseModel):
    """Deep health check response model including database connectivity.

    Attributes:
        status: Current database health status.
        version: Application version string.
        timestamp: ISO 8601 timestamp of the health check.
    """

    status: Literal["healthy", "degraded"]
    version: str
    timestamp: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2026-01-14T12:00:00Z",
            }
        }
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the current health status of the application.",
)
async def health_check() -> HealthResponse:
    """Check application health status.

    Returns the current health status of the application including
    version information and timestamp. This endpoint is used by
    load balancers and monitoring systems to verify the application
    is running correctly.

    Returns:
        HealthResponse with status, version, and UTC timestamp.
    """
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/health/db",
    response_model=DbHealthResponse,
    summary="Database Health Check",
    description=(
        "Verifies Supabase connectivity by performing a lightweight query. "
        "Returns 'degraded' if the database is unreachable."
    ),
)
async def db_health_check() -> DbHealthResponse:
    """Check database connectivity status.

    Attempts a lightweight Supabase query to verify the connection is live.
    This endpoint is intended for monitoring systems that need to distinguish
    application health from database health.

    Returns:
        DbHealthResponse with status ``"healthy"`` if Supabase is reachable,
        or ``"degraded"`` if the connection fails.
    """
    settings = get_settings()
    db_status: Literal["healthy", "degraded"] = "healthy"

    try:
        client = await asyncio.to_thread(get_supabase_client)
        await asyncio.to_thread(
            lambda: client.table("routes").select("id").limit(1).execute()
        )
    except Exception:  # pylint: disable=broad-except
        db_status = "degraded"

    return DbHealthResponse(
        status=db_status,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
    )
