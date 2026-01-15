"""Health check endpoint.

Provides health status information for load balancers,
monitoring systems, and orchestration platforms.
"""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from src.config import get_settings

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
