"""Shared Pydantic models used across multiple route modules."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response model for all API endpoints.

    Attributes:
        detail: Human-readable error message.
        error_code: Optional machine-readable error code.
    """

    detail: str
    error_code: str | None = None
