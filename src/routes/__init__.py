"""API routes package.

This package contains all FastAPI route modules organized by domain.
"""

from src.routes.health import router as health_router

__all__ = ["health_router"]
