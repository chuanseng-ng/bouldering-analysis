"""API routes package.

This package contains all FastAPI route modules organized by domain.
"""

from src.routes.health import router as health_router
from src.routes.routes import router as routes_router
from src.routes.upload import router as upload_router

__all__ = ["health_router", "routes_router", "upload_router"]
