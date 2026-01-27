"""FastAPI application factory.

This module provides the application factory pattern for creating
configured FastAPI instances with all middleware and routes.
"""

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings, get_settings, get_settings_override
from src.logging_config import configure_logging, get_logger
from src.routes import health_router, routes_router, upload_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown events for the application.
    Use this for initializing and cleaning up resources like
    database connections, ML models, etc.

    Args:
        app: The FastAPI application instance.

    Yields:
        None during application lifetime.
    """
    # Startup
    logger.info(
        "Application starting",
        extra={
            "app_name": app.state.settings.app_name,
            "version": app.state.settings.app_version,
        },
    )
    yield
    # Shutdown
    logger.info("Application shutting down")


def create_app(config_override: dict[str, Any] | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    This factory function creates a new FastAPI instance with all
    middleware, routes, and configuration applied. Supports
    configuration override for testing.

    Args:
        config_override: Optional dictionary to override default
            configuration values. Used primarily for testing.

    Returns:
        Configured FastAPI application instance ready to serve requests.

    Example:
        >>> app = create_app()
        >>> # For testing with custom config
        >>> test_app = create_app({"debug": True, "testing": True})
    """
    # Load settings
    if config_override:
        settings = get_settings_override(config_override)
    else:
        settings = get_settings()

    # Configure logging
    json_output = not settings.debug
    configure_logging(settings.log_level, json_output=json_output)

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Bouldering route analysis and grade prediction API",
        docs_url="/docs" if settings.debug or settings.testing else None,
        redoc_url="/redoc" if settings.debug or settings.testing else None,
        openapi_url="/openapi.json" if settings.debug or settings.testing else None,
        lifespan=lifespan,
    )

    # Store settings in app state for access in routes
    app.state.settings = settings

    # Add middleware
    _configure_middleware(app, settings)

    # Register routes
    _register_routes(app)

    return app


def _configure_middleware(app: FastAPI, settings: Settings) -> None:
    """Configure application middleware.

    Args:
        app: FastAPI application instance.
        settings: Application settings.
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add unique request ID to each request for tracing.

        If the request includes an X-Request-ID header, it will be
        preserved. Otherwise, a new UUID is generated.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response with X-Request-ID header.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


def _register_routes(app: FastAPI) -> None:
    """Register all API routes.

    Args:
        app: FastAPI application instance.
    """
    # Root-level health check
    app.include_router(health_router, tags=["health"])

    # Versioned API routes
    app.include_router(health_router, prefix="/api/v1", tags=["health-v1"])
    app.include_router(upload_router)
    app.include_router(routes_router)


# Create default app instance for uvicorn
# Usage: uvicorn src.app:application --reload
# Or with factory: uvicorn src.app:create_app --factory --reload
application = create_app()
