"""FastAPI application factory.

This module provides the application factory pattern for creating
configured FastAPI instances with all middleware and routes.
"""

import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import Settings, get_settings, get_settings_override
from src.logging_config import configure_logging, get_logger
from src.routes import health_router, routes_router, upload_router

logger = get_logger(__name__)

# Health-check paths that bypass API key authentication
_HEALTH_PATHS = {"/health", "/api/v1/health"}


class _UploadRateLimiter:
    """Thread-safe sliding-window rate limiter for upload requests.

    Tracks request timestamps per client IP in a 60-second window.
    A fresh instance is created for each application instance so that
    test isolation is guaranteed when :func:`create_app` is called repeatedly.
    """

    _WINDOW_SECONDS: int = 60

    def __init__(self) -> None:
        self._timestamps: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str, max_requests: int) -> bool:
        """Return True if the request is within the rate limit.

        Args:
            client_ip: Identifier for the client (typically remote IP).
            max_requests: Maximum allowed requests per window.

        Returns:
            True if the request is allowed; False if the limit is exceeded.
        """
        now = time.monotonic()
        cutoff = now - self._WINDOW_SECONDS
        with self._lock:
            timestamps = [t for t in self._timestamps.get(client_ip, []) if t > cutoff]
            if len(timestamps) >= max_requests:
                if timestamps:
                    self._timestamps[client_ip] = timestamps
                elif client_ip in self._timestamps:
                    del self._timestamps[client_ip]
                return False
            timestamps.append(now)
            self._timestamps[client_ip] = timestamps
            return True


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

    # API key authentication middleware
    @app.middleware("http")
    async def api_key_auth(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Enforce API key authentication on non-health endpoints.

        Skips authentication when ``settings.api_key`` is empty (open mode)
        or when the request targets a health-check path.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response from the next handler, or a 401 JSON response.
        """
        path = request.url.path
        is_health = any(path == p or path.startswith(p + "/") for p in _HEALTH_PATHS)
        if settings.api_key and not is_health:
            provided_key = request.headers.get("X-API-Key", "")
            if provided_key != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)

    # Upload rate-limit middleware (per-IP sliding window, 60 s)
    _upload_rate_limiter = _UploadRateLimiter()
    _upload_path = "/api/v1/routes/upload"

    @app.middleware("http")
    async def rate_limit_upload(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Rate-limit POST requests to the upload endpoint.

        Uses a per-IP sliding window counter.  Disabled when
        ``settings.rate_limit_upload`` is 0.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            Response from the next handler, or a 429 JSON response.
        """
        max_requests = settings.rate_limit_upload
        if (
            max_requests > 0
            and request.method == "POST"
            and request.url.path == _upload_path
        ):
            client_ip = request.client.host if request.client else "unknown"
            if not _upload_rate_limiter.is_allowed(client_ip, max_requests):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please try again later."},
                )
        return await call_next(request)


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
