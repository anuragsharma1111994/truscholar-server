"""Main FastAPI application module for TruScholar.

This module creates and configures the FastAPI application instance with all
necessary middleware, routers, and event handlers.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.dependencies import get_request_id
from src.api.middleware.error_handler import ErrorHandlerMiddleware
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.middleware.rate_limiter import RateLimiterMiddleware
from src.api.middleware.request_id import RequestIDMiddleware
from src.core.config import get_settings
from src.core.events import create_start_app_handler, create_stop_app_handler
from src.utils.logger import get_logger

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info(
        "Starting TruScholar API",
        extra={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
        }
    )

    # Execute startup handler
    startup_handler = create_start_app_handler(app)
    await startup_handler()

    logger.info("TruScholar API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down TruScholar API")

    # Execute shutdown handler
    shutdown_handler = create_stop_app_handler(app)
    await shutdown_handler()

    logger.info("TruScholar API shut down successfully")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Create FastAPI instance
    app = FastAPI(
        title=settings.APP_NAME,
        description="AI-powered RAISEC-based career counselling platform",
        version=settings.APP_VERSION,
        docs_url=f"{settings.API_V1_PREFIX}/docs" if settings.ENABLE_API_DOCS else None,
        redoc_url=f"{settings.API_V1_PREFIX}/redoc" if settings.ENABLE_API_DOCS else None,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json" if settings.ENABLE_API_DOCS else None,
        lifespan=lifespan,
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
        },
    )

    # Set custom exception handlers
    app = register_exception_handlers(app)

    # Add middleware
    app = register_middleware(app)

    # Include routers
    app = register_routers(app)

    # Add health check endpoints
    app = register_health_checks(app)

    # Setup metrics if enabled
    if settings.ENABLE_METRICS:
        setup_metrics(app)

    return app


def register_exception_handlers(app: FastAPI) -> FastAPI:
    """Register custom exception handlers.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI: Application with exception handlers registered
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        request_id = get_request_id()

        logger.warning(
            f"HTTP exception: {exc.detail}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "path": request.url.path,
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors."""
        request_id = get_request_id()

        logger.warning(
            "Validation error",
            extra={
                "request_id": request_id,
                "errors": exc.errors(),
                "body": exc.body,
                "path": request.url.path,
            }
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message": "Validation error",
                    "details": exc.errors(),
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle uncaught exceptions."""
        request_id = get_request_id()

        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                "request_id": request_id,
                "exception_type": type(exc).__name__,
                "path": request.url.path,
            },
            exc_info=True,
        )

        # Don't expose internal errors in production
        if settings.APP_ENV == "production":
            message = "An internal error occurred"
        else:
            message = str(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": message,
                    "request_id": request_id,
                }
            },
        )

    return app


def register_middleware(app: FastAPI) -> FastAPI:
    """Register application middleware.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI: Application with middleware registered
    """
    # Add custom middleware (order matters - executed in reverse order)

    # Request ID middleware (should be first to execute)
    app.add_middleware(RequestIDMiddleware)

    # Logging middleware
    app.add_middleware(LoggingMiddleware)

    # Error handler middleware
    app.add_middleware(ErrorHandlerMiddleware)

    # Rate limiting middleware
    if settings.ENABLE_RATE_LIMITING:
        app.add_middleware(
            RateLimiterMiddleware,
            rate_limit=settings.API_RATE_LIMIT,
            rate_limit_period=settings.API_RATE_LIMIT_PERIOD,
        )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # GZip middleware for response compression
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # Trusted host middleware for security
    if settings.APP_ENV == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # Add timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time to response headers."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        return response

    return app


def register_routers(app: FastAPI) -> FastAPI:
    """Register API routers.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI: Application with routers registered
    """
    # Import routers here to avoid circular imports
    from src.routers import auth, careers, health, questions, reports, tests, users

    # API v1 routers
    api_prefix = settings.API_V1_PREFIX

    app.include_router(
        health.router,
        prefix=f"{api_prefix}/health",
        tags=["Health"],
    )

    app.include_router(
        auth.router,
        prefix=f"{api_prefix}/auth",
        tags=["Authentication"],
    )

    app.include_router(
        users.router,
        prefix=f"{api_prefix}/users",
        tags=["Users"],
    )

    app.include_router(
        tests.router,
        prefix=f"{api_prefix}/tests",
        tags=["Tests"],
    )

    app.include_router(
        questions.router,
        prefix=f"{api_prefix}/questions",
        tags=["Questions"],
    )

    app.include_router(
        careers.router,
        prefix=f"{api_prefix}/careers",
        tags=["Careers"],
    )

    app.include_router(
        reports.router,
        prefix=f"{api_prefix}/reports",
        tags=["Reports"],
    )

    return app


def register_health_checks(app: FastAPI) -> FastAPI:
    """Register health check endpoints.

    Args:
        app: FastAPI application instance

    Returns:
        FastAPI: Application with health checks registered
    """

    @app.get(
        "/health",
        tags=["Health"],
        summary="Basic health check",
        response_model=Dict[str, Any],
    )
    async def health_check() -> Dict[str, Any]:
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.APP_ENV,
        }

    @app.get(
        "/",
        tags=["Root"],
        summary="Root endpoint",
        response_model=Dict[str, str],
    )
    async def root() -> Dict[str, str]:
        """Root endpoint with API information."""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs": f"{settings.API_V1_PREFIX}/docs",
            "health": "/health",
        }

    return app


def setup_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics.

    Args:
        app: FastAPI application instance
    """
    # Initialize Prometheus instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*health.*", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="truscholar_inprogress",
        inprogress_labels=True,
    )

    # Add custom metrics
    @instrumentator.add
    def custom_metrics(info):
        """Add custom application metrics."""
        # You can add custom metrics here
        pass

    # Instrument the app
    instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        tags=["Metrics"],
        include_in_schema=False,
    )

    logger.info("Prometheus metrics enabled at /metrics")


# Create the application instance
app = create_application()


# Additional app configuration
@app.on_event("startup")
async def log_startup_info():
    """Log application startup information."""
    logger.info(
        "Application configuration",
        extra={
            "cors_origins": settings.CORS_ORIGINS,
            "rate_limiting": settings.ENABLE_RATE_LIMITING,
            "api_docs": settings.ENABLE_API_DOCS,
            "metrics": settings.ENABLE_METRICS,
            "cache": settings.ENABLE_CACHE,
        }
    )


# Export the app instance
__all__ = ["app"]


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
        log_config=None,  # Use our custom logging
        access_log=False,  # Handled by middleware
        workers=1 if settings.APP_ENV == "development" else 4,
    )
