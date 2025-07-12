"""Health check routes for TruScholar API.

This module provides health check endpoints for monitoring the application
and its dependencies.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_db, get_redis
from src.core.config import get_settings
from src.utils.datetime_utils import utc_now
from src.utils.logger import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)
settings = get_settings()


class HealthStatus(BaseModel):
    """Health check status response."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, Dict[str, Any]]


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Basic health check endpoint.
    
    Returns:
        HealthStatus: Application health status
    """
    return HealthStatus(
        status="healthy",
        timestamp=utc_now(),
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        services={}
    )


@router.get("/health/detailed", response_model=HealthStatus)
async def detailed_health_check(
    db=Depends(get_db),
    redis=Depends(get_redis)
) -> HealthStatus:
    """Detailed health check including all services.
    
    Returns:
        HealthStatus: Detailed health status with service checks
    """
    services = {}
    overall_status = "healthy"
    
    # Check database
    try:
        # Simple ping to check connection
        await db.command("ping")
        services["database"] = {
            "status": "healthy",
            "type": "mongodb",
            "response_time_ms": 0  # Could implement actual timing
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        services["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"
    
    # Check Redis
    try:
        await redis.ping()
        services["cache"] = {
            "status": "healthy",
            "type": "redis",
            "response_time_ms": 0  # Could implement actual timing
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        services["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
    
    # Add application info
    services["application"] = {
        "status": "healthy",
        "uptime_seconds": 0,  # Could track actual uptime
        "memory_usage_mb": 0,  # Could get actual memory usage
        "cpu_usage_percent": 0  # Could get actual CPU usage
    }
    
    return HealthStatus(
        status=overall_status,
        timestamp=utc_now(),
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        services=services
    )


@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint.
    
    Returns:
        Dict: Simple status response
    """
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness_probe(
    db=Depends(get_db),
    redis=Depends(get_redis)
) -> Dict[str, str]:
    """Kubernetes readiness probe endpoint.
    
    Returns:
        Dict: Ready status if all services are available
    """
    try:
        # Check critical services
        await db.command("ping")
        await redis.ping()
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}")
        return {"status": "not_ready", "error": str(e)}