"""Connection pool management for MongoDB and Redis.

This module provides unified connection pool management, monitoring,
and optimization for database connections in the TruScholar application.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

from src.core.config import get_settings
from src.core.settings import SystemConstants
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for a connection pool."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0

    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    last_reset: datetime = field(default_factory=datetime.utcnow)

    def add_request(self, success: bool, wait_time_ms: float) -> None:
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_wait_time_ms += wait_time_ms
        self.max_wait_time_ms = max(self.max_wait_time_ms, wait_time_ms)

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.failed_connections += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_wait_time_ms(self) -> float:
        """Calculate average wait time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_wait_time_ms / self.total_requests

    def reset(self) -> None:
        """Reset statistics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_wait_time_ms = 0.0
        self.max_wait_time_ms = 0.0
        self.last_reset = datetime.utcnow()


@dataclass
class PoolHealth:
    """Health status of a connection pool."""

    name: str
    status: str  # healthy, degraded, unhealthy
    active_connections: int
    idle_connections: int
    total_connections: int
    success_rate: float
    average_latency_ms: float
    last_error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        """Check if pool is healthy."""
        return self.status == "healthy"

    @property
    def utilization(self) -> float:
        """Calculate pool utilization percentage."""
        if self.total_connections == 0:
            return 0.0
        return (self.active_connections / self.total_connections) * 100


class ConnectionPoolManager:
    """Manages connection pools for all databases."""

    def __init__(self):
        """Initialize connection pool manager."""
        self._pools: Dict[str, Any] = {}
        self._stats: Dict[str, ConnectionStats] = defaultdict(ConnectionStats)
        self._monitors: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._check_interval = SystemConstants.HEALTH_CHECK_INTERVAL_SECONDS

    async def start(self) -> None:
        """Start connection pool monitoring."""
        async with self._lock:
            if self._running:
                logger.warning("Connection pool manager already running")
                return

            self._running = True

            # Register existing pools
            await self._register_pools()

            # Start monitors
            self._monitors["mongodb"] = asyncio.create_task(
                self._monitor_pool("mongodb", self._check_mongodb_health)
            )
            self._monitors["redis"] = asyncio.create_task(
                self._monitor_pool("redis", self._check_redis_health)
            )

            # Start stats reporter
            self._monitors["stats"] = asyncio.create_task(self._report_stats())

            logger.info("Connection pool manager started")

    async def stop(self) -> None:
        """Stop connection pool monitoring."""
        async with self._lock:
            if not self._running:
                return

            self._running = False

            # Cancel all monitors
            for name, task in self._monitors.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            self._monitors.clear()
            logger.info("Connection pool manager stopped")

    async def _register_pools(self) -> None:
        """Register existing connection pools."""
        # Register MongoDB pool
        if MongoDB._client:
            self._pools["mongodb"] = MongoDB._client
            logger.info("Registered MongoDB connection pool")

        # Register Redis pool
        if RedisClient._client:
            self._pools["redis"] = RedisClient._client
            logger.info("Registered Redis connection pool")

    async def _monitor_pool(
        self,
        pool_name: str,
        health_check_func: Any,
    ) -> None:
        """Monitor a connection pool.

        Args:
            pool_name: Name of the pool
            health_check_func: Function to check pool health
        """
        logger.info(f"Starting monitor for {pool_name} pool")

        while self._running:
            try:
                # Perform health check
                start_time = time.time()
                health = await health_check_func()
                check_time_ms = (time.time() - start_time) * 1000

                # Update stats
                stats = self._stats[pool_name]
                stats.add_request(health.is_healthy, check_time_ms)

                if not health.is_healthy:
                    stats.record_error(health.last_error or "Health check failed")
                    logger.warning(
                        f"{pool_name} pool health check failed",
                        extra={
                            "status": health.status,
                            "error": health.last_error,
                            "success_rate": health.success_rate,
                        }
                    )

                # Log pool status
                if health.utilization > 80:
                    logger.warning(
                        f"{pool_name} pool high utilization",
                        extra={
                            "utilization": health.utilization,
                            "active": health.active_connections,
                            "total": health.total_connections,
                        }
                    )

                # Sleep until next check
                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Error monitoring {pool_name} pool: {str(e)}",
                    exc_info=True
                )
                await asyncio.sleep(self._check_interval)

    async def _check_mongodb_health(self) -> PoolHealth:
        """Check MongoDB connection pool health."""
        try:
            if not MongoDB._client:
                return PoolHealth(
                    name="mongodb",
                    status="unhealthy",
                    active_connections=0,
                    idle_connections=0,
                    total_connections=0,
                    success_rate=0.0,
                    average_latency_ms=0.0,
                    last_error="Client not initialized",
                )

            # Get pool stats from MongoDB
            start_time = time.time()
            await MongoDB.ping()
            latency_ms = (time.time() - start_time) * 1000

            # Get connection pool info
            # Note: Motor doesn't expose pool stats directly, so we estimate
            stats = self._stats["mongodb"]

            return PoolHealth(
                name="mongodb",
                status="healthy",
                active_connections=0,  # Estimated
                idle_connections=settings.MONGODB_MIN_POOL_SIZE,
                total_connections=settings.MONGODB_MAX_POOL_SIZE,
                success_rate=stats.success_rate,
                average_latency_ms=latency_ms,
            )

        except Exception as e:
            return PoolHealth(
                name="mongodb",
                status="unhealthy",
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                success_rate=0.0,
                average_latency_ms=0.0,
                last_error=str(e),
            )

    async def _check_redis_health(self) -> PoolHealth:
        """Check Redis connection pool health."""
        try:
            if not RedisClient._client:
                return PoolHealth(
                    name="redis",
                    status="unhealthy",
                    active_connections=0,
                    idle_connections=0,
                    total_connections=0,
                    success_rate=0.0,
                    average_latency_ms=0.0,
                    last_error="Client not initialized",
                )

            # Get pool stats
            start_time = time.time()
            await RedisClient.ping()
            latency_ms = (time.time() - start_time) * 1000

            # Get connection pool stats
            pool = RedisClient._pool
            if pool:
                # Get pool statistics
                active = len(pool._in_use_connections)
                idle = len(pool._available_connections)
                total = pool.max_connections
            else:
                active = idle = total = 0

            stats = self._stats["redis"]

            return PoolHealth(
                name="redis",
                status="healthy",
                active_connections=active,
                idle_connections=idle,
                total_connections=total,
                success_rate=stats.success_rate,
                average_latency_ms=latency_ms,
            )

        except Exception as e:
            return PoolHealth(
                name="redis",
                status="unhealthy",
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                success_rate=0.0,
                average_latency_ms=0.0,
                last_error=str(e),
            )

    async def _report_stats(self) -> None:
        """Periodically report pool statistics."""
        report_interval = 300  # 5 minutes

        while self._running:
            try:
                await asyncio.sleep(report_interval)

                # Collect all stats
                report = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "pools": {},
                }

                for pool_name, stats in self._stats.items():
                    report["pools"][pool_name] = {
                        "total_requests": stats.total_requests,
                        "success_rate": stats.success_rate,
                        "average_wait_time_ms": stats.average_wait_time_ms,
                        "max_wait_time_ms": stats.max_wait_time_ms,
                        "failed_connections": stats.failed_connections,
                        "last_error": stats.last_error,
                        "uptime_minutes": (
                            datetime.utcnow() - stats.created_at
                        ).total_seconds() / 60,
                    }

                logger.info(
                    "Connection pool statistics",
                    extra={"report": report}
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reporting stats: {str(e)}", exc_info=True)

    async def get_pool_health(self, pool_name: str) -> Optional[PoolHealth]:
        """Get current health status of a pool.

        Args:
            pool_name: Name of the pool

        Returns:
            PoolHealth or None
        """
        if pool_name == "mongodb":
            return await self._check_mongodb_health()
        elif pool_name == "redis":
            return await self._check_redis_health()
        return None

    async def get_all_pool_health(self) -> Dict[str, PoolHealth]:
        """Get health status of all pools.

        Returns:
            Dictionary of pool health statuses
        """
        health_status = {}

        for pool_name in ["mongodb", "redis"]:
            health = await self.get_pool_health(pool_name)
            if health:
                health_status[pool_name] = health

        return health_status

    def get_pool_stats(self, pool_name: str) -> Optional[ConnectionStats]:
        """Get statistics for a pool.

        Args:
            pool_name: Name of the pool

        Returns:
            ConnectionStats or None
        """
        return self._stats.get(pool_name)

    def get_all_pool_stats(self) -> Dict[str, ConnectionStats]:
        """Get statistics for all pools.

        Returns:
            Dictionary of pool statistics
        """
        return dict(self._stats)

    def reset_pool_stats(self, pool_name: Optional[str] = None) -> None:
        """Reset statistics for a pool or all pools.

        Args:
            pool_name: Name of the pool to reset, or None for all
        """
        if pool_name:
            if pool_name in self._stats:
                self._stats[pool_name].reset()
                logger.info(f"Reset stats for {pool_name} pool")
        else:
            for stats in self._stats.values():
                stats.reset()
            logger.info("Reset stats for all pools")

    async def optimize_pools(self) -> Dict[str, Any]:
        """Optimize connection pool settings based on usage.

        Returns:
            Optimization recommendations
        """
        recommendations = {}

        for pool_name, stats in self._stats.items():
            health = await self.get_pool_health(pool_name)
            if not health:
                continue

            pool_recommendations = {
                "current_settings": {
                    "total_connections": health.total_connections,
                    "utilization": health.utilization,
                },
                "recommendations": [],
            }

            # Check utilization
            if health.utilization > 80:
                pool_recommendations["recommendations"].append({
                    "type": "increase_pool_size",
                    "reason": f"High utilization: {health.utilization:.1f}%",
                    "suggestion": "Consider increasing max_connections",
                })
            elif health.utilization < 20 and health.total_connections > 10:
                pool_recommendations["recommendations"].append({
                    "type": "decrease_pool_size",
                    "reason": f"Low utilization: {health.utilization:.1f}%",
                    "suggestion": "Consider decreasing max_connections",
                })

            # Check latency
            if stats.average_wait_time_ms > 100:
                pool_recommendations["recommendations"].append({
                    "type": "performance_issue",
                    "reason": f"High average latency: {stats.average_wait_time_ms:.1f}ms",
                    "suggestion": "Check database performance or increase pool size",
                })

            # Check error rate
            error_rate = 100 - stats.success_rate
            if error_rate > 5:
                pool_recommendations["recommendations"].append({
                    "type": "high_error_rate",
                    "reason": f"Error rate: {error_rate:.1f}%",
                    "suggestion": "Investigate connection errors",
                    "last_error": stats.last_error,
                })

            recommendations[pool_name] = pool_recommendations

        return recommendations


# Global connection pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None


async def get_pool_manager() -> ConnectionPoolManager:
    """Get or create the global connection pool manager.

    Returns:
        ConnectionPoolManager instance
    """
    global _pool_manager

    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
        await _pool_manager.start()

    return _pool_manager


async def get_connection_health() -> Dict[str, Any]:
    """Get health status of all database connections.

    Returns:
        Health status dictionary
    """
    manager = await get_pool_manager()

    health_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "pools": {},
        "overall_status": "healthy",
    }

    pool_health = await manager.get_all_pool_health()

    for pool_name, health in pool_health.items():
        health_data["pools"][pool_name] = {
            "status": health.status,
            "active_connections": health.active_connections,
            "total_connections": health.total_connections,
            "utilization": f"{health.utilization:.1f}%",
            "success_rate": f"{health.success_rate:.1f}%",
            "average_latency_ms": health.average_latency_ms,
        }

        if health.status != "healthy":
            health_data["overall_status"] = "degraded"

    return health_data


async def optimize_connection_pools() -> Dict[str, Any]:
    """Get optimization recommendations for connection pools.

    Returns:
        Optimization recommendations
    """
    manager = await get_pool_manager()
    return await manager.optimize_pools()


# Connection pool lifecycle functions

async def initialize_connection_pools() -> None:
    """Initialize all connection pools."""
    try:
        # Initialize MongoDB
        await MongoDB.connect()

        # Initialize Redis
        await RedisClient.connect()

        # Start pool manager
        await get_pool_manager()

        logger.info("All connection pools initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize connection pools: {str(e)}", exc_info=True)
        raise


async def shutdown_connection_pools() -> None:
    """Shutdown all connection pools."""
    try:
        # Stop pool manager
        if _pool_manager:
            await _pool_manager.stop()

        # Disconnect databases
        await MongoDB.disconnect()
        await RedisClient.disconnect()

        logger.info("All connection pools shut down successfully")

    except Exception as e:
        logger.error(f"Error shutting down connection pools: {str(e)}", exc_info=True)


# Export main functions
__all__ = [
    "ConnectionPoolManager",
    "ConnectionStats",
    "PoolHealth",
    "get_pool_manager",
    "get_connection_health",
    "optimize_connection_pools",
    "initialize_connection_pools",
    "shutdown_connection_pools",
]
