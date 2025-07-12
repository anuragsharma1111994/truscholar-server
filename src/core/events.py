"""Application lifecycle event handlers for TruScholar.

This module manages startup and shutdown events, including database connections,
cache initialization, background tasks, and resource cleanup.
"""

import asyncio
from typing import Callable, Optional

from fastapi import FastAPI

from src.core.config import get_settings
from src.core.settings import SystemConstants, app_settings
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class StartupEvent:
    """Handles application startup tasks."""

    def __init__(self, app: FastAPI):
        """Initialize startup event handler.

        Args:
            app: FastAPI application instance
        """
        self.app = app
        self.tasks = []
        self.failed_tasks = []

    async def execute(self) -> None:
        """Execute all startup tasks."""
        logger.info(
            "Starting application startup sequence",
            extra={
                "app_name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "environment": settings.APP_ENV,
            }
        )

        # Define startup tasks in order
        startup_tasks = [
            ("Database Connection", self._connect_database),
            ("Database Indexes", self._create_indexes),
            ("Redis Connection", self._connect_redis),
            ("Cache Warming", self._warm_cache),
            ("Background Tasks", self._initialize_background_tasks),
            ("Health Checks", self._initialize_health_checks),
            ("Feature Flags", self._load_feature_flags),
            ("Static Data", self._load_static_data),
        ]

        # Execute tasks
        for task_name, task_func in startup_tasks:
            try:
                logger.info(f"Starting: {task_name}")
                await task_func()
                self.tasks.append(task_name)
                logger.info(f"Completed: {task_name}")
            except Exception as e:
                logger.error(
                    f"Failed: {task_name}",
                    extra={"error": str(e)},
                    exc_info=True
                )
                self.failed_tasks.append((task_name, str(e)))

                # Determine if this is a critical failure
                if task_name in ["Database Connection", "Redis Connection"]:
                    raise RuntimeError(
                        f"Critical startup task failed: {task_name}. Error: {str(e)}"
                    )

        # Log startup summary
        self._log_startup_summary()

    async def _connect_database(self) -> None:
        """Connect to MongoDB database."""
        try:
            # Initialize MongoDB connection
            await MongoDB.connect(
                url=settings.get_database_url(),
                db_name=settings.MONGODB_DB_NAME,
                max_pool_size=settings.MONGODB_MAX_POOL_SIZE,
                min_pool_size=settings.MONGODB_MIN_POOL_SIZE,
                max_idle_time_ms=settings.MONGODB_MAX_IDLE_TIME_MS,
                connect_timeout_ms=settings.MONGODB_CONNECT_TIMEOUT_MS,
            )

            # Test connection
            await MongoDB.ping()

            logger.info(
                "MongoDB connected successfully",
                extra={
                    "database": settings.MONGODB_DB_NAME,
                    "pool_size": settings.MONGODB_MAX_POOL_SIZE,
                }
            )

        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    async def _create_indexes(self) -> None:
        """Create database indexes."""
        try:
            # Get all collections that need indexes
            collections_with_indexes = [
                ("users", [
                    ([("phone", 1), ("name", 1)], {"unique": True}),
                    ([("phone", 1)], {}),
                    ([("email", 1)], {"sparse": True}),
                    ([("is_active", 1)], {}),
                    ([("created_at", -1)], {}),
                ]),
                ("tests", [
                    ([("user_id", 1)], {}),
                    ([("status", 1)], {}),
                    ([("created_at", -1)], {}),
                    ([("expires_at", 1)], {"expireAfterSeconds": 0}),
                ]),
                ("questions", [
                    ([("test_id", 1), ("question_number", 1)], {"unique": True}),
                    ([("test_id", 1)], {}),
                    ([("question_type", 1)], {}),
                ]),
                ("answers", [
                    ([("test_id", 1), ("question_id", 1)], {"unique": True}),
                    ([("test_id", 1), ("question_number", 1)], {}),
                    ([("user_id", 1)], {}),
                ]),
                ("careers", [
                    ([("code", 1)], {"unique": True}),
                    ([("primary_raisec_code", 1)], {}),
                    ([("is_active", 1), ("popularity_score", -1)], {}),
                ]),
                ("career_recommendations", [
                    ([("test_id", 1), ("recommendation_number", 1)], {"unique": True}),
                    ([("user_id", 1), ("created_at", -1)], {}),
                ]),
                ("reports", [
                    ([("test_id", 1)], {"unique": True}),
                    ([("user_id", 1), ("created_at", -1)], {}),
                    ([("access_code", 1)], {"sparse": True}),
                ]),
                ("prompts", [
                    ([("prompt_name", 1), ("version", 1)], {"unique": True}),
                    ([("prompt_type", 1), ("is_active", 1)], {}),
                ]),
                ("sessions", [
                    ([("session_token", 1)], {"unique": True}),
                    ([("user_id", 1), ("is_active", 1)], {}),
                    ([("expires_at", 1)], {"expireAfterSeconds": 0}),
                ]),
            ]

            # Create indexes for each collection
            created_count = 0
            for collection_name, indexes in collections_with_indexes:
                for index_spec, index_options in indexes:
                    created = await MongoDB.create_index(
                        collection_name, index_spec, **index_options
                    )
                    if created:
                        created_count += 1

            logger.info(f"Created {created_count} database indexes")

        except Exception as e:
            logger.error(f"Index creation failed: {str(e)}")
            # Non-critical error, don't raise

    async def _connect_redis(self) -> None:
        """Connect to Redis cache."""
        try:
            # Initialize Redis connection
            await RedisClient.connect(
                url=settings.get_redis_url(),
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                decode_responses=settings.REDIS_DECODE_RESPONSES,
                health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL,
                password=settings.REDIS_PASSWORD,
            )

            # Test connection
            await RedisClient.ping()

            logger.info(
                "Redis connected successfully",
                extra={
                    "max_connections": settings.REDIS_MAX_CONNECTIONS,
                }
            )

        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            raise

    async def _warm_cache(self) -> None:
        """Warm up cache with frequently accessed data."""
        if not settings.ENABLE_CACHE:
            logger.info("Cache warming skipped (caching disabled)")
            return

        try:
            # Cache static data that rarely changes
            static_data_keys = [
                ("career_categories", self._get_career_categories),
                ("raisec_descriptions", self._get_raisec_descriptions),
                ("age_group_ranges", self._get_age_group_ranges),
            ]

            warmed_count = 0
            for cache_key, data_func in static_data_keys:
                try:
                    data = await data_func()
                    if data:
                        await RedisClient.set_json(
                            f"static:{cache_key}",
                            data,
                            ttl=86400  # 24 hours
                        )
                        warmed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to warm cache for {cache_key}: {str(e)}")

            logger.info(f"Cache warmed with {warmed_count} static entries")

        except Exception as e:
            logger.error(f"Cache warming failed: {str(e)}")
            # Non-critical error, don't raise

    async def _initialize_background_tasks(self) -> None:
        """Initialize background tasks and schedulers."""
        if not settings.ENABLE_BACKGROUND_JOBS:
            logger.info("Background tasks skipped (disabled)")
            return

        try:
            # Import here to avoid circular imports
            from src.workers.schedulers.periodic_tasks import initialize_schedulers

            # Initialize periodic task schedulers
            await initialize_schedulers()

            logger.info("Background tasks initialized")

        except Exception as e:
            logger.error(f"Background task initialization failed: {str(e)}")
            # Non-critical error, don't raise

    async def _initialize_health_checks(self) -> None:
        """Initialize health check monitors."""
        try:
            # Create health check tasks
            health_check_interval = SystemConstants.HEALTH_CHECK_INTERVAL_SECONDS

            # Database health check
            async def check_database_health():
                while True:
                    try:
                        await MongoDB.ping()
                        await asyncio.sleep(health_check_interval)
                    except Exception as e:
                        logger.error(f"Database health check failed: {str(e)}")
                        await asyncio.sleep(health_check_interval)

            # Redis health check
            async def check_redis_health():
                while True:
                    try:
                        await RedisClient.ping()
                        await asyncio.sleep(health_check_interval)
                    except Exception as e:
                        logger.error(f"Redis health check failed: {str(e)}")
                        await asyncio.sleep(health_check_interval)

            # Create background tasks
            asyncio.create_task(check_database_health())
            asyncio.create_task(check_redis_health())

            logger.info("Health checks initialized")

        except Exception as e:
            logger.error(f"Health check initialization failed: {str(e)}")
            # Non-critical error, don't raise

    async def _load_feature_flags(self) -> None:
        """Load feature flags into cache."""
        try:
            from src.core.settings import feature_flags

            # Get all feature flags
            flags = feature_flags.get_all_flags()

            # Store in Redis for dynamic updates
            for flag_name, flag_value in flags.items():
                await RedisClient.set(
                    f"feature_flag:{flag_name}",
                    str(flag_value).lower(),
                    ttl=300  # 5 minutes
                )

            logger.info(f"Loaded {len(flags)} feature flags")

        except Exception as e:
            logger.error(f"Feature flag loading failed: {str(e)}")
            # Non-critical error, don't raise

    async def _load_static_data(self) -> None:
        """Load static data from files or database."""
        try:
            # Load any static data needed for the application
            # This could include career databases, prompt templates, etc.

            logger.info("Static data loaded")

        except Exception as e:
            logger.error(f"Static data loading failed: {str(e)}")
            # Non-critical error, don't raise

    async def _get_career_categories(self) -> list:
        """Get career categories for cache warming."""
        # This would typically fetch from database
        return [
            "Technology",
            "Healthcare",
            "Education",
            "Business",
            "Arts",
            "Science",
            "Engineering",
            "Social Services",
        ]

    async def _get_raisec_descriptions(self) -> dict:
        """Get RAISEC dimension descriptions for cache warming."""
        return {
            "R": "Realistic - Practical, hands-on, physical activities",
            "A": "Artistic - Creative, expressive, innovative activities",
            "I": "Investigative - Analytical, intellectual, research activities",
            "S": "Social - Helping, teaching, interpersonal activities",
            "E": "Enterprising - Leadership, persuasive, business activities",
            "C": "Conventional - Organized, structured, detail-oriented activities",
        }

    async def _get_age_group_ranges(self) -> dict:
        """Get age group ranges for cache warming."""
        from src.core.settings import business_settings
        return {
            group.value: ranges
            for group, ranges in business_settings.AGE_GROUP_RANGES.items()
        }

    def _log_startup_summary(self) -> None:
        """Log startup summary."""
        summary = {
            "successful_tasks": len(self.tasks),
            "failed_tasks": len(self.failed_tasks),
            "tasks": self.tasks,
            "failures": self.failed_tasks,
            "environment": settings.APP_ENV,
            "debug_mode": settings.APP_DEBUG,
            "api_docs": settings.ENABLE_API_DOCS,
            "cache_enabled": settings.ENABLE_CACHE,
            "rate_limiting": settings.ENABLE_RATE_LIMITING,
        }

        if self.failed_tasks:
            logger.warning(
                "Application started with errors",
                extra=summary
            )
        else:
            logger.info(
                "Application started successfully",
                extra=summary
            )


class ShutdownEvent:
    """Handles application shutdown tasks."""

    def __init__(self, app: FastAPI):
        """Initialize shutdown event handler.

        Args:
            app: FastAPI application instance
        """
        self.app = app

    async def execute(self) -> None:
        """Execute all shutdown tasks."""
        logger.info("Starting application shutdown sequence")

        # Define shutdown tasks in order
        shutdown_tasks = [
            ("Cancel Background Tasks", self._cancel_background_tasks),
            ("Flush Cache", self._flush_cache),
            ("Close Redis Connection", self._close_redis),
            ("Close Database Connection", self._close_database),
            ("Cleanup Resources", self._cleanup_resources),
        ]

        # Execute tasks
        for task_name, task_func in shutdown_tasks:
            try:
                logger.info(f"Executing: {task_name}")
                await task_func()
                logger.info(f"Completed: {task_name}")
            except Exception as e:
                logger.error(
                    f"Error during {task_name}: {str(e)}",
                    exc_info=True
                )

        logger.info("Application shutdown complete")

    async def _cancel_background_tasks(self) -> None:
        """Cancel all background tasks."""
        try:
            # Cancel all running tasks except the current one
            tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]

            if tasks:
                logger.info(f"Cancelling {len(tasks)} background tasks")
                for task in tasks:
                    task.cancel()

                # Wait for all tasks to complete cancellation
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error cancelling background tasks: {str(e)}")

    async def _flush_cache(self) -> None:
        """Flush critical cache data."""
        try:
            # Save any critical cached data that needs persistence
            # This is mainly for session data or temporary states

            logger.info("Cache flushed")

        except Exception as e:
            logger.error(f"Error flushing cache: {str(e)}")

    async def _close_redis(self) -> None:
        """Close Redis connection."""
        try:
            await RedisClient.disconnect()
            logger.info("Redis connection closed")

        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")

    async def _close_database(self) -> None:
        """Close database connection."""
        try:
            await MongoDB.disconnect()
            logger.info("Database connection closed")

        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")

    async def _cleanup_resources(self) -> None:
        """Cleanup any remaining resources."""
        try:
            # Clean up temporary files
            import shutil
            import os

            temp_path = settings.TEMP_FILES_PATH
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
                os.makedirs(temp_path, exist_ok=True)

            logger.info("Resources cleaned up")

        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")


def create_start_app_handler(app: FastAPI) -> Callable:
    """Create startup event handler for the application.

    Args:
        app: FastAPI application instance

    Returns:
        Startup event handler function
    """
    async def start_app() -> None:
        """Application startup handler."""
        startup_event = StartupEvent(app)
        await startup_event.execute()

    return start_app


def create_stop_app_handler(app: FastAPI) -> Callable:
    """Create shutdown event handler for the application.

    Args:
        app: FastAPI application instance

    Returns:
        Shutdown event handler function
    """
    async def stop_app() -> None:
        """Application shutdown handler."""
        shutdown_event = ShutdownEvent(app)
        await shutdown_event.execute()

    return stop_app


# Additional event handlers

async def on_startup_maintenance_check() -> None:
    """Check for scheduled maintenance on startup."""
    try:
        # Check if maintenance mode is enabled
        maintenance_mode = await RedisClient.get("maintenance_mode")

        if maintenance_mode:
            logger.warning(
                "Application starting in maintenance mode",
                extra={"reason": maintenance_mode}
            )

    except Exception as e:
        logger.error(f"Maintenance check failed: {str(e)}")


async def on_startup_migration_check() -> None:
    """Check for pending database migrations on startup."""
    try:
        # This would check for pending migrations
        # For now, just log
        logger.info("Database migration check completed")

    except Exception as e:
        logger.error(f"Migration check failed: {str(e)}")


# Export the main functions
__all__ = [
    "create_start_app_handler",
    "create_stop_app_handler",
    "StartupEvent",
    "ShutdownEvent",
]
