"""Common dependencies for FastAPI routes.

This module provides reusable dependencies for authentication, database access,
caching, and other common functionality across API endpoints.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis import Redis

from src.core.config import get_settings
from src.database.mongodb import get_database
from src.database.redis_client import get_redis_client
from src.models.user import User
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


# Request context dependencies
def get_request_id(request: Request) -> str:
    """Get request ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string
    """
    return getattr(request.state, "request_id", "unknown")


def get_client_ip(request: Request) -> str:
    """Get client IP address from request.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address
    """
    # Check for X-Forwarded-For header (when behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Get the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    # Check for X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client IP
    return request.client.host if request.client else "unknown"


def get_user_agent(request: Request) -> str:
    """Get user agent from request headers.

    Args:
        request: FastAPI request object

    Returns:
        User agent string
    """
    return request.headers.get("User-Agent", "unknown")


# Database dependencies
async def get_db() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance.

    Yields:
        AsyncIOMotorDatabase: MongoDB database instance
    """
    db = await get_database()
    if not db:
        logger.error("Failed to get database connection")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection unavailable"
        )
    return db


async def get_redis() -> Redis:
    """Get Redis client instance.

    Yields:
        Redis: Redis client instance
    """
    redis = await get_redis_client()
    if not redis:
        logger.error("Failed to get Redis connection")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service unavailable"
        )
    return redis


async def get_cache():
    """Get cache client instance (alias for get_redis).

    Yields:
        Redis: Redis client instance
    """
    return await get_redis()


# Authentication dependencies
async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token (optional).

    Args:
        request: FastAPI request object
        credentials: JWT credentials from Authorization header
        db: Database instance

    Returns:
        User data dict or None if not authenticated
    """
    if not credentials:
        return None

    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        # Get user ID from token
        user_id: str = payload.get("sub")
        if not user_id:
            return None

        # Verify token type
        token_type: str = payload.get("type", "access")
        if token_type != "access":
            return None

        # Check token expiration (handled by jwt.decode, but we can add custom logic)
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            return None

        # Get user from database
        user_data = await db.users.find_one({"_id": user_id})
        if not user_data:
            return None

        # Check if user is active
        if not user_data.get("is_active", True):
            return None

        # Add token data to user dict
        user_data["token_data"] = {
            "jti": payload.get("jti"),
            "iat": payload.get("iat"),
            "exp": payload.get("exp"),
        }

        return user_data

    except JWTError as e:
        logger.warning(
            f"JWT validation error: {str(e)}",
            extra={
                "request_id": get_request_id(request),
                "ip_address": get_client_ip(request),
            }
        )
        return None
    except Exception as e:
        logger.error(
            f"Error getting current user: {str(e)}",
            extra={"request_id": get_request_id(request)},
            exc_info=True
        )
        return None


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> Dict[str, Any]:
    """Get current user from JWT token (required).

    Args:
        request: FastAPI request object
        credentials: JWT credentials from Authorization header
        db: Database instance

    Returns:
        User data dict

    Raises:
        HTTPException: If user is not authenticated
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user_optional(request, credentials, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current active user.

    Args:
        current_user: Current user data

    Returns:
        User data dict

    Raises:
        HTTPException: If user is not active
    """
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )

    return current_user


async def get_current_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get current admin user.

    Args:
        current_user: Current user data

    Returns:
        User data dict

    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    return current_user


# Rate limiting dependencies
async def check_rate_limit(
    request: Request,
    redis: Redis = Depends(get_redis),
    user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> None:
    """Check rate limit for the current request.

    Args:
        request: FastAPI request object
        redis: Redis client
        user: Optional user data

    Raises:
        HTTPException: If rate limit is exceeded
    """
    if not settings.ENABLE_RATE_LIMITING:
        return

    # Determine rate limit key
    if user:
        # User-specific rate limit
        key = f"rate_limit:user:{user['_id']}:{request.url.path}"
        limit = settings.API_RATE_LIMIT_USER
    else:
        # IP-based rate limit for anonymous users
        ip = get_client_ip(request)
        key = f"rate_limit:ip:{ip}:{request.url.path}"
        limit = settings.API_RATE_LIMIT_ANONYMOUS

    try:
        # Increment counter
        current = await redis.incr(key)

        # Set expiry on first request
        if current == 1:
            await redis.expire(key, settings.API_RATE_LIMIT_PERIOD)

        # Check limit
        if current > limit:
            logger.warning(
                f"Rate limit exceeded",
                extra={
                    "request_id": get_request_id(request),
                    "key": key,
                    "limit": limit,
                    "current": current,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )

    except HTTPException:
        raise
    except Exception as e:
        # Don't fail the request if rate limiting fails
        logger.error(
            f"Rate limit check failed: {str(e)}",
            extra={"request_id": get_request_id(request)},
            exc_info=True
        )


# Cache dependencies
class CacheKeyBuilder:
    """Helper class for building cache keys."""

    @staticmethod
    def build(*parts: Any) -> str:
        """Build a cache key from parts.

        Args:
            *parts: Parts to join into a cache key

        Returns:
            Cache key string
        """
        return ":".join(str(part) for part in parts)

    @staticmethod
    def user_key(user_id: str, *parts: Any) -> str:
        """Build a user-specific cache key.

        Args:
            user_id: User ID
            *parts: Additional key parts

        Returns:
            Cache key string
        """
        return CacheKeyBuilder.build("user", user_id, *parts)

    @staticmethod
    def test_key(test_id: str, *parts: Any) -> str:
        """Build a test-specific cache key.

        Args:
            test_id: Test ID
            *parts: Additional key parts

        Returns:
            Cache key string
        """
        return CacheKeyBuilder.build("test", test_id, *parts)


async def get_cached_data(
    key: str,
    redis: Redis = Depends(get_redis),
) -> Optional[Any]:
    """Get cached data by key.

    Args:
        key: Cache key
        redis: Redis client

    Returns:
        Cached data or None
    """
    if not settings.ENABLE_CACHE:
        return None

    try:
        data = await redis.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(
            f"Cache get error: {str(e)}",
            extra={"key": key},
            exc_info=True
        )
        return None


async def set_cached_data(
    key: str,
    data: Any,
    ttl: Optional[int] = None,
    redis: Redis = Depends(get_redis),
) -> bool:
    """Set cached data with optional TTL.

    Args:
        key: Cache key
        data: Data to cache
        ttl: Time to live in seconds
        redis: Redis client

    Returns:
        Success boolean
    """
    if not settings.ENABLE_CACHE:
        return False

    try:
        serialized = json.dumps(data)
        if ttl:
            await redis.setex(key, ttl, serialized)
        else:
            await redis.set(key, serialized)
        return True
    except Exception as e:
        logger.error(
            f"Cache set error: {str(e)}",
            extra={"key": key},
            exc_info=True
        )
        return False


# Pagination dependencies
class PaginationParams:
    """Common pagination parameters."""

    def __init__(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "created_at",
        sort_order: int = -1,
    ):
        """Initialize pagination parameters.

        Args:
            page: Page number (1-based)
            limit: Items per page
            sort_by: Field to sort by
            sort_order: Sort order (-1 for desc, 1 for asc)
        """
        self.page = max(1, page)
        self.limit = min(max(1, limit), 100)  # Max 100 items per page
        self.sort_by = sort_by
        self.sort_order = sort_order if sort_order in [-1, 1] else -1

    @property
    def skip(self) -> int:
        """Calculate skip value for database query."""
        return (self.page - 1) * self.limit

    def get_sort_list(self) -> list:
        """Get sort list for MongoDB query."""
        return [(self.sort_by, self.sort_order)]


# Request context dependencies
class RequestContext:
    """Container for request context data."""

    def __init__(
        self,
        request: Request,
        request_id: str = Depends(get_request_id),
        client_ip: str = Depends(get_client_ip),
        user_agent: str = Depends(get_user_agent),
        user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    ):
        """Initialize request context.

        Args:
            request: FastAPI request object
            request_id: Request ID
            client_ip: Client IP address
            user_agent: User agent string
            user: Optional user data
        """
        self.request = request
        self.request_id = request_id
        self.client_ip = client_ip
        self.user_agent = user_agent
        self.user = user
        self.user_id = user.get("_id") if user else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "user_id": self.user_id,
            "path": self.request.url.path,
            "method": self.request.method,
        }


# Feature flag dependencies
async def check_feature_flag(
    flag_name: str,
    default: bool = False,
    redis: Redis = Depends(get_redis),
) -> bool:
    """Check if a feature flag is enabled.

    Args:
        flag_name: Name of the feature flag
        default: Default value if flag is not set
        redis: Redis client

    Returns:
        Boolean indicating if feature is enabled
    """
    try:
        # Check Redis for dynamic feature flags
        key = f"feature_flag:{flag_name}"
        value = await redis.get(key)

        if value is not None:
            return value.lower() in ["true", "1", "yes", "on"]

        # Fall back to settings
        return getattr(settings, f"FEATURE_{flag_name.upper()}", default)

    except Exception as e:
        logger.error(
            f"Feature flag check failed: {str(e)}",
            extra={"flag": flag_name},
            exc_info=True
        )
        return default


# Dependency injection helpers
def get_service(service_class: type):
    """Factory function for service dependencies.

    Args:
        service_class: Service class to instantiate

    Returns:
        Dependency function that returns service instance
    """
    async def _get_service(
        db: AsyncIOMotorDatabase = Depends(get_db),
        redis: Redis = Depends(get_redis),
    ):
        return service_class(db=db, redis=redis)

    return _get_service
