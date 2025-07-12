"""Redis client and cache management for TruScholar.

This module provides Redis connection management, caching operations,
and distributed locking functionality for the application.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    RedisError,
    ResponseError,
    TimeoutError as RedisTimeoutError,
)

from src.core.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class RedisClient:
    """Redis client manager with connection pooling and operations."""

    _pool: Optional[ConnectionPool] = None
    _client: Optional[redis.Redis] = None
    _initialized: bool = False
    _lock: asyncio.Lock = asyncio.Lock()
    _pubsub_clients: Dict[str, redis.client.PubSub] = {}

    @classmethod
    async def connect(
        cls,
        url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Connect to Redis with connection pooling.

        Args:
            url: Redis connection URL
            **kwargs: Additional connection parameters
        """
        async with cls._lock:
            if cls._initialized:
                logger.warning("Redis already connected")
                return

            try:
                # Use provided URL or from settings
                connection_url = url or settings.get_redis_url()

                # Connection pool parameters
                pool_kwargs = {
                    "max_connections": kwargs.get("max_connections", settings.REDIS_MAX_CONNECTIONS),
                    "decode_responses": kwargs.get("decode_responses", settings.REDIS_DECODE_RESPONSES),
                    "encoding": kwargs.get("encoding", "utf-8"),
                    "health_check_interval": kwargs.get(
                        "health_check_interval",
                        settings.REDIS_HEALTH_CHECK_INTERVAL
                    ),
                    "socket_keepalive": kwargs.get("socket_keepalive", True),
                    "socket_connect_timeout": kwargs.get("socket_connect_timeout", 5),
                    "retry_on_timeout": kwargs.get("retry_on_timeout", True),
                    "retry_on_error": kwargs.get("retry_on_error", [RedisConnectionError, RedisTimeoutError]),
                }

                # Add password if provided
                if settings.REDIS_PASSWORD:
                    pool_kwargs["password"] = settings.REDIS_PASSWORD

                # Create connection pool
                cls._pool = ConnectionPool.from_url(connection_url, **pool_kwargs)

                # Create Redis client
                cls._client = redis.Redis(connection_pool=cls._pool)

                # Test connection
                await cls._client.ping()

                cls._initialized = True
                logger.info(
                    "Redis connected successfully",
                    extra={
                        "max_connections": pool_kwargs["max_connections"],
                        "decode_responses": pool_kwargs["decode_responses"],
                    }
                )

            except Exception as e:
                cls._pool = None
                cls._client = None
                cls._initialized = False
                logger.error(f"Redis connection failed: {str(e)}", exc_info=True)
                raise

    @classmethod
    async def disconnect(cls) -> None:
        """Disconnect from Redis and cleanup resources."""
        async with cls._lock:
            if cls._client:
                try:
                    # Close all pubsub connections
                    for channel, pubsub in cls._pubsub_clients.items():
                        await pubsub.unsubscribe()
                        await pubsub.close()
                    cls._pubsub_clients.clear()

                    # Close main client
                    await cls._client.close()

                    # Close connection pool
                    if cls._pool:
                        await cls._pool.disconnect()

                    cls._client = None
                    cls._pool = None
                    cls._initialized = False

                    logger.info("Redis disconnected successfully")

                except Exception as e:
                    logger.error(f"Error disconnecting from Redis: {str(e)}", exc_info=True)

    @classmethod
    async def ping(cls) -> bool:
        """Check if Redis connection is alive.

        Returns:
            bool: True if connection is alive
        """
        if not cls._initialized or not cls._client:
            return False

        try:
            response = await cls._client.ping()
            return response is True
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False

    @classmethod
    def get_client(cls) -> Optional[redis.Redis]:
        """Get the Redis client instance.

        Returns:
            Redis client or None
        """
        return cls._client

    # Basic Operations

    @classmethod
    async def get(cls, key: str) -> Optional[str]:
        """Get value by key.

        Args:
            key: Cache key

        Returns:
            Value or None if not found
        """
        if not cls._client:
            logger.error("Redis client not initialized")
            return None

        try:
            value = await cls._client.get(key)
            return value
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {str(e)}")
            return None

    @classmethod
    async def set(
        cls,
        key: str,
        value: Union[str, int, float],
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set key-value pair with optional TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            bool: Success status
        """
        if not cls._client:
            logger.error("Redis client not initialized")
            return False

        try:
            result = await cls._client.set(
                key,
                value,
                ex=ttl,
                nx=nx,
                xx=xx,
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {str(e)}")
            return False

    @classmethod
    async def delete(cls, *keys: str) -> int:
        """Delete one or more keys.

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        if not cls._client:
            logger.error("Redis client not initialized")
            return 0

        try:
            result = await cls._client.delete(*keys)
            return result
        except Exception as e:
            logger.error(f"Redis DELETE error: {str(e)}")
            return 0

    @classmethod
    async def exists(cls, *keys: str) -> int:
        """Check if keys exist.

        Args:
            *keys: Keys to check

        Returns:
            Number of keys that exist
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.exists(*keys)
            return result
        except Exception as e:
            logger.error(f"Redis EXISTS error: {str(e)}")
            return 0

    @classmethod
    async def expire(cls, key: str, seconds: int) -> bool:
        """Set expiration time for a key.

        Args:
            key: Cache key
            seconds: Expiration time in seconds

        Returns:
            bool: Success status
        """
        if not cls._client:
            return False

        try:
            result = await cls._client.expire(key, seconds)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {str(e)}")
            return False

    @classmethod
    async def ttl(cls, key: str) -> int:
        """Get time to live for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no expiry, -2 if not exists
        """
        if not cls._client:
            return -2

        try:
            result = await cls._client.ttl(key)
            return result
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {str(e)}")
            return -2

    # JSON Operations

    @classmethod
    async def get_json(cls, key: str) -> Optional[Any]:
        """Get and deserialize JSON value.

        Args:
            key: Cache key

        Returns:
            Deserialized value or None
        """
        value = await cls.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for key {key}: {str(e)}")
        return None

    @classmethod
    async def set_json(
        cls,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Serialize and set JSON value.

        Args:
            key: Cache key
            value: Value to serialize and store
            ttl: Time to live in seconds

        Returns:
            bool: Success status
        """
        try:
            json_value = json.dumps(value, default=str)
            return await cls.set(key, json_value, ttl)
        except Exception as e:
            logger.error(f"JSON encode error for key {key}: {str(e)}")
            return False

    # Counter Operations

    @classmethod
    async def incr(cls, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter.

        Args:
            key: Counter key
            amount: Increment amount

        Returns:
            New counter value or None
        """
        if not cls._client:
            return None

        try:
            result = await cls._client.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Redis INCR error for key {key}: {str(e)}")
            return None

    @classmethod
    async def decr(cls, key: str, amount: int = 1) -> Optional[int]:
        """Decrement counter.

        Args:
            key: Counter key
            amount: Decrement amount

        Returns:
            New counter value or None
        """
        if not cls._client:
            return None

        try:
            result = await cls._client.decrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Redis DECR error for key {key}: {str(e)}")
            return None

    # Set Operations

    @classmethod
    async def sadd(cls, key: str, *members: str) -> int:
        """Add members to a set.

        Args:
            key: Set key
            *members: Members to add

        Returns:
            Number of members added
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.sadd(key, *members)
            return result
        except Exception as e:
            logger.error(f"Redis SADD error for key {key}: {str(e)}")
            return 0

    @classmethod
    async def srem(cls, key: str, *members: str) -> int:
        """Remove members from a set.

        Args:
            key: Set key
            *members: Members to remove

        Returns:
            Number of members removed
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.srem(key, *members)
            return result
        except Exception as e:
            logger.error(f"Redis SREM error for key {key}: {str(e)}")
            return 0

    @classmethod
    async def smembers(cls, key: str) -> Set[str]:
        """Get all members of a set.

        Args:
            key: Set key

        Returns:
            Set of members
        """
        if not cls._client:
            return set()

        try:
            result = await cls._client.smembers(key)
            return result
        except Exception as e:
            logger.error(f"Redis SMEMBERS error for key {key}: {str(e)}")
            return set()

    @classmethod
    async def sismember(cls, key: str, member: str) -> bool:
        """Check if member exists in set.

        Args:
            key: Set key
            member: Member to check

        Returns:
            bool: True if member exists
        """
        if not cls._client:
            return False

        try:
            result = await cls._client.sismember(key, member)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis SISMEMBER error for key {key}: {str(e)}")
            return False

    # Hash Operations

    @classmethod
    async def hset(
        cls,
        key: str,
        field: str,
        value: Union[str, int, float],
    ) -> int:
        """Set hash field value.

        Args:
            key: Hash key
            field: Field name
            value: Field value

        Returns:
            Number of fields added
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.hset(key, field, value)
            return result
        except Exception as e:
            logger.error(f"Redis HSET error for key {key}: {str(e)}")
            return 0

    @classmethod
    async def hget(cls, key: str, field: str) -> Optional[str]:
        """Get hash field value.

        Args:
            key: Hash key
            field: Field name

        Returns:
            Field value or None
        """
        if not cls._client:
            return None

        try:
            result = await cls._client.hget(key, field)
            return result
        except Exception as e:
            logger.error(f"Redis HGET error for key {key}: {str(e)}")
            return None

    @classmethod
    async def hgetall(cls, key: str) -> Dict[str, str]:
        """Get all hash fields and values.

        Args:
            key: Hash key

        Returns:
            Dictionary of field-value pairs
        """
        if not cls._client:
            return {}

        try:
            result = await cls._client.hgetall(key)
            return result
        except Exception as e:
            logger.error(f"Redis HGETALL error for key {key}: {str(e)}")
            return {}

    @classmethod
    async def hdel(cls, key: str, *fields: str) -> int:
        """Delete hash fields.

        Args:
            key: Hash key
            *fields: Fields to delete

        Returns:
            Number of fields deleted
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.hdel(key, *fields)
            return result
        except Exception as e:
            logger.error(f"Redis HDEL error for key {key}: {str(e)}")
            return 0

    # List Operations

    @classmethod
    async def lpush(cls, key: str, *values: str) -> int:
        """Push values to the left of list.

        Args:
            key: List key
            *values: Values to push

        Returns:
            List length after push
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.lpush(key, *values)
            return result
        except Exception as e:
            logger.error(f"Redis LPUSH error for key {key}: {str(e)}")
            return 0

    @classmethod
    async def rpop(cls, key: str) -> Optional[str]:
        """Pop value from the right of list.

        Args:
            key: List key

        Returns:
            Popped value or None
        """
        if not cls._client:
            return None

        try:
            result = await cls._client.rpop(key)
            return result
        except Exception as e:
            logger.error(f"Redis RPOP error for key {key}: {str(e)}")
            return None

    @classmethod
    async def lrange(cls, key: str, start: int, stop: int) -> List[str]:
        """Get range of list elements.

        Args:
            key: List key
            start: Start index
            stop: Stop index (-1 for end)

        Returns:
            List of values
        """
        if not cls._client:
            return []

        try:
            result = await cls._client.lrange(key, start, stop)
            return result
        except Exception as e:
            logger.error(f"Redis LRANGE error for key {key}: {str(e)}")
            return []

    # Pub/Sub Operations

    @classmethod
    async def publish(cls, channel: str, message: str) -> int:
        """Publish message to channel.

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        if not cls._client:
            return 0

        try:
            result = await cls._client.publish(channel, message)
            return result
        except Exception as e:
            logger.error(f"Redis PUBLISH error for channel {channel}: {str(e)}")
            return 0

    @classmethod
    async def subscribe(cls, *channels: str) -> Optional[redis.client.PubSub]:
        """Subscribe to channels.

        Args:
            *channels: Channel names

        Returns:
            PubSub instance or None
        """
        if not cls._client:
            return None

        try:
            pubsub = cls._client.pubsub()
            await pubsub.subscribe(*channels)

            # Store pubsub client for cleanup
            for channel in channels:
                cls._pubsub_clients[channel] = pubsub

            return pubsub
        except Exception as e:
            logger.error(f"Redis SUBSCRIBE error: {str(e)}")
            return None

    # Transaction Operations

    @classmethod
    @asynccontextmanager
    async def pipeline(cls, transaction: bool = True):
        """Create a pipeline for atomic operations.

        Args:
            transaction: Whether to use MULTI/EXEC

        Yields:
            Pipeline instance
        """
        if not cls._client:
            raise RuntimeError("Redis client not initialized")

        pipeline = cls._client.pipeline(transaction=transaction)
        try:
            yield pipeline
            await pipeline.execute()
        except Exception as e:
            logger.error(f"Redis pipeline error: {str(e)}")
            raise
        finally:
            await pipeline.reset()

    # Lock Operations

    @classmethod
    @asynccontextmanager
    async def lock(
        cls,
        key: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: int = 5,
    ):
        """Distributed lock using Redis.

        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            blocking: Whether to block waiting for lock
            blocking_timeout: Max time to wait for lock

        Yields:
            Lock instance
        """
        if not cls._client:
            raise RuntimeError("Redis client not initialized")

        lock = cls._client.lock(
            key,
            timeout=timeout,
            blocking=blocking,
            blocking_timeout=blocking_timeout,
        )

        acquired = False
        try:
            acquired = await lock.acquire()
            if not acquired and blocking:
                raise TimeoutError(f"Failed to acquire lock for key: {key}")
            yield lock
        finally:
            if acquired:
                try:
                    await lock.release()
                except Exception as e:
                    logger.error(f"Error releasing lock for key {key}: {str(e)}")

    # Utility Methods

    @classmethod
    async def keys(cls, pattern: str = "*") -> List[str]:
        """Get keys matching pattern.

        Args:
            pattern: Key pattern (use with caution in production)

        Returns:
            List of matching keys
        """
        if not cls._client:
            return []

        try:
            # Use SCAN instead of KEYS for production
            keys = []
            async for key in cls._client.scan_iter(pattern):
                keys.append(key)
            return keys
        except Exception as e:
            logger.error(f"Redis KEYS error: {str(e)}")
            return []

    @classmethod
    async def flushdb(cls) -> bool:
        """Flush current database (use with caution).

        Returns:
            bool: Success status
        """
        if not cls._client:
            return False

        try:
            await cls._client.flushdb()
            logger.warning("Redis database flushed")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {str(e)}")
            return False

    @classmethod
    async def info(cls) -> Dict[str, Any]:
        """Get Redis server info.

        Returns:
            Server information dictionary
        """
        if not cls._client:
            return {}

        try:
            info = await cls._client.info()
            return info
        except Exception as e:
            logger.error(f"Redis INFO error: {str(e)}")
            return {}


# Cache helper functions

async def cache_get(key: str) -> Optional[Any]:
    """Get cached value with JSON deserialization.

    Args:
        key: Cache key

    Returns:
        Cached value or None
    """
    return await RedisClient.get_json(key)


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
) -> bool:
    """Set cache value with JSON serialization.

    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds

    Returns:
        bool: Success status
    """
    return await RedisClient.set_json(key, value, ttl)


async def cache_delete(*keys: str) -> int:
    """Delete cache keys.

    Args:
        *keys: Keys to delete

    Returns:
        Number of keys deleted
    """
    return await RedisClient.delete(*keys)


async def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client instance.

    Returns:
        Redis client or None
    """
    return RedisClient.get_client()


# Export main classes and functions
__all__ = [
    "RedisClient",
    "cache_get",
    "cache_set",
    "cache_delete",
    "get_redis_client",
]
