"""Cache manager for TruScholar application.

This module provides a high-level interface for caching operations using Redis,
with support for JSON serialization, TTL management, and cache invalidation.
"""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from datetime import datetime, timedelta
from functools import wraps
import asyncio

from src.database.redis_client import RedisClient
from src.cache.cache_keys import CacheKeys, CacheNamespace
from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

T = TypeVar('T')


class CacheManager:
    """High-level cache management interface."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        """Initialize cache manager.
        
        Args:
            redis_client: Optional Redis client instance
        """
        self.redis = redis_client or RedisClient
        self.enabled = settings.ENABLE_CACHE
        
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        if not self.enabled:
            return default
            
        try:
            value = await self.redis.get(key)
            if value is None:
                return default
                
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return True
            
        try:
            # Serialize value if needed
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, (str, int, float, bytes)):
                value = json.dumps(value, default=str)
                
            return await self.redis.set(key, value, ttl=ttl, nx=nx, xx=xx)
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys from cache.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not keys:
            return 0
            
        try:
            return await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return 0
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist.
        
        Args:
            *keys: Keys to check
            
        Returns:
            Number of keys that exist
        """
        if not self.enabled or not keys:
            return 0
            
        try:
            return await self.redis.exists(*keys)
        except Exception as e:
            logger.error(f"Cache exists error: {str(e)}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return True
            
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {str(e)}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expire, -2 if not exists
        """
        if not self.enabled:
            return -2
            
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {str(e)}")
            return -2
    
    # High-level operations
    
    async def get_or_set(
        self,
        key: str,
        func: callable,
        ttl: Optional[int] = None,
        force_refresh: bool = False
    ) -> Any:
        """Get value from cache or compute and cache it.
        
        Args:
            key: Cache key
            func: Async function to compute value
            ttl: Time to live in seconds
            force_refresh: Force recompute value
            
        Returns:
            Cached or computed value
        """
        if not self.enabled:
            return await func()
            
        # Check cache first unless forced refresh
        if not force_refresh:
            cached = await self.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for key: {key}")
                return cached
        
        logger.debug(f"Cache miss for key: {key}")
        
        # Compute value
        value = await func()
        
        # Cache the result
        await self.set(key, value, ttl=ttl)
        
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., "user:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0
            
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate pattern error: {str(e)}")
            return 0
    
    # User cache operations
    
    async def cache_user(self, user_id: str, user_data: dict, ttl: Optional[int] = None) -> bool:
        """Cache user data.
        
        Args:
            user_id: User ID
            user_data: User data to cache
            ttl: Custom TTL or use default
            
        Returns:
            True if successful
        """
        key = CacheKeys.user_by_id(user_id)
        ttl = ttl or 3600  # Default 1 hour TTL for user data
        return await self.set(key, user_data, ttl=ttl)
    
    async def get_cached_user(self, user_id: str) -> Optional[dict]:
        """Get cached user data.
        
        Args:
            user_id: User ID
            
        Returns:
            User data or None
        """
        key = CacheKeys.user_by_id(user_id)
        return await self.get(key)
    
    async def invalidate_user(self, user_id: str) -> int:
        """Invalidate all user-related cache.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of keys deleted
        """
        pattern = CacheKeys.pattern_for_user(user_id)
        count = await self.invalidate_pattern(pattern)
        
        # Also invalidate specific keys
        keys = [
            CacheKeys.user_by_id(user_id),
            CacheKeys.user_sessions(user_id),
            CacheKeys.user_tests(user_id),
            CacheKeys.user_active_test(user_id)
        ]
        count += await self.delete(*keys)
        
        return count
    
    # Test cache operations
    
    async def cache_test(self, test_id: str, test_data: dict, ttl: Optional[int] = None) -> bool:
        """Cache test data.
        
        Args:
            test_id: Test ID
            test_data: Test data to cache
            ttl: Custom TTL or use default
            
        Returns:
            True if successful
        """
        key = CacheKeys.test_by_id(test_id)
        ttl = ttl or 7200  # Default 2 hour TTL for test progress
        return await self.set(key, test_data, ttl=ttl)
    
    async def get_cached_test(self, test_id: str) -> Optional[dict]:
        """Get cached test data.
        
        Args:
            test_id: Test ID
            
        Returns:
            Test data or None
        """
        key = CacheKeys.test_by_id(test_id)
        return await self.get(key)
    
    async def invalidate_test(self, test_id: str) -> int:
        """Invalidate all test-related cache.
        
        Args:
            test_id: Test ID
            
        Returns:
            Number of keys deleted
        """
        pattern = CacheKeys.pattern_for_test(test_id)
        count = await self.invalidate_pattern(pattern)
        
        # Also invalidate specific keys
        keys = [
            CacheKeys.test_by_id(test_id),
            CacheKeys.test_questions(test_id),
            CacheKeys.test_answers(test_id),
            CacheKeys.test_progress(test_id),
            CacheKeys.test_score(test_id)
        ]
        count += await self.delete(*keys)
        
        return count
    
    # Question cache operations
    
    async def cache_questions(self, test_id: str, questions: List[dict], ttl: Optional[int] = None) -> bool:
        """Cache test questions.
        
        Args:
            test_id: Test ID
            questions: List of question data
            ttl: Custom TTL or use default
            
        Returns:
            True if successful
        """
        key = CacheKeys.test_questions(test_id)
        ttl = ttl or 3600  # Default 1 hour TTL for test questions
        return await self.set(key, questions, ttl=ttl)
    
    async def get_cached_questions(self, test_id: str) -> Optional[List[dict]]:
        """Get cached test questions.
        
        Args:
            test_id: Test ID
            
        Returns:
            List of questions or None
        """
        key = CacheKeys.test_questions(test_id)
        return await self.get(key)
    
    # Lock operations for distributed systems
    
    async def acquire_lock(self, resource: str, ttl: int = 30, retry_times: int = 3) -> bool:
        """Acquire a distributed lock.
        
        Args:
            resource: Resource identifier
            ttl: Lock timeout in seconds
            retry_times: Number of retry attempts
            
        Returns:
            True if lock acquired
        """
        lock_key = f"lock:{resource}"
        lock_value = f"{datetime.utcnow().isoformat()}"
        
        for _ in range(retry_times):
            success = await self.set(lock_key, lock_value, ttl=ttl, nx=True)
            if success:
                return True
            await asyncio.sleep(0.1)
            
        return False
    
    async def release_lock(self, resource: str) -> bool:
        """Release a distributed lock.
        
        Args:
            resource: Resource identifier
            
        Returns:
            True if lock released
        """
        lock_key = f"lock:{resource}"
        return await self.delete(lock_key) > 0
    
    # Hash operations for LLM caching
    
    def compute_hash(self, data: Union[str, dict]) -> str:
        """Compute hash for cache key.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def cache_llm_response(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache LLM response.
        
        Args:
            prompt: LLM prompt
            response: LLM response
            metadata: Optional metadata
            ttl: Custom TTL or use default
            
        Returns:
            True if successful
        """
        hash_key = self.compute_hash(prompt)
        key = CacheKeys.llm_response(hash_key)
        
        data = {
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "cached_at": datetime.utcnow().isoformat()
        }
        
        ttl = ttl or 86400  # Default 24 hour TTL for LLM responses
        return await self.set(key, data, ttl=ttl)
    
    async def get_cached_llm_response(self, prompt: str) -> Optional[str]:
        """Get cached LLM response.
        
        Args:
            prompt: LLM prompt
            
        Returns:
            Cached response or None
        """
        hash_key = self.compute_hash(prompt)
        key = CacheKeys.llm_response(hash_key)
        
        data = await self.get(key)
        if data and isinstance(data, dict):
            return data.get("response")
        return None


# Decorator for caching function results
def cache_result(
    key_func: callable,
    ttl: Optional[int] = None,
    namespace: Optional[str] = None
):
    """Decorator for caching async function results.
    
    Args:
        key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds
        namespace: Optional namespace prefix
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_func(*args, **kwargs)
            if namespace:
                cache_key = f"{namespace}:{cache_key}"
            
            # Get cache manager
            cache_manager = CacheManager()
            
            # Try to get from cache
            cached = await cache_manager.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
cache_manager = CacheManager()