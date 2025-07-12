"""Cache decorators for TruScholar application.

This module provides decorators for caching function results
to improve performance and reduce database/API calls.
"""

import functools
import hashlib
import json
from typing import Any, Callable, Optional, Union
from datetime import timedelta

from src.cache import Cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_cache_key(
    prefix: str,
    func_name: str,
    args: tuple,
    kwargs: dict,
    key_params: Optional[list] = None
) -> str:
    """Generate a cache key from function name and arguments.
    
    Args:
        prefix: Cache key prefix
        func_name: Function name
        args: Function positional arguments
        kwargs: Function keyword arguments
        key_params: List of parameter names to include in key
        
    Returns:
        str: Generated cache key
    """
    # Start with prefix and function name
    key_parts = [prefix, func_name]
    
    if key_params:
        # Use only specified parameters
        for i, param in enumerate(key_params):
            if i < len(args):
                key_parts.append(str(args[i]))
            elif param in kwargs:
                key_parts.append(str(kwargs[param]))
    else:
        # Use all arguments
        key_parts.extend(str(arg) for arg in args)
        
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        for k, v in sorted_kwargs:
            key_parts.extend([str(k), str(v)])
    
    # Create hash of key parts for shorter keys
    key_string = ':'.join(key_parts)
    if len(key_string) > 200:
        # Hash long keys
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{func_name}:{key_hash}"
    
    return key_string


def cached(
    prefix: str = "cache",
    expire: Union[int, timedelta] = 300,
    key_params: Optional[list] = None,
    cache_none: bool = False,
    namespace: Optional[str] = None
):
    """Decorator to cache function results.
    
    Args:
        prefix: Cache key prefix
        expire: Cache expiration in seconds or timedelta
        key_params: List of parameter names to include in cache key
        cache_none: Whether to cache None results
        namespace: Optional namespace for cache keys
        
    Usage:
        @cached(prefix="user", expire=3600)
        async def get_user(user_id: str):
            return await db.get_user(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache instance
            cache = kwargs.get('cache')
            if not cache or not isinstance(cache, Cache):
                # No cache available, call function directly
                return await func(*args, **kwargs)
            
            # Remove cache from kwargs to avoid passing it to function
            kwargs_without_cache = {k: v for k, v in kwargs.items() if k != 'cache'}
            
            # Generate cache key
            full_prefix = f"{namespace}:{prefix}" if namespace else prefix
            cache_key = generate_cache_key(
                full_prefix,
                func.__name__,
                args,
                kwargs_without_cache,
                key_params
            )
            
            # Try to get from cache
            try:
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_value
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
            
            # Call the function
            result = await func(*args, **kwargs_without_cache)
            
            # Cache the result
            if result is not None or cache_none:
                try:
                    # Convert timedelta to seconds
                    expire_seconds = expire.total_seconds() if isinstance(expire, timedelta) else expire
                    
                    await cache.set(cache_key, result, expire=int(expire_seconds))
                    logger.debug(f"Cached result for key: {cache_key}")
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async cache
            # Just call the function directly
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def invalidate_cache(
    prefix: str = "cache",
    namespace: Optional[str] = None,
    patterns: Optional[list] = None
):
    """Decorator to invalidate cache entries after function execution.
    
    Args:
        prefix: Cache key prefix to invalidate
        namespace: Optional namespace for cache keys
        patterns: List of cache key patterns to invalidate
        
    Usage:
        @invalidate_cache(prefix="user", patterns=["user:*", "users:list"])
        async def update_user(user_id: str, data: dict):
            return await db.update_user(user_id, data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Call the function first
            result = await func(*args, **kwargs)
            
            # Get cache instance
            cache = kwargs.get('cache')
            if cache and isinstance(cache, Cache):
                try:
                    # Build patterns to invalidate
                    invalidate_patterns = []
                    
                    if patterns:
                        invalidate_patterns.extend(patterns)
                    else:
                        # Default pattern based on prefix
                        full_prefix = f"{namespace}:{prefix}" if namespace else prefix
                        invalidate_patterns.append(f"{full_prefix}:*")
                    
                    # Invalidate cache entries
                    for pattern in invalidate_patterns:
                        await cache.delete_pattern(pattern)
                        logger.debug(f"Invalidated cache pattern: {pattern}")
                        
                except Exception as e:
                    logger.warning(f"Cache invalidation error: {e}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, just call the function
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_aside(
    key_generator: Callable,
    expire: Union[int, timedelta] = 300,
    cache_none: bool = False
):
    """Cache-aside pattern decorator with custom key generation.
    
    Args:
        key_generator: Function to generate cache key from arguments
        expire: Cache expiration in seconds or timedelta
        cache_none: Whether to cache None results
        
    Usage:
        def user_key(user_id: str, include_profile: bool = False):
            return f"user:{user_id}:profile:{include_profile}"
            
        @cache_aside(key_generator=user_key, expire=3600)
        async def get_user_data(user_id: str, include_profile: bool = False):
            return await fetch_user_data(user_id, include_profile)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            cache = kwargs.get('cache')
            if not cache or not isinstance(cache, Cache):
                # No cache available, call function directly
                return await func(*args, **kwargs)
            
            # Remove cache from kwargs
            kwargs_without_cache = {k: v for k, v in kwargs.items() if k != 'cache'}
            
            # Generate cache key
            try:
                cache_key = key_generator(*args, **kwargs_without_cache)
            except Exception as e:
                logger.warning(f"Key generation error: {e}")
                return await func(*args, **kwargs_without_cache)
            
            # Try to get from cache
            try:
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for custom key: {cache_key}")
                    return cached_value
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
            
            # Call the function
            result = await func(*args, **kwargs_without_cache)
            
            # Cache the result
            if result is not None or cache_none:
                try:
                    # Convert timedelta to seconds
                    expire_seconds = expire.total_seconds() if isinstance(expire, timedelta) else expire
                    
                    await cache.set(cache_key, result, expire=int(expire_seconds))
                    logger.debug(f"Cached result for custom key: {cache_key}")
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
            
            return result
        
        return wrapper
    
    return decorator


def memoize(maxsize: int = 128):
    """In-memory memoization decorator.
    
    Args:
        maxsize: Maximum number of cached results
        
    Usage:
        @memoize(maxsize=256)
        def expensive_calculation(x: int, y: int):
            return x ** y
    """
    def decorator(func: Callable) -> Callable:
        # Use functools.lru_cache for implementation
        cached_func = functools.lru_cache(maxsize=maxsize)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)
        
        # Add cache info and clear methods
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        
        return wrapper
    
    return decorator


# Import asyncio for async function detection
import asyncio


# Conditional caching decorator
def cached_if(
    condition: Callable[..., bool],
    prefix: str = "cache",
    expire: Union[int, timedelta] = 300
):
    """Cache results only if condition is met.
    
    Args:
        condition: Function that returns True if result should be cached
        prefix: Cache key prefix
        expire: Cache expiration
        
    Usage:
        @cached_if(condition=lambda result: result is not None and result.success)
        async def api_call(endpoint: str):
            return await make_api_request(endpoint)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance
            cache = kwargs.get('cache')
            if not cache or not isinstance(cache, Cache):
                return await func(*args, **kwargs)
            
            # Remove cache from kwargs
            kwargs_without_cache = {k: v for k, v in kwargs.items() if k != 'cache'}
            
            # Generate cache key
            cache_key = generate_cache_key(
                prefix,
                func.__name__,
                args,
                kwargs_without_cache
            )
            
            # Try to get from cache
            try:
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
            
            # Call the function
            result = await func(*args, **kwargs_without_cache)
            
            # Check condition and cache if met
            try:
                if condition(result):
                    expire_seconds = expire.total_seconds() if isinstance(expire, timedelta) else expire
                    await cache.set(cache_key, result, expire=int(expire_seconds))
                    logger.debug(f"Conditionally cached result for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Conditional cache error: {e}")
            
            return result
        
        return wrapper
    
    return decorator


# Export decorators
__all__ = [
    "cached",
    "invalidate_cache",
    "cache_aside",
    "memoize",
    "cached_if",
    "generate_cache_key",
]