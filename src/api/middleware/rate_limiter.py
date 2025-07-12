"""Rate limiting middleware for TruScholar API."""

import time
from typing import Callable, Dict, Optional
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""

    def __init__(self, app, requests_per_minute: int = 60):
        """Initialize rate limiter middleware.
        
        Args:
            app: FastAPI application instance
            requests_per_minute: Maximum requests allowed per minute
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        # Store request counts: {client_id: [(timestamp, count)]}
        self.request_counts: Dict[str, list] = defaultdict(list)
        self.window_size = 60  # 1 minute window

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier from request.
        
        Args:
            request: HTTP request
            
        Returns:
            str: Client identifier
        """
        # Try to get authenticated user ID first
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"

    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier
            
        Returns:
            bool: True if rate limited
        """
        current_time = time.time()
        
        # Clean up old entries
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                (timestamp, count) 
                for timestamp, count in self.request_counts[client_id]
                if current_time - timestamp < self.window_size
            ]
        
        # Count requests in current window
        total_requests = sum(
            count for _, count in self.request_counts[client_id]
        )
        
        return total_requests >= self.requests_per_minute

    def _record_request(self, client_id: str):
        """Record a request for rate limiting.
        
        Args:
            client_id: Client identifier
        """
        current_time = time.time()
        
        # Add request to current window
        if self.request_counts[client_id]:
            last_timestamp, last_count = self.request_counts[client_id][-1]
            # Group requests within same second
            if int(current_time) == int(last_timestamp):
                self.request_counts[client_id][-1] = (last_timestamp, last_count + 1)
            else:
                self.request_counts[client_id].append((current_time, 1))
        else:
            self.request_counts[client_id].append((current_time, 1))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            Response: HTTP response
        """
        # Skip rate limiting if disabled
        if not settings.ENABLE_RATE_LIMITING:
            return await call_next(request)
        
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/api/v1/health"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if self._is_rate_limited(client_id):
            logger.warning(
                f"Rate limit exceeded for client: {client_id}",
                extra={"client_id": client_id, "path": request.url.path}
            )
            
            return Response(
                content='{"detail": "Rate limit exceeded. Please try again later."}',
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(self.window_size),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Window": f"{self.window_size}s"
                }
            )
        
        # Record request
        self._record_request(client_id)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Window"] = f"{self.window_size}s"
        
        return response


class AdvancedRateLimiter:
    """Advanced rate limiter with different limits for different endpoints."""
    
    def __init__(self):
        """Initialize advanced rate limiter."""
        self.limits = {
            # API endpoints with custom limits
            "/api/v1/auth/login": 5,  # 5 requests per minute
            "/api/v1/auth/register": 3,  # 3 requests per minute
            "/api/v1/tests/submit": 10,  # 10 requests per minute
            "/api/v1/careers/recommend": 20,  # 20 requests per minute
            # Default limit for other endpoints
            "default": settings.API_RATE_LIMIT_USER if hasattr(settings, 'API_RATE_LIMIT_USER') else 60
        }
        self.request_history: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.window_size = 60  # 1 minute
    
    def get_limit_for_endpoint(self, path: str) -> int:
        """Get rate limit for specific endpoint.
        
        Args:
            path: Request path
            
        Returns:
            int: Rate limit
        """
        # Check exact match first
        if path in self.limits:
            return self.limits[path]
        
        # Check prefix match
        for endpoint, limit in self.limits.items():
            if path.startswith(endpoint):
                return limit
        
        return self.limits["default"]
    
    def check_rate_limit(self, client_id: str, path: str) -> tuple[bool, Optional[int]]:
        """Check if request is within rate limit.
        
        Args:
            client_id: Client identifier
            path: Request path
            
        Returns:
            tuple[bool, Optional[int]]: (is_allowed, retry_after_seconds)
        """
        current_time = time.time()
        limit = self.get_limit_for_endpoint(path)
        
        # Clean old entries
        self.request_history[client_id][path] = [
            timestamp for timestamp in self.request_history[client_id][path]
            if current_time - timestamp < self.window_size
        ]
        
        # Check if limit exceeded
        request_count = len(self.request_history[client_id][path])
        if request_count >= limit:
            # Calculate retry after
            oldest_request = min(self.request_history[client_id][path])
            retry_after = int(oldest_request + self.window_size - current_time) + 1
            return False, retry_after
        
        # Record request
        self.request_history[client_id][path].append(current_time)
        return True, None