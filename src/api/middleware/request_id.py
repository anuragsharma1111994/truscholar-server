"""Request ID middleware for TruScholar API.

This middleware generates and propagates unique request IDs for tracking
requests throughout the application lifecycle.
"""

import contextvars
from typing import Callable, Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Context variable for request ID
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id",
    default=None
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and propagate request IDs."""
    
    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Request-ID",
        generate_request_id: Optional[Callable[[], str]] = None
    ):
        """Initialize request ID middleware.
        
        Args:
            app: The ASGI application
            header_name: Header name for request ID
            generate_request_id: Custom function to generate request IDs
        """
        super().__init__(app)
        self.header_name = header_name
        self.generate_request_id = generate_request_id or self._default_request_id_generator
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add request ID.
        
        Args:
            request: The incoming request
            call_next: The next middleware/endpoint
            
        Returns:
            Response: The response with request ID header
        """
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)
        
        if not request_id:
            request_id = self.generate_request_id()
            logger.debug(f"Generated new request ID: {request_id}")
        else:
            logger.debug(f"Using existing request ID from header: {request_id}")
            
        # Validate request ID format
        if not self._is_valid_request_id(request_id):
            logger.warning(f"Invalid request ID format: {request_id}, generating new one")
            request_id = self.generate_request_id()
            
        # Store request ID in request state
        request.state.request_id = request_id
        
        # Set context variable
        token = request_id_var.set(request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers[self.header_name] = request_id
            
            return response
            
        finally:
            # Reset context variable
            request_id_var.reset(token)
            
    def _default_request_id_generator(self) -> str:
        """Generate a default request ID.
        
        Returns:
            str: Generated request ID
        """
        return str(uuid4())
        
    def _is_valid_request_id(self, request_id: str) -> bool:
        """Validate request ID format.
        
        Args:
            request_id: Request ID to validate
            
        Returns:
            bool: True if valid
        """
        if not request_id:
            return False
            
        # Check length (UUID is 36 chars with hyphens)
        if len(request_id) < 32 or len(request_id) > 128:
            return False
            
        # Check for potentially malicious content
        if any(char in request_id for char in ['<', '>', '"', "'", '\n', '\r', '\0']):
            return False
            
        return True


def get_request_id() -> Optional[str]:
    """Get the current request ID from context.
    
    Returns:
        Optional[str]: Current request ID or None
    """
    return request_id_var.get()


def set_request_id(request_id: str) -> contextvars.Token:
    """Set the request ID in context.
    
    Args:
        request_id: Request ID to set
        
    Returns:
        contextvars.Token: Token for resetting context
    """
    return request_id_var.set(request_id)


class RequestIDGenerator:
    """Customizable request ID generator."""
    
    def __init__(self, prefix: str = "", include_timestamp: bool = False):
        """Initialize request ID generator.
        
        Args:
            prefix: Prefix for generated IDs
            include_timestamp: Whether to include timestamp in ID
        """
        self.prefix = prefix
        self.include_timestamp = include_timestamp
        
    def generate(self) -> str:
        """Generate a request ID.
        
        Returns:
            str: Generated request ID
        """
        parts = []
        
        if self.prefix:
            parts.append(self.prefix)
            
        if self.include_timestamp:
            import time
            parts.append(str(int(time.time() * 1000)))
            
        parts.append(str(uuid4()).replace('-', ''))
        
        return '-'.join(parts)


def with_request_id(request_id: Optional[str] = None):
    """Decorator to ensure function runs with a request ID context.
    
    Args:
        request_id: Optional request ID to use
        
    Usage:
        @with_request_id()
        async def process_data():
            request_id = get_request_id()
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Check if request ID already exists in context
            existing_id = get_request_id()
            
            if existing_id:
                # Use existing request ID
                return await func(*args, **kwargs)
            else:
                # Set new request ID
                new_id = request_id or str(uuid4())
                token = set_request_id(new_id)
                
                try:
                    return await func(*args, **kwargs)
                finally:
                    request_id_var.reset(token)
                    
        def sync_wrapper(*args, **kwargs):
            # Check if request ID already exists in context
            existing_id = get_request_id()
            
            if existing_id:
                # Use existing request ID
                return func(*args, **kwargs)
            else:
                # Set new request ID
                new_id = request_id or str(uuid4())
                token = set_request_id(new_id)
                
                try:
                    return func(*args, **kwargs)
                finally:
                    request_id_var.reset(token)
                    
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


class RequestIDLoggerAdapter:
    """Logger adapter that automatically includes request ID in logs."""
    
    def __init__(self, logger):
        """Initialize logger adapter.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        
    def _get_extra(self, extra: Optional[dict] = None) -> dict:
        """Get extra data with request ID.
        
        Args:
            extra: Additional extra data
            
        Returns:
            dict: Extra data with request ID
        """
        request_id = get_request_id()
        
        if extra is None:
            extra = {}
            
        if request_id:
            extra["request_id"] = request_id
            
        return extra
        
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with request ID."""
        kwargs["extra"] = self._get_extra(kwargs.get("extra"))
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        """Log info message with request ID."""
        kwargs["extra"] = self._get_extra(kwargs.get("extra"))
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with request ID."""
        kwargs["extra"] = self._get_extra(kwargs.get("extra"))
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        """Log error message with request ID."""
        kwargs["extra"] = self._get_extra(kwargs.get("extra"))
        self.logger.error(msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with request ID."""
        kwargs["extra"] = self._get_extra(kwargs.get("extra"))
        self.logger.critical(msg, *args, **kwargs)


# Export middleware and utilities
__all__ = [
    "RequestIDMiddleware",
    "get_request_id",
    "set_request_id",
    "RequestIDGenerator",
    "with_request_id",
    "RequestIDLoggerAdapter"
]