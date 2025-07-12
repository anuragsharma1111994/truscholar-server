"""Logging middleware for TruScholar API.

This middleware logs all incoming requests and outgoing responses,
providing comprehensive request tracking and monitoring.
"""

import json
import time
from typing import Callable, Dict, Any, Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[list] = None,
        mask_fields: Optional[list] = None
    ):
        """Initialize logging middleware.
        
        Args:
            app: The ASGI application
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            exclude_paths: List of paths to exclude from logging
            mask_fields: List of field names to mask in logs
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.mask_fields = mask_fields or ["password", "token", "secret", "api_key", "authorization"]
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log details.
        
        Args:
            request: The incoming request
            call_next: The next middleware/endpoint
            
        Returns:
            Response: The response
        """
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
            
        # Get or generate request ID
        request_id = getattr(request.state, "request_id", str(uuid4()))
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        await self._log_response(response, request, request_id, duration_ms)
        
        return response
        
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details.
        
        Args:
            request: The request object
            request_id: Request ID for tracking
        """
        # Build log context
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
        
        # Add headers (excluding sensitive ones)
        headers = self._mask_sensitive_headers(dict(request.headers))
        log_data["headers"] = headers
        
        # Add request body if enabled and present
        if self.log_request_body and request.headers.get("content-type") == "application/json":
            try:
                body = await request.body()
                if body:
                    # Store body for later use
                    request.state.body = body
                    # Parse and mask sensitive fields
                    body_data = json.loads(body)
                    log_data["body"] = self._mask_sensitive_data(body_data)
            except Exception as e:
                log_data["body_error"] = str(e)
                
        # Log the request
        logger.info(
            f"Request: {request.method} {request.url.path}",
            extra=log_data
        )
        
    async def _log_response(
        self,
        response: Response,
        request: Request,
        request_id: str,
        duration_ms: float
    ):
        """Log outgoing response details.
        
        Args:
            response: The response object
            request: The original request
            request_id: Request ID for tracking
            duration_ms: Request duration in milliseconds
        """
        # Build log context
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "response_headers": dict(response.headers)
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logger.error
            log_msg = f"Error Response: {request.method} {request.url.path} - {response.status_code}"
        elif response.status_code >= 400:
            log_level = logger.warning
            log_msg = f"Client Error: {request.method} {request.url.path} - {response.status_code}"
        else:
            log_level = logger.info
            log_msg = f"Response: {request.method} {request.url.path} - {response.status_code}"
            
        # Log the response
        log_level(log_msg, extra=log_data)
        
    def _mask_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Mask sensitive header values.
        
        Args:
            headers: Request headers
            
        Returns:
            Dict[str, str]: Headers with sensitive values masked
        """
        masked_headers = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            if any(field in key_lower for field in self.mask_fields):
                masked_headers[key] = "***MASKED***"
            else:
                masked_headers[key] = value
                
        return masked_headers
        
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Recursively mask sensitive fields in data.
        
        Args:
            data: Data to mask
            
        Returns:
            Any: Data with sensitive fields masked
        """
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(field in key_lower for field in self.mask_fields):
                    masked_data[key] = "***MASKED***"
                else:
                    masked_data[key] = self._mask_sensitive_data(value)
            return masked_data
            
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
            
        else:
            return data


class RequestLoggingContext:
    """Context manager for structured request logging."""
    
    def __init__(self, request: Request, operation: str):
        """Initialize logging context.
        
        Args:
            request: The request object
            operation: Operation name for logging
        """
        self.request = request
        self.operation = operation
        self.request_id = getattr(request.state, "request_id", str(uuid4()))
        self.start_time = None
        self.extra_data = {}
        
    def add_context(self, **kwargs):
        """Add additional context to logs.
        
        Args:
            **kwargs: Additional context data
        """
        self.extra_data.update(kwargs)
        
    def __enter__(self):
        """Enter the context and log operation start."""
        self.start_time = time.time()
        
        logger.info(
            f"Operation started: {self.operation}",
            extra={
                "request_id": self.request_id,
                "operation": self.operation,
                "path": self.request.url.path,
                **self.extra_data
            }
        )
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and log operation completion."""
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type:
            # Log operation failure
            logger.error(
                f"Operation failed: {self.operation}",
                extra={
                    "request_id": self.request_id,
                    "operation": self.operation,
                    "duration_ms": round(duration_ms, 2),
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    **self.extra_data
                },
                exc_info=True
            )
        else:
            # Log operation success
            logger.info(
                f"Operation completed: {self.operation}",
                extra={
                    "request_id": self.request_id,
                    "operation": self.operation,
                    "duration_ms": round(duration_ms, 2),
                    **self.extra_data
                }
            )


def log_operation(operation: str):
    """Decorator to log function operations.
    
    Args:
        operation: Operation name for logging
        
    Usage:
        @log_operation("create_user")
        async def create_user(data: dict):
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = kwargs.get("request_id", str(uuid4()))
            
            logger.info(
                f"Operation started: {operation}",
                extra={
                    "request_id": request_id,
                    "operation": operation,
                    "function": func.__name__
                }
            )
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Operation completed: {operation}",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "function": func.__name__,
                        "duration_ms": round(duration_ms, 2)
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Operation failed: {operation}",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "function": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    exc_info=True
                )
                
                raise
                
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = kwargs.get("request_id", str(uuid4()))
            
            logger.info(
                f"Operation started: {operation}",
                extra={
                    "request_id": request_id,
                    "operation": operation,
                    "function": func.__name__
                }
            )
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    f"Operation completed: {operation}",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "function": func.__name__,
                        "duration_ms": round(duration_ms, 2)
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    f"Operation failed: {operation}",
                    extra={
                        "request_id": request_id,
                        "operation": operation,
                        "function": func.__name__,
                        "duration_ms": round(duration_ms, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    },
                    exc_info=True
                )
                
                raise
                
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Export middleware and utilities
__all__ = [
    "LoggingMiddleware",
    "RequestLoggingContext",
    "log_operation"
]