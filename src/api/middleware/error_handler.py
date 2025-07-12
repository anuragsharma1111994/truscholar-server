"""Error handler middleware for TruScholar API.

This middleware catches and handles all exceptions, ensuring consistent
error responses across the application.
"""

import sys
import traceback
from typing import Callable, Optional
from uuid import uuid4

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.exceptions import (
    TruScholarError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    BusinessLogicError,
    RateLimitError,
    ExternalServiceError,
    DatabaseError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and format error responses."""
    
    def __init__(self, app: ASGIApp, debug: bool = False):
        """Initialize error handler middleware.
        
        Args:
            app: The ASGI application
            debug: Whether to include detailed error info
        """
        super().__init__(app)
        self.debug = debug
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and handle any exceptions.
        
        Args:
            request: The incoming request
            call_next: The next middleware/endpoint
            
        Returns:
            Response: The response or error response
        """
        request_id = request.state.request_id if hasattr(request.state, "request_id") else str(uuid4())
        
        try:
            response = await call_next(request)
            return response
            
        except TruScholarError as e:
            # Handle custom application exceptions
            return await self._handle_application_error(e, request_id, request)
            
        except ValueError as e:
            # Handle validation errors
            return await self._handle_validation_error(e, request_id, request)
            
        except Exception as e:
            # Handle unexpected errors
            return await self._handle_unexpected_error(e, request_id, request)
            
    async def _handle_application_error(
        self,
        error: TruScholarError,
        request_id: str,
        request: Request
    ) -> JSONResponse:
        """Handle custom application exceptions.
        
        Args:
            error: The application exception
            request_id: Request ID for tracking
            request: The request object
            
        Returns:
            JSONResponse: Formatted error response
        """
        # Determine status code based on error type
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(error, ValidationError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(error, AuthenticationError):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(error, AuthorizationError):
            status_code = status.HTTP_403_FORBIDDEN
        elif isinstance(error, ResourceNotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(error, RateLimitError):
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif isinstance(error, ExternalServiceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        # Log the error
        logger.warning(
            f"Application error: {error.message}",
            extra={
                "request_id": request_id,
                "error_code": error.error_code,
                "error_type": error.__class__.__name__,
                "status_code": status_code,
                "path": request.url.path,
                "method": request.method,
                "details": error.details
            }
        )
        
        # Build error response
        error_response = {
            "error": {
                "code": error.error_code or error.__class__.__name__,
                "message": error.message,
                "request_id": request_id
            }
        }
        
        # Add details if available
        if error.details:
            error_response["error"]["details"] = error.details
            
        # Add field errors for validation errors
        if isinstance(error, ValidationError) and hasattr(error, "validation_errors"):
            error_response["error"]["validation_errors"] = error.validation_errors
            
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
        
    async def _handle_validation_error(
        self,
        error: ValueError,
        request_id: str,
        request: Request
    ) -> JSONResponse:
        """Handle validation errors.
        
        Args:
            error: The validation error
            request_id: Request ID for tracking
            request: The request object
            
        Returns:
            JSONResponse: Formatted error response
        """
        logger.warning(
            f"Validation error: {str(error)}",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(error),
                    "request_id": request_id
                }
            }
        )
        
    async def _handle_unexpected_error(
        self,
        error: Exception,
        request_id: str,
        request: Request
    ) -> JSONResponse:
        """Handle unexpected errors.
        
        Args:
            error: The unexpected error
            request_id: Request ID for tracking
            request: The request object
            
        Returns:
            JSONResponse: Formatted error response
        """
        # Log the full error with traceback
        logger.error(
            f"Unexpected error: {str(error)}",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        # Build error response
        error_response = {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "request_id": request_id
            }
        }
        
        # Add debug info if enabled
        if self.debug:
            error_response["error"]["debug"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc().split("\n")
            }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )


def create_error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[dict] = None
) -> JSONResponse:
    """Create a standardized error response.
    
    Args:
        status_code: HTTP status code
        code: Error code
        message: Error message
        request_id: Request ID for tracking
        details: Additional error details
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_content = {
        "error": {
            "code": code,
            "message": message
        }
    }
    
    if request_id:
        error_content["error"]["request_id"] = request_id
        
    if details:
        error_content["error"]["details"] = details
        
    return JSONResponse(
        status_code=status_code,
        content=error_content
    )


# Error response factory functions

def bad_request_error(
    message: str = "Bad request",
    code: str = "BAD_REQUEST",
    request_id: Optional[str] = None,
    details: Optional[dict] = None
) -> JSONResponse:
    """Create a 400 Bad Request error response."""
    return create_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        code=code,
        message=message,
        request_id=request_id,
        details=details
    )


def unauthorized_error(
    message: str = "Unauthorized",
    code: str = "UNAUTHORIZED",
    request_id: Optional[str] = None
) -> JSONResponse:
    """Create a 401 Unauthorized error response."""
    return create_error_response(
        status_code=status.HTTP_401_UNAUTHORIZED,
        code=code,
        message=message,
        request_id=request_id
    )


def forbidden_error(
    message: str = "Forbidden",
    code: str = "FORBIDDEN",
    request_id: Optional[str] = None
) -> JSONResponse:
    """Create a 403 Forbidden error response."""
    return create_error_response(
        status_code=status.HTTP_403_FORBIDDEN,
        code=code,
        message=message,
        request_id=request_id
    )


def not_found_error(
    message: str = "Resource not found",
    code: str = "NOT_FOUND",
    request_id: Optional[str] = None
) -> JSONResponse:
    """Create a 404 Not Found error response."""
    return create_error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        code=code,
        message=message,
        request_id=request_id
    )


def conflict_error(
    message: str = "Resource conflict",
    code: str = "CONFLICT",
    request_id: Optional[str] = None,
    details: Optional[dict] = None
) -> JSONResponse:
    """Create a 409 Conflict error response."""
    return create_error_response(
        status_code=status.HTTP_409_CONFLICT,
        code=code,
        message=message,
        request_id=request_id,
        details=details
    )


def rate_limit_error(
    message: str = "Rate limit exceeded",
    code: str = "RATE_LIMIT_EXCEEDED",
    request_id: Optional[str] = None,
    retry_after: Optional[int] = None
) -> JSONResponse:
    """Create a 429 Too Many Requests error response."""
    response = create_error_response(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        code=code,
        message=message,
        request_id=request_id
    )
    
    if retry_after:
        response.headers["Retry-After"] = str(retry_after)
        
    return response


def internal_error(
    message: str = "Internal server error",
    code: str = "INTERNAL_ERROR",
    request_id: Optional[str] = None
) -> JSONResponse:
    """Create a 500 Internal Server Error response."""
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=code,
        message=message,
        request_id=request_id
    )


def service_unavailable_error(
    message: str = "Service temporarily unavailable",
    code: str = "SERVICE_UNAVAILABLE",
    request_id: Optional[str] = None,
    retry_after: Optional[int] = None
) -> JSONResponse:
    """Create a 503 Service Unavailable error response."""
    response = create_error_response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        code=code,
        message=message,
        request_id=request_id
    )
    
    if retry_after:
        response.headers["Retry-After"] = str(retry_after)
        
    return response


# Export middleware and utilities
__all__ = [
    "ErrorHandlerMiddleware",
    "create_error_response",
    "bad_request_error",
    "unauthorized_error",
    "forbidden_error",
    "not_found_error",
    "conflict_error",
    "rate_limit_error",
    "internal_error",
    "service_unavailable_error"
]