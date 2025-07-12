"""Response schemas for TruScholar API.

This module defines standardized response schemas for API endpoints,
ensuring consistent response formats across the application.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, TypeVar, Generic
from pydantic import BaseModel, Field

from src.schemas.base import BaseSchema

# Generic type for response data
T = TypeVar('T')


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "field": "email",
                "message": "Invalid email format",
                "code": "INVALID_FORMAT"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    validation_errors: Optional[List[ErrorDetail]] = Field(None, description="Field validation errors")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "validation_errors": [
                    {
                        "field": "age",
                        "message": "Age must be between 13 and 35",
                        "code": "OUT_OF_RANGE"
                    }
                ]
            }
        }
    }


class SuccessResponse(BaseSchema, Generic[T]):
    """Standard success response format with generic data type."""
    
    success: bool = Field(default=True, description="Success indicator")
    data: T = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Optional success message")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "success": True,
                "data": {"id": "123", "name": "Example"},
                "message": "Operation completed successfully"
            }
        }
    }


class PaginatedResponse(BaseSchema, Generic[T]):
    """Standard paginated response format."""
    
    success: bool = Field(default=True, description="Success indicator")
    data: List[T] = Field(..., description="List of items")
    pagination: "PaginationInfo" = Field(..., description="Pagination information")
    message: Optional[str] = Field(None, description="Optional message")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "success": True,
                "data": [{"id": "1"}, {"id": "2"}],
                "pagination": {
                    "page": 1,
                    "per_page": 20,
                    "total": 100,
                    "total_pages": 5,
                    "has_next": True,
                    "has_prev": False
                }
            }
        }
    }


class PaginationInfo(BaseModel):
    """Pagination metadata."""
    
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "page": 2,
                "per_page": 20,
                "total": 95,
                "total_pages": 5,
                "has_next": True,
                "has_prev": True
            }
        }
    }


class BatchOperationResult(BaseSchema):
    """Result of a batch operation."""
    
    success: bool = Field(..., description="Overall success status")
    total: int = Field(..., ge=0, description="Total items processed")
    succeeded: int = Field(..., ge=0, description="Number of successful operations")
    failed: int = Field(..., ge=0, description="Number of failed operations")
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of errors")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Individual operation results")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "success": False,
                "total": 10,
                "succeeded": 8,
                "failed": 2,
                "errors": [
                    {
                        "message": "Item not found",
                        "code": "NOT_FOUND"
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    dependencies: Optional[Dict[str, "DependencyStatus"]] = Field(None, description="Dependency statuses")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "service": "TruScholar API",
                "version": "1.0.0",
                "environment": "production",
                "timestamp": "2024-01-01T12:00:00Z",
                "dependencies": {
                    "database": {"status": "healthy", "latency_ms": 5},
                    "cache": {"status": "healthy", "latency_ms": 1}
                }
            }
        }
    }


class DependencyStatus(BaseModel):
    """Status of a service dependency."""
    
    status: str = Field(..., description="Dependency status")
    latency_ms: Optional[float] = Field(None, description="Latency in milliseconds")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "latency_ms": 3.5
            }
        }
    }


class TaskStatusResponse(BaseSchema):
    """Response for async task status."""
    
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[Any] = Field(None, description="Task result if completed")
    error: Optional[ErrorResponse] = Field(None, description="Error if failed")
    created_at: datetime = Field(..., description="Task creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "task_id": "task_123456",
                "status": "processing",
                "progress": 45,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:05:00Z"
            }
        }
    }


class FileUploadResponse(BaseSchema):
    """Response for file upload operations."""
    
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    url: Optional[str] = Field(None, description="File access URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL for images")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "file_id": "file_789abc",
                "filename": "report.pdf",
                "size": 1024000,
                "content_type": "application/pdf",
                "url": "https://api.truscholar.com/files/file_789abc",
                "uploaded_at": "2024-01-01T12:00:00Z"
            }
        }
    }


class MessageResponse(BaseSchema):
    """Simple message response."""
    
    message: str = Field(..., description="Response message")
    code: Optional[str] = Field(None, description="Message code")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "message": "Email sent successfully",
                "code": "EMAIL_SENT"
            }
        }
    }


class CountResponse(BaseSchema):
    """Response containing a count."""
    
    count: int = Field(..., ge=0, description="Count value")
    label: Optional[str] = Field(None, description="Label for the count")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "count": 42,
                "label": "Total active users"
            }
        }
    }


class StatisticsResponse(BaseSchema):
    """Response containing statistics."""
    
    total: int = Field(..., ge=0, description="Total count")
    average: Optional[float] = Field(None, description="Average value")
    min: Optional[float] = Field(None, description="Minimum value")
    max: Optional[float] = Field(None, description="Maximum value")
    sum: Optional[float] = Field(None, description="Sum of values")
    breakdown: Optional[Dict[str, int]] = Field(None, description="Breakdown by category")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "total": 100,
                "average": 75.5,
                "min": 10,
                "max": 100,
                "sum": 7550,
                "breakdown": {
                    "category_a": 40,
                    "category_b": 60
                }
            }
        }
    }


# Update forward references
PaginatedResponse.model_rebuild()
HealthResponse.model_rebuild()


# Utility functions for creating responses

def success_response(
    data: Any,
    message: Optional[str] = None
) -> SuccessResponse:
    """Create a success response.
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        SuccessResponse: Formatted success response
    """
    return SuccessResponse(data=data, message=message)


def error_response(
    code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    validation_errors: Optional[List[ErrorDetail]] = None
) -> Dict[str, Any]:
    """Create an error response dictionary.
    
    Args:
        code: Error code
        message: Error message
        request_id: Request ID
        details: Additional details
        validation_errors: Field validation errors
        
    Returns:
        Dict[str, Any]: Error response dictionary
    """
    error = ErrorResponse(
        code=code,
        message=message,
        request_id=request_id,
        details=details,
        validation_errors=validation_errors
    )
    
    return {
        "success": False,
        "error": error.model_dump(exclude_none=True)
    }


def paginated_response(
    data: List[Any],
    page: int,
    per_page: int,
    total: int,
    message: Optional[str] = None
) -> PaginatedResponse:
    """Create a paginated response.
    
    Args:
        data: List of items
        page: Current page
        per_page: Items per page
        total: Total items
        message: Optional message
        
    Returns:
        PaginatedResponse: Formatted paginated response
    """
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
    
    pagination = PaginationInfo(
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    return PaginatedResponse(
        data=data,
        pagination=pagination,
        message=message
    )


# Export all schemas and utilities
__all__ = [
    # Response schemas
    "ErrorDetail",
    "ErrorResponse",
    "SuccessResponse",
    "PaginatedResponse",
    "PaginationInfo",
    "BatchOperationResult",
    "HealthResponse",
    "DependencyStatus",
    "TaskStatusResponse",
    "FileUploadResponse",
    "MessageResponse",
    "CountResponse",
    "StatisticsResponse",
    
    # Utility functions
    "success_response",
    "error_response",
    "paginated_response"
]