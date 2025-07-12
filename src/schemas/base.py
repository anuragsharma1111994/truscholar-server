"""Base Pydantic schemas for TruScholar API.

This module provides base schemas, response models, and common data structures
used across all API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator

from src.utils.constants import ErrorCodes, ResponseFormats
from src.utils.enums import SortOrder
from src.utils.validators import ValidationResult

# Generic type variable for data
DataType = TypeVar('DataType')


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "arbitrary_types_allowed": False,
        "str_strip_whitespace": True,
        "json_schema_extra": {
            "example": {}
        }
    }


class ResponseMetadata(BaseSchema):
    """Metadata included in API responses."""

    request_id: Optional[str] = Field(None, description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    version: str = Field(default="1.0", description="API version")
    processing_time_ms: Optional[float] = Field(None, description="Request processing time in milliseconds")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "request_id": "req_123456789",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0",
                "processing_time_ms": 150.5
            }
        }
    }


class BaseResponse(BaseSchema, Generic[DataType]):
    """Base response schema for all API endpoints."""

    success: bool = Field(..., description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    data: Optional[DataType] = Field(None, description="Response data")
    meta: Optional[ResponseMetadata] = Field(None, description="Response metadata")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {},
                "meta": {
                    "request_id": "req_123456789",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "version": "1.0"
                }
            }
        }
    }


class SuccessResponse(BaseResponse[DataType]):
    """Success response schema."""

    success: bool = Field(default=True, description="Always true for success responses")

    @classmethod
    def create(
        cls,
        data: DataType,
        message: Optional[str] = None,
        meta: Optional[ResponseMetadata] = None
    ) -> "SuccessResponse[DataType]":
        """Create a success response.

        Args:
            data: Response data
            message: Optional success message
            meta: Optional metadata

        Returns:
            SuccessResponse: Success response instance
        """
        return cls(
            success=True,
            data=data,
            message=message,
            meta=meta or ResponseMetadata()
        )


class ErrorDetail(BaseSchema):
    """Individual error detail."""

    code: Optional[str] = Field(None, description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid phone number format",
                "field": "phone",
                "details": {"pattern": "Must be 10 digits starting with 6-9"}
            }
        }
    }


class ErrorResponse(BaseResponse[None]):
    """Error response schema."""

    success: bool = Field(default=False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error information")

    @classmethod
    def create(
        cls,
        message: str,
        code: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        meta: Optional[ResponseMetadata] = None
    ) -> "ErrorResponse":
        """Create an error response.

        Args:
            message: Error message
            code: Error code
            field: Field that caused the error
            details: Additional error details
            meta: Optional metadata

        Returns:
            ErrorResponse: Error response instance
        """
        return cls(
            success=False,
            error=ErrorDetail(
                code=code,
                message=message,
                field=field,
                details=details
            ),
            meta=meta or ResponseMetadata()
        )

    @classmethod
    def from_validation_result(
        cls,
        validation_result: ValidationResult,
        meta: Optional[ResponseMetadata] = None
    ) -> "ErrorResponse":
        """Create error response from validation result.

        Args:
            validation_result: Failed validation result
            meta: Optional metadata

        Returns:
            ErrorResponse: Error response instance
        """
        if validation_result.is_valid:
            raise ValueError("Cannot create error response from successful validation")

        return cls(
            success=False,
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message=validation_result.errors[0] if validation_result.errors else "Validation failed",
                details={
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            ),
            meta=meta or ResponseMetadata()
        )


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with multiple errors."""

    @classmethod
    def create_multiple(
        cls,
        errors: List[ErrorDetail],
        message: str = "Validation failed",
        meta: Optional[ResponseMetadata] = None
    ) -> "ValidationErrorResponse":
        """Create validation error response with multiple errors.

        Args:
            errors: List of error details
            message: Overall error message
            meta: Optional metadata

        Returns:
            ValidationErrorResponse: Validation error response
        """
        return cls(
            success=False,
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message=message,
                details={"errors": [error.model_dump() for error in errors]}
            ),
            meta=meta or ResponseMetadata()
        )


class PaginationInfo(BaseSchema):
    """Pagination information."""

    page: int = Field(..., ge=1, description="Current page number")
    limit: int = Field(..., ge=1, le=100, description="Items per page")
    total: int = Field(..., ge=0, description="Total number of items")
    pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

    @classmethod
    def create(cls, page: int, limit: int, total: int) -> "PaginationInfo":
        """Create pagination info from parameters.

        Args:
            page: Current page number
            limit: Items per page
            total: Total number of items

        Returns:
            PaginationInfo: Pagination information
        """
        pages = (total + limit - 1) // limit if total > 0 else 0

        return cls(
            page=page,
            limit=limit,
            total=total,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "page": 1,
                "limit": 20,
                "total": 150,
                "pages": 8,
                "has_next": True,
                "has_prev": False
            }
        }
    }


class PaginatedResponse(BaseResponse[List[DataType]]):
    """Paginated response schema."""

    pagination: PaginationInfo = Field(..., description="Pagination information")

    @classmethod
    def create(
        cls,
        data: List[DataType],
        page: int,
        limit: int,
        total: int,
        message: Optional[str] = None,
        meta: Optional[ResponseMetadata] = None
    ) -> "PaginatedResponse[DataType]":
        """Create a paginated response.

        Args:
            data: List of items for current page
            page: Current page number
            limit: Items per page
            total: Total number of items
            message: Optional message
            meta: Optional metadata

        Returns:
            PaginatedResponse: Paginated response instance
        """
        return cls(
            success=True,
            data=data,
            message=message,
            pagination=PaginationInfo.create(page, limit, total),
            meta=meta or ResponseMetadata()
        )


# Request parameter schemas

class PaginationParams(BaseSchema):
    """Query parameters for pagination."""

    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def skip(self) -> int:
        """Calculate skip value for database queries."""
        return (self.page - 1) * self.limit

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "page": 1,
                "limit": 20
            }
        }
    }


class SortParams(BaseSchema):
    """Query parameters for sorting."""

    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")

    @property
    def mongo_sort(self) -> List[tuple]:
        """Get MongoDB sort parameters."""
        return [(self.sort_by, self.sort_order.mongo_value)]

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        }
    }


class FilterParams(BaseSchema):
    """Base class for filter parameters."""

    search: Optional[str] = Field(None, min_length=2, max_length=100, description="Search query")
    created_after: Optional[datetime] = Field(None, description="Filter items created after this date")
    created_before: Optional[datetime] = Field(None, description="Filter items created before this date")

    @field_validator("search")
    @classmethod
    def validate_search(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean search query."""
        if v:
            # Remove extra whitespace and normalize
            cleaned = " ".join(v.strip().split())
            return cleaned if len(cleaned) >= 2 else None
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "search": "john doe",
                "created_after": "2024-01-01T00:00:00Z",
                "created_before": "2024-12-31T23:59:59Z"
            }
        }
    }


class BulkOperationResult(BaseSchema):
    """Result of a bulk operation."""

    total_requested: int = Field(..., ge=0, description="Total items requested for operation")
    successful: int = Field(..., ge=0, description="Number of successful operations")
    failed: int = Field(..., ge=0, description="Number of failed operations")
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of errors that occurred")

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requested == 0:
            return 0.0
        return (self.successful / self.total_requested) * 100

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "total_requested": 100,
                "successful": 95,
                "failed": 5,
                "errors": []
            }
        }
    }


class HealthCheckResponse(BaseSchema):
    """Health check response schema."""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., ge=0, description="Application uptime in seconds")

    # Service statuses
    database: Dict[str, Any] = Field(default_factory=dict, description="Database health")
    cache: Dict[str, Any] = Field(default_factory=dict, description="Cache health")
    external_services: Dict[str, Any] = Field(default_factory=dict, description="External services health")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "database": {"status": "connected", "response_time_ms": 5.2},
                "cache": {"status": "connected", "response_time_ms": 1.1},
                "external_services": {"openai": {"status": "available"}}
            }
        }
    }


# Utility functions for creating responses

def create_success_response(
    data: Any,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> SuccessResponse:
    """Create a success response with metadata.

    Args:
        data: Response data
        message: Optional success message
        request_id: Optional request ID

    Returns:
        SuccessResponse: Success response
    """
    meta = ResponseMetadata()
    if request_id:
        meta.request_id = request_id

    return SuccessResponse.create(data=data, message=message, meta=meta)


def create_error_response(
    message: str,
    code: Optional[str] = None,
    field: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create an error response with metadata.

    Args:
        message: Error message
        code: Error code
        field: Field that caused the error
        details: Additional error details
        request_id: Optional request ID

    Returns:
        ErrorResponse: Error response
    """
    meta = ResponseMetadata()
    if request_id:
        meta.request_id = request_id

    return ErrorResponse.create(
        message=message,
        code=code,
        field=field,
        details=details,
        meta=meta
    )


def create_paginated_response(
    data: List[Any],
    page: int,
    limit: int,
    total: int,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> PaginatedResponse:
    """Create a paginated response with metadata.

    Args:
        data: List of items for current page
        page: Current page number
        limit: Items per page
        total: Total number of items
        message: Optional message
        request_id: Optional request ID

    Returns:
        PaginatedResponse: Paginated response
    """
    meta = ResponseMetadata()
    if request_id:
        meta.request_id = request_id

    return PaginatedResponse.create(
        data=data,
        page=page,
        limit=limit,
        total=total,
        message=message,
        meta=meta
    )


# Export all schemas and utilities
__all__ = [
    # Base schemas
    "BaseSchema",
    "ResponseMetadata",
    "BaseResponse",
    "SuccessResponse",
    "ErrorDetail",
    "ErrorResponse",
    "ValidationErrorResponse",
    "PaginationInfo",
    "PaginatedResponse",
    "BulkOperationResult",
    "HealthCheckResponse",

    # Parameter schemas
    "PaginationParams",
    "SortParams",
    "FilterParams",

    # Utility functions
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
]
