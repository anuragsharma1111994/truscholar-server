"""Pydantic schemas for TruScholar API.

This module provides all Pydantic schemas for request/response validation,
data serialization, and API documentation.
"""

from src.schemas.auth_schemas import (
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    LogoutRequest,
    VerifyTokenRequest,
    VerifyTokenResponse,
    UserRegistration,
    UserRegistrationResponse,
)
from src.schemas.base import (
    BaseResponse,
    ErrorResponse,
    PaginatedResponse,
    SuccessResponse,
    ValidationErrorResponse,
    PaginationParams,
    SortParams,
    FilterParams,
    ResponseMetadata,
)
from src.schemas.user_schemas import (
    UserBase,
    UserCreate,
    UserUpdate,
    UserResponse,
    UserProfile,
    UserProfileUpdate,
    UserPreferences,
    UserPreferencesUpdate,
    UserStats,
    UserDevice,
    UserSummary,
    UserList,
    UserSearch,
)

# Version info
__version__ = "1.0.0"

# Export all schemas
__all__ = [
    # Base schemas
    "BaseResponse",
    "ErrorResponse",
    "PaginatedResponse",
    "SuccessResponse",
    "ValidationErrorResponse",
    "PaginationParams",
    "SortParams",
    "FilterParams",
    "ResponseMetadata",

    # Authentication schemas
    "LoginRequest",
    "LoginResponse",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "LogoutRequest",
    "VerifyTokenRequest",
    "VerifyTokenResponse",
    "UserRegistration",
    "UserRegistrationResponse",

    # User schemas
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserProfile",
    "UserProfileUpdate",
    "UserPreferences",
    "UserPreferencesUpdate",
    "UserStats",
    "UserDevice",
    "UserSummary",
    "UserList",
    "UserSearch",
]
