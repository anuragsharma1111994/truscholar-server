"""Authentication and authorization schemas for TruScholar API.

This module provides Pydantic schemas for authentication endpoints including
login, registration, token management, and user verification.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from src.schemas.base import BaseSchema
from src.utils.constants import AgeGroup, UserAccountType
from src.utils.validators import validate_name, validate_phone


class LoginRequest(BaseSchema):
    """Request schema for user login."""

    phone: str = Field(..., description="User's phone number (10 digits)")
    name: str = Field(..., description="User's name")
    device_info: Optional[Dict[str, str]] = Field(None, description="Device information")

    @field_validator("phone")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        result = validate_phone(v)
        if not result.is_valid:
            raise ValueError(result.errors[0])
        return result.cleaned_value

    @field_validator("name")
    @classmethod
    def validate_user_name(cls, v: str) -> str:
        """Validate user name."""
        result = validate_name(v)
        if not result.is_valid:
            raise ValueError(result.errors[0])
        return result.cleaned_value

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "phone": "9876543210",
                "name": "John Doe",
                "device_info": {
                    "device_type": "mobile",
                    "os": "iOS",
                    "browser": "Safari",
                    "app_version": "1.0.0"
                }
            }
        }
    }


class TokenInfo(BaseSchema):
    """Token information schema."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    scope: Optional[str] = Field(None, description="Token scope")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "expires_at": "2024-01-02T12:00:00Z",
                "scope": "user:read user:write"
            }
        }
    }


class UserBasicInfo(BaseSchema):
    """Basic user information for auth responses."""

    id: str = Field(..., description="User ID")
    phone: str = Field(..., description="User's phone number (masked)")
    name: str = Field(..., description="User's name")
    account_type: UserAccountType = Field(..., description="Account type")
    is_active: bool = Field(..., description="Whether user account is active")
    is_verified: bool = Field(..., description="Whether user is verified")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "phone": "****3210",
                "name": "John Doe",
                "account_type": "free",
                "is_active": True,
                "is_verified": False,
                "created_at": "2024-01-01T12:00:00Z",
                "last_login": "2024-01-02T10:30:00Z"
            }
        }
    }


class LoginResponse(BaseSchema):
    """Response schema for successful login."""

    user: UserBasicInfo = Field(..., description="User information")
    tokens: TokenInfo = Field(..., description="Authentication tokens")
    session_id: str = Field(..., description="Session ID")
    permissions: List[str] = Field(default_factory=list, description="User permissions")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "user": {
                    "id": "507f1f77bcf86cd799439011",
                    "phone": "****3210",
                    "name": "John Doe",
                    "account_type": "free",
                    "is_active": True,
                    "is_verified": False,
                    "created_at": "2024-01-01T12:00:00Z",
                    "last_login": "2024-01-02T10:30:00Z"
                },
                "tokens": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 86400,
                    "expires_at": "2024-01-02T12:00:00Z"
                },
                "session_id": "sess_123456789",
                "permissions": ["user:read", "user:write", "test:create"]
            }
        }
    }


class RefreshTokenRequest(BaseSchema):
    """Request schema for token refresh."""

    refresh_token: str = Field(..., description="Valid refresh token")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    }


class RefreshTokenResponse(BaseSchema):
    """Response schema for token refresh."""

    tokens: TokenInfo = Field(..., description="New authentication tokens")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "tokens": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 86400,
                    "expires_at": "2024-01-02T12:00:00Z"
                }
            }
        }
    }


class LogoutRequest(BaseSchema):
    """Request schema for user logout."""

    refresh_token: Optional[str] = Field(None, description="Refresh token to invalidate")
    logout_all_sessions: bool = Field(default=False, description="Whether to logout from all sessions")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "logout_all_sessions": False
            }
        }
    }


class VerifyTokenRequest(BaseSchema):
    """Request schema for token verification."""

    access_token: str = Field(..., description="Access token to verify")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }
    }


class VerifyTokenResponse(BaseSchema):
    """Response schema for token verification."""

    valid: bool = Field(..., description="Whether token is valid")
    user_id: Optional[str] = Field(None, description="User ID if token is valid")
    expires_at: Optional[datetime] = Field(None, description="Token expiration time")
    permissions: List[str] = Field(default_factory=list, description="Token permissions")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "valid": True,
                "user_id": "507f1f77bcf86cd799439011",
                "expires_at": "2024-01-02T12:00:00Z",
                "permissions": ["user:read", "user:write", "test:create"]
            }
        }
    }


class UserRegistration(BaseSchema):
    """Request schema for user registration."""

    phone: str = Field(..., description="User's phone number (10 digits)")
    name: str = Field(..., description="User's full name")
    age: Optional[int] = Field(None, ge=13, le=99, description="User's age")
    email: Optional[str] = Field(None, description="User's email address")

    # Privacy and terms
    terms_accepted: bool = Field(..., description="Whether user accepted terms of service")
    privacy_accepted: bool = Field(..., description="Whether user accepted privacy policy")
    data_consent: bool = Field(default=True, description="Consent for data processing")

    # Optional profile information
    location_city: Optional[str] = Field(None, max_length=100, description="User's city")
    location_state: Optional[str] = Field(None, max_length=100, description="User's state")
    education_level: Optional[str] = Field(None, max_length=100, description="Education level")
    current_occupation: Optional[str] = Field(None, max_length=100, description="Current occupation")

    # Marketing preferences
    marketing_emails: bool = Field(default=False, description="Consent for marketing emails")

    # Device information
    device_info: Optional[Dict[str, str]] = Field(None, description="Registration device info")

    @field_validator("phone")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        result = validate_phone(v)
        if not result.is_valid:
            raise ValueError(result.errors[0])
        return result.cleaned_value

    @field_validator("name")
    @classmethod
    def validate_user_name(cls, v: str) -> str:
        """Validate user name."""
        result = validate_name(v)
        if not result.is_valid:
            raise ValueError(result.errors[0])
        return result.cleaned_value

    @field_validator("email")
    @classmethod
    def validate_email_if_provided(cls, v: Optional[str]) -> Optional[str]:
        """Validate email if provided."""
        if v:
            from src.utils.validators import validate_email
            result = validate_email(v)
            if not result.is_valid:
                raise ValueError(result.errors[0])
            return result.cleaned_value
        return v

    @field_validator("terms_accepted", "privacy_accepted")
    @classmethod
    def validate_required_acceptance(cls, v: bool, info) -> bool:
        """Ensure required terms are accepted."""
        if not v:
            field_name = info.field_name.replace('_', ' ').title()
            raise ValueError(f"{field_name} must be accepted")
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "phone": "9876543210",
                "name": "John Doe",
                "age": 25,
                "email": "john.doe@example.com",
                "terms_accepted": True,
                "privacy_accepted": True,
                "data_consent": True,
                "location_city": "Mumbai",
                "location_state": "Maharashtra",
                "education_level": "Bachelor's Degree",
                "current_occupation": "Software Engineer",
                "marketing_emails": False,
                "device_info": {
                    "device_type": "mobile",
                    "os": "Android",
                    "browser": "Chrome"
                }
            }
        }
    }


class UserRegistrationResponse(BaseSchema):
    """Response schema for user registration."""

    user: UserBasicInfo = Field(..., description="Created user information")
    tokens: TokenInfo = Field(..., description="Authentication tokens")
    session_id: str = Field(..., description="Session ID")
    welcome_message: str = Field(..., description="Welcome message for new user")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "user": {
                    "id": "507f1f77bcf86cd799439011",
                    "phone": "****3210",
                    "name": "John Doe",
                    "account_type": "free",
                    "is_active": True,
                    "is_verified": False,
                    "created_at": "2024-01-01T12:00:00Z",
                    "last_login": None
                },
                "tokens": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 86400,
                    "expires_at": "2024-01-02T12:00:00Z"
                },
                "session_id": "sess_123456789",
                "welcome_message": "Welcome to TruScholar! Let's discover your perfect career path.",
                "next_steps": [
                    "Complete your profile",
                    "Take your first RAISEC assessment",
                    "Explore career recommendations"
                ]
            }
        }
    }


class PasswordResetRequest(BaseSchema):
    """Request schema for password reset (future use)."""

    phone: str = Field(..., description="User's phone number")

    @field_validator("phone")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        result = validate_phone(v)
        if not result.is_valid:
            raise ValueError(result.errors[0])
        return result.cleaned_value

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "phone": "9876543210"
            }
        }
    }


class PasswordResetConfirm(BaseSchema):
    """Request schema for password reset confirmation (future use)."""

    phone: str = Field(..., description="User's phone number")
    reset_code: str = Field(..., min_length=6, max_length=10, description="Reset verification code")
    new_password: str = Field(..., min_length=8, description="New password")

    @field_validator("phone")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        result = validate_phone(v)
        if not result.is_valid:
            raise ValueError(result.errors[0])
        return result.cleaned_value

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "phone": "9876543210",
                "reset_code": "123456",
                "new_password": "newSecurePassword123"
            }
        }
    }


class SessionInfo(BaseSchema):
    """Session information schema."""

    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    device_info: Optional[Dict[str, str]] = Field(None, description="Device information")
    ip_address: Optional[str] = Field(None, description="IP address (masked)")
    created_at: datetime = Field(..., description="Session creation time")
    last_accessed: datetime = Field(..., description="Last access time")
    expires_at: datetime = Field(..., description="Session expiration time")
    is_active: bool = Field(..., description="Whether session is active")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "session_id": "sess_123456789",
                "user_id": "507f1f77bcf86cd799439011",
                "device_info": {
                    "device_type": "mobile",
                    "os": "iOS",
                    "browser": "Safari"
                },
                "ip_address": "192.168.***.***",
                "created_at": "2024-01-01T12:00:00Z",
                "last_accessed": "2024-01-01T12:30:00Z",
                "expires_at": "2024-01-02T12:00:00Z",
                "is_active": True
            }
        }
    }


class UserSessionsResponse(BaseSchema):
    """Response schema for user sessions list."""

    current_session: SessionInfo = Field(..., description="Current session information")
    other_sessions: List[SessionInfo] = Field(default_factory=list, description="Other active sessions")
    total_sessions: int = Field(..., description="Total number of active sessions")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "current_session": {
                    "session_id": "sess_123456789",
                    "user_id": "507f1f77bcf86cd799439011",
                    "device_info": {"device_type": "mobile"},
                    "created_at": "2024-01-01T12:00:00Z",
                    "is_active": True
                },
                "other_sessions": [],
                "total_sessions": 1
            }
        }
    }


# Export all authentication schemas
__all__ = [
    "LoginRequest",
    "TokenInfo",
    "UserBasicInfo",
    "LoginResponse",
    "RefreshTokenRequest",
    "RefreshTokenResponse",
    "LogoutRequest",
    "VerifyTokenRequest",
    "VerifyTokenResponse",
    "UserRegistration",
    "UserRegistrationResponse",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "SessionInfo",
    "UserSessionsResponse",
]
