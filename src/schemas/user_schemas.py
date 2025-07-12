"""User management schemas for TruScholar API.

This module provides Pydantic schemas for user-related operations including
user profiles, preferences, statistics, and user management endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from src.schemas.base import BaseSchema, FilterParams
from src.utils.constants import AgeGroup, UserAccountType
from src.utils.validators import validate_age, validate_email, validate_name, validate_phone


class UserBase(BaseSchema):
    """Base user schema with common fields."""

    phone: str = Field(..., description="User's phone number (10 digits)")
    name: str = Field(..., description="User's full name")
    email: Optional[str] = Field(None, description="User's email address")

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
            result = validate_email(v)
            if not result.is_valid:
                raise ValueError(result.errors[0])
            return result.cleaned_value
        return v


class UserProfile(BaseSchema):
    """User profile information schema."""

    age: Optional[int] = Field(None, ge=13, le=99, description="User's age")
    age_group: Optional[AgeGroup] = Field(None, description="User's age group")
    education_level: Optional[str] = Field(None, max_length=100, description="Education level")
    current_occupation: Optional[str] = Field(None, max_length=100, description="Current occupation")
    location_city: Optional[str] = Field(None, max_length=100, description="City")
    location_state: Optional[str] = Field(None, max_length=100, description="State")
    location_country: str = Field(default="IN", description="Country code")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    interests: List[str] = Field(default_factory=list, max_length=10, description="User interests")

    @field_validator("age")
    @classmethod
    def validate_user_age(cls, v: Optional[int]) -> Optional[int]:
        """Validate user age if provided."""
        if v is not None:
            result = validate_age(v)
            if not result.is_valid:
                raise ValueError(result.errors[0])
            return result.cleaned_value["age"]
        return v

    @model_validator(mode="after")
    def set_age_group_from_age(self) -> "UserProfile":
        """Set age group based on age."""
        if self.age is not None:
            from src.utils.constants import get_age_group_from_age
            try:
                self.age_group = get_age_group_from_age(self.age)
            except ValueError:
                # Age outside supported range, leave age_group as None
                pass
        return self

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "age": 25,
                "age_group": "18-25",
                "education_level": "Bachelor's Degree",
                "current_occupation": "Software Engineer",
                "location_city": "Mumbai",
                "location_state": "Maharashtra",
                "location_country": "IN",
                "bio": "Passionate about technology and career development",
                "interests": ["technology", "career growth", "learning"]
            }
        }
    }


class UserPreferences(BaseSchema):
    """User preferences and settings schema."""

    language: str = Field(default="en", pattern="^[a-z]{2}$", description="Language preference")
    timezone: str = Field(default="Asia/Kolkata", description="Timezone preference")
    email_notifications: bool = Field(default=True, description="Email notifications enabled")
    sms_notifications: bool = Field(default=False, description="SMS notifications enabled")
    marketing_emails: bool = Field(default=False, description="Marketing emails enabled")
    theme: str = Field(default="light", pattern="^(light|dark|auto)$", description="UI theme preference")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "language": "en",
                "timezone": "Asia/Kolkata",
                "email_notifications": True,
                "sms_notifications": False,
                "marketing_emails": False,
                "theme": "light"
            }
        }
    }


class UserStats(BaseSchema):
    """User statistics and metrics schema."""

    total_tests_taken: int = Field(default=0, ge=0, description="Total tests taken")
    tests_completed: int = Field(default=0, ge=0, description="Tests completed")
    tests_abandoned: int = Field(default=0, ge=0, description="Tests abandoned")
    average_test_duration_minutes: float = Field(default=0.0, ge=0.0, description="Average test duration")
    last_test_date: Optional[datetime] = Field(None, description="Last test date")
    career_paths_viewed: int = Field(default=0, ge=0, description="Career paths viewed")
    reports_generated: int = Field(default=0, ge=0, description="Reports generated")

    @property
    def completion_rate(self) -> float:
        """Calculate test completion rate as percentage."""
        if self.total_tests_taken == 0:
            return 0.0
        return (self.tests_completed / self.total_tests_taken) * 100

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "total_tests_taken": 5,
                "tests_completed": 4,
                "tests_abandoned": 1,
                "average_test_duration_minutes": 25.5,
                "last_test_date": "2024-01-15T14:30:00Z",
                "career_paths_viewed": 12,
                "reports_generated": 4
            }
        }
    }


class UserDevice(BaseSchema):
    """User device information schema."""

    device_id: Optional[str] = Field(None, description="Device identifier")
    device_type: Optional[str] = Field(None, description="Device type")
    os: Optional[str] = Field(None, description="Operating system")
    browser: Optional[str] = Field(None, description="Browser")
    app_version: Optional[str] = Field(None, description="App version")
    last_ip: Optional[str] = Field(None, description="Last IP address (masked)")
    last_user_agent: Optional[str] = Field(None, description="Last user agent")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "device_id": "device_123456",
                "device_type": "mobile",
                "os": "iOS 17.0",
                "browser": "Safari",
                "app_version": "1.0.0",
                "last_ip": "192.168.***.***",
                "last_user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0..."
            }
        }
    }


class UserCreate(UserBase):
    """Schema for creating a new user."""

    profile: Optional[UserProfile] = Field(None, description="User profile information")
    preferences: Optional[UserPreferences] = Field(None, description="User preferences")

    # Account settings
    account_type: UserAccountType = Field(default=UserAccountType.FREE, description="Account type")
    is_active: bool = Field(default=True, description="Whether account is active")

    # Consent and legal
    terms_accepted: bool = Field(..., description="Terms of service accepted")
    privacy_accepted: bool = Field(..., description="Privacy policy accepted")
    data_consent: bool = Field(default=True, description="Data processing consent")

    # Additional fields
    tags: List[str] = Field(default_factory=list, max_length=10, description="User tags")
    notes: Optional[str] = Field(None, max_length=1000, description="Admin notes")
    referral_code: Optional[str] = Field(None, max_length=20, description="Referral code")

    @field_validator("terms_accepted", "privacy_accepted")
    @classmethod
    def validate_required_acceptance(cls, v: bool) -> bool:
        """Ensure required terms are accepted."""
        if not v:
            raise ValueError("Required acceptance must be true")
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "phone": "9876543210",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "profile": {
                    "age": 25,
                    "education_level": "Bachelor's Degree",
                    "current_occupation": "Software Engineer",
                    "location_city": "Mumbai",
                    "location_state": "Maharashtra"
                },
                "preferences": {
                    "language": "en",
                    "email_notifications": True,
                    "marketing_emails": False
                },
                "account_type": "free",
                "terms_accepted": True,
                "privacy_accepted": True,
                "data_consent": True,
                "tags": ["new_user"],
                "referral_code": "REF123"
            }
        }
    }


class UserUpdate(BaseSchema):
    """Schema for updating user information."""

    name: Optional[str] = Field(None, description="User's full name")
    email: Optional[str] = Field(None, description="User's email address")

    @field_validator("name")
    @classmethod
    def validate_user_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate user name if provided."""
        if v:
            result = validate_name(v)
            if not result.is_valid:
                raise ValueError(result.errors[0])
            return result.cleaned_value
        return v

    @field_validator("email")
    @classmethod
    def validate_email_if_provided(cls, v: Optional[str]) -> Optional[str]:
        """Validate email if provided."""
        if v:
            result = validate_email(v)
            if not result.is_valid:
                raise ValueError(result.errors[0])
            return result.cleaned_value
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "name": "John Smith",
                "email": "john.smith@example.com"
            }
        }
    }


class UserProfileUpdate(BaseSchema):
    """Schema for updating user profile."""

    age: Optional[int] = Field(None, ge=13, le=99, description="User's age")
    education_level: Optional[str] = Field(None, max_length=100, description="Education level")
    current_occupation: Optional[str] = Field(None, max_length=100, description="Current occupation")
    location_city: Optional[str] = Field(None, max_length=100, description="City")
    location_state: Optional[str] = Field(None, max_length=100, description="State")
    location_country: Optional[str] = Field(None, max_length=3, description="Country code")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    interests: Optional[List[str]] = Field(None, max_length=10, description="User interests")

    @field_validator("age")
    @classmethod
    def validate_user_age(cls, v: Optional[int]) -> Optional[int]:
        """Validate user age if provided."""
        if v is not None:
            result = validate_age(v)
            if not result.is_valid:
                raise ValueError(result.errors[0])
            return result.cleaned_value["age"]
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "age": 26,
                "education_level": "Master's Degree",
                "current_occupation": "Senior Software Engineer",
                "location_city": "Bangalore",
                "location_state": "Karnataka",
                "bio": "Experienced developer with passion for AI and career guidance",
                "interests": ["artificial intelligence", "career development", "technology"]
            }
        }
    }


class UserPreferencesUpdate(BaseSchema):
    """Schema for updating user preferences."""

    language: Optional[str] = Field(None, pattern="^[a-z]{2}$", description="Language preference")
    timezone: Optional[str] = Field(None, description="Timezone preference")
    email_notifications: Optional[bool] = Field(None, description="Email notifications enabled")
    sms_notifications: Optional[bool] = Field(None, description="SMS notifications enabled")
    marketing_emails: Optional[bool] = Field(None, description="Marketing emails enabled")
    theme: Optional[str] = Field(None, pattern="^(light|dark|auto)$", description="UI theme preference")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "language": "hi",
                "timezone": "Asia/Kolkata",
                "email_notifications": False,
                "theme": "dark"
            }
        }
    }


class UserResponse(BaseSchema):
    """Schema for user response data."""

    id: str = Field(..., description="User ID")
    phone: str = Field(..., description="User's phone number (masked)")
    name: str = Field(..., description="User's full name")
    email: Optional[str] = Field(None, description="User's email address")

    # Profile information
    profile: Optional[UserProfile] = Field(None, description="User profile")
    preferences: Optional[UserPreferences] = Field(None, description="User preferences")
    stats: Optional[UserStats] = Field(None, description="User statistics")
    device: Optional[UserDevice] = Field(None, description="Device information")

    # Account information
    account_type: UserAccountType = Field(..., description="Account type")
    is_active: bool = Field(..., description="Whether account is active")
    is_verified: bool = Field(..., description="Whether user is verified")
    is_admin: bool = Field(..., description="Whether user is admin")

    # Timestamps
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_active: datetime = Field(..., description="Last activity timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    login_count: int = Field(..., description="Total login count")

    # Related data counts
    test_ids_count: int = Field(default=0, description="Number of tests taken")
    report_ids_count: int = Field(default=0, description="Number of reports generated")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "phone": "****3210",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "profile": {
                    "age": 25,
                    "age_group": "18-25",
                    "education_level": "Bachelor's Degree",
                    "current_occupation": "Software Engineer",
                    "location_city": "Mumbai",
                    "location_state": "Maharashtra"
                },
                "preferences": {
                    "language": "en",
                    "email_notifications": True,
                    "theme": "light"
                },
                "stats": {
                    "total_tests_taken": 3,
                    "tests_completed": 2,
                    "tests_abandoned": 1
                },
                "account_type": "free",
                "is_active": True,
                "is_verified": False,
                "is_admin": False,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "last_active": "2024-01-15T14:45:00Z",
                "last_login": "2024-01-15T09:00:00Z",
                "login_count": 15,
                "test_ids_count": 3,
                "report_ids_count": 2
            }
        }
    }


class UserSummary(BaseSchema):
    """Schema for user summary (minimal user information)."""

    id: str = Field(..., description="User ID")
    name: str = Field(..., description="User's name")
    phone: str = Field(..., description="User's phone (masked)")
    account_type: UserAccountType = Field(..., description="Account type")
    is_active: bool = Field(..., description="Whether account is active")
    last_active: datetime = Field(..., description="Last activity timestamp")
    tests_completed: int = Field(default=0, description="Number of completed tests")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "name": "John Doe",
                "phone": "****3210",
                "account_type": "free",
                "is_active": True,
                "last_active": "2024-01-15T14:45:00Z",
                "tests_completed": 2
            }
        }
    }


class UserList(BaseSchema):
    """Schema for user list response."""

    users: List[UserSummary] = Field(..., description="List of users")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "users": [
                    {
                        "id": "507f1f77bcf86cd799439011",
                        "name": "John Doe",
                        "phone": "****3210",
                        "account_type": "free",
                        "is_active": True,
                        "last_active": "2024-01-15T14:45:00Z",
                        "tests_completed": 2
                    }
                ]
            }
        }
    }


class UserSearch(FilterParams):
    """Schema for user search parameters."""

    name: Optional[str] = Field(None, min_length=2, max_length=100, description="Search by name")
    phone: Optional[str] = Field(None, description="Search by phone number")
    email: Optional[str] = Field(None, description="Search by email")
    account_type: Optional[UserAccountType] = Field(None, description="Filter by account type")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    is_verified: Optional[bool] = Field(None, description="Filter by verification status")
    age_group: Optional[AgeGroup] = Field(None, description="Filter by age group")
    location_city: Optional[str] = Field(None, description="Filter by city")
    location_state: Optional[str] = Field(None, description="Filter by state")
    has_completed_tests: Optional[bool] = Field(None, description="Filter users with completed tests")
    registered_after: Optional[datetime] = Field(None, description="Filter users registered after date")
    registered_before: Optional[datetime] = Field(None, description="Filter users registered before date")
    last_active_after: Optional[datetime] = Field(None, description="Filter by last active after date")
    last_active_before: Optional[datetime] = Field(None, description="Filter by last active before date")

    @field_validator("phone")
    @classmethod
    def validate_phone_search(cls, v: Optional[str]) -> Optional[str]:
        """Validate phone number for search if provided."""
        if v:
            # For search, we allow partial phone numbers
            if len(v) < 4:
                raise ValueError("Phone search must be at least 4 digits")
            if not v.isdigit():
                raise ValueError("Phone search must contain only digits")
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "search": "john",
                "name": "John",
                "account_type": "free",
                "is_active": True,
                "age_group": "18-25",
                "location_city": "Mumbai",
                "has_completed_tests": True,
                "registered_after": "2024-01-01T00:00:00Z",
                "last_active_after": "2024-01-10T00:00:00Z"
            }
        }
    }


class UserActivityLog(BaseSchema):
    """Schema for user activity log entry."""

    activity_type: str = Field(..., description="Type of activity")
    description: str = Field(..., description="Activity description")
    metadata: Optional[Dict[str, str]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(..., description="Activity timestamp")
    ip_address: Optional[str] = Field(None, description="IP address (masked)")
    user_agent: Optional[str] = Field(None, description="User agent")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "activity_type": "login",
                "description": "User logged in successfully",
                "metadata": {
                    "device_type": "mobile",
                    "login_method": "phone"
                },
                "timestamp": "2024-01-15T09:00:00Z",
                "ip_address": "192.168.***.***",
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS..."
            }
        }
    }


class UserDeletionRequest(BaseSchema):
    """Schema for user account deletion request."""

    reason: str = Field(..., min_length=10, max_length=500, description="Reason for deletion")
    confirm_deletion: bool = Field(..., description="Confirmation of deletion intent")
    delete_all_data: bool = Field(default=True, description="Whether to delete all associated data")

    @field_validator("confirm_deletion")
    @classmethod
    def validate_deletion_confirmation(cls, v: bool) -> bool:
        """Ensure deletion is confirmed."""
        if not v:
            raise ValueError("Deletion must be confirmed")
        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "reason": "No longer need the service",
                "confirm_deletion": True,
                "delete_all_data": True
            }
        }
    }


class UserBulkUpdate(BaseSchema):
    """Schema for bulk user updates."""

    user_ids: List[str] = Field(..., min_length=1, max_length=100, description="List of user IDs to update")
    updates: Dict[str, str] = Field(..., description="Fields to update")

    @field_validator("user_ids")
    @classmethod
    def validate_user_ids(cls, v: List[str]) -> List[str]:
        """Validate user IDs format."""
        if not v:
            raise ValueError("At least one user ID is required")

        for user_id in v:
            if not user_id or len(user_id) < 10:
                raise ValueError(f"Invalid user ID: {user_id}")

        return v

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "user_ids": ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"],
                "updates": {
                    "account_type": "premium",
                    "is_verified": "true"
                }
            }
        }
    }


class UserExportRequest(BaseSchema):
    """Schema for user data export request."""

    export_format: str = Field(default="json", pattern="^(json|csv|excel)$", description="Export format")
    include_tests: bool = Field(default=True, description="Include test data")
    include_reports: bool = Field(default=True, description="Include report data")
    include_activity: bool = Field(default=False, description="Include activity logs")
    date_from: Optional[datetime] = Field(None, description="Export data from this date")
    date_to: Optional[datetime] = Field(None, description="Export data until this date")

    @model_validator(mode="after")
    def validate_date_range(self) -> "UserExportRequest":
        """Validate date range."""
        if self.date_from and self.date_to:
            if self.date_from >= self.date_to:
                raise ValueError("date_from must be before date_to")
        return self

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "export_format": "json",
                "include_tests": True,
                "include_reports": True,
                "include_activity": False,
                "date_from": "2024-01-01T00:00:00Z",
                "date_to": "2024-12-31T23:59:59Z"
            }
        }
    }


class UserImportRequest(BaseSchema):
    """Schema for bulk user import request."""

    import_format: str = Field(..., pattern="^(json|csv|excel)$", description="Import format")
    data: str = Field(..., description="Import data (base64 encoded or JSON string)")
    skip_duplicates: bool = Field(default=True, description="Skip duplicate phone numbers")
    validate_only: bool = Field(default=False, description="Only validate without importing")
    default_account_type: UserAccountType = Field(default=UserAccountType.FREE, description="Default account type")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "import_format": "csv",
                "data": "data:text/csv;base64,cGhvbmUsbmFtZSxlbWFpbA==",
                "skip_duplicates": True,
                "validate_only": False,
                "default_account_type": "free"
            }
        }
    }


class UserImportResult(BaseSchema):
    """Schema for user import result."""

    total_records: int = Field(..., ge=0, description="Total records in import")
    successful_imports: int = Field(..., ge=0, description="Successfully imported users")
    failed_imports: int = Field(..., ge=0, description="Failed import attempts")
    skipped_duplicates: int = Field(..., ge=0, description="Skipped duplicate records")
    validation_errors: List[Dict[str, str]] = Field(default_factory=list, description="Validation errors")
    imported_user_ids: List[str] = Field(default_factory=list, description="IDs of imported users")

    @property
    def success_rate(self) -> float:
        """Calculate import success rate."""
        if self.total_records == 0:
            return 0.0
        return (self.successful_imports / self.total_records) * 100

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "total_records": 100,
                "successful_imports": 95,
                "failed_imports": 3,
                "skipped_duplicates": 2,
                "validation_errors": [
                    {"row": "15", "error": "Invalid phone number format"},
                    {"row": "23", "error": "Missing required field: name"}
                ],
                "imported_user_ids": ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]
            }
        }
    }


class UserPermissions(BaseSchema):
    """Schema for user permissions."""

    user_permissions: List[str] = Field(default_factory=list, description="User-specific permissions")
    role_permissions: List[str] = Field(default_factory=list, description="Role-based permissions")
    account_type_permissions: List[str] = Field(default_factory=list, description="Account type permissions")
    all_permissions: List[str] = Field(default_factory=list, description="All effective permissions")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "user_permissions": ["test:create", "report:view"],
                "role_permissions": ["user:read", "user:update"],
                "account_type_permissions": ["basic:access"],
                "all_permissions": ["test:create", "report:view", "user:read", "user:update", "basic:access"]
            }
        }
    }


class UserNotificationSettings(BaseSchema):
    """Schema for user notification settings."""

    email_enabled: bool = Field(default=True, description="Email notifications enabled")
    sms_enabled: bool = Field(default=False, description="SMS notifications enabled")
    push_enabled: bool = Field(default=True, description="Push notifications enabled")
    marketing_enabled: bool = Field(default=False, description="Marketing notifications enabled")

    # Specific notification types
    test_reminders: bool = Field(default=True, description="Test reminder notifications")
    test_completed: bool = Field(default=True, description="Test completion notifications")
    report_ready: bool = Field(default=True, description="Report ready notifications")
    career_updates: bool = Field(default=False, description="Career update notifications")
    account_security: bool = Field(default=True, description="Security-related notifications")

    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "email_enabled": True,
                "sms_enabled": False,
                "push_enabled": True,
                "marketing_enabled": False,
                "test_reminders": True,
                "test_completed": True,
                "report_ready": True,
                "career_updates": False,
                "account_security": True
            }
        }
    }


# Export all user schemas
__all__ = [
    "UserBase",
    "UserProfile",
    "UserPreferences",
    "UserStats",
    "UserDevice",
    "UserCreate",
    "UserUpdate",
    "UserProfileUpdate",
    "UserPreferencesUpdate",
    "UserResponse",
    "UserSummary",
    "UserList",
    "UserSearch",
    "UserActivityLog",
    "UserDeletionRequest",
    "UserBulkUpdate",
    "UserExportRequest",
    "UserImportRequest",
    "UserImportResult",
    "UserPermissions",
    "UserNotificationSettings",
]
