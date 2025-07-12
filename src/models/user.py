"""User model for the TruScholar application.

This module defines the User model and related schemas for storing user
information in MongoDB.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import AgeGroup
from src.utils.validators import validate_age, validate_name, validate_phone


class UserPreferences(EmbeddedDocument):
    """User preferences and settings."""

    language: str = Field(default="en", pattern="^[a-z]{2}$")
    timezone: str = Field(default="UTC")
    email_notifications: bool = Field(default=True)
    sms_notifications: bool = Field(default=False)
    marketing_emails: bool = Field(default=False)
    theme: str = Field(default="light", pattern="^(light|dark|auto)$")


class UserStats(EmbeddedDocument):
    """User statistics and metrics."""

    total_tests_taken: int = Field(default=0, ge=0)
    tests_completed: int = Field(default=0, ge=0)
    tests_abandoned: int = Field(default=0, ge=0)
    average_test_duration_minutes: float = Field(default=0.0, ge=0.0)
    last_test_date: Optional[datetime] = None
    career_paths_viewed: int = Field(default=0, ge=0)
    reports_generated: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_test_counts(self) -> "UserStats":
        """Ensure test counts are consistent."""
        total = self.tests_completed + self.tests_abandoned
        if self.total_tests_taken < total:
            self.total_tests_taken = total
        return self


class UserProfile(EmbeddedDocument):
    """Extended user profile information."""

    age: Optional[int] = Field(default=None, ge=13, le=99)
    age_group: Optional[AgeGroup] = None
    education_level: Optional[str] = None
    current_occupation: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_country: str = Field(default="IN")
    bio: Optional[str] = Field(default=None, max_length=500)
    interests: List[str] = Field(default_factory=list, max_length=10)

    @field_validator("age")
    @classmethod
    def validate_age_field(cls, value: Optional[int]) -> Optional[int]:
        """Validate age if provided."""
        if value is not None:
            return validate_age(value)
        return value

    @model_validator(mode="after")
    def set_age_group(self) -> "UserProfile":
        """Automatically set age group based on age."""
        if self.age is not None:
            if 13 <= self.age <= 17:
                self.age_group = AgeGroup.TEEN
            elif 18 <= self.age <= 25:
                self.age_group = AgeGroup.YOUNG_ADULT
            elif 26 <= self.age <= 35:
                self.age_group = AgeGroup.ADULT
            else:
                self.age_group = None
        return self


class UserDevice(EmbeddedDocument):
    """User device information for tracking."""

    device_id: Optional[str] = None
    device_type: Optional[str] = None  # mobile, tablet, desktop
    os: Optional[str] = None
    browser: Optional[str] = None
    app_version: Optional[str] = None
    last_ip: Optional[str] = None
    last_user_agent: Optional[str] = None


class User(BaseDocument):
    """Main user model for TruScholar application.

    Stores user authentication details, profile information, and related data.
    Phone number + name combination must be unique.
    """

    # Core fields - required for registration
    phone: str = Field(..., description="10-digit phone number")
    name: str = Field(..., min_length=2, max_length=100)

    # Optional authentication fields (for future use)
    email: Optional[str] = Field(default=None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password_hash: Optional[str] = Field(default=None, exclude=True)

    # Embedded documents
    profile: UserProfile = Field(default_factory=UserProfile)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    stats: UserStats = Field(default_factory=UserStats)
    device: UserDevice = Field(default_factory=UserDevice)

    # Account status
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)
    is_admin: bool = Field(default=False)
    account_type: str = Field(default="free", pattern="^(free|premium|enterprise)$")

    # Timestamps and tracking
    last_active: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_count: int = Field(default=0, ge=0)

    # Related data references
    test_ids: List[PyObjectId] = Field(default_factory=list)
    report_ids: List[PyObjectId] = Field(default_factory=list)

    # Compliance and consent
    terms_accepted: bool = Field(default=False)
    terms_accepted_at: Optional[datetime] = None
    privacy_accepted: bool = Field(default=False)
    privacy_accepted_at: Optional[datetime] = None
    data_consent: bool = Field(default=False)

    # Additional metadata
    referral_code: Optional[str] = Field(default=None, max_length=20)
    referred_by: Optional[PyObjectId] = None
    tags: List[str] = Field(default_factory=list, max_length=10)
    notes: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("phone")
    @classmethod
    def validate_phone_field(cls, value: str) -> str:
        """Validate phone number format."""
        return validate_phone(value)

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, value: str) -> str:
        """Validate and clean name."""
        return validate_name(value)

    @model_validator(mode="after")
    def validate_consent_timestamps(self) -> "User":
        """Set consent timestamps when consent is given."""
        if self.terms_accepted and not self.terms_accepted_at:
            self.terms_accepted_at = datetime.utcnow()
        if self.privacy_accepted and not self.privacy_accepted_at:
            self.privacy_accepted_at = datetime.utcnow()
        return self

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the user collection."""
        return [
            # Unique compound index for phone + name
            ([("phone", 1), ("name", 1)], {"unique": True}),
            # Single field indexes
            ([("phone", 1)], {}),
            ([("email", 1)], {"sparse": True}),
            ([("is_active", 1)], {}),
            ([("created_at", -1)], {}),
            ([("last_active", -1)], {}),
            # Compound indexes for queries
            ([("is_active", 1), ("account_type", 1)], {}),
            ([("profile.age_group", 1), ("is_active", 1)], {}),
            # Text search index
            ([("name", "text"), ("profile.bio", "text")], {}),
        ]

    def update_activity(self) -> None:
        """Update last active timestamp."""
        self.last_active = datetime.utcnow()
        self.update_timestamps()

    def record_login(self, device_info: Optional[Dict[str, Any]] = None) -> None:
        """Record a user login event.

        Args:
            device_info: Optional device information to update
        """
        self.last_login = datetime.utcnow()
        self.login_count += 1
        self.update_activity()

        if device_info:
            self.device = UserDevice(**device_info)

    def add_test(self, test_id: PyObjectId) -> None:
        """Add a test ID to user's test list.

        Args:
            test_id: The test ID to add
        """
        if test_id not in self.test_ids:
            self.test_ids.append(test_id)
            self.stats.total_tests_taken += 1
            self.update_timestamps()

    def add_report(self, report_id: PyObjectId) -> None:
        """Add a report ID to user's report list.

        Args:
            report_id: The report ID to add
        """
        if report_id not in self.report_ids:
            self.report_ids.append(report_id)
            self.stats.reports_generated += 1
            self.update_timestamps()

    def complete_test(self, duration_minutes: float) -> None:
        """Record test completion.

        Args:
            duration_minutes: Duration of the test in minutes
        """
        self.stats.tests_completed += 1
        self.stats.last_test_date = datetime.utcnow()

        # Update average duration
        total_duration = (
            self.stats.average_test_duration_minutes *
            (self.stats.tests_completed - 1)
        ) + duration_minutes
        self.stats.average_test_duration_minutes = (
            total_duration / self.stats.tests_completed
        )

        self.update_timestamps()

    def abandon_test(self) -> None:
        """Record test abandonment."""
        self.stats.tests_abandoned += 1
        self.update_timestamps()

    def can_take_test(self) -> bool:
        """Check if user can take a new test.

        Returns:
            bool: True if user can take a test
        """
        return (
            self.is_active and
            self.terms_accepted and
            self.privacy_accepted
        )

    def to_public_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with only public fields.

        Returns:
            Dict[str, Any]: Public user data
        """
        return self.to_dict(
            exclude={
                "password_hash",
                "test_ids",
                "report_ids",
                "device",
                "notes",
                "tags",
            }
        )

    def __repr__(self) -> str:
        """String representation of User."""
        return f"<User(id={self.id}, phone={self.phone}, name={self.name})>"


class UserSession(BaseDocument):
    """User session model for tracking active sessions."""

    user_id: PyObjectId = Field(..., description="Reference to User")
    session_token: str = Field(..., min_length=32)
    refresh_token: Optional[str] = Field(default=None, min_length=32)

    # Session metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Optional[UserDevice] = None

    # Session validity
    is_active: bool = Field(default=True)
    expires_at: datetime
    refresh_expires_at: Optional[datetime] = None
    last_accessed: datetime = Field(default_factory=datetime.utcnow)

    # Security tracking
    login_method: str = Field(default="phone", pattern="^(phone|email|oauth)$")
    security_flags: List[str] = Field(default_factory=list)

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the session collection."""
        return [
            ([("session_token", 1)], {"unique": True}),
            ([("refresh_token", 1)], {"sparse": True}),
            ([("user_id", 1), ("is_active", 1)], {}),
            ([("expires_at", 1)], {"expireAfterSeconds": 0}),
        ]

    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return (
            self.is_active and
            self.expires_at > datetime.utcnow()
        )

    def refresh(self, new_expires_at: datetime) -> None:
        """Refresh the session expiry.

        Args:
            new_expires_at: New expiration time
        """
        self.expires_at = new_expires_at
        self.last_accessed = datetime.utcnow()
        self.update_timestamps()

    def invalidate(self) -> None:
        """Invalidate the session."""
        self.is_active = False
        self.update_timestamps()
