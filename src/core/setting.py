"""Global application settings and feature flags for TruScholar.

This module provides application-wide settings, feature flags, and business
constants that complement the configuration in config.py.
"""

from datetime import timedelta
from typing import Dict, List, Set, Tuple

from src.utils.constants import AgeGroup, QuestionType, RaisecDimension


class ApplicationSettings:
    """Application-wide settings and constants."""

    # API Settings
    API_TITLE = "TruScholar API"
    API_DESCRIPTION = "AI-powered RAISEC-based career counselling platform"
    API_CONTACT = {
        "name": "TruScholar Support",
        "email": "support@truscholar.com",
        "url": "https://truscholar.com/support"
    }
    API_LICENSE = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    API_TAGS_METADATA = [
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints",
        },
        {
            "name": "Users",
            "description": "User management endpoints",
        },
        {
            "name": "Tests",
            "description": "RAISEC test management endpoints",
        },
        {
            "name": "Questions",
            "description": "Test question endpoints",
        },
        {
            "name": "Careers",
            "description": "Career recommendation endpoints",
        },
        {
            "name": "Reports",
            "description": "Test report endpoints",
        },
        {
            "name": "Health",
            "description": "Health check and monitoring endpoints",
        },
    ]

    # Request Settings
    REQUEST_ID_HEADER = "X-Request-ID"
    REQUEST_TIMEOUT_HEADER = "X-Request-Timeout"
    PROCESS_TIME_HEADER = "X-Process-Time"

    # Response Headers
    CACHE_CONTROL_HEADER = "Cache-Control"
    RATE_LIMIT_HEADER = "X-RateLimit-Limit"
    RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
    RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"

    # Security Headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }

    # CORS Settings
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    CORS_ALLOW_HEADERS = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Request-ID",
    ]
    CORS_EXPOSE_HEADERS = [
        "X-Request-ID",
        "X-Process-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ]


class BusinessSettings:
    """Business logic settings and rules."""

    # User Settings
    USERNAME_MIN_LENGTH = 2
    USERNAME_MAX_LENGTH = 100
    PHONE_NUMBER_LENGTH = 10
    PHONE_NUMBER_PATTERN = r"^[0-9]{10}$"

    # Age Groups Configuration
    AGE_GROUP_RANGES = {
        AgeGroup.TEEN: (13, 17),
        AgeGroup.YOUNG_ADULT: (18, 25),
        AgeGroup.ADULT: (26, 35),
    }

    # Test Settings
    TEST_EXPIRY_HOURS = 2
    TEST_EXTENSION_HOURS = 1
    MAX_TEST_ATTEMPTS_PER_DAY = 5
    INCOMPLETE_TEST_CLEANUP_DAYS = 7

    # Question Distribution
    QUESTION_DISTRIBUTION: Dict[QuestionType, int] = {
        QuestionType.MCQ: 2,
        QuestionType.STATEMENT_SET: 2,
        QuestionType.SCENARIO_MCQ: 2,
        QuestionType.SCENARIO_MULTI_SELECT: 2,
        QuestionType.THIS_OR_THAT: 2,
        QuestionType.SCALE_RATING: 1,
        QuestionType.PLOT_DAY: 1,
    }

    # RAISEC Scoring
    DIMENSION_WEIGHTS = {
        "single": {"R": 1.5, "A": 1.5, "I": 1.3, "S": 1.3, "E": 1.2, "C": 1.2},
        "multi": [["R", "I"], ["A", "S"], ["E", "C"], ["R", "S", "E"]],
    }

    # Scoring Points
    SINGLE_DIMENSION_POINTS = 10.0
    MULTI_DIMENSION_PRIMARY_POINTS = 6.0
    MULTI_DIMENSION_SECONDARY_POINTS = 4.0
    PLOT_DAY_TASK_POINTS = 5.0

    # Likert Scale Mapping
    LIKERT_SCALE_MAP = {
        1: 0.0,    # Strongly Disagree
        2: 2.5,    # Disagree
        3: 5.0,    # Neutral
        4: 7.5,    # Agree
        5: 10.0,   # Strongly Agree
    }

    # Plot Day Time Slots
    PLOT_DAY_TIME_SLOTS = [
        "9:00-12:00",
        "12:00-15:00",
        "15:00-18:00",
        "18:00-21:00",
    ]

    PLOT_DAY_TIME_WEIGHTS = {
        "9:00-12:00": 1.2,    # Morning - higher weight
        "12:00-15:00": 1.0,   # Afternoon - normal weight
        "15:00-18:00": 1.0,   # Late afternoon - normal weight
        "18:00-21:00": 0.8,   # Evening - lower weight
    }

    # Career Recommendations
    CAREER_FIT_SCORE_WEIGHTS = {
        "code_match": 0.4,      # 40% weight for RAISEC code match
        "dimension_correlation": 0.6,  # 60% weight for dimension correlation
    }

    MIN_CAREER_FIT_SCORE = 60.0  # Minimum score to recommend a career

    # Report Settings
    REPORT_EXPIRY_DAYS = 90
    REPORT_ACCESS_CODE_LENGTH = 8
    MAX_REPORT_VIEWS = 1000

    # Session Settings
    SESSION_IDLE_TIMEOUT_MINUTES = 30
    MAX_CONCURRENT_SESSIONS = 5
    SESSION_EXTENSION_MINUTES = 15


class CacheSettings:
    """Cache configuration and TTL settings."""

    # Cache Key Prefixes
    CACHE_PREFIX = "truscholar"

    # TTL Settings (in seconds)
    TTL_USER_SESSION = int(timedelta(hours=24).total_seconds())
    TTL_USER_PROFILE = int(timedelta(hours=12).total_seconds())
    TTL_TEST_QUESTIONS = int(timedelta(hours=2).total_seconds())
    TTL_TEST_PROGRESS = int(timedelta(minutes=5).total_seconds())
    TTL_CAREER_RECOMMENDATIONS = int(timedelta(hours=24).total_seconds())
    TTL_REPORT = int(timedelta(days=7).total_seconds())
    TTL_STATIC_CONTENT = int(timedelta(days=30).total_seconds())
    TTL_RATE_LIMIT = int(timedelta(hours=1).total_seconds())
    TTL_FEATURE_FLAG = int(timedelta(minutes=5).total_seconds())

    # Cache Patterns
    CACHE_PATTERNS = {
        "user_session": f"{CACHE_PREFIX}:session:{{user_id}}",
        "user_profile": f"{CACHE_PREFIX}:user:{{user_id}}",
        "test_questions": f"{CACHE_PREFIX}:test:{{test_id}}:questions",
        "test_progress": f"{CACHE_PREFIX}:test:{{test_id}}:progress",
        "career_recommendations": f"{CACHE_PREFIX}:test:{{test_id}}:careers",
        "report": f"{CACHE_PREFIX}:report:{{report_id}}",
        "rate_limit": f"{CACHE_PREFIX}:rate_limit:{{key}}",
        "feature_flag": f"{CACHE_PREFIX}:feature:{{flag_name}}",
    }


class FeatureFlags:
    """Feature flag definitions and defaults."""

    # Test Features
    ENABLE_DYNAMIC_QUESTIONS = True
    ENABLE_ADAPTIVE_TESTING = False
    ENABLE_QUESTION_TIMING = True
    ENABLE_SKIP_QUESTIONS = False

    # Career Features
    ENABLE_AI_CAREER_MATCHING = True
    ENABLE_CAREER_DATABASE = True
    ENABLE_CAREER_TRENDS = False
    ENABLE_SALARY_DATA = True

    # Report Features
    ENABLE_PDF_EXPORT = False
    ENABLE_EMAIL_REPORTS = False
    ENABLE_REPORT_SHARING = True
    ENABLE_REPORT_ANALYTICS = True

    # User Features
    ENABLE_USER_PROFILES = True
    ENABLE_SOCIAL_LOGIN = False
    ENABLE_EMAIL_VERIFICATION = False
    ENABLE_TWO_FACTOR_AUTH = False

    # System Features
    ENABLE_MAINTENANCE_MODE = False
    ENABLE_READ_ONLY_MODE = False
    ENABLE_BACKGROUND_JOBS = True
    ENABLE_WEBHOOKS = False

    @classmethod
    def get_all_flags(cls) -> Dict[str, bool]:
        """Get all feature flags as a dictionary."""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if key.startswith("ENABLE_") and isinstance(value, bool)
        }


class ValidationSettings:
    """Input validation settings."""

    # Text Length Limits
    MAX_TEXT_LENGTH = 5000
    MAX_CAREER_INTEREST_LENGTH = 1000
    MAX_FEEDBACK_LENGTH = 1000
    MAX_SKIP_REASON_LENGTH = 200

    # Array Length Limits
    MAX_ARRAY_LENGTH = 100
    MAX_TAGS_COUNT = 20
    MAX_SKILLS_COUNT = 50

    # Numeric Limits
    MAX_PAGE_SIZE = 100
    DEFAULT_PAGE_SIZE = 20
    MAX_EXPORT_RECORDS = 10000

    # File Upload Limits
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".csv", ".json"}
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    # Rate Limiting Rules
    RATE_LIMIT_RULES = {
        "default": (100, 3600),  # 100 requests per hour
        "auth": (5, 300),        # 5 auth attempts per 5 minutes
        "test_start": (10, 3600), # 10 tests per hour
        "report_generate": (20, 3600),  # 20 reports per hour
        "export": (10, 86400),   # 10 exports per day
    }


class ErrorMessages:
    """Standardized error messages."""

    # Authentication Errors
    INVALID_CREDENTIALS = "Invalid phone number or name"
    TOKEN_EXPIRED = "Authentication token has expired"
    TOKEN_INVALID = "Invalid authentication token"
    UNAUTHORIZED = "Authentication required"
    FORBIDDEN = "You don't have permission to access this resource"

    # Validation Errors
    INVALID_INPUT = "Invalid input data"
    MISSING_REQUIRED_FIELD = "Required field is missing: {field}"
    INVALID_FIELD_VALUE = "Invalid value for field: {field}"
    FIELD_TOO_LONG = "Field exceeds maximum length: {field}"

    # Resource Errors
    USER_NOT_FOUND = "User not found"
    TEST_NOT_FOUND = "Test not found"
    QUESTION_NOT_FOUND = "Question not found"
    REPORT_NOT_FOUND = "Report not found"
    CAREER_NOT_FOUND = "Career not found"

    # Business Logic Errors
    TEST_EXPIRED = "Test has expired"
    TEST_ALREADY_COMPLETED = "Test has already been completed"
    INVALID_TEST_STATE = "Invalid test state for this operation"
    QUESTIONS_NOT_READY = "Questions are not ready yet"
    ALL_QUESTIONS_MUST_BE_ANSWERED = "All questions must be answered before submission"

    # System Errors
    DATABASE_ERROR = "Database operation failed"
    CACHE_ERROR = "Cache operation failed"
    EXTERNAL_SERVICE_ERROR = "External service is unavailable"
    INTERNAL_ERROR = "An internal error occurred"
    SERVICE_UNAVAILABLE = "Service is temporarily unavailable"


class SuccessMessages:
    """Standardized success messages."""

    # Authentication
    LOGIN_SUCCESS = "Login successful"
    LOGOUT_SUCCESS = "Logout successful"
    TOKEN_REFRESH_SUCCESS = "Token refreshed successfully"

    # User Operations
    USER_CREATED = "User created successfully"
    USER_UPDATED = "User updated successfully"
    USER_DELETED = "User deleted successfully"

    # Test Operations
    TEST_STARTED = "Test started successfully"
    ANSWER_SUBMITTED = "Answer submitted successfully"
    TEST_COMPLETED = "Test completed successfully"

    # Report Operations
    REPORT_GENERATED = "Report generated successfully"
    REPORT_EXPORTED = "Report exported successfully"

    # Career Operations
    CAREERS_RECOMMENDED = "Career recommendations generated successfully"
    CAREER_FEEDBACK_SUBMITTED = "Career feedback submitted successfully"


class SystemConstants:
    """System-wide constants."""

    # Supported Languages
    SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "hi", "zh"]
    DEFAULT_LANGUAGE = "en"

    # Supported Timezones
    DEFAULT_TIMEZONE = "UTC"

    # Pagination
    MAX_PAGE_NUMBER = 1000

    # Search
    MIN_SEARCH_LENGTH = 2
    MAX_SEARCH_LENGTH = 100
    SEARCH_DEBOUNCE_MS = 300

    # Monitoring
    HEALTH_CHECK_INTERVAL_SECONDS = 30
    METRICS_COLLECTION_INTERVAL_SECONDS = 60

    # Maintenance
    MAINTENANCE_WARNING_HOURS = 24
    SCHEDULED_MAINTENANCE_DURATION_HOURS = 2


# Export all settings classes
__all__ = [
    "ApplicationSettings",
    "BusinessSettings",
    "CacheSettings",
    "FeatureFlags",
    "ValidationSettings",
    "ErrorMessages",
    "SuccessMessages",
    "SystemConstants",
]


# Create singleton instances
app_settings = ApplicationSettings()
business_settings = BusinessSettings()
cache_settings = CacheSettings()
feature_flags = FeatureFlags()
validation_settings = ValidationSettings()
error_messages = ErrorMessages()
success_messages = SuccessMessages()
system_constants = SystemConstants()
