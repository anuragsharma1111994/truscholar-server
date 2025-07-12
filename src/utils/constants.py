"""Constants and enums for the TruScholar application.

This module defines all constants, enums, and mappings used throughout the
application for consistency and type safety.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple


# ============================================================================
# CORE ENUMS
# ============================================================================

class AgeGroup(str, Enum):
    """Age group classifications for users."""

    TEEN = "13-17"
    YOUNG_ADULT = "18-25"
    ADULT = "26-35"

    @classmethod
    def from_age(cls, age: int) -> "AgeGroup":
        """Get age group from numeric age.

        Args:
            age: User's age

        Returns:
            AgeGroup: Corresponding age group

        Raises:
            ValueError: If age is outside supported ranges
        """
        if 13 <= age <= 17:
            return cls.TEEN
        elif 18 <= age <= 25:
            return cls.YOUNG_ADULT
        elif 26 <= age <= 35:
            return cls.ADULT
        else:
            raise ValueError(f"Age {age} is outside supported ranges (13-35)")

    def get_age_range(self) -> Tuple[int, int]:
        """Get the age range for this group.

        Returns:
            Tuple[int, int]: (min_age, max_age)
        """
        if self == self.TEEN:
            return (13, 17)
        elif self == self.YOUNG_ADULT:
            return (18, 25)
        elif self == self.ADULT:
            return (26, 35)


class RaisecDimension(str, Enum):
    """RAISEC personality dimensions."""

    REALISTIC = "R"
    ARTISTIC = "A"
    INVESTIGATIVE = "I"
    SOCIAL = "S"
    ENTERPRISING = "E"
    CONVENTIONAL = "C"

    @property
    def name_full(self) -> str:
        """Get full name of the dimension."""
        return RAISEC_DIMENSION_NAMES[self]

    @property
    def description(self) -> str:
        """Get description of the dimension."""
        return RAISEC_DIMENSION_DESCRIPTIONS[self]

    @classmethod
    def get_all_codes(cls) -> List[str]:
        """Get all RAISEC codes as a list."""
        return [dim.value for dim in cls]

    @classmethod
    def from_code(cls, code: str) -> "RaisecDimension":
        """Get dimension from single character code.

        Args:
            code: Single character code (R, A, I, S, E, C)

        Returns:
            RaisecDimension: Corresponding dimension

        Raises:
            ValueError: If code is invalid
        """
        code = code.upper()
        for dim in cls:
            if dim.value == code:
                return dim
        raise ValueError(f"Invalid RAISEC code: {code}")


class QuestionType(str, Enum):
    """Types of questions in the RAISEC assessment."""

    MCQ = "mcq"  # Multiple Choice Question
    STATEMENT_SET = "statement_set"  # Likert scale statements
    SCENARIO_MCQ = "scenario_mcq"  # Scenario-based multiple choice
    SCENARIO_MULTI_SELECT = "scenario_multi_select"  # Scenario with multiple selections
    THIS_OR_THAT = "this_or_that"  # Binary choice questions
    SCALE_RATING = "scale_rating"  # Scale of 1-10 rating
    PLOT_DAY = "plot_day"  # Drag and drop daily schedule

    @property
    def display_name(self) -> str:
        """Get display name for the question type."""
        return QUESTION_TYPE_DISPLAY_NAMES[self]

    @property
    def description(self) -> str:
        """Get description for the question type."""
        return QUESTION_TYPE_DESCRIPTIONS[self]

    @classmethod
    def get_distribution(cls) -> Dict["QuestionType", int]:
        """Get the standard distribution of question types."""
        return QUESTION_TYPE_DISTRIBUTION.copy()


class TestStatus(str, Enum):
    """Status values for test progression."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    QUESTIONS_READY = "questions_ready"
    SCORING = "scoring"
    SCORED = "scored"
    INTERESTS_SUBMITTED = "interests_submitted"
    RECOMMENDATIONS_READY = "recommendations_ready"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    EXPIRED = "expired"

    @property
    def is_active(self) -> bool:
        """Check if test is in an active state."""
        return self in ACTIVE_TEST_STATUSES

    @property
    def is_complete(self) -> bool:
        """Check if test is in a completed state."""
        return self in COMPLETED_TEST_STATUSES

    @property
    def allows_questions(self) -> bool:
        """Check if test status allows answering questions."""
        return self in [self.QUESTIONS_READY, self.IN_PROGRESS]

    @property
    def display_name(self) -> str:
        """Get display name for the status."""
        return TEST_STATUS_DISPLAY_NAMES[self]


class UserAccountType(str, Enum):
    """User account types."""

    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

    @property
    def display_name(self) -> str:
        """Get display name for account type."""
        return USER_ACCOUNT_TYPE_NAMES[self]


class ReportType(str, Enum):
    """Types of reports that can be generated."""

    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    DETAILED = "detailed"

    @property
    def display_name(self) -> str:
        """Get display name for report type."""
        return REPORT_TYPE_NAMES[self]


class PromptType(str, Enum):
    """Types of LLM prompts."""

    QUESTION_GENERATION = "question_generation"
    CAREER_RECOMMENDATION = "career_recommendation"
    REPORT_GENERATION = "report_generation"
    SCORING = "scoring"
    VALIDATION = "validation"
    GENERAL = "general"

    @property
    def display_name(self) -> str:
        """Get display name for prompt type."""
        return PROMPT_TYPE_NAMES[self]


class RecommendationType(str, Enum):
    """Types of career recommendations."""

    TRADITIONAL = "traditional"
    INNOVATIVE = "innovative"
    HYBRID = "hybrid"

    @property
    def display_name(self) -> str:
        """Get display name for recommendation type."""
        return RECOMMENDATION_TYPE_NAMES[self]


# ============================================================================
# QUESTION TYPE MAPPINGS
# ============================================================================

QUESTION_TYPE_DISTRIBUTION: Dict[QuestionType, int] = {
    QuestionType.MCQ: 2,
    QuestionType.STATEMENT_SET: 2,
    QuestionType.SCENARIO_MCQ: 2,
    QuestionType.SCENARIO_MULTI_SELECT: 2,
    QuestionType.THIS_OR_THAT: 2,
    QuestionType.SCALE_RATING: 1,
    QuestionType.PLOT_DAY: 1,
}

QUESTION_TYPE_DISPLAY_NAMES: Dict[QuestionType, str] = {
    QuestionType.MCQ: "Multiple Choice",
    QuestionType.STATEMENT_SET: "Statement Rating",
    QuestionType.SCENARIO_MCQ: "Scenario Choice",
    QuestionType.SCENARIO_MULTI_SELECT: "Scenario Multi-Select",
    QuestionType.THIS_OR_THAT: "This or That",
    QuestionType.SCALE_RATING: "Scale Rating",
    QuestionType.PLOT_DAY: "Day Planning",
}

QUESTION_TYPE_DESCRIPTIONS: Dict[QuestionType, str] = {
    QuestionType.MCQ: "Choose one option from multiple choices",
    QuestionType.STATEMENT_SET: "Rate your agreement with statements",
    QuestionType.SCENARIO_MCQ: "Choose your response to a scenario",
    QuestionType.SCENARIO_MULTI_SELECT: "Select all applicable responses to a scenario",
    QuestionType.THIS_OR_THAT: "Choose between two options",
    QuestionType.SCALE_RATING: "Rate on a scale from 1 to 10",
    QuestionType.PLOT_DAY: "Organize tasks throughout your day",
}


# ============================================================================
# RAISEC DIMENSION MAPPINGS
# ============================================================================

RAISEC_DIMENSION_NAMES: Dict[RaisecDimension, str] = {
    RaisecDimension.REALISTIC: "Realistic",
    RaisecDimension.ARTISTIC: "Artistic",
    RaisecDimension.INVESTIGATIVE: "Investigative",
    RaisecDimension.SOCIAL: "Social",
    RaisecDimension.ENTERPRISING: "Enterprising",
    RaisecDimension.CONVENTIONAL: "Conventional",
}

RAISEC_DIMENSION_DESCRIPTIONS: Dict[RaisecDimension, str] = {
    RaisecDimension.REALISTIC: "Practical, hands-on, physical activities and working with tools, machines, or animals",
    RaisecDimension.ARTISTIC: "Creative, expressive, innovative activities involving art, music, writing, or design",
    RaisecDimension.INVESTIGATIVE: "Analytical, intellectual, research activities involving problem-solving and data analysis",
    RaisecDimension.SOCIAL: "Helping, teaching, interpersonal activities focused on supporting and developing others",
    RaisecDimension.ENTERPRISING: "Leadership, persuasive, business activities involving managing and influencing others",
    RaisecDimension.CONVENTIONAL: "Organized, structured, detail-oriented activities involving data and systematic processes",
}

RAISEC_DIMENSION_KEYWORDS: Dict[RaisecDimension, List[str]] = {
    RaisecDimension.REALISTIC: [
        "hands-on", "practical", "mechanical", "physical", "tools", "machines",
        "building", "outdoor", "technical", "manual", "concrete", "tangible"
    ],
    RaisecDimension.ARTISTIC: [
        "creative", "artistic", "imaginative", "expressive", "innovative", "aesthetic",
        "design", "music", "writing", "visual", "original", "inspiration"
    ],
    RaisecDimension.INVESTIGATIVE: [
        "analytical", "research", "intellectual", "scientific", "logical", "theoretical",
        "investigation", "data", "analysis", "study", "facts", "knowledge"
    ],
    RaisecDimension.SOCIAL: [
        "helping", "teaching", "counseling", "social", "interpersonal", "collaborative",
        "community", "service", "support", "communication", "empathy", "cooperation"
    ],
    RaisecDimension.ENTERPRISING: [
        "leadership", "management", "business", "entrepreneurial", "persuasive", "ambitious",
        "competitive", "influential", "sales", "negotiation", "strategic", "results-oriented"
    ],
    RaisecDimension.CONVENTIONAL: [
        "organized", "systematic", "detailed", "structured", "precise", "methodical",
        "administrative", "procedural", "clerical", "orderly", "careful", "routine"
    ],
}


# ============================================================================
# TEST STATUS MAPPINGS
# ============================================================================

ACTIVE_TEST_STATUSES = [
    TestStatus.CREATED,
    TestStatus.IN_PROGRESS,
    TestStatus.QUESTIONS_READY,
    TestStatus.SCORING,
    TestStatus.SCORED,
    TestStatus.INTERESTS_SUBMITTED,
    TestStatus.RECOMMENDATIONS_READY,
]

COMPLETED_TEST_STATUSES = [
    TestStatus.COMPLETED,
    TestStatus.ABANDONED,
    TestStatus.EXPIRED,
]

TEST_STATUS_DISPLAY_NAMES: Dict[TestStatus, str] = {
    TestStatus.CREATED: "Created",
    TestStatus.IN_PROGRESS: "In Progress",
    TestStatus.QUESTIONS_READY: "Questions Ready",
    TestStatus.SCORING: "Scoring",
    TestStatus.SCORED: "Scored",
    TestStatus.INTERESTS_SUBMITTED: "Interests Submitted",
    TestStatus.RECOMMENDATIONS_READY: "Recommendations Ready",
    TestStatus.COMPLETED: "Completed",
    TestStatus.ABANDONED: "Abandoned",
    TestStatus.EXPIRED: "Expired",
}

TEST_STATUS_FLOW = [
    TestStatus.CREATED,
    TestStatus.QUESTIONS_READY,
    TestStatus.IN_PROGRESS,
    TestStatus.SCORING,
    TestStatus.SCORED,
    TestStatus.INTERESTS_SUBMITTED,
    TestStatus.RECOMMENDATIONS_READY,
    TestStatus.COMPLETED,
]


# ============================================================================
# USER AND ACCOUNT MAPPINGS
# ============================================================================

USER_ACCOUNT_TYPE_NAMES: Dict[UserAccountType, str] = {
    UserAccountType.FREE: "Free",
    UserAccountType.PREMIUM: "Premium",
    UserAccountType.ENTERPRISE: "Enterprise",
}


# ============================================================================
# REPORT MAPPINGS
# ============================================================================

REPORT_TYPE_NAMES: Dict[ReportType, str] = {
    ReportType.COMPREHENSIVE: "Comprehensive Report",
    ReportType.SUMMARY: "Summary Report",
    ReportType.DETAILED: "Detailed Report",
}


# ============================================================================
# PROMPT MAPPINGS
# ============================================================================

PROMPT_TYPE_NAMES: Dict[PromptType, str] = {
    PromptType.QUESTION_GENERATION: "Question Generation",
    PromptType.CAREER_RECOMMENDATION: "Career Recommendation",
    PromptType.REPORT_GENERATION: "Report Generation",
    PromptType.SCORING: "Scoring",
    PromptType.VALIDATION: "Validation",
    PromptType.GENERAL: "General",
}


# ============================================================================
# RECOMMENDATION MAPPINGS
# ============================================================================

RECOMMENDATION_TYPE_NAMES: Dict[RecommendationType, str] = {
    RecommendationType.TRADITIONAL: "Traditional Career Path",
    RecommendationType.INNOVATIVE: "Innovative Career Path",
    RecommendationType.HYBRID: "Hybrid Career Path",
}


# ============================================================================
# SCORING CONSTANTS
# ============================================================================

class ScoringConstants:
    """Constants for RAISEC scoring calculations."""

    # Base points for different question types
    SINGLE_DIMENSION_POINTS = 10.0
    PRIMARY_DIMENSION_POINTS = 6.0
    SECONDARY_DIMENSION_POINTS = 4.0
    PLOT_DAY_TASK_POINTS = 5.0

    # Likert scale mapping (1-5 scale to points)
    LIKERT_SCALE_MAP = {
        1: 0.0,    # Strongly Disagree
        2: 2.5,    # Disagree
        3: 5.0,    # Neutral
        4: 7.5,    # Agree
        5: 10.0,   # Strongly Agree
    }

    # Scale rating multiplier (1-10 scale)
    SCALE_RATING_MULTIPLIER = 1.0

    # Plot day time slot weights
    PLOT_DAY_TIME_WEIGHTS = {
        "9:00-12:00": 1.2,    # Morning - higher weight
        "12:00-15:00": 1.0,   # Afternoon - normal weight
        "15:00-18:00": 1.0,   # Late afternoon - normal weight
        "18:00-21:00": 0.8,   # Evening - lower weight
    }

    # Career fit scoring weights
    CAREER_FIT_WEIGHTS = {
        "code_match": 0.4,      # 40% weight for RAISEC code match
        "dimension_correlation": 0.6,  # 60% weight for dimension correlation
    }

    # Minimum scores and thresholds
    MIN_CAREER_FIT_SCORE = 60.0
    MIN_DIMENSION_SCORE = 0.0
    MAX_DIMENSION_SCORE = 100.0

    # Confidence calculation parameters
    MIN_CONFIDENCE = 50.0
    MAX_CONFIDENCE = 100.0
    REVISION_PENALTY = 10.0  # Penalty per revision beyond 2
    HESITATION_PENALTY = 0.5  # Penalty multiplier for hesitation score


# ============================================================================
# TIME AND SCHEDULE CONSTANTS
# ============================================================================

class TimeConstants:
    """Time-related constants."""

    # Plot day time slots
    PLOT_DAY_TIME_SLOTS = [
        "9:00-12:00",
        "12:00-15:00",
        "15:00-18:00",
        "18:00-21:00",
    ]

    # Test timeouts
    TEST_TIMEOUT_MINUTES = 60
    TEST_EXTENSION_MINUTES = 30
    QUESTION_TIMEOUT_MINUTES = 5

    # Session timeouts
    SESSION_TIMEOUT_MINUTES = 30
    IDLE_TIMEOUT_MINUTES = 15

    # Cache TTL (in seconds)
    CACHE_TTL_SHORT = 300      # 5 minutes
    CACHE_TTL_MEDIUM = 3600    # 1 hour
    CACHE_TTL_LONG = 86400     # 24 hours
    CACHE_TTL_WEEK = 604800    # 7 days


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

class ValidationConstants:
    """Validation rules and patterns."""

    # Phone number validation (Indian format)
    PHONE_PATTERN = re.compile(r"^[6-9]\d{9}$")  # Indian mobile numbers
    PHONE_LENGTH = 10

    # Name validation
    NAME_MIN_LENGTH = 2
    NAME_MAX_LENGTH = 100
    NAME_PATTERN = re.compile(r"^[a-zA-Z\s\.\-']+$")

    # Age validation
    MIN_AGE = 13
    MAX_AGE = 35

    # RAISEC code validation
    RAISEC_CODE_PATTERN = re.compile(r"^[RIASEC]{3}$")
    RAISEC_CODE_LENGTH = 3

    # Text length limits
    MAX_TEXT_SHORT = 200
    MAX_TEXT_MEDIUM = 1000
    MAX_TEXT_LONG = 5000

    # Array limits
    MAX_ARRAY_LENGTH = 100
    MAX_TAGS_COUNT = 20

    # File upload limits
    MAX_FILE_SIZE_MB = 10
    ALLOWED_FILE_EXTENSIONS = [".pdf", ".doc", ".docx", ".txt", ".csv"]
    ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp"]


# ============================================================================
# BUSINESS LOGIC CONSTANTS
# ============================================================================

class BusinessConstants:
    """Business logic constants."""

    # Test configuration
    TOTAL_QUESTIONS = 12
    MIN_QUESTIONS_FOR_SCORING = 10
    MAX_SKIP_ALLOWED = 2

    # Career recommendations
    CAREER_RECOMMENDATION_COUNT = 3
    MIN_CAREERS_IN_DATABASE = 100

    # Report configuration
    REPORT_EXPIRY_DAYS = 90
    REPORT_ACCESS_CODE_LENGTH = 8
    MAX_REPORT_VIEWS = 1000

    # User limits
    MAX_TESTS_PER_DAY = 5
    MAX_CONCURRENT_SESSIONS = 3

    # Rate limiting
    RATE_LIMIT_DEFAULT = 100  # requests per hour
    RATE_LIMIT_AUTH = 5       # auth attempts per 5 minutes
    RATE_LIMIT_TEST = 10      # tests per hour


# ============================================================================
# ERROR CODES
# ============================================================================

class ErrorCodes:
    """Standardized error codes."""

    # Authentication errors (1000-1099)
    INVALID_CREDENTIALS = 1001
    TOKEN_EXPIRED = 1002
    TOKEN_INVALID = 1003
    UNAUTHORIZED = 1004
    FORBIDDEN = 1005

    # Validation errors (1100-1199)
    INVALID_INPUT = 1101
    MISSING_FIELD = 1102
    INVALID_FIELD_VALUE = 1103
    FIELD_TOO_LONG = 1104
    INVALID_PHONE = 1105
    INVALID_AGE = 1106

    # Resource errors (1200-1299)
    USER_NOT_FOUND = 1201
    TEST_NOT_FOUND = 1202
    QUESTION_NOT_FOUND = 1203
    REPORT_NOT_FOUND = 1204
    CAREER_NOT_FOUND = 1205

    # Business logic errors (1300-1399)
    TEST_EXPIRED = 1301
    TEST_COMPLETED = 1302
    INVALID_TEST_STATE = 1303
    QUESTIONS_NOT_READY = 1304
    ALL_QUESTIONS_REQUIRED = 1305
    MAX_ATTEMPTS_EXCEEDED = 1306

    # System errors (1400-1499)
    DATABASE_ERROR = 1401
    CACHE_ERROR = 1402
    EXTERNAL_SERVICE_ERROR = 1403
    INTERNAL_ERROR = 1404
    SERVICE_UNAVAILABLE = 1405

    # Rate limiting errors (1500-1599)
    RATE_LIMIT_EXCEEDED = 1501
    TOO_MANY_REQUESTS = 1502


# ============================================================================
# HTTP STATUS CODE MAPPINGS
# ============================================================================

HTTP_STATUS_MAPPINGS = {
    # Success
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content",

    # Client errors
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",

    # Server errors
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


# ============================================================================
# API RESPONSE FORMATS
# ============================================================================

class ResponseFormats:
    """Standard API response formats."""

    SUCCESS = {
        "success": True,
        "data": None,
        "message": None,
        "meta": None,
    }

    ERROR = {
        "success": False,
        "error": {
            "code": None,
            "message": None,
            "details": None,
        },
        "meta": None,
    }

    PAGINATED = {
        "success": True,
        "data": [],
        "pagination": {
            "page": 1,
            "limit": 20,
            "total": 0,
            "pages": 0,
            "has_next": False,
            "has_prev": False,
        },
        "meta": None,
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_raisec_code_from_scores(scores: Dict[RaisecDimension, float]) -> str:
    """Generate RAISEC code from dimension scores.

    Args:
        scores: Dictionary of dimension scores

    Returns:
        str: 3-letter RAISEC code
    """
    sorted_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return "".join([dim.value for dim, _ in sorted_dims[:3]])


def validate_raisec_code(code: str) -> bool:
    """Validate RAISEC code format.

    Args:
        code: RAISEC code to validate

    Returns:
        bool: True if valid
    """
    return bool(ValidationConstants.RAISEC_CODE_PATTERN.match(code))


def validate_phone_number(phone: str) -> bool:
    """Validate Indian phone number format.

    Args:
        phone: Phone number to validate

    Returns:
        bool: True if valid
    """
    return bool(ValidationConstants.PHONE_PATTERN.match(phone))


def get_age_group_from_age(age: int) -> AgeGroup:
    """Get age group from numeric age.

    Args:
        age: User's age

    Returns:
        AgeGroup: Corresponding age group

    Raises:
        ValueError: If age is outside supported ranges
    """
    return AgeGroup.from_age(age)


# Export all constants and enums
__all__ = [
    # Enums
    "AgeGroup",
    "RaisecDimension",
    "QuestionType",
    "TestStatus",
    "UserAccountType",
    "ReportType",
    "PromptType",
    "RecommendationType",

    # Mappings
    "QUESTION_TYPE_DISTRIBUTION",
    "QUESTION_TYPE_DISPLAY_NAMES",
    "QUESTION_TYPE_DESCRIPTIONS",
    "RAISEC_DIMENSION_NAMES",
    "RAISEC_DIMENSION_DESCRIPTIONS",
    "RAISEC_DIMENSION_KEYWORDS",
    "TEST_STATUS_DISPLAY_NAMES",
    "TEST_STATUS_FLOW",
    "ACTIVE_TEST_STATUSES",
    "COMPLETED_TEST_STATUSES",

    # Constants classes
    "ScoringConstants",
    "TimeConstants",
    "ValidationConstants",
    "BusinessConstants",
    "ErrorCodes",
    "ResponseFormats",

    # Utility functions
    "get_raisec_code_from_scores",
    "validate_raisec_code",
    "validate_phone_number",
    "get_age_group_from_age",

    # Other mappings
    "HTTP_STATUS_MAPPINGS",
]
