"""Test management schemas for TruScholar API.

This module defines Pydantic schemas for test-related API requests and responses,
including test creation, progress tracking, and question management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.schemas.base import BaseResponse
from src.utils.constants import AgeGroup, QuestionType, TestStatus


# Request Schemas

class TestStartRequest(BaseModel):
    """Request schema for starting a new test."""

    age: int = Field(..., ge=13, le=99, description="User's age for age group determination")
    test_type: str = Field(default="raisec", description="Type of test to start")
    is_practice: bool = Field(default=False, description="Whether this is a practice test")


class AnswerSubmissionRequest(BaseModel):
    """Request schema for submitting an answer to a question."""

    question_id: str = Field(..., description="ID of the question being answered")
    answer_data: Dict[str, Any] = Field(..., description="Answer content based on question type")
    time_spent_seconds: int = Field(default=0, ge=0, description="Time spent on question")

    # Optional tracking data
    changed_count: int = Field(default=0, ge=0, description="Number of times answer was changed")
    device_type: Optional[str] = Field(default=None, description="Device type used")


class TestSubmissionRequest(BaseModel):
    """Request schema for submitting a completed test."""

    test_id: str = Field(..., description="ID of the test being submitted")
    final_review: bool = Field(default=True, description="Whether user reviewed answers")
    feedback: Optional[str] = Field(default=None, max_length=500, description="Optional feedback")


class CareerInterestsRequest(BaseModel):
    """Request schema for submitting career interests."""

    interests_text: str = Field(..., min_length=10, max_length=1000, description="What they want to do")
    current_status: str = Field(..., min_length=5, max_length=500, description="What they're currently doing")


class ValidationQuestionsRequest(BaseModel):
    """Request schema for validation questions responses."""

    responses: List[bool] = Field(..., min_length=3, max_length=3, description="Yes/No responses to validation questions")


# Response Schemas

class QuestionResponse(BaseModel):
    """Response schema for a single question."""

    question_id: str
    question_number: int
    question_type: str
    question_text: str
    instructions: Optional[str] = None

    # Question-specific content (populated based on type)
    options: List[Dict[str, Any]] = Field(default_factory=list)
    statements: List[Dict[str, Any]] = Field(default_factory=list)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    option_a: Optional[Dict[str, Any]] = None
    option_b: Optional[Dict[str, Any]] = None
    scale: Optional[Dict[str, Any]] = None
    time_slots: List[str] = Field(default_factory=list)

    # Metadata
    is_required: bool = True
    allow_skip: bool = False
    time_limit_seconds: Optional[int] = None


class TestProgressResponse(BaseModel):
    """Response schema for test progress information."""

    test_id: str
    status: str
    current_question: int
    questions_answered: int
    total_questions: int = 12
    completion_percentage: float
    time_spent_minutes: float
    expires_at: str
    is_expired: bool

    # Progress milestones
    started_at: Optional[str] = None
    questions_ready_at: Optional[str] = None
    last_activity: Optional[str] = None


class TestResponse(BaseModel):
    """Response schema for complete test information."""

    test_id: str
    user_id: str
    age_group: str
    status: str
    progress: TestProgressResponse

    # Test components
    questions: List[QuestionResponse] = Field(default_factory=list)
    is_practice: bool = False

    # Timestamps
    created_at: str
    updated_at: str
    expires_at: str

    # Flags
    can_answer: bool
    can_submit: bool
    can_extend: bool


class AnswerResponse(BaseModel):
    """Response schema for answer submission."""

    answer_id: str
    question_id: str
    question_number: int
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    time_spent_seconds: int
    submitted_at: str


class TestScoresResponse(BaseModel):
    """Response schema for test scores."""

    test_id: str
    raisec_code: str
    dimension_scores: Dict[str, Dict[str, Any]]
    total_score: float
    consistency_score: float
    scored_at: str


class CareerInterestsResponse(BaseModel):
    """Response schema for career interests submission."""

    interests_submitted: bool
    validation_questions: List[str]
    submitted_at: str


class ValidationQuestionsResponse(BaseModel):
    """Response schema for validation questions."""

    questions: List[str] = Field(..., min_length=3, max_length=3)
    based_on_raisec: str
    question_id: str = "validation_questions"


class TestSummaryResponse(BaseModel):
    """Response schema for test summary."""

    test_id: str
    status: str
    completion_percentage: float
    raisec_code: Optional[str] = None
    completed_at: Optional[str] = None
    duration_minutes: Optional[float] = None
    can_view_results: bool = False


class TestListResponse(BaseModel):
    """Response schema for user's test list."""

    tests: List[TestSummaryResponse]
    total_tests: int
    completed_tests: int
    in_progress_tests: int


# Internal Processing Schemas

class QuestionGenerationContext(BaseModel):
    """Context for question generation."""

    age_group: AgeGroup
    question_number: int
    question_type: QuestionType
    previous_questions: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class ScoringContext(BaseModel):
    """Context for test scoring."""

    test_id: str
    answers: List[Dict[str, Any]]
    age_group: AgeGroup
    question_distribution: Dict[str, int]


class TestValidation(BaseModel):
    """Test validation results."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    completeness_score: float = Field(default=100.0, ge=0, le=100)


# Utility Schemas

class TestConfiguration(BaseModel):
    """Configuration for test generation."""

    question_distribution: Dict[QuestionType, int] = Field(
        default_factory=lambda: {
            QuestionType.MCQ: 2,
            QuestionType.STATEMENT_SET: 2,
            QuestionType.SCENARIO_MCQ: 2,
            QuestionType.SCENARIO_MULTI_SELECT: 2,
            QuestionType.THIS_OR_THAT: 2,
            QuestionType.SCALE_RATING: 1,
            QuestionType.PLOT_DAY: 1,
        }
    )

    time_limit_hours: int = Field(default=2, ge=1, le=6)
    allow_question_skip: bool = Field(default=False)
    require_all_answers: bool = Field(default=True)
    enable_question_timing: bool = Field(default=True)


class ErrorDetail(BaseModel):
    """Detailed error information."""

    error_code: str
    error_message: str
    error_context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Validation Methods

class TestRequestValidator:
    """Validator for test-related requests."""

    @staticmethod
    def validate_age_for_group(age: int) -> AgeGroup:
        """Validate age and return appropriate age group."""
        if 13 <= age <= 17:
            return AgeGroup.TEEN
        elif 18 <= age <= 25:
            return AgeGroup.YOUNG_ADULT
        elif 26 <= age <= 35:
            return AgeGroup.ADULT
        else:
            raise ValueError(f"Age {age} is not supported. Must be between 13-35.")

    @staticmethod
    def validate_answer_format(
        question_type: QuestionType,
        answer_data: Dict[str, Any]
    ) -> bool:
        """Validate answer format for question type."""
        if question_type == QuestionType.MCQ:
            return "selected_option" in answer_data
        elif question_type == QuestionType.STATEMENT_SET:
            return "ratings" in answer_data and isinstance(answer_data["ratings"], dict)
        elif question_type == QuestionType.SCENARIO_MCQ:
            return "selected_option" in answer_data
        elif question_type == QuestionType.SCENARIO_MULTI_SELECT:
            return "selected_options" in answer_data and isinstance(answer_data["selected_options"], list)
        elif question_type == QuestionType.THIS_OR_THAT:
            return "selected" in answer_data and answer_data["selected"] in ["a", "b"]
        elif question_type == QuestionType.SCALE_RATING:
            return "rating" in answer_data and isinstance(answer_data["rating"], int)
        elif question_type == QuestionType.PLOT_DAY:
            return "placements" in answer_data and isinstance(answer_data["placements"], dict)

        return False


# Export all schemas
__all__ = [
    # Request schemas
    "TestStartRequest",
    "AnswerSubmissionRequest",
    "TestSubmissionRequest",
    "CareerInterestsRequest",
    "ValidationQuestionsRequest",

    # Response schemas
    "QuestionResponse",
    "TestProgressResponse",
    "TestResponse",
    "AnswerResponse",
    "TestScoresResponse",
    "CareerInterestsResponse",
    "ValidationQuestionsResponse",
    "TestSummaryResponse",
    "TestListResponse",

    # Processing schemas
    "QuestionGenerationContext",
    "ScoringContext",
    "TestValidation",
    "TestConfiguration",
    "ErrorDetail",

    # Validators
    "TestRequestValidator",
]
