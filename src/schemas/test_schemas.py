"""Test management schemas for TruScholar API.

This module defines Pydantic schemas for test-related API requests and responses,
including test creation, progress tracking, and question management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.schemas.base import BaseResponse
from src.utils.constants import AgeGroup, QuestionType, TestStatus, RaisecDimension


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


class DimensionScoreResponse(BaseModel):
    """Response schema for individual dimension scores."""
    
    dimension: str
    raw_score: float = Field(..., ge=0, description="Raw dimensional score")
    weighted_score: float = Field(..., ge=0, description="Score after applying weights")
    normalized_score: float = Field(..., ge=0, le=100, description="Normalized score (0-100)")
    confidence_level: float = Field(..., ge=0, le=100, description="Confidence in this score")
    question_count: int = Field(..., ge=0, description="Number of questions contributing")
    consistency_rating: str = Field(..., description="Consistency level (high/medium/low)")


class TestScoresResponse(BaseModel):
    """Response schema for comprehensive test scores."""

    test_id: str
    user_id: str
    raisec_code: str = Field(..., description="Primary RAISEC code (e.g., 'RIA')")
    raisec_profile: str = Field(..., description="Full RAISEC profile description")
    
    # Individual dimension scores
    dimension_scores: Dict[str, DimensionScoreResponse] = Field(
        ..., description="Scores for each RAISEC dimension"
    )
    
    # Overall scoring metrics
    total_score: float = Field(..., ge=0, description="Total composite score")
    average_score: float = Field(..., ge=0, le=100, description="Average normalized score")
    consistency_score: float = Field(..., ge=0, le=100, description="Overall consistency rating")
    confidence_score: float = Field(..., ge=0, le=100, description="Overall confidence in results")
    
    # Test completion metrics
    completion_percentage: float = Field(..., ge=0, le=100, description="Percentage of test completed")
    questions_answered: int = Field(..., ge=0, description="Number of questions answered")
    total_questions: int = Field(..., ge=0, description="Total questions in test")
    
    # Timing and behavior analysis
    total_time_minutes: float = Field(..., ge=0, description="Total time spent on test")
    average_time_per_question: float = Field(..., ge=0, description="Average seconds per question")
    timing_consistency: str = Field(..., description="Timing pattern analysis")
    
    # Scoring metadata
    scoring_version: str = Field(default="v2.0", description="Version of scoring algorithm used")
    scored_at: str = Field(..., description="When scoring was completed")
    scoring_notes: List[str] = Field(default_factory=list, description="Additional scoring notes")
    
    # Validity and reliability indicators
    validity_flags: List[str] = Field(default_factory=list, description="Any validity concerns")
    reliability_score: float = Field(..., ge=0, le=100, description="Test reliability indicator")


class ScoringAnalyticsResponse(BaseModel):
    """Response schema for detailed scoring analytics."""
    
    test_id: str
    
    # Answer pattern analysis
    answer_patterns: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Patterns in user responses"
    )
    
    # Time-based analysis
    timing_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed timing behavior analysis"
    )
    
    # Consistency analysis
    consistency_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Various consistency measurements"
    )
    
    # Question type performance
    performance_by_type: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance breakdown by question type"
    )
    
    # Dimensional insights
    dimensional_insights: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Insights for each RAISEC dimension"
    )
    
    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations based on analysis"
    )
    
    # Confidence indicators
    confidence_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence levels for different aspects"
    )


class ScoringExplanationResponse(BaseModel):
    """Response schema for scoring explanation."""
    
    test_id: str
    explanation_type: str = Field(..., description="Type of explanation (summary/detailed)")
    
    # Overall explanation
    overall_summary: str = Field(..., description="High-level summary of results")
    raisec_code_explanation: str = Field(..., description="Explanation of assigned RAISEC code")
    
    # Dimension explanations
    dimension_explanations: Dict[str, str] = Field(
        default_factory=dict,
        description="Explanation for each dimension score"
    )
    
    # Strengths and development areas
    key_strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    development_areas: List[str] = Field(default_factory=list, description="Areas for development")
    
    # Scoring methodology
    methodology_notes: List[str] = Field(
        default_factory=list, 
        description="Notes about scoring methodology"
    )
    
    # Caveats and limitations
    caveats: List[str] = Field(
        default_factory=list,
        description="Important caveats about the results"
    )
    
    generated_at: str = Field(..., description="When explanation was generated")


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
    user_timezone: Optional[str] = None
    scoring_parameters: Dict[str, Any] = Field(default_factory=dict)


class ScoringRequest(BaseModel):
    """Request schema for test scoring."""
    
    test_id: str = Field(..., description="ID of test to score")
    force_rescore: bool = Field(default=False, description="Force rescoring even if already scored")
    include_analytics: bool = Field(default=True, description="Include detailed analytics in response")
    scoring_version: Optional[str] = Field(default=None, description="Specific scoring version to use")
    explanation_level: str = Field(default="summary", description="Level of explanation (summary/detailed)")


class ScoringConfigurationUpdate(BaseModel):
    """Schema for updating scoring configuration."""
    
    question_weights: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Updated weights for question types"
    )
    time_adjustment_factors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Updated time adjustment factors"
    )
    consistency_thresholds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Updated consistency analysis thresholds"
    )
    confidence_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated confidence calculation parameters"
    )


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
    "ScoringRequest",
    "ScoringConfigurationUpdate",

    # Response schemas
    "QuestionResponse",
    "TestProgressResponse",
    "TestResponse",
    "AnswerResponse",
    "DimensionScoreResponse",
    "TestScoresResponse",
    "ScoringAnalyticsResponse",
    "ScoringExplanationResponse",
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
