"""Test model for the TruScholar RAISEC assessment.

This module defines the Test model and related schemas for storing test
information, progress, and results in MongoDB.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import (
    AgeGroup,
    QuestionType,
    RaisecDimension,
    TestStatus,
)


class DimensionScore(EmbeddedDocument):
    """Individual RAISEC dimension score."""

    dimension: RaisecDimension
    raw_score: float = Field(default=0.0, ge=0.0)
    percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    rank: int = Field(default=0, ge=1, le=6)
    confidence: float = Field(default=0.0, ge=0.0, le=100.0)


class TestScores(EmbeddedDocument):
    """Complete test scoring results."""

    # Individual dimension scores
    realistic: DimensionScore = Field(
        default_factory=lambda: DimensionScore(dimension=RaisecDimension.REALISTIC)
    )
    artistic: DimensionScore = Field(
        default_factory=lambda: DimensionScore(dimension=RaisecDimension.ARTISTIC)
    )
    investigative: DimensionScore = Field(
        default_factory=lambda: DimensionScore(dimension=RaisecDimension.INVESTIGATIVE)
    )
    social: DimensionScore = Field(
        default_factory=lambda: DimensionScore(dimension=RaisecDimension.SOCIAL)
    )
    enterprising: DimensionScore = Field(
        default_factory=lambda: DimensionScore(dimension=RaisecDimension.ENTERPRISING)
    )
    conventional: DimensionScore = Field(
        default_factory=lambda: DimensionScore(dimension=RaisecDimension.CONVENTIONAL)
    )

    # Calculated RAISEC code (top 3)
    raisec_code: Optional[str] = Field(default=None, pattern="^[RIASEC]{3}$")

    # Secondary codes for nuanced analysis
    secondary_code: Optional[str] = Field(default=None, pattern="^[RIASEC]{3}$")

    # Overall test metrics
    total_score: float = Field(default=0.0, ge=0.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    differentiation_index: float = Field(default=0.0, ge=0.0, le=100.0)

    def get_dimension_scores(self) -> Dict[str, DimensionScore]:
        """Get all dimension scores as a dictionary."""
        return {
            "R": self.realistic,
            "A": self.artistic,
            "I": self.investigative,
            "S": self.social,
            "E": self.enterprising,
            "C": self.conventional,
        }

    def get_top_dimensions(self, count: int = 3) -> List[str]:
        """Get top scoring dimensions.

        Args:
            count: Number of top dimensions to return

        Returns:
            List of dimension codes sorted by score
        """
        scores = self.get_dimension_scores()
        sorted_dims = sorted(
            scores.items(),
            key=lambda x: (x[1].percentage, x[0]),
            reverse=True
        )
        return [dim for dim, _ in sorted_dims[:count]]


class TestProgress(EmbeddedDocument):
    """Test progress tracking."""

    current_question: int = Field(default=0, ge=0, le=12)
    questions_answered: int = Field(default=0, ge=0, le=12)
    questions_skipped: int = Field(default=0, ge=0)
    time_spent_seconds: int = Field(default=0, ge=0)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

    # Progress milestones
    started_at: Optional[datetime] = None
    questions_generated_at: Optional[datetime] = None
    first_question_answered_at: Optional[datetime] = None
    last_question_answered_at: Optional[datetime] = None
    scoring_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        return (self.questions_answered / 12) * 100 if self.questions_answered > 0 else 0

    @property
    def is_complete(self) -> bool:
        """Check if all questions are answered."""
        return self.questions_answered >= 12


class CareerInterests(EmbeddedDocument):
    """User's career interests and preferences."""

    interests_text: Optional[str] = Field(default=None, max_length=1000)
    current_status: Optional[str] = Field(default=None, max_length=500)

    # Extracted keywords for analysis
    interest_keywords: List[str] = Field(default_factory=list)
    status_keywords: List[str] = Field(default_factory=list)

    # Yes/No validation questions
    validation_questions: List[str] = Field(default_factory=list, max_length=3)
    validation_responses: List[bool] = Field(default_factory=list, max_length=3)

    # Timestamp
    submitted_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_responses(self) -> "CareerInterests":
        """Ensure validation responses match questions."""
        if len(self.validation_responses) > len(self.validation_questions):
            self.validation_responses = self.validation_responses[:len(self.validation_questions)]
        return self


class TestMetadata(EmbeddedDocument):
    """Test metadata and configuration."""

    test_version: str = Field(default="1.0")
    prompt_version: str = Field(default="v1.0")

    # Question generation metadata
    generation_model: str = Field(default="gpt-4-turbo-preview")
    generation_temperature: float = Field(default=0.7)
    generation_attempts: int = Field(default=1, ge=1)
    fallback_used: bool = Field(default=False)

    # Scoring metadata
    scoring_algorithm: str = Field(default="weighted_raisec_v1")
    scoring_model: Optional[str] = None

    # Client information
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    platform: Optional[str] = None  # web, mobile, api

    # A/B testing flags
    experiment_group: Optional[str] = None
    feature_flags: Dict[str, bool] = Field(default_factory=dict)


class Test(BaseDocument):
    """Main test model for RAISEC assessment.

    Represents a single test instance with questions, answers, scores,
    and career recommendations.
    """

    # User reference
    user_id: PyObjectId = Field(..., description="Reference to User")

    # Test configuration
    age_group: AgeGroup = Field(..., description="User's age group")
    status: TestStatus = Field(default=TestStatus.CREATED)

    # Test components
    progress: TestProgress = Field(default_factory=TestProgress)
    scores: Optional[TestScores] = None
    career_interests: Optional[CareerInterests] = None
    metadata: TestMetadata = Field(default_factory=TestMetadata)

    # Question and answer references
    question_ids: List[PyObjectId] = Field(default_factory=list, max_length=12)
    answer_ids: List[PyObjectId] = Field(default_factory=list, max_length=12)

    # Career recommendations
    recommendation_ids: List[PyObjectId] = Field(default_factory=list, max_length=3)
    report_id: Optional[PyObjectId] = None

    # Timing
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=2))

    # Flags
    is_practice: bool = Field(default=False)
    is_complete: bool = Field(default=False)
    is_scored: bool = Field(default=False)
    has_recommendations: bool = Field(default=False)

    # Analytics
    completion_time_minutes: Optional[float] = None
    abandonment_reason: Optional[str] = None
    feedback_rating: Optional[int] = Field(default=None, ge=1, le=5)
    feedback_text: Optional[str] = Field(default=None, max_length=500)

    @field_validator("age_group", mode="before")
    @classmethod
    def validate_age_group(cls, value: Any) -> AgeGroup:
        """Validate and convert age group."""
        if isinstance(value, str):
            return AgeGroup(value)
        return value

    @model_validator(mode="after")
    def update_status_flags(self) -> "Test":
        """Update status flags based on test state."""
        # Update completion flag
        self.is_complete = (
            self.progress.is_complete and
            self.scores is not None and
            self.career_interests is not None
        )

        # Update scored flag
        self.is_scored = self.scores is not None and self.scores.raisec_code is not None

        # Update recommendations flag
        self.has_recommendations = len(self.recommendation_ids) > 0

        return self

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the test collection."""
        return [
            # Single field indexes
            ([("user_id", 1)], {}),
            ([("status", 1)], {}),
            ([("created_at", -1)], {}),
            ([("expires_at", 1)], {"expireAfterSeconds": 0}),

            # Compound indexes
            ([("user_id", 1), ("status", 1), ("created_at", -1)], {}),
            ([("age_group", 1), ("status", 1)], {}),
            ([("is_complete", 1), ("created_at", -1)], {}),

            # Analytics indexes
            ([("completion_time_minutes", 1)], {"sparse": True}),
            ([("feedback_rating", 1)], {"sparse": True}),
        ]

    def start_test(self) -> None:
        """Mark test as started."""
        self.status = TestStatus.IN_PROGRESS
        self.progress.started_at = datetime.utcnow()
        self.update_timestamps()

    def mark_questions_ready(self) -> None:
        """Mark that questions have been generated."""
        self.status = TestStatus.QUESTIONS_READY
        self.progress.questions_generated_at = datetime.utcnow()
        self.update_timestamps()

    def add_answer(self, answer_id: PyObjectId) -> None:
        """Add an answer to the test.

        Args:
            answer_id: The answer ID to add
        """
        if answer_id not in self.answer_ids:
            self.answer_ids.append(answer_id)
            self.progress.questions_answered += 1
            self.progress.last_activity = datetime.utcnow()

            if self.progress.first_question_answered_at is None:
                self.progress.first_question_answered_at = datetime.utcnow()

            self.progress.last_question_answered_at = datetime.utcnow()
            self.update_timestamps()

    def skip_question(self) -> None:
        """Record a skipped question."""
        self.progress.questions_skipped += 1
        self.progress.last_activity = datetime.utcnow()
        self.update_timestamps()

    def start_scoring(self) -> None:
        """Mark test as being scored."""
        self.status = TestStatus.SCORING
        self.progress.scoring_started_at = datetime.utcnow()
        self.update_timestamps()

    def complete_scoring(self, scores: TestScores) -> None:
        """Complete test scoring.

        Args:
            scores: The calculated test scores
        """
        self.scores = scores
        self.status = TestStatus.SCORED
        self.is_scored = True
        self.update_timestamps()

    def add_career_interests(self, interests: CareerInterests) -> None:
        """Add career interests to the test.

        Args:
            interests: Career interests data
        """
        interests.submitted_at = datetime.utcnow()
        self.career_interests = interests
        self.status = TestStatus.INTERESTS_SUBMITTED
        self.update_timestamps()

    def add_recommendations(self, recommendation_ids: List[PyObjectId]) -> None:
        """Add career recommendations.

        Args:
            recommendation_ids: List of recommendation IDs
        """
        self.recommendation_ids = recommendation_ids[:3]  # Max 3 recommendations
        self.has_recommendations = True
        self.status = TestStatus.RECOMMENDATIONS_READY
        self.update_timestamps()

    def complete_test(self, report_id: PyObjectId) -> None:
        """Mark test as completed.

        Args:
            report_id: The generated report ID
        """
        self.report_id = report_id
        self.status = TestStatus.COMPLETED
        self.is_complete = True
        self.progress.completed_at = datetime.utcnow()

        # Calculate completion time
        if self.progress.started_at:
            duration = self.progress.completed_at - self.progress.started_at
            self.completion_time_minutes = duration.total_seconds() / 60

        self.update_timestamps()

    def abandon_test(self, reason: Optional[str] = None) -> None:
        """Mark test as abandoned.

        Args:
            reason: Optional reason for abandonment
        """
        self.status = TestStatus.ABANDONED
        self.abandonment_reason = reason
        self.update_timestamps()

    def extend_expiry(self, hours: int = 2) -> None:
        """Extend test expiry time.

        Args:
            hours: Number of hours to extend
        """
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.update_timestamps()

    def can_answer_questions(self) -> bool:
        """Check if test is ready for answering questions."""
        return (
            self.status in [TestStatus.QUESTIONS_READY, TestStatus.IN_PROGRESS] and
            datetime.utcnow() < self.expires_at
        )

    def can_submit_interests(self) -> bool:
        """Check if test is ready for career interests submission."""
        return (
            self.is_scored and
            self.career_interests is None
        )

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get test progress summary.

        Returns:
            Dictionary with progress information
        """
        return {
            "test_id": str(self.id),
            "status": self.status.value,
            "progress": {
                "current_question": self.progress.current_question,
                "questions_answered": self.progress.questions_answered,
                "total_questions": 12,
                "completion_percentage": self.progress.completion_percentage,
                "time_spent_minutes": self.progress.time_spent_seconds / 60,
            },
            "expires_at": self.expires_at.isoformat(),
            "is_expired": datetime.utcnow() > self.expires_at,
        }

    def __repr__(self) -> str:
        """String representation of Test."""
        return (
            f"<Test(id={self.id}, user_id={self.user_id}, "
            f"status={self.status.value}, progress={self.progress.completion_percentage}%)>"
        )
