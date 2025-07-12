"""Answer model for the TruScholar RAISEC assessment.

This module defines the Answer model for storing and processing user responses
to assessment questions, including validation and scoring logic.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import QuestionType, RaisecDimension


class AnswerValidation(EmbeddedDocument):
    """Answer validation details and error tracking."""

    is_valid: bool = Field(default=True)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Validation metrics
    completeness_score: float = Field(default=100.0, ge=0, le=100)
    consistency_score: float = Field(default=100.0, ge=0, le=100)

    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


class AnswerMetrics(EmbeddedDocument):
    """Metrics and analytics for answer behavior."""

    # Time metrics
    first_view_timestamp: Optional[datetime] = None
    first_interaction_timestamp: Optional[datetime] = None
    final_submission_timestamp: Optional[datetime] = None
    total_time_seconds: int = Field(default=0, ge=0)
    active_time_seconds: int = Field(default=0, ge=0)

    # Interaction metrics
    focus_events: int = Field(default=0, ge=0)
    blur_events: int = Field(default=0, ge=0)
    change_events: int = Field(default=0, ge=0)

    # Answer changes
    initial_answer: Optional[Dict[str, Any]] = None
    answer_history: List[Dict[str, Any]] = Field(default_factory=list, max_length=10)
    revision_count: int = Field(default=0, ge=0)

    # Confidence metrics
    confidence_level: Optional[int] = Field(default=None, ge=1, le=5)
    hesitation_score: float = Field(default=0.0, ge=0, le=100)

    def record_change(self, previous_answer: Dict[str, Any]) -> None:
        """Record an answer change."""
        self.answer_history.append({
            "answer": previous_answer,
            "timestamp": datetime.utcnow(),
            "revision": self.revision_count
        })
        self.revision_count += 1
        self.change_events += 1


class DimensionScore(EmbeddedDocument):
    """Individual dimension scoring details."""

    dimension: RaisecDimension
    raw_score: float = Field(default=0.0, ge=0)
    weighted_score: float = Field(default=0.0, ge=0)
    confidence: float = Field(default=100.0, ge=0, le=100)
    contribution_percentage: float = Field(default=0.0, ge=0, le=100)


class Answer(BaseDocument):
    """User's answer to a question with comprehensive tracking and scoring."""

    # Core references
    test_id: PyObjectId = Field(..., description="Reference to Test")
    question_id: PyObjectId = Field(..., description="Reference to Question")
    user_id: PyObjectId = Field(..., description="Reference to User")

    # Question metadata (denormalized for performance)
    question_number: int = Field(..., ge=1, le=12)
    question_type: QuestionType
    age_group: str = Field(..., pattern="^(13-17|18-25|26-35)$")

    # Answer content based on question type
    answer_data: Dict[str, Any] = Field(..., description="Answer content")

    # Answer-specific data for different types
    selected_option: Optional[str] = None  # For MCQ, SCENARIO_MCQ, THIS_OR_THAT
    selected_options: List[str] = Field(default_factory=list)  # For SCENARIO_MULTI_SELECT
    ratings: Dict[str, int] = Field(default_factory=dict)  # For STATEMENT_SET
    scale_rating: Optional[int] = None  # For SCALE_RATING
    task_placements: Dict[str, List[str]] = Field(default_factory=dict)  # For PLOT_DAY

    # Validation
    validation: AnswerValidation = Field(default_factory=AnswerValidation)

    # Metrics
    metrics: AnswerMetrics = Field(default_factory=AnswerMetrics)

    # Scoring
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    total_points: float = Field(default=0.0, ge=0)
    is_scored: bool = Field(default=False)
    scored_at: Optional[datetime] = None
    scoring_version: str = Field(default="v1.0")

    # Status
    is_final: bool = Field(default=False)
    is_skipped: bool = Field(default=False)
    skip_reason: Optional[str] = None

    # Device/Context information
    device_type: Optional[str] = None  # mobile, tablet, desktop
    browser: Optional[str] = None
    screen_resolution: Optional[str] = None

    @field_validator("question_type", mode="before")
    @classmethod
    def validate_question_type(cls, value: Any) -> QuestionType:
        """Validate and convert question type."""
        if isinstance(value, str):
            return QuestionType(value)
        return value

    @model_validator(mode="after")
    def extract_answer_components(self) -> "Answer":
        """Extract answer components based on question type."""
        if not self.answer_data:
            return self

        if self.question_type in [QuestionType.MCQ, QuestionType.SCENARIO_MCQ]:
            self.selected_option = self.answer_data.get("selected_option")

        elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            self.selected_options = self.answer_data.get("selected_options", [])

        elif self.question_type == QuestionType.THIS_OR_THAT:
            self.selected_option = self.answer_data.get("selected")

        elif self.question_type == QuestionType.STATEMENT_SET:
            self.ratings = self.answer_data.get("ratings", {})

        elif self.question_type == QuestionType.SCALE_RATING:
            self.scale_rating = self.answer_data.get("rating")

        elif self.question_type == QuestionType.PLOT_DAY:
            self.task_placements = self.answer_data.get("placements", {})

        return self

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the answer collection."""
        return [
            # Unique constraint
            ([("test_id", 1), ("question_id", 1)], {"unique": True}),

            # Query indexes
            ([("test_id", 1), ("question_number", 1)], {}),
            ([("user_id", 1), ("created_at", -1)], {}),
            ([("question_type", 1), ("is_scored", 1)], {}),

            # Analytics indexes
            ([("age_group", 1), ("question_type", 1)], {}),
            ([("is_skipped", 1), ("skip_reason", 1)], {"sparse": True}),
            ([("metrics.total_time_seconds", 1)], {}),
            ([("metrics.revision_count", 1)], {}),
        ]

    def validate_answer_content(self) -> bool:
        """Validate answer content based on question type.

        Returns:
            bool: True if answer is valid
        """
        self.validation = AnswerValidation()

        try:
            if self.is_skipped:
                if not self.skip_reason:
                    self.validation.add_warning("Skipped without reason")
                return True

            if self.question_type == QuestionType.MCQ:
                if not self.selected_option:
                    self.validation.add_error("No option selected")
                elif not isinstance(self.selected_option, str):
                    self.validation.add_error("Selected option must be a string")

            elif self.question_type == QuestionType.STATEMENT_SET:
                if not self.ratings:
                    self.validation.add_error("No ratings provided")
                else:
                    for key, rating in self.ratings.items():
                        if not isinstance(rating, int) or rating < 1 or rating > 5:
                            self.validation.add_error(f"Invalid rating {rating} for statement {key}")

            elif self.question_type == QuestionType.SCENARIO_MCQ:
                if not self.selected_option:
                    self.validation.add_error("No scenario option selected")

            elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
                if not self.selected_options:
                    self.validation.add_error("No options selected")
                elif len(self.selected_options) == 0:
                    self.validation.add_error("At least one option must be selected")

            elif self.question_type == QuestionType.THIS_OR_THAT:
                if self.selected_option not in ["a", "b"]:
                    self.validation.add_error(f"Invalid selection: {self.selected_option}")

            elif self.question_type == QuestionType.SCALE_RATING:
                if self.scale_rating is None:
                    self.validation.add_error("No rating provided")
                elif not isinstance(self.scale_rating, int):
                    self.validation.add_error("Rating must be an integer")
                elif self.scale_rating < 1 or self.scale_rating > 10:
                    self.validation.add_error(f"Rating {self.scale_rating} out of range (1-10)")

            elif self.question_type == QuestionType.PLOT_DAY:
                if not self.task_placements:
                    self.validation.add_error("No task placements provided")
                else:
                    placed_tasks = []
                    valid_slots = ["9:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00", "not_interested"]

                    for slot, tasks in self.task_placements.items():
                        if slot not in valid_slots:
                            self.validation.add_error(f"Invalid time slot: {slot}")
                        for task in tasks:
                            if task in placed_tasks:
                                self.validation.add_error(f"Task {task} placed in multiple slots")
                            placed_tasks.append(task)

            # Calculate completeness score
            if self.validation.errors:
                self.validation.completeness_score = 0.0
            elif self.validation.warnings:
                self.validation.completeness_score = 80.0

        except Exception as e:
            self.validation.add_error(f"Validation exception: {str(e)}")

        return self.validation.is_valid

    def record_interaction(self, event_type: str, timestamp: Optional[datetime] = None) -> None:
        """Record user interaction with the question.

        Args:
            event_type: Type of interaction (view, focus, blur, change)
            timestamp: Timestamp of the interaction
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if event_type == "view" and not self.metrics.first_view_timestamp:
            self.metrics.first_view_timestamp = timestamp
        elif event_type == "interaction" and not self.metrics.first_interaction_timestamp:
            self.metrics.first_interaction_timestamp = timestamp
        elif event_type == "focus":
            self.metrics.focus_events += 1
        elif event_type == "blur":
            self.metrics.blur_events += 1
        elif event_type == "change":
            if self.metrics.initial_answer is None:
                self.metrics.initial_answer = self.answer_data.copy()
            else:
                self.metrics.record_change(self.answer_data.copy())

    def finalize_answer(self, time_spent: Optional[int] = None) -> None:
        """Finalize the answer for scoring.

        Args:
            time_spent: Total time spent on the question in seconds
        """
        self.is_final = True
        self.metrics.final_submission_timestamp = datetime.utcnow()

        if time_spent:
            self.metrics.total_time_seconds = time_spent

        # Calculate hesitation score based on changes and time
        if self.metrics.revision_count > 0:
            self.metrics.hesitation_score = min(
                100,
                (self.metrics.revision_count * 20) +
                (self.metrics.blur_events * 5)
            )

        self.update_timestamps()

    def skip_question(self, reason: Optional[str] = None) -> None:
        """Mark question as skipped.

        Args:
            reason: Reason for skipping
        """
        self.is_skipped = True
        self.skip_reason = reason or "User choice"
        self.is_final = True
        self.metrics.final_submission_timestamp = datetime.utcnow()
        self.update_timestamps()

    def calculate_dimension_scores(
        self,
        scoring_rules: Dict[str, Any],
        question_dimensions: List[RaisecDimension]
    ) -> List[DimensionScore]:
        """Calculate dimension scores for the answer.

        Args:
            scoring_rules: Scoring configuration from question
            question_dimensions: Dimensions evaluated by the question

        Returns:
            List of dimension scores
        """
        if self.is_skipped:
            return []

        scores = []
        total_score = 0.0

        # Calculate raw scores based on question type
        raw_scores = self._calculate_raw_scores(scoring_rules)

        # Create DimensionScore objects
        for dimension, raw_score in raw_scores.items():
            # Apply confidence based on answer behavior
            confidence = 100.0
            if self.metrics.revision_count > 2:
                confidence -= 10 * (self.metrics.revision_count - 2)
            if self.metrics.hesitation_score > 50:
                confidence -= (self.metrics.hesitation_score - 50) * 0.5
            confidence = max(confidence, 50.0)  # Minimum 50% confidence

            weighted_score = raw_score * (confidence / 100)
            total_score += weighted_score

            scores.append(DimensionScore(
                dimension=dimension,
                raw_score=raw_score,
                weighted_score=weighted_score,
                confidence=confidence
            ))

        # Calculate contribution percentages
        if total_score > 0:
            for score in scores:
                score.contribution_percentage = (score.weighted_score / total_score) * 100

        self.dimension_scores = scores
        self.total_points = total_score
        self.is_scored = True
        self.scored_at = datetime.utcnow()

        return scores

    def _calculate_raw_scores(self, rules: Dict[str, Any]) -> Dict[RaisecDimension, float]:
        """Calculate raw scores based on answer type.

        Args:
            rules: Scoring rules from question

        Returns:
            Dictionary of dimension scores
        """
        scores = {}

        if self.question_type == QuestionType.MCQ:
            # MCQ: Single dimension gets full points
            if self.selected_option:
                # This would need the actual question data to map option to dimension
                # For now, returning placeholder
                pass

        elif self.question_type == QuestionType.STATEMENT_SET:
            # Likert: Each statement contributes to its dimension
            likert_map = rules.get("likert_scale_map", {1: 0, 2: 2.5, 3: 5, 4: 7.5, 5: 10})
            for stmt_id, rating in self.ratings.items():
                # Would need statement-dimension mapping from question
                score = likert_map.get(rating, 5.0)
                # Placeholder for dimension mapping

        elif self.question_type == QuestionType.SCALE_RATING:
            # Scale: Normalized rating applied to all evaluated dimensions
            if self.scale_rating:
                normalized = self.scale_rating / 10.0
                base_points = rules.get("single_dimension_points", 10.0)
                score = normalized * base_points
                # Apply to all dimensions equally (would need question data)

        elif self.question_type == QuestionType.PLOT_DAY:
            # Plot day: Time slot weights and task dimensions
            slot_weights = rules.get("plot_day_time_slot_weights", {})
            task_points = rules.get("plot_day_task_points", 5.0)

            for slot, tasks in self.task_placements.items():
                if slot == "not_interested":
                    continue
                weight = slot_weights.get(slot, 1.0)
                # Would need task-dimension mapping from question

        # Note: Actual implementation would need the Question object
        # to properly map options/statements/tasks to dimensions

        return scores

    def get_summary(self) -> Dict[str, Any]:
        """Get answer summary for reporting.

        Returns:
            Dictionary with answer summary
        """
        summary = {
            "question_number": self.question_number,
            "question_type": self.question_type.value,
            "is_skipped": self.is_skipped,
            "is_valid": self.validation.is_valid,
            "total_points": self.total_points,
            "time_spent_seconds": self.metrics.total_time_seconds,
            "revision_count": self.metrics.revision_count,
            "confidence_average": sum(s.confidence for s in self.dimension_scores) / len(self.dimension_scores) if self.dimension_scores else 0,
        }

        # Add type-specific summary
        if self.question_type == QuestionType.MCQ:
            summary["selected_option"] = self.selected_option
        elif self.question_type == QuestionType.STATEMENT_SET:
            summary["average_rating"] = sum(self.ratings.values()) / len(self.ratings) if self.ratings else 0
        elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            summary["options_selected"] = len(self.selected_options)
        elif self.question_type == QuestionType.SCALE_RATING:
            summary["rating"] = self.scale_rating
        elif self.question_type == QuestionType.PLOT_DAY:
            summary["tasks_placed"] = sum(len(tasks) for slot, tasks in self.task_placements.items() if slot != "not_interested")

        return summary

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display.

        Returns:
            Dictionary with display-safe answer data
        """
        return {
            "question_number": self.question_number,
            "question_type": self.question_type.value,
            "answer_data": self.answer_data,
            "is_skipped": self.is_skipped,
            "skip_reason": self.skip_reason,
            "time_spent_seconds": self.metrics.total_time_seconds,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        """String representation of Answer."""
        return (
            f"<Answer(id={self.id}, test_id={self.test_id}, "
            f"question={self.question_number}, type={self.question_type.value}, "
            f"scored={self.is_scored})>"
        )
