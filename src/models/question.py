"""Question and Answer models for the TruScholar RAISEC assessment.

This module defines models for test questions and user answers, supporting
various question types and scoring mechanisms.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import QuestionType, RaisecDimension


class QuestionOption(EmbeddedDocument):
    """Individual option for multiple choice questions."""

    id: str = Field(..., pattern="^[a-z]$")  # a, b, c, d, etc.
    text: str = Field(..., min_length=1, max_length=500)
    image_url: Optional[str] = None
    is_correct: bool = Field(default=False)  # For validation purposes
    dimension_weights: Dict[RaisecDimension, float] = Field(default_factory=dict)


class LikertStatement(EmbeddedDocument):
    """Individual statement for Likert scale questions."""

    id: int = Field(..., ge=0, le=9)
    text: str = Field(..., min_length=1, max_length=300)
    dimension: RaisecDimension
    reverse_scored: bool = Field(default=False)


class PlotDayTask(EmbeddedDocument):
    """Task option for PLOT_DAY question type."""

    id: str = Field(..., min_length=1, max_length=50)
    title: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=200)
    icon: Optional[str] = None  # Icon identifier
    primary_dimension: RaisecDimension
    secondary_dimensions: List[RaisecDimension] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "PlotDayTask":
        """Ensure primary dimension is not in secondary dimensions."""
        if self.primary_dimension in self.secondary_dimensions:
            self.secondary_dimensions.remove(self.primary_dimension)
        return self


class ScoringRule(EmbeddedDocument):
    """Scoring rules for different question types."""

    # Points allocation
    single_dimension_points: float = Field(default=10.0, ge=0)
    primary_dimension_points: float = Field(default=6.0, ge=0)
    secondary_dimension_points: float = Field(default=4.0, ge=0)

    # Likert scale mapping (1-5 scale)
    likert_scale_map: Dict[int, float] = Field(
        default_factory=lambda: {1: 0, 2: 2.5, 3: 5, 4: 7.5, 5: 10}
    )

    # Scale rating mapping (1-10 scale)
    scale_rating_multiplier: float = Field(default=1.0, ge=0)

    # Plot day scoring
    plot_day_task_points: float = Field(default=5.0, ge=0)
    plot_day_time_slot_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "9:00-12:00": 1.2,    # Morning - higher weight
            "12:00-15:00": 1.0,   # Afternoon - normal weight
            "15:00-18:00": 1.0,   # Late afternoon - normal weight
            "18:00-21:00": 0.8,   # Evening - lower weight
        }
    )


class Question(BaseDocument):
    """Question model for RAISEC assessment.

    Supports multiple question types with appropriate data structures
    and scoring mechanisms.
    """

    # Test reference
    test_id: PyObjectId = Field(..., description="Reference to Test")

    # Question metadata
    question_number: int = Field(..., ge=1, le=12)
    question_type: QuestionType
    question_text: str = Field(..., min_length=10, max_length=1000)
    instructions: Optional[str] = Field(default=None, max_length=500)

    # Age group specific
    age_group: str = Field(..., pattern="^(13-17|18-25|26-35)$")

    # Question content based on type
    options: List[QuestionOption] = Field(default_factory=list)  # For MCQ types
    statements: List[LikertStatement] = Field(default_factory=list)  # For statement sets
    tasks: List[PlotDayTask] = Field(default_factory=list)  # For plot day

    # Binary choice for THIS_OR_THAT
    option_a: Optional[QuestionOption] = None
    option_b: Optional[QuestionOption] = None

    # Scale parameters for SCALE_RATING
    scale_min: int = Field(default=1, ge=1)
    scale_max: int = Field(default=10, le=10)
    scale_labels: Dict[str, str] = Field(default_factory=dict)  # e.g., {"1": "Not at all", "10": "Extremely"}

    # Dimensions being evaluated
    dimensions_evaluated: List[RaisecDimension] = Field(..., min_length=1, max_length=6)
    primary_dimension: Optional[RaisecDimension] = None

    # Scoring configuration
    scoring_rule: ScoringRule = Field(default_factory=ScoringRule)

    # Generation metadata
    generated_by: str = Field(default="gpt-4-turbo-preview")
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_static: bool = Field(default=False)  # True if from static questions

    # Validation and quality
    is_validated: bool = Field(default=True)
    validation_score: Optional[float] = Field(default=None, ge=0, le=100)

    # Display configuration
    display_order: Optional[int] = None
    time_limit_seconds: Optional[int] = None
    is_required: bool = Field(default=True)
    allow_skip: bool = Field(default=False)

    @field_validator("question_type", mode="before")
    @classmethod
    def validate_question_type(cls, value: Any) -> QuestionType:
        """Validate and convert question type."""
        if isinstance(value, str):
            return QuestionType(value)
        return value

    @model_validator(mode="after")
    def validate_question_content(self) -> "Question":
        """Validate question content based on type."""
        if self.question_type in [QuestionType.MCQ, QuestionType.SCENARIO_MCQ]:
            if len(self.options) < 2:
                raise ValueError(f"{self.question_type} must have at least 2 options")

        elif self.question_type == QuestionType.STATEMENT_SET:
            if len(self.statements) < 3:
                raise ValueError("Statement set must have at least 3 statements")

        elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            if len(self.options) < 3:
                raise ValueError("Multi-select must have at least 3 options")

        elif self.question_type == QuestionType.THIS_OR_THAT:
            if not self.option_a or not self.option_b:
                raise ValueError("This or That must have both options A and B")

        elif self.question_type == QuestionType.PLOT_DAY:
            if len(self.tasks) < 8:
                raise ValueError("Plot day must have at least 8 tasks")

        # Set primary dimension if not set
        if not self.primary_dimension and self.dimensions_evaluated:
            self.primary_dimension = self.dimensions_evaluated[0]

        return self

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the question collection."""
        return [
            ([("test_id", 1), ("question_number", 1)], {"unique": True}),
            ([("test_id", 1)], {}),
            ([("question_type", 1)], {}),
            ([("age_group", 1)], {}),
            ([("is_static", 1)], {}),
        ]

    def get_answer_format(self) -> Dict[str, Any]:
        """Get the expected answer format for this question type.

        Returns:
            Dictionary describing expected answer structure
        """
        if self.question_type == QuestionType.MCQ:
            return {
                "type": "single_choice",
                "format": {"selected_option": "string (option id)"},
                "options": [opt.id for opt in self.options],
            }

        elif self.question_type == QuestionType.STATEMENT_SET:
            return {
                "type": "likert_ratings",
                "format": {"ratings": "dict[statement_id: int(1-5)]"},
                "statements": [stmt.id for stmt in self.statements],
            }

        elif self.question_type == QuestionType.SCENARIO_MCQ:
            return {
                "type": "single_choice",
                "format": {"selected_option": "string (option id)"},
                "options": [opt.id for opt in self.options],
            }

        elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            return {
                "type": "multi_choice",
                "format": {"selected_options": "list[string] (option ids)"},
                "options": [opt.id for opt in self.options],
                "min_selections": 1,
                "max_selections": len(self.options),
            }

        elif self.question_type == QuestionType.THIS_OR_THAT:
            return {
                "type": "binary_choice",
                "format": {"selected": "string (a or b)"},
                "options": ["a", "b"],
            }

        elif self.question_type == QuestionType.SCALE_RATING:
            return {
                "type": "scale",
                "format": {"rating": f"int({self.scale_min}-{self.scale_max})"},
                "min": self.scale_min,
                "max": self.scale_max,
            }

        elif self.question_type == QuestionType.PLOT_DAY:
            return {
                "type": "time_allocation",
                "format": {
                    "placements": {
                        "9:00-12:00": "list[task_id]",
                        "12:00-15:00": "list[task_id]",
                        "15:00-18:00": "list[task_id]",
                        "18:00-21:00": "list[task_id]",
                        "not_interested": "list[task_id]",
                    }
                },
                "tasks": [task.id for task in self.tasks],
            }

        return {}

    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display to user.

        Returns:
            Dictionary with display-ready question data
        """
        base_dict = {
            "question_id": str(self.id),
            "question_number": self.question_number,
            "question_type": self.question_type.value,
            "question_text": self.question_text,
            "instructions": self.instructions,
            "is_required": self.is_required,
            "allow_skip": self.allow_skip,
        }

        # Add type-specific content
        if self.question_type in [QuestionType.MCQ, QuestionType.SCENARIO_MCQ, QuestionType.SCENARIO_MULTI_SELECT]:
            base_dict["options"] = [
                {
                    "id": opt.id,
                    "text": opt.text,
                    "image_url": opt.image_url,
                }
                for opt in self.options
            ]

        elif self.question_type == QuestionType.STATEMENT_SET:
            base_dict["statements"] = [
                {
                    "id": stmt.id,
                    "text": stmt.text,
                }
                for stmt in self.statements
            ]
            base_dict["scale"] = {
                "min": 1,
                "max": 5,
                "labels": {
                    "1": "Strongly Disagree",
                    "2": "Disagree",
                    "3": "Neutral",
                    "4": "Agree",
                    "5": "Strongly Agree",
                }
            }

        elif self.question_type == QuestionType.THIS_OR_THAT:
            base_dict["option_a"] = {
                "id": "a",
                "text": self.option_a.text,
                "image_url": self.option_a.image_url,
            }
            base_dict["option_b"] = {
                "id": "b",
                "text": self.option_b.text,
                "image_url": self.option_b.image_url,
            }

        elif self.question_type == QuestionType.SCALE_RATING:
            base_dict["scale"] = {
                "min": self.scale_min,
                "max": self.scale_max,
                "labels": self.scale_labels,
            }

        elif self.question_type == QuestionType.PLOT_DAY:
            base_dict["tasks"] = [
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "icon": task.icon,
                }
                for task in self.tasks
            ]
            base_dict["time_slots"] = [
                "9:00-12:00",
                "12:00-15:00",
                "15:00-18:00",
                "18:00-21:00",
            ]

        if self.time_limit_seconds:
            base_dict["time_limit_seconds"] = self.time_limit_seconds

        return base_dict


class Answer(BaseDocument):
    """User's answer to a question."""

    # References
    test_id: PyObjectId = Field(..., description="Reference to Test")
    question_id: PyObjectId = Field(..., description="Reference to Question")
    user_id: PyObjectId = Field(..., description="Reference to User")

    # Question metadata (denormalized for performance)
    question_number: int = Field(..., ge=1, le=12)
    question_type: QuestionType

    # Answer content (varies by question type)
    answer_data: Dict[str, Any] = Field(..., description="Answer content based on question type")

    # Timing
    time_spent_seconds: int = Field(default=0, ge=0)
    answered_at: datetime = Field(default_factory=datetime.utcnow)

    # Scoring
    dimension_scores: Dict[RaisecDimension, float] = Field(default_factory=dict)
    total_points: float = Field(default=0.0, ge=0)
    is_scored: bool = Field(default=False)

    # Validation
    is_valid: bool = Field(default=True)
    validation_errors: List[str] = Field(default_factory=list)

    # User behavior
    changed_count: int = Field(default=0, ge=0)  # Number of times answer was changed
    is_skipped: bool = Field(default=False)
    skip_reason: Optional[str] = None

    @field_validator("question_type", mode="before")
    @classmethod
    def validate_question_type(cls, value: Any) -> QuestionType:
        """Validate and convert question type."""
        if isinstance(value, str):
            return QuestionType(value)
        return value

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the answer collection."""
        return [
            ([("test_id", 1), ("question_id", 1)], {"unique": True}),
            ([("test_id", 1), ("question_number", 1)], {}),
            ([("user_id", 1)], {}),
            ([("is_scored", 1)], {}),
            ([("answered_at", -1)], {}),
        ]

    def validate_answer(self, question: Question) -> bool:
        """Validate answer against question requirements.

        Args:
            question: The question this answer is for

        Returns:
            bool: True if answer is valid
        """
        self.validation_errors = []

        try:
            if self.question_type == QuestionType.MCQ:
                selected = self.answer_data.get("selected_option")
                valid_options = [opt.id for opt in question.options]
                if selected not in valid_options:
                    self.validation_errors.append(f"Invalid option: {selected}")

            elif self.question_type == QuestionType.STATEMENT_SET:
                ratings = self.answer_data.get("ratings", {})
                for stmt in question.statements:
                    rating = ratings.get(str(stmt.id))
                    if rating is None:
                        self.validation_errors.append(f"Missing rating for statement {stmt.id}")
                    elif not (1 <= rating <= 5):
                        self.validation_errors.append(f"Invalid rating {rating} for statement {stmt.id}")

            elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
                selected = self.answer_data.get("selected_options", [])
                valid_options = [opt.id for opt in question.options]
                for opt in selected:
                    if opt not in valid_options:
                        self.validation_errors.append(f"Invalid option: {opt}")
                if len(selected) == 0:
                    self.validation_errors.append("At least one option must be selected")

            elif self.question_type == QuestionType.THIS_OR_THAT:
                selected = self.answer_data.get("selected")
                if selected not in ["a", "b"]:
                    self.validation_errors.append(f"Invalid selection: {selected}")

            elif self.question_type == QuestionType.SCALE_RATING:
                rating = self.answer_data.get("rating")
                if not isinstance(rating, int):
                    self.validation_errors.append("Rating must be an integer")
                elif not (question.scale_min <= rating <= question.scale_max):
                    self.validation_errors.append(f"Rating {rating} out of range")

            elif self.question_type == QuestionType.PLOT_DAY:
                placements = self.answer_data.get("placements", {})
                valid_tasks = [task.id for task in question.tasks]
                placed_tasks = []

                for time_slot, tasks in placements.items():
                    if time_slot not in ["9:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00", "not_interested"]:
                        self.validation_errors.append(f"Invalid time slot: {time_slot}")
                    for task in tasks:
                        if task not in valid_tasks:
                            self.validation_errors.append(f"Invalid task: {task}")
                        if task in placed_tasks:
                            self.validation_errors.append(f"Task {task} placed multiple times")
                        placed_tasks.append(task)

        except Exception as e:
            self.validation_errors.append(f"Validation error: {str(e)}")

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid

    def calculate_scores(self, question: Question) -> Dict[RaisecDimension, float]:
        """Calculate dimension scores for this answer.

        Args:
            question: The question this answer is for

        Returns:
            Dictionary of dimension scores
        """
        scores = {}
        rule = question.scoring_rule

        if self.question_type == QuestionType.MCQ:
            selected = self.answer_data.get("selected_option")
            for opt in question.options:
                if opt.id == selected:
                    for dim, weight in opt.dimension_weights.items():
                        scores[dim] = weight * rule.single_dimension_points

        elif self.question_type == QuestionType.STATEMENT_SET:
            ratings = self.answer_data.get("ratings", {})
            for stmt in question.statements:
                rating = ratings.get(str(stmt.id), 3)  # Default to neutral
                if stmt.reverse_scored:
                    rating = 6 - rating  # Reverse the scale
                score = rule.likert_scale_map.get(rating, 5.0)
                scores[stmt.dimension] = scores.get(stmt.dimension, 0) + score

        elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            selected = self.answer_data.get("selected_options", [])
            for opt in question.options:
                if opt.id in selected:
                    for dim, weight in opt.dimension_weights.items():
                        scores[dim] = scores.get(dim, 0) + (weight * rule.secondary_dimension_points)

        elif self.question_type == QuestionType.THIS_OR_THAT:
            selected = self.answer_data.get("selected")
            option = question.option_a if selected == "a" else question.option_b
            for dim, weight in option.dimension_weights.items():
                scores[dim] = weight * rule.single_dimension_points

        elif self.question_type == QuestionType.SCALE_RATING:
            rating = self.answer_data.get("rating", 5)
            normalized_rating = rating / question.scale_max
            for dim in question.dimensions_evaluated:
                scores[dim] = normalized_rating * rule.single_dimension_points * rule.scale_rating_multiplier

        elif self.question_type == QuestionType.PLOT_DAY:
            placements = self.answer_data.get("placements", {})
            for time_slot, tasks in placements.items():
                if time_slot == "not_interested":
                    continue
                slot_weight = rule.plot_day_time_slot_weights.get(time_slot, 1.0)
                for task_id in tasks:
                    task = next((t for t in question.tasks if t.id == task_id), None)
                    if task:
                        # Primary dimension gets full points
                        scores[task.primary_dimension] = scores.get(task.primary_dimension, 0) + (
                            rule.plot_day_task_points * slot_weight
                        )
                        # Secondary dimensions get partial points
                        for dim in task.secondary_dimensions:
                            scores[dim] = scores.get(dim, 0) + (
                                rule.plot_day_task_points * slot_weight * 0.5
                            )

        self.dimension_scores = scores
        self.total_points = sum(scores.values())
        self.is_scored = True

        return scores

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary.

        Returns:
            Dictionary with answer summary
        """
        return {
            "question_number": self.question_number,
            "question_type": self.question_type.value,
            "answered_at": self.answered_at.isoformat(),
            "time_spent_seconds": self.time_spent_seconds,
            "total_points": self.total_points,
            "is_valid": self.is_valid,
            "is_skipped": self.is_skipped,
        }
