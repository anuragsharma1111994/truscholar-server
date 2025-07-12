"""Answer schemas for TruScholar API.

This module defines Pydantic schemas for answer-related requests and responses,
including answer submission, validation, and scoring.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from bson import ObjectId

from src.schemas.base import BaseSchema
from src.utils.enums import QuestionType, RaisecDimension


# Answer content schemas for different question types

class MCQAnswerData(BaseSchema):
    """Answer data for multiple choice questions."""
    
    selected_option: str = Field(
        ..., 
        pattern="^[a-d]$", 
        description="Selected option ID (a, b, c, or d)"
    )
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "selected_option": "a"
            }
        }
    }


class StatementSetAnswerData(BaseSchema):
    """Answer data for statement rating questions."""
    
    ratings: Dict[str, int] = Field(
        ..., 
        description="Statement ratings (statement_id -> rating)"
    )
    
    @field_validator("ratings")
    @classmethod
    def validate_ratings(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate rating values are within range."""
        for statement_id, rating in v.items():
            if not 1 <= rating <= 5:
                raise ValueError(f"Rating for statement {statement_id} must be between 1 and 5")
        return v
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "ratings": {
                    "1": 4,
                    "2": 3,
                    "3": 5,
                    "4": 2,
                    "5": 4
                }
            }
        }
    }


class ScenarioMCQAnswerData(BaseSchema):
    """Answer data for scenario-based MCQ."""
    
    selected_option: str = Field(
        ..., 
        pattern="^[a-d]$", 
        description="Selected option ID"
    )
    confidence_level: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer (0-1)"
    )
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "selected_option": "b",
                "confidence_level": 0.8
            }
        }
    }


class ScenarioMultiSelectAnswerData(BaseSchema):
    """Answer data for scenario multi-select questions."""
    
    selected_options: List[str] = Field(
        ..., 
        min_length=1,
        max_length=6,
        description="List of selected option IDs"
    )
    
    @field_validator("selected_options")
    @classmethod
    def validate_options(cls, v: List[str]) -> List[str]:
        """Validate option IDs and check for duplicates."""
        if len(set(v)) != len(v):
            raise ValueError("Duplicate options selected")
        for option in v:
            if not option.isalpha() or len(option) != 1:
                raise ValueError(f"Invalid option ID: {option}")
        return v
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "selected_options": ["a", "c", "e"]
            }
        }
    }


class ThisOrThatAnswerData(BaseSchema):
    """Answer data for binary choice questions."""
    
    selected: str = Field(
        ..., 
        pattern="^[AB]$", 
        description="Selected option (A or B)"
    )
    decision_time_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Time taken to decide in milliseconds"
    )
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "selected": "A",
                "decision_time_ms": 2500
            }
        }
    }


class ScaleRatingAnswerData(BaseSchema):
    """Answer data for scale rating questions."""
    
    rating: int = Field(
        ..., 
        ge=1, 
        le=10, 
        description="Rating value (1-10)"
    )
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "rating": 7
            }
        }
    }


class PlotDayAnswerData(BaseSchema):
    """Answer data for plot day scheduling questions."""
    
    placements: Dict[str, List[str]] = Field(
        ..., 
        description="Task placements (time_slot -> list of task_ids)"
    )
    not_interested: List[str] = Field(
        default_factory=list,
        description="Tasks placed in 'not interested' slot"
    )
    
    @model_validator(mode="after")
    def validate_placements(self) -> "PlotDayAnswerData":
        """Validate all tasks are placed and no duplicates."""
        all_placed_tasks = []
        
        # Collect all placed tasks
        for tasks in self.placements.values():
            all_placed_tasks.extend(tasks)
        all_placed_tasks.extend(self.not_interested)
        
        # Check for duplicates
        if len(set(all_placed_tasks)) != len(all_placed_tasks):
            raise ValueError("Duplicate task placements found")
            
        return self
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "placements": {
                    "morning": ["task1", "task3"],
                    "afternoon": ["task2", "task5"],
                    "late_afternoon": ["task4"],
                    "evening": ["task7"]
                },
                "not_interested": ["task6", "task8"]
            }
        }
    }


# Request schemas

class AnswerSubmitRequest(BaseSchema):
    """Request to submit an answer to a question."""
    
    question_id: str = Field(..., description="Question ID being answered")
    question_type: QuestionType = Field(..., description="Type of question")
    answer_data: Union[
        MCQAnswerData,
        StatementSetAnswerData,
        ScenarioMCQAnswerData,
        ScenarioMultiSelectAnswerData,
        ThisOrThatAnswerData,
        ScaleRatingAnswerData,
        PlotDayAnswerData
    ] = Field(..., discriminator="question_type", description="Answer data based on question type")
    time_spent_seconds: int = Field(..., ge=0, description="Time spent on the question")
    
    # Optional metadata
    device_type: Optional[str] = Field(None, description="Device used (mobile/desktop/tablet)")
    changed_count: int = Field(default=0, ge=0, description="Number of times answer was changed")
    
    @field_validator("question_id")
    @classmethod
    def validate_object_id(cls, v: str) -> str:
        """Validate MongoDB ObjectId format."""
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid question ID format")
        return v


class BulkAnswerSubmitRequest(BaseSchema):
    """Request to submit multiple answers at once."""
    
    test_id: str = Field(..., description="Test ID")
    answers: List[AnswerSubmitRequest] = Field(
        ..., 
        min_length=1,
        max_length=12,
        description="List of answers to submit"
    )
    
    @field_validator("test_id")
    @classmethod
    def validate_test_id(cls, v: str) -> str:
        """Validate MongoDB ObjectId format."""
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid test ID format")
        return v


# Response schemas

class AnswerValidationResult(BaseSchema):
    """Result of answer validation."""
    
    is_valid: bool = Field(..., description="Whether answer is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "is_valid": True,
                "errors": [],
                "warnings": ["Answer changed multiple times"]
            }
        }
    }


class AnswerScoreResult(BaseSchema):
    """Result of answer scoring."""
    
    dimension_scores: Dict[RaisecDimension, float] = Field(
        ..., 
        description="Scores for each RAISEC dimension"
    )
    total_score: float = Field(..., ge=0, description="Total score for this answer")
    weighted_score: float = Field(..., ge=0, description="Weighted score based on question type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the score")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "dimension_scores": {
                    "R": 0.8,
                    "I": 0.2
                },
                "total_score": 1.0,
                "weighted_score": 0.9,
                "confidence": 0.85
            }
        }
    }


class AnswerResponse(BaseSchema):
    """Complete answer response."""
    
    id: str = Field(..., description="Answer ID")
    test_id: str = Field(..., description="Test ID")
    question_id: str = Field(..., description="Question ID")
    question_number: int = Field(..., ge=1, le=12, description="Question number")
    question_type: QuestionType = Field(..., description="Type of question answered")
    
    # Answer data
    answer_data: Dict[str, Any] = Field(..., description="Submitted answer data")
    
    # Validation and scoring
    validation: AnswerValidationResult = Field(..., description="Validation result")
    score: Optional[AnswerScoreResult] = Field(None, description="Score if calculated")
    
    # Metadata
    time_spent_seconds: int = Field(..., description="Time spent on question")
    changed_count: int = Field(..., description="Number of times changed")
    device_type: Optional[str] = Field(None, description="Device used")
    
    # Timestamps
    submitted_at: datetime = Field(..., description="Submission timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "test_id": "507f1f77bcf86cd799439012",
                "question_id": "507f1f77bcf86cd799439013",
                "question_number": 1,
                "question_type": "mcq",
                "answer_data": {
                    "selected_option": "a"
                },
                "validation": {
                    "is_valid": True,
                    "errors": [],
                    "warnings": []
                },
                "score": {
                    "dimension_scores": {"R": 1.0},
                    "total_score": 1.0,
                    "weighted_score": 1.0,
                    "confidence": 0.9
                },
                "time_spent_seconds": 45,
                "changed_count": 0,
                "device_type": "desktop",
                "submitted_at": "2024-01-01T12:00:45Z"
            }
        }
    }


class AnswerListResponse(BaseSchema):
    """Response containing multiple answers."""
    
    answers: List[AnswerResponse] = Field(..., description="List of answers")
    total: int = Field(..., ge=0, description="Total number of answers")
    test_id: str = Field(..., description="Associated test ID")
    questions_answered: int = Field(..., ge=0, le=12, description="Number of questions answered")
    questions_remaining: int = Field(..., ge=0, le=12, description="Number of questions remaining")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "answers": [],
                "total": 5,
                "test_id": "507f1f77bcf86cd799439012",
                "questions_answered": 5,
                "questions_remaining": 7
            }
        }
    }


class AnswerStatistics(BaseSchema):
    """Statistics about answer patterns."""
    
    total_time_seconds: int = Field(..., ge=0, description="Total time spent")
    average_time_per_question: float = Field(..., ge=0, description="Average time per question")
    fastest_answer_seconds: int = Field(..., ge=0, description="Fastest answer time")
    slowest_answer_seconds: int = Field(..., ge=0, description="Slowest answer time")
    total_changes: int = Field(..., ge=0, description="Total number of answer changes")
    questions_by_type: Dict[QuestionType, int] = Field(..., description="Count by question type")
    completion_rate: float = Field(..., ge=0, le=1, description="Percentage of questions answered")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "total_time_seconds": 720,
                "average_time_per_question": 60,
                "fastest_answer_seconds": 15,
                "slowest_answer_seconds": 120,
                "total_changes": 3,
                "questions_by_type": {
                    "mcq": 2,
                    "statement_set": 2,
                    "scenario_mcq": 2
                },
                "completion_rate": 0.5
            }
        }
    }


# Utility schemas for answer processing

class AnswerContext(BaseSchema):
    """Context for answer processing."""
    
    test_id: str
    question_id: str
    question_type: QuestionType
    question_data: Dict[str, Any]
    user_age_group: str
    previous_answers: Optional[List[Dict[str, Any]]] = None


class AnswerProcessingResult(BaseSchema):
    """Result of answer processing."""
    
    answer_id: str
    is_processed: bool
    validation_result: AnswerValidationResult
    score_result: Optional[AnswerScoreResult] = None
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)


# Export all schemas
__all__ = [
    # Answer data schemas
    "MCQAnswerData",
    "StatementSetAnswerData",
    "ScenarioMCQAnswerData",
    "ScenarioMultiSelectAnswerData",
    "ThisOrThatAnswerData",
    "ScaleRatingAnswerData",
    "PlotDayAnswerData",
    
    # Request schemas
    "AnswerSubmitRequest",
    "BulkAnswerSubmitRequest",
    
    # Response schemas
    "AnswerValidationResult",
    "AnswerScoreResult",
    "AnswerResponse",
    "AnswerListResponse",
    "AnswerStatistics",
    
    # Utility schemas
    "AnswerContext",
    "AnswerProcessingResult",
]