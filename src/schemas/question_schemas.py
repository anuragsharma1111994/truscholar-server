"""Question schemas for TruScholar API.

This module defines Pydantic schemas for question-related requests and responses,
including AI-generated questions, question types, and RAISEC dimensions.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from bson import ObjectId

from src.schemas.base import BaseSchema
from src.utils.constants import AgeGroup, QuestionType, RaisecDimension


# Question content schemas for different types

class MCQOption(BaseSchema):
    """Multiple choice question option."""
    
    id: str = Field(..., description="Option identifier (A, B, C, D)")
    text: str = Field(..., min_length=1, max_length=500, description="Option text")
    raisec_dimension: RaisecDimension = Field(..., description="Associated RAISEC dimension")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "A",
                "text": "Build and repair mechanical devices",
                "raisec_dimension": "realistic"
            }
        }
    }


class StatementRating(BaseSchema):
    """Statement for rating scale."""
    
    id: str = Field(..., description="Statement identifier")
    text: str = Field(..., min_length=10, max_length=500, description="Statement text")
    raisec_dimension: RaisecDimension = Field(..., description="Associated RAISEC dimension")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "S1",
                "text": "I enjoy working with tools and machinery",
                "raisec_dimension": "realistic"
            }
        }
    }


class ScenarioOption(BaseSchema):
    """Scenario-based question option."""
    
    id: str = Field(..., description="Option identifier")
    text: str = Field(..., min_length=10, max_length=1000, description="Option description")
    raisec_dimensions: List[RaisecDimension] = Field(
        ..., 
        min_length=1,
        max_length=3,
        description="Associated RAISEC dimensions"
    )
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "A",
                "text": "Lead a team meeting to brainstorm solutions",
                "raisec_dimensions": ["enterprising", "social"]
            }
        }
    }


class ThisOrThatOption(BaseSchema):
    """Binary choice option."""
    
    id: str = Field(..., pattern="^[AB]$", description="Option A or B")
    text: str = Field(..., min_length=5, max_length=200, description="Option text")
    image_url: Optional[str] = Field(None, description="Optional image URL")
    raisec_dimension: RaisecDimension = Field(..., description="Associated RAISEC dimension")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "A",
                "text": "Working outdoors",
                "raisec_dimension": "realistic"
            }
        }
    }


class ScaleEndpoint(BaseSchema):
    """Scale rating endpoint description."""
    
    value: int = Field(..., ge=1, le=10, description="Scale value")
    label: str = Field(..., min_length=1, max_length=50, description="Label for this endpoint")
    description: Optional[str] = Field(None, max_length=200, description="Description")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "value": 1,
                "label": "Strongly Dislike",
                "description": "I would avoid this activity"
            }
        }
    }


class PlotDayTask(BaseSchema):
    """Task for daily schedule plotting."""
    
    id: str = Field(..., description="Task identifier")
    text: str = Field(..., min_length=5, max_length=200, description="Task description")
    category: str = Field(..., description="Task category")
    typical_duration: str = Field(..., description="Typical duration (e.g., '1-2 hours')")
    raisec_dimensions: List[RaisecDimension] = Field(
        ..., 
        min_length=1,
        max_length=2,
        description="Associated RAISEC dimensions"
    )
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "T1",
                "text": "Analyze data and create reports",
                "category": "analytical",
                "typical_duration": "2-3 hours",
                "raisec_dimensions": ["investigative", "conventional"]
            }
        }
    }


class PlotDayTimeSlot(BaseSchema):
    """Time slot for daily schedule."""
    
    id: str = Field(..., description="Slot identifier")
    time_range: str = Field(..., pattern=r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$", description="Time range")
    label: str = Field(..., description="Slot label")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "morning",
                "time_range": "9:00-12:00",
                "label": "Morning"
            }
        }
    }


# Request schemas

class QuestionGenerateRequest(BaseSchema):
    """Request to generate questions for a test."""
    
    test_id: str = Field(..., description="Test ID")
    age_group: AgeGroup = Field(..., description="Age group for question generation")
    question_types: Optional[Dict[QuestionType, int]] = Field(
        None,
        description="Optional override for question type distribution"
    )
    include_static: bool = Field(
        default=False,
        description="Include static fallback questions"
    )
    
    @field_validator("test_id")
    @classmethod
    def validate_object_id(cls, v: str) -> str:
        """Validate MongoDB ObjectId format."""
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid test ID format")
        return v


class QuestionRegenerateRequest(BaseSchema):
    """Request to regenerate a specific question."""
    
    test_id: str = Field(..., description="Test ID")
    question_number: int = Field(..., ge=1, le=12, description="Question number to regenerate")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for regeneration")
    
    @field_validator("test_id")
    @classmethod
    def validate_object_id(cls, v: str) -> str:
        """Validate MongoDB ObjectId format."""
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid test ID format")
        return v


# Response schemas

class QuestionContentResponse(BaseSchema):
    """Unified question content response."""
    
    # MCQ fields
    options: Optional[List[MCQOption]] = Field(None, description="MCQ options")
    
    # Statement set fields
    statements: Optional[List[StatementRating]] = Field(None, description="Statements to rate")
    scale_labels: Optional[Dict[str, str]] = Field(None, description="Scale labels")
    
    # Scenario fields
    scenario: Optional[str] = Field(None, description="Scenario description")
    scenario_options: Optional[List[ScenarioOption]] = Field(None, description="Scenario options")
    
    # This or That fields
    option_a: Optional[ThisOrThatOption] = Field(None, description="First option")
    option_b: Optional[ThisOrThatOption] = Field(None, description="Second option")
    
    # Scale rating fields
    scale_min: Optional[ScaleEndpoint] = Field(None, description="Minimum scale endpoint")
    scale_max: Optional[ScaleEndpoint] = Field(None, description="Maximum scale endpoint")
    scale_item: Optional[str] = Field(None, description="Item to rate")
    
    # Plot day fields
    tasks: Optional[List[PlotDayTask]] = Field(None, description="Tasks to schedule")
    time_slots: Optional[List[PlotDayTimeSlot]] = Field(None, description="Available time slots")
    special_slot: Optional[Dict[str, str]] = Field(None, description="Special slot (Not Interested)")
    
    @model_validator(mode="after")
    def validate_content_by_type(self) -> "QuestionContentResponse":
        """Ensure required fields are present based on inferred question type."""
        # This validation would be enhanced when integrated with actual question type
        return self


class QuestionResponse(BaseSchema):
    """Complete question response."""
    
    id: str = Field(..., description="Question ID")
    test_id: str = Field(..., description="Test ID")
    question_number: int = Field(..., ge=1, le=12, description="Question number in sequence")
    question_type: QuestionType = Field(..., description="Type of question")
    question_text: str = Field(..., min_length=10, description="Main question text")
    instructions: Optional[str] = Field(None, description="Instructions for answering")
    
    # Question content
    content: QuestionContentResponse = Field(..., description="Question-specific content")
    
    # Metadata
    raisec_dimensions: List[RaisecDimension] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="RAISEC dimensions evaluated"
    )
    time_estimate_seconds: Optional[int] = Field(
        None,
        ge=30,
        le=600,
        description="Estimated time to answer"
    )
    is_required: bool = Field(default=True, description="Whether question must be answered")
    generated_by: str = Field(..., description="Generation method (ai/static)")
    prompt_version: Optional[str] = Field(None, description="Prompt version used")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    regenerated_at: Optional[datetime] = Field(None, description="Last regeneration timestamp")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "test_id": "507f1f77bcf86cd799439012",
                "question_number": 1,
                "question_type": "mcq",
                "question_text": "Which of these activities appeals to you most?",
                "instructions": "Select the option that best describes your preference",
                "content": {
                    "options": [
                        {
                            "id": "A",
                            "text": "Building or fixing things",
                            "raisec_dimension": "realistic"
                        },
                        {
                            "id": "B",
                            "text": "Creating art or music",
                            "raisec_dimension": "artistic"
                        }
                    ]
                },
                "raisec_dimensions": ["realistic", "artistic"],
                "time_estimate_seconds": 60,
                "is_required": True,
                "generated_by": "ai",
                "prompt_version": "v1.0",
                "created_at": "2024-01-01T12:00:00Z"
            }
        }
    }


class QuestionListResponse(BaseSchema):
    """Response containing multiple questions."""
    
    questions: List[QuestionResponse] = Field(..., description="List of questions")
    total: int = Field(..., ge=0, description="Total number of questions")
    test_id: str = Field(..., description="Associated test ID")
    generation_complete: bool = Field(..., description="Whether all questions are generated")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "questions": [],
                "total": 12,
                "test_id": "507f1f77bcf86cd799439012",
                "generation_complete": True
            }
        }
    }


class QuestionGenerationStatusResponse(BaseSchema):
    """Response for question generation status."""
    
    test_id: str = Field(..., description="Test ID")
    status: str = Field(..., description="Generation status")
    questions_generated: int = Field(..., ge=0, le=12, description="Number of questions generated")
    total_questions: int = Field(default=12, description="Total questions to generate")
    estimated_completion_seconds: Optional[int] = Field(None, description="Estimated time remaining")
    errors: List[str] = Field(default_factory=list, description="Any generation errors")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "test_id": "507f1f77bcf86cd799439012",
                "status": "generating",
                "questions_generated": 8,
                "total_questions": 12,
                "estimated_completion_seconds": 10,
                "errors": []
            }
        }
    }


class QuestionValidationResponse(BaseSchema):
    """Response for question validation."""
    
    question_id: str = Field(..., description="Question ID")
    is_valid: bool = Field(..., description="Whether question is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "question_id": "507f1f77bcf86cd799439011",
                "is_valid": True,
                "errors": [],
                "warnings": ["Question text could be more specific"],
                "suggestions": ["Consider adding context about work environment"]
            }
        }
    }


# Static question schema for fallback

class StaticQuestionData(BaseSchema):
    """Schema for static question data files."""
    
    version: str = Field(..., description="Static data version")
    age_group: AgeGroup = Field(..., description="Target age group")
    question_type: QuestionType = Field(..., description="Question type")
    questions: List[Dict[str, Any]] = Field(..., description="Static question templates")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        **BaseSchema.model_config,
        "json_schema_extra": {
            "example": {
                "version": "1.0",
                "age_group": "18-25",
                "question_type": "mcq",
                "questions": [
                    {
                        "text": "What type of work environment do you prefer?",
                        "options": [
                            {"id": "A", "text": "Outdoor physical work", "dimension": "R"},
                            {"id": "B", "text": "Creative studio", "dimension": "A"}
                        ]
                    }
                ],
                "metadata": {
                    "created_date": "2024-01-01",
                    "author": "TruScholar Team"
                }
            }
        }
    }


# Export all schemas
__all__ = [
    # Content schemas
    "MCQOption",
    "StatementRating",
    "ScenarioOption",
    "ThisOrThatOption",
    "ScaleEndpoint",
    "PlotDayTask",
    "PlotDayTimeSlot",
    
    # Request schemas
    "QuestionGenerateRequest",
    "QuestionRegenerateRequest",
    
    # Response schemas
    "QuestionContentResponse",
    "QuestionResponse",
    "QuestionListResponse",
    "QuestionGenerationStatusResponse",
    "QuestionValidationResponse",
    
    # Static data schema
    "StaticQuestionData",
]