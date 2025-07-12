"""Utility formatters for question generation and display.

This module provides formatting utilities for converting between different
question data formats, cleaning LLM responses, and preparing data for display.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from src.utils.constants import QuestionType, RaisecDimension, AgeGroup
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuestionFormatter:
    """Formatter for question data and LLM responses."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.validation_patterns = {
            "question_text": r"^.{10,500}$",  # 10-500 characters
            "option_text": r"^.{5,200}$",    # 5-200 characters  
            "statement_text": r"^.{10,250}$", # 10-250 characters
            "task_title": r"^.{5,100}$",     # 5-100 characters
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        }
    
    def format_llm_response(
        self,
        response_data: Dict[str, Any],
        question_type: QuestionType
    ) -> Dict[str, Any]:
        """Format and validate LLM response data.
        
        Args:
            response_data: Raw response from LLM
            question_type: Type of question
            
        Returns:
            Dict[str, Any]: Formatted and validated response data
        """
        logger.debug(f"Formatting LLM response for {question_type.value}")
        
        # Clean and validate basic structure
        formatted = self._clean_response_structure(response_data)
        
        # Apply question type specific formatting
        if question_type == QuestionType.MCQ:
            formatted = self._format_mcq_response(formatted)
        elif question_type == QuestionType.STATEMENT_SET:
            formatted = self._format_statement_set_response(formatted)
        elif question_type == QuestionType.SCENARIO_MCQ:
            formatted = self._format_scenario_mcq_response(formatted)
        elif question_type == QuestionType.SCENARIO_MULTI_SELECT:
            formatted = self._format_scenario_multi_select_response(formatted)
        elif question_type == QuestionType.THIS_OR_THAT:
            formatted = self._format_this_or_that_response(formatted)
        elif question_type == QuestionType.SCALE_RATING:
            formatted = self._format_scale_rating_response(formatted)
        elif question_type == QuestionType.PLOT_DAY:
            formatted = self._format_plot_day_response(formatted)
        
        # Validate final result
        self._validate_formatted_response(formatted, question_type)
        
        return formatted
    
    def format_question_for_display(
        self,
        question_data: Dict[str, Any],
        age_group: AgeGroup,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Format question data for frontend display.
        
        Args:
            question_data: Question data
            age_group: Target age group
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Display-formatted question
        """
        display_data = {
            "id": question_data.get("id"),
            "question_number": question_data.get("question_number"),
            "question_type": question_data.get("question_type"),
            "question_text": self._format_text_for_age_group(
                question_data.get("question_text", ""), age_group
            ),
            "instructions": self._format_instructions(
                question_data.get("instructions"), age_group
            ),
            "time_estimate_seconds": question_data.get("estimated_time_seconds", 45),
            "is_required": question_data.get("is_required", True)
        }
        
        # Add content based on question type
        question_type = QuestionType(question_data.get("question_type"))
        
        if question_type == QuestionType.MCQ:
            display_data["options"] = self._format_mcq_options_for_display(
                question_data.get("options", []), age_group
            )
        elif question_type == QuestionType.STATEMENT_SET:
            display_data["statements"] = self._format_statements_for_display(
                question_data.get("statements", []), age_group
            )
            display_data["scale_info"] = self._get_likert_scale_info(age_group)
        elif question_type == QuestionType.PLOT_DAY:
            display_data["tasks"] = self._format_plot_day_tasks_for_display(
                question_data.get("tasks", []), age_group
            )
            display_data["time_slots"] = self._get_time_slots_for_age_group(age_group)
        
        # Add metadata if requested
        if include_metadata:
            display_data["metadata"] = {
                "generated_by": "ai" if not question_data.get("is_static", True) else "static",
                "dimensions_evaluated": question_data.get("dimensions_evaluated", []),
                "created_at": question_data.get("created_at"),
                "llm_metadata": question_data.get("llm_metadata", {})
            }
        
        return display_data
    
    def format_question_summary(
        self,
        question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format question data for summary/list view.
        
        Args:
            question_data: Question data
            
        Returns:
            Dict[str, Any]: Summary-formatted question
        """
        return {
            "id": question_data.get("id"),
            "question_number": question_data.get("question_number"),
            "question_type": question_data.get("question_type"),
            "question_preview": self._truncate_text(
                question_data.get("question_text", ""), 100
            ),
            "dimensions_evaluated": question_data.get("dimensions_evaluated", []),
            "time_estimate_seconds": question_data.get("estimated_time_seconds", 45),
            "generated_by": "ai" if not question_data.get("is_static", True) else "static",
            "status": "active"  # Could be derived from other data
        }
    
    def clean_text_input(self, text: str, max_length: Optional[int] = None) -> str:
        """Clean and normalize text input.
        
        Args:
            text: Input text
            max_length: Maximum length (optional)
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove potentially harmful characters
        cleaned = re.sub(r'[<>"\']', '', cleaned)
        
        # Truncate if needed
        if max_length and len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + "..."
        
        return cleaned
    
    def validate_question_text(self, text: str, question_type: QuestionType) -> bool:
        """Validate question text format.
        
        Args:
            text: Question text
            question_type: Type of question
            
        Returns:
            bool: True if valid
        """
        if not text or len(text.strip()) < 10:
            return False
        
        # Type-specific validation
        max_lengths = {
            QuestionType.MCQ: 300,
            QuestionType.STATEMENT_SET: 200,
            QuestionType.SCENARIO_MCQ: 500,
            QuestionType.SCENARIO_MULTI_SELECT: 500,
            QuestionType.THIS_OR_THAT: 250,
            QuestionType.SCALE_RATING: 200,
            QuestionType.PLOT_DAY: 300
        }
        
        max_length = max_lengths.get(question_type, 300)
        return len(text) <= max_length
    
    def _clean_response_structure(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean basic response structure.
        
        Args:
            response_data: Raw response data
            
        Returns:
            Dict[str, Any]: Cleaned response data
        """
        cleaned = {}
        
        # Clean question text
        if "question_text" in response_data:
            cleaned["question_text"] = self.clean_text_input(
                response_data["question_text"], 500
            )
        
        # Clean instructions
        if "instructions" in response_data:
            cleaned["instructions"] = self.clean_text_input(
                response_data["instructions"], 200
            )
        
        # Preserve dimensions_evaluated
        if "dimensions_evaluated" in response_data:
            cleaned["dimensions_evaluated"] = self._clean_dimensions_list(
                response_data["dimensions_evaluated"]
            )
        
        # Preserve metadata
        if "generation_metadata" in response_data:
            cleaned["generation_metadata"] = response_data["generation_metadata"]
        
        return cleaned
    
    def _format_mcq_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format MCQ specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted MCQ data
        """
        if "options" not in response_data:
            return response_data
        
        formatted_options = []
        
        for i, option in enumerate(response_data["options"]):
            if isinstance(option, dict):
                formatted_option = {
                    "id": option.get("id", chr(65 + i)),  # A, B, C, D
                    "text": self.clean_text_input(option.get("text", ""), 200)
                }
                
                # Handle scoring guide
                if "scoring_guide" in option:
                    formatted_option["scoring_guide"] = option["scoring_guide"]
                elif "dimension" in option:
                    formatted_option["scoring_guide"] = {
                        formatted_option["id"]: {
                            "dimension": option["dimension"],
                            "score": option.get("score", 3)
                        }
                    }
                
                formatted_options.append(formatted_option)
        
        response_data["options"] = formatted_options
        return response_data
    
    def _format_statement_set_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format statement set specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted statement set data
        """
        if "statements" not in response_data:
            return response_data
        
        formatted_statements = []
        
        for i, statement in enumerate(response_data["statements"]):
            if isinstance(statement, dict):
                formatted_statement = {
                    "id": statement.get("id", i + 1),
                    "text": self.clean_text_input(statement.get("text", ""), 250),
                    "dimension": statement.get("dimension", "R"),
                    "reverse_scored": statement.get("reverse_scored", False)
                }
                formatted_statements.append(formatted_statement)
        
        response_data["statements"] = formatted_statements
        return response_data
    
    def _format_scenario_mcq_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format scenario MCQ specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted scenario MCQ data
        """
        # Clean scenario description
        if "scenario" in response_data:
            response_data["scenario"] = self.clean_text_input(
                response_data["scenario"], 400
            )
        
        # Format options similar to MCQ
        return self._format_mcq_response(response_data)
    
    def _format_scenario_multi_select_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format scenario multi-select specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted scenario multi-select data
        """
        # Similar to scenario MCQ but allows multiple selections
        return self._format_scenario_mcq_response(response_data)
    
    def _format_this_or_that_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format this-or-that specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted this-or-that data
        """
        # Format option A
        if "option_a" in response_data:
            option_a = response_data["option_a"]
            if isinstance(option_a, dict):
                response_data["option_a"] = {
                    "text": self.clean_text_input(option_a.get("text", ""), 150),
                    "scoring_guide": option_a.get("scoring_guide", {})
                }
        
        # Format option B
        if "option_b" in response_data:
            option_b = response_data["option_b"]
            if isinstance(option_b, dict):
                response_data["option_b"] = {
                    "text": self.clean_text_input(option_b.get("text", ""), 150),
                    "scoring_guide": option_b.get("scoring_guide", {})
                }
        
        return response_data
    
    def _format_scale_rating_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format scale rating specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted scale rating data
        """
        # Ensure scale properties are present and valid
        response_data["scale_min"] = response_data.get("scale_min", 1)
        response_data["scale_max"] = response_data.get("scale_max", 10)
        
        # Clean scale labels
        if "scale_labels" in response_data:
            labels = response_data["scale_labels"]
            if isinstance(labels, dict):
                cleaned_labels = {}
                for key, value in labels.items():
                    cleaned_labels[str(key)] = self.clean_text_input(str(value), 50)
                response_data["scale_labels"] = cleaned_labels
        
        # Clean scale item
        if "scale_item" in response_data:
            response_data["scale_item"] = self.clean_text_input(
                response_data["scale_item"], 200
            )
        
        return response_data
    
    def _format_plot_day_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format plot day specific response data.
        
        Args:
            response_data: Response data
            
        Returns:
            Dict[str, Any]: Formatted plot day data
        """
        if "tasks" not in response_data:
            return response_data
        
        formatted_tasks = []
        
        for i, task in enumerate(response_data["tasks"]):
            if isinstance(task, dict):
                formatted_task = {
                    "id": task.get("id", f"task_{i + 1}"),
                    "title": self.clean_text_input(task.get("title", ""), 100),
                    "description": self.clean_text_input(task.get("description", ""), 200),
                    "category": task.get("category", "general"),
                    "duration": task.get("duration", "1-2 hours"),
                    "primary_dimension": task.get("primary_dimension", "R"),
                    "secondary_dimensions": task.get("secondary_dimensions", [])
                }
                formatted_tasks.append(formatted_task)
        
        response_data["tasks"] = formatted_tasks
        return response_data
    
    def _validate_formatted_response(
        self,
        response_data: Dict[str, Any],
        question_type: QuestionType
    ) -> None:
        """Validate formatted response data.
        
        Args:
            response_data: Formatted response data
            question_type: Question type
            
        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        if not response_data.get("question_text"):
            raise ValueError("Question text is required")
        
        if not self.validate_question_text(response_data["question_text"], question_type):
            raise ValueError("Invalid question text format")
        
        # Type-specific validation
        if question_type == QuestionType.MCQ:
            if not response_data.get("options") or len(response_data["options"]) < 2:
                raise ValueError("MCQ must have at least 2 options")
        
        elif question_type == QuestionType.STATEMENT_SET:
            if not response_data.get("statements") or len(response_data["statements"]) < 3:
                raise ValueError("Statement set must have at least 3 statements")
        
        elif question_type == QuestionType.PLOT_DAY:
            if not response_data.get("tasks") or len(response_data["tasks"]) < 5:
                raise ValueError("Plot day must have at least 5 tasks")
    
    def _clean_dimensions_list(self, dimensions: List[str]) -> List[str]:
        """Clean dimensions list.
        
        Args:
            dimensions: List of dimension strings
            
        Returns:
            List[str]: Cleaned dimensions
        """
        valid_dimensions = {d.value for d in RaisecDimension}
        cleaned = []
        
        for dim in dimensions:
            if isinstance(dim, str) and dim.upper() in valid_dimensions:
                cleaned.append(dim.upper())
        
        return cleaned
    
    def _format_text_for_age_group(self, text: str, age_group: AgeGroup) -> str:
        """Format text appropriately for age group.
        
        Args:
            text: Original text
            age_group: Target age group
            
        Returns:
            str: Age-appropriate text
        """
        if age_group == AgeGroup.TEEN:
            # Replace complex terms with simpler ones
            replacements = {
                "career": "future interests",
                "professional": "work-related",
                "colleagues": "classmates or teammates",
                "workplace": "work environment"
            }
            
            for old, new in replacements.items():
                text = re.sub(rf'\b{old}\b', new, text, flags=re.IGNORECASE)
        
        return text
    
    def _format_instructions(
        self,
        instructions: Optional[str],
        age_group: AgeGroup
    ) -> str:
        """Format instructions for age group.
        
        Args:
            instructions: Original instructions
            age_group: Target age group
            
        Returns:
            str: Age-appropriate instructions
        """
        if not instructions:
            default_instructions = {
                AgeGroup.TEEN: "Choose the option that best describes what you enjoy or find interesting.",
                AgeGroup.YOUNG_ADULT: "Select the option that best matches your preferences and interests.",
                AgeGroup.ADULT: "Choose the option that best aligns with your professional preferences and career interests."
            }
            return default_instructions.get(age_group, "Select your preferred option.")
        
        return self._format_text_for_age_group(instructions, age_group)
    
    def _format_mcq_options_for_display(
        self,
        options: List[Dict[str, Any]],
        age_group: AgeGroup
    ) -> List[Dict[str, Any]]:
        """Format MCQ options for display.
        
        Args:
            options: Option data
            age_group: Target age group
            
        Returns:
            List[Dict[str, Any]]: Display-formatted options
        """
        formatted = []
        
        for option in options:
            formatted_option = {
                "id": option.get("id"),
                "text": self._format_text_for_age_group(
                    option.get("text", ""), age_group
                ),
                "order": len(formatted) + 1
            }
            formatted.append(formatted_option)
        
        return formatted
    
    def _format_statements_for_display(
        self,
        statements: List[Dict[str, Any]],
        age_group: AgeGroup
    ) -> List[Dict[str, Any]]:
        """Format statements for display.
        
        Args:
            statements: Statement data
            age_group: Target age group
            
        Returns:
            List[Dict[str, Any]]: Display-formatted statements
        """
        formatted = []
        
        for statement in statements:
            formatted_statement = {
                "id": statement.get("id"),
                "text": self._format_text_for_age_group(
                    statement.get("text", ""), age_group
                ),
                "order": len(formatted) + 1
            }
            formatted.append(formatted_statement)
        
        return formatted
    
    def _format_plot_day_tasks_for_display(
        self,
        tasks: List[Dict[str, Any]],
        age_group: AgeGroup
    ) -> List[Dict[str, Any]]:
        """Format plot day tasks for display.
        
        Args:
            tasks: Task data
            age_group: Target age group
            
        Returns:
            List[Dict[str, Any]]: Display-formatted tasks
        """
        formatted = []
        
        for task in tasks:
            formatted_task = {
                "id": task.get("id"),
                "title": self._format_text_for_age_group(
                    task.get("title", ""), age_group
                ),
                "description": self._format_text_for_age_group(
                    task.get("description", ""), age_group
                ),
                "category": task.get("category", "general"),
                "typical_duration": task.get("duration", "1-2 hours"),
                "order": len(formatted) + 1
            }
            formatted.append(formatted_task)
        
        return formatted
    
    def _get_likert_scale_info(self, age_group: AgeGroup) -> Dict[str, Any]:
        """Get Likert scale information for age group.
        
        Args:
            age_group: Target age group
            
        Returns:
            Dict[str, Any]: Scale information
        """
        base_scale = {
            "min": 1,
            "max": 5,
            "labels": {
                "1": "Strongly Disagree",
                "2": "Disagree", 
                "3": "Neutral",
                "4": "Agree",
                "5": "Strongly Agree"
            }
        }
        
        if age_group == AgeGroup.TEEN:
            base_scale["labels"].update({
                "1": "Really Don't Like",
                "2": "Don't Like",
                "3": "It's OK",
                "4": "Like",
                "5": "Really Like"
            })
        
        return base_scale
    
    def _get_time_slots_for_age_group(self, age_group: AgeGroup) -> List[Dict[str, str]]:
        """Get time slots appropriate for age group.
        
        Args:
            age_group: Target age group
            
        Returns:
            List[Dict[str, str]]: Time slots
        """
        if age_group == AgeGroup.TEEN:
            return [
                {"id": "morning", "label": "Morning (9 AM - 12 PM)", "time_range": "09:00-12:00"},
                {"id": "afternoon", "label": "Afternoon (12 PM - 3 PM)", "time_range": "12:00-15:00"},
                {"id": "late_afternoon", "label": "Late Afternoon (3 PM - 6 PM)", "time_range": "15:00-18:00"},
                {"id": "evening", "label": "Evening (6 PM - 9 PM)", "time_range": "18:00-21:00"}
            ]
        else:
            return [
                {"id": "morning", "label": "Morning (9:00 - 12:00)", "time_range": "09:00-12:00"},
                {"id": "afternoon", "label": "Afternoon (12:00 - 15:00)", "time_range": "12:00-15:00"},
                {"id": "late_afternoon", "label": "Late Afternoon (15:00 - 18:00)", "time_range": "15:00-18:00"},
                {"id": "evening", "label": "Evening (18:00 - 21:00)", "time_range": "18:00-21:00"}
            ]
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Input text
            max_length: Maximum length
            
        Returns:
            str: Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can break reasonably close
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."


# Convenience functions

def format_question_for_api(
    question_data: Dict[str, Any],
    age_group: AgeGroup
) -> Dict[str, Any]:
    """Format question for API response.
    
    Args:
        question_data: Question data
        age_group: Target age group
        
    Returns:
        Dict[str, Any]: API-formatted question
    """
    formatter = QuestionFormatter()
    return formatter.format_question_for_display(question_data, age_group, include_metadata=True)


def clean_user_input(text: str, input_type: str = "general") -> str:
    """Clean user input text.
    
    Args:
        text: Input text
        input_type: Type of input (general, email, etc.)
        
    Returns:
        str: Cleaned text
    """
    formatter = QuestionFormatter()
    
    if input_type == "email":
        # Basic email cleaning
        text = text.lower().strip()
        if re.match(formatter.validation_patterns["email"], text):
            return text
        else:
            raise ValueError("Invalid email format")
    
    return formatter.clean_text_input(text)


def validate_question_format(
    question_data: Dict[str, Any],
    question_type: QuestionType
) -> Tuple[bool, List[str]]:
    """Validate question format.
    
    Args:
        question_data: Question data
        question_type: Question type
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, errors)
    """
    formatter = QuestionFormatter()
    errors = []
    
    try:
        formatter._validate_formatted_response(question_data, question_type)
        return True, []
    except ValueError as e:
        return False, [str(e)]


# Export main classes and functions
__all__ = [
    "QuestionFormatter",
    "format_question_for_api",
    "clean_user_input",
    "validate_question_format"
]