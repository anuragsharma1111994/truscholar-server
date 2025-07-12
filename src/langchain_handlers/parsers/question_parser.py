"""Question parser for RAISEC question generation responses.

This module provides parsing and validation for LLM-generated
RAISEC assessment questions across different question types.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

from src.utils.constants import QuestionType, RaisecDimension
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuestionParser(BaseOutputParser):
    """Parser for question generation outputs.
    
    Handles parsing and validation of LLM responses for different
    question types in the RAISEC assessment.
    """
    
    def __init__(self, question_type: QuestionType):
        """Initialize the parser for a specific question type.
        
        Args:
            question_type: Type of question being parsed
        """
        self.question_type = question_type
        self.version = "1.0.0"
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM.
        
        Returns:
            str: Format instructions
        """
        return f"""Return a valid JSON object for a {self.question_type.value} question.
        The JSON must follow the exact structure specified in the prompt.
        Do not include any text before or after the JSON."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output text synchronously.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            Dict[str, Any]: Parsed and validated question data
            
        Raises:
            OutputParserException: If parsing fails
        """
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Parse JSON
            parsed_json = json.loads(cleaned_text)
            
            # Validate structure
            validated_data = self._validate_question_structure(parsed_json)
            
            # Add metadata
            validated_data["parser_metadata"] = {
                "parser_version": self.version,
                "question_type": self.question_type.value,
                "parsed_at": datetime.utcnow().isoformat(),
                "validation_passed": True
            }
            
            return validated_data
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in LLM response: {str(e)}"
            logger.error(error_msg)
            raise OutputParserException(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to parse question: {str(e)}"
            logger.error(error_msg)
            raise OutputParserException(error_msg)
    
    async def aparse(self, text: str) -> Dict[str, Any]:
        """Parse the output text asynchronously.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            Dict[str, Any]: Parsed and validated question data
        """
        # For now, just call the sync version
        return self.parse(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean the raw text to extract JSON.
        
        Args:
            text: Raw text from LLM
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove common prefixes/suffixes
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        
        # Find JSON object boundaries
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in response")
        
        return text[json_start:json_end]
    
    def _validate_question_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the question structure based on type.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Dict[str, Any]: Validated and normalized data
            
        Raises:
            ValueError: If validation fails
        """
        if self.question_type == QuestionType.MCQ:
            return self._validate_mcq(data)
        elif self.question_type == QuestionType.STATEMENT_SET:
            return self._validate_statement_set(data)
        elif self.question_type == QuestionType.SCENARIO_MCQ:
            return self._validate_scenario_mcq(data)
        elif self.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            return self._validate_scenario_multi_select(data)
        elif self.question_type == QuestionType.THIS_OR_THAT:
            return self._validate_this_or_that(data)
        elif self.question_type == QuestionType.SCALE_RATING:
            return self._validate_scale_rating(data)
        elif self.question_type == QuestionType.PLOT_DAY:
            return self._validate_plot_day(data)
        else:
            raise ValueError(f"Unsupported question type: {self.question_type}")
    
    def _validate_mcq(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate MCQ structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        required_fields = ["question_text", "options", "dimensions_evaluated"]
        self._check_required_fields(data, required_fields)
        
        # Validate options
        options = data["options"]
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("MCQ must have at least 2 options")
        
        option_ids = set()
        for option in options:
            self._validate_option(option)
            if option["id"] in option_ids:
                raise ValueError(f"Duplicate option ID: {option['id']}")
            option_ids.add(option["id"])
        
        # Validate dimensions
        self._validate_dimensions(data["dimensions_evaluated"])
        
        return data
    
    def _validate_statement_set(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statement set structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        required_fields = ["question_text", "statements", "dimensions_evaluated"]
        self._check_required_fields(data, required_fields)
        
        # Validate statements
        statements = data["statements"]
        if not isinstance(statements, list) or len(statements) < 3:
            raise ValueError("Statement set must have at least 3 statements")
        
        statement_ids = set()
        for statement in statements:
            self._validate_statement(statement)
            if statement["id"] in statement_ids:
                raise ValueError(f"Duplicate statement ID: {statement['id']}")
            statement_ids.add(statement["id"])
        
        # Validate dimensions
        self._validate_dimensions(data["dimensions_evaluated"])
        
        return data
    
    def _validate_scenario_mcq(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario MCQ structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        # Same validation as regular MCQ
        return self._validate_mcq(data)
    
    def _validate_scenario_multi_select(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scenario multi-select structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        required_fields = ["question_text", "options", "dimensions_evaluated"]
        self._check_required_fields(data, required_fields)
        
        # Validate options
        options = data["options"]
        if not isinstance(options, list) or len(options) < 3:
            raise ValueError("Multi-select must have at least 3 options")
        
        # Validate min/max selections
        min_sel = data.get("min_selections", 1)
        max_sel = data.get("max_selections", len(options))
        
        if min_sel < 1 or max_sel > len(options) or min_sel > max_sel:
            raise ValueError("Invalid min/max selection values")
        
        option_ids = set()
        for option in options:
            self._validate_option(option)
            if option["id"] in option_ids:
                raise ValueError(f"Duplicate option ID: {option['id']}")
            option_ids.add(option["id"])
        
        # Validate dimensions
        self._validate_dimensions(data["dimensions_evaluated"])
        
        return data
    
    def _validate_this_or_that(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate this-or-that structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        required_fields = ["question_text", "option_a", "option_b", "dimensions_evaluated"]
        self._check_required_fields(data, required_fields)
        
        # Validate both options
        self._validate_option(data["option_a"])
        self._validate_option(data["option_b"])
        
        # Validate dimensions
        self._validate_dimensions(data["dimensions_evaluated"])
        
        return data
    
    def _validate_scale_rating(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scale rating structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        required_fields = ["question_text", "scale_min", "scale_max", "dimensions_evaluated"]
        self._check_required_fields(data, required_fields)
        
        # Validate scale
        scale_min = data["scale_min"]
        scale_max = data["scale_max"]
        
        if not isinstance(scale_min, int) or not isinstance(scale_max, int):
            raise ValueError("Scale min/max must be integers")
        
        if scale_min >= scale_max or scale_min < 1 or scale_max > 10:
            raise ValueError("Invalid scale range")
        
        # Validate dimensions
        self._validate_dimensions(data["dimensions_evaluated"])
        
        return data
    
    def _validate_plot_day(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plot day structure.
        
        Args:
            data: Question data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        required_fields = ["question_text", "tasks", "time_slots", "dimensions_evaluated"]
        self._check_required_fields(data, required_fields)
        
        # Validate tasks
        tasks = data["tasks"]
        if not isinstance(tasks, list) or len(tasks) < 8:
            raise ValueError("Plot day must have at least 8 tasks")
        
        task_ids = set()
        for task in tasks:
            self._validate_task(task)
            if task["id"] in task_ids:
                raise ValueError(f"Duplicate task ID: {task['id']}")
            task_ids.add(task["id"])
        
        # Validate time slots
        expected_slots = ["9:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00"]
        time_slots = data["time_slots"]
        
        if not isinstance(time_slots, list) or set(time_slots) != set(expected_slots):
            raise ValueError("Invalid time slots")
        
        # Validate dimensions
        self._validate_dimensions(data["dimensions_evaluated"])
        
        return data
    
    def _validate_option(self, option: Dict[str, Any]) -> None:
        """Validate a single option.
        
        Args:
            option: Option data
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["id", "text", "primary_dimension", "dimension_weights"]
        self._check_required_fields(option, required_fields)
        
        # Validate ID format
        if not isinstance(option["id"], str) or len(option["id"]) != 1:
            raise ValueError("Option ID must be a single character")
        
        # Validate text
        if not isinstance(option["text"], str) or len(option["text"].strip()) == 0:
            raise ValueError("Option text cannot be empty")
        
        # Validate primary dimension
        try:
            RaisecDimension(option["primary_dimension"])
        except ValueError:
            raise ValueError(f"Invalid primary dimension: {option['primary_dimension']}")
        
        # Validate dimension weights
        weights = option["dimension_weights"]
        if not isinstance(weights, dict):
            raise ValueError("Dimension weights must be a dictionary")
        
        for dim, weight in weights.items():
            try:
                RaisecDimension(dim)
            except ValueError:
                raise ValueError(f"Invalid dimension in weights: {dim}")
            
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Invalid weight for {dim}: {weight}")
    
    def _validate_statement(self, statement: Dict[str, Any]) -> None:
        """Validate a single statement.
        
        Args:
            statement: Statement data
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["id", "text", "dimension"]
        self._check_required_fields(statement, required_fields)
        
        # Validate ID
        if not isinstance(statement["id"], int) or statement["id"] < 0:
            raise ValueError("Statement ID must be a non-negative integer")
        
        # Validate text
        if not isinstance(statement["text"], str) or len(statement["text"].strip()) == 0:
            raise ValueError("Statement text cannot be empty")
        
        # Validate dimension
        try:
            RaisecDimension(statement["dimension"])
        except ValueError:
            raise ValueError(f"Invalid dimension: {statement['dimension']}")
        
        # Validate reverse_scored (optional)
        if "reverse_scored" in statement:
            if not isinstance(statement["reverse_scored"], bool):
                raise ValueError("reverse_scored must be a boolean")
    
    def _validate_task(self, task: Dict[str, Any]) -> None:
        """Validate a single task.
        
        Args:
            task: Task data
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["id", "title", "primary_dimension"]
        self._check_required_fields(task, required_fields)
        
        # Validate ID
        if not isinstance(task["id"], str) or len(task["id"].strip()) == 0:
            raise ValueError("Task ID cannot be empty")
        
        # Validate title
        if not isinstance(task["title"], str) or len(task["title"].strip()) == 0:
            raise ValueError("Task title cannot be empty")
        
        # Validate primary dimension
        try:
            RaisecDimension(task["primary_dimension"])
        except ValueError:
            raise ValueError(f"Invalid primary dimension: {task['primary_dimension']}")
        
        # Validate secondary dimensions (optional)
        if "secondary_dimensions" in task:
            secondary = task["secondary_dimensions"]
            if not isinstance(secondary, list):
                raise ValueError("Secondary dimensions must be a list")
            
            for dim in secondary:
                try:
                    RaisecDimension(dim)
                except ValueError:
                    raise ValueError(f"Invalid secondary dimension: {dim}")
    
    def _validate_dimensions(self, dimensions: List[str]) -> None:
        """Validate a list of RAISEC dimensions.
        
        Args:
            dimensions: List of dimension codes
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(dimensions, list) or len(dimensions) == 0:
            raise ValueError("Dimensions evaluated cannot be empty")
        
        for dim in dimensions:
            try:
                RaisecDimension(dim)
            except ValueError:
                raise ValueError(f"Invalid dimension: {dim}")
    
    def _check_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """Check if required fields are present.
        
        Args:
            data: Data to check
            required_fields: List of required field names
            
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
    
    def get_version(self) -> str:
        """Get parser version.
        
        Returns:
            str: Parser version
        """
        return self.version
    
    @property
    def _type(self) -> str:
        """Return the parser type."""
        return "question_parser"


# Export the parser
__all__ = ["QuestionParser"]