"""Validation module for LangChain handlers and prompts.

This module provides comprehensive validation for prompt templates,
LLM responses, and configuration files to ensure data integrity
and consistency across the TruScholar platform.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_core.prompts import ChatPromptTemplate

from src.utils.constants import QuestionType, AgeGroup, RecommendationType, ReportType, RaisecDimension
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptValidationError(Exception):
    """Custom exception for prompt validation errors."""
    pass


class PromptVersionInfo(BaseModel):
    """Model for prompt version information."""
    
    version: str = Field(..., pattern=r"^\d+\.\d+(\.\d+)?$")
    created_at: str = Field(..., description="ISO 8601 timestamp")
    description: str = Field(..., min_length=10)
    author: str = Field(..., min_length=2)
    
    @field_validator('created_at')
    @classmethod
    def validate_created_at(cls, v: str) -> str:
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Invalid ISO 8601 timestamp format")


class QuestionPromptSchema(BaseModel):
    """Schema for question prompt validation."""
    
    system_prompt: str = Field(..., min_length=100)
    age_groups: Dict[str, Dict[str, str]] = Field(...)
    
    @field_validator('age_groups')
    @classmethod
    def validate_age_groups(cls, v: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """Validate age group structure."""
        expected_groups = {"13-17", "18-25", "26-35"}
        if set(v.keys()) != expected_groups:
            raise ValueError(f"Expected age groups: {expected_groups}, got: {set(v.keys())}")
        
        for age_group, content in v.items():
            if "user_template" not in content:
                raise ValueError(f"Missing user_template for age group {age_group}")
            if len(content["user_template"]) < 100:
                raise ValueError(f"user_template too short for age group {age_group}")
        
        return v


class CareerPromptSchema(BaseModel):
    """Schema for career prompt validation."""
    
    system_prompt: str = Field(..., min_length=100)
    user_template: str = Field(..., min_length=200)


class ReportPromptSchema(BaseModel):
    """Schema for report prompt validation."""
    
    system_prompt: str = Field(..., min_length=100)
    user_template: str = Field(..., min_length=300)


class PromptFileValidator:
    """Validator for prompt configuration files."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize validator with base path.
        
        Args:
            base_path: Base path for prompt files (defaults to project data/prompts)
        """
        self.base_path = Path(base_path) if base_path else Path("data/prompts")
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def validate_prompt_file(self, file_path: Union[str, Path]) -> Tuple[bool, List[str], List[str]]:
        """Validate a single prompt file.
        
        Args:
            file_path: Path to the prompt file
            
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        try:
            file_path = Path(file_path)
            
            # Check file exists
            if not file_path.exists():
                self.validation_errors.append(f"File does not exist: {file_path}")
                return False, self.validation_errors, self.validation_warnings
            
            # Load and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate basic structure
            self._validate_basic_structure(data, file_path.name)
            
            # Validate specific prompt type
            if "question" in file_path.name:
                self._validate_question_prompts(data)
            elif "career" in file_path.name:
                self._validate_career_prompts(data)
            elif "report" in file_path.name:
                self._validate_report_prompts(data)
            else:
                self.validation_warnings.append(f"Unknown prompt type: {file_path.name}")
            
            return len(self.validation_errors) == 0, self.validation_errors, self.validation_warnings
            
        except json.JSONDecodeError as e:
            self.validation_errors.append(f"Invalid JSON in {file_path}: {e}")
            return False, self.validation_errors, self.validation_warnings
        
        except Exception as e:
            self.validation_errors.append(f"Error validating {file_path}: {e}")
            return False, self.validation_errors, self.validation_warnings
    
    def validate_all_prompts(self, version: str = "current") -> Dict[str, Any]:
        """Validate all prompt files for a version.
        
        Args:
            version: Version to validate (default: "current")
            
        Returns:
            Dict[str, Any]: Validation results
        """
        version_path = self.base_path / version
        
        if not version_path.exists():
            return {
                "valid": False,
                "errors": [f"Version path does not exist: {version_path}"],
                "warnings": [],
                "files_validated": 0,
                "files_passed": 0
            }
        
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "files_validated": 0,
            "files_passed": 0,
            "file_results": {}
        }
        
        # Find all JSON files
        json_files = list(version_path.glob("*.json"))
        
        for file_path in json_files:
            results["files_validated"] += 1
            
            is_valid, errors, warnings = self.validate_prompt_file(file_path)
            
            results["file_results"][file_path.name] = {
                "valid": is_valid,
                "errors": errors,
                "warnings": warnings
            }
            
            if is_valid:
                results["files_passed"] += 1
            else:
                results["valid"] = False
                results["errors"].extend([f"{file_path.name}: {error}" for error in errors])
            
            results["warnings"].extend([f"{file_path.name}: {warning}" for warning in warnings])
        
        return results
    
    def _validate_basic_structure(self, data: Dict[str, Any], filename: str) -> None:
        """Validate basic prompt file structure."""
        required_fields = ["version", "created_at", "description", "author", "prompts"]
        
        for field in required_fields:
            if field not in data:
                self.validation_errors.append(f"Missing required field: {field}")
        
        # Validate version info
        if "version" in data:
            try:
                PromptVersionInfo(
                    version=data["version"],
                    created_at=data.get("created_at", ""),
                    description=data.get("description", ""),
                    author=data.get("author", "")
                )
            except ValidationError as e:
                self.validation_errors.append(f"Invalid version info: {e}")
        
        # Check prompts structure
        if "prompts" not in data or not isinstance(data["prompts"], dict):
            self.validation_errors.append("Missing or invalid 'prompts' section")
        elif len(data["prompts"]) == 0:
            self.validation_warnings.append("No prompts defined in file")
    
    def _validate_question_prompts(self, data: Dict[str, Any]) -> None:
        """Validate question prompt structure."""
        prompts = data.get("prompts", {})
        
        # Expected question types
        expected_types = {qt.value for qt in QuestionType}
        actual_types = set(prompts.keys())
        
        missing_types = expected_types - actual_types
        if missing_types:
            self.validation_warnings.append(f"Missing question types: {missing_types}")
        
        extra_types = actual_types - expected_types
        if extra_types:
            self.validation_warnings.append(f"Unexpected question types: {extra_types}")
        
        # Validate each question type
        for q_type, q_data in prompts.items():
            try:
                QuestionPromptSchema(**q_data)
            except ValidationError as e:
                self.validation_errors.append(f"Invalid {q_type} prompt: {e}")
            
            # Check for required template variables
            self._validate_template_variables(q_data, "question")
    
    def _validate_career_prompts(self, data: Dict[str, Any]) -> None:
        """Validate career prompt structure."""
        prompts = data.get("prompts", {})
        
        # Expected recommendation types
        expected_types = {rt.value for rt in RecommendationType}
        actual_types = set(prompts.keys())
        
        missing_types = expected_types - actual_types
        if missing_types:
            self.validation_warnings.append(f"Missing recommendation types: {missing_types}")
        
        # Validate each recommendation type
        for r_type, r_data in prompts.items():
            try:
                CareerPromptSchema(**r_data)
            except ValidationError as e:
                self.validation_errors.append(f"Invalid {r_type} prompt: {e}")
            
            # Check for required template variables
            self._validate_template_variables(r_data, "career")
    
    def _validate_report_prompts(self, data: Dict[str, Any]) -> None:
        """Validate report prompt structure."""
        prompts = data.get("prompts", {})
        
        # Expected report types
        expected_types = {rt.value for rt in ReportType}
        actual_types = set(prompts.keys())
        
        missing_types = expected_types - actual_types
        if missing_types:
            self.validation_warnings.append(f"Missing report types: {missing_types}")
        
        # Validate each report type
        for r_type, r_data in prompts.items():
            try:
                ReportPromptSchema(**r_data)
            except ValidationError as e:
                self.validation_errors.append(f"Invalid {r_type} prompt: {e}")
            
            # Check for required template variables
            self._validate_template_variables(r_data, "report")
    
    def _validate_template_variables(self, prompt_data: Dict[str, Any], prompt_type: str) -> None:
        """Validate template variables in prompts."""
        if prompt_type == "question":
            required_vars = {
                "question_number", "dimensions_focus", "context", "constraints",
                "age_group", "age_range"
            }
            
            # Check age group templates
            age_groups = prompt_data.get("age_groups", {})
            for age_group, templates in age_groups.items():
                user_template = templates.get("user_template", "")
                self._check_template_variables(user_template, required_vars, f"{age_group} user_template")
        
        elif prompt_type == "career":
            required_vars = {
                "raisec_code", "top_three_dimensions", "all_dimensions", "score_spread",
                "user_age", "age_group", "career_stage", "user_location",
                "education_level", "experience_level", "interests", "constraints"
            }
            
            user_template = prompt_data.get("user_template", "")
            self._check_template_variables(user_template, required_vars, "user_template")
        
        elif prompt_type == "report":
            required_vars = {
                "user_name", "user_age", "age_group", "career_stage", "user_location",
                "education_level", "experience_level", "raisec_code", "dominant_code",
                "total_assessment_score", "primary_dimension", "secondary_dimension",
                "tertiary_dimension", "all_dimensions", "score_analysis",
                "career_recommendations", "completion_metrics", "additional_insights"
            }
            
            user_template = prompt_data.get("user_template", "")
            self._check_template_variables(user_template, required_vars, "user_template")
    
    def _check_template_variables(self, template: str, required_vars: set, template_name: str) -> None:
        """Check if template contains required variables."""
        # Find variables in template (look for {variable_name})
        var_pattern = r'\{([^}]+)\}'
        found_vars = set(re.findall(var_pattern, template))
        
        # Extract base variable names (handle nested access like {primary_dimension[name]})
        base_vars = set()
        for var in found_vars:
            base_var = var.split('[')[0].split('.')[0]
            base_vars.add(base_var)
        
        missing_vars = required_vars - base_vars
        if missing_vars:
            self.validation_warnings.append(
                f"{template_name}: Missing template variables: {missing_vars}"
            )


class PromptTemplateValidator:
    """Validator for LangChain prompt templates."""
    
    def __init__(self):
        """Initialize template validator."""
        self.validation_errors: List[str] = []
    
    def validate_chat_prompt_template(self, template: ChatPromptTemplate) -> Tuple[bool, List[str]]:
        """Validate a ChatPromptTemplate.
        
        Args:
            template: ChatPromptTemplate to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        self.validation_errors.clear()
        
        try:
            # Check if template has messages
            if not hasattr(template, 'messages') or len(template.messages) == 0:
                self.validation_errors.append("Template has no messages")
                return False, self.validation_errors
            
            # Check each message
            for i, message in enumerate(template.messages):
                if not hasattr(message, 'prompt'):
                    self.validation_errors.append(f"Message {i} has no prompt")
                    continue
                
                # Check for valid template syntax
                try:
                    # Try to format with dummy variables
                    dummy_vars = {
                        'question_number': 1,
                        'dimensions_focus': ['R', 'A'],
                        'context': 'test',
                        'constraints': {},
                        'age_group': '18-25',
                        'age_range': '18-25',
                        'raisec_code': 'RIA',
                        'user_age': 25,
                        'user_name': 'Test User'
                    }
                    
                    message.prompt.format(**dummy_vars)
                    
                except KeyError as e:
                    self.validation_errors.append(f"Message {i}: Missing variable {e}")
                except Exception as e:
                    self.validation_errors.append(f"Message {i}: Template error {e}")
            
            return len(self.validation_errors) == 0, self.validation_errors
            
        except Exception as e:
            self.validation_errors.append(f"Error validating template: {e}")
            return False, self.validation_errors


class ResponseValidator:
    """Validator for LLM responses."""
    
    def __init__(self):
        """Initialize response validator."""
        pass
    
    def validate_question_response(self, response: str, question_type: QuestionType) -> Tuple[bool, List[str]]:
        """Validate a question generation response.
        
        Args:
            response: Raw response from LLM
            question_type: Expected question type
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        errors = []
        
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Basic structure validation
            required_fields = ["question_text", "dimensions_evaluated"]
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Type-specific validation
            if question_type == QuestionType.MCQ:
                if "options" not in data or len(data.get("options", [])) < 2:
                    errors.append("MCQ must have at least 2 options")
            
            elif question_type == QuestionType.STATEMENT_SET:
                if "statements" not in data or len(data.get("statements", [])) < 3:
                    errors.append("Statement set must have at least 3 statements")
            
            # Validate RAISEC dimensions
            dimensions = data.get("dimensions_evaluated", [])
            for dim in dimensions:
                try:
                    RaisecDimension(dim)
                except ValueError:
                    errors.append(f"Invalid RAISEC dimension: {dim}")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON response: {e}")
            return False, errors
        
        except Exception as e:
            errors.append(f"Error validating response: {e}")
            return False, errors
    
    def validate_career_response(self, response: str) -> Tuple[bool, List[str]]:
        """Validate a career recommendation response.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        errors = []
        
        try:
            data = json.loads(response)
            
            # Check required sections
            required_sections = ["recommendations", "summary", "key_strengths", "development_areas"]
            for section in required_sections:
                if section not in data:
                    errors.append(f"Missing required section: {section}")
            
            # Validate recommendations
            recommendations = data.get("recommendations", [])
            if len(recommendations) == 0:
                errors.append("No recommendations provided")
            
            for i, rec in enumerate(recommendations):
                required_fields = ["title", "description", "raisec_match", "match_score"]
                for field in required_fields:
                    if field not in rec:
                        errors.append(f"Recommendation {i+1}: Missing field {field}")
                
                # Validate match score
                match_score = rec.get("match_score", 0)
                if not isinstance(match_score, (int, float)) or not (0 <= match_score <= 100):
                    errors.append(f"Recommendation {i+1}: Invalid match_score {match_score}")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON response: {e}")
            return False, errors
        
        except Exception as e:
            errors.append(f"Error validating response: {e}")
            return False, errors
    
    def validate_report_response(self, response: str, report_type: ReportType) -> Tuple[bool, List[str]]:
        """Validate a report generation response.
        
        Args:
            response: Raw response from LLM
            report_type: Expected report type
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        errors = []
        
        try:
            data = json.loads(response)
            
            # Check for report wrapper
            if "report" not in data:
                errors.append("Response must contain a 'report' field")
                return False, errors
            
            report = data["report"]
            
            # Type-specific validation
            if report_type == ReportType.COMPREHENSIVE:
                required_sections = [
                    "title", "executive_summary", "personality_profile",
                    "career_recommendations", "development_plan", "next_steps"
                ]
            elif report_type == ReportType.SUMMARY:
                required_sections = [
                    "title", "key_insights", "raisec_profile", "career_matches", "action_plan"
                ]
            elif report_type == ReportType.DETAILED:
                required_sections = [
                    "title", "personality_analysis", "career_exploration", "development_strategy"
                ]
            else:
                errors.append(f"Unknown report type: {report_type}")
                return False, errors
            
            # Check required sections
            for section in required_sections:
                if section not in report:
                    errors.append(f"Missing required section: {section}")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON response: {e}")
            return False, errors
        
        except Exception as e:
            errors.append(f"Error validating response: {e}")
            return False, errors


def validate_prompts_directory(directory_path: str) -> Dict[str, Any]:
    """Validate an entire prompts directory.
    
    Args:
        directory_path: Path to prompts directory
        
    Returns:
        Dict[str, Any]: Comprehensive validation results
    """
    validator = PromptFileValidator(directory_path)
    
    # Get all versions
    base_path = Path(directory_path)
    versions = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    results = {
        "valid": True,
        "versions_found": versions,
        "version_results": {},
        "overall_errors": [],
        "overall_warnings": []
    }
    
    for version in versions:
        version_results = validator.validate_all_prompts(version)
        results["version_results"][version] = version_results
        
        if not version_results["valid"]:
            results["valid"] = False
            results["overall_errors"].extend([f"v{version}: {error}" for error in version_results["errors"]])
        
        results["overall_warnings"].extend([f"v{version}: {warning}" for warning in version_results["warnings"]])
    
    return results


# Export all classes and functions
__all__ = [
    "PromptValidationError",
    "PromptVersionInfo", 
    "QuestionPromptSchema",
    "CareerPromptSchema",
    "ReportPromptSchema",
    "PromptFileValidator",
    "PromptTemplateValidator",
    "ResponseValidator",
    "validate_prompts_directory"
]