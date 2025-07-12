"""Input validation utilities for TruScholar application.

This module provides comprehensive validation functions for user inputs,
business logic validation, and data integrity checks.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from src.utils.constants import (
    AgeGroup,
    QuestionType,
    RaisecDimension,
    ValidationConstants,
    get_age_group_from_age,
)


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    cleaned_value: Optional[Any] = Field(default=None, description="Cleaned/normalized value")

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    @classmethod
    def success(cls, cleaned_value: Optional[Any] = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, cleaned_value=cleaned_value)

    @classmethod
    def failure(cls, errors: Union[str, List[str]]) -> "ValidationResult":
        """Create a failed validation result."""
        if isinstance(errors, str):
            errors = [errors]
        return cls(is_valid=False, errors=errors)


def validate_phone(phone: str) -> ValidationResult:
    """Validate Indian phone number.

    Args:
        phone: Phone number to validate

    Returns:
        ValidationResult: Validation result with cleaned phone number
    """
    if not phone:
        return ValidationResult.failure("Phone number is required")

    # Remove any non-digit characters
    cleaned_phone = re.sub(r'\D', '', phone)

    # Check length
    if len(cleaned_phone) != ValidationConstants.PHONE_LENGTH:
        return ValidationResult.failure(
            f"Phone number must be exactly {ValidationConstants.PHONE_LENGTH} digits"
        )

    # Check Indian mobile number pattern (starts with 6, 7, 8, or 9)
    if not ValidationConstants.PHONE_PATTERN.match(cleaned_phone):
        return ValidationResult.failure(
            "Phone number must be a valid Indian mobile number (starting with 6, 7, 8, or 9)"
        )

    return ValidationResult.success(cleaned_phone)


def validate_name(name: str) -> ValidationResult:
    """Validate user name.

    Args:
        name: Name to validate

    Returns:
        ValidationResult: Validation result with cleaned name
    """
    if not name:
        return ValidationResult.failure("Name is required")

    # Remove extra whitespace and normalize
    cleaned_name = " ".join(name.strip().split())

    # Check length
    if len(cleaned_name) < ValidationConstants.NAME_MIN_LENGTH:
        return ValidationResult.failure(
            f"Name must be at least {ValidationConstants.NAME_MIN_LENGTH} characters long"
        )

    if len(cleaned_name) > ValidationConstants.NAME_MAX_LENGTH:
        return ValidationResult.failure(
            f"Name cannot exceed {ValidationConstants.NAME_MAX_LENGTH} characters"
        )

    # Check pattern (letters, spaces, dots, hyphens, apostrophes)
    if not ValidationConstants.NAME_PATTERN.match(cleaned_name):
        return ValidationResult.failure(
            "Name can only contain letters, spaces, dots, hyphens, and apostrophes"
        )

    # Additional checks
    result = ValidationResult.success(cleaned_name)

    # Check for potentially suspicious patterns
    if len(cleaned_name.split()) > 5:
        result.add_warning("Name has many parts - please verify accuracy")

    if any(char.isdigit() for char in cleaned_name):
        result.add_error("Name cannot contain numbers")

    return result


def validate_age(age: int) -> ValidationResult:
    """Validate user age.

    Args:
        age: Age to validate

    Returns:
        ValidationResult: Validation result with age group
    """
    if age is None:
        return ValidationResult.failure("Age is required")

    if not isinstance(age, int) or age <= 0:
        return ValidationResult.failure("Age must be a positive integer")

    if age < ValidationConstants.MIN_AGE:
        return ValidationResult.failure(
            f"Minimum age is {ValidationConstants.MIN_AGE} years"
        )

    if age > ValidationConstants.MAX_AGE:
        return ValidationResult.failure(
            f"Maximum age is {ValidationConstants.MAX_AGE} years"
        )

    try:
        age_group = get_age_group_from_age(age)
        return ValidationResult.success({
            "age": age,
            "age_group": age_group
        })
    except ValueError as e:
        return ValidationResult.failure(str(e))


def validate_email(email: str) -> ValidationResult:
    """Validate email address.

    Args:
        email: Email to validate

    Returns:
        ValidationResult: Validation result with normalized email
    """
    if not email:
        return ValidationResult.failure("Email is required")

    # Normalize email (lowercase, strip whitespace)
    cleaned_email = email.strip().lower()

    # Basic email pattern
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    if not email_pattern.match(cleaned_email):
        return ValidationResult.failure("Invalid email format")

    # Additional checks
    if len(cleaned_email) > 254:  # RFC 5321 limit
        return ValidationResult.failure("Email address is too long")

    local_part, domain = cleaned_email.split('@', 1)

    if len(local_part) > 64:  # RFC 5321 limit
        return ValidationResult.failure("Email local part is too long")

    # Check for common suspicious patterns
    result = ValidationResult.success(cleaned_email)

    if '..' in cleaned_email:
        result.add_error("Email cannot contain consecutive dots")

    if cleaned_email.startswith('.') or cleaned_email.endswith('.'):
        result.add_error("Email cannot start or end with a dot")

    return result


def validate_raisec_scores(scores: Dict[str, float]) -> ValidationResult:
    """Validate RAISEC dimension scores.

    Args:
        scores: Dictionary of dimension scores

    Returns:
        ValidationResult: Validation result with normalized scores
    """
    if not scores:
        return ValidationResult.failure("RAISEC scores are required")

    if not isinstance(scores, dict):
        return ValidationResult.failure("RAISEC scores must be a dictionary")

    # Validate all dimensions are present
    required_dimensions = [dim.value for dim in RaisecDimension]
    provided_dimensions = set(scores.keys())
    missing_dimensions = set(required_dimensions) - provided_dimensions

    if missing_dimensions:
        return ValidationResult.failure(
            f"Missing RAISEC dimensions: {', '.join(missing_dimensions)}"
        )

    # Validate score values
    cleaned_scores = {}
    errors = []

    for dim_code, score in scores.items():
        if dim_code not in required_dimensions:
            errors.append(f"Invalid RAISEC dimension: {dim_code}")
            continue

        if not isinstance(score, (int, float)):
            errors.append(f"Score for {dim_code} must be a number")
            continue

        if score < 0 or score > 100:
            errors.append(f"Score for {dim_code} must be between 0 and 100")
            continue

        cleaned_scores[dim_code] = float(score)

    if errors:
        return ValidationResult.failure(errors)

    # Additional validations
    result = ValidationResult.success(cleaned_scores)

    # Check for extreme patterns
    max_score = max(cleaned_scores.values())
    min_score = min(cleaned_scores.values())

    if max_score - min_score < 10:
        result.add_warning("All scores are very similar - results may not be differentiated")

    if max_score < 30:
        result.add_warning("All scores are quite low - consider retaking the test")

    return result


def validate_test_answers(answers: List[Dict[str, Any]]) -> ValidationResult:
    """Validate test answers completeness and format.

    Args:
        answers: List of answer dictionaries

    Returns:
        ValidationResult: Validation result
    """
    if not answers:
        return ValidationResult.failure("No answers provided")

    if len(answers) < 10:  # Minimum questions for valid test
        return ValidationResult.failure("Insufficient answers for test completion")

    errors = []
    warnings = []

    for i, answer in enumerate(answers):
        if not isinstance(answer, dict):
            errors.append(f"Answer {i+1} must be a dictionary")
            continue

        # Check required fields
        required_fields = ["question_id", "question_type", "answer_data"]
        for field in required_fields:
            if field not in answer:
                errors.append(f"Answer {i+1} missing required field: {field}")

        # Validate question type
        question_type = answer.get("question_type")
        if question_type:
            try:
                QuestionType(question_type)
            except ValueError:
                errors.append(f"Answer {i+1} has invalid question type: {question_type}")

        # Check if answer is skipped
        if answer.get("is_skipped", False):
            warnings.append(f"Question {i+1} was skipped")

        # Basic answer data validation
        answer_data = answer.get("answer_data")
        if not answer_data and not answer.get("is_skipped", False):
            errors.append(f"Answer {i+1} has no answer data")

    if errors:
        return ValidationResult.failure(errors)

    result = ValidationResult.success(answers)
    for warning in warnings:
        result.add_warning(warning)

    return result


def validate_user_input(
    data: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None
) -> ValidationResult:
    """Validate general user input data.

    Args:
        data: Input data dictionary
        required_fields: List of required field names
        optional_fields: List of optional field names

    Returns:
        ValidationResult: Validation result with cleaned data
    """
    if not isinstance(data, dict):
        return ValidationResult.failure("Input must be a dictionary")

    errors = []
    warnings = []
    cleaned_data = {}

    # Check required fields
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Required field missing: {field}")
        else:
            cleaned_data[field] = data[field]

    # Process optional fields
    if optional_fields:
        for field in optional_fields:
            if field in data and data[field] is not None:
                cleaned_data[field] = data[field]

    # Check for unexpected fields
    allowed_fields = set(required_fields + (optional_fields or []))
    provided_fields = set(data.keys())
    unexpected_fields = provided_fields - allowed_fields

    if unexpected_fields:
        warnings.append(f"Unexpected fields will be ignored: {', '.join(unexpected_fields)}")

    if errors:
        return ValidationResult.failure(errors)

    result = ValidationResult.success(cleaned_data)
    for warning in warnings:
        result.add_warning(warning)

    return result


def validate_text_length(
    text: str,
    min_length: int = 0,
    max_length: int = ValidationConstants.MAX_TEXT_LONG,
    field_name: str = "text"
) -> ValidationResult:
    """Validate text length constraints.

    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length
        field_name: Field name for error messages

    Returns:
        ValidationResult: Validation result
    """
    if text is None:
        text = ""

    if not isinstance(text, str):
        return ValidationResult.failure(f"{field_name} must be a string")

    # Clean the text
    cleaned_text = text.strip()

    if len(cleaned_text) < min_length:
        return ValidationResult.failure(
            f"{field_name} must be at least {min_length} characters long"
        )

    if len(cleaned_text) > max_length:
        return ValidationResult.failure(
            f"{field_name} cannot exceed {max_length} characters"
        )

    return ValidationResult.success(cleaned_text)


def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    field_name: str = "value"
) -> ValidationResult:
    """Validate numeric value within range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        field_name: Field name for error messages

    Returns:
        ValidationResult: Validation result
    """
    if value is None:
        return ValidationResult.failure(f"{field_name} is required")

    if not isinstance(value, (int, float)):
        return ValidationResult.failure(f"{field_name} must be a number")

    if min_value is not None and value < min_value:
        return ValidationResult.failure(
            f"{field_name} must be at least {min_value}"
        )

    if max_value is not None and value > max_value:
        return ValidationResult.failure(
            f"{field_name} cannot exceed {max_value}"
        )

    return ValidationResult.success(value)


def validate_array_length(
    array: List[Any],
    min_length: int = 0,
    max_length: int = ValidationConstants.MAX_ARRAY_LENGTH,
    field_name: str = "array"
) -> ValidationResult:
    """Validate array length constraints.

    Args:
        array: Array to validate
        min_length: Minimum length
        max_length: Maximum length
        field_name: Field name for error messages

    Returns:
        ValidationResult: Validation result
    """
    if array is None:
        array = []

    if not isinstance(array, list):
        return ValidationResult.failure(f"{field_name} must be an array")

    if len(array) < min_length:
        return ValidationResult.failure(
            f"{field_name} must have at least {min_length} items"
        )

    if len(array) > max_length:
        return ValidationResult.failure(
            f"{field_name} cannot have more than {max_length} items"
        )

    return ValidationResult.success(array)


def validate_file_upload(
    file_data: Dict[str, Any],
    allowed_extensions: Optional[List[str]] = None,
    max_size_mb: int = ValidationConstants.MAX_FILE_SIZE_MB
) -> ValidationResult:
    """Validate file upload data.

    Args:
        file_data: File data dictionary with 'filename', 'size', 'content_type'
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB

    Returns:
        ValidationResult: Validation result
    """
    if not isinstance(file_data, dict):
        return ValidationResult.failure("File data must be a dictionary")

    filename = file_data.get("filename", "")
    file_size = file_data.get("size", 0)
    content_type = file_data.get("content_type", "")

    if not filename:
        return ValidationResult.failure("Filename is required")

    # Validate file extension
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ""

    if allowed_extensions is None:
        allowed_extensions = [ext.lstrip('.') for ext in ValidationConstants.ALLOWED_FILE_EXTENSIONS]

    if file_extension not in allowed_extensions:
        return ValidationResult.failure(
            f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Validate file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        return ValidationResult.failure(
            f"File size cannot exceed {max_size_mb} MB"
        )

    if file_size == 0:
        return ValidationResult.failure("File cannot be empty")

    # Additional security checks
    result = ValidationResult.success(file_data)

    # Check for potentially dangerous filenames
    dangerous_patterns = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js']
    if any(pattern in filename.lower() for pattern in dangerous_patterns):
        result.add_error("File type not allowed for security reasons")

    # Check filename length
    if len(filename) > 255:
        result.add_error("Filename is too long")

    return result


class BulkValidator:
    """Utility class for validating multiple items."""

    @staticmethod
    def validate_users(users_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate bulk user data.

        Args:
            users_data: List of user data dictionaries

        Returns:
            Dict with validation results
        """
        results = {
            "valid_users": [],
            "invalid_users": [],
            "errors": [],
            "warnings": []
        }

        for i, user_data in enumerate(users_data):
            user_errors = []

            # Validate phone
            phone_result = validate_phone(user_data.get("phone", ""))
            if not phone_result.is_valid:
                user_errors.extend([f"Phone: {error}" for error in phone_result.errors])

            # Validate name
            name_result = validate_name(user_data.get("name", ""))
            if not name_result.is_valid:
                user_errors.extend([f"Name: {error}" for error in name_result.errors])

            # Validate age if provided
            age = user_data.get("age")
            if age is not None:
                age_result = validate_age(age)
                if not age_result.is_valid:
                    user_errors.extend([f"Age: {error}" for error in age_result.errors])

            if user_errors:
                results["invalid_users"].append({
                    "index": i,
                    "data": user_data,
                    "errors": user_errors
                })
            else:
                cleaned_user = {
                    "phone": phone_result.cleaned_value,
                    "name": name_result.cleaned_value
                }
                if age is not None:
                    cleaned_user["age"] = age

                results["valid_users"].append(cleaned_user)

        return results


# Export all validation functions
__all__ = [
    "ValidationResult",
    "validate_phone",
    "validate_name",
    "validate_age",
    "validate_email",
    "validate_raisec_scores",
    "validate_test_answers",
    "validate_user_input",
    "validate_text_length",
    "validate_numeric_range",
    "validate_array_length",
    "validate_file_upload",
    "BulkValidator",
]
