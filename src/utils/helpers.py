"""Helper utilities for TruScholar application.

This module provides various utility functions used across the application,
including data transformations, calculations, and common operations.
"""

import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4

from src.utils.constants import (
    RaisecDimension,
    QuestionType,
    AgeGroup,
    TestStatus
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# String utilities

def generate_random_string(length: int = 10, include_digits: bool = True) -> str:
    """Generate a random string of specified length.
    
    Args:
        length: Length of the string
        include_digits: Whether to include digits
        
    Returns:
        str: Random string
    """
    chars = string.ascii_letters
    if include_digits:
        chars += string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        str: Unique identifier
    """
    unique_id = str(uuid4()).replace('-', '')
    return f"{prefix}{unique_id}" if prefix else unique_id


def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
    """Sanitize string for safe storage and display.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length-3] + "..."
        
    return text


# Date and time utilities

def get_age_from_birthdate(birthdate: datetime) -> int:
    """Calculate age from birthdate.
    
    Args:
        birthdate: Date of birth
        
    Returns:
        int: Age in years
    """
    today = datetime.utcnow().date()
    birth_date = birthdate.date() if isinstance(birthdate, datetime) else birthdate
    
    age = today.year - birth_date.year
    
    # Adjust if birthday hasn't occurred this year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
        
    return age


def get_age_group(age: int) -> AgeGroup:
    """Get age group from age.
    
    Args:
        age: Age in years
        
    Returns:
        AgeGroup: Corresponding age group
    """
    if 13 <= age <= 17:
        return AgeGroup.AGE_13_17
    elif 18 <= age <= 25:
        return AgeGroup.AGE_18_25
    elif 26 <= age <= 35:
        return AgeGroup.AGE_26_35
    else:
        # Default to middle group if outside range
        logger.warning(f"Age {age} outside supported range, defaulting to 18-25")
        return AgeGroup.AGE_18_25


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration (e.g., "2h 15m", "45m", "30s")
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if minutes:
            return f"{hours}h {minutes}m"
        return f"{hours}h"


# RAISEC utilities

def calculate_dimension_percentages(
    dimension_scores: Dict[RaisecDimension, float]
) -> Dict[RaisecDimension, float]:
    """Calculate percentage distribution of RAISEC dimensions.
    
    Args:
        dimension_scores: Raw dimension scores
        
    Returns:
        Dict[RaisecDimension, float]: Percentage for each dimension
    """
    total_score = sum(dimension_scores.values())
    
    if total_score == 0:
        # Equal distribution if no scores
        return {dim: 100.0 / 6 for dim in RaisecDimension}
        
    percentages = {}
    for dimension, score in dimension_scores.items():
        percentages[dimension] = (score / total_score) * 100
        
    return percentages


def get_top_dimensions(
    dimension_scores: Dict[RaisecDimension, float],
    top_n: int = 3
) -> List[Tuple[RaisecDimension, float]]:
    """Get top N dimensions by score.
    
    Args:
        dimension_scores: Dimension scores
        top_n: Number of top dimensions to return
        
    Returns:
        List[Tuple[RaisecDimension, float]]: Top dimensions with scores
    """
    sorted_dims = sorted(
        dimension_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_dims[:top_n]


def get_dimension_combination_name(dimensions: List[RaisecDimension]) -> str:
    """Get name for a combination of RAISEC dimensions.
    
    Args:
        dimensions: List of dimensions
        
    Returns:
        str: Combination name (e.g., "RIA", "SEC")
    """
    # Get first letter of each dimension
    letters = [dim.value for dim in dimensions]
    return ''.join(letters)


# Question and answer utilities

def calculate_question_weight(question_type: QuestionType) -> float:
    """Get weight multiplier for question type.
    
    Args:
        question_type: Type of question
        
    Returns:
        float: Weight multiplier
    """
    weight_map = {
        QuestionType.MCQ: 1.0,
        QuestionType.STATEMENT_SET: 1.2,
        QuestionType.SCENARIO_MCQ: 1.1,
        QuestionType.SCENARIO_MULTI_SELECT: 1.15,
        QuestionType.THIS_OR_THAT: 0.9,
        QuestionType.SCALE_RATING: 1.0,
        QuestionType.PLOT_DAY: 1.3
    }
    return weight_map.get(question_type, 1.0)


def calculate_test_completion_time(
    questions_count: int,
    avg_time_per_question: int = 60
) -> int:
    """Calculate estimated test completion time.
    
    Args:
        questions_count: Number of questions
        avg_time_per_question: Average time per question in seconds
        
    Returns:
        int: Estimated completion time in seconds
    """
    # Add buffer time for reading instructions and transitions
    buffer_time = 120  # 2 minutes
    return (questions_count * avg_time_per_question) + buffer_time


def validate_dimension_scores(scores: Dict[str, float]) -> Dict[RaisecDimension, float]:
    """Validate and normalize dimension scores.
    
    Args:
        scores: Raw dimension scores (str keys)
        
    Returns:
        Dict[RaisecDimension, float]: Validated scores with enum keys
    """
    validated_scores = {}
    
    for dim_str, score in scores.items():
        try:
            # Convert string to enum
            dimension = RaisecDimension(dim_str)
            
            # Ensure score is within valid range
            normalized_score = max(0.0, min(1.0, float(score)))
            
            validated_scores[dimension] = normalized_score
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid dimension or score: {dim_str}={score}, error: {e}")
            
    return validated_scores


# Security utilities

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password with salt.
    
    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple[str, str]: (hashed_password, salt)
    """
    if not salt:
        salt = generate_random_string(32)
        
    # Combine password and salt
    salted_password = f"{password}{salt}".encode('utf-8')
    
    # Hash using SHA256
    hashed = hashlib.sha256(salted_password).hexdigest()
    
    return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify password against hash.
    
    Args:
        password: Plain text password
        hashed_password: Stored hash
        salt: Salt used for hashing
        
    Returns:
        bool: True if password matches
    """
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hashed_password


# Data transformation utilities

def flatten_dict(
    nested_dict: Dict[str, Any],
    parent_key: str = '',
    separator: str = '.'
) -> Dict[str, Any]:
    """Flatten nested dictionary.
    
    Args:
        nested_dict: Nested dictionary
        parent_key: Parent key prefix
        separator: Key separator
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    items = []
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
            
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# Test status utilities

def can_modify_test(status: TestStatus) -> bool:
    """Check if test can be modified based on status.
    
    Args:
        status: Test status
        
    Returns:
        bool: True if test can be modified
    """
    return status in [TestStatus.DRAFT, TestStatus.IN_PROGRESS]


def can_submit_answers(status: TestStatus) -> bool:
    """Check if answers can be submitted to test.
    
    Args:
        status: Test status
        
    Returns:
        bool: True if answers can be submitted
    """
    return status == TestStatus.IN_PROGRESS


def get_test_status_display(status: TestStatus) -> str:
    """Get user-friendly display text for test status.
    
    Args:
        status: Test status
        
    Returns:
        str: Display text
    """
    display_map = {
        TestStatus.DRAFT: "Not Started",
        TestStatus.IN_PROGRESS: "In Progress",
        TestStatus.COMPLETED: "Completed",
        TestStatus.ARCHIVED: "Archived"
    }
    return display_map.get(status, str(status.value))


# Scoring utilities

def normalize_score(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0
) -> float:
    """Normalize score to 0-1 range.
    
    Args:
        score: Raw score
        min_score: Minimum possible score
        max_score: Maximum possible score
        
    Returns:
        float: Normalized score
    """
    if max_score == min_score:
        return 0.5
        
    normalized = (score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized))


def calculate_weighted_average(
    values: List[Tuple[float, float]]
) -> float:
    """Calculate weighted average.
    
    Args:
        values: List of (value, weight) tuples
        
    Returns:
        float: Weighted average
    """
    if not values:
        return 0.0
        
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return 0.0
        
    weighted_sum = sum(value * weight for value, weight in values)
    return weighted_sum / total_weight


# Cache key utilities

def generate_cache_key(*parts: str) -> str:
    """Generate cache key from parts.
    
    Args:
        *parts: Key components
        
    Returns:
        str: Cache key
    """
    return ':'.join(str(part) for part in parts)


def parse_cache_key(key: str) -> List[str]:
    """Parse cache key into components.
    
    Args:
        key: Cache key
        
    Returns:
        List[str]: Key components
    """
    return key.split(':')


# Export all utilities
__all__ = [
    # String utilities
    "generate_random_string",
    "generate_unique_id",
    "sanitize_string",
    
    # Date and time utilities
    "get_age_from_birthdate",
    "get_age_group",
    "format_duration",
    
    # RAISEC utilities
    "calculate_dimension_percentages",
    "get_top_dimensions",
    "get_dimension_combination_name",
    "validate_dimension_scores",
    
    # Question and answer utilities
    "calculate_question_weight",
    "calculate_test_completion_time",
    
    # Security utilities
    "hash_password",
    "verify_password",
    
    # Data transformation utilities
    "flatten_dict",
    "chunk_list",
    
    # Test status utilities
    "can_modify_test",
    "can_submit_answers",
    "get_test_status_display",
    
    # Scoring utilities
    "normalize_score",
    "calculate_weighted_average",
    
    # Cache key utilities
    "generate_cache_key",
    "parse_cache_key",
]