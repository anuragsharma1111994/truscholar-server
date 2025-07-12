"""Enums and enumeration utilities for TruScholar application.

This module provides enums used throughout the application for
standardizing values and ensuring type safety.
"""

from enum import Enum, EnumMeta
from typing import Any, List, Type, Union, Callable


class ResponseStatus(Enum):
    """Standard response status values."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"
    FAILED = "failed"
    PROCESSING = "processing"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheKeyType(Enum):
    """Cache key type identifiers."""
    USER = "user"
    TEST = "test"
    QUESTION = "question"
    CAREER = "career"
    RECOMMENDATION = "recommendation"
    RAISEC_SCORES = "raisec_scores"
    ANALYTICS = "analytics"
    SESSION = "session"
    RATE_LIMIT = "rate_limit"


class NotificationStatus(Enum):
    """Notification status values."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    READ = "read"
    DISMISSED = "dismissed"


class APIEndpointCategory(Enum):
    """API endpoint categories for monitoring."""
    AUTH = "auth"
    USER = "user"
    TEST = "test"
    QUESTION = "question"
    CAREER = "career"
    RECOMMENDATION = "recommendation"
    ANALYTICS = "analytics"
    ADMIN = "admin"
    HEALTH = "health"


class DataExportFormat(Enum):
    """Data export format options."""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    EXCEL = "excel"
    XML = "xml"


class SystemMetricType(Enum):
    """System metric types for monitoring."""
    RESPONSE_TIME = "response_time"
    REQUEST_COUNT = "request_count"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_CONNECTION_COUNT = "db_connection_count"
    LLM_REQUEST_COUNT = "llm_request_count"


class BackgroundTaskStatus(Enum):
    """Background task status values."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"
    ASCENDING = "ascending"
    DESCENDING = "descending"


# Utility functions for working with enums


def create_enum_validator(enum_class: Type[Enum]) -> Callable[[Any], bool]:
    """Create a validator function for an enum.
    
    Args:
        enum_class: The enum class to validate against
        
    Returns:
        Validator function that returns True if value is valid
    """
    valid_values = {item.value for item in enum_class}
    
    def validator(value: Any) -> bool:
        return value in valid_values
    
    return validator


def get_enum_values(enum_class: Type[Enum], as_string: bool = True) -> List[Union[str, Any]]:
    """Get all values from an enum.
    
    Args:
        enum_class: The enum class to get values from
        as_string: Whether to return values as strings
        
    Returns:
        List of enum values
    """
    if as_string:
        return [item.value for item in enum_class]
    return [item for item in enum_class]


def get_enum_names(enum_class: Type[Enum]) -> List[str]:
    """Get all names from an enum.
    
    Args:
        enum_class: The enum class to get names from
        
    Returns:
        List of enum names
    """
    return [item.name for item in enum_class]


def enum_to_dict(enum_class: Type[Enum]) -> dict:
    """Convert enum to dictionary.
    
    Args:
        enum_class: The enum class to convert
        
    Returns:
        Dictionary with name-value pairs
    """
    return {item.name: item.value for item in enum_class}


def is_valid_enum_value(enum_class: Type[Enum], value: Any) -> bool:
    """Check if a value is valid for the given enum.
    
    Args:
        enum_class: The enum class to check against
        value: Value to validate
        
    Returns:
        True if value is valid for the enum
    """
    try:
        enum_class(value)
        return True
    except ValueError:
        return False


def get_enum_by_value(enum_class: Type[Enum], value: Any) -> Union[Enum, None]:
    """Get enum item by its value.
    
    Args:
        enum_class: The enum class to search in
        value: Value to find
        
    Returns:
        Enum item if found, None otherwise
    """
    try:
        return enum_class(value)
    except ValueError:
        return None


def get_enum_description(enum_item: Enum) -> str:
    """Get a human-readable description for an enum item.
    
    Args:
        enum_item: The enum item
        
    Returns:
        Human-readable description
    """
    # Default implementation - can be extended with custom descriptions
    return enum_item.name.replace('_', ' ').title()


# Export all enums and utility functions
__all__ = [
    # Enum classes
    "ResponseStatus",
    "LogLevel", 
    "CacheKeyType",
    "NotificationStatus",
    "APIEndpointCategory",
    "DataExportFormat",
    "SystemMetricType",
    "BackgroundTaskStatus",
    "SortOrder",
    
    # Utility functions
    "create_enum_validator",
    "get_enum_values",
    "get_enum_names",
    "enum_to_dict",
    "is_valid_enum_value",
    "get_enum_by_value",
    "get_enum_description",
]