"""TruScholar utilities package.

This package provides common utilities, validators, constants, and helper
functions used throughout the TruScholar application.
"""

from src.utils.constants import (
    AgeGroup,
    BusinessConstants,
    ErrorCodes,
    QuestionType,
    RaisecDimension,
    ResponseFormats,
    ScoringConstants,
    TestStatus,
    TimeConstants,
    UserAccountType,
    ValidationConstants,
    get_age_group_from_age,
    get_raisec_code_from_scores,
    validate_phone_number,
    validate_raisec_code,
)
from src.utils.datetime_utils import (
    format_datetime,
    get_current_timestamp,
    get_timezone_aware_datetime,
    parse_datetime,
    utc_now,
)
from src.utils.enums import (
    ResponseStatus,
    LogLevel,
    CacheKeyType,
    create_enum_validator,
    get_enum_values,
)
from src.utils.exceptions import (
    TruScholarError,
    ValidationError,
    BusinessLogicError,
    ExternalServiceError,
    DatabaseError,
    CacheError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    TestError,
    UserError,
)
from src.utils.helpers import (
    generate_random_string,
    generate_uuid,
    mask_sensitive_data,
    sanitize_string,
    truncate_string,
    calculate_percentage,
    round_to_decimal_places,
    deep_merge_dicts,
    flatten_dict,
    chunk_list,
    remove_duplicates,
    convert_to_snake_case,
    convert_to_camel_case,
)
from src.utils.logger import (
    get_logger,
    get_api_logger,
    get_database_logger,
    get_llm_logger,
    setup_logging,
    log_api_request,
    log_api_response,
    log_database_operation,
    log_llm_request,
    PerformanceLogger,
)
from src.utils.validators import (
    validate_age,
    validate_email,
    validate_name,
    validate_phone,
    validate_raisec_scores,
    validate_test_answers,
    validate_user_input,
    ValidationResult,
)

# Version information
__version__ = "1.0.0"
__author__ = "TruScholar Team"

# Export all utility functions and classes
__all__ = [
    # Constants and Enums
    "AgeGroup",
    "BusinessConstants",
    "ErrorCodes",
    "QuestionType",
    "RaisecDimension",
    "ResponseFormats",
    "ResponseStatus",
    "ScoringConstants",
    "TestStatus",
    "TimeConstants",
    "UserAccountType",
    "ValidationConstants",
    "LogLevel",
    "CacheKeyType",

    # DateTime utilities
    "format_datetime",
    "get_current_timestamp",
    "get_timezone_aware_datetime",
    "parse_datetime",
    "utc_now",

    # Enum utilities
    "create_enum_validator",
    "get_enum_values",

    # Exception classes
    "TruScholarError",
    "ValidationError",
    "BusinessLogicError",
    "ExternalServiceError",
    "DatabaseError",
    "CacheError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "TestError",
    "UserError",

    # Helper functions
    "generate_random_string",
    "generate_uuid",
    "mask_sensitive_data",
    "sanitize_string",
    "truncate_string",
    "calculate_percentage",
    "round_to_decimal_places",
    "deep_merge_dicts",
    "flatten_dict",
    "chunk_list",
    "remove_duplicates",
    "convert_to_snake_case",
    "convert_to_camel_case",

    # Logger functions
    "get_logger",
    "get_api_logger",
    "get_database_logger",
    "get_llm_logger",
    "setup_logging",
    "log_api_request",
    "log_api_response",
    "log_database_operation",
    "log_llm_request",
    "PerformanceLogger",

    # Validator functions
    "validate_age",
    "validate_email",
    "validate_name",
    "validate_phone",
    "validate_raisec_scores",
    "validate_test_answers",
    "validate_user_input",
    "ValidationResult",

    # Utility functions from constants
    "get_age_group_from_age",
    "get_raisec_code_from_scores",
    "validate_phone_number",
    "validate_raisec_code",
]

# Package metadata
__package_info__ = {
    "name": "truscholar-utils",
    "version": __version__,
    "description": "Utility functions and classes for TruScholar application",
    "author": __author__,
    "python_requires": ">=3.9",
}
