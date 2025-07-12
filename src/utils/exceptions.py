"""Custom exception classes for TruScholar application.

This module defines a hierarchy of custom exceptions for different types
of errors that can occur in the application.
"""

from typing import Any, Dict, List, Optional, Union


class TruScholarError(Exception):
    """Base exception class for all TruScholar application errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize TruScholar error.

        Args:
            message: Error message
            error_code: Application-specific error code
            details: Additional error details
            cause: Underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation.

        Returns:
            Dict[str, Any]: Exception data
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """String representation of the exception."""
        parts = [self.message]
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)


class ValidationError(TruScholarError):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize validation error.

        Args:
            message: Error message
            field: Field name that failed validation
            value: Invalid value
            validation_errors: List of specific validation errors
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if validation_errors:
            details["validation_errors"] = validation_errors

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.field = field
        self.value = value
        self.validation_errors = validation_errors or []


class BusinessLogicError(TruScholarError):
    """Exception for business logic violations."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize business logic error.

        Args:
            message: Error message
            operation: Operation that failed
            resource_id: ID of resource involved
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if resource_id:
            details["resource_id"] = resource_id

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.operation = operation
        self.resource_id = resource_id


class AuthenticationError(TruScholarError):
    """Exception for authentication failures."""

    def __init__(
        self,
        message: str = "Authentication failed",
        user_id: Optional[str] = None,
        auth_method: Optional[str] = None,
        **kwargs
    ):
        """Initialize authentication error.

        Args:
            message: Error message
            user_id: User ID if known
            auth_method: Authentication method used
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if user_id:
            details["user_id"] = user_id
        if auth_method:
            details["auth_method"] = auth_method

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.user_id = user_id
        self.auth_method = auth_method


class AuthorizationError(TruScholarError):
    """Exception for authorization/permission failures."""

    def __init__(
        self,
        message: str = "Access denied",
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        """Initialize authorization error.

        Args:
            message: Error message
            user_id: User ID
            resource: Resource being accessed
            action: Action being attempted
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if user_id:
            details["user_id"] = user_id
        if resource:
            details["resource"] = resource
        if action:
            details["action"] = action

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.user_id = user_id
        self.resource = resource
        self.action = action


class ResourceNotFoundError(TruScholarError):
    """Exception for when a requested resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize resource not found error.

        Args:
            message: Error message
            resource_type: Type of resource (user, test, etc.)
            resource_id: ID of the resource
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.resource_type = resource_type
        self.resource_id = resource_id


class DatabaseError(TruScholarError):
    """Exception for database operation failures."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize database error.

        Args:
            message: Error message
            operation: Database operation (find, insert, update, delete)
            collection: Collection name
            query: Query that failed (sensitive data will be masked)
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if collection:
            details["collection"] = collection
        if query:
            # Mask sensitive data in query
            masked_query = _mask_sensitive_query_data(query)
            details["query"] = masked_query

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.operation = operation
        self.collection = collection
        self.query = query


class CacheError(TruScholarError):
    """Exception for cache operation failures."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize cache error.

        Args:
            message: Error message
            operation: Cache operation (get, set, delete)
            cache_key: Cache key involved
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if cache_key:
            details["cache_key"] = cache_key

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.operation = operation
        self.cache_key = cache_key


class ExternalServiceError(TruScholarError):
    """Exception for external service failures."""

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None,
        **kwargs
    ):
        """Initialize external service error.

        Args:
            message: Error message
            service: External service name (openai, anthropic, etc.)
            status_code: HTTP status code if applicable
            response_data: Response data from service
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if service:
            details["service"] = service
        if status_code:
            details["status_code"] = status_code
        if response_data:
            details["response_data"] = response_data

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.service = service
        self.status_code = status_code
        self.response_data = response_data


class LLMError(ExternalServiceError):
    """Exception for LLM service failures."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        **kwargs
    ):
        """Initialize LLM error.

        Args:
            message: Error message
            model: Model name
            provider: LLM provider (openai, anthropic, google)
            tokens_used: Number of tokens used
            cost: Cost incurred
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if model:
            details["model"] = model
        if provider:
            details["provider"] = provider
        if tokens_used:
            details["tokens_used"] = tokens_used
        if cost:
            details["cost"] = cost

        kwargs["service"] = provider
        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.model = model
        self.provider = provider
        self.tokens_used = tokens_used
        self.cost = cost


class RateLimitError(TruScholarError):
    """Exception for rate limit violations."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        reset_time: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            limit: Rate limit value
            reset_time: When limit resets (Unix timestamp)
            retry_after: Seconds to wait before retry
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if reset_time:
            details["reset_time"] = reset_time
        if retry_after:
            details["retry_after"] = retry_after

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.limit = limit
        self.reset_time = reset_time
        self.retry_after = retry_after


# Domain-specific exceptions

class UserError(TruScholarError):
    """Exception for user-related errors."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        user_phone: Optional[str] = None,
        **kwargs
    ):
        """Initialize user error.

        Args:
            message: Error message
            user_id: User ID
            user_phone: User phone number (masked)
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if user_id:
            details["user_id"] = user_id
        if user_phone:
            # Mask phone number for privacy
            details["user_phone"] = _mask_phone_number(user_phone)

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.user_id = user_id
        self.user_phone = user_phone


class TestError(TruScholarError):
    """Exception for test-related errors."""

    def __init__(
        self,
        message: str,
        test_id: Optional[str] = None,
        user_id: Optional[str] = None,
        test_status: Optional[str] = None,
        **kwargs
    ):
        """Initialize test error.

        Args:
            message: Error message
            test_id: Test ID
            user_id: User ID
            test_status: Current test status
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if test_id:
            details["test_id"] = test_id
        if user_id:
            details["user_id"] = user_id
        if test_status:
            details["test_status"] = test_status

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.test_id = test_id
        self.user_id = user_id
        self.test_status = test_status


class QuestionError(TruScholarError):
    """Exception for question-related errors."""

    def __init__(
        self,
        message: str,
        question_id: Optional[str] = None,
        test_id: Optional[str] = None,
        question_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize question error.

        Args:
            message: Error message
            question_id: Question ID
            test_id: Test ID
            question_type: Question type
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if question_id:
            details["question_id"] = question_id
        if test_id:
            details["test_id"] = test_id
        if question_type:
            details["question_type"] = question_type

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.question_id = question_id
        self.test_id = test_id
        self.question_type = question_type


class CareerError(TruScholarError):
    """Exception for career recommendation errors."""

    def __init__(
        self,
        message: str,
        career_id: Optional[str] = None,
        raisec_code: Optional[str] = None,
        **kwargs
    ):
        """Initialize career error.

        Args:
            message: Error message
            career_id: Career ID
            raisec_code: RAISEC code
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if career_id:
            details["career_id"] = career_id
        if raisec_code:
            details["raisec_code"] = raisec_code

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.career_id = career_id
        self.raisec_code = raisec_code


class ReportError(TruScholarError):
    """Exception for report generation errors."""

    def __init__(
        self,
        message: str,
        report_id: Optional[str] = None,
        test_id: Optional[str] = None,
        report_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize report error.

        Args:
            message: Error message
            report_id: Report ID
            test_id: Test ID
            report_type: Report type
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if report_id:
            details["report_id"] = report_id
        if test_id:
            details["test_id"] = test_id
        if report_type:
            details["report_type"] = report_type

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.report_id = report_id
        self.test_id = test_id
        self.report_type = report_type


class ConfigurationError(TruScholarError):
    """Exception for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key
            config_value: Configuration value
            **kwargs: Additional arguments for parent class
        """
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)

        kwargs["details"] = details
        super().__init__(message, **kwargs)

        self.config_key = config_key
        self.config_value = config_value


# Utility functions for error handling

def _mask_phone_number(phone: str) -> str:
    """Mask phone number for privacy.

    Args:
        phone: Phone number to mask

    Returns:
        str: Masked phone number
    """
    if not phone or len(phone) < 4:
        return "****"
    return f"****{phone[-4:]}"


def _mask_sensitive_query_data(query: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive data in database query.

    Args:
        query: Query dictionary

    Returns:
        Dict[str, Any]: Query with sensitive data masked
    """
    sensitive_fields = ["password", "token", "key", "secret", "phone"]
    masked_query = {}

    for key, value in query.items():
        if any(field in key.lower() for field in sensitive_fields):
            masked_query[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked_query[key] = _mask_sensitive_query_data(value)
        else:
            masked_query[key] = value

    return masked_query


def create_error_response(
    error: TruScholarError,
    include_details: bool = True
) -> Dict[str, Any]:
    """Create standardized error response dictionary.

    Args:
        error: TruScholar error instance
        include_details: Whether to include error details

    Returns:
        Dict[str, Any]: Error response dictionary
    """
    response = {
        "success": False,
        "error": {
            "type": error.__class__.__name__,
            "message": error.message,
            "code": error.error_code,
        }
    }

    if include_details and error.details:
        response["error"]["details"] = error.details

    return response


def handle_exception_chain(exception: Exception) -> List[Dict[str, Any]]:
    """Handle exception chain and create detailed error information.

    Args:
        exception: Exception to process

    Returns:
        List[Dict[str, Any]]: List of error information
    """
    errors = []
    current_exception = exception

    while current_exception:
        error_info = {
            "type": current_exception.__class__.__name__,
            "message": str(current_exception),
        }

        if isinstance(current_exception, TruScholarError):
            error_info.update({
                "error_code": current_exception.error_code,
                "details": current_exception.details,
            })

        errors.append(error_info)

        # Get the next exception in the chain
        if isinstance(current_exception, TruScholarError) and current_exception.cause:
            current_exception = current_exception.cause
        else:
            current_exception = getattr(current_exception, "__cause__", None)

    return errors


# Export all exception classes and utilities
__all__ = [
    # Base exception
    "TruScholarError",

    # General exceptions
    "ValidationError",
    "BusinessLogicError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "DatabaseError",
    "CacheError",
    "ExternalServiceError",
    "LLMError",
    "RateLimitError",
    "ConfigurationError",

    # Domain-specific exceptions
    "UserError",
    "TestError",
    "QuestionError",
    "CareerError",
    "ReportError",

    # Utility functions
    "create_error_response",
    "handle_exception_chain",
]
