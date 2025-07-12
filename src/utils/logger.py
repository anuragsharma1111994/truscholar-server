"""Production-grade logging configuration for TruScholar.

This module provides structured logging with different handlers for development
and production environments, including JSON formatting for GCP Cloud Logging.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger


class TruScholarFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for TruScholar application logs."""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record.

        Args:
            log_record: The log record dictionary to modify
            record: The original logging record
            message_dict: Additional message data
        """
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

        # Add application context
        log_record['application'] = 'truscholar'
        log_record['service'] = 'career-api'

        # Add thread and process info for debugging
        log_record['thread_id'] = record.thread
        log_record['process_id'] = record.process

        # Handle exception info
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records."""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize context filter.

        Args:
            context: Additional context to add to all log records
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record.

        Args:
            record: The log record to modify

        Returns:
            bool: Always True to allow all records
        """
        # Add context fields
        for key, value in self.context.items():
            setattr(record, key, value)

        return True


class RequestContextFilter(logging.Filter):
    """Filter to add request context from FastAPI."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context if available.

        Args:
            record: The log record to modify

        Returns:
            bool: Always True to allow all records
        """
        try:
            # Try to get request context from contextvars (FastAPI)
            import contextvars

            # These would be set by middleware
            request_id = getattr(contextvars, 'request_id', None)
            user_id = getattr(contextvars, 'user_id', None)
            session_id = getattr(contextvars, 'session_id', None)

            if request_id:
                record.request_id = request_id.get() if hasattr(request_id, 'get') else str(request_id)
            if user_id:
                record.user_id = user_id.get() if hasattr(user_id, 'get') else str(user_id)
            if session_id:
                record.session_id = session_id.get() if hasattr(session_id, 'get') else str(session_id)

        except (ImportError, AttributeError):
            # Context not available, continue without it
            pass

        return True


class LoggerConfig:
    """Logger configuration manager."""

    # Log levels
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    # Component loggers
    COMPONENTS = {
        'api': 'truscholar.api',
        'database': 'truscholar.database',
        'llm': 'truscholar.llm',
        'langchain': 'truscholar.langchain',
        'cache': 'truscholar.cache',
        'worker': 'truscholar.worker',
        'security': 'truscholar.security',
        'business': 'truscholar.business',
    }

    def __init__(self, environment: str = 'development', log_level: str = 'INFO'):
        """Initialize logger configuration.

        Args:
            environment: Environment name (development, production, test)
            log_level: Default log level
        """
        self.environment = environment
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Configure root logger
        self._configure_root_logger()

        # Configure component loggers
        self._configure_component_loggers()

    def _configure_root_logger(self) -> None:
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Clear existing handlers
        root_logger.handlers.clear()

        if self.environment == 'production':
            self._add_production_handlers(root_logger)
        elif self.environment == 'test':
            self._add_test_handlers(root_logger)
        else:
            self._add_development_handlers(root_logger)

    def _add_production_handlers(self, logger: logging.Logger) -> None:
        """Add production-grade handlers.

        Args:
            logger: Logger to configure
        """
        # Console handler with JSON formatting for GCP
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        json_formatter = TruScholarFormatter(
            fmt='%(timestamp)s %(level)s %(logger)s %(message)s'
        )
        console_handler.setFormatter(json_formatter)
        console_handler.addFilter(RequestContextFilter())

        logger.addHandler(console_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        error_handler.addFilter(RequestContextFilter())

        logger.addHandler(error_handler)

        # Application log handler
        app_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "application.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(json_formatter)
        app_handler.addFilter(RequestContextFilter())

        logger.addHandler(app_handler)

    def _add_development_handlers(self, logger: logging.Logger) -> None:
        """Add development-friendly handlers.

        Args:
            logger: Logger to configure
        """
        # Console handler with readable formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        dev_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-3d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(dev_formatter)

        logger.addHandler(console_handler)

        # Debug file handler
        debug_handler = logging.FileHandler(
            filename=self.log_dir / "debug.log",
            mode='a',
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(dev_formatter)

        logger.addHandler(debug_handler)

    def _add_test_handlers(self, logger: logging.Logger) -> None:
        """Add test environment handlers.

        Args:
            logger: Logger to configure
        """
        # Only console handler for tests
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors during tests

        test_formatter = logging.Formatter(
            fmt='TEST | %(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(test_formatter)

        logger.addHandler(console_handler)

    def _configure_component_loggers(self) -> None:
        """Configure individual component loggers."""
        for component, logger_name in self.COMPONENTS.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(self.log_level)

            # Add component context
            context_filter = ContextFilter({'component': component})
            logger.addFilter(context_filter)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name

        Returns:
            logging.Logger: Configured logger instance
        """
        return logging.getLogger(name)

    def get_component_logger(self, component: str) -> logging.Logger:
        """Get a component-specific logger.

        Args:
            component: Component name (api, database, llm, etc.)

        Returns:
            logging.Logger: Component logger

        Raises:
            ValueError: If component is not recognized
        """
        if component not in self.COMPONENTS:
            raise ValueError(f"Unknown component: {component}. Available: {list(self.COMPONENTS.keys())}")

        return logging.getLogger(self.COMPONENTS[component])


# Global logger configuration instance
_logger_config: Optional[LoggerConfig] = None


def setup_logging(environment: str = 'development', log_level: str = 'INFO') -> LoggerConfig:
    """Setup application logging.

    Args:
        environment: Environment name
        log_level: Log level

    Returns:
        LoggerConfig: Configured logger instance
    """
    global _logger_config
    _logger_config = LoggerConfig(environment, log_level)
    return _logger_config


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name, defaults to caller's module name

    Returns:
        logging.Logger: Logger instance
    """
    if _logger_config is None:
        # Auto-setup with defaults
        setup_logging()

    return _logger_config.get_logger(name)


def get_component_logger(component: str) -> logging.Logger:
    """Get a component-specific logger.

    Args:
        component: Component name

    Returns:
        logging.Logger: Component logger
    """
    if _logger_config is None:
        setup_logging()

    return _logger_config.get_component_logger(component)


# Convenience functions for different components
def get_api_logger() -> logging.Logger:
    """Get API component logger."""
    return get_component_logger('api')


def get_database_logger() -> logging.Logger:
    """Get database component logger."""
    return get_component_logger('database')


def get_llm_logger() -> logging.Logger:
    """Get LLM component logger."""
    return get_component_logger('llm')


def get_langchain_logger() -> logging.Logger:
    """Get LangChain component logger."""
    return get_component_logger('langchain')


def get_cache_logger() -> logging.Logger:
    """Get cache component logger."""
    return get_component_logger('cache')


def get_worker_logger() -> logging.Logger:
    """Get worker component logger."""
    return get_component_logger('worker')


def get_security_logger() -> logging.Logger:
    """Get security component logger."""
    return get_component_logger('security')


def get_business_logger() -> logging.Logger:
    """Get business logic logger."""
    return get_component_logger('business')


# Log helper functions
def log_function_call(func_name: str, args: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """Log function call with arguments.

    Args:
        func_name: Function name
        args: Function arguments
        logger: Logger instance, defaults to root logger
    """
    if logger is None:
        logger = get_logger()

    logger.debug(f"Calling {func_name}", extra={
        'function_name': func_name,
        'arguments': args,
        'event_type': 'function_call'
    })


def log_api_request(method: str, path: str, user_id: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
    """Log API request.

    Args:
        method: HTTP method
        path: Request path
        user_id: User ID if authenticated
        logger: Logger instance
    """
    if logger is None:
        logger = get_api_logger()

    logger.info(f"{method} {path}", extra={
        'http_method': method,
        'request_path': path,
        'user_id': user_id,
        'event_type': 'api_request'
    })


def log_api_response(method: str, path: str, status_code: int, duration_ms: float, logger: Optional[logging.Logger] = None) -> None:
    """Log API response.

    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        logger: Logger instance
    """
    if logger is None:
        logger = get_api_logger()

    level = logging.WARNING if status_code >= 400 else logging.INFO

    logger.log(level, f"{method} {path} - {status_code}", extra={
        'http_method': method,
        'request_path': path,
        'status_code': status_code,
        'duration_ms': duration_ms,
        'event_type': 'api_response'
    })


def log_llm_request(model: str, prompt_type: str, tokens: int, cost: float, logger: Optional[logging.Logger] = None) -> None:
    """Log LLM API request.

    Args:
        model: Model name
        prompt_type: Type of prompt
        tokens: Token count
        cost: Cost in USD
        logger: Logger instance
    """
    if logger is None:
        logger = get_llm_logger()

    logger.info(f"LLM request to {model}", extra={
        'model': model,
        'prompt_type': prompt_type,
        'token_count': tokens,
        'cost_usd': cost,
        'event_type': 'llm_request'
    })


def log_database_operation(operation: str, collection: str, duration_ms: float, logger: Optional[logging.Logger] = None) -> None:
    """Log database operation.

    Args:
        operation: Operation type (find, insert, update, delete)
        collection: Collection name
        duration_ms: Operation duration in milliseconds
        logger: Logger instance
    """
    if logger is None:
        logger = get_database_logger()

    logger.info(f"DB {operation} on {collection}", extra={
        'database_operation': operation,
        'collection': collection,
        'duration_ms': duration_ms,
        'event_type': 'database_operation'
    })


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = 'INFO', logger: Optional[logging.Logger] = None) -> None:
    """Log security event.

    Args:
        event_type: Type of security event
        details: Event details
        severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        logger: Logger instance
    """
    if logger is None:
        logger = get_security_logger()

    level = getattr(logging, severity.upper(), logging.INFO)

    logger.log(level, f"Security event: {event_type}", extra={
        'security_event_type': event_type,
        'event_details': details,
        'severity': severity,
        'event_type': 'security_event'
    })


# Performance monitoring
class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, extra: Optional[Dict[str, Any]] = None):
        """Initialize performance logger.

        Args:
            operation: Operation name
            logger: Logger instance
            extra: Additional fields to log
        """
        self.operation = operation
        self.logger = logger or get_logger()
        self.extra = extra or {}
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> 'PerformanceLogger':
        """Start timing."""
        self.start_time = datetime.utcnow()
        self.logger.debug(f"Starting {self.operation}", extra={
            'operation': self.operation,
            'event_type': 'performance_start',
            **self.extra
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing and log result."""
        if self.start_time:
            duration = datetime.utcnow() - self.start_time
            duration_ms = duration.total_seconds() * 1000

            level = logging.WARNING if duration_ms > 5000 else logging.INFO  # Warn if > 5 seconds

            self.logger.log(level, f"Completed {self.operation}", extra={
                'operation': self.operation,
                'duration_ms': duration_ms,
                'event_type': 'performance_end',
                'success': exc_type is None,
                **self.extra
            })


# Example usage functions
def example_usage():
    """Example of how to use the logging system."""
    # Setup logging (typically done in main.py)
    setup_logging(environment='development', log_level='DEBUG')

    # Get loggers
    api_logger = get_api_logger()
    db_logger = get_database_logger()

    # Log API request
    log_api_request('GET', '/api/v1/users', user_id='user123')

    # Log with performance monitoring
    with PerformanceLogger('database_query', db_logger, {'table': 'users'}):
        # Simulate database operation
        import time
        time.sleep(0.1)

    # Log security event
    log_security_event('failed_login', {
        'ip_address': '192.168.1.1',
        'user_agent': 'Mozilla/5.0...',
        'attempted_username': 'admin'
    }, severity='WARNING')


if __name__ == '__main__':
    example_usage()
