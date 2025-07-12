"""Configuration management for TruScholar application.

This module handles all configuration loading, validation, and management
using Pydantic Settings for type safety and environment variable support.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import setup_logging


class Settings(BaseSettings):
    """Application settings with validation and type hints."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application Settings
    APP_NAME: str = Field(default="TruScholar", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    APP_ENV: str = Field(
        default="development",
        description="Application environment",
        pattern="^(development|test|staging|production)$",
    )
    APP_DEBUG: bool = Field(default=True, description="Debug mode")
    APP_HOST: str = Field(default="0.0.0.0", description="Application host")
    APP_PORT: int = Field(default=8000, description="Application port", ge=1, le=65535)

    # API Settings
    API_V1_PREFIX: str = Field(default="/api/v1", description="API v1 prefix")
    API_TIMEOUT: int = Field(default=30, description="API timeout in seconds", ge=1)
    API_RATE_LIMIT: int = Field(
        default=100, description="Rate limit per period", ge=1
    )
    API_RATE_LIMIT_PERIOD: int = Field(
        default=3600, description="Rate limit period in seconds", ge=1
    )
    API_RATE_LIMIT_USER: int = Field(
        default=1000, description="Rate limit for authenticated users", ge=1
    )
    API_RATE_LIMIT_ANONYMOUS: int = Field(
        default=100, description="Rate limit for anonymous users", ge=1
    )

    # Security Settings
    SECRET_KEY: str = Field(
        default="your-secret-key-here-change-in-production",
        description="Application secret key",
        min_length=32,
    )
    JWT_SECRET_KEY: str = Field(
        default="your-jwt-secret-key-here-change-in-production",
        description="JWT secret key",
        min_length=32,
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=1440, description="Access token expiration in minutes", ge=1
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=30, description="Refresh token expiration in days", ge=1
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts for the application",
    )
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    # Database Configuration
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URL",
    )
    MONGODB_DB_NAME: str = Field(
        default="truscholar", description="MongoDB database name"
    )
    MONGODB_MAX_POOL_SIZE: int = Field(
        default=50, description="MongoDB max connection pool size", ge=1
    )
    MONGODB_MIN_POOL_SIZE: int = Field(
        default=10, description="MongoDB min connection pool size", ge=0
    )
    MONGODB_MAX_IDLE_TIME_MS: int = Field(
        default=10000, description="MongoDB max idle time in milliseconds", ge=0
    )
    MONGODB_CONNECT_TIMEOUT_MS: int = Field(
        default=10000, description="MongoDB connection timeout in milliseconds", ge=1000
    )

    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_MAX_CONNECTIONS: int = Field(
        default=100, description="Redis max connections", ge=1
    )
    REDIS_DECODE_RESPONSES: bool = Field(
        default=True, description="Redis decode responses"
    )
    REDIS_HEALTH_CHECK_INTERVAL: int = Field(
        default=30, description="Redis health check interval in seconds", ge=0
    )

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1", description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2", description="Celery result backend"
    )
    CELERY_TASK_SERIALIZER: str = Field(
        default="json", description="Celery task serializer"
    )
    CELERY_RESULT_SERIALIZER: str = Field(
        default="json", description="Celery result serializer"
    )
    CELERY_ACCEPT_CONTENT: List[str] = Field(
        default=["json"], description="Celery accepted content types"
    )
    CELERY_TIMEZONE: str = Field(default="UTC", description="Celery timezone")
    CELERY_ENABLE_UTC: bool = Field(default=True, description="Celery enable UTC")
    CELERY_TASK_TRACK_STARTED: bool = Field(
        default=True, description="Track task start"
    )
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=300, description="Task time limit in seconds", ge=1
    )
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(
        default=240, description="Task soft time limit in seconds", ge=1
    )

    # LLM Configuration
    OPENAI_API_KEY: str = Field(
        default="your-openai-api-key-here", description="OpenAI API key"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview", description="OpenAI model name"
    )
    OPENAI_MAX_TOKENS: int = Field(
        default=2000, description="OpenAI max tokens", ge=1
    )
    OPENAI_TEMPERATURE: float = Field(
        default=0.7, description="OpenAI temperature", ge=0.0, le=2.0
    )
    OPENAI_TIMEOUT: int = Field(
        default=60, description="OpenAI request timeout in seconds", ge=1
    )
    OPENAI_MAX_RETRIES: int = Field(
        default=3, description="OpenAI max retries", ge=0
    )

    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    ANTHROPIC_MODEL: str = Field(
        default="claude-3-opus", description="Anthropic model name"
    )
    ANTHROPIC_MAX_TOKENS: int = Field(
        default=2000, description="Anthropic max tokens", ge=1
    )

    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google API key")
    GOOGLE_MODEL: str = Field(default="gemini-pro", description="Google model name")

    # LLM Fallback Configuration
    LLM_PRIMARY_PROVIDER: str = Field(
        default="openai",
        description="Primary LLM provider",
        pattern="^(openai|anthropic|google|static)$",
    )
    LLM_FALLBACK_PROVIDERS: List[str] = Field(
        default=["anthropic", "google", "static"],
        description="Fallback LLM providers in order",
    )
    LLM_RETRY_DELAY: int = Field(
        default=2, description="LLM retry delay in seconds", ge=0
    )
    LLM_RETRY_MAX_DELAY: int = Field(
        default=60, description="LLM max retry delay in seconds", ge=1
    )

    # Feature Flags
    ENABLE_CACHE: bool = Field(default=True, description="Enable caching")
    ENABLE_RATE_LIMITING: bool = Field(
        default=True, description="Enable rate limiting"
    )
    ENABLE_API_DOCS: bool = Field(default=True, description="Enable API documentation")
    ENABLE_METRICS: bool = Field(default=True, description="Enable metrics collection")
    ENABLE_TRACING: bool = Field(default=False, description="Enable distributed tracing")

    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Log level",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    LOG_FORMAT: str = Field(
        default="json", description="Log format", pattern="^(json|text)$"
    )
    LOG_FILE_PATH: Optional[str] = Field(
        default="logs/app.log", description="Log file path"
    )
    LOG_FILE_MAX_SIZE: int = Field(
        default=10485760, description="Log file max size in bytes", ge=1024
    )
    LOG_FILE_BACKUP_COUNT: int = Field(
        default=5, description="Log file backup count", ge=0
    )

    # Monitoring Configuration
    PROMETHEUS_ENABLED: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    PROMETHEUS_PORT: int = Field(
        default=9090, description="Prometheus metrics port", ge=1, le=65535
    )
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN")
    SENTRY_ENVIRONMENT: str = Field(
        default="development", description="Sentry environment"
    )
    SENTRY_TRACES_SAMPLE_RATE: float = Field(
        default=1.0, description="Sentry traces sample rate", ge=0.0, le=1.0
    )

    # External Services
    SMTP_HOST: str = Field(default="localhost", description="SMTP host")
    SMTP_PORT: int = Field(default=587, description="SMTP port", ge=1, le=65535)
    SMTP_USERNAME: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASSWORD: Optional[str] = Field(default=None, description="SMTP password")
    SMTP_FROM_EMAIL: str = Field(
        default="noreply@truscholar.com", description="SMTP from email"
    )
    SMTP_USE_TLS: bool = Field(default=True, description="SMTP use TLS")

    # Test Configuration
    TEST_MODE: bool = Field(default=False, description="Test mode enabled")
    TEST_DATABASE_URL: str = Field(
        default="mongodb://localhost:27017/truscholar_test",
        description="Test database URL",
    )
    TEST_REDIS_URL: str = Field(
        default="redis://localhost:6379/15", description="Test Redis URL"
    )

    # Business Logic Configuration
    MIN_USER_AGE: int = Field(
        default=13, description="Minimum user age", ge=1, le=100
    )
    MAX_USER_AGE: int = Field(
        default=99, description="Maximum user age", ge=1, le=150
    )
    TEST_QUESTION_COUNT: int = Field(
        default=12, description="Number of test questions", ge=1
    )
    RAISEC_CODE_LENGTH: int = Field(
        default=3, description="RAISEC code length", ge=1, le=6
    )
    CAREER_RECOMMENDATION_COUNT: int = Field(
        default=3, description="Number of career recommendations", ge=1, le=10
    )
    TEST_TIMEOUT_MINUTES: int = Field(
        default=60, description="Test timeout in minutes", ge=1
    )
    REPORT_CACHE_TTL_HOURS: int = Field(
        default=24, description="Report cache TTL in hours", ge=0
    )

    # File Storage Configuration
    UPLOAD_MAX_SIZE_MB: int = Field(
        default=10, description="Max upload size in MB", ge=1
    )
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = Field(
        default=["pdf", "doc", "docx", "txt"],
        description="Allowed upload file extensions",
    )
    STATIC_FILES_PATH: str = Field(
        default="./static", description="Static files path"
    )
    TEMP_FILES_PATH: str = Field(default="./temp", description="Temp files path")

    # Admin Configuration
    ADMIN_EMAIL: str = Field(
        default="admin@truscholar.com", description="Admin email"
    )
    ADMIN_DASHBOARD_ENABLED: bool = Field(
        default=False, description="Admin dashboard enabled"
    )
    ADMIN_API_KEY: Optional[str] = Field(
        default=None, description="Admin API key"
    )

    @field_validator("APP_ENV")
    def validate_app_env(cls, v: str) -> str:
        """Validate application environment."""
        valid_envs = ["development", "test", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"APP_ENV must be one of {valid_envs}")
        return v

    @field_validator("SECRET_KEY", "JWT_SECRET_KEY")
    def validate_secret_keys(cls, v: str, info) -> str:
        """Validate secret keys are changed in production."""
        if info.data.get("APP_ENV") == "production" and "change-in-production" in v:
            raise ValueError(
                f"{info.field_name} must be changed from default in production"
            )
        return v

    @field_validator("MONGODB_URL", "TEST_DATABASE_URL")
    def validate_mongodb_url(cls, v: str) -> str:
        """Validate MongoDB URL format."""
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MongoDB URL must start with mongodb:// or mongodb+srv://")
        return v

    @field_validator("REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "TEST_REDIS_URL")
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with redis://")
        return v

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validate settings after all fields are set."""
        # Ensure test environment uses test databases
        if self.APP_ENV == "test":
            self.MONGODB_URL = self.TEST_DATABASE_URL
            self.REDIS_URL = self.TEST_REDIS_URL
            self.CELERY_BROKER_URL = self.TEST_REDIS_URL
            self.CELERY_RESULT_BACKEND = self.TEST_REDIS_URL

        # Disable certain features in test mode
        if self.TEST_MODE:
            self.ENABLE_RATE_LIMITING = False
            self.ENABLE_METRICS = False
            self.ENABLE_TRACING = False

        # Ensure Celery time limits are consistent
        if self.CELERY_TASK_SOFT_TIME_LIMIT >= self.CELERY_TASK_TIME_LIMIT:
            self.CELERY_TASK_SOFT_TIME_LIMIT = int(self.CELERY_TASK_TIME_LIMIT * 0.8)

        # Set appropriate defaults for production
        if self.APP_ENV == "production":
            self.APP_DEBUG = False
            self.LOG_LEVEL = "INFO" if self.LOG_LEVEL == "DEBUG" else self.LOG_LEVEL

        return self

    def get_database_url(self) -> str:
        """Get the appropriate database URL based on environment."""
        if self.APP_ENV == "test" or self.TEST_MODE:
            return self.TEST_DATABASE_URL
        return self.MONGODB_URL

    def get_redis_url(self) -> str:
        """Get the appropriate Redis URL based on environment."""
        if self.APP_ENV == "test" or self.TEST_MODE:
            return self.TEST_REDIS_URL
        return self.REDIS_URL

    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration as a dictionary."""
        return {
            "broker_url": self.CELERY_BROKER_URL,
            "result_backend": self.CELERY_RESULT_BACKEND,
            "task_serializer": self.CELERY_TASK_SERIALIZER,
            "result_serializer": self.CELERY_RESULT_SERIALIZER,
            "accept_content": self.CELERY_ACCEPT_CONTENT,
            "timezone": self.CELERY_TIMEZONE,
            "enable_utc": self.CELERY_ENABLE_UTC,
            "task_track_started": self.CELERY_TASK_TRACK_STARTED,
            "task_time_limit": self.CELERY_TASK_TIME_LIMIT,
            "task_soft_time_limit": self.CELERY_TASK_SOFT_TIME_LIMIT,
        }

    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration for a specific provider."""
        provider = provider or self.LLM_PRIMARY_PROVIDER

        if provider == "openai":
            return {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
                "max_tokens": self.OPENAI_MAX_TOKENS,
                "temperature": self.OPENAI_TEMPERATURE,
                "timeout": self.OPENAI_TIMEOUT,
                "max_retries": self.OPENAI_MAX_RETRIES,
            }
        elif provider == "anthropic":
            return {
                "api_key": self.ANTHROPIC_API_KEY,
                "model": self.ANTHROPIC_MODEL,
                "max_tokens": self.ANTHROPIC_MAX_TOKENS,
            }
        elif provider == "google":
            return {
                "api_key": self.GOOGLE_API_KEY,
                "model": self.GOOGLE_MODEL,
            }
        else:
            return {}

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.APP_ENV == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.APP_ENV == "development"

    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self.APP_ENV == "test" or self.TEST_MODE


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings instance
    """
    settings = Settings()

    # Setup logging based on settings
    setup_logging(
        level=settings.LOG_LEVEL,
        format_type=settings.LOG_FORMAT,
        log_file=settings.LOG_FILE_PATH,
        max_bytes=settings.LOG_FILE_MAX_SIZE,
        backup_count=settings.LOG_FILE_BACKUP_COUNT,
    )

    return settings


# Create a global settings instance
settings = get_settings()
