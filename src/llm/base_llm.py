"""Base LLM interface and common types for TruScholar.

This module defines the abstract base class and common data structures
for all LLM providers in the TruScholar application.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMModelType(str, Enum):
    """Enumeration of LLM model types."""
    
    # OpenAI models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Google models
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class LLMRole(str, Enum):
    """Message roles in LLM conversations."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""
    
    role: LLMRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "role": "user",
                "content": "Generate a career assessment question for a 20-year-old",
                "metadata": {"age_group": "18-25", "question_type": "mcq"}
            }
        }
    }


class LLMUsage(BaseModel):
    """Token usage information from LLM response."""
    
    prompt_tokens: int = Field(..., ge=0, description="Tokens used in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Tokens used in the completion")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    estimated_cost: Optional[float] = Field(None, ge=0, description="Estimated cost in USD")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt_tokens": 150,
                "completion_tokens": 300,
                "total_tokens": 450,
                "estimated_cost": 0.009
            }
        }
    }


class LLMRequest(BaseModel):
    """Request to an LLM provider."""
    
    messages: List[LLMMessage] = Field(..., min_length=1, description="Conversation messages")
    model: LLMModelType = Field(..., description="Model to use")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0, le=1, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream the response")
    
    # Request metadata
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    user_id: Optional[str] = Field(None, description="User making the request")
    purpose: Optional[str] = Field(None, description="Purpose of the request")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a career counseling expert."},
                    {"role": "user", "content": "Generate an MCQ question about interests."}
                ],
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 500,
                "purpose": "question_generation"
            }
        }
    }


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model that generated the response")
    usage: LLMUsage = Field(..., description="Token usage information")
    finish_reason: Optional[str] = Field(None, description="Reason the generation stopped")
    
    # Response metadata
    response_id: Optional[str] = Field(None, description="Unique response identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    provider: LLMProvider = Field(..., description="LLM provider used")
    latency_ms: Optional[float] = Field(None, ge=0, description="Response latency in milliseconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "What type of work environment do you prefer?\na) Quiet office\nb) Collaborative space\nc) Outdoor setting\nd) Home office",
                "model": "gpt-4",
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 50,
                    "total_tokens": 200,
                    "estimated_cost": 0.004
                },
                "finish_reason": "stop",
                "provider": "openai",
                "latency_ms": 1500.0
            }
        }
    }


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """Initialize LLM error.
        
        Args:
            message: Error message
            provider: LLM provider where error occurred
            model: Model being used when error occurred
            error_code: Provider-specific error code
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.error_code = error_code
        self.original_error = original_error


class LLMRateLimitError(LLMError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMQuotaError(LLMError):
    """Exception raised when usage quotas are exceeded."""
    pass


class LLMAuthenticationError(LLMError):
    """Exception raised when authentication fails."""
    pass


class LLMValidationError(LLMError):
    """Exception raised when request validation fails."""
    pass


class LLMTimeoutError(LLMError):
    """Exception raised when requests timeout."""
    pass


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(
        self,
        api_key: str,
        model: LLMModelType,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs
    ):
        """Initialize base LLM.
        
        Args:
            api_key: API key for the provider
            model: Default model to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            **kwargs: Additional provider-specific arguments
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize provider-specific settings
        self._setup_client(**kwargs)
    
    @abstractmethod
    def _setup_client(self, **kwargs) -> None:
        """Setup the provider-specific client.
        
        Args:
            **kwargs: Provider-specific configuration options
        """
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: LLM request with messages and parameters
            
        Returns:
            LLMResponse: Generated response with usage information
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def validate_request(self, request: LLMRequest) -> None:
        """Validate an LLM request.
        
        Args:
            request: Request to validate
            
        Raises:
            LLMValidationError: If request is invalid
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate the cost of a request.
        
        Args:
            request: Request to estimate cost for
            
        Returns:
            float: Estimated cost in USD
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information including limits and capabilities
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy.
        
        Returns:
            bool: True if provider is healthy
        """
        try:
            # Simple test request
            test_request = LLMRequest(
                messages=[
                    LLMMessage(role=LLMRole.USER, content="Hello")
                ],
                model=self.model,
                max_tokens=5
            )
            
            await self.generate(test_request)
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for {self.__class__.__name__}: {e}")
            return False
    
    def _calculate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        This is a rough estimation. Providers should override with more accurate methods.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4
    
    def _validate_common_params(self, request: LLMRequest) -> None:
        """Validate common request parameters.
        
        Args:
            request: Request to validate
            
        Raises:
            LLMValidationError: If validation fails
        """
        if not request.messages:
            raise LLMValidationError("Messages cannot be empty")
        
        if request.temperature < 0 or request.temperature > 2:
            raise LLMValidationError("Temperature must be between 0 and 2")
        
        if request.max_tokens is not None and request.max_tokens < 1:
            raise LLMValidationError("max_tokens must be positive")
        
        # Check message content
        for i, message in enumerate(request.messages):
            if not message.content.strip():
                raise LLMValidationError(f"Message {i} cannot be empty")


# Export all classes
__all__ = [
    "LLMProvider",
    "LLMModelType", 
    "LLMRole",
    "LLMMessage",
    "LLMUsage",
    "LLMRequest",
    "LLMResponse",
    "LLMError",
    "LLMRateLimitError",
    "LLMQuotaError", 
    "LLMAuthenticationError",
    "LLMValidationError",
    "LLMTimeoutError",
    "BaseLLM"
]