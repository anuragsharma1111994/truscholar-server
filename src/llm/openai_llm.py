"""OpenAI LLM integration for TruScholar.

This module provides integration with OpenAI's GPT models including
GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime

from .base_llm import (
    BaseLLM, LLMRequest, LLMResponse, LLMMessage, LLMUsage,
    LLMProvider, LLMModelType, LLMRole,
    LLMError, LLMRateLimitError, LLMQuotaError, 
    LLMAuthenticationError, LLMValidationError, LLMTimeoutError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    # Model pricing per 1K tokens (input, output)
    MODEL_PRICING = {
        LLMModelType.GPT_4: (0.03, 0.06),
        LLMModelType.GPT_4_TURBO: (0.01, 0.03),
        LLMModelType.GPT_3_5_TURBO: (0.0015, 0.002),
    }
    
    # Model context limits
    MODEL_LIMITS = {
        LLMModelType.GPT_4: 8192,
        LLMModelType.GPT_4_TURBO: 128000,
        LLMModelType.GPT_3_5_TURBO: 4096,
    }
    
    def __init__(
        self,
        api_key: str,
        model: LLMModelType = LLMModelType.GPT_4,
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        **kwargs
    ):
        """Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            base_url: Base URL for OpenAI API
            organization: OpenAI organization ID
            **kwargs: Additional arguments for parent class
        """
        self.base_url = base_url
        self.organization = organization
        
        super().__init__(api_key, model, **kwargs)
    
    def _setup_client(self, **kwargs) -> None:
        """Setup OpenAI HTTP client."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API.
        
        Args:
            request: LLM request
            
        Returns:
            LLMResponse: Generated response
        """
        start_time = time.time()
        
        # Validate request
        await self.validate_request(request)
        
        # Prepare OpenAI API request
        api_request = self._prepare_api_request(request)
        
        # Make request with retries
        response_data = await self._make_request_with_retries(api_request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        return self._parse_response(response_data, request, latency_ms)
    
    def _prepare_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare OpenAI API request payload.
        
        Args:
            request: LLM request
            
        Returns:
            Dict[str, Any]: API request payload
        """
        # Convert messages to OpenAI format
        messages = []
        for msg in request.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Build request payload
        payload = {
            "model": request.model.value,
            "messages": messages,
            "temperature": request.temperature,
        }
        
        # Add optional parameters
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        
        if request.stop:
            payload["stop"] = request.stop
        
        if request.stream:
            payload["stream"] = request.stream
        
        return payload
    
    async def _make_request_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with exponential backoff retries.
        
        Args:
            payload: Request payload
            
        Returns:
            Dict[str, Any]: Response data
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post("/chat/completions", json=payload)
                
                if response.status_code == 200:
                    return response.json()
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key",
                        provider=LLMProvider.OPENAI,
                        model=payload["model"]
                    )
                
                if response.status_code == 429:
                    error_data = response.json().get("error", {})
                    retry_after = response.headers.get("Retry-After")
                    
                    raise LLMRateLimitError(
                        error_data.get("message", "Rate limit exceeded"),
                        provider=LLMProvider.OPENAI,
                        model=payload["model"],
                        retry_after=int(retry_after) if retry_after else None
                    )
                
                if response.status_code == 402:
                    raise LLMQuotaError(
                        "Quota exceeded",
                        provider=LLMProvider.OPENAI,
                        model=payload["model"]
                    )
                
                if response.status_code == 400:
                    error_data = response.json().get("error", {})
                    raise LLMValidationError(
                        error_data.get("message", "Invalid request"),
                        provider=LLMProvider.OPENAI,
                        model=payload["model"]
                    )
                
                # Generic error for other status codes
                error_data = response.json().get("error", {})
                raise LLMError(
                    f"API request failed: {error_data.get('message', 'Unknown error')}",
                    provider=LLMProvider.OPENAI,
                    model=payload["model"],
                    error_code=str(response.status_code)
                )
                
            except httpx.TimeoutException as e:
                last_exception = LLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider=LLMProvider.OPENAI,
                    model=payload["model"],
                    original_error=e
                )
                
            except httpx.RequestError as e:
                last_exception = LLMError(
                    f"Request failed: {str(e)}",
                    provider=LLMProvider.OPENAI,
                    model=payload["model"],
                    original_error=e
                )
            
            except (LLMAuthenticationError, LLMQuotaError, LLMValidationError) as e:
                # Don't retry these errors
                raise e
            
            except LLMRateLimitError as e:
                if e.retry_after and attempt < self.max_retries:
                    logger.warning(
                        f"Rate limited, waiting {e.retry_after}s before retry {attempt + 1}"
                    )
                    await asyncio.sleep(e.retry_after)
                    last_exception = e
                else:
                    raise e
            
            except Exception as e:
                last_exception = LLMError(
                    f"Unexpected error: {str(e)}",
                    provider=LLMProvider.OPENAI,
                    model=payload["model"],
                    original_error=e
                )
            
            # Exponential backoff for retries
            if attempt < self.max_retries:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                logger.warning(f"Request failed, retrying in {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise LLMError(
                "All retries exhausted",
                provider=LLMProvider.OPENAI,
                model=payload["model"]
            )
    
    def _parse_response(
        self, 
        response_data: Dict[str, Any], 
        request: LLMRequest,
        latency_ms: float
    ) -> LLMResponse:
        """Parse OpenAI API response.
        
        Args:
            response_data: Raw response data
            request: Original request
            latency_ms: Response latency
            
        Returns:
            LLMResponse: Parsed response
        """
        choice = response_data["choices"][0]
        usage_data = response_data["usage"]
        
        # Calculate cost
        input_cost, output_cost = self.MODEL_PRICING.get(request.model, (0, 0))
        estimated_cost = (
            (usage_data["prompt_tokens"] / 1000) * input_cost +
            (usage_data["completion_tokens"] / 1000) * output_cost
        )
        
        # Create usage object
        usage = LLMUsage(
            prompt_tokens=usage_data["prompt_tokens"],
            completion_tokens=usage_data["completion_tokens"],
            total_tokens=usage_data["total_tokens"],
            estimated_cost=estimated_cost
        )
        
        return LLMResponse(
            content=choice["message"]["content"],
            model=response_data["model"],
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            response_id=response_data.get("id"),
            provider=LLMProvider.OPENAI,
            latency_ms=latency_ms
        )
    
    async def validate_request(self, request: LLMRequest) -> None:
        """Validate OpenAI-specific request parameters.
        
        Args:
            request: Request to validate
        """
        # Call parent validation
        self._validate_common_params(request)
        
        # Check model support
        if request.model not in self.MODEL_PRICING:
            raise LLMValidationError(f"Unsupported model: {request.model}")
        
        # Check context length
        total_tokens = sum(self._calculate_tokens(msg.content) for msg in request.messages)
        
        if request.max_tokens:
            total_tokens += request.max_tokens
        
        model_limit = self.MODEL_LIMITS.get(request.model, 4096)
        if total_tokens > model_limit:
            raise LLMValidationError(
                f"Request exceeds model context limit: {total_tokens} > {model_limit}"
            )
        
        # Validate OpenAI-specific parameters
        if request.top_p is not None and (request.top_p < 0 or request.top_p > 1):
            raise LLMValidationError("top_p must be between 0 and 1")
        
        if request.frequency_penalty is not None:
            if request.frequency_penalty < -2 or request.frequency_penalty > 2:
                raise LLMValidationError("frequency_penalty must be between -2 and 2")
        
        if request.presence_penalty is not None:
            if request.presence_penalty < -2 or request.presence_penalty > 2:
                raise LLMValidationError("presence_penalty must be between -2 and 2")
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for OpenAI request.
        
        Args:
            request: Request to estimate
            
        Returns:
            float: Estimated cost in USD
        """
        input_cost, output_cost = self.MODEL_PRICING.get(request.model, (0, 0))
        
        # Estimate input tokens
        input_tokens = sum(self._calculate_tokens(msg.content) for msg in request.messages)
        
        # Estimate output tokens (use max_tokens or reasonable default)
        output_tokens = request.max_tokens or 500
        
        return (input_tokens / 1000) * input_cost + (output_tokens / 1000) * output_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "provider": LLMProvider.OPENAI.value,
            "model": self.model.value,
            "context_limit": self.MODEL_LIMITS.get(self.model, 4096),
            "input_cost_per_1k": self.MODEL_PRICING.get(self.model, (0, 0))[0],
            "output_cost_per_1k": self.MODEL_PRICING.get(self.model, (0, 0))[1],
            "supports_streaming": True,
            "supports_function_calling": self.model in [
                LLMModelType.GPT_4, 
                LLMModelType.GPT_4_TURBO,
                LLMModelType.GPT_3_5_TURBO
            ]
        }
    
    def _calculate_tokens(self, text: str) -> int:
        """More accurate token estimation for OpenAI models.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # More accurate estimation for OpenAI models
        # This is still an approximation - for exact counts, use tiktoken library
        words = text.split()
        
        # Average tokens per word is roughly 1.3 for English
        return int(len(words) * 1.3)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()


# Export the class
__all__ = ["OpenAILLM"]