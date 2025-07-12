"""Anthropic Claude LLM integration for TruScholar.

This module provides integration with Anthropic's Claude models including
Claude 3 Opus, Sonnet, and Haiku.
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


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""
    
    # Model pricing per 1K tokens (input, output)
    MODEL_PRICING = {
        LLMModelType.CLAUDE_3_OPUS: (0.015, 0.075),
        LLMModelType.CLAUDE_3_SONNET: (0.003, 0.015),
        LLMModelType.CLAUDE_3_HAIKU: (0.00025, 0.00125),
    }
    
    # Model context limits
    MODEL_LIMITS = {
        LLMModelType.CLAUDE_3_OPUS: 200000,
        LLMModelType.CLAUDE_3_SONNET: 200000,
        LLMModelType.CLAUDE_3_HAIKU: 200000,
    }
    
    def __init__(
        self,
        api_key: str,
        model: LLMModelType = LLMModelType.CLAUDE_3_SONNET,
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        **kwargs
    ):
        """Initialize Anthropic LLM.
        
        Args:
            api_key: Anthropic API key
            model: Model to use
            base_url: Base URL for Anthropic API
            anthropic_version: API version
            **kwargs: Additional arguments for parent class
        """
        self.base_url = base_url
        self.anthropic_version = anthropic_version
        
        super().__init__(api_key, model, **kwargs)
    
    def _setup_client(self, **kwargs) -> None:
        """Setup Anthropic HTTP client."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API.
        
        Args:
            request: LLM request
            
        Returns:
            LLMResponse: Generated response
        """
        start_time = time.time()
        
        # Validate request
        await self.validate_request(request)
        
        # Prepare Anthropic API request
        api_request = self._prepare_api_request(request)
        
        # Make request with retries
        response_data = await self._make_request_with_retries(api_request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        return self._parse_response(response_data, request, latency_ms)
    
    def _prepare_api_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Prepare Anthropic API request payload.
        
        Args:
            request: LLM request
            
        Returns:
            Dict[str, Any]: API request payload
        """
        # Convert messages to Anthropic format
        # Anthropic expects system message separate from conversation
        system_message = None
        messages = []
        
        for msg in request.messages:
            if msg.role == LLMRole.SYSTEM:
                system_message = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        # Build request payload
        payload = {
            "model": request.model.value,
            "messages": messages,
            "max_tokens": request.max_tokens or 1000,  # Required for Anthropic
        }
        
        # Add system message if present
        if system_message:
            payload["system"] = system_message
        
        # Add optional parameters
        if request.temperature != 0.7:  # Only include if different from default
            payload["temperature"] = request.temperature
        
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        
        if request.stop:
            payload["stop_sequences"] = request.stop
        
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
                response = await self.client.post("/messages", json=payload)
                
                if response.status_code == 200:
                    return response.json()
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key",
                        provider=LLMProvider.ANTHROPIC,
                        model=payload["model"]
                    )
                
                if response.status_code == 429:
                    error_data = response.json().get("error", {})
                    retry_after = response.headers.get("retry-after")
                    
                    raise LLMRateLimitError(
                        error_data.get("message", "Rate limit exceeded"),
                        provider=LLMProvider.ANTHROPIC,
                        model=payload["model"],
                        retry_after=int(retry_after) if retry_after else None
                    )
                
                if response.status_code == 402:
                    raise LLMQuotaError(
                        "Credit limit exceeded",
                        provider=LLMProvider.ANTHROPIC,
                        model=payload["model"]
                    )
                
                if response.status_code == 400:
                    error_data = response.json().get("error", {})
                    raise LLMValidationError(
                        error_data.get("message", "Invalid request"),
                        provider=LLMProvider.ANTHROPIC,
                        model=payload["model"]
                    )
                
                # Generic error for other status codes
                error_data = response.json().get("error", {})
                raise LLMError(
                    f"API request failed: {error_data.get('message', 'Unknown error')}",
                    provider=LLMProvider.ANTHROPIC,
                    model=payload["model"],
                    error_code=str(response.status_code)
                )
                
            except httpx.TimeoutException as e:
                last_exception = LLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider=LLMProvider.ANTHROPIC,
                    model=payload["model"],
                    original_error=e
                )
                
            except httpx.RequestError as e:
                last_exception = LLMError(
                    f"Request failed: {str(e)}",
                    provider=LLMProvider.ANTHROPIC,
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
                    provider=LLMProvider.ANTHROPIC,
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
                provider=LLMProvider.ANTHROPIC,
                model=payload["model"]
            )
    
    def _parse_response(
        self, 
        response_data: Dict[str, Any], 
        request: LLMRequest,
        latency_ms: float
    ) -> LLMResponse:
        """Parse Anthropic API response.
        
        Args:
            response_data: Raw response data
            request: Original request
            latency_ms: Response latency
            
        Returns:
            LLMResponse: Parsed response
        """
        # Extract content from response
        content = ""
        if "content" in response_data and response_data["content"]:
            content = response_data["content"][0].get("text", "")
        
        # Extract usage information
        usage_data = response_data.get("usage", {})
        
        # Calculate cost
        input_cost, output_cost = self.MODEL_PRICING.get(request.model, (0, 0))
        estimated_cost = (
            (usage_data.get("input_tokens", 0) / 1000) * input_cost +
            (usage_data.get("output_tokens", 0) / 1000) * output_cost
        )
        
        # Create usage object
        usage = LLMUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
            estimated_cost=estimated_cost
        )
        
        return LLMResponse(
            content=content,
            model=response_data.get("model", request.model.value),
            usage=usage,
            finish_reason=response_data.get("stop_reason"),
            response_id=response_data.get("id"),
            provider=LLMProvider.ANTHROPIC,
            latency_ms=latency_ms
        )
    
    async def validate_request(self, request: LLMRequest) -> None:
        """Validate Anthropic-specific request parameters.
        
        Args:
            request: Request to validate
        """
        # Call parent validation
        self._validate_common_params(request)
        
        # Check model support
        if request.model not in self.MODEL_PRICING:
            raise LLMValidationError(f"Unsupported model: {request.model}")
        
        # Anthropic requires max_tokens
        if request.max_tokens is None:
            raise LLMValidationError("max_tokens is required for Anthropic models")
        
        # Check context length
        total_tokens = sum(self._calculate_tokens(msg.content) for msg in request.messages)
        total_tokens += request.max_tokens
        
        model_limit = self.MODEL_LIMITS.get(request.model, 200000)
        if total_tokens > model_limit:
            raise LLMValidationError(
                f"Request exceeds model context limit: {total_tokens} > {model_limit}"
            )
        
        # Validate Anthropic-specific parameters
        if request.top_p is not None and (request.top_p < 0 or request.top_p > 1):
            raise LLMValidationError("top_p must be between 0 and 1")
        
        # Anthropic doesn't support frequency_penalty or presence_penalty
        if request.frequency_penalty is not None:
            logger.warning("frequency_penalty is not supported by Anthropic, ignoring")
        
        if request.presence_penalty is not None:
            logger.warning("presence_penalty is not supported by Anthropic, ignoring")
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Anthropic request.
        
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
        """Get Anthropic model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "provider": LLMProvider.ANTHROPIC.value,
            "model": self.model.value,
            "context_limit": self.MODEL_LIMITS.get(self.model, 200000),
            "input_cost_per_1k": self.MODEL_PRICING.get(self.model, (0, 0))[0],
            "output_cost_per_1k": self.MODEL_PRICING.get(self.model, (0, 0))[1],
            "supports_streaming": True,
            "supports_function_calling": False,  # Claude doesn't support function calling yet
            "requires_max_tokens": True
        }
    
    def _calculate_tokens(self, text: str) -> int:
        """Token estimation for Anthropic models.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Anthropic uses similar tokenization to OpenAI
        # This is an approximation - for exact counts, use Anthropic's tokenizer
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
__all__ = ["AnthropicLLM"]