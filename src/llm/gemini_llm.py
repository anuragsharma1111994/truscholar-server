"""Google Gemini LLM integration for TruScholar.

This module provides integration with Google's Gemini models including
Gemini Pro and Gemini Pro Vision.
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


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation."""
    
    # Model pricing per 1K tokens (input, output)
    # Note: Gemini pricing may vary, these are example rates
    MODEL_PRICING = {
        LLMModelType.GEMINI_PRO: (0.0005, 0.0015),
        LLMModelType.GEMINI_PRO_VISION: (0.0005, 0.0015),
    }
    
    # Model context limits
    MODEL_LIMITS = {
        LLMModelType.GEMINI_PRO: 32768,
        LLMModelType.GEMINI_PRO_VISION: 16384,
    }
    
    def __init__(
        self,
        api_key: str,
        model: LLMModelType = LLMModelType.GEMINI_PRO,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        **kwargs
    ):
        """Initialize Gemini LLM.
        
        Args:
            api_key: Google API key
            model: Model to use
            base_url: Base URL for Google AI API
            **kwargs: Additional arguments for parent class
        """
        self.base_url = base_url
        
        super().__init__(api_key, model, **kwargs)
    
    def _setup_client(self, **kwargs) -> None:
        """Setup Google AI HTTP client."""
        # Google AI uses API key as query parameter
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Gemini API.
        
        Args:
            request: LLM request
            
        Returns:
            LLMResponse: Generated response
        """
        start_time = time.time()
        
        # Validate request
        await self.validate_request(request)
        
        # Prepare Gemini API request
        api_request, endpoint = self._prepare_api_request(request)
        
        # Make request with retries
        response_data = await self._make_request_with_retries(api_request, endpoint)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        return self._parse_response(response_data, request, latency_ms)
    
    def _prepare_api_request(self, request: LLMRequest) -> tuple[Dict[str, Any], str]:
        """Prepare Gemini API request payload.
        
        Args:
            request: LLM request
            
        Returns:
            Tuple[Dict[str, Any], str]: API request payload and endpoint
        """
        # Convert messages to Gemini format
        contents = []
        
        for msg in request.messages:
            if msg.role == LLMRole.SYSTEM:
                # Gemini doesn't have explicit system role, prepend to first user message
                if contents and contents[-1].get("role") == "user":
                    contents[-1]["parts"][0]["text"] = f"System: {msg.content}\n\nUser: {contents[-1]['parts'][0]['text']}"
                else:
                    contents.append({
                        "role": "user",
                        "parts": [{"text": f"System: {msg.content}\n\nPlease respond as if this was a user message:"}]
                    })
            else:
                # Map roles
                role = "user" if msg.role == LLMRole.USER else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        # Build generation config
        generation_config = {
            "temperature": request.temperature,
        }
        
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        
        if request.stop:
            generation_config["stopSequences"] = request.stop
        
        # Build request payload
        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        # Determine endpoint
        model_name = request.model.value
        if request.stream:
            endpoint = f"/models/{model_name}:streamGenerateContent"
        else:
            endpoint = f"/models/{model_name}:generateContent"
        
        return payload, endpoint
    
    async def _make_request_with_retries(
        self, 
        payload: Dict[str, Any], 
        endpoint: str
    ) -> Dict[str, Any]:
        """Make API request with exponential backoff retries.
        
        Args:
            payload: Request payload
            endpoint: API endpoint
            
        Returns:
            Dict[str, Any]: Response data
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add API key as query parameter
                params = {"key": self.api_key}
                
                response = await self.client.post(endpoint, json=payload, params=params)
                
                if response.status_code == 200:
                    return response.json()
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key",
                        provider=LLMProvider.GEMINI,
                        model=payload.get("model", "unknown")
                    )
                
                if response.status_code == 429:
                    error_data = response.json().get("error", {})
                    retry_after = response.headers.get("retry-after")
                    
                    raise LLMRateLimitError(
                        error_data.get("message", "Rate limit exceeded"),
                        provider=LLMProvider.GEMINI,
                        model=payload.get("model", "unknown"),
                        retry_after=int(retry_after) if retry_after else None
                    )
                
                if response.status_code == 402:
                    raise LLMQuotaError(
                        "Quota exceeded",
                        provider=LLMProvider.GEMINI,
                        model=payload.get("model", "unknown")
                    )
                
                if response.status_code == 400:
                    error_data = response.json().get("error", {})
                    raise LLMValidationError(
                        error_data.get("message", "Invalid request"),
                        provider=LLMProvider.GEMINI,
                        model=payload.get("model", "unknown")
                    )
                
                # Generic error for other status codes
                error_data = response.json().get("error", {})
                raise LLMError(
                    f"API request failed: {error_data.get('message', 'Unknown error')}",
                    provider=LLMProvider.GEMINI,
                    model=payload.get("model", "unknown"),
                    error_code=str(response.status_code)
                )
                
            except httpx.TimeoutException as e:
                last_exception = LLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider=LLMProvider.GEMINI,
                    model=payload.get("model", "unknown"),
                    original_error=e
                )
                
            except httpx.RequestError as e:
                last_exception = LLMError(
                    f"Request failed: {str(e)}",
                    provider=LLMProvider.GEMINI,
                    model=payload.get("model", "unknown"),
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
                    provider=LLMProvider.GEMINI,
                    model=payload.get("model", "unknown"),
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
                provider=LLMProvider.GEMINI,
                model=payload.get("model", "unknown")
            )
    
    def _parse_response(
        self, 
        response_data: Dict[str, Any], 
        request: LLMRequest,
        latency_ms: float
    ) -> LLMResponse:
        """Parse Gemini API response.
        
        Args:
            response_data: Raw response data
            request: Original request
            latency_ms: Response latency
            
        Returns:
            LLMResponse: Parsed response
        """
        # Extract content from response
        content = ""
        candidates = response_data.get("candidates", [])
        
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "")
        
        # Extract usage information (if available)
        usage_metadata = response_data.get("usageMetadata", {})
        
        # Calculate estimated tokens and cost
        input_tokens = usage_metadata.get("promptTokenCount", 0)
        output_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", input_tokens + output_tokens)
        
        # Calculate cost
        input_cost, output_cost = self.MODEL_PRICING.get(request.model, (0, 0))
        estimated_cost = (
            (input_tokens / 1000) * input_cost +
            (output_tokens / 1000) * output_cost
        )
        
        # Create usage object
        usage = LLMUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost
        )
        
        # Get finish reason
        finish_reason = None
        if candidates:
            finish_reason = candidates[0].get("finishReason")
        
        return LLMResponse(
            content=content,
            model=request.model.value,
            usage=usage,
            finish_reason=finish_reason,
            response_id=None,  # Gemini doesn't provide response IDs
            provider=LLMProvider.GEMINI,
            latency_ms=latency_ms
        )
    
    async def validate_request(self, request: LLMRequest) -> None:
        """Validate Gemini-specific request parameters.
        
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
        
        model_limit = self.MODEL_LIMITS.get(request.model, 32768)
        if total_tokens > model_limit:
            raise LLMValidationError(
                f"Request exceeds model context limit: {total_tokens} > {model_limit}"
            )
        
        # Validate Gemini-specific parameters
        if request.top_p is not None and (request.top_p < 0 or request.top_p > 1):
            raise LLMValidationError("top_p must be between 0 and 1")
        
        # Gemini doesn't support frequency_penalty or presence_penalty
        if request.frequency_penalty is not None:
            logger.warning("frequency_penalty is not supported by Gemini, ignoring")
        
        if request.presence_penalty is not None:
            logger.warning("presence_penalty is not supported by Gemini, ignoring")
    
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate cost for Gemini request.
        
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
        """Get Gemini model information.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "provider": LLMProvider.GEMINI.value,
            "model": self.model.value,
            "context_limit": self.MODEL_LIMITS.get(self.model, 32768),
            "input_cost_per_1k": self.MODEL_PRICING.get(self.model, (0, 0))[0],
            "output_cost_per_1k": self.MODEL_PRICING.get(self.model, (0, 0))[1],
            "supports_streaming": True,
            "supports_function_calling": False,  # Gemini has different function calling
            "supports_vision": self.model == LLMModelType.GEMINI_PRO_VISION
        }
    
    def _calculate_tokens(self, text: str) -> int:
        """Token estimation for Gemini models.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Gemini uses similar tokenization patterns
        # This is an approximation
        words = text.split()
        
        # Average tokens per word for Gemini
        return int(len(words) * 1.2)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()


# Export the class
__all__ = ["GeminiLLM"]