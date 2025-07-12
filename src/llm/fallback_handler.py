"""Fallback handler for LLM requests in TruScholar.

This module provides sophisticated fallback mechanisms when primary
LLM providers fail, including retry logic, provider switching, and
graceful degradation strategies.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field

from .base_llm import (
    BaseLLM, LLMRequest, LLMResponse, LLMProvider,
    LLMError, LLMRateLimitError, LLMQuotaError,
    LLMAuthenticationError, LLMTimeoutError
)
from .llm_factory import LLMFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FallbackStrategy(str, Enum):
    """Fallback strategies for handling LLM failures."""
    
    PROVIDER_SWITCH = "provider_switch"  # Switch to different provider
    MODEL_DOWNGRADE = "model_downgrade"  # Use simpler model
    STATIC_RESPONSE = "static_response"  # Return predefined response
    CACHE_LOOKUP = "cache_lookup"       # Look for cached similar response
    RETRY_EXPONENTIAL = "retry_exponential"  # Exponential backoff retry


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    
    # Maximum total attempts across all strategies
    max_total_attempts: int = 5
    
    # Maximum time to spend on a single request (seconds)
    max_total_time: float = 60.0
    
    # Enabled fallback strategies in order of preference
    strategies: List[FallbackStrategy] = field(default_factory=lambda: [
        FallbackStrategy.RETRY_EXPONENTIAL,
        FallbackStrategy.PROVIDER_SWITCH,
        FallbackStrategy.MODEL_DOWNGRADE,
        FallbackStrategy.STATIC_RESPONSE
    ])
    
    # Provider priority order for switching
    provider_priority: List[LLMProvider] = field(default_factory=lambda: [
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.GEMINI
    ])
    
    # Static responses for emergency fallback
    static_responses: Dict[str, str] = field(default_factory=lambda: {
        "question_generation": "I apologize, but I'm currently unable to generate questions. Please try again later.",
        "default": "I'm experiencing technical difficulties. Please try your request again."
    })
    
    # Errors that should trigger immediate fallback (no retry)
    immediate_fallback_errors: List[type] = field(default_factory=lambda: [
        LLMAuthenticationError,
        LLMQuotaError
    ])
    
    # Errors that should be retried
    retryable_errors: List[type] = field(default_factory=lambda: [
        LLMRateLimitError,
        LLMTimeoutError,
        LLMError  # Generic LLM errors
    ])


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    
    strategy: FallbackStrategy
    provider: Optional[LLMProvider]
    model: Optional[str]
    error: Optional[Exception]
    response: Optional[LLMResponse]
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


class FallbackHandler:
    """Handles LLM request fallbacks and retries."""
    
    def __init__(
        self,
        config: Optional[FallbackConfig] = None,
        cache_lookup_fn: Optional[Callable] = None,
        health_check_interval: float = 300.0  # 5 minutes
    ):
        """Initialize fallback handler.
        
        Args:
            config: Fallback configuration
            cache_lookup_fn: Function to lookup cached responses
            health_check_interval: Interval for provider health checks
        """
        self.config = config or FallbackConfig()
        self.cache_lookup_fn = cache_lookup_fn
        self.health_check_interval = health_check_interval
        
        # Track provider health
        self._provider_health: Dict[LLMProvider, bool] = {}
        self._last_health_check: float = 0
        
        # Track attempt history for learning
        self._attempt_history: List[FallbackAttempt] = []
    
    async def execute_with_fallback(
        self,
        primary_llm: BaseLLM,
        request: LLMRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Execute LLM request with fallback handling.
        
        Args:
            primary_llm: Primary LLM to try first
            request: LLM request
            context: Additional context for fallback decisions
            
        Returns:
            LLMResponse: Response from successful attempt
            
        Raises:
            LLMError: If all fallback strategies are exhausted
        """
        start_time = time.time()
        attempts = []
        context = context or {}
        
        # Update provider health if needed
        await self._update_provider_health_if_needed()
        
        logger.info(f"Starting LLM request with fallback handling")
        
        try:
            # Try primary LLM first
            response = await self._attempt_request(primary_llm, request)
            
            # Log successful attempt
            attempt = FallbackAttempt(
                strategy=FallbackStrategy.RETRY_EXPONENTIAL,  # Primary attempt
                provider=self._get_provider_from_llm(primary_llm),
                model=primary_llm.model.value,
                error=None,
                response=response,
                duration_ms=(time.time() - start_time) * 1000
            )
            attempts.append(attempt)
            self._attempt_history.append(attempt)
            
            logger.info(f"Primary LLM succeeded on first attempt")
            return response
            
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}")
            
            # Log failed attempt
            attempt = FallbackAttempt(
                strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                provider=self._get_provider_from_llm(primary_llm),
                model=primary_llm.model.value,
                error=e,
                response=None,
                duration_ms=(time.time() - start_time) * 1000
            )
            attempts.append(attempt)
            self._attempt_history.append(attempt)
            
            # Check if we should immediately fallback
            if any(isinstance(e, error_type) for error_type in self.config.immediate_fallback_errors):
                logger.warning(f"Immediate fallback triggered by {type(e).__name__}")
            else:
                # Try retrying primary LLM first
                retry_response = await self._try_retry_strategy(primary_llm, request, e, attempts)
                if retry_response:
                    return retry_response
        
        # Execute fallback strategies
        for strategy in self.config.strategies:
            if len(attempts) >= self.config.max_total_attempts:
                break
                
            if (time.time() - start_time) > self.config.max_total_time:
                break
            
            try:
                response = await self._execute_strategy(
                    strategy, primary_llm, request, context, attempts
                )
                if response:
                    logger.info(f"Fallback successful with strategy: {strategy}")
                    return response
                    
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy} failed: {e}")
                continue
        
        # All strategies exhausted
        total_duration = time.time() - start_time
        logger.error(f"All fallback strategies exhausted after {total_duration:.2f}s")
        
        # Raise the most recent error
        last_error = attempts[-1].error if attempts else LLMError("Unknown error")
        raise LLMError(
            f"All fallback strategies failed. Total attempts: {len(attempts)}",
            original_error=last_error
        )
    
    async def _attempt_request(self, llm: BaseLLM, request: LLMRequest) -> LLMResponse:
        """Attempt a request with a specific LLM.
        
        Args:
            llm: LLM instance
            request: Request to execute
            
        Returns:
            LLMResponse: Response from LLM
        """
        return await llm.generate(request)
    
    async def _try_retry_strategy(
        self,
        llm: BaseLLM,
        request: LLMRequest,
        initial_error: Exception,
        attempts: List[FallbackAttempt]
    ) -> Optional[LLMResponse]:
        """Try retry strategy with exponential backoff.
        
        Args:
            llm: LLM instance
            request: Request to retry
            initial_error: Initial error that triggered retry
            attempts: List to track attempts
            
        Returns:
            Optional[LLMResponse]: Response if successful, None otherwise
        """
        # Check if error is retryable
        if not any(isinstance(initial_error, error_type) for error_type in self.config.retryable_errors):
            return None
        
        # Calculate retry attempts based on error type
        max_retries = 3
        if isinstance(initial_error, LLMRateLimitError):
            max_retries = 5  # More retries for rate limits
        
        for retry_attempt in range(max_retries):
            if len(attempts) >= self.config.max_total_attempts:
                break
            
            # Calculate wait time
            if isinstance(initial_error, LLMRateLimitError) and initial_error.retry_after:
                wait_time = initial_error.retry_after
            else:
                wait_time = (2 ** retry_attempt) + (0.1 * retry_attempt)
            
            logger.info(f"Retrying in {wait_time}s (attempt {retry_attempt + 1})")
            await asyncio.sleep(wait_time)
            
            try:
                start_time = time.time()
                response = await self._attempt_request(llm, request)
                
                # Log successful retry
                attempt = FallbackAttempt(
                    strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                    provider=self._get_provider_from_llm(llm),
                    model=llm.model.value,
                    error=None,
                    response=response,
                    duration_ms=(time.time() - start_time) * 1000
                )
                attempts.append(attempt)
                self._attempt_history.append(attempt)
                
                logger.info(f"Retry successful on attempt {retry_attempt + 1}")
                return response
                
            except Exception as e:
                start_time = time.time()
                attempt = FallbackAttempt(
                    strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                    provider=self._get_provider_from_llm(llm),
                    model=llm.model.value,
                    error=e,
                    response=None,
                    duration_ms=(time.time() - start_time) * 1000
                )
                attempts.append(attempt)
                self._attempt_history.append(attempt)
                
                logger.warning(f"Retry {retry_attempt + 1} failed: {e}")
                
                # Update error for next iteration
                initial_error = e
        
        return None
    
    async def _execute_strategy(
        self,
        strategy: FallbackStrategy,
        primary_llm: BaseLLM,
        request: LLMRequest,
        context: Dict[str, Any],
        attempts: List[FallbackAttempt]
    ) -> Optional[LLMResponse]:
        """Execute a specific fallback strategy.
        
        Args:
            strategy: Fallback strategy to execute
            primary_llm: Primary LLM instance
            request: Original request
            context: Additional context
            attempts: List to track attempts
            
        Returns:
            Optional[LLMResponse]: Response if successful
        """
        if strategy == FallbackStrategy.PROVIDER_SWITCH:
            return await self._try_provider_switch(request, attempts)
            
        elif strategy == FallbackStrategy.MODEL_DOWNGRADE:
            return await self._try_model_downgrade(primary_llm, request, attempts)
            
        elif strategy == FallbackStrategy.STATIC_RESPONSE:
            return self._try_static_response(request, context)
            
        elif strategy == FallbackStrategy.CACHE_LOOKUP:
            return await self._try_cache_lookup(request, context)
        
        return None
    
    async def _try_provider_switch(
        self,
        request: LLMRequest,
        attempts: List[FallbackAttempt]
    ) -> Optional[LLMResponse]:
        """Try switching to a different provider.
        
        Args:
            request: Request to execute
            attempts: List to track attempts
            
        Returns:
            Optional[LLMResponse]: Response if successful
        """
        # Get providers to try (excluding already failed ones)
        tried_providers = {attempt.provider for attempt in attempts if attempt.provider}
        available_providers = [
            p for p in self.config.provider_priority 
            if p not in tried_providers and self._provider_health.get(p, True)
        ]
        
        for provider in available_providers:
            try:
                start_time = time.time()
                
                # Create LLM for this provider
                llm = LLMFactory.create_llm(provider)
                
                # Execute request
                response = await self._attempt_request(llm, request)
                
                # Log successful attempt
                attempt = FallbackAttempt(
                    strategy=FallbackStrategy.PROVIDER_SWITCH,
                    provider=provider,
                    model=llm.model.value,
                    error=None,
                    response=response,
                    duration_ms=(time.time() - start_time) * 1000
                )
                attempts.append(attempt)
                self._attempt_history.append(attempt)
                
                logger.info(f"Provider switch to {provider} successful")
                return response
                
            except Exception as e:
                # Log failed attempt
                attempt = FallbackAttempt(
                    strategy=FallbackStrategy.PROVIDER_SWITCH,
                    provider=provider,
                    model=getattr(llm, 'model', {}).get('value', 'unknown') if 'llm' in locals() else 'unknown',
                    error=e,
                    response=None,
                    duration_ms=(time.time() - start_time) * 1000
                )
                attempts.append(attempt)
                self._attempt_history.append(attempt)
                
                logger.warning(f"Provider switch to {provider} failed: {e}")
                
                # Mark provider as unhealthy if authentication fails
                if isinstance(e, LLMAuthenticationError):
                    self._provider_health[provider] = False
                
                continue
        
        return None
    
    async def _try_model_downgrade(
        self,
        primary_llm: BaseLLM,
        request: LLMRequest,
        attempts: List[FallbackAttempt]
    ) -> Optional[LLMResponse]:
        """Try using a simpler/cheaper model.
        
        Args:
            primary_llm: Primary LLM instance
            request: Request to execute
            attempts: List to track attempts
            
        Returns:
            Optional[LLMResponse]: Response if successful
        """
        provider = self._get_provider_from_llm(primary_llm)
        
        # Define model downgrade paths
        downgrade_map = {
            # OpenAI downgrades
            "gpt-4-turbo-preview": "gpt-4",
            "gpt-4": "gpt-3.5-turbo",
            
            # Anthropic downgrades
            "claude-3-opus-20240229": "claude-3-sonnet-20240229",
            "claude-3-sonnet-20240229": "claude-3-haiku-20240307",
            
            # Gemini downgrades (limited options)
            "gemini-pro-vision": "gemini-pro",
        }
        
        current_model = primary_llm.model.value
        downgraded_model = downgrade_map.get(current_model)
        
        if not downgraded_model:
            logger.info(f"No downgrade available for model {current_model}")
            return None
        
        try:
            start_time = time.time()
            
            # Create LLM with downgraded model
            from .base_llm import LLMModelType
            downgraded_model_enum = LLMModelType(downgraded_model)
            
            llm = LLMFactory.create_llm(provider, model=downgraded_model_enum)
            
            # Execute request
            response = await self._attempt_request(llm, request)
            
            # Log successful attempt
            attempt = FallbackAttempt(
                strategy=FallbackStrategy.MODEL_DOWNGRADE,
                provider=provider,
                model=downgraded_model,
                error=None,
                response=response,
                duration_ms=(time.time() - start_time) * 1000
            )
            attempts.append(attempt)
            self._attempt_history.append(attempt)
            
            logger.info(f"Model downgrade to {downgraded_model} successful")
            return response
            
        except Exception as e:
            # Log failed attempt
            attempt = FallbackAttempt(
                strategy=FallbackStrategy.MODEL_DOWNGRADE,
                provider=provider,
                model=downgraded_model,
                error=e,
                response=None,
                duration_ms=(time.time() - start_time) * 1000
            )
            attempts.append(attempt)
            self._attempt_history.append(attempt)
            
            logger.warning(f"Model downgrade to {downgraded_model} failed: {e}")
            return None
    
    def _try_static_response(
        self,
        request: LLMRequest,
        context: Dict[str, Any]
    ) -> Optional[LLMResponse]:
        """Return a static response as last resort.
        
        Args:
            request: Original request
            context: Additional context
            
        Returns:
            Optional[LLMResponse]: Static response
        """
        # Determine response type from context
        purpose = context.get("purpose", "default")
        static_content = self.config.static_responses.get(purpose)
        
        if not static_content:
            static_content = self.config.static_responses.get("default")
        
        if not static_content:
            return None
        
        # Create a static response
        from .base_llm import LLMUsage
        
        usage = LLMUsage(
            prompt_tokens=0,
            completion_tokens=len(static_content.split()),
            total_tokens=len(static_content.split()),
            estimated_cost=0.0
        )
        
        response = LLMResponse(
            content=static_content,
            model="static",
            usage=usage,
            finish_reason="static",
            provider=LLMProvider.OPENAI,  # Arbitrary
            latency_ms=0.0
        )
        
        logger.info("Returning static fallback response")
        return response
    
    async def _try_cache_lookup(
        self,
        request: LLMRequest,
        context: Dict[str, Any]
    ) -> Optional[LLMResponse]:
        """Try to find a cached similar response.
        
        Args:
            request: Original request
            context: Additional context
            
        Returns:
            Optional[LLMResponse]: Cached response if found
        """
        if not self.cache_lookup_fn:
            return None
        
        try:
            cached_response = await self.cache_lookup_fn(request, context)
            if cached_response:
                logger.info("Found cached response for fallback")
                return cached_response
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
        
        return None
    
    async def _update_provider_health_if_needed(self):
        """Update provider health status if interval has passed."""
        current_time = time.time()
        
        if (current_time - self._last_health_check) > self.health_check_interval:
            logger.info("Updating provider health status")
            
            try:
                health_status = await LLMFactory.health_check_all()
                self._provider_health.update(health_status)
                self._last_health_check = current_time
                
                logger.info(f"Provider health updated: {health_status}")
                
            except Exception as e:
                logger.warning(f"Provider health check failed: {e}")
    
    def _get_provider_from_llm(self, llm: BaseLLM) -> Optional[LLMProvider]:
        """Extract provider from LLM instance.
        
        Args:
            llm: LLM instance
            
        Returns:
            Optional[LLMProvider]: Provider if identifiable
        """
        class_name = llm.__class__.__name__
        
        if "OpenAI" in class_name:
            return LLMProvider.OPENAI
        elif "Anthropic" in class_name:
            return LLMProvider.ANTHROPIC
        elif "Gemini" in class_name:
            return LLMProvider.GEMINI
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback handler statistics.
        
        Returns:
            Dict[str, Any]: Statistics about fallback performance
        """
        if not self._attempt_history:
            return {"total_attempts": 0}
        
        # Calculate statistics
        total_attempts = len(self._attempt_history)
        successful_attempts = len([a for a in self._attempt_history if a.response])
        
        # Success rate by strategy
        strategy_stats = {}
        for strategy in FallbackStrategy:
            strategy_attempts = [a for a in self._attempt_history if a.strategy == strategy]
            if strategy_attempts:
                successes = len([a for a in strategy_attempts if a.response])
                strategy_stats[strategy.value] = {
                    "attempts": len(strategy_attempts),
                    "successes": successes,
                    "success_rate": successes / len(strategy_attempts)
                }
        
        # Provider health
        provider_health = dict(self._provider_health)
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "overall_success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
            "strategy_statistics": strategy_stats,
            "provider_health": provider_health,
            "last_health_check": self._last_health_check
        }
    
    def clear_history(self):
        """Clear attempt history."""
        self._attempt_history.clear()
        logger.info("Fallback attempt history cleared")


# Export the handler and related classes
__all__ = [
    "FallbackHandler",
    "FallbackStrategy", 
    "FallbackConfig",
    "FallbackAttempt"
]