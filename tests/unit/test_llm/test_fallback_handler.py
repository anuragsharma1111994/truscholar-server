"""Unit tests for Fallback Handler in TruScholar.

This module tests the LLM fallback handler functionality including retry logic,
provider switching, model downgrading, and graceful degradation strategies.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional

from src.llm.fallback_handler import (
    FallbackHandler, FallbackStrategy, FallbackConfig, FallbackAttempt
)
from src.llm.base_llm import (
    BaseLLM, LLMRequest, LLMResponse, LLMMessage, LLMUsage, LLMProvider, 
    LLMModelType, LLMRole, LLMError, LLMRateLimitError, LLMQuotaError,
    LLMAuthenticationError, LLMTimeoutError
)


class TestFallbackConfig:
    """Test cases for FallbackConfig."""
    
    def test_default_config(self):
        """Test default fallback configuration."""
        config = FallbackConfig()
        
        assert config.max_total_attempts == 5
        assert config.max_total_time == 60.0
        assert len(config.strategies) == 4
        assert FallbackStrategy.RETRY_EXPONENTIAL in config.strategies
        assert FallbackStrategy.PROVIDER_SWITCH in config.strategies
        assert FallbackStrategy.MODEL_DOWNGRADE in config.strategies
        assert FallbackStrategy.STATIC_RESPONSE in config.strategies
        
        assert len(config.provider_priority) == 3
        assert LLMProvider.OPENAI in config.provider_priority
        assert LLMProvider.ANTHROPIC in config.provider_priority
        assert LLMProvider.GEMINI in config.provider_priority
        
        assert "default" in config.static_responses
        assert "question_generation" in config.static_responses
    
    def test_custom_config(self):
        """Test custom fallback configuration."""
        config = FallbackConfig(
            max_total_attempts=10,
            max_total_time=120.0,
            strategies=[FallbackStrategy.PROVIDER_SWITCH, FallbackStrategy.STATIC_RESPONSE],
            provider_priority=[LLMProvider.ANTHROPIC, LLMProvider.OPENAI],
            static_responses={"custom": "Custom response"}
        )
        
        assert config.max_total_attempts == 10
        assert config.max_total_time == 120.0
        assert len(config.strategies) == 2
        assert config.provider_priority[0] == LLMProvider.ANTHROPIC
        assert config.static_responses["custom"] == "Custom response"


class TestFallbackAttempt:
    """Test cases for FallbackAttempt."""
    
    def test_fallback_attempt_creation(self):
        """Test creating fallback attempt record."""
        mock_error = LLMError("Test error")
        mock_response = Mock(spec=LLMResponse)
        
        attempt = FallbackAttempt(
            strategy=FallbackStrategy.PROVIDER_SWITCH,
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            error=mock_error,
            response=mock_response,
            duration_ms=150.5
        )
        
        assert attempt.strategy == FallbackStrategy.PROVIDER_SWITCH
        assert attempt.provider == LLMProvider.OPENAI
        assert attempt.model == "gpt-4"
        assert attempt.error == mock_error
        assert attempt.response == mock_response
        assert attempt.duration_ms == 150.5
        assert attempt.timestamp > 0


class TestFallbackHandler:
    """Test cases for FallbackHandler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = FallbackConfig(max_total_attempts=3, max_total_time=30.0)
        self.handler = FallbackHandler(config=self.config)
        
        # Mock LLM request
        self.mock_request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.USER, content="Test message")
            ],
            model=LLMModelType.GPT_4,
            max_tokens=100
        )
        
        # Mock successful response
        self.mock_response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, estimated_cost=0.001),
            finish_reason="stop",
            provider=LLMProvider.OPENAI,
            latency_ms=150.0
        )
    
    def test_init_default_config(self):
        """Test initializing handler with default config."""
        handler = FallbackHandler()
        
        assert handler.config is not None
        assert handler.cache_lookup_fn is None
        assert handler.health_check_interval == 300.0
        assert len(handler._provider_health) == 0
        assert len(handler._attempt_history) == 0
    
    def test_init_custom_config(self):
        """Test initializing handler with custom config."""
        cache_fn = Mock()
        config = FallbackConfig(max_total_attempts=10)
        
        handler = FallbackHandler(
            config=config,
            cache_lookup_fn=cache_fn,
            health_check_interval=600.0
        )
        
        assert handler.config == config
        assert handler.cache_lookup_fn == cache_fn
        assert handler.health_check_interval == 600.0
    
    async def test_execute_with_fallback_success_primary(self):
        """Test successful execution with primary LLM."""
        mock_llm = AsyncMock(spec=BaseLLM)
        mock_llm.generate.return_value = self.mock_response
        mock_llm.model = LLMModelType.GPT_4
        
        with patch.object(self.handler, '_update_provider_health_if_needed'):
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                result = await self.handler.execute_with_fallback(mock_llm, self.mock_request)
        
        assert result == self.mock_response
        assert len(self.handler._attempt_history) == 1
        assert self.handler._attempt_history[0].error is None
        assert self.handler._attempt_history[0].response == self.mock_response
    
    async def test_execute_with_fallback_retry_success(self):
        """Test successful execution after retry."""
        mock_llm = AsyncMock(spec=BaseLLM)
        # First call fails, second succeeds
        mock_llm.generate.side_effect = [
            LLMRateLimitError("Rate limited", provider=LLMProvider.OPENAI, model="gpt-4"),
            self.mock_response
        ]
        mock_llm.model = LLMModelType.GPT_4
        
        with patch.object(self.handler, '_update_provider_health_if_needed'):
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                with patch('asyncio.sleep'):  # Speed up test
                    result = await self.handler.execute_with_fallback(mock_llm, self.mock_request)
        
        assert result == self.mock_response
        assert len(self.handler._attempt_history) >= 2
        # First attempt should have error, second should succeed
        assert self.handler._attempt_history[0].error is not None
        assert self.handler._attempt_history[-1].response == self.mock_response
    
    async def test_execute_with_fallback_immediate_fallback_error(self):
        """Test immediate fallback on authentication error."""
        mock_llm = AsyncMock(spec=BaseLLM)
        auth_error = LLMAuthenticationError("Invalid API key", provider=LLMProvider.OPENAI, model="gpt-4")
        mock_llm.generate.side_effect = auth_error
        mock_llm.model = LLMModelType.GPT_4
        
        with patch.object(self.handler, '_update_provider_health_if_needed'):
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                with patch.object(self.handler, '_execute_strategy', return_value=self.mock_response):
                    result = await self.handler.execute_with_fallback(mock_llm, self.mock_request)
        
        assert result == self.mock_response
        # Should not retry authentication errors
        assert len(self.handler._attempt_history) >= 1
        assert isinstance(self.handler._attempt_history[0].error, LLMAuthenticationError)
    
    async def test_execute_with_fallback_all_strategies_fail(self):
        """Test when all fallback strategies fail."""
        mock_llm = AsyncMock(spec=BaseLLM)
        error = LLMError("Persistent error", provider=LLMProvider.OPENAI, model="gpt-4")
        mock_llm.generate.side_effect = error
        mock_llm.model = LLMModelType.GPT_4
        
        with patch.object(self.handler, '_update_provider_health_if_needed'):
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                with patch.object(self.handler, '_execute_strategy', return_value=None):
                    with pytest.raises(LLMError, match="All fallback strategies failed"):
                        await self.handler.execute_with_fallback(mock_llm, self.mock_request)
    
    async def test_execute_with_fallback_max_attempts_exceeded(self):
        """Test when max attempts are exceeded."""
        # Set very low max attempts
        self.handler.config.max_total_attempts = 1
        
        mock_llm = AsyncMock(spec=BaseLLM)
        error = LLMError("Error", provider=LLMProvider.OPENAI, model="gpt-4")
        mock_llm.generate.side_effect = error
        mock_llm.model = LLMModelType.GPT_4
        
        with patch.object(self.handler, '_update_provider_health_if_needed'):
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                with pytest.raises(LLMError):
                    await self.handler.execute_with_fallback(mock_llm, self.mock_request)
    
    async def test_execute_with_fallback_timeout_exceeded(self):
        """Test when total time limit is exceeded."""
        # Set very low timeout
        self.handler.config.max_total_time = 0.1
        
        mock_llm = AsyncMock(spec=BaseLLM)
        
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.2)  # Exceed timeout
            return self.mock_response
        
        mock_llm.generate.side_effect = slow_generate
        mock_llm.model = LLMModelType.GPT_4
        
        with patch.object(self.handler, '_update_provider_health_if_needed'):
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                with pytest.raises(LLMError):
                    await self.handler.execute_with_fallback(mock_llm, self.mock_request)
    
    async def test_try_retry_strategy_rate_limit_with_retry_after(self):
        """Test retry strategy with rate limit and retry_after header."""
        mock_llm = AsyncMock(spec=BaseLLM)
        rate_limit_error = LLMRateLimitError(
            "Rate limited", 
            provider=LLMProvider.OPENAI, 
            model="gpt-4",
            retry_after=1
        )
        
        # First call fails with rate limit, second succeeds
        mock_llm.generate.side_effect = [rate_limit_error, self.mock_response]
        mock_llm.model = LLMModelType.GPT_4
        
        attempts = []
        
        with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
            with patch('asyncio.sleep') as mock_sleep:  # Speed up test
                result = await self.handler._try_retry_strategy(
                    mock_llm, self.mock_request, rate_limit_error, attempts
                )
        
        assert result == self.mock_response
        mock_sleep.assert_called_with(1)  # Should use retry_after value
        assert len(attempts) == 2  # Failed attempt + successful retry
    
    async def test_try_retry_strategy_non_retryable_error(self):
        """Test retry strategy with non-retryable error."""
        mock_llm = AsyncMock(spec=BaseLLM)
        auth_error = LLMAuthenticationError("Invalid API key", provider=LLMProvider.OPENAI, model="gpt-4")
        
        attempts = []
        
        result = await self.handler._try_retry_strategy(
            mock_llm, self.mock_request, auth_error, attempts
        )
        
        assert result is None
        assert len(attempts) == 0  # No retry attempts for auth errors
    
    async def test_try_provider_switch_success(self):
        """Test successful provider switching."""
        attempts = []
        
        with patch('src.llm.fallback_handler.LLMFactory') as mock_factory:
            mock_new_llm = AsyncMock(spec=BaseLLM)
            mock_new_llm.generate.return_value = self.mock_response
            mock_new_llm.model = LLMModelType.CLAUDE_3_SONNET
            mock_factory.create_llm.return_value = mock_new_llm
            
            result = await self.handler._try_provider_switch(self.mock_request, attempts)
        
        assert result == self.mock_response
        assert len(attempts) == 1
        assert attempts[0].strategy == FallbackStrategy.PROVIDER_SWITCH
        assert attempts[0].error is None
    
    async def test_try_provider_switch_all_providers_fail(self):
        """Test provider switching when all providers fail."""
        attempts = []
        
        with patch('src.llm.fallback_handler.LLMFactory') as mock_factory:
            error = LLMError("Provider failed")
            mock_factory.create_llm.side_effect = error
            
            result = await self.handler._try_provider_switch(self.mock_request, attempts)
        
        assert result is None
        # Should try all available providers
        assert len(attempts) >= 1
        assert all(attempt.error is not None for attempt in attempts)
    
    async def test_try_provider_switch_excludes_tried_providers(self):
        """Test that provider switching excludes already tried providers."""
        # Add attempt for OpenAI provider
        attempts = [
            FallbackAttempt(
                strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                error=LLMError("Failed"),
                response=None,
                duration_ms=100
            )
        ]
        
        with patch('src.llm.fallback_handler.LLMFactory') as mock_factory:
            mock_new_llm = AsyncMock(spec=BaseLLM)
            mock_new_llm.generate.return_value = self.mock_response
            mock_new_llm.model = LLMModelType.CLAUDE_3_SONNET
            mock_factory.create_llm.return_value = mock_new_llm
            
            result = await self.handler._try_provider_switch(self.mock_request, attempts)
        
        # Should not try OpenAI again (already in attempts)
        create_calls = mock_factory.create_llm.call_args_list
        providers_tried = [call[0][0] for call in create_calls]
        assert LLMProvider.OPENAI not in providers_tried
    
    async def test_try_model_downgrade_success(self):
        """Test successful model downgrading."""
        mock_llm = AsyncMock(spec=BaseLLM)
        mock_llm.model = LLMModelType.GPT_4_TURBO
        
        attempts = []
        
        with patch('src.llm.fallback_handler.LLMFactory') as mock_factory:
            with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
                mock_downgraded_llm = AsyncMock(spec=BaseLLM)
                mock_downgraded_llm.generate.return_value = self.mock_response
                mock_factory.create_llm.return_value = mock_downgraded_llm
                
                result = await self.handler._try_model_downgrade(mock_llm, self.mock_request, attempts)
        
        assert result == self.mock_response
        assert len(attempts) == 1
        assert attempts[0].strategy == FallbackStrategy.MODEL_DOWNGRADE
        assert attempts[0].model == "gpt-4"  # Downgraded from GPT-4 Turbo to GPT-4
    
    async def test_try_model_downgrade_no_downgrade_available(self):
        """Test model downgrade when no downgrade is available."""
        mock_llm = AsyncMock(spec=BaseLLM)
        mock_llm.model = LLMModelType.GPT_3_5_TURBO  # Already lowest tier
        
        attempts = []
        
        with patch.object(self.handler, '_get_provider_from_llm', return_value=LLMProvider.OPENAI):
            result = await self.handler._try_model_downgrade(mock_llm, self.mock_request, attempts)
        
        assert result is None
        assert len(attempts) == 0
    
    def test_try_static_response_with_purpose(self):
        """Test static response with specific purpose."""
        context = {"purpose": "question_generation"}
        
        result = self.handler._try_static_response(self.mock_request, context)
        
        assert result is not None
        assert "unable to generate questions" in result.content.lower()
        assert result.model == "static"
        assert result.provider == LLMProvider.OPENAI  # Arbitrary
        assert result.usage.estimated_cost == 0.0
    
    def test_try_static_response_default(self):
        """Test static response with default fallback."""
        context = {"purpose": "unknown_purpose"}
        
        result = self.handler._try_static_response(self.mock_request, context)
        
        assert result is not None
        assert "technical difficulties" in result.content.lower()
    
    def test_try_static_response_no_responses_configured(self):
        """Test static response when no responses are configured."""
        # Configure handler with empty static responses
        config = FallbackConfig(static_responses={})
        handler = FallbackHandler(config=config)
        
        result = handler._try_static_response(self.mock_request, {})
        
        assert result is None
    
    async def test_try_cache_lookup_success(self):
        """Test successful cache lookup."""
        cache_fn = AsyncMock(return_value=self.mock_response)
        handler = FallbackHandler(cache_lookup_fn=cache_fn)
        
        result = await handler._try_cache_lookup(self.mock_request, {})
        
        assert result == self.mock_response
        cache_fn.assert_called_once_with(self.mock_request, {})
    
    async def test_try_cache_lookup_no_function(self):
        """Test cache lookup when no function is provided."""
        result = await self.handler._try_cache_lookup(self.mock_request, {})
        
        assert result is None
    
    async def test_try_cache_lookup_failure(self):
        """Test cache lookup when function raises exception."""
        cache_fn = AsyncMock(side_effect=Exception("Cache error"))
        handler = FallbackHandler(cache_lookup_fn=cache_fn)
        
        result = await handler._try_cache_lookup(self.mock_request, {})
        
        assert result is None
    
    async def test_update_provider_health_if_needed(self):
        """Test updating provider health status."""
        # Set last check to old time to trigger update
        self.handler._last_health_check = 0
        
        mock_health_status = {
            LLMProvider.OPENAI: True,
            LLMProvider.ANTHROPIC: False,
            LLMProvider.GEMINI: True
        }
        
        with patch('src.llm.fallback_handler.LLMFactory') as mock_factory:
            mock_factory.health_check_all.return_value = mock_health_status
            
            await self.handler._update_provider_health_if_needed()
        
        assert self.handler._provider_health == mock_health_status
        assert self.handler._last_health_check > 0
    
    async def test_update_provider_health_skip_if_recent(self):
        """Test skipping health update if recently checked."""
        # Set recent check time
        self.handler._last_health_check = time.time()
        
        with patch('src.llm.fallback_handler.LLMFactory') as mock_factory:
            await self.handler._update_provider_health_if_needed()
        
        # Should not call health check
        mock_factory.health_check_all.assert_not_called()
    
    def test_get_provider_from_llm_openai(self):
        """Test identifying OpenAI provider from LLM instance."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "OpenAILLM"
        
        provider = self.handler._get_provider_from_llm(mock_llm)
        
        assert provider == LLMProvider.OPENAI
    
    def test_get_provider_from_llm_anthropic(self):
        """Test identifying Anthropic provider from LLM instance."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "AnthropicLLM"
        
        provider = self.handler._get_provider_from_llm(mock_llm)
        
        assert provider == LLMProvider.ANTHROPIC
    
    def test_get_provider_from_llm_gemini(self):
        """Test identifying Gemini provider from LLM instance."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "GeminiLLM"
        
        provider = self.handler._get_provider_from_llm(mock_llm)
        
        assert provider == LLMProvider.GEMINI
    
    def test_get_provider_from_llm_unknown(self):
        """Test handling unknown provider."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "UnknownLLM"
        
        provider = self.handler._get_provider_from_llm(mock_llm)
        
        assert provider is None
    
    def test_get_statistics_empty(self):
        """Test getting statistics with no attempts."""
        stats = self.handler.get_statistics()
        
        assert stats["total_attempts"] == 0
    
    def test_get_statistics_with_attempts(self):
        """Test getting statistics with attempt history."""
        # Add some mock attempts
        self.handler._attempt_history = [
            FallbackAttempt(
                strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                error=None,
                response=self.mock_response,
                duration_ms=100
            ),
            FallbackAttempt(
                strategy=FallbackStrategy.PROVIDER_SWITCH,
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-sonnet",
                error=LLMError("Failed"),
                response=None,
                duration_ms=150
            ),
            FallbackAttempt(
                strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                error=None,
                response=self.mock_response,
                duration_ms=120
            )
        ]
        
        stats = self.handler.get_statistics()
        
        assert stats["total_attempts"] == 3
        assert stats["successful_attempts"] == 2
        assert stats["overall_success_rate"] == 2/3
        
        # Check strategy statistics
        retry_stats = stats["strategy_statistics"][FallbackStrategy.RETRY_EXPONENTIAL.value]
        assert retry_stats["attempts"] == 2
        assert retry_stats["successes"] == 2
        assert retry_stats["success_rate"] == 1.0
        
        switch_stats = stats["strategy_statistics"][FallbackStrategy.PROVIDER_SWITCH.value]
        assert switch_stats["attempts"] == 1
        assert switch_stats["successes"] == 0
        assert switch_stats["success_rate"] == 0.0
    
    def test_clear_history(self):
        """Test clearing attempt history."""
        # Add some attempts
        self.handler._attempt_history.append(
            FallbackAttempt(
                strategy=FallbackStrategy.RETRY_EXPONENTIAL,
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                error=None,
                response=self.mock_response,
                duration_ms=100
            )
        )
        
        assert len(self.handler._attempt_history) == 1
        
        self.handler.clear_history()
        
        assert len(self.handler._attempt_history) == 0