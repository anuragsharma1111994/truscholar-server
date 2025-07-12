"""Unit tests for LLM Factory in TruScholar.

This module tests the LLM factory functionality including provider registration,
instance creation, configuration validation, and caching.
"""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.llm.llm_factory import LLMFactory, create_openai_llm, create_anthropic_llm, create_gemini_llm
from src.llm.base_llm import LLMProvider, LLMModelType, BaseLLM
from src.llm.openai_llm import OpenAILLM
from src.llm.anthropic_llm import AnthropicLLM
from src.llm.gemini_llm import GeminiLLM


class TestLLMFactory:
    """Test cases for LLM Factory."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Clear factory cache before each test
        LLMFactory.clear_cache()
        
        # Mock environment variables
        self.env_vars = {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key", 
            "GOOGLE_API_KEY": "test-google-key"
        }
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = LLMFactory.get_available_providers()
        
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.ANTHROPIC in providers
        assert LLMProvider.GEMINI in providers
        assert len(providers) == 3
    
    def test_get_supported_models_openai(self):
        """Test getting supported models for OpenAI."""
        models = LLMFactory.get_supported_models(LLMProvider.OPENAI)
        
        assert LLMModelType.GPT_4 in models
        assert LLMModelType.GPT_4_TURBO in models
        assert LLMModelType.GPT_3_5_TURBO in models
        assert len(models) == 3
    
    def test_get_supported_models_anthropic(self):
        """Test getting supported models for Anthropic."""
        models = LLMFactory.get_supported_models(LLMProvider.ANTHROPIC)
        
        assert LLMModelType.CLAUDE_3_OPUS in models
        assert LLMModelType.CLAUDE_3_SONNET in models
        assert LLMModelType.CLAUDE_3_HAIKU in models
        assert len(models) == 3
    
    def test_get_supported_models_gemini(self):
        """Test getting supported models for Gemini."""
        models = LLMFactory.get_supported_models(LLMProvider.GEMINI)
        
        assert LLMModelType.GEMINI_PRO in models
        assert LLMModelType.GEMINI_PRO_VISION in models
        assert len(models) == 2
    
    def test_get_supported_models_invalid_provider(self):
        """Test getting supported models for invalid provider."""
        # Create a mock provider that doesn't exist
        with patch('src.llm.llm_factory.LLMProvider') as mock_provider:
            mock_provider.INVALID = "invalid"
            models = LLMFactory.get_supported_models(mock_provider.INVALID)
            assert models == []
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.OpenAILLM')
    def test_create_llm_openai_success(self, mock_openai_class):
        """Test successful OpenAI LLM creation."""
        mock_instance = Mock(spec=OpenAILLM)
        mock_openai_class.return_value = mock_instance
        
        llm = LLMFactory.create_llm(
            provider=LLMProvider.OPENAI,
            model=LLMModelType.GPT_4
        )
        
        assert llm == mock_instance
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            model=LLMModelType.GPT_4
        )
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.AnthropicLLM')
    def test_create_llm_anthropic_success(self, mock_anthropic_class):
        """Test successful Anthropic LLM creation."""
        mock_instance = Mock(spec=AnthropicLLM)
        mock_anthropic_class.return_value = mock_instance
        
        llm = LLMFactory.create_llm(
            provider=LLMProvider.ANTHROPIC,
            model=LLMModelType.CLAUDE_3_SONNET
        )
        
        assert llm == mock_instance
        mock_anthropic_class.assert_called_once_with(
            api_key="test-key",
            model=LLMModelType.CLAUDE_3_SONNET
        )
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.GeminiLLM')
    def test_create_llm_gemini_success(self, mock_gemini_class):
        """Test successful Gemini LLM creation."""
        mock_instance = Mock(spec=GeminiLLM)
        mock_gemini_class.return_value = mock_instance
        
        llm = LLMFactory.create_llm(
            provider=LLMProvider.GEMINI,
            model=LLMModelType.GEMINI_PRO
        )
        
        assert llm == mock_instance
        mock_gemini_class.assert_called_once_with(
            api_key="test-key",
            model=LLMModelType.GEMINI_PRO
        )
    
    def test_create_llm_unsupported_provider(self):
        """Test creating LLM with unsupported provider."""
        with patch('src.llm.llm_factory.LLMProvider') as mock_provider:
            mock_provider.INVALID = "invalid"
            
            with pytest.raises(ValueError, match="Unsupported provider"):
                LLMFactory.create_llm(provider=mock_provider.INVALID)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_llm_missing_api_key(self):
        """Test creating LLM without API key."""
        with pytest.raises(ValueError, match="API key not provided"):
            LLMFactory.create_llm(provider=LLMProvider.OPENAI)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.OpenAILLM')
    def test_create_llm_with_custom_api_key(self, mock_openai_class):
        """Test creating LLM with custom API key."""
        mock_instance = Mock(spec=OpenAILLM)
        mock_openai_class.return_value = mock_instance
        
        custom_key = "custom-api-key"
        llm = LLMFactory.create_llm(
            provider=LLMProvider.OPENAI,
            api_key=custom_key
        )
        
        assert llm == mock_instance
        mock_openai_class.assert_called_once_with(
            api_key=custom_key,
            model=LLMModelType.GPT_4  # Default model
        )
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.OpenAILLM')
    def test_create_llm_default_model(self, mock_openai_class):
        """Test creating LLM uses default model when not specified."""
        mock_instance = Mock(spec=OpenAILLM)
        mock_openai_class.return_value = mock_instance
        
        llm = LLMFactory.create_llm(provider=LLMProvider.OPENAI)
        
        mock_openai_class.assert_called_once_with(
            api_key="test-key",
            model=LLMModelType.GPT_4  # Default for OpenAI
        )
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.OpenAILLM')
    def test_create_llm_caching(self, mock_openai_class):
        """Test that LLM instances are cached."""
        mock_instance = Mock(spec=OpenAILLM)
        mock_openai_class.return_value = mock_instance
        
        # Create same LLM twice
        llm1 = LLMFactory.create_llm(
            provider=LLMProvider.OPENAI,
            model=LLMModelType.GPT_4
        )
        llm2 = LLMFactory.create_llm(
            provider=LLMProvider.OPENAI,
            model=LLMModelType.GPT_4
        )
        
        # Should return same instance from cache
        assert llm1 == llm2
        # Should only create instance once
        assert mock_openai_class.call_count == 1
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.llm.llm_factory.OpenAILLM')
    def test_create_llm_creation_failure(self, mock_openai_class):
        """Test handling LLM creation failure."""
        mock_openai_class.side_effect = Exception("Creation failed")
        
        with pytest.raises(ValueError, match="Failed to create LLM instance"):
            LLMFactory.create_llm(provider=LLMProvider.OPENAI)
    
    def test_create_from_config_valid(self):
        """Test creating LLM from valid configuration."""
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
            "timeout": 60.0,
            "max_retries": 5
        }
        
        with patch.object(LLMFactory, 'create_llm') as mock_create:
            mock_instance = Mock(spec=BaseLLM)
            mock_create.return_value = mock_instance
            
            llm = LLMFactory.create_from_config(config)
            
            mock_create.assert_called_once_with(
                provider=LLMProvider.OPENAI,
                model=LLMModelType.GPT_4,
                api_key="test-key",
                timeout=60.0,
                max_retries=5
            )
    
    def test_create_from_config_missing_provider(self):
        """Test creating LLM from config without provider."""
        config = {"model": "gpt-4"}
        
        with pytest.raises(ValueError, match="Provider not specified"):
            LLMFactory.create_from_config(config)
    
    def test_create_from_config_invalid_provider(self):
        """Test creating LLM from config with invalid provider."""
        config = {"provider": "invalid-provider"}
        
        with pytest.raises(ValueError, match="Invalid provider"):
            LLMFactory.create_from_config(config)
    
    def test_create_from_config_invalid_model(self):
        """Test creating LLM from config with invalid model."""
        config = {
            "provider": "openai",
            "model": "invalid-model"
        }
        
        with pytest.raises(ValueError, match="Invalid model"):
            LLMFactory.create_from_config(config)
    
    @patch('src.llm.llm_factory.get_settings')
    def test_create_from_settings(self, mock_get_settings):
        """Test creating LLM from application settings."""
        mock_settings = Mock()
        mock_settings.DEFAULT_LLM_PROVIDER = "anthropic"
        mock_settings.ANTHROPIC_DEFAULT_MODEL = "claude-3-sonnet-20240229"
        mock_settings.LLM_TIMEOUT = 45.0
        mock_settings.LLM_MAX_RETRIES = 4
        mock_get_settings.return_value = mock_settings
        
        with patch.object(LLMFactory, 'create_llm') as mock_create:
            mock_instance = Mock(spec=BaseLLM)
            mock_create.return_value = mock_instance
            
            llm = LLMFactory.create_from_settings()
            
            mock_create.assert_called_once_with(
                provider=LLMProvider.ANTHROPIC,
                model=LLMModelType.CLAUDE_3_SONNET,
                timeout=45.0,
                max_retries=4
            )
    
    @patch('src.llm.llm_factory.get_settings')
    def test_create_from_settings_with_provider_override(self, mock_get_settings):
        """Test creating LLM from settings with provider override."""
        mock_settings = Mock()
        mock_settings.LLM_TIMEOUT = 30.0
        mock_settings.LLM_MAX_RETRIES = 3
        mock_get_settings.return_value = mock_settings
        
        with patch.object(LLMFactory, 'create_llm') as mock_create:
            mock_instance = Mock(spec=BaseLLM)
            mock_create.return_value = mock_instance
            
            llm = LLMFactory.create_from_settings(provider=LLMProvider.GEMINI)
            
            mock_create.assert_called_once_with(
                provider=LLMProvider.GEMINI,
                model=None,
                timeout=30.0,
                max_retries=3
            )
    
    def test_validate_configuration_valid(self):
        """Test validating valid configuration."""
        with patch.object(LLMFactory, '_get_api_key_from_env', return_value="test-key"):
            is_valid = LLMFactory.validate_configuration(
                provider=LLMProvider.OPENAI,
                model=LLMModelType.GPT_4
            )
            assert is_valid is True
    
    def test_validate_configuration_unsupported_provider(self):
        """Test validating configuration with unsupported provider."""
        with patch('src.llm.llm_factory.LLMProvider') as mock_provider:
            mock_provider.INVALID = "invalid"
            
            is_valid = LLMFactory.validate_configuration(
                provider=mock_provider.INVALID,
                model=LLMModelType.GPT_4
            )
            assert is_valid is False
    
    def test_validate_configuration_unsupported_model(self):
        """Test validating configuration with unsupported model."""
        with patch.object(LLMFactory, 'get_supported_models', return_value=[]):
            is_valid = LLMFactory.validate_configuration(
                provider=LLMProvider.OPENAI,
                model=LLMModelType.GPT_4
            )
            assert is_valid is False
    
    def test_validate_configuration_missing_api_key(self):
        """Test validating configuration with missing API key."""
        with patch.object(LLMFactory, '_get_api_key_from_env', return_value=None):
            is_valid = LLMFactory.validate_configuration(
                provider=LLMProvider.OPENAI,
                model=LLMModelType.GPT_4
            )
            assert is_valid is False
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai", "ANTHROPIC_API_KEY": "test-anthropic"})
    async def test_health_check_all(self):
        """Test health check for all providers."""
        with patch.object(LLMFactory, 'create_llm') as mock_create:
            # Mock LLM instances
            mock_openai = AsyncMock()
            mock_openai.health_check.return_value = True
            
            mock_anthropic = AsyncMock()
            mock_anthropic.health_check.return_value = False
            
            # Configure mock to return different instances based on provider
            def create_side_effect(provider, **kwargs):
                if provider == LLMProvider.OPENAI:
                    return mock_openai
                elif provider == LLMProvider.ANTHROPIC:
                    return mock_anthropic
                else:
                    raise Exception("No API key")
            
            mock_create.side_effect = create_side_effect
            
            results = await LLMFactory.health_check_all()
            
            assert results[LLMProvider.OPENAI] is True
            assert results[LLMProvider.ANTHROPIC] is False
            assert results[LLMProvider.GEMINI] is False  # No API key
    
    def test_clear_cache(self):
        """Test clearing the instance cache."""
        # Add something to cache first
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('src.llm.llm_factory.OpenAILLM'):
                LLMFactory.create_llm(provider=LLMProvider.OPENAI)
                
                stats_before = LLMFactory.get_cache_stats()
                assert stats_before["cached_instances"] > 0
                
                LLMFactory.clear_cache()
                
                stats_after = LLMFactory.get_cache_stats()
                assert stats_after["cached_instances"] == 0
    
    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        stats = LLMFactory.get_cache_stats()
        
        assert "cached_instances" in stats
        assert "cache_keys" in stats
        assert isinstance(stats["cached_instances"], int)
        assert isinstance(stats["cache_keys"], list)
    
    def test_register_provider(self):
        """Test registering a new provider."""
        # Create mock provider and implementation
        mock_provider = Mock()
        mock_implementation = Mock()
        mock_model = Mock()
        
        LLMFactory.register_provider(
            provider=mock_provider,
            implementation=mock_implementation,
            default_model=mock_model
        )
        
        # Verify provider was registered
        assert mock_provider in LLMFactory._providers
        assert LLMFactory._providers[mock_provider] == mock_implementation
        assert LLMFactory._default_models[mock_provider] == mock_model


class TestConvenienceFunctions:
    """Test convenience functions for creating LLMs."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(LLMFactory, 'create_llm')
    def test_create_openai_llm(self, mock_create):
        """Test convenience function for creating OpenAI LLM."""
        mock_instance = Mock(spec=OpenAILLM)
        mock_create.return_value = mock_instance
        
        llm = create_openai_llm(
            model=LLMModelType.GPT_4_TURBO,
            api_key="custom-key",
            timeout=60
        )
        
        mock_create.assert_called_once_with(
            provider=LLMProvider.OPENAI,
            model=LLMModelType.GPT_4_TURBO,
            api_key="custom-key",
            timeout=60
        )
        assert llm == mock_instance
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch.object(LLMFactory, 'create_llm')
    def test_create_anthropic_llm(self, mock_create):
        """Test convenience function for creating Anthropic LLM."""
        mock_instance = Mock(spec=AnthropicLLM)
        mock_create.return_value = mock_instance
        
        llm = create_anthropic_llm(
            model=LLMModelType.CLAUDE_3_OPUS,
            api_key="custom-key",
            timeout=45
        )
        
        mock_create.assert_called_once_with(
            provider=LLMProvider.ANTHROPIC,
            model=LLMModelType.CLAUDE_3_OPUS,
            api_key="custom-key",
            timeout=45
        )
        assert llm == mock_instance
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @patch.object(LLMFactory, 'create_llm')
    def test_create_gemini_llm(self, mock_create):
        """Test convenience function for creating Gemini LLM."""
        mock_instance = Mock(spec=GeminiLLM)
        mock_create.return_value = mock_instance
        
        llm = create_gemini_llm(
            model=LLMModelType.GEMINI_PRO_VISION,
            api_key="custom-key",
            timeout=30
        )
        
        mock_create.assert_called_once_with(
            provider=LLMProvider.GEMINI,
            model=LLMModelType.GEMINI_PRO_VISION,
            api_key="custom-key",
            timeout=30
        )
        assert llm == mock_instance