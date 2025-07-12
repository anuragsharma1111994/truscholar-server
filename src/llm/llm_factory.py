"""LLM Factory for creating and managing LLM instances.

This module provides a factory pattern for creating LLM instances
and managing their configuration and lifecycle.
"""

from typing import Dict, Any, Optional, Type, List
from enum import Enum
import os

from .base_llm import BaseLLM, LLMProvider, LLMModelType
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .gemini_llm import GeminiLLM
from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory for creating and managing LLM instances."""
    
    # Registry of LLM implementations
    _providers: Dict[LLMProvider, Type[BaseLLM]] = {
        LLMProvider.OPENAI: OpenAILLM,
        LLMProvider.ANTHROPIC: AnthropicLLM,
        LLMProvider.GEMINI: GeminiLLM,
    }
    
    # Default models for each provider
    _default_models: Dict[LLMProvider, LLMModelType] = {
        LLMProvider.OPENAI: LLMModelType.GPT_4,
        LLMProvider.ANTHROPIC: LLMModelType.CLAUDE_3_SONNET,
        LLMProvider.GEMINI: LLMModelType.GEMINI_PRO,
    }
    
    # Cache for LLM instances
    _instances: Dict[str, BaseLLM] = {}
    
    @classmethod
    def create_llm(
        cls,
        provider: LLMProvider,
        model: Optional[LLMModelType] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM instance.
        
        Args:
            provider: LLM provider to use
            model: Model to use (uses default if not specified)
            api_key: API key (uses environment variable if not specified)
            **kwargs: Additional configuration for the LLM
            
        Returns:
            BaseLLM: Configured LLM instance
            
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        # Check if provider is supported
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Get model (use default if not specified)
        if model is None:
            model = cls._default_models[provider]
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = cls._get_api_key_from_env(provider)
        
        if not api_key:
            raise ValueError(f"API key not provided for {provider}")
        
        # Create cache key
        cache_key = f"{provider.value}:{model.value}:{hash(api_key)}"
        
        # Check cache first
        if cache_key in cls._instances:
            logger.debug(f"Returning cached LLM instance for {provider}:{model}")
            return cls._instances[cache_key]
        
        # Get LLM class
        llm_class = cls._providers[provider]
        
        # Create instance
        try:
            instance = llm_class(
                api_key=api_key,
                model=model,
                **kwargs
            )
            
            # Cache the instance
            cls._instances[cache_key] = instance
            
            logger.info(f"Created new LLM instance: {provider}:{model}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create LLM instance for {provider}: {e}")
            raise ValueError(f"Failed to create LLM instance: {e}")
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseLLM:
        """Create LLM from configuration dictionary.
        
        Args:
            config: Configuration dictionary with provider, model, etc.
            
        Returns:
            BaseLLM: Configured LLM instance
        """
        provider_str = config.get("provider")
        if not provider_str:
            raise ValueError("Provider not specified in config")
        
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise ValueError(f"Invalid provider: {provider_str}")
        
        model_str = config.get("model")
        model = None
        if model_str:
            try:
                model = LLMModelType(model_str)
            except ValueError:
                raise ValueError(f"Invalid model: {model_str}")
        
        # Extract other config parameters
        api_key = config.get("api_key")
        timeout = config.get("timeout", 30.0)
        max_retries = config.get("max_retries", 3)
        
        # Provider-specific configurations
        provider_config = config.get("provider_config", {})
        
        return cls.create_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **provider_config
        )
    
    @classmethod
    def create_from_settings(cls, provider: Optional[LLMProvider] = None) -> BaseLLM:
        """Create LLM from application settings.
        
        Args:
            provider: Provider to use (uses default from settings if not specified)
            
        Returns:
            BaseLLM: Configured LLM instance
        """
        settings = get_settings()
        
        # Use specified provider or default from settings
        if provider is None:
            provider_str = getattr(settings, "DEFAULT_LLM_PROVIDER", "openai")
            provider = LLMProvider(provider_str)
        
        # Get model from settings
        model_str = getattr(settings, f"{provider.value.upper()}_DEFAULT_MODEL", None)
        model = None
        if model_str:
            try:
                model = LLMModelType(model_str)
            except ValueError:
                logger.warning(f"Invalid model in settings: {model_str}")
        
        # Get timeout and retries from settings
        timeout = getattr(settings, "LLM_TIMEOUT", 30.0)
        max_retries = getattr(settings, "LLM_MAX_RETRIES", 3)
        
        return cls.create_llm(
            provider=provider,
            model=model,
            timeout=timeout,
            max_retries=max_retries
        )
    
    @classmethod
    def get_available_providers(cls) -> List[LLMProvider]:
        """Get list of available LLM providers.
        
        Returns:
            List[LLMProvider]: Available providers
        """
        return list(cls._providers.keys())
    
    @classmethod
    def get_supported_models(cls, provider: LLMProvider) -> List[LLMModelType]:
        """Get supported models for a provider.
        
        Args:
            provider: LLM provider
            
        Returns:
            List[LLMModelType]: Supported models
        """
        if provider == LLMProvider.OPENAI:
            return [
                LLMModelType.GPT_4,
                LLMModelType.GPT_4_TURBO,
                LLMModelType.GPT_3_5_TURBO,
            ]
        elif provider == LLMProvider.ANTHROPIC:
            return [
                LLMModelType.CLAUDE_3_OPUS,
                LLMModelType.CLAUDE_3_SONNET,
                LLMModelType.CLAUDE_3_HAIKU,
            ]
        elif provider == LLMProvider.GEMINI:
            return [
                LLMModelType.GEMINI_PRO,
                LLMModelType.GEMINI_PRO_VISION,
            ]
        else:
            return []
    
    @classmethod
    def validate_configuration(
        cls,
        provider: LLMProvider,
        model: LLMModelType,
        api_key: Optional[str] = None
    ) -> bool:
        """Validate LLM configuration.
        
        Args:
            provider: LLM provider
            model: Model to validate
            api_key: API key to validate
            
        Returns:
            bool: True if configuration is valid
        """
        # Check if provider is supported
        if provider not in cls._providers:
            logger.error(f"Unsupported provider: {provider}")
            return False
        
        # Check if model is supported by provider
        supported_models = cls.get_supported_models(provider)
        if model not in supported_models:
            logger.error(f"Model {model} not supported by {provider}")
            return False
        
        # Check API key
        if api_key is None:
            api_key = cls._get_api_key_from_env(provider)
        
        if not api_key:
            logger.error(f"No API key found for {provider}")
            return False
        
        return True
    
    @classmethod
    async def health_check_all(cls) -> Dict[LLMProvider, bool]:
        """Perform health check on all available providers.
        
        Returns:
            Dict[LLMProvider, bool]: Health status for each provider
        """
        results = {}
        
        for provider in cls._providers:
            try:
                # Get API key
                api_key = cls._get_api_key_from_env(provider)
                if not api_key:
                    results[provider] = False
                    continue
                
                # Create LLM instance
                llm = cls.create_llm(provider)
                
                # Perform health check
                is_healthy = await llm.health_check()
                results[provider] = is_healthy
                
            except Exception as e:
                logger.warning(f"Health check failed for {provider}: {e}")
                results[provider] = False
        
        return results
    
    @classmethod
    def clear_cache(cls):
        """Clear the instance cache."""
        cls._instances.clear()
        logger.info("LLM instance cache cleared")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        return {
            "cached_instances": len(cls._instances),
            "cache_keys": list(cls._instances.keys())
        }
    
    @classmethod
    def _get_api_key_from_env(cls, provider: LLMProvider) -> Optional[str]:
        """Get API key from environment variables.
        
        Args:
            provider: LLM provider
            
        Returns:
            Optional[str]: API key if found
        """
        env_var_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.GEMINI: "GOOGLE_API_KEY",
        }
        
        env_var = env_var_map.get(provider)
        if not env_var:
            return None
        
        return os.getenv(env_var)
    
    @classmethod
    def register_provider(
        cls,
        provider: LLMProvider,
        implementation: Type[BaseLLM],
        default_model: LLMModelType
    ):
        """Register a new LLM provider implementation.
        
        Args:
            provider: Provider enum value
            implementation: LLM implementation class
            default_model: Default model for this provider
        """
        cls._providers[provider] = implementation
        cls._default_models[provider] = default_model
        
        logger.info(f"Registered new LLM provider: {provider}")


# Convenience functions

def create_openai_llm(
    model: LLMModelType = LLMModelType.GPT_4,
    api_key: Optional[str] = None,
    **kwargs
) -> OpenAILLM:
    """Create OpenAI LLM instance.
    
    Args:
        model: OpenAI model to use
        api_key: API key (uses environment if not provided)
        **kwargs: Additional configuration
        
    Returns:
        OpenAILLM: Configured OpenAI LLM
    """
    return LLMFactory.create_llm(
        provider=LLMProvider.OPENAI,
        model=model,
        api_key=api_key,
        **kwargs
    )


def create_anthropic_llm(
    model: LLMModelType = LLMModelType.CLAUDE_3_SONNET,
    api_key: Optional[str] = None,
    **kwargs
) -> AnthropicLLM:
    """Create Anthropic LLM instance.
    
    Args:
        model: Anthropic model to use
        api_key: API key (uses environment if not provided)
        **kwargs: Additional configuration
        
    Returns:
        AnthropicLLM: Configured Anthropic LLM
    """
    return LLMFactory.create_llm(
        provider=LLMProvider.ANTHROPIC,
        model=model,
        api_key=api_key,
        **kwargs
    )


def create_gemini_llm(
    model: LLMModelType = LLMModelType.GEMINI_PRO,
    api_key: Optional[str] = None,
    **kwargs
) -> GeminiLLM:
    """Create Gemini LLM instance.
    
    Args:
        model: Gemini model to use
        api_key: API key (uses environment if not provided)
        **kwargs: Additional configuration
        
    Returns:
        GeminiLLM: Configured Gemini LLM
    """
    return LLMFactory.create_llm(
        provider=LLMProvider.GEMINI,
        model=model,
        api_key=api_key,
        **kwargs
    )


# Export factory and convenience functions
__all__ = [
    "LLMFactory",
    "create_openai_llm",
    "create_anthropic_llm", 
    "create_gemini_llm"
]