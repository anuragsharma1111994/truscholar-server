"""LLM integration module for TruScholar.

This module provides interfaces and implementations for various Large Language Model
providers, including OpenAI, Anthropic, and Google Gemini. It includes fallback
handling, prompt management, and unified interfaces for AI-powered features.
"""

from .base_llm import BaseLLM, LLMRequest, LLMResponse, LLMUsage
from .llm_factory import LLMFactory, LLMProvider
from .fallback_handler import FallbackHandler
from .prompt_manager import PromptManager, PromptTemplate
from .openai_llm import OpenAILLM
from .anthropic_llm import AnthropicLLM
from .gemini_llm import GeminiLLM

# Export main components
__all__ = [
    # Base classes and types
    "BaseLLM",
    "LLMRequest", 
    "LLMResponse",
    "LLMUsage",
    
    # Factory and management
    "LLMFactory",
    "LLMProvider",
    "FallbackHandler",
    "PromptManager",
    "PromptTemplate",
    
    # Provider implementations
    "OpenAILLM",
    "AnthropicLLM", 
    "GeminiLLM",
]

# Version info
__version__ = "1.0.0"