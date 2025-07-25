"""LLM provider integrations for Virtual Agora.

This module provides factory functions and configuration for creating
LangChain chat model instances for various providers (Google, OpenAI, 
Anthropic, Grok) using LangChain's init_chat_model pattern with fallback support.
"""

from typing import TYPE_CHECKING

# Public API imports
from .config import (
    ProviderConfig,
    ProviderType,
    GoogleProviderConfig,
    OpenAIProviderConfig,
    AnthropicProviderConfig,
    GrokProviderConfig,
    create_provider_config,
)
from .factory import ProviderFactory, create_provider, create_provider_with_fallbacks
from .registry import registry, ProviderInfo, ModelInfo

if TYPE_CHECKING:
    # Type imports for better IDE support
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    # Configuration
    "ProviderConfig",
    "ProviderType",
    "GoogleProviderConfig",
    "OpenAIProviderConfig",
    "AnthropicProviderConfig",
    "GrokProviderConfig",
    "create_provider_config",
    # Factory
    "ProviderFactory",
    "create_provider",
    "create_provider_with_fallbacks",
    # Registry
    "registry",
    "ProviderInfo",
    "ModelInfo",
]