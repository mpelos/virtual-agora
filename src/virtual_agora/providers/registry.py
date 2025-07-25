"""Provider registry for Virtual Agora.

This module maintains a registry of supported LLM providers and their
available models. It provides validation and metadata for provider
configurations.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from virtual_agora.providers.config import ProviderType


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    display_name: str
    max_tokens: int
    supports_streaming: bool = True
    supports_functions: bool = True
    notes: Optional[str] = None


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""
    provider_type: ProviderType
    display_name: str
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    requires_api_key: bool = True
    api_key_env_var: Optional[str] = None
    
    def add_model(self, model: ModelInfo) -> None:
        """Add a model to the provider."""
        self.models[model.name] = model
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def is_model_supported(self, model_name: str) -> bool:
        """Check if a model is supported."""
        return model_name in self.models


class ProviderRegistry:
    """Registry of LLM providers and their models."""
    
    def __init__(self):
        """Initialize the registry with known providers."""
        self._providers: Dict[ProviderType, ProviderInfo] = {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize the registry with known providers and models."""
        
        # Google Gemini
        google = ProviderInfo(
            provider_type=ProviderType.GOOGLE,
            display_name="Google Gemini",
            api_key_env_var="GOOGLE_API_KEY"
        )
        google.add_model(ModelInfo(
            name="gemini-1.5-pro",
            display_name="Gemini 1.5 Pro",
            max_tokens=8192,
            notes="Latest Gemini model with improved capabilities"
        ))
        google.add_model(ModelInfo(
            name="gemini-1.5-flash",
            display_name="Gemini 1.5 Flash",
            max_tokens=8192,
            notes="Faster, more efficient Gemini model"
        ))
        google.add_model(ModelInfo(
            name="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            max_tokens=8192,
            notes="Latest flash model with improved performance"
        ))
        self._providers[ProviderType.GOOGLE] = google
        
        # OpenAI
        openai = ProviderInfo(
            provider_type=ProviderType.OPENAI,
            display_name="OpenAI",
            api_key_env_var="OPENAI_API_KEY"
        )
        openai.add_model(ModelInfo(
            name="gpt-4o",
            display_name="GPT-4 Optimized",
            max_tokens=4096,
            notes="Optimized GPT-4 with improved performance"
        ))
        openai.add_model(ModelInfo(
            name="gpt-4o-mini",
            display_name="GPT-4 Optimized Mini",
            max_tokens=4096,
            notes="Smaller, faster GPT-4 variant"
        ))
        openai.add_model(ModelInfo(
            name="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            max_tokens=4096,
            notes="Latest GPT-4 with vision capabilities"
        ))
        openai.add_model(ModelInfo(
            name="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            max_tokens=4096,
            notes="Fast and efficient for most tasks"
        ))
        self._providers[ProviderType.OPENAI] = openai
        
        # Anthropic
        anthropic = ProviderInfo(
            provider_type=ProviderType.ANTHROPIC,
            display_name="Anthropic",
            api_key_env_var="ANTHROPIC_API_KEY"
        )
        anthropic.add_model(ModelInfo(
            name="claude-3-opus-20240229",
            display_name="Claude 3 Opus",
            max_tokens=4096,
            notes="Most capable Claude model"
        ))
        anthropic.add_model(ModelInfo(
            name="claude-3-sonnet-20240229",
            display_name="Claude 3 Sonnet",
            max_tokens=4096,
            notes="Balanced performance and cost"
        ))
        anthropic.add_model(ModelInfo(
            name="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            max_tokens=4096,
            notes="Fast and efficient Claude model"
        ))
        anthropic.add_model(ModelInfo(
            name="claude-3-5-sonnet-latest",
            display_name="Claude 3.5 Sonnet",
            max_tokens=8192,
            notes="Latest Claude model with enhanced capabilities"
        ))
        self._providers[ProviderType.ANTHROPIC] = anthropic
        
        # Grok (placeholder - actual models TBD)
        grok = ProviderInfo(
            provider_type=ProviderType.GROK,
            display_name="Grok",
            api_key_env_var="GROK_API_KEY"
        )
        grok.add_model(ModelInfo(
            name="grok-1",
            display_name="Grok 1",
            max_tokens=4096,
            notes="Grok's primary model",
            supports_functions=False  # Unknown, being conservative
        ))
        self._providers[ProviderType.GROK] = grok
    
    def get_provider(self, provider_type: ProviderType) -> Optional[ProviderInfo]:
        """Get provider information."""
        return self._providers.get(provider_type)
    
    def list_providers(self) -> List[ProviderType]:
        """List all registered providers."""
        return list(self._providers.keys())
    
    def is_provider_supported(self, provider_type: ProviderType) -> bool:
        """Check if a provider is supported."""
        return provider_type in self._providers
    
    def get_model_info(
        self, 
        provider_type: ProviderType, 
        model_name: str
    ) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        provider = self.get_provider(provider_type)
        if provider:
            return provider.get_model(model_name)
        return None
    
    def is_model_supported(
        self, 
        provider_type: ProviderType, 
        model_name: str
    ) -> bool:
        """Check if a model is supported by a provider."""
        provider = self.get_provider(provider_type)
        if provider:
            return provider.is_model_supported(model_name)
        return False
    
    def list_models_for_provider(
        self, 
        provider_type: ProviderType
    ) -> List[str]:
        """List all models for a provider."""
        provider = self.get_provider(provider_type)
        if provider:
            return provider.list_models()
        return []
    
    def get_api_key_env_var(self, provider_type: ProviderType) -> Optional[str]:
        """Get the environment variable name for a provider's API key."""
        provider = self.get_provider(provider_type)
        if provider:
            return provider.api_key_env_var
        return None
    
    def get_all_models(self) -> Dict[str, List[str]]:
        """Get all models grouped by provider."""
        result = {}
        for provider_type in self._providers:
            provider = self._providers[provider_type]
            result[provider.display_name] = [
                f"{model.display_name} ({model.name})"
                for model in provider.models.values()
            ]
        return result
    
    def validate_model_config(
        self, 
        provider_type: ProviderType, 
        model_name: str
    ) -> tuple[bool, Optional[str]]:
        """Validate a provider and model combination.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.is_provider_supported(provider_type):
            return False, f"Provider '{provider_type}' is not supported"
        
        if not self.is_model_supported(provider_type, model_name):
            available_models = self.list_models_for_provider(provider_type)
            return False, (
                f"Model '{model_name}' is not supported for provider '{provider_type}'. "
                f"Available models: {', '.join(available_models)}"
            )
        
        return True, None


# Global registry instance
registry = ProviderRegistry()