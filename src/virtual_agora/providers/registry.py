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
    context_window: int  # Total context window size
    output_tokens: int  # Maximum output tokens
    supports_streaming: bool = True
    supports_functions: bool = True  # Deprecated, use supports_tools
    supports_tools: bool = True  # Tool/function calling capability
    supports_vision: bool = False  # Multimodal image support
    supports_audio: bool = False  # Audio input support
    supports_structured_output: bool = False  # JSON mode/structured responses
    supports_web_search: bool = False  # Built-in web search capability
    supports_code_execution: bool = False  # Code execution capability
    notes: Optional[str] = None

    @property
    def max_tokens(self) -> int:
        """Backward compatibility for max_tokens."""
        return self.output_tokens


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""

    provider_type: ProviderType
    display_name: str
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    requires_api_key: bool = True
    api_key_env_var: Optional[str] = None
    supports_rate_limiting: bool = True  # Provider-level rate limit handling
    supports_fallbacks: bool = True  # Native fallback support
    supports_batch: bool = False  # Batch API support
    default_timeout: int = 60  # Provider-recommended timeout in seconds
    requires_org_id: bool = False  # For providers needing organization ID
    base_url: Optional[str] = None  # Custom base URL support

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
            api_key_env_var="GOOGLE_API_KEY",
            supports_batch=True,
            default_timeout=120,
        )
        google.add_model(
            ModelInfo(
                name="gemini-1.5-pro-002",
                display_name="Gemini 1.5 Pro 002",
                context_window=2097152,  # 2M tokens
                output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_structured_output=True,
                supports_code_execution=True,
                notes="Latest Gemini Pro with 2M context window",
            )
        )
        google.add_model(
            ModelInfo(
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                context_window=2097152,  # 2M tokens
                output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_structured_output=True,
                notes="Gemini Pro with multimodal capabilities",
            )
        )
        google.add_model(
            ModelInfo(
                name="gemini-1.5-flash-002",
                display_name="Gemini 1.5 Flash 002",
                context_window=1048576,  # 1M tokens
                output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_structured_output=True,
                notes="Latest fast Gemini model with 1M context",
            )
        )
        google.add_model(
            ModelInfo(
                name="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                context_window=1048576,  # 1M tokens
                output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_structured_output=True,
                notes="Fast, efficient Gemini model",
            )
        )
        google.add_model(
            ModelInfo(
                name="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                context_window=1048576,  # 1M tokens
                output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_structured_output=True,
                supports_web_search=True,
                supports_code_execution=True,
                notes="Latest Gemini 2.0 with web search and code execution",
            )
        )
        self._providers[ProviderType.GOOGLE] = google

        # OpenAI
        openai = ProviderInfo(
            provider_type=ProviderType.OPENAI,
            display_name="OpenAI",
            api_key_env_var="OPENAI_API_KEY",
            supports_batch=True,
            default_timeout=60,
            requires_org_id=True,  # Optional but supported
        )
        openai.add_model(
            ModelInfo(
                name="gpt-4o-2024-11-20",
                display_name="GPT-4o (Nov 2024)",
                context_window=128000,  # 128K tokens
                output_tokens=16384,
                supports_vision=True,
                supports_structured_output=True,
                supports_audio=True,
                notes="Latest GPT-4o with vision and audio support",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-4o",
                display_name="GPT-4o",
                context_window=128000,  # 128K tokens
                output_tokens=16384,
                supports_vision=True,
                supports_structured_output=True,
                supports_audio=True,
                notes="GPT-4 Optimized with multimodal capabilities",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-4o-mini-2024-07-18",
                display_name="GPT-4o Mini (July 2024)",
                context_window=128000,  # 128K tokens
                output_tokens=16384,
                supports_vision=True,
                supports_structured_output=True,
                notes="Latest GPT-4o Mini version",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-4o-mini",
                display_name="GPT-4o Mini",
                context_window=128000,  # 128K tokens
                output_tokens=16384,
                supports_vision=True,
                supports_structured_output=True,
                notes="Smaller, faster GPT-4 variant",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-4-turbo-2024-04-09",
                display_name="GPT-4 Turbo (April 2024)",
                context_window=128000,  # 128K tokens
                output_tokens=4096,
                supports_vision=True,
                supports_structured_output=True,
                notes="GPT-4 Turbo with vision capabilities",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                context_window=128000,  # 128K tokens
                output_tokens=4096,
                supports_vision=True,
                supports_structured_output=True,
                notes="Latest GPT-4 Turbo",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-3.5-turbo-0125",
                display_name="GPT-3.5 Turbo (Jan 2025)",
                context_window=16385,  # 16K tokens
                output_tokens=4096,
                supports_structured_output=True,
                notes="Latest GPT-3.5 Turbo version",
            )
        )
        openai.add_model(
            ModelInfo(
                name="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                context_window=16385,  # 16K tokens
                output_tokens=4096,
                supports_structured_output=True,
                notes="Fast and efficient for most tasks",
            )
        )
        openai.add_model(
            ModelInfo(
                name="o1-preview",
                display_name="O1 Preview",
                context_window=128000,  # 128K tokens
                output_tokens=32768,
                supports_streaming=False,  # O1 models don't support streaming
                supports_tools=False,  # O1 models don't support tools yet
                notes="Advanced reasoning model (preview)",
            )
        )
        openai.add_model(
            ModelInfo(
                name="o1-mini",
                display_name="O1 Mini",
                context_window=128000,  # 128K tokens
                output_tokens=65536,
                supports_streaming=False,  # O1 models don't support streaming
                supports_tools=False,  # O1 models don't support tools yet
                notes="Faster reasoning model",
            )
        )
        self._providers[ProviderType.OPENAI] = openai

        # Anthropic
        anthropic = ProviderInfo(
            provider_type=ProviderType.ANTHROPIC,
            display_name="Anthropic",
            api_key_env_var="ANTHROPIC_API_KEY",
            default_timeout=60,
        )
        anthropic.add_model(
            ModelInfo(
                name="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet (Oct 2024)",
                context_window=200000,  # 200K tokens
                output_tokens=8192,
                supports_vision=True,
                supports_structured_output=True,
                supports_web_search=True,  # With langchain-anthropic>=0.3.13
                supports_code_execution=True,  # With langchain-anthropic>=0.3.14
                notes="Latest Claude 3.5 Sonnet with web search and code execution",
            )
        )
        anthropic.add_model(
            ModelInfo(
                name="claude-3-5-sonnet-latest",
                display_name="Claude 3.5 Sonnet Latest",
                context_window=200000,  # 200K tokens
                output_tokens=8192,
                supports_vision=True,
                supports_structured_output=True,
                supports_web_search=True,
                supports_code_execution=True,
                notes="Always points to latest Claude 3.5 Sonnet",
            )
        )
        anthropic.add_model(
            ModelInfo(
                name="claude-3-5-haiku-20241022",
                display_name="Claude 3.5 Haiku (Oct 2024)",
                context_window=200000,  # 200K tokens
                output_tokens=8192,
                supports_vision=True,
                supports_structured_output=True,
                notes="Fast Claude 3.5 model",
            )
        )
        anthropic.add_model(
            ModelInfo(
                name="claude-3-opus-20240229",
                display_name="Claude 3 Opus",
                context_window=200000,  # 200K tokens
                output_tokens=4096,
                supports_vision=True,
                supports_structured_output=True,
                notes="Most capable Claude 3 model",
            )
        )
        anthropic.add_model(
            ModelInfo(
                name="claude-3-sonnet-20240229",
                display_name="Claude 3 Sonnet",
                context_window=200000,  # 200K tokens
                output_tokens=4096,
                supports_vision=True,
                supports_structured_output=True,
                notes="Balanced Claude 3 model",
            )
        )
        anthropic.add_model(
            ModelInfo(
                name="claude-3-haiku-20240307",
                display_name="Claude 3 Haiku",
                context_window=200000,  # 200K tokens
                output_tokens=4096,
                supports_vision=True,
                supports_structured_output=True,
                notes="Fast and efficient Claude 3 model",
            )
        )
        self._providers[ProviderType.ANTHROPIC] = anthropic

        # Grok (xAI)
        grok = ProviderInfo(
            provider_type=ProviderType.GROK,
            display_name="Grok",
            api_key_env_var="GROK_API_KEY",
            base_url="https://api.x.ai/v1",  # OpenAI-compatible endpoint
            default_timeout=60,
        )
        grok.add_model(
            ModelInfo(
                name="grok-2-1212",
                display_name="Grok 2 (Dec 2024)",
                context_window=131072,  # 128K tokens
                output_tokens=4096,
                supports_tools=True,
                supports_structured_output=True,
                notes="Latest Grok 2 model",
            )
        )
        grok.add_model(
            ModelInfo(
                name="grok-2-vision-1212",
                display_name="Grok 2 Vision (Dec 2024)",
                context_window=32768,  # 32K tokens
                output_tokens=4096,
                supports_vision=True,
                supports_tools=True,
                supports_structured_output=True,
                notes="Grok 2 with vision capabilities",
            )
        )
        grok.add_model(
            ModelInfo(
                name="grok-1",
                display_name="Grok 1",
                context_window=8192,
                output_tokens=4096,
                notes="Original Grok model",
            )
        )
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
        self, provider_type: ProviderType, model_name: str
    ) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        provider = self.get_provider(provider_type)
        if provider:
            return provider.get_model(model_name)
        return None

    def is_model_supported(self, provider_type: ProviderType, model_name: str) -> bool:
        """Check if a model is supported by a provider."""
        provider = self.get_provider(provider_type)
        if provider:
            return provider.is_model_supported(model_name)
        return False

    def list_models_for_provider(self, provider_type: ProviderType) -> List[str]:
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
        self, provider_type: ProviderType, model_name: str
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

    def get_model_capabilities(
        self, provider_type: ProviderType, model_name: str
    ) -> Optional[Dict[str, bool]]:
        """Get all capabilities for a specific model.

        Returns:
            Dictionary of capability names to boolean values, or None if model not found
        """
        model = self.get_model_info(provider_type, model_name)
        if not model:
            return None

        return {
            "streaming": model.supports_streaming,
            "tools": model.supports_tools,
            "vision": model.supports_vision,
            "audio": model.supports_audio,
            "structured_output": model.supports_structured_output,
            "web_search": model.supports_web_search,
            "code_execution": model.supports_code_execution,
        }

    def supports_feature(
        self, provider_type: ProviderType, model_name: str, feature: str
    ) -> bool:
        """Check if a model supports a specific feature.

        Args:
            provider_type: Provider type
            model_name: Model name
            feature: Feature name (e.g., 'vision', 'tools', 'audio')

        Returns:
            True if the model supports the feature, False otherwise
        """
        capabilities = self.get_model_capabilities(provider_type, model_name)
        if not capabilities:
            return False
        return capabilities.get(feature, False)

    def get_models_by_capability(
        self,
        required_capabilities: List[str],
        provider_type: Optional[ProviderType] = None,
    ) -> List[tuple[ProviderType, str, ModelInfo]]:
        """Get all models that support the required capabilities.

        Args:
            required_capabilities: List of required capability names
            provider_type: Optional provider filter

        Returns:
            List of tuples (provider_type, model_name, model_info)
        """
        results = []

        providers = [provider_type] if provider_type else self.list_providers()

        for prov_type in providers:
            provider = self.get_provider(prov_type)
            if not provider:
                continue

            for model_name, model_info in provider.models.items():
                capabilities = self.get_model_capabilities(prov_type, model_name)
                if not capabilities:
                    continue

                # Check if all required capabilities are supported
                if all(capabilities.get(cap, False) for cap in required_capabilities):
                    results.append((prov_type, model_name, model_info))

        return results

    def get_latest_model(self, provider_type: ProviderType) -> Optional[str]:
        """Get the latest/recommended model for a provider.

        Returns:
            Model name or None if provider not found
        """
        # Define latest models for each provider
        latest_models = {
            ProviderType.GOOGLE: "gemini-2.0-flash",
            ProviderType.OPENAI: "gpt-4o-2024-11-20",
            ProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
            ProviderType.GROK: "grok-2-1212",
        }

        provider = self.get_provider(provider_type)
        if not provider:
            return None

        # Return the latest model if it exists, otherwise return the first model
        latest = latest_models.get(provider_type)
        if latest and self.is_model_supported(provider_type, latest):
            return latest

        # Fallback to first model
        models = self.list_models_for_provider(provider_type)
        return models[0] if models else None

    def get_model_aliases(self, provider_type: ProviderType) -> Dict[str, str]:
        """Get model name aliases for a provider.

        Returns:
            Dictionary mapping aliases to canonical model names
        """
        # Define common aliases
        aliases = {
            ProviderType.OPENAI: {
                "gpt-4o": "gpt-4o-2024-11-20",
                "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
                "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
                "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
            },
            ProviderType.ANTHROPIC: {
                "claude-3-opus": "claude-3-opus-20240229",
                "claude-3-sonnet": "claude-3-sonnet-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-3.5-haiku": "claude-3-5-haiku-20241022",
                "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
            },
            ProviderType.GOOGLE: {
                "gemini-pro": "gemini-1.5-pro-002",
                "gemini-flash": "gemini-1.5-flash-002",
                "gemini-2-flash": "gemini-2.0-flash",
            },
            ProviderType.GROK: {
                "grok-2": "grok-2-1212",
                "grok-2-vision": "grok-2-vision-1212",
            },
        }

        return aliases.get(provider_type, {})


# Global registry instance
registry = ProviderRegistry()
