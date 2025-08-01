"""Provider configuration models for Virtual Agora.

This module defines configuration models for LLM providers using Pydantic
for validation. These configurations are used to initialize LangChain
chat models with consistent settings.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"
    MODERATOR = "moderator"


class ProviderConfig(BaseModel):
    """Base configuration for all LLM providers.

    This class defines common settings that apply to all providers.
    Provider-specific configurations can extend this class.
    """

    provider: ProviderType = Field(description="The LLM provider to use")

    model: str = Field(description="The model name/ID to use")

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for text generation (0.0-2.0)",
    )

    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )

    timeout: int = Field(default=300, gt=0, description="Request timeout in seconds")

    streaming: bool = Field(default=False, description="Enable streaming responses")

    api_key: Optional[str] = Field(
        default=None, description="API key (loaded from environment if not provided)"
    )

    # Provider-specific additional kwargs
    extra_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)


class GoogleProviderConfig(ProviderConfig):
    """Configuration specific to Google Gemini models."""

    provider: Literal[ProviderType.GOOGLE] = Field(default=ProviderType.GOOGLE)

    # Google-specific settings
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )

    top_k: Optional[int] = Field(
        default=None, gt=0, description="Top-k sampling parameter"
    )

    safety_settings: Optional[Dict[str, str]] = Field(
        default=None, description="Google safety settings"
    )


class OpenAIProviderConfig(ProviderConfig):
    """Configuration specific to OpenAI models."""

    provider: Literal[ProviderType.OPENAI] = Field(default=ProviderType.OPENAI)

    # OpenAI-specific settings
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for token repetition",
    )

    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for token repetition",
    )

    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )

    seed: Optional[int] = Field(
        default=None, description="Random seed for deterministic generation"
    )


class AnthropicProviderConfig(ProviderConfig):
    """Configuration specific to Anthropic Claude models."""

    provider: Literal[ProviderType.ANTHROPIC] = Field(default=ProviderType.ANTHROPIC)

    # Anthropic-specific settings
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )

    top_k: Optional[int] = Field(
        default=None, gt=0, description="Top-k sampling parameter"
    )


class GrokProviderConfig(ProviderConfig):
    """Configuration specific to Grok models.

    Note: This is a placeholder. Actual Grok configuration
    will depend on their API specification.
    """

    provider: Literal[ProviderType.GROK] = Field(default=ProviderType.GROK)


def create_provider_config(provider: str, model: str, **kwargs) -> ProviderConfig:
    """Factory function to create appropriate provider configuration.

    Args:
        provider: Provider type (google, openai, anthropic, grok)
        model: Model name
        **kwargs: Additional configuration parameters

    Returns:
        Provider-specific configuration instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    config_map = {
        ProviderType.GOOGLE: GoogleProviderConfig,
        ProviderType.OPENAI: OpenAIProviderConfig,
        ProviderType.ANTHROPIC: AnthropicProviderConfig,
        ProviderType.GROK: GrokProviderConfig,
    }

    if provider_lower not in [p.value for p in ProviderType]:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {[p.value for p in ProviderType]}"
        )

    config_class = config_map[ProviderType(provider_lower)]
    return config_class(model=model, **kwargs)
