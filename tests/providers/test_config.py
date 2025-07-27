"""Tests for provider configuration models."""

import pytest
from pydantic import ValidationError

from virtual_agora.providers.config import (
    ProviderType,
    ProviderConfig,
    GoogleProviderConfig,
    OpenAIProviderConfig,
    AnthropicProviderConfig,
    GrokProviderConfig,
    create_provider_config,
)


class TestProviderType:
    """Test ProviderType enum."""

    def test_provider_types(self):
        """Test all provider types are defined."""
        assert ProviderType.GOOGLE.value == "google"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.GROK.value == "grok"

    def test_provider_type_from_string(self):
        """Test creating provider type from string."""
        assert ProviderType("google") == ProviderType.GOOGLE
        assert ProviderType("openai") == ProviderType.OPENAI


class TestProviderConfig:
    """Test base ProviderConfig."""

    def test_basic_config(self):
        """Test creating basic provider config."""
        config = ProviderConfig(provider=ProviderType.GOOGLE, model="gemini-2.5-pro")
        assert config.provider == ProviderType.GOOGLE
        assert config.model == "gemini-2.5-pro"
        assert config.temperature == 0.7  # default
        assert config.max_tokens is None  # default
        assert config.timeout == 30  # default
        assert config.streaming is False  # default

    def test_config_with_all_fields(self):
        """Test config with all fields specified."""
        config = ProviderConfig(
            provider=ProviderType.OPENAI,
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            timeout=60,
            streaming=True,
            api_key="test-key",
            extra_kwargs={"custom": "value"},
        )
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 60
        assert config.streaming is True
        assert config.api_key == "test-key"
        assert config.extra_kwargs == {"custom": "value"}

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        config = ProviderConfig(
            provider=ProviderType.GOOGLE, model="test", temperature=0.0
        )
        assert config.temperature == 0.0

        config = ProviderConfig(
            provider=ProviderType.GOOGLE, model="test", temperature=2.0
        )
        assert config.temperature == 2.0

        # Invalid temperatures
        with pytest.raises(ValidationError):
            ProviderConfig(provider=ProviderType.GOOGLE, model="test", temperature=-0.1)

        with pytest.raises(ValidationError):
            ProviderConfig(provider=ProviderType.GOOGLE, model="test", temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        config = ProviderConfig(
            provider=ProviderType.GOOGLE, model="test", max_tokens=1
        )
        assert config.max_tokens == 1

        # Invalid max_tokens
        with pytest.raises(ValidationError):
            ProviderConfig(provider=ProviderType.GOOGLE, model="test", max_tokens=0)

        with pytest.raises(ValidationError):
            ProviderConfig(provider=ProviderType.GOOGLE, model="test", max_tokens=-1)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeout
        config = ProviderConfig(provider=ProviderType.GOOGLE, model="test", timeout=1)
        assert config.timeout == 1

        # Invalid timeout
        with pytest.raises(ValidationError):
            ProviderConfig(provider=ProviderType.GOOGLE, model="test", timeout=0)


class TestGoogleProviderConfig:
    """Test GoogleProviderConfig."""

    def test_google_specific_fields(self):
        """Test Google-specific configuration fields."""
        config = GoogleProviderConfig(
            model="gemini-2.5-pro",
            top_p=0.9,
            top_k=40,
            safety_settings={"HARASSMENT": "BLOCK_LOW"},
        )
        assert config.provider == ProviderType.GOOGLE
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.safety_settings == {"HARASSMENT": "BLOCK_LOW"}

    def test_top_p_validation(self):
        """Test top_p validation."""
        # Valid top_p
        config = GoogleProviderConfig(model="test", top_p=0.0)
        assert config.top_p == 0.0

        config = GoogleProviderConfig(model="test", top_p=1.0)
        assert config.top_p == 1.0

        # Invalid top_p
        with pytest.raises(ValidationError):
            GoogleProviderConfig(model="test", top_p=-0.1)

        with pytest.raises(ValidationError):
            GoogleProviderConfig(model="test", top_p=1.1)

    def test_top_k_validation(self):
        """Test top_k validation."""
        # Valid top_k
        config = GoogleProviderConfig(model="test", top_k=1)
        assert config.top_k == 1

        # Invalid top_k
        with pytest.raises(ValidationError):
            GoogleProviderConfig(model="test", top_k=0)


class TestOpenAIProviderConfig:
    """Test OpenAIProviderConfig."""

    def test_openai_specific_fields(self):
        """Test OpenAI-specific configuration fields."""
        config = OpenAIProviderConfig(
            model="gpt-4",
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            seed=42,
        )
        assert config.provider == ProviderType.OPENAI
        assert config.top_p == 0.9
        assert config.presence_penalty == 0.1
        assert config.frequency_penalty == 0.2
        assert config.seed == 42

    def test_penalty_validation(self):
        """Test presence and frequency penalty validation."""
        # Valid penalties
        config = OpenAIProviderConfig(
            model="test", presence_penalty=-2.0, frequency_penalty=2.0
        )
        assert config.presence_penalty == -2.0
        assert config.frequency_penalty == 2.0

        # Invalid penalties
        with pytest.raises(ValidationError):
            OpenAIProviderConfig(model="test", presence_penalty=-2.1)

        with pytest.raises(ValidationError):
            OpenAIProviderConfig(model="test", frequency_penalty=2.1)


class TestAnthropicProviderConfig:
    """Test AnthropicProviderConfig."""

    def test_anthropic_specific_fields(self):
        """Test Anthropic-specific configuration fields."""
        config = AnthropicProviderConfig(model="claude-3-opus", top_p=0.9, top_k=40)
        assert config.provider == ProviderType.ANTHROPIC
        assert config.top_p == 0.9
        assert config.top_k == 40


class TestGrokProviderConfig:
    """Test GrokProviderConfig."""

    def test_grok_basic_config(self):
        """Test basic Grok configuration."""
        config = GrokProviderConfig(model="grok-1")
        assert config.provider == ProviderType.GROK
        assert config.model == "grok-1"


class TestCreateProviderConfig:
    """Test create_provider_config factory function."""

    def test_create_google_config(self):
        """Test creating Google provider config."""
        config = create_provider_config(
            provider="google", model="gemini-2.5-pro", temperature=0.5, top_p=0.9
        )
        assert isinstance(config, GoogleProviderConfig)
        assert config.provider == ProviderType.GOOGLE
        assert config.model == "gemini-2.5-pro"
        assert config.temperature == 0.5
        assert config.top_p == 0.9

    def test_create_openai_config(self):
        """Test creating OpenAI provider config."""
        config = create_provider_config(
            provider="openai", model="gpt-4", presence_penalty=0.1
        )
        assert isinstance(config, OpenAIProviderConfig)
        assert config.provider == ProviderType.OPENAI
        assert config.presence_penalty == 0.1

    def test_create_anthropic_config(self):
        """Test creating Anthropic provider config."""
        config = create_provider_config(
            provider="anthropic", model="claude-3-opus", top_k=40
        )
        assert isinstance(config, AnthropicProviderConfig)
        assert config.provider == ProviderType.ANTHROPIC
        assert config.top_k == 40

    def test_create_grok_config(self):
        """Test creating Grok provider config."""
        config = create_provider_config(provider="grok", model="grok-1")
        assert isinstance(config, GrokProviderConfig)
        assert config.provider == ProviderType.GROK

    def test_create_with_provider_enum(self):
        """Test creating config with ProviderType enum."""
        config = create_provider_config(
            provider=ProviderType.GOOGLE, model="gemini-2.5-pro"
        )
        assert isinstance(config, GoogleProviderConfig)

    def test_invalid_provider(self):
        """Test creating config with invalid provider."""
        with pytest.raises(ValueError) as exc_info:
            create_provider_config(provider="invalid", model="test")
        assert "Unsupported provider" in str(exc_info.value)

    def test_ignore_invalid_kwargs(self):
        """Test that invalid kwargs are ignored for specific providers."""
        # Create OpenAI config with Google-specific field
        config = create_provider_config(
            provider="openai", model="gpt-4", top_k=40  # Google/Anthropic specific
        )
        assert isinstance(config, OpenAIProviderConfig)
        # top_k should be ignored, not raise an error
        assert not hasattr(config, "top_k")
