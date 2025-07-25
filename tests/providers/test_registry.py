"""Tests for provider registry."""

import pytest

from virtual_agora.providers.config import ProviderType
from virtual_agora.providers.registry import (
    ModelInfo,
    ProviderInfo,
    ProviderRegistry,
    registry,
)


class TestModelInfo:
    """Test ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        model = ModelInfo(
            name="test-model",
            display_name="Test Model",
            max_tokens=4096,
            supports_streaming=True,
            supports_functions=False,
            notes="Test notes"
        )
        assert model.name == "test-model"
        assert model.display_name == "Test Model"
        assert model.max_tokens == 4096
        assert model.supports_streaming is True
        assert model.supports_functions is False
        assert model.notes == "Test notes"
    
    def test_model_info_defaults(self):
        """Test ModelInfo default values."""
        model = ModelInfo(
            name="test",
            display_name="Test",
            max_tokens=1000
        )
        assert model.supports_streaming is True
        assert model.supports_functions is True
        assert model.notes is None


class TestProviderInfo:
    """Test ProviderInfo dataclass."""
    
    def test_provider_info_creation(self):
        """Test creating ProviderInfo."""
        provider = ProviderInfo(
            provider_type=ProviderType.GOOGLE,
            display_name="Google Gemini",
            requires_api_key=True,
            api_key_env_var="GOOGLE_API_KEY"
        )
        assert provider.provider_type == ProviderType.GOOGLE
        assert provider.display_name == "Google Gemini"
        assert provider.requires_api_key is True
        assert provider.api_key_env_var == "GOOGLE_API_KEY"
        assert provider.models == {}
    
    def test_add_model(self):
        """Test adding models to provider."""
        provider = ProviderInfo(
            provider_type=ProviderType.GOOGLE,
            display_name="Google"
        )
        
        model = ModelInfo(
            name="gemini-pro",
            display_name="Gemini Pro",
            max_tokens=8192
        )
        provider.add_model(model)
        
        assert "gemini-pro" in provider.models
        assert provider.models["gemini-pro"] == model
    
    def test_get_model(self):
        """Test getting model from provider."""
        provider = ProviderInfo(
            provider_type=ProviderType.GOOGLE,
            display_name="Google"
        )
        
        model = ModelInfo(
            name="gemini-pro",
            display_name="Gemini Pro",
            max_tokens=8192
        )
        provider.add_model(model)
        
        # Get existing model
        retrieved = provider.get_model("gemini-pro")
        assert retrieved == model
        
        # Get non-existent model
        assert provider.get_model("non-existent") is None
    
    def test_list_models(self):
        """Test listing models."""
        provider = ProviderInfo(
            provider_type=ProviderType.GOOGLE,
            display_name="Google"
        )
        
        # Empty initially
        assert provider.list_models() == []
        
        # Add models
        provider.add_model(ModelInfo("model1", "Model 1", 1000))
        provider.add_model(ModelInfo("model2", "Model 2", 2000))
        
        models = provider.list_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models
    
    def test_is_model_supported(self):
        """Test checking if model is supported."""
        provider = ProviderInfo(
            provider_type=ProviderType.GOOGLE,
            display_name="Google"
        )
        
        provider.add_model(ModelInfo("supported", "Supported", 1000))
        
        assert provider.is_model_supported("supported") is True
        assert provider.is_model_supported("unsupported") is False


class TestProviderRegistry:
    """Test ProviderRegistry."""
    
    def test_registry_initialization(self):
        """Test registry is initialized with providers."""
        reg = ProviderRegistry()
        
        # Check all providers are registered
        assert reg.is_provider_supported(ProviderType.GOOGLE)
        assert reg.is_provider_supported(ProviderType.OPENAI)
        assert reg.is_provider_supported(ProviderType.ANTHROPIC)
        assert reg.is_provider_supported(ProviderType.GROK)
    
    def test_get_provider(self):
        """Test getting provider info."""
        reg = ProviderRegistry()
        
        # Get existing provider
        google = reg.get_provider(ProviderType.GOOGLE)
        assert google is not None
        assert google.display_name == "Google Gemini"
        assert google.api_key_env_var == "GOOGLE_API_KEY"
        
        # Get non-existent provider (if we add more enum values)
        # This test is more for completeness
        providers = list(ProviderType)
        for p in providers:
            assert reg.get_provider(p) is not None
    
    def test_list_providers(self):
        """Test listing all providers."""
        reg = ProviderRegistry()
        providers = reg.list_providers()
        
        assert len(providers) == 4
        assert ProviderType.GOOGLE in providers
        assert ProviderType.OPENAI in providers
        assert ProviderType.ANTHROPIC in providers
        assert ProviderType.GROK in providers
    
    def test_get_model_info(self):
        """Test getting model info."""
        reg = ProviderRegistry()
        
        # Get existing model
        model = reg.get_model_info(ProviderType.GOOGLE, "gemini-1.5-pro")
        assert model is not None
        assert model.display_name == "Gemini 1.5 Pro"
        assert model.max_tokens == 8192
        
        # Get non-existent model
        model = reg.get_model_info(ProviderType.GOOGLE, "non-existent")
        assert model is None
    
    def test_is_model_supported(self):
        """Test checking if model is supported."""
        reg = ProviderRegistry()
        
        # Supported models
        assert reg.is_model_supported(ProviderType.GOOGLE, "gemini-1.5-pro")
        assert reg.is_model_supported(ProviderType.OPENAI, "gpt-4o")
        assert reg.is_model_supported(ProviderType.ANTHROPIC, "claude-3-opus-20240229")
        
        # Unsupported models
        assert not reg.is_model_supported(ProviderType.GOOGLE, "unsupported")
        assert not reg.is_model_supported(ProviderType.OPENAI, "gpt-5")
    
    def test_list_models_for_provider(self):
        """Test listing models for a provider."""
        reg = ProviderRegistry()
        
        # Google models
        google_models = reg.list_models_for_provider(ProviderType.GOOGLE)
        assert "gemini-1.5-pro" in google_models
        assert "gemini-1.5-flash" in google_models
        assert "gemini-2.0-flash" in google_models
        
        # OpenAI models
        openai_models = reg.list_models_for_provider(ProviderType.OPENAI)
        assert "gpt-4o" in openai_models
        assert "gpt-4o-mini" in openai_models
        assert "gpt-4-turbo" in openai_models
        assert "gpt-3.5-turbo" in openai_models
    
    def test_get_api_key_env_var(self):
        """Test getting API key environment variable."""
        reg = ProviderRegistry()
        
        assert reg.get_api_key_env_var(ProviderType.GOOGLE) == "GOOGLE_API_KEY"
        assert reg.get_api_key_env_var(ProviderType.OPENAI) == "OPENAI_API_KEY"
        assert reg.get_api_key_env_var(ProviderType.ANTHROPIC) == "ANTHROPIC_API_KEY"
        assert reg.get_api_key_env_var(ProviderType.GROK) == "GROK_API_KEY"
    
    def test_get_all_models(self):
        """Test getting all models grouped by provider."""
        reg = ProviderRegistry()
        all_models = reg.get_all_models()
        
        assert "Google Gemini" in all_models
        assert "OpenAI" in all_models
        assert "Anthropic" in all_models
        assert "Grok" in all_models
        
        # Check format includes display name and model name
        google_models = all_models["Google Gemini"]
        assert any("Gemini 1.5 Pro (gemini-1.5-pro)" in model for model in google_models)
    
    def test_validate_model_config(self):
        """Test validating model configuration."""
        reg = ProviderRegistry()
        
        # Valid configuration
        is_valid, error = reg.validate_model_config(
            ProviderType.GOOGLE,
            "gemini-1.5-pro"
        )
        assert is_valid is True
        assert error is None
        
        # Invalid model
        is_valid, error = reg.validate_model_config(
            ProviderType.GOOGLE,
            "invalid-model"
        )
        assert is_valid is False
        assert "not supported" in error
        assert "Available models:" in error


class TestGlobalRegistry:
    """Test the global registry instance."""
    
    def test_global_registry_exists(self):
        """Test that global registry is available."""
        assert registry is not None
        assert isinstance(registry, ProviderRegistry)
    
    def test_global_registry_has_providers(self):
        """Test that global registry has providers initialized."""
        assert registry.is_provider_supported(ProviderType.GOOGLE)
        assert registry.is_provider_supported(ProviderType.OPENAI)
        assert registry.is_provider_supported(ProviderType.ANTHROPIC)
        assert registry.is_provider_supported(ProviderType.GROK)