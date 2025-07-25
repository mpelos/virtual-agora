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
            context_window=128000,
            output_tokens=4096,
            supports_streaming=True,
            supports_functions=False,
            supports_tools=True,
            supports_vision=True,
            notes="Test notes"
        )
        assert model.name == "test-model"
        assert model.display_name == "Test Model"
        assert model.context_window == 128000
        assert model.output_tokens == 4096
        assert model.max_tokens == 4096  # Backward compatibility
        assert model.supports_streaming is True
        assert model.supports_functions is False
        assert model.supports_tools is True
        assert model.supports_vision is True
        assert model.notes == "Test notes"
    
    def test_model_info_defaults(self):
        """Test ModelInfo default values."""
        model = ModelInfo(
            name="test",
            display_name="Test",
            context_window=128000,
            output_tokens=1000
        )
        assert model.supports_streaming is True
        assert model.supports_functions is True
        assert model.supports_tools is True
        assert model.supports_vision is False
        assert model.supports_audio is False
        assert model.supports_structured_output is False
        assert model.supports_web_search is False
        assert model.supports_code_execution is False
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
            context_window=2000000,
            output_tokens=8192
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
            context_window=2000000,
            output_tokens=8192
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
        provider.add_model(ModelInfo("model1", "Model 1", 128000, 1000))
        provider.add_model(ModelInfo("model2", "Model 2", 128000, 2000))
        
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
        
        provider.add_model(ModelInfo("supported", "Supported", 128000, 1000))
        
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
        model = reg.get_model_info(ProviderType.GOOGLE, "gemini-2.0-flash")
        assert model is not None
        assert model.display_name == "Gemini 2.0 Flash"
        assert model.max_tokens == 8192
        
        # Get non-existent model
        model = reg.get_model_info(ProviderType.GOOGLE, "non-existent")
        assert model is None
    
    def test_is_model_supported(self):
        """Test checking if model is supported."""
        reg = ProviderRegistry()
        
        # Supported models
        assert reg.is_model_supported(ProviderType.GOOGLE, "gemini-2.0-flash")
        assert reg.is_model_supported(ProviderType.OPENAI, "gpt-4o-2024-11-20")
        assert reg.is_model_supported(ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022")
        
        # Unsupported models
        assert not reg.is_model_supported(ProviderType.GOOGLE, "unsupported")
        assert not reg.is_model_supported(ProviderType.OPENAI, "gpt-5")
    
    def test_list_models_for_provider(self):
        """Test listing models for a provider."""
        reg = ProviderRegistry()
        
        # Google models
        google_models = reg.list_models_for_provider(ProviderType.GOOGLE)
        assert "gemini-1.5-pro-002" in google_models
        assert "gemini-1.5-flash-002" in google_models
        assert "gemini-2.0-flash" in google_models
        
        # OpenAI models
        openai_models = reg.list_models_for_provider(ProviderType.OPENAI)
        assert "gpt-4o-2024-11-20" in openai_models
        assert "gpt-4o-mini-2024-07-18" in openai_models
        assert "gpt-4-turbo-2024-04-09" in openai_models
        assert "gpt-3.5-turbo-0125" in openai_models
    
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
        assert any("Gemini 2.0 Flash (gemini-2.0-flash)" in model for model in google_models)
    
    def test_validate_model_config(self):
        """Test validating model configuration."""
        reg = ProviderRegistry()
        
        # Valid configuration
        is_valid, error = reg.validate_model_config(
            ProviderType.GOOGLE,
            "gemini-2.0-flash"
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


class TestNewRegistryFeatures:
    """Test new registry features."""
    
    def test_get_model_capabilities(self):
        """Test getting model capabilities."""
        reg = ProviderRegistry()
        
        # Test Gemini 2.0 Flash capabilities
        capabilities = reg.get_model_capabilities(
            ProviderType.GOOGLE, 
            "gemini-2.0-flash"
        )
        assert capabilities is not None
        assert capabilities["streaming"] is True
        assert capabilities["tools"] is True
        assert capabilities["vision"] is True
        assert capabilities["audio"] is True
        assert capabilities["structured_output"] is True
        assert capabilities["web_search"] is True
        assert capabilities["code_execution"] is True
        
        # Test non-existent model
        capabilities = reg.get_model_capabilities(
            ProviderType.GOOGLE, 
            "non-existent"
        )
        assert capabilities is None
    
    def test_supports_feature(self):
        """Test checking if model supports a specific feature."""
        reg = ProviderRegistry()
        
        # Test vision support
        assert reg.supports_feature(
            ProviderType.GOOGLE, 
            "gemini-2.0-flash", 
            "vision"
        ) is True
        
        # Test feature not supported
        assert reg.supports_feature(
            ProviderType.OPENAI, 
            "o1-preview", 
            "tools"
        ) is False
        
        # Test non-existent model
        assert reg.supports_feature(
            ProviderType.GOOGLE, 
            "non-existent", 
            "vision"
        ) is False
    
    def test_get_models_by_capability(self):
        """Test getting models by required capabilities."""
        reg = ProviderRegistry()
        
        # Test models with vision support
        vision_models = reg.get_models_by_capability(["vision"])
        assert len(vision_models) > 0
        
        # Check that all returned models support vision
        for provider_type, model_name, model_info in vision_models:
            assert model_info.supports_vision is True
        
        # Test models with both vision and tools
        vision_tool_models = reg.get_models_by_capability(["vision", "tools"])
        assert len(vision_tool_models) > 0
        
        # Test with provider filter
        google_vision_models = reg.get_models_by_capability(
            ["vision"], 
            ProviderType.GOOGLE
        )
        assert len(google_vision_models) > 0
        for provider_type, model_name, model_info in google_vision_models:
            assert provider_type == ProviderType.GOOGLE
            assert model_info.supports_vision is True
    
    def test_get_latest_model(self):
        """Test getting latest model for each provider."""
        reg = ProviderRegistry()
        
        # Test latest models
        assert reg.get_latest_model(ProviderType.GOOGLE) == "gemini-2.0-flash"
        assert reg.get_latest_model(ProviderType.OPENAI) == "gpt-4o-2024-11-20"
        assert reg.get_latest_model(ProviderType.ANTHROPIC) == "claude-3-5-sonnet-20241022"
        assert reg.get_latest_model(ProviderType.GROK) == "grok-2-1212"
    
    def test_get_model_aliases(self):
        """Test getting model aliases."""
        reg = ProviderRegistry()
        
        # Test OpenAI aliases
        openai_aliases = reg.get_model_aliases(ProviderType.OPENAI)
        assert "gpt-4o" in openai_aliases
        assert openai_aliases["gpt-4o"] == "gpt-4o-2024-11-20"
        assert openai_aliases["gpt-4o-mini"] == "gpt-4o-mini-2024-07-18"
        
        # Test Anthropic aliases
        anthropic_aliases = reg.get_model_aliases(ProviderType.ANTHROPIC)
        assert "claude-3.5-sonnet" in anthropic_aliases
        assert anthropic_aliases["claude-3.5-sonnet"] == "claude-3-5-sonnet-20241022"
        
        # Test Google aliases
        google_aliases = reg.get_model_aliases(ProviderType.GOOGLE)
        assert "gemini-pro" in google_aliases
        assert google_aliases["gemini-pro"] == "gemini-1.5-pro-002"
    
    def test_backward_compatibility(self):
        """Test backward compatibility for max_tokens property."""
        reg = ProviderRegistry()
        
        model = reg.get_model_info(ProviderType.GOOGLE, "gemini-2.0-flash")
        assert model is not None
        assert model.max_tokens == model.output_tokens
        assert model.max_tokens == 8192
    
    def test_new_model_features(self):
        """Test new model features are properly set."""
        reg = ProviderRegistry()
        
        # Test OpenAI O1 models have correct limitations
        o1_model = reg.get_model_info(ProviderType.OPENAI, "o1-preview")
        assert o1_model is not None
        assert o1_model.supports_streaming is False
        assert o1_model.supports_tools is False
        
        # Test Anthropic models have new capabilities
        claude_model = reg.get_model_info(
            ProviderType.ANTHROPIC, 
            "claude-3-5-sonnet-20241022"
        )
        assert claude_model is not None
        assert claude_model.supports_web_search is True
        assert claude_model.supports_code_execution is True
        
        # Test Grok models have correct setup
        grok_model = reg.get_model_info(ProviderType.GROK, "grok-2-vision-1212")
        assert grok_model is not None
        assert grok_model.supports_vision is True
        assert grok_model.supports_tools is True
    
    def test_provider_features(self):
        """Test provider-level features."""
        reg = ProviderRegistry()
        
        # Test Google provider features
        google = reg.get_provider(ProviderType.GOOGLE)
        assert google is not None
        assert google.supports_batch is True
        assert google.default_timeout == 120
        
        # Test OpenAI provider features
        openai = reg.get_provider(ProviderType.OPENAI)
        assert openai is not None
        assert openai.supports_batch is True
        assert openai.requires_org_id is True
        
        # Test Grok provider features
        grok = reg.get_provider(ProviderType.GROK)
        assert grok is not None
        assert grok.base_url == "https://api.x.ai/v1"