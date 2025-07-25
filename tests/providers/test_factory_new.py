"""Tests for the new init_chat_model pattern and fallback functionality."""

import pytest
from unittest.mock import Mock, patch
import os

from virtual_agora.providers.config import (
    ProviderType,
    GoogleProviderConfig,
    OpenAIProviderConfig,
    GrokProviderConfig,
)
from virtual_agora.providers.factory import (
    ProviderFactory,
    create_provider,
    create_provider_with_fallbacks,
)
from virtual_agora.utils.exceptions import ConfigurationError


class TestInitChatModelPattern:
    """Test the new init_chat_model pattern implementation."""
    
    def setup_method(self):
        """Set up test method."""
        ProviderFactory.clear_cache()
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_init_chat_model_called_with_correct_params(self, mock_init_chat_model):
        """Test that init_chat_model is called with correct parameters."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        config = GoogleProviderConfig(
            model="gemini-1.5-pro",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        
        provider = ProviderFactory.create_provider(config, use_cache=False)
        
        # Verify init_chat_model was called
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args
        
        # Check model identifier
        assert call_args[0] == "google_genai:gemini-1.5-pro"
        
        # Check parameters
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["timeout"] == 30.0
        assert "disable_streaming" in call_kwargs  # Google uses disable_streaming
        
        assert provider == mock_instance
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_provider_mapping(self, mock_init_chat_model):
        """Test correct provider string mapping."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        test_cases = [
            (ProviderType.GOOGLE, "google_genai", "gemini-1.5-pro"),
            (ProviderType.OPENAI, "openai", "gpt-4o"),
            (ProviderType.ANTHROPIC, "anthropic", "claude-3-opus-20240229"),
            (ProviderType.GROK, "openai", "grok-1"),  # Grok uses OpenAI-compatible
        ]
        
        for provider_type, expected_prefix, model_name in test_cases:
            config = {
                "provider": provider_type,
                "model": model_name,
                "api_key": "test-key"
            }
            provider = create_provider(provider_type, model_name, api_key="test-key")
            
            # Get the last call
            call_args = mock_init_chat_model.call_args[0]
            assert call_args[0].startswith(f"{expected_prefix}:")
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_streaming_parameter_handling(self, mock_init_chat_model):
        """Test correct handling of streaming parameter for different providers."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        # Test Google (uses disable_streaming)
        google_config = GoogleProviderConfig(
            model="gemini-1.5-pro",
            api_key="test-key",
            streaming=False
        )
        ProviderFactory.create_provider(google_config, use_cache=False)
        call_kwargs = mock_init_chat_model.call_args[1]
        assert "disable_streaming" in call_kwargs
        assert call_kwargs["disable_streaming"] is True
        assert "streaming" not in call_kwargs
        
        # Test OpenAI (uses streaming)
        openai_config = OpenAIProviderConfig(
            model="gpt-4o",
            api_key="test-key",
            streaming=True
        )
        ProviderFactory.create_provider(openai_config, use_cache=False)
        call_kwargs = mock_init_chat_model.call_args[1]
        assert "streaming" in call_kwargs
        assert call_kwargs["streaming"] is True
        assert "disable_streaming" not in call_kwargs


class TestFallbackFunctionality:
    """Test the fallback functionality using .with_fallbacks()."""
    
    def setup_method(self):
        """Set up test method."""
        ProviderFactory.clear_cache()
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_create_provider_with_fallbacks(self, mock_init_chat_model):
        """Test creating a provider with fallbacks."""
        # Create mock providers
        primary_mock = Mock()
        primary_mock.with_fallbacks = Mock()
        fallback_mock = Mock()
        
        # Mock the with_fallbacks to return a special object
        fallback_chain_mock = Mock()
        primary_mock.with_fallbacks.return_value = fallback_chain_mock
        
        # Set up init_chat_model to return different mocks
        mock_init_chat_model.side_effect = [primary_mock, fallback_mock]
        
        # Create provider with fallbacks
        primary_config = {
            "provider": "google",
            "model": "gemini-1.5-pro",
            "api_key": "test-key"
        }
        fallback_configs = [
            {
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "test-key"
            }
        ]
        
        result = ProviderFactory.create_provider_with_fallbacks(
            primary_config,
            fallback_configs
        )
        
        # Verify init_chat_model was called twice
        assert mock_init_chat_model.call_count == 2
        
        # Verify with_fallbacks was called
        primary_mock.with_fallbacks.assert_called_once()
        
        # Verify the result is the fallback chain
        assert result == fallback_chain_mock
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_create_provider_with_fallbacks_convenience_function(self, mock_init_chat_model):
        """Test the convenience function for creating providers with fallbacks."""
        primary_mock = Mock()
        fallback_mock1 = Mock()
        fallback_mock2 = Mock()
        fallback_chain_mock = Mock()
        
        # Setup the with_fallbacks method
        primary_mock.with_fallbacks = Mock(return_value=fallback_chain_mock)
        
        mock_init_chat_model.side_effect = [primary_mock, fallback_mock1, fallback_mock2]
        
        # Use convenience function
        result = create_provider_with_fallbacks(
            primary_provider="google",
            primary_model="gemini-1.5-pro",
            fallback_configs=[
                {"provider": "openai", "model": "gpt-4o", "api_key": "test-key"},
                {"provider": "anthropic", "model": "claude-3-opus-20240229", "api_key": "test-key"}
            ],
            api_key="test-key",
            temperature=0.7
        )
        
        # Verify three providers were created (1 primary + 2 fallbacks)
        assert mock_init_chat_model.call_count == 3
        
        # Verify with_fallbacks was called with the fallback providers
        primary_mock.with_fallbacks.assert_called_once_with([fallback_mock1, fallback_mock2])
        
        # Verify we got the fallback chain
        assert result == fallback_chain_mock


class TestErrorHandling:
    """Test error handling and fallback to legacy methods."""
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    @patch('virtual_agora.providers.factory.ProviderFactory._create_provider_instance_legacy')
    def test_fallback_to_legacy_on_import_error(self, mock_legacy, mock_init_chat_model):
        """Test fallback to legacy method on ImportError."""
        # Make init_chat_model raise ImportError
        mock_init_chat_model.side_effect = ImportError("Module not found")
        
        # Set up legacy method
        legacy_instance = Mock()
        mock_legacy.return_value = legacy_instance
        
        config = GoogleProviderConfig(
            model="gemini-1.5-pro",
            api_key="test-key"
        )
        
        result = ProviderFactory.create_provider(config, use_cache=False)
        
        # Verify fallback was used
        mock_legacy.assert_called_once()
        assert result == legacy_instance
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_configuration_error_not_caught(self, mock_init_chat_model):
        """Test that ConfigurationError is not caught and re-raised."""
        # Make init_chat_model raise ConfigurationError
        mock_init_chat_model.side_effect = ConfigurationError("Invalid config")
        
        config = GoogleProviderConfig(
            model="gemini-1.5-pro",
            api_key="test-key"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            ProviderFactory.create_provider(config, use_cache=False)
        
        assert "Invalid config" in str(exc_info.value)


class TestProviderSpecificParameters:
    """Test provider-specific parameter handling."""
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_google_specific_parameters(self, mock_init_chat_model):
        """Test Google-specific parameters are passed correctly."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        config = GoogleProviderConfig(
            model="gemini-1.5-pro",
            api_key="test-key",
            top_p=0.9,
            top_k=40,
            safety_settings={"HARASSMENT": "BLOCK_LOW"}
        )
        
        ProviderFactory.create_provider(config, use_cache=False)
        
        call_kwargs = mock_init_chat_model.call_args[1]
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["safety_settings"] == {"HARASSMENT": "BLOCK_LOW"}
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_openai_specific_parameters(self, mock_init_chat_model):
        """Test OpenAI-specific parameters are passed correctly."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        config = OpenAIProviderConfig(
            model="gpt-4o",
            api_key="test-key",
            presence_penalty=0.1,
            frequency_penalty=0.2,
            seed=42
        )
        
        ProviderFactory.create_provider(config, use_cache=False)
        
        call_kwargs = mock_init_chat_model.call_args[1]
        assert call_kwargs["presence_penalty"] == 0.1
        assert call_kwargs["frequency_penalty"] == 0.2
        assert call_kwargs["seed"] == 42
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_grok_base_url_handling(self, mock_init_chat_model):
        """Test Grok provider uses correct base_url."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        # Test with default base_url
        provider = create_provider("grok", "grok-1", api_key="test-key")
        call_kwargs = mock_init_chat_model.call_args[1]
        assert call_kwargs["base_url"] == "https://api.x.ai/v1"
        
        # Test with custom base_url
        mock_init_chat_model.reset_mock()
        config = GrokProviderConfig(
            model="grok-1",
            api_key="test-key",
            extra_kwargs={"base_url": "https://custom.api/v1"}
        )
        provider = ProviderFactory.create_provider(config, use_cache=False)
        call_kwargs = mock_init_chat_model.call_args[1]
        assert call_kwargs["base_url"] == "https://custom.api/v1"


class TestApiKeyHandling:
    """Test API key handling from environment variables."""
    
    @patch('virtual_agora.providers.factory.init_chat_model')
    def test_api_key_from_config(self, mock_init_chat_model):
        """Test API key is set in environment when provided in config."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance
        
        # Clear any existing API key
        os.environ.pop("GOOGLE_API_KEY", None)
        
        config = GoogleProviderConfig(
            model="gemini-1.5-pro",
            api_key="config-api-key"
        )
        
        ProviderFactory.create_provider(config, use_cache=False)
        
        # Verify API key was set in environment
        assert os.environ.get("GOOGLE_API_KEY") == "config-api-key"
        
        # Clean up
        os.environ.pop("GOOGLE_API_KEY", None)
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ConfigurationError."""
        # Clear any existing API key
        os.environ.pop("GOOGLE_API_KEY", None)
        
        config = GoogleProviderConfig(
            model="gemini-1.5-pro"
            # No api_key provided
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            ProviderFactory.create_provider(config, use_cache=False)
        
        assert "API key not found" in str(exc_info.value)
        assert "GOOGLE_API_KEY" in str(exc_info.value)