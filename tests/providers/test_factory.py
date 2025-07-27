"""Tests for provider factory."""

import os
from unittest.mock import Mock, patch, MagicMock
import pytest

from virtual_agora.providers.config import (
    ProviderType,
    GoogleProviderConfig,
    OpenAIProviderConfig,
    AnthropicProviderConfig,
    GrokProviderConfig,
)
from virtual_agora.providers.factory import ProviderFactory, create_provider
from virtual_agora.utils.exceptions import ConfigurationError


class TestProviderFactory:
    """Test ProviderFactory class."""

    def setup_method(self):
        """Set up test method."""
        # Clear cache before each test
        ProviderFactory.clear_cache()

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_google_provider(self, mock_init_chat_model):
        """Test creating Google provider."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance

        config = GoogleProviderConfig(
            model="gemini-2.5-pro",
            api_key="test-key",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
            top_k=40,
        )

        provider = ProviderFactory.create_provider(config, use_cache=False)

        # Check that init_chat_model was called with correct parameters
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args

        # model identifier
        assert call_args[0] == "google_genai:gemini-2.5-pro"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40

        assert provider == mock_instance

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_openai_provider(self, mock_init_chat_model):
        """Test creating OpenAI provider."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance

        config = OpenAIProviderConfig(
            model="gpt-4o",
            api_key="test-key",
            temperature=0.7,
            max_tokens=2000,
            presence_penalty=0.1,
        )

        provider = ProviderFactory.create_provider(config, use_cache=False)

        # Check that init_chat_model was called with correct parameters
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args

        assert call_args[0] == "openai:gpt-4o"  # model identifier
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2000
        assert call_kwargs["presence_penalty"] == 0.1

        assert provider == mock_instance

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_anthropic_provider(self, mock_init_chat_model):
        """Test creating Anthropic provider."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance

        config = AnthropicProviderConfig(
            model="claude-3-opus-20240229",
            api_key="test-key",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.9,
        )

        provider = ProviderFactory.create_provider(config, use_cache=False)

        # Check that init_chat_model was called with correct parameters
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args

        # model identifier
        assert call_args[0] == "anthropic:claude-3-opus-20240229"
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 1500
        assert call_kwargs["top_p"] == 0.9

        assert provider == mock_instance

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_grok_provider(self, mock_init_chat_model):
        """Test creating Grok provider (using OpenAI-compatible API)."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance

        config = GrokProviderConfig(
            model="grok-1",
            api_key="test-key",
            temperature=0.6,
            extra_kwargs={"base_url": "https://api.x.ai/v1"},
        )

        provider = ProviderFactory.create_provider(config, use_cache=False)

        # Check that init_chat_model was called with correct parameters
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args

        assert (
            call_args[0] == "openai:grok-1"
        )  # model identifier (uses openai for Grok)
        assert call_kwargs["temperature"] == 0.6
        assert call_kwargs["base_url"] == "https://api.x.ai/v1"

        assert provider == mock_instance

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_provider_with_dict_config(self, mock_init_chat_model):
        """Test creating provider with dictionary configuration."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance

        provider = ProviderFactory.create_provider(
            {"provider": "google", "model": "gemini-2.5-pro", "api_key": "test-key"},
            use_cache=False,
        )

        assert provider == mock_instance

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_api_key_from_environment(self, mock_init_chat_model):
        """Test that API key is read from environment if not provided."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            mock_instance = Mock()
            mock_init_chat_model.return_value = mock_instance

            config = GoogleProviderConfig(
                model="gemini-2.5-pro",
                # No api_key provided
            )

            provider = ProviderFactory.create_provider(config, use_cache=False)

            # Check that the provider was created successfully
            # (The API key handling is done via environment variables)
            assert provider == mock_instance

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ConfigurationError."""
        config = GoogleProviderConfig(
            model="gemini-2.5-pro",
            # No api_key provided and no environment variable set
        )

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                ProviderFactory.create_provider(config, use_cache=False)

            assert "API key not found" in str(exc_info.value)
            assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_invalid_model_raises_error(self):
        """Test that invalid model raises ConfigurationError."""
        config = GoogleProviderConfig(model="invalid-model", api_key="test-key")

        with pytest.raises(ConfigurationError) as exc_info:
            ProviderFactory.create_provider(config, use_cache=False)

        assert "not supported" in str(exc_info.value)

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ConfigurationError."""

        # Mock an unsupported provider type
        class UnsupportedProviderType:
            value = "unsupported"

        # Create a config with unsupported provider
        config = Mock()
        config.provider = UnsupportedProviderType()
        config.model = "test-model"
        config.api_key = "test-key"

        # Mock registry validation to pass
        with patch(
            "virtual_agora.providers.factory.registry.validate_model_config"
        ) as mock_validate:
            mock_validate.return_value = (True, None)

            with pytest.raises(ConfigurationError) as exc_info:
                ProviderFactory.create_provider(config, use_cache=False)

            assert "Unsupported provider" in str(exc_info.value)

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_caching_behavior(self, mock_init_chat_model):
        """Test provider instance caching."""
        mock_instance = Mock()
        mock_init_chat_model.return_value = mock_instance

        config = GoogleProviderConfig(model="gemini-2.5-pro", api_key="test-key")

        # First call should create instance
        provider1 = ProviderFactory.create_provider(config, use_cache=True)
        assert mock_init_chat_model.call_count == 1
        assert provider1 == mock_instance

        # Second call with same config should use cache
        provider2 = ProviderFactory.create_provider(config, use_cache=True)
        assert mock_init_chat_model.call_count == 1  # Should not have increased
        assert provider2 == mock_instance

        # Call with use_cache=False should create new instance
        provider3 = ProviderFactory.create_provider(config, use_cache=False)
        assert mock_init_chat_model.call_count == 2  # Should have increased
        assert provider3 == mock_instance

    def test_cache_key_generation(self):
        """Test cache key generation."""
        config1 = GoogleProviderConfig(
            model="gemini-2.5-pro", temperature=0.7, streaming=False
        )

        config2 = GoogleProviderConfig(
            model="gemini-2.5-pro", temperature=0.7, streaming=False
        )

        config3 = GoogleProviderConfig(
            model="gemini-2.5-pro",
            temperature=0.8,  # Different temperature
            streaming=False,
        )

        key1 = ProviderFactory._generate_cache_key(config1)
        key2 = ProviderFactory._generate_cache_key(config2)
        key3 = ProviderFactory._generate_cache_key(config3)

        # Same config should generate same key
        assert key1 == key2

        # Different config should generate different key
        assert key1 != key3

    def test_cache_management(self):
        """Test cache management methods."""
        # Test cache size
        assert ProviderFactory.get_cache_size() == 0

        # Add something to cache (mock)
        ProviderFactory._instance_cache["test"] = Mock()
        assert ProviderFactory.get_cache_size() == 1

        # Clear cache
        ProviderFactory.clear_cache()
        assert ProviderFactory.get_cache_size() == 0

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_import_error_handling(self, mock_init_chat_model):
        """Test handling of missing LangChain packages with fallback to legacy."""
        config = GoogleProviderConfig(model="gemini-2.5-pro", api_key="test-key")

        # Mock init_chat_model to raise ImportError
        mock_init_chat_model.side_effect = ImportError(
            "langchain-google-genai package is required"
        )

        # Mock the legacy method to also raise ImportError
        with patch.object(
            ProviderFactory, "_create_provider_instance_legacy"
        ) as mock_legacy:
            mock_legacy.side_effect = ImportError(
                "langchain-google-genai package is required for Google provider. "
                "Install it with: pip install langchain-google-genai"
            )

            with pytest.raises(ImportError) as exc_info:
                ProviderFactory.create_provider(config, use_cache=False)

            assert "langchain-google-genai package is required" in str(exc_info.value)


class TestCreateProviderFunction:
    """Test create_provider convenience function."""

    def test_create_provider_with_string(self):
        """Test create_provider with string provider type."""
        with patch.object(ProviderFactory, "create_provider") as mock_create:
            mock_instance = Mock()
            mock_create.return_value = mock_instance

            provider = create_provider("google", "gemini-2.5-pro", temperature=0.5)

            # Check that factory was called
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]  # First argument (config)

            assert call_args.provider == ProviderType.GOOGLE
            assert call_args.model == "gemini-2.5-pro"
            assert call_args.temperature == 0.5

            assert provider == mock_instance

    def test_create_provider_with_enum(self):
        """Test create_provider with ProviderType enum."""
        with patch.object(ProviderFactory, "create_provider") as mock_create:
            mock_instance = Mock()
            mock_create.return_value = mock_instance

            provider = create_provider(ProviderType.OPENAI, "gpt-4o", max_tokens=1000)

            # Check that factory was called
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0][0]  # First argument (config)

            assert call_args.provider == ProviderType.OPENAI
            assert call_args.model == "gpt-4o"
            assert call_args.max_tokens == 1000

            assert provider == mock_instance
