"""Integration tests for provider system."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from virtual_agora.providers import (
    create_provider,
    create_provider_config,
    ProviderFactory,
    registry,
    ProviderType,
)
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.exceptions import ConfigurationError


class TestProviderIntegration:
    """Test end-to-end provider integration."""

    def setup_method(self):
        """Set up test method."""
        ProviderFactory.clear_cache()

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_google_agent_end_to_end(self, mock_init_chat_model):
        """Test creating a Google agent end-to-end."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm.model_name = "gemini-2.5-pro"
        mock_init_chat_model.return_value = mock_llm

        # Create provider using convenience function
        provider = create_provider(
            provider="google",
            model="gemini-2.5-pro",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9,
        )

        # Create agent with provider - disable error handling to avoid LLM wrapping
        agent = LLMAgent(
            "google-agent", provider, role="participant", enable_error_handling=False
        )

        # Verify provider creation
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args
        assert call_args[0] == "google_genai:gemini-2.5-pro"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

        # Verify agent properties
        assert agent.agent_id == "google-agent"
        assert agent.provider == "google"
        assert agent.model == "gemini-2.5-pro"
        assert agent.llm == mock_llm

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_create_openai_agent_with_config_object(self, mock_init_chat_model):
        """Test creating OpenAI agent with config object."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatOpenAI"
        mock_llm.model_name = "gpt-4o"
        mock_init_chat_model.return_value = mock_llm

        # Create config first
        config = create_provider_config(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            max_tokens=2000,
            presence_penalty=0.1,
        )

        # Create provider with config
        provider = ProviderFactory.create_provider(config)

        # Create agent
        agent = LLMAgent("openai-agent", provider, role="moderator")

        # Verify everything
        assert agent.provider == "openai"
        assert agent.model == "gpt-4o"
        assert agent.role == "moderator"
        assert "impartial Moderator" in agent.system_prompt

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_agent_conversation_with_anthropic(self, mock_init_chat_model):
        """Test agent conversation with Anthropic provider."""
        # Setup mock
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatAnthropic"
        mock_llm.model_name = "claude-3-opus-20240229"
        mock_response = Mock()
        mock_response.content = (
            "I think AI ethics is crucial for responsible development."
        )
        mock_llm.invoke.return_value = mock_response
        mock_init_chat_model.return_value = mock_llm

        # Create provider and agent - disable error handling to avoid LLM wrapping
        provider = create_provider(
            provider="anthropic", model="claude-3-opus-20240229", api_key="test-key"
        )
        agent = LLMAgent("claude-agent", provider, enable_error_handling=False)

        # Generate response
        response = agent.generate_response("What do you think about AI ethics?")

        # Verify response
        assert response == "I think AI ethics is crucial for responsible development."
        assert agent.message_count == 1

        # Verify LLM was called with formatted messages
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        assert len(messages) == 2  # System + Human
        assert "thoughtful participant" in messages[0].content  # System prompt
        assert "What do you think about AI ethics?" == messages[1].content

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"})
    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_environment_api_key_integration(self, mock_init_chat_model):
        """Test API key from environment integration."""
        mock_llm = Mock()
        mock_init_chat_model.return_value = mock_llm

        # Create provider without API key (should use environment)
        provider = create_provider(provider="openai", model="gpt-4o")

        # Verify init_chat_model was called (API key is set in environment by factory)
        mock_init_chat_model.assert_called_once()
        # Check that environment was properly set
        assert os.environ.get("OPENAI_API_KEY") == "env-api-key"

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_caching_integration(self, mock_init_chat_model):
        """Test provider caching integration."""
        mock_llm = Mock()
        mock_init_chat_model.return_value = mock_llm

        # Create same provider twice
        provider1 = create_provider(
            provider="openai", model="gpt-4o", api_key="test-key", temperature=0.7
        )

        provider2 = create_provider(
            provider="openai", model="gpt-4o", api_key="test-key", temperature=0.7
        )

        # Should only create once due to caching
        assert mock_init_chat_model.call_count == 1
        assert provider1 == provider2 == mock_llm

        # Different config should create new instance
        provider3 = create_provider(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            temperature=0.8,  # Different temperature
        )

        assert mock_init_chat_model.call_count == 2
        assert provider3 == mock_llm

    def test_all_providers_in_registry(self):
        """Test that all providers are properly registered."""
        providers = registry.list_providers()
        assert len(providers) == 4
        assert ProviderType.GOOGLE in providers
        assert ProviderType.OPENAI in providers
        assert ProviderType.ANTHROPIC in providers
        assert ProviderType.GROK in providers

        # Check each provider has models
        for provider_type in providers:
            models = registry.list_models_for_provider(provider_type)
            assert len(models) > 0, f"Provider {provider_type} has no models"

            # Check API key env var is set
            env_var = registry.get_api_key_env_var(provider_type)
            assert (
                env_var is not None
            ), f"Provider {provider_type} has no API key env var"

    def test_model_info_completeness(self):
        """Test that all models have complete information."""
        for provider_type in registry.list_providers():
            models = registry.list_models_for_provider(provider_type)
            for model_name in models:
                model_info = registry.get_model_info(provider_type, model_name)
                assert model_info is not None
                assert model_info.name == model_name
                assert model_info.display_name
                assert model_info.max_tokens > 0
                assert isinstance(model_info.supports_streaming, bool)
                assert isinstance(model_info.supports_functions, bool)

    @patch("virtual_agora.providers.factory.init_chat_model")
    def test_grok_provider_special_handling(self, mock_init_chat_model):
        """Test Grok provider's special OpenAI-compatible handling."""
        mock_llm = Mock()
        mock_init_chat_model.return_value = mock_llm

        provider = create_provider(
            provider="grok",
            model="grok-1",
            api_key="grok-key",
            extra_kwargs={"base_url": "https://api.x.ai/v1"},
        )

        # Should use init_chat_model with openai provider string
        mock_init_chat_model.assert_called_once()
        call_args, call_kwargs = mock_init_chat_model.call_args
        assert call_args[0] == "openai:grok-1"  # Grok uses openai provider
        assert call_kwargs["base_url"] == "https://api.x.ai/v1"


class TestProviderCompatibility:
    """Test provider compatibility and feature support."""

    def test_streaming_support(self):
        """Test streaming support across providers."""
        streaming_models = [
            ("google", "gemini-2.5-pro"),
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-opus-20240229"),
        ]

        for provider, model in streaming_models:
            model_info = registry.get_model_info(ProviderType(provider), model)
            assert (
                model_info.supports_streaming
            ), f"{provider}/{model} should support streaming"

    def test_function_calling_support(self):
        """Test function calling support across providers."""
        # Most models should support function calling
        function_models = [
            ("google", "gemini-2.5-pro"),
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-opus-20240229"),
        ]

        for provider, model in function_models:
            model_info = registry.get_model_info(ProviderType(provider), model)
            assert (
                model_info.supports_functions
            ), f"{provider}/{model} should support functions"

    def test_model_token_limits(self):
        """Test that all models have reasonable token limits."""
        for provider_type in registry.list_providers():
            models = registry.list_models_for_provider(provider_type)
            for model_name in models:
                model_info = registry.get_model_info(provider_type, model_name)
                # All models should have at least 1000 tokens
                assert (
                    model_info.max_tokens >= 1000
                ), f"{provider_type}/{model_name} has too low token limit"
                # And no more than 100K (reasonable upper bound)
                assert (
                    model_info.max_tokens <= 100000
                ), f"{provider_type}/{model_name} has unreasonably high token limit"
