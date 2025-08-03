"""Mock provider factory for deterministic LLM testing.

This module provides a replacement for the Virtual Agora provider system
that creates deterministic LLMs instead of real API-based providers, while
maintaining the same interfaces and behavior patterns.
"""

from typing import Dict, Any, Optional, Union, List
from unittest.mock import Mock

from virtual_agora.config.models import Config as VirtualAgoraConfig, Provider
from virtual_agora.utils.logging import get_logger

from .role_specific_llms import (
    ModeratorDeterministicLLM,
    DiscussionAgentDeterministicLLM,
    SummarizerDeterministicLLM,
    ReportWriterDeterministicLLM,
)

logger = get_logger(__name__)


class DeterministicProviderFactory:
    """Factory for creating deterministic LLM providers for testing.

    This factory replaces the real provider creation system with deterministic
    implementations that maintain realistic behavior while being completely
    predictable for testing purposes.
    """

    def __init__(self):
        self.created_providers = {}
        self.provider_call_count = 0
        self.agent_counter = {"agent": 0}

    def create_provider(
        self, config_model: str, config_provider: str = None, role: str = None, **kwargs
    ) -> Union[
        ModeratorDeterministicLLM,
        DiscussionAgentDeterministicLLM,
        SummarizerDeterministicLLM,
        ReportWriterDeterministicLLM,
    ]:
        """Create a deterministic LLM provider based on configuration.

        Args:
            config_model: Model name from configuration (e.g., "gpt-4o", "claude-3-opus-20240229")
            config_provider: Provider name from configuration (e.g., "openai", "anthropic")
            role: Role of the agent (moderator, agent, summarizer, report_writer)
            **kwargs: Additional configuration parameters

        Returns:
            Appropriate deterministic LLM instance for the role
        """
        self.provider_call_count += 1

        # Determine role if not explicitly provided
        if not role:
            role = self._infer_role_from_config(config_model, config_provider, **kwargs)

        # Create appropriate LLM based on role
        provider_key = f"{role}_{self.provider_call_count}"

        if role == "moderator":
            llm = ModeratorDeterministicLLM(model=config_model, **kwargs)
        elif role == "summarizer":
            llm = SummarizerDeterministicLLM(model=config_model, **kwargs)
        elif role == "report_writer":
            llm = ReportWriterDeterministicLLM(model=config_model, **kwargs)
        else:  # Default to discussion agent
            self.agent_counter["agent"] += 1
            agent_id = f"agent_{self.agent_counter['agent']}"
            llm = DiscussionAgentDeterministicLLM(
                model=config_model, agent_id=agent_id, **kwargs
            )

        # Store for tracking
        self.created_providers[provider_key] = llm

        logger.info(
            f"Created deterministic LLM: {role} using {config_model} ({config_provider})"
        )

        return llm

    def _infer_role_from_config(
        self, config_model: str, config_provider: str, **kwargs
    ) -> str:
        """Infer agent role from configuration parameters.

        Args:
            config_model: Model name
            config_provider: Provider name (normalized string, not enum)
            **kwargs: Additional configuration that might contain role hints

        Returns:
            Inferred role name
        """
        # Check for explicit role indicators in kwargs
        if "role" in kwargs:
            return kwargs["role"]

        # Infer from model patterns
        if "moderator" in config_model.lower():
            return "moderator"
        elif "summary" in config_model.lower() or "summariz" in config_model.lower():
            return "summarizer"
        elif "report" in config_model.lower() or "writer" in config_model.lower():
            return "report_writer"

        # Check creation order - first few calls are typically special roles
        if self.provider_call_count == 1:
            return "moderator"
        elif self.provider_call_count == 2:
            return "summarizer"
        elif self.provider_call_count == 3:
            return "report_writer"
        else:
            return "agent"

    def get_provider_summary(self) -> Dict[str, Any]:
        """Get summary of all created providers.

        Returns:
            Dictionary containing provider creation statistics
        """
        role_counts = {}
        for provider in self.created_providers.values():
            role = provider.role
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total_providers": len(self.created_providers),
            "role_counts": role_counts,
            "creation_order": list(self.created_providers.keys()),
            "all_providers": {
                key: {
                    "role": provider.role,
                    "model": provider.model_name,
                    "provider": provider.provider,
                    "call_count": provider.call_count,
                }
                for key, provider in self.created_providers.items()
            },
        }

    def reset_factory(self) -> None:
        """Reset factory state for new test runs."""
        for provider in self.created_providers.values():
            provider.reset_state()

        self.created_providers.clear()
        self.provider_call_count = 0
        self.agent_counter = {"agent": 0}

        logger.debug("Reset DeterministicProviderFactory state")


# Global factory instance
_deterministic_factory = DeterministicProviderFactory()


def create_deterministic_provider(provider: str, model: str, **kwargs) -> Union[
    ModeratorDeterministicLLM,
    DiscussionAgentDeterministicLLM,
    SummarizerDeterministicLLM,
    ReportWriterDeterministicLLM,
]:
    """Create a deterministic provider using the global factory.

    This function provides the same interface as the real provider creation
    function but returns deterministic LLMs for testing.

    Args:
        provider: Provider type (e.g., "OpenAI", "Anthropic") - Provider enum value
        model: Model name (e.g., "gpt-4o", "claude-3-opus-20240229")
        **kwargs: Additional configuration parameters

    Returns:
        Deterministic LLM instance
    """
    # Normalize provider string to handle both enum values and strings
    provider_normalized = provider
    if hasattr(provider, "value"):
        provider_normalized = provider.value

    return _deterministic_factory.create_provider(
        config_model=model, config_provider=provider_normalized, **kwargs
    )


def get_provider_factory() -> DeterministicProviderFactory:
    """Get the global deterministic provider factory.

    Returns:
        Global DeterministicProviderFactory instance
    """
    return _deterministic_factory


def reset_provider_factory() -> None:
    """Reset the global provider factory state."""
    _deterministic_factory.reset_factory()


class VirtualAgoraConfigMock:
    """Mock configuration for Virtual Agora testing.

    Provides realistic configuration objects that work with the deterministic
    provider system while maintaining compatibility with the real config system.
    """

    @staticmethod
    def create_test_config(
        moderator_model: str = "gpt-4o",
        moderator_provider: str = "OpenAI",
        summarizer_model: str = "gpt-4o",
        summarizer_provider: str = "OpenAI",
        report_writer_model: str = "claude-3-opus-20240229",
        report_writer_provider: str = "Anthropic",
        agent_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> Mock:
        """Create a mock configuration for testing.

        Args:
            moderator_model: Model for moderator agent
            moderator_provider: Provider for moderator agent
            summarizer_model: Model for summarizer agent
            summarizer_provider: Provider for summarizer agent
            report_writer_model: Model for report writer agent
            report_writer_provider: Provider for report writer agent
            agent_configs: List of agent configurations

        Returns:
            Mock configuration object
        """
        if agent_configs is None:
            agent_configs = [
                {"model": "gpt-4o", "provider": "OpenAI", "count": 1},
                {
                    "model": "claude-3-opus-20240229",
                    "provider": "Anthropic",
                    "count": 1,
                },
            ]

        config = Mock(spec=VirtualAgoraConfig)

        # Moderator configuration
        config.moderator = Mock()
        config.moderator.model = moderator_model
        config.moderator.provider = Provider(moderator_provider)
        config.moderator.temperature = 0.7
        config.moderator.max_tokens = 4000

        # Summarizer configuration
        config.summarizer = Mock()
        config.summarizer.model = summarizer_model
        config.summarizer.provider = Provider(summarizer_provider)
        config.summarizer.temperature = 0.6
        config.summarizer.max_tokens = 3000

        # Report writer configuration
        config.report_writer = Mock()
        config.report_writer.model = report_writer_model
        config.report_writer.provider = Provider(report_writer_provider)
        config.report_writer.temperature = 0.5
        config.report_writer.max_tokens = 5000

        # Agent configurations
        config.agents = []
        for agent_config in agent_configs:
            agent = Mock()
            agent.model = agent_config["model"]
            agent.provider = Provider(agent_config["provider"])
            agent.count = agent_config["count"]
            agent.temperature = agent_config.get("temperature", 0.7)
            agent.max_tokens = agent_config.get("max_tokens", 3000)
            config.agents.append(agent)

        # Additional methods
        config.get_total_agent_count = Mock(
            return_value=sum(agent["count"] for agent in agent_configs)
        )

        return config


def setup_deterministic_testing_environment():
    """Set up the complete deterministic testing environment.

    This function should be called at the beginning of test sessions to
    ensure all LLM providers are replaced with deterministic versions.

    Returns:
        Dictionary containing setup information
    """
    from unittest.mock import patch

    # Reset factory state
    reset_provider_factory()

    # Set up patching context
    patches = {
        "create_provider": patch(
            "virtual_agora.providers.factory.create_provider",
            side_effect=create_deterministic_provider,
        ),
        "graph_create_provider": patch(
            "virtual_agora.flow.graph_v13.create_provider",
            side_effect=create_deterministic_provider,
        ),
    }

    # Start all patches
    started_patches = {}
    for name, patch_obj in patches.items():
        started_patches[name] = patch_obj.start()

    return {
        "factory": get_provider_factory(),
        "patches": patches,
        "started_patches": started_patches,
        "config_mock": VirtualAgoraConfigMock,
    }


def teardown_deterministic_testing_environment(setup_info: Dict[str, Any]):
    """Clean up the deterministic testing environment.

    Args:
        setup_info: Setup information returned by setup_deterministic_testing_environment
    """
    # Stop all patches
    for patch_name, patch_obj in setup_info.get("patches", {}).items():
        try:
            patch_obj.stop()
        except Exception as e:
            logger.warning(f"Error stopping patch {patch_name}: {e}")

    # Reset factory
    reset_provider_factory()

    logger.debug("Deterministic testing environment cleaned up")
