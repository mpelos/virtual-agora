"""Tests for AgentFactory implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel

from virtual_agora.agents.agent_factory import (
    AgentFactory,
    create_agents_from_config,
    validate_agent_requirements,
)
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.config.models import (
    Config,
    AgentConfig,
    ModeratorConfig,
    SummarizerConfig,
    TopicReportConfig,
    EcclesiaReportConfig,
    Provider,
)
from virtual_agora.utils.exceptions import ConfigurationError


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    agent_configs = [
        AgentConfig(provider=Provider.OPENAI, model="gpt-4", count=2),
        AgentConfig(provider=Provider.ANTHROPIC, model="claude-3-opus", count=1),
    ]

    moderator_config = ModeratorConfig(provider=Provider.OPENAI, model="gpt-4")
    summarizer_config = SummarizerConfig(provider=Provider.OPENAI, model="gpt-4")
    topic_report_config = TopicReportConfig(
        provider=Provider.ANTHROPIC, model="claude-3-opus"
    )
    ecclesia_report_config = EcclesiaReportConfig(
        provider=Provider.GOOGLE, model="gemini-pro"
    )

    return Config(
        agents=agent_configs,
        moderator=moderator_config,
        summarizer=summarizer_config,
        topic_report=topic_report_config,
        ecclesia_report=ecclesia_report_config,
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    llm = Mock(spec=BaseChatModel)
    llm.__class__.__name__ = "ChatOpenAI"
    llm.model_name = "gpt-4"
    return llm


class TestAgentFactory:
    """Test AgentFactory class."""

    def setup_method(self):
        """Set up test method."""
        # Mock provider factory
        self.mock_provider_factory = Mock()
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_provider_factory.create_provider.return_value = self.mock_llm

    def test_factory_initialization(self, mock_config):
        """Test factory initialization."""
        factory = AgentFactory(mock_config)

        assert factory.config == mock_config
        assert factory.provider_factory is not None
        assert len(factory._created_agents) == 0
        assert factory._moderator is None

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_all_agents_success(self, mock_provider_class, mock_config):
        """Test successful agent creation."""
        # Setup mock provider factory
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        agents = factory.create_all_agents()

        # Should create 3 agents total (2 + 1)
        assert len(agents) == 3

        # Check agent IDs follow the format model-index
        expected_ids = ["gpt-4-1", "gpt-4-2", "claude-3-opus-1"]
        assert set(agents.keys()) == set(expected_ids)

        # All should be DiscussionAgent instances
        for agent in agents.values():
            assert isinstance(agent, DiscussionAgent)

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_moderator_success(self, mock_provider_class, mock_config):
        """Test successful moderator creation."""
        # Setup mock provider factory
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        moderator = factory.create_moderator()

        assert isinstance(moderator, ModeratorAgent)
        assert moderator.agent_id == "moderator"
        assert factory._moderator == moderator

        # Creating again should return same instance
        moderator2 = factory.create_moderator()
        assert moderator2 is moderator

    def test_generate_agent_id(self, mock_config):
        """Test agent ID generation."""
        factory = AgentFactory(mock_config)

        # Test different models and indices
        assert factory._generate_agent_id("gpt-4", 1) == "gpt-4-1"
        assert factory._generate_agent_id("claude-3-opus", 2) == "claude-3-opus-2"
        assert factory._generate_agent_id("llama-2", 10) == "llama-2-10"

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_agents_from_config_single(self, mock_provider_class, mock_config):
        """Test creating agents from single configuration."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)

        # Test with first config (2 agents)
        agent_config = mock_config.agents[0]
        agents = factory._create_agents_from_config(agent_config)

        assert len(agents) == 2
        assert "gpt-4-1" in agents
        assert "gpt-4-2" in agents

        for agent in agents.values():
            assert isinstance(agent, DiscussionAgent)

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_get_agent_names(self, mock_provider_class, mock_config):
        """Test getting agent names."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        factory.create_all_agents()

        names = factory.get_agent_names()
        expected_names = ["gpt-4-1", "gpt-4-2", "claude-3-opus-1"]

        assert set(names) == set(expected_names)

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_get_agent_by_id(self, mock_provider_class, mock_config):
        """Test getting agent by ID."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        factory.create_all_agents()

        # Test existing agent
        agent = factory.get_agent_by_id("gpt-4-1")
        assert agent is not None
        assert isinstance(agent, DiscussionAgent)
        assert agent.agent_id == "gpt-4-1"

        # Test non-existing agent
        agent = factory.get_agent_by_id("non-existent")
        assert agent is None

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_get_all_agents(self, mock_provider_class, mock_config):
        """Test getting all agents."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        created_agents = factory.create_all_agents()
        all_agents = factory.get_all_agents()

        assert len(all_agents) == len(created_agents)
        assert all_agents is not created_agents  # Should be a copy

        for agent_id, agent in all_agents.items():
            assert agent_id == agent.agent_id
            assert isinstance(agent, DiscussionAgent)

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_get_agent_info(self, mock_provider_class, mock_config):
        """Test getting agent information."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        factory.create_all_agents()

        agent_info = factory.get_agent_info()

        assert len(agent_info) == 3
        for info in agent_info:
            assert "id" in info
            assert "model" in info
            assert "provider" in info
            assert "role" in info
            assert "message_count" in info
            assert "created_at" in info

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_validate_agent_creation(self, mock_provider_class, mock_config):
        """Test agent creation validation."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        factory.create_all_agents()

        validation = factory.validate_agent_creation()

        assert validation["success"] is True
        assert validation["total_expected"] == 3
        assert validation["total_created"] == 3
        assert validation["creation_rate"] == 1.0
        assert validation["missing_agents"] == 0
        assert len(validation["agent_details"]) == 2  # 2 different configs

        # Check agent details
        for detail in validation["agent_details"]:
            assert detail["success"] is True
            assert detail["created_count"] > 0


class TestAgentFactoryErrorHandling:
    """Test error handling in AgentFactory."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_agents_provider_error(self, mock_provider_class, mock_config):
        """Test handling provider creation errors."""
        # Setup mock to raise exception
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.side_effect = Exception("Provider error")

        factory = AgentFactory(mock_config)

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_all_agents()

        assert "Agent creation failed" in str(exc_info.value)

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_agents_partial_failure(self, mock_provider_class, mock_config):
        """Test handling partial agent creation failures."""
        # Setup mock to fail on some agents
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance

        def side_effect(config, use_cache=False):
            if "claude" in config["model"]:
                raise Exception("Claude provider error")
            return self.mock_llm

        mock_provider_instance.create_provider.side_effect = side_effect

        factory = AgentFactory(mock_config)

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_all_agents()

        assert "Agent creation failed" in str(exc_info.value)
        assert "claude-3-opus" in str(exc_info.value)

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_moderator_error(self, mock_provider_class, mock_config):
        """Test handling moderator creation errors."""
        # Setup mock to raise exception
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.side_effect = Exception(
            "Moderator error"
        )

        factory = AgentFactory(mock_config)

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_moderator()

        assert "Moderator creation failed" in str(exc_info.value)

    def test_no_agents_created_error(self, mock_config):
        """Test error when no agents are created."""
        from pydantic_core import ValidationError

        # Create config with no agents
        with pytest.raises(ValidationError):
            Config(
                agents=[],
                moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4"),
            )


class TestAgentFactoryAdvancedFeatures:
    """Test advanced AgentFactory features."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_agent_pool(self, mock_provider_class, mock_config):
        """Test creating complete agent pool."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        pool = factory.create_agent_pool(include_moderator=True)

        assert "discussion_agents" in pool
        assert "moderator" in pool
        assert "summarizer" in pool
        assert "topic_report" in pool
        assert "ecclesia_report" in pool
        assert "created_at" in pool
        assert "total_agents" in pool
        assert "validation" in pool

        assert len(pool["discussion_agents"]) == 3
        assert isinstance(pool["moderator"], ModeratorAgent)
        # In v1.3, we have 3 discussion agents + 4 specialized agents
        assert pool["total_agents"] == 7
        assert isinstance(pool["created_at"], datetime)
        assert pool["validation"]["success"] is True

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_agent_pool_no_moderator(self, mock_provider_class, mock_config):
        """Test creating agent pool without moderator."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)
        pool = factory.create_agent_pool(include_moderator=False)

        assert len(pool["discussion_agents"]) == 3
        assert pool["moderator"] is None
        assert pool["total_agents"] == 3

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_create_fallback_agents(self, mock_provider_class, mock_config):
        """Test creating fallback agents."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)

        # Create fallback agents for failed configs
        failed_configs = [mock_config.agents[1]]  # claude config
        fallback_agents = factory.create_fallback_agents(
            failed_configs, fallback_provider="openai", fallback_model="gpt-4o-mini"
        )

        assert len(fallback_agents) == 1
        assert "claude-3-opus-1-fallback" in fallback_agents

        fallback_agent = fallback_agents["claude-3-opus-1-fallback"]
        assert isinstance(fallback_agent, DiscussionAgent)

    def test_reset_factory(self, mock_config):
        """Test resetting factory state."""
        factory = AgentFactory(mock_config)

        # Add some state
        factory._created_agents["test"] = Mock()
        factory._moderator = Mock()

        # Reset
        factory.reset_factory()

        assert len(factory._created_agents) == 0
        assert factory._moderator is None


class TestAgentFactoryUtilityFunctions:
    """Test utility functions in agent_factory module."""

    @patch("virtual_agora.agents.agent_factory.AgentFactory")
    def test_create_agents_from_config_function(self, mock_factory_class, mock_config):
        """Test create_agents_from_config utility function."""
        # Setup mock factory
        mock_factory_instance = Mock()
        mock_factory_class.return_value = mock_factory_instance

        expected_pool = {"discussion_agents": {}, "moderator": None}
        mock_factory_instance.create_agent_pool.return_value = expected_pool

        # Call function
        result = create_agents_from_config(mock_config)

        # Verify
        mock_factory_class.assert_called_once_with(mock_config)
        mock_factory_instance.create_agent_pool.assert_called_once()
        assert result == expected_pool

    def test_validate_agent_requirements_success(self):
        """Test successful agent requirements validation."""
        config = Config(
            agents=[
                AgentConfig(provider=Provider.OPENAI, model="gpt-4", count=2),
                AgentConfig(
                    provider=Provider.ANTHROPIC, model="claude-3-opus", count=1
                ),
            ],
            moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4"),
            summarizer=SummarizerConfig(provider=Provider.OPENAI, model="gpt-4"),
            topic_report=TopicReportConfig(
                provider=Provider.ANTHROPIC, model="claude-3-opus"
            ),
            ecclesia_report=EcclesiaReportConfig(
                provider=Provider.GOOGLE, model="gemini-pro"
            ),
        )

        results = validate_agent_requirements(config)

        assert results["valid"] is True
        assert results["total_agents"] == 3
        assert len(results["issues"]) == 0
        assert len(results["unique_models"]) == 2
        assert len(results["providers"]) == 2

    def test_validate_agent_requirements_too_few(self):
        """Test validation with too few agents."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            Config(
                agents=[AgentConfig(provider=Provider.OPENAI, model="gpt-4", count=1)],
                moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4"),
            )

    def test_validate_agent_requirements_too_many(self):
        """Test validation with too many agents."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            Config(
                agents=[AgentConfig(provider=Provider.OPENAI, model="gpt-4", count=25)],
                moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4"),
            )

    def test_validate_agent_requirements_diversity_warnings(self):
        """Test validation warnings for lack of diversity."""
        # Same model config
        config = Config(
            agents=[AgentConfig(provider=Provider.OPENAI, model="gpt-4", count=3)],
            moderator=ModeratorConfig(provider=Provider.OPENAI, model="gpt-4"),
            summarizer=SummarizerConfig(provider=Provider.OPENAI, model="gpt-4"),
            topic_report=TopicReportConfig(provider=Provider.OPENAI, model="gpt-4"),
            ecclesia_report=EcclesiaReportConfig(
                provider=Provider.OPENAI, model="gpt-4"
            ),
        )

        results = validate_agent_requirements(config)

        assert results["valid"] is True
        assert (
            "Consider using diverse models for richer discussions"
            in results["warnings"]
        )
        assert "Consider using multiple providers for resilience" in results["warnings"]


class TestAgentFactoryIntegration:
    """Integration tests for AgentFactory."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=BaseChatModel)
        llm.model_name = "mock-model"
        llm.temperature = 0.7
        return llm

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_full_factory_workflow(self, mock_provider_class, mock_config):
        """Test complete factory workflow."""
        # Setup mock
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        # Create factory
        factory = AgentFactory(mock_config)

        # 1. Create all agents
        agents = factory.create_all_agents()
        assert len(agents) == 3

        # 2. Create moderator
        moderator = factory.create_moderator()
        assert isinstance(moderator, ModeratorAgent)

        # 3. Get agent info
        info = factory.get_agent_info()
        assert len(info) == 3

        # 4. Validate creation
        validation = factory.validate_agent_creation()
        assert validation["success"] is True

        # 5. Get individual agents
        agent = factory.get_agent_by_id("gpt-4-1")
        assert agent is not None

        # 6. Create complete pool
        pool = factory.create_agent_pool()
        # In v1.3, we have 3 discussion agents + 4 specialized agents
        assert pool["total_agents"] == 7

        # 7. Reset factory
        factory.reset_factory()
        assert len(factory._created_agents) == 0

    @patch("virtual_agora.agents.agent_factory.ProviderFactory")
    def test_factory_with_caching(self, mock_provider_class, mock_config):
        """Test factory behavior with provider caching."""
        # Setup mock with call tracking
        mock_provider_instance = Mock()
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.create_provider.return_value = self.mock_llm

        factory = AgentFactory(mock_config)

        # Create agents (should use caching)
        agents = factory.create_all_agents()

        # Verify caching was used for agents
        call_args_list = mock_provider_instance.create_provider.call_args_list

        # Should have calls with use_cache=True for agents
        agent_calls = [call for call in call_args_list if "use_cache" in call.kwargs]
        assert len(agent_calls) >= 3  # At least 3 agent creation calls

        for call in agent_calls:
            assert call.kwargs.get("use_cache") is True

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
