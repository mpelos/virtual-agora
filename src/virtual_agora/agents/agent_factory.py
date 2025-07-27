"""Agent Factory for Virtual Agora.

This module provides factory functionality for creating discussion agents
based on configuration specifications.
"""

from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel

from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.topic_report_agent import TopicReportAgent
from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent
from virtual_agora.config.models import Config, AgentConfig, ModeratorConfig
from virtual_agora.providers.factory import ProviderFactory
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import ConfigurationError

logger = get_logger(__name__)


class AgentFactory:
    """Factory for creating discussion agents based on configuration."""

    def __init__(self, config: Config):
        """Initialize the agent factory.

        Args:
            config: Virtual Agora configuration
        """
        self.config = config
        self.provider_factory = ProviderFactory()
        self._created_agents: Dict[str, DiscussionAgent] = {}
        self._moderator: Optional[ModeratorAgent] = None
        self._summarizer: Optional[SummarizerAgent] = None
        self._topic_report: Optional[TopicReportAgent] = None
        self._ecclesia_report: Optional[EcclesiaReportAgent] = None

    def create_all_agents(self) -> Dict[str, DiscussionAgent]:
        """Create all discussion agents based on configuration.

        Returns:
            Dictionary mapping agent IDs to DiscussionAgent instances

        Raises:
            ConfigurationError: If agent creation fails
        """
        agents = {}

        try:
            for agent_config in self.config.agents:
                agent_instances = self._create_agents_from_config(agent_config)
                agents.update(agent_instances)

            self._created_agents = agents
            logger.info(f"Successfully created {len(agents)} discussion agents")
            return agents

        except Exception as e:
            logger.error(f"Failed to create agents: {e}")
            raise ConfigurationError(f"Agent creation failed: {e}") from e

    def create_moderator(self) -> ModeratorAgent:
        """Create the moderator agent based on configuration.

        Returns:
            ModeratorAgent instance

        Raises:
            ConfigurationError: If moderator creation fails
        """
        if self._moderator is not None:
            return self._moderator

        try:
            # Create provider configuration for moderator
            provider_config = {
                "provider": self.config.moderator.provider.value,
                "model": self.config.moderator.model,
            }

            # Create LLM instance
            llm = self.provider_factory.create_provider(provider_config)

            # Create moderator agent
            moderator_id = "moderator"
            self._moderator = ModeratorAgent(
                agent_id=moderator_id, llm=llm, enable_error_handling=True
            )

            logger.info(
                f"Successfully created moderator with model {self.config.moderator.model}"
            )
            return self._moderator

        except Exception as e:
            logger.error(f"Failed to create moderator: {e}")
            raise ConfigurationError(f"Moderator creation failed: {e}") from e

    def create_summarizer(self) -> SummarizerAgent:
        """Create the summarizer agent based on configuration.

        Returns:
            SummarizerAgent instance

        Raises:
            ConfigurationError: If summarizer creation fails
        """
        if self._summarizer is not None:
            return self._summarizer

        try:
            # Create provider configuration for summarizer
            provider_config = {
                "provider": self.config.summarizer.provider.value,
                "model": self.config.summarizer.model,
            }

            # Create LLM instance
            llm = self.provider_factory.create_provider(provider_config)

            # Create summarizer agent
            summarizer_id = "summarizer"
            self._summarizer = SummarizerAgent(
                agent_id=summarizer_id,
                llm=llm,
                compression_ratio=0.3,
                max_summary_tokens=self.config.summarizer.max_tokens or 500,
                enable_error_handling=True,
            )

            logger.info(
                f"Successfully created summarizer with model {self.config.summarizer.model}"
            )
            return self._summarizer

        except Exception as e:
            logger.error(f"Failed to create summarizer: {e}")
            raise ConfigurationError(f"Summarizer creation failed: {e}") from e

    def create_topic_report(self) -> TopicReportAgent:
        """Create the topic report agent based on configuration.

        Returns:
            TopicReportAgent instance

        Raises:
            ConfigurationError: If topic report agent creation fails
        """
        if self._topic_report is not None:
            return self._topic_report

        try:
            # Create provider configuration for topic report
            provider_config = {
                "provider": self.config.topic_report.provider.value,
                "model": self.config.topic_report.model,
            }

            # Create LLM instance
            llm = self.provider_factory.create_provider(provider_config)

            # Create topic report agent
            topic_report_id = "topic_report"
            self._topic_report = TopicReportAgent(
                agent_id=topic_report_id, llm=llm, enable_error_handling=True
            )

            logger.info(
                f"Successfully created topic report agent with model {self.config.topic_report.model}"
            )
            return self._topic_report

        except Exception as e:
            logger.error(f"Failed to create topic report agent: {e}")
            raise ConfigurationError(f"Topic report agent creation failed: {e}") from e

    def create_ecclesia_report(self) -> EcclesiaReportAgent:
        """Create the ecclesia report agent based on configuration.

        Returns:
            EcclesiaReportAgent instance

        Raises:
            ConfigurationError: If ecclesia report agent creation fails
        """
        if self._ecclesia_report is not None:
            return self._ecclesia_report

        try:
            # Create provider configuration for ecclesia report
            provider_config = {
                "provider": self.config.ecclesia_report.provider.value,
                "model": self.config.ecclesia_report.model,
            }

            # Create LLM instance
            llm = self.provider_factory.create_provider(provider_config)

            # Create ecclesia report agent
            ecclesia_report_id = "ecclesia_report"
            self._ecclesia_report = EcclesiaReportAgent(
                agent_id=ecclesia_report_id, llm=llm, enable_error_handling=True
            )

            logger.info(
                f"Successfully created ecclesia report agent with model {self.config.ecclesia_report.model}"
            )
            return self._ecclesia_report

        except Exception as e:
            logger.error(f"Failed to create ecclesia report agent: {e}")
            raise ConfigurationError(
                f"Ecclesia report agent creation failed: {e}"
            ) from e

    def _create_agents_from_config(
        self, agent_config: AgentConfig
    ) -> Dict[str, DiscussionAgent]:
        """Create agents from a single agent configuration.

        Args:
            agent_config: Configuration for creating agents

        Returns:
            Dictionary mapping agent IDs to DiscussionAgent instances
        """
        agents = {}

        # Create provider configuration
        provider_config = {
            "provider": agent_config.provider.value,
            "model": agent_config.model,
        }

        # Create agents with unique IDs
        for i in range(1, agent_config.count + 1):
            agent_id = self._generate_agent_id(agent_config.model, i)

            try:
                # Create LLM instance for this agent
                llm = self.provider_factory.create_provider(
                    provider_config, use_cache=True
                )

                # Create discussion agent
                agent = DiscussionAgent(
                    agent_id=agent_id, llm=llm, enable_error_handling=True
                )

                agents[agent_id] = agent
                logger.debug(
                    f"Created agent {agent_id} with model {agent_config.model}"
                )

            except Exception as e:
                logger.error(f"Failed to create agent {agent_id}: {e}")
                # Continue creating other agents even if one fails
                continue

        if not agents:
            raise ConfigurationError(
                f"Failed to create any agents for configuration: "
                f"{agent_config.provider.value}:{agent_config.model}"
            )

        logger.info(
            f"Created {len(agents)}/{agent_config.count} agents for "
            f"{agent_config.provider.value}:{agent_config.model}"
        )

        return agents

    def _generate_agent_id(self, model: str, index: int) -> str:
        """Generate unique agent ID based on model and index.

        Args:
            model: Model name
            index: Agent index (1-based)

        Returns:
            Unique agent identifier
        """
        # Use the format specified in the project spec: model-index
        return f"{model}-{index}"

    def get_agent_names(self) -> List[str]:
        """Get list of all created agent names.

        Returns:
            List of agent IDs
        """
        return list(self._created_agents.keys())

    def get_agent_by_id(self, agent_id: str) -> Optional[DiscussionAgent]:
        """Get an agent by its ID.

        Args:
            agent_id: The ID of the agent to retrieve

        Returns:
            DiscussionAgent instance or None if not found
        """
        return self._created_agents.get(agent_id)

    def get_all_agents(self) -> Dict[str, DiscussionAgent]:
        """Get all created agents.

        Returns:
            Dictionary mapping agent IDs to DiscussionAgent instances
        """
        return self._created_agents.copy()

    def get_agent_info(self) -> List[Dict[str, Any]]:
        """Get information about all created agents.

        Returns:
            List of agent information dictionaries
        """
        agent_info = []

        for agent_id, agent in self._created_agents.items():
            info = agent.get_agent_info()
            info.update(agent.get_agent_state_info())
            agent_info.append(info)

        return agent_info

    def validate_agent_creation(self) -> Dict[str, Any]:
        """Validate that agents were created successfully.

        Returns:
            Validation results dictionary
        """
        total_expected = sum(agent_config.count for agent_config in self.config.agents)
        total_created = len(self._created_agents)

        validation_results = {
            "success": total_created > 0,
            "total_expected": total_expected,
            "total_created": total_created,
            "creation_rate": (
                total_created / total_expected if total_expected > 0 else 0
            ),
            "missing_agents": total_expected - total_created,
            "agent_details": [],
        }

        # Add details for each configured agent type
        for agent_config in self.config.agents:
            created_count = sum(
                1
                for agent_id in self._created_agents.keys()
                if agent_id.startswith(agent_config.model)
            )

            agent_detail = {
                "provider": agent_config.provider.value,
                "model": agent_config.model,
                "expected_count": agent_config.count,
                "created_count": created_count,
                "success": created_count > 0,
            }
            validation_results["agent_details"].append(agent_detail)

        return validation_results

    def create_agent_pool(self, include_moderator: bool = True) -> Dict[str, Any]:
        """Create complete agent pool including all specialized agents and discussion agents.

        Args:
            include_moderator: Whether to include specialized agents (kept for backward compatibility)

        Returns:
            Dictionary containing all agents and metadata
        """
        pool = {
            "discussion_agents": {},
            "moderator": None,
            "summarizer": None,
            "topic_report": None,
            "ecclesia_report": None,
            "created_at": datetime.now(),
            "total_agents": 0,
            "validation": {},
        }

        try:
            # Create discussion agents
            pool["discussion_agents"] = self.create_all_agents()

            # Create specialized agents if requested
            if include_moderator:
                specialized = self.create_specialized_agents()
                pool["moderator"] = specialized["moderator"]
                pool["summarizer"] = specialized["summarizer"]
                pool["topic_report"] = specialized["topic_report"]
                pool["ecclesia_report"] = specialized["ecclesia_report"]

            # Update totals
            pool["total_agents"] = len(pool["discussion_agents"])
            if include_moderator:
                pool["total_agents"] += 4  # Four specialized agents

            # Add validation results
            pool["validation"] = self.validate_agent_creation()

            logger.info(
                f"Successfully created agent pool with {pool['total_agents']} total agents"
            )

            return pool

        except Exception as e:
            logger.error(f"Failed to create agent pool: {e}")
            raise

    def create_fallback_agents(
        self,
        failed_configs: List[AgentConfig],
        fallback_provider: str = "openai",
        fallback_model: str = "gpt-4o-mini",
    ) -> Dict[str, DiscussionAgent]:
        """Create fallback agents for failed configurations.

        Args:
            failed_configs: List of configurations that failed
            fallback_provider: Provider to use for fallback agents
            fallback_model: Model to use for fallback agents

        Returns:
            Dictionary of fallback agents
        """
        fallback_agents = {}

        for config in failed_configs:
            # Create fallback configuration
            fallback_config = {"provider": fallback_provider, "model": fallback_model}

            for i in range(1, config.count + 1):
                agent_id = f"{config.model}-{i}-fallback"

                try:
                    llm = self.provider_factory.create_provider(fallback_config)
                    agent = DiscussionAgent(
                        agent_id=agent_id, llm=llm, enable_error_handling=True
                    )

                    fallback_agents[agent_id] = agent
                    logger.info(f"Created fallback agent {agent_id}")

                except Exception as e:
                    logger.error(f"Failed to create fallback agent {agent_id}: {e}")
                    continue

        return fallback_agents

    def create_specialized_agents(self) -> Dict[str, Any]:
        """Create all specialized agents based on configuration.

        Returns:
            Dictionary mapping agent type to agent instance

        Raises:
            ConfigurationError: If any agent creation fails
        """
        agents = {}

        try:
            # Create all specialized agents
            agents["moderator"] = self.create_moderator()
            agents["summarizer"] = self.create_summarizer()
            agents["topic_report"] = self.create_topic_report()
            agents["ecclesia_report"] = self.create_ecclesia_report()

            logger.info(f"Successfully created all {len(agents)} specialized agents")
            return agents

        except Exception as e:
            logger.error(f"Failed to create specialized agents: {e}")
            raise

    def reset_factory(self) -> None:
        """Reset the factory state."""
        self._created_agents.clear()
        self._moderator = None
        self._summarizer = None
        self._topic_report = None
        self._ecclesia_report = None
        logger.info("Agent factory reset")


def create_agents_from_config(config: Config) -> Dict[str, Any]:
    """Convenience function to create agents from configuration.

    Args:
        config: Virtual Agora configuration

    Returns:
        Dictionary containing agent pool
    """
    factory = AgentFactory(config)
    return factory.create_agent_pool()


def create_specialized_agents(config: Config) -> Dict[str, Any]:
    """Convenience function to create all specialized agents from configuration.

    Args:
        config: Virtual Agora configuration

    Returns:
        Dictionary mapping agent type to agent instance
    """
    factory = AgentFactory(config)
    return factory.create_specialized_agents()


def validate_agent_requirements(config: Config) -> Dict[str, Any]:
    """Validate that agent configuration meets requirements.

    Args:
        config: Virtual Agora configuration

    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "total_agents": 0,
        "unique_models": set(),
        "providers": set(),
    }

    # Calculate totals
    results["total_agents"] = sum(agent.count for agent in config.agents)

    # Check minimum requirements
    if results["total_agents"] < 2:
        results["valid"] = False
        results["issues"].append("At least 2 agents required for discussion")

    if results["total_agents"] > 20:
        results["valid"] = False
        results["issues"].append("Maximum 20 agents allowed")

    # Analyze diversity
    for agent_config in config.agents:
        results["unique_models"].add(agent_config.model)
        results["providers"].add(agent_config.provider.value)

    # Recommendations
    if len(results["unique_models"]) == 1:
        results["warnings"].append(
            "Consider using diverse models for richer discussions"
        )

    if len(results["providers"]) == 1:
        results["warnings"].append("Consider using multiple providers for resilience")

    return results
