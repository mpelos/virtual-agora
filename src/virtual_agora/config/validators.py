"""Configuration validators for Virtual Agora.

This module provides additional validation logic beyond the basic
Pydantic schema validation.
"""

from collections import Counter
from typing import Optional, Set, Dict, Any

from virtual_agora.config.models import (
    Config, Provider, 
    SummarizerConfig, TopicReportConfig, EcclesiaReportConfig
)
from virtual_agora.utils.exceptions import ConfigurationError
from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


class ConfigValidator:
    """Provides additional validation for configurations."""

    def __init__(self, config: Config):
        """Initialize validator with a configuration.

        Args:
            config: Configuration to validate.
        """
        self.config = config

    def validate_all(self) -> None:
        """Run all validation checks.

        Raises:
            ConfigurationError: If any validation fails.
        """
        self.validate_provider_diversity()
        self.validate_model_compatibility()
        self.validate_resource_requirements()
        self.validate_agent_naming_conflicts()
        self.validate_specialized_agents()

    def validate_provider_diversity(self) -> None:
        """Check if configuration has reasonable provider diversity.

        This is a warning-level check to encourage diverse perspectives.
        """
        # Count providers (including specialized agents in v1.3)
        providers = [
            self.config.moderator.provider,
            self.config.summarizer.provider,
            self.config.topic_report.provider,
            self.config.ecclesia_report.provider,
        ]
        for agent in self.config.agents:
            providers.extend([agent.provider] * agent.count)

        provider_counts = Counter(providers)
        total_entities = len(providers)

        # Check if one provider dominates (>75% of all agents + moderator)
        for provider, count in provider_counts.items():
            percentage = (count / total_entities) * 100
            if percentage > 75:
                logger.warning(
                    f"Provider '{provider.value}' represents {percentage:.0f}% "
                    f"of all agents. Consider adding more provider diversity "
                    f"for richer discussions."
                )

    def validate_model_compatibility(self) -> None:
        """Validate that selected models are appropriate for their roles."""
        # Check if moderator model is powerful enough
        moderator_model = self.config.moderator.model.lower()

        # These are considered less capable models that might struggle with moderation
        basic_models = ["gpt-3.5", "claude-2", "gemini-flash"]

        if any(basic in moderator_model for basic in basic_models):
            logger.warning(
                f"Moderator is using '{self.config.moderator.model}' which may "
                f"have limited capabilities for complex moderation tasks. "
                f"Consider using a more capable model like gpt-4 or gemini-2.5-pro."
            )

    def validate_resource_requirements(self) -> None:
        """Estimate resource requirements and warn if excessive."""
        total_agents = self.config.get_total_agent_count()

        # Estimate tokens per round (rough approximation)
        avg_tokens_per_response = 500
        avg_context_per_agent = 2000

        estimated_tokens_per_round = (
            total_agents * (avg_tokens_per_response + avg_context_per_agent)
            + avg_context_per_agent  # Moderator
        )

        # Warn if estimated usage is high
        if estimated_tokens_per_round > 50000:
            logger.warning(
                f"Configuration may use approximately {estimated_tokens_per_round:,} "
                f"tokens per discussion round with {total_agents} agents. "
                f"This could result in high API costs."
            )

        # Check for expensive model combinations
        expensive_models = ["gpt-4", "claude-3-opus", "gemini-2.5-pro"]
        expensive_count = 0

        if any(
            model in self.config.moderator.model.lower() for model in expensive_models
        ):
            expensive_count += 1

        for agent in self.config.agents:
            if any(model in agent.model.lower() for model in expensive_models):
                expensive_count += agent.count

        if expensive_count > 5:
            logger.warning(
                f"Configuration uses {expensive_count} instances of expensive models. "
                f"Consider mixing in some less expensive models to reduce costs."
            )

    def validate_agent_naming_conflicts(self) -> None:
        """Check for potential agent naming conflicts."""
        agent_names = self.config.get_agent_names()

        # Check for duplicates (shouldn't happen with our naming scheme)
        name_counts = Counter(agent_names)
        duplicates = [name for name, count in name_counts.items() if count > 1]

        if duplicates:
            raise ConfigurationError(
                f"Agent naming conflict detected. Duplicate names: {', '.join(duplicates)}",
                details={"duplicate_names": duplicates},
            )

        # Check for excessively long names that might cause display issues
        max_name_length = 50
        long_names = [name for name in agent_names if len(name) > max_name_length]

        if long_names:
            logger.warning(
                f"Some agent names exceed {max_name_length} characters "
                f"and may cause display issues: {', '.join(long_names[:3])}"
            )
    
    def validate_specialized_agents(self) -> None:
        """Validate specialized agent configurations for v1.3.
        
        Checks that specialized agents use compatible models and
        appropriate configurations for their roles.
        """
        # Check summarizer configuration
        if hasattr(self.config.summarizer, 'temperature'):
            if self.config.summarizer.temperature > 0.5:
                logger.warning(
                    f"Summarizer temperature is {self.config.summarizer.temperature}. "
                    f"Lower temperatures (0.2-0.4) typically produce more consistent summaries."
                )
        
        # Check topic report agent
        topic_model = self.config.topic_report.model.lower()
        if hasattr(self.config.topic_report, 'max_tokens'):
            if self.config.topic_report.max_tokens < 1000:
                logger.warning(
                    f"Topic report max_tokens is {self.config.topic_report.max_tokens}. "
                    f"Consider increasing to 1500+ for comprehensive reports."
                )
        
        # Check ecclesia report agent
        if hasattr(self.config.ecclesia_report, 'max_tokens'):
            if self.config.ecclesia_report.max_tokens < 1500:
                logger.warning(
                    f"Ecclesia report max_tokens is {self.config.ecclesia_report.max_tokens}. "
                    f"Consider increasing to 2000+ for detailed final reports."
                )
        
        # Validate model availability for all specialized agents
        specialized_agents = [
            ('moderator', self.config.moderator),
            ('summarizer', self.config.summarizer),
            ('topic_report', self.config.topic_report),
            ('ecclesia_report', self.config.ecclesia_report),
        ]
        
        for agent_type, agent_config in specialized_agents:
            if not self._is_model_available(agent_config.provider, agent_config.model):
                logger.warning(
                    f"{agent_type} model '{agent_config.model}' may not be available "
                    f"for provider {agent_config.provider.value}"
                )
    
    def _is_model_available(self, provider: Provider, model: str) -> bool:
        """Check if a model is likely available for a provider.
        
        This is a basic check based on known model patterns.
        """
        model_lower = model.lower()
        
        # Known model patterns by provider
        provider_patterns = {
            Provider.GOOGLE: ['gemini', 'bard'],
            Provider.OPENAI: ['gpt', 'davinci', 'babbage', 'ada'],
            Provider.ANTHROPIC: ['claude'],
            Provider.GROK: ['grok'],
        }
        
        patterns = provider_patterns.get(provider, [])
        return any(pattern in model_lower for pattern in patterns)

    def get_validation_report(self) -> dict[str, Any]:
        """Generate a validation report without raising exceptions.

        Returns:
            Dictionary containing validation results and warnings.
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stats": {
                "total_agents": self.config.get_total_agent_count(),
                "providers": {},
                "models": {},
            },
        }

        # Collect provider stats (including specialized agents)
        providers = [
            self.config.moderator.provider.value,
            self.config.summarizer.provider.value,
            self.config.topic_report.provider.value,
            self.config.ecclesia_report.provider.value,
        ]
        for agent in self.config.agents:
            providers.extend([agent.provider.value] * agent.count)
        provider_counts = Counter(providers)
        report["stats"]["providers"] = dict(provider_counts)

        # Collect model stats
        models = []
        for agent in self.config.agents:
            models.extend([agent.model] * agent.count)
        model_counts = Counter(models)
        report["stats"]["models"] = dict(model_counts)

        # Run validation checks and collect issues
        try:
            self.validate_provider_diversity()
        except ConfigurationError as e:
            report["errors"].append(str(e))
            report["valid"] = False

        try:
            self.validate_model_compatibility()
        except ConfigurationError as e:
            report["errors"].append(str(e))
            report["valid"] = False

        try:
            self.validate_resource_requirements()
        except ConfigurationError as e:
            report["errors"].append(str(e))
            report["valid"] = False

        try:
            self.validate_agent_naming_conflicts()
        except ConfigurationError as e:
            report["errors"].append(str(e))
            report["valid"] = False
            
        try:
            self.validate_specialized_agents()
        except ConfigurationError as e:
            report["errors"].append(str(e))
            report["valid"] = False

        return report
