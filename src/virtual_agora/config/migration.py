"""Configuration migration utilities for Virtual Agora.

This module provides utilities to migrate configuration files from
v1.1 format to v1.3 format with specialized agents.
"""

from typing import Dict, Any, Optional
from copy import deepcopy

from virtual_agora.config.models import Provider
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


def get_default_model(provider: str, agent_type: str) -> str:
    """Get a sensible default model for a given provider and agent type.

    Args:
        provider: The provider name (e.g., 'Google', 'OpenAI')
        agent_type: The type of agent (e.g., 'summarizer', 'topic_report')

    Returns:
        A default model string appropriate for the provider and agent type
    """
    provider_lower = provider.lower()

    # Default models by provider
    default_models = {
        "google": {
            "summarizer": "gemini-2.5-flash-lite",
            "topic_report": "gemini-2.5-pro",
            "ecclesia_report": "gemini-2.5-pro",
        },
        "openai": {
            "summarizer": "gpt-4o",
            "topic_report": "gpt-4o",
            "ecclesia_report": "gpt-4o",
        },
        "anthropic": {
            "summarizer": "claude-3-opus-20240229",
            "topic_report": "claude-3-opus-20240229",
            "ecclesia_report": "claude-3-opus-20240229",
        },
        "grok": {
            "summarizer": "grok-beta",
            "topic_report": "grok-beta",
            "ecclesia_report": "grok-beta",
        },
    }

    return default_models.get(provider_lower, {}).get(agent_type, "default-model")


def migrate_config_v1_to_v3(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert v1.1 config to v1.3 format.

    This function adds the required specialized agents (summarizer, topic_report,
    ecclesia_report) to a v1.1 configuration that only has a moderator and agents.

    Args:
        old_config: The v1.1 configuration dictionary

    Returns:
        A v1.3 compatible configuration dictionary
    """
    # Deep copy to avoid modifying the original
    new_config = deepcopy(old_config)

    # Get moderator provider as a fallback for specialized agents
    moderator_provider = old_config.get("moderator", {}).get("provider", "Google")

    # Add specialized agents if they don't exist
    if "summarizer" not in new_config:
        logger.info(
            f"Adding default summarizer configuration using provider: {moderator_provider}"
        )
        new_config["summarizer"] = {
            "provider": moderator_provider,
            "model": get_default_model(moderator_provider, "summarizer"),
        }

    if "topic_report" not in new_config:
        # For topic_report, we might prefer a more capable model
        # Try to use Anthropic if available in agents, otherwise use moderator's provider
        topic_provider = moderator_provider
        for agent in old_config.get("agents", []):
            if agent.get("provider", "").lower() == "anthropic":
                topic_provider = "Anthropic"
                break

        logger.info(
            f"Adding default topic_report configuration using provider: {topic_provider}"
        )
        new_config["topic_report"] = {
            "provider": topic_provider,
            "model": get_default_model(topic_provider, "topic_report"),
        }

    if "ecclesia_report" not in new_config:
        logger.info(
            f"Adding default ecclesia_report configuration using provider: {moderator_provider}"
        )
        new_config["ecclesia_report"] = {
            "provider": moderator_provider,
            "model": get_default_model(moderator_provider, "ecclesia_report"),
        }

    return new_config


def detect_config_version(config: Dict[str, Any]) -> str:
    """Detect the version of a configuration dictionary.

    Args:
        config: The configuration dictionary

    Returns:
        The detected version string ('1.1' or '1.3')
    """
    # v1.3 has specialized agents
    if all(key in config for key in ["summarizer", "topic_report", "ecclesia_report"]):
        return "1.3"

    # v1.1 only has moderator and agents
    if "moderator" in config and "agents" in config:
        return "1.1"

    # Unknown version
    return "unknown"


def is_migration_needed(config: Dict[str, Any]) -> bool:
    """Check if a configuration needs migration from v1.1 to v1.3.

    Args:
        config: The configuration dictionary

    Returns:
        True if migration is needed, False otherwise
    """
    version = detect_config_version(config)
    return version == "1.1"


def validate_migrated_config(config: Dict[str, Any]) -> bool:
    """Validate that a migrated configuration has all required v1.3 fields.

    Args:
        config: The configuration dictionary

    Returns:
        True if the configuration is valid for v1.3, False otherwise
    """
    required_fields = [
        "moderator",
        "summarizer",
        "topic_report",
        "ecclesia_report",
        "agents",
    ]

    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field in v1.3 config: {field}")
            return False

        # Check that each agent config has provider and model
        if field != "agents":
            agent_config = config[field]
            if not isinstance(agent_config, dict):
                logger.error(
                    f"Invalid config for {field}: expected dict, got {type(agent_config)}"
                )
                return False
            if "provider" not in agent_config or "model" not in agent_config:
                logger.error(f"Missing provider or model in {field} configuration")
                return False

    # Validate agents is a list
    if not isinstance(config["agents"], list) or len(config["agents"]) == 0:
        logger.error("Invalid agents configuration: expected non-empty list")
        return False

    return True


def migrate_config_with_validation(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a configuration with full validation.

    Args:
        old_config: The configuration dictionary (v1.1 or v1.3)

    Returns:
        A validated v1.3 configuration dictionary

    Raises:
        ValueError: If the configuration cannot be migrated or validated
    """
    # Check version
    version = detect_config_version(old_config)
    if version == "unknown":
        raise ValueError(
            f"Unknown configuration version, cannot migrate. Config: {old_config}"
        )

    # Check if migration is needed
    if not is_migration_needed(old_config):
        logger.info("Configuration is already v1.3, no migration needed")
        return old_config

    # Perform migration
    logger.info("Migrating configuration from v1.1 to v1.3")
    new_config = migrate_config_v1_to_v3(old_config)

    # Validate the result
    if not validate_migrated_config(new_config):
        raise ValueError("Migration failed: resulting configuration is invalid")

    logger.info("Configuration migration completed successfully")
    return new_config
