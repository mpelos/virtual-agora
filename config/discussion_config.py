"""Discussion flow configuration for Virtual Agora.

This module provides configuration classes for the discussion flow,
specifically for controlling user participation timing and creating
discussion round nodes with the appropriate strategies.
"""

from typing import Dict, Any, List

from virtual_agora.flow.participation_strategies import (
    ParticipationTiming,
    UserParticipationStrategy,
    create_participation_strategy,
)
from virtual_agora.flow.nodes.discussion_round import DiscussionRoundNode
from virtual_agora.flow.state_manager import FlowStateManager
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class DiscussionFlowConfig:
    """Configuration for discussion flow behavior.

    This class provides a simple interface for configuring discussion flow
    behavior, particularly user participation timing. The primary requirement
    is that switching user participation timing should require only a single
    configuration change.

    Key features:
    - Single property controls user participation timing
    - Factory method creates discussion nodes with proper strategy
    - Easy switching between timing modes
    - Backward compatible defaults
    """

    def __init__(self):
        """Initialize discussion flow configuration with defaults."""
        # EASY TO CHANGE: Single line configuration change switches timing
        # Default to current behavior (end-of-round) for backward compatibility
        self.user_participation_timing = ParticipationTiming.END_OF_ROUND

        # Alternative configurations (commented out):
        # self.user_participation_timing = ParticipationTiming.START_OF_ROUND
        # self.user_participation_timing = ParticipationTiming.DISABLED

        logger.info(
            f"Initialized DiscussionFlowConfig with timing: {self.user_participation_timing.value}"
        )

    def set_participation_timing(self, timing: ParticipationTiming) -> None:
        """Set user participation timing.

        Args:
            timing: Desired participation timing mode
        """
        logger.info(
            f"Changing participation timing from {self.user_participation_timing.value} to {timing.value}"
        )
        self.user_participation_timing = timing

    def create_discussion_node(
        self,
        flow_state_manager: FlowStateManager,
        discussing_agents: List[LLMAgent],
        specialized_agents: Dict[str, LLMAgent],
    ) -> DiscussionRoundNode:
        """Create discussion round node with configured timing strategy.

        Args:
            flow_state_manager: FlowStateManager for round state operations
            discussing_agents: List of discussion agents
            specialized_agents: Dictionary of specialized agents

        Returns:
            DiscussionRoundNode configured with the appropriate participation strategy
        """
        # Create participation strategy based on configuration
        participation_strategy = create_participation_strategy(
            self.user_participation_timing
        )

        # Create and return the discussion node
        node = DiscussionRoundNode(
            flow_state_manager=flow_state_manager,
            discussing_agents=discussing_agents,
            specialized_agents=specialized_agents,
            participation_strategy=participation_strategy,
        )

        logger.info(
            f"Created DiscussionRoundNode with {participation_strategy.get_timing_name()}"
        )
        return node

    def get_timing_description(self) -> str:
        """Get human-readable description of current timing configuration.

        Returns:
            Description of the current participation timing
        """
        descriptions = {
            ParticipationTiming.START_OF_ROUND: "User guides discussion before agents speak",
            ParticipationTiming.END_OF_ROUND: "User responds after agents complete discussion",
            ParticipationTiming.DISABLED: "No user participation during discussion rounds",
        }
        return descriptions.get(self.user_participation_timing, "Unknown timing mode")

    def is_user_participation_enabled(self) -> bool:
        """Check if user participation is enabled.

        Returns:
            True if user participation is enabled, False if disabled
        """
        return self.user_participation_timing != ParticipationTiming.DISABLED

    def is_start_of_round_participation(self) -> bool:
        """Check if user participation happens at start of round.

        Returns:
            True if participation is at start of round, False otherwise
        """
        return self.user_participation_timing == ParticipationTiming.START_OF_ROUND

    def is_end_of_round_participation(self) -> bool:
        """Check if user participation happens at end of round.

        Returns:
            True if participation is at end of round, False otherwise
        """
        return self.user_participation_timing == ParticipationTiming.END_OF_ROUND

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "user_participation_timing": self.user_participation_timing.value,
            "timing_description": self.get_timing_description(),
            "participation_enabled": self.is_user_participation_enabled(),
            "start_of_round": self.is_start_of_round_participation(),
            "end_of_round": self.is_end_of_round_participation(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DiscussionFlowConfig":
        """Create configuration from dictionary representation.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            DiscussionFlowConfig instance with the specified configuration

        Raises:
            ValueError: If timing value is not recognized
        """
        config = cls()

        timing_value = config_dict.get("user_participation_timing")
        if timing_value:
            try:
                timing = ParticipationTiming(timing_value)
                config.set_participation_timing(timing)
            except ValueError:
                raise ValueError(f"Unknown participation timing: {timing_value}")

        return config

    def __str__(self) -> str:
        """String representation of the configuration.

        Returns:
            Human-readable string describing the configuration
        """
        return f"DiscussionFlowConfig(timing={self.user_participation_timing.value})"

    def __repr__(self) -> str:
        """Developer representation of the configuration.

        Returns:
            Detailed string representation for debugging
        """
        return f"DiscussionFlowConfig(user_participation_timing={self.user_participation_timing!r})"


# Convenience functions for common configuration patterns


def create_start_of_round_config() -> DiscussionFlowConfig:
    """Create configuration for start-of-round user participation.

    Returns:
        DiscussionFlowConfig configured for start-of-round participation
    """
    config = DiscussionFlowConfig()
    config.set_participation_timing(ParticipationTiming.START_OF_ROUND)
    return config


def create_end_of_round_config() -> DiscussionFlowConfig:
    """Create configuration for end-of-round user participation.

    Returns:
        DiscussionFlowConfig configured for end-of-round participation
    """
    config = DiscussionFlowConfig()
    config.set_participation_timing(ParticipationTiming.END_OF_ROUND)
    return config


def create_disabled_participation_config() -> DiscussionFlowConfig:
    """Create configuration with disabled user participation.

    Returns:
        DiscussionFlowConfig configured with disabled participation
    """
    config = DiscussionFlowConfig()
    config.set_participation_timing(ParticipationTiming.DISABLED)
    return config


# Example usage demonstrating the single configuration change requirement
def example_configuration_switching():
    """Example demonstrating how easy it is to switch participation timing."""

    # Create base configuration (defaults to end-of-round for backward compatibility)
    config = DiscussionFlowConfig()
    print(f"Default: {config}")

    # SINGLE LINE CHANGE to switch to start-of-round participation
    config.user_participation_timing = ParticipationTiming.START_OF_ROUND
    print(f"Start of round: {config}")

    # SINGLE LINE CHANGE to disable participation
    config.user_participation_timing = ParticipationTiming.DISABLED
    print(f"Disabled: {config}")

    # SINGLE LINE CHANGE to return to end-of-round
    config.user_participation_timing = ParticipationTiming.END_OF_ROUND
    print(f"End of round: {config}")


if __name__ == "__main__":
    # Run example if script is executed directly
    example_configuration_switching()
