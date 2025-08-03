"""User participation timing strategies for Virtual Agora discussion flow.

This module implements the strategy pattern for configurable user participation
timing in discussion rounds. It provides three timing strategies:
- StartOfRoundParticipation: User guides before agents speak
- EndOfRoundParticipation: User responds after agents speak (current behavior)
- DisabledParticipation: No user participation

The strategy pattern enables easy switching between participation timing modes
without requiring changes to the graph structure or discussion flow logic.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ParticipationTiming(Enum):
    """When user participation should occur in discussion flow."""

    START_OF_ROUND = "start_of_round"  # User input before agents speak
    END_OF_ROUND = "end_of_round"  # User input after agents speak
    DISABLED = "disabled"  # No user participation


class UserParticipationStrategy(ABC):
    """Strategy pattern for user participation timing.

    This abstract base class defines the interface for different user participation
    timing strategies. Each strategy determines when and how user participation
    should be requested during discussion rounds.

    Key responsibilities:
    - Determine if participation needed before agents speak
    - Determine if participation needed after agents speak
    - Generate appropriate context for user participation prompts
    """

    @abstractmethod
    def should_request_participation_before_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Check if user participation needed before agents speak.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if user participation should be requested before agents, False otherwise
        """
        pass

    @abstractmethod
    def should_request_participation_after_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Check if user participation needed after agents speak.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if user participation should be requested after agents, False otherwise
        """
        pass

    @abstractmethod
    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Get context for user participation prompt.

        Args:
            state: Current Virtual Agora state

        Returns:
            Dictionary containing context information for the participation prompt
        """
        pass

    def get_timing_name(self) -> str:
        """Get human-readable name for this timing strategy.

        Returns:
            String name of the timing strategy
        """
        return self.__class__.__name__


class StartOfRoundParticipation(UserParticipationStrategy):
    """User participates at the beginning of each round.

    This strategy allows users to provide guidance and direction before agents
    begin speaking. User input influences the discussion context for the
    current round rather than the next round.

    Key characteristics:
    - User participation happens before agents speak
    - User messages are included in agent context for the same round
    - Enables proactive guidance rather than reactive feedback
    - Users can set discussion direction and priorities
    """

    def should_request_participation_before_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Request participation before agents from round 1 onwards.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if round >= 1, False otherwise
        """
        current_round = state.get("current_round", 0)
        should_participate = current_round >= 1

        logger.debug(
            f"StartOfRound strategy: round {current_round}, participate before: {should_participate}"
        )
        return should_participate

    def should_request_participation_after_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Never request participation after agents in start-of-round mode.

        Args:
            state: Current Virtual Agora state

        Returns:
            Always False
        """
        return False

    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Get context for start-of-round participation.

        Args:
            state: Current Virtual Agora state

        Returns:
            Context dictionary with round start messaging and options
        """
        current_round = state.get("current_round", 0)
        current_topic = state.get("active_topic", "Unknown Topic")

        return {
            "timing": "round_start",
            "message": (
                f"Round {current_round} is about to begin.\n"
                f"Topic: {current_topic}\n\n"
                "You can provide guidance to shape the upcoming discussion.\n"
                "Your input will be available to all agents in this round."
            ),
            "show_previous_summary": True,
            "participation_type": "proactive_guidance",
            "round_phase": "pre_discussion",
        }


class EndOfRoundParticipation(UserParticipationStrategy):
    """User participates at the end of each round (current behavior).

    This strategy maintains the current system behavior where users respond
    to agent discussions after they complete. User input influences the
    context for the next round.

    Key characteristics:
    - User participation happens after agents speak
    - User messages are included in agent context for the next round
    - Enables reactive feedback to agent discussions
    - Users can redirect or conclude based on agent output
    """

    def should_request_participation_before_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Never request participation before agents in end-of-round mode.

        Args:
            state: Current Virtual Agora state

        Returns:
            Always False
        """
        return False

    def should_request_participation_after_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Request participation after agents from round 1 onwards.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if round >= 1, False otherwise
        """
        current_round = state.get("current_round", 0)
        should_participate = current_round >= 1

        logger.debug(
            f"EndOfRound strategy: round {current_round}, participate after: {should_participate}"
        )
        return should_participate

    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Get context for end-of-round participation.

        Args:
            state: Current Virtual Agora state

        Returns:
            Context dictionary with round end messaging and options
        """
        current_round = state.get("current_round", 0)
        current_topic = state.get("active_topic", "Unknown Topic")

        return {
            "timing": "round_end",
            "message": (
                f"Round {current_round} has completed.\n"
                f"Topic: {current_topic}\n\n"
                "You can respond to the discussion and guide the next steps.\n"
                "What would you like to do next?"
            ),
            "show_round_summary": True,
            "participation_type": "reactive_feedback",
            "round_phase": "post_discussion",
        }


class DisabledParticipation(UserParticipationStrategy):
    """No user participation during discussion rounds.

    This strategy completely disables user participation during discussion
    rounds, allowing agents to proceed autonomously without user input.

    Key characteristics:
    - No user participation requests at any point
    - Agents proceed without user guidance
    - Useful for automated or batch processing scenarios
    - Can be used for testing agent-only discussions
    """

    def should_request_participation_before_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Never request participation before agents.

        Args:
            state: Current Virtual Agora state

        Returns:
            Always False
        """
        return False

    def should_request_participation_after_agents(
        self, state: VirtualAgoraState
    ) -> bool:
        """Never request participation after agents.

        Args:
            state: Current Virtual Agora state

        Returns:
            Always False
        """
        return False

    def get_participation_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Get context for disabled participation (should not be called).

        Args:
            state: Current Virtual Agora state

        Returns:
            Empty context dictionary

        Note:
            This method should not be called since participation is disabled,
            but is implemented for interface compliance.
        """
        logger.warning(
            "get_participation_context called on DisabledParticipation strategy"
        )
        return {
            "timing": "disabled",
            "message": "User participation is disabled",
            "participation_type": "disabled",
            "round_phase": "none",
        }


def create_participation_strategy(
    timing: ParticipationTiming,
) -> UserParticipationStrategy:
    """Factory function to create participation strategy instances.

    Args:
        timing: Desired participation timing mode

    Returns:
        Appropriate strategy instance for the timing mode

    Raises:
        ValueError: If timing mode is not recognized
    """
    strategies = {
        ParticipationTiming.START_OF_ROUND: StartOfRoundParticipation(),
        ParticipationTiming.END_OF_ROUND: EndOfRoundParticipation(),
        ParticipationTiming.DISABLED: DisabledParticipation(),
    }

    if timing not in strategies:
        raise ValueError(f"Unknown participation timing: {timing}")

    strategy = strategies[timing]
    logger.info(f"Created participation strategy: {strategy.get_timing_name()}")
    return strategy


def get_available_timings() -> Dict[ParticipationTiming, str]:
    """Get available participation timing modes with descriptions.

    Returns:
        Dictionary mapping timing modes to human-readable descriptions
    """
    return {
        ParticipationTiming.START_OF_ROUND: "User guides discussion before agents speak",
        ParticipationTiming.END_OF_ROUND: "User responds after agents complete discussion",
        ParticipationTiming.DISABLED: "No user participation during discussion rounds",
    }
