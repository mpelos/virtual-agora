"""Round management abstraction for Virtual Agora.

This module provides centralized round state and transition management,
replacing scattered round management logic throughout the codebase.
"""

from typing import Dict, Any
import logging

from ..state.schema import VirtualAgoraState

logger = logging.getLogger(__name__)


class RoundManager:
    """Centralized round state and transition management.

    This class provides a single source of truth for round numbers and
    related operations, replacing the scattered round management logic
    that previously existed across multiple files.

    Key responsibilities:
    - Get current round number with consistent fallback logic
    - Calculate new round numbers for transitions
    - Validate preconditions for round operations
    - Provide round-specific metadata and context
    """

    def get_current_round(self, state: VirtualAgoraState) -> int:
        """Get current round number with consistent logic.

        Args:
            state: Current state of the Virtual Agora session

        Returns:
            Current round number (0 if not set or None)

        Note:
            This replaces the scattered `state.get("current_round", 0)`
            pattern used throughout the codebase. Explicitly handles None values.
        """
        current_round = state.get("current_round", 0)
        return current_round if current_round is not None else 0

    def start_new_round(self, state: VirtualAgoraState) -> int:
        """Increment and return new round number.

        Args:
            state: Current state of the Virtual Agora session

        Returns:
            New round number (current + 1)

        Note:
            This replaces the scattered `state.get("current_round", 0) + 1`
            pattern used in discussion nodes.
        """
        current = self.get_current_round(state)
        new_round = current + 1
        logger.debug(f"Starting new round: {current} -> {new_round}")
        return new_round

    def can_start_round(self, state: VirtualAgoraState) -> bool:
        """Determine if conditions allow starting new round.

        Args:
            state: Current state of the Virtual Agora session

        Returns:
            True if round can be started, False otherwise

        Note:
            Currently checks for active topic. Can be extended for
            additional preconditions as needed.
        """
        has_active_topic = state.get("active_topic") is not None

        if not has_active_topic:
            logger.warning("Cannot start round: no active topic")

        return has_active_topic

    def get_round_metadata(
        self, state: VirtualAgoraState, round_num: int
    ) -> Dict[str, Any]:
        """Get round-specific metadata and context.

        Args:
            state: Current state of the Virtual Agora session
            round_num: Round number to get metadata for

        Returns:
            Dictionary containing round metadata including:
            - round_number: The round number
            - topic: Active topic for this round
            - phase: Current phase of discussion
            - is_threshold_round: Whether this round triggers polling (>= 3)
            - is_checkpoint_round: Whether this triggers periodic stop
        """
        active_topic = state.get("active_topic", "Unknown Topic")
        current_phase = state.get("current_phase", 0)
        checkpoint_interval = state.get("checkpoint_interval", 3)

        metadata = {
            "round_number": round_num,
            "topic": active_topic,
            "phase": current_phase,
            "is_threshold_round": round_num >= 3,
            "is_checkpoint_round": (
                round_num > 0
                and checkpoint_interval > 0
                and round_num % checkpoint_interval == 0
            ),
        }

        logger.debug(f"Round {round_num} metadata: {metadata}")
        return metadata

    def should_trigger_polling(self, state: VirtualAgoraState) -> bool:
        """Check if current round should trigger topic conclusion polling.

        Args:
            state: Current state of the Virtual Agora session

        Returns:
            True if polling should be triggered (round >= 3)

        Note:
            This centralizes the threshold logic previously in edges_v13.py
        """
        current_round = self.get_current_round(state)
        should_poll = current_round >= 3

        if should_poll:
            logger.debug(f"Round {current_round} >= 3, triggering polling")
        else:
            logger.debug(f"Round {current_round} < 3, no polling")

        return should_poll

    def should_trigger_checkpoint(self, state: VirtualAgoraState) -> bool:
        """Check if current round should trigger periodic user stop.

        Args:
            state: Current state of the Virtual Agora session

        Returns:
            True if checkpoint should be triggered

        Note:
            This centralizes the checkpoint logic from various files.
        """
        current_round = self.get_current_round(state)
        checkpoint_interval = state.get("checkpoint_interval", 3)

        should_checkpoint = (
            current_round > 0
            and checkpoint_interval > 0
            and current_round % checkpoint_interval == 0
        )

        if should_checkpoint:
            logger.debug(
                f"Round {current_round} triggers checkpoint (interval: {checkpoint_interval})"
            )
        else:
            logger.debug(
                f"Round {current_round} no checkpoint (interval: {checkpoint_interval})"
            )

        return should_checkpoint
