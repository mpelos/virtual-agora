"""Flow State Manager for Virtual Agora.

This module provides centralized state transition management, establishing clear
boundaries between state preparation, user participation application, and round
finalization. It builds on RoundManager and MessageCoordinator to provide a
unified interface for all flow state operations.
"""

from typing import Dict, Any, List, Optional, NamedTuple
from datetime import datetime
import logging
import uuid
from copy import deepcopy

from ..state.schema import VirtualAgoraState
from .round_manager import RoundManager
from .message_coordinator import MessageCoordinator

logger = logging.getLogger(__name__)


class RoundState(NamedTuple):
    """Prepared state information for a discussion round."""

    round_number: int
    current_topic: str
    speaking_order: List[str]
    round_id: str
    round_start_time: datetime
    theme: str


class FlowStateManager:
    """Manages discussion flow state and transitions.

    This class provides centralized state management by coordinating with
    RoundManager and MessageCoordinator to handle state transitions in a
    consistent and predictable manner.

    Key responsibilities:
    - Round state preparation with speaking order management
    - User participation application with proper coordination
    - Round finalization with state cleanup
    - Immutable state operations that return new state
    """

    def __init__(
        self, round_manager: RoundManager, message_coordinator: MessageCoordinator
    ):
        """Initialize FlowStateManager with required dependencies.

        Args:
            round_manager: RoundManager instance for round state management
            message_coordinator: MessageCoordinator instance for message operations
        """
        self.round_manager = round_manager
        self.message_coordinator = message_coordinator

    def prepare_round_state(self, state: VirtualAgoraState) -> RoundState:
        """Prepare state for new discussion round.

        This method centralizes the round preparation logic that was previously
        scattered in nodes_v13.py:1290-1330, providing consistent round setup
        across all discussion flows.

        Args:
            state: Current state of the Virtual Agora session

        Returns:
            RoundState object with prepared round information

        Raises:
            ValueError: If no active topic is set for the round

        Note:
            This replaces the manual round setup logic and speaking order
            rotation that was previously duplicated across multiple nodes.
        """
        # Validate required state
        current_topic = state.get("active_topic")
        if not current_topic:
            raise ValueError("No active topic set for discussion round")

        # Prepare round number using RoundManager
        round_number = self.round_manager.start_new_round(state)

        # Get theme for context
        theme = state.get("main_topic") or "Unknown Topic"

        # Prepare speaking order with rotation
        speaking_order = self._prepare_speaking_order(state, round_number)

        # Generate round metadata
        round_id = str(uuid.uuid4())
        round_start_time = datetime.now()

        logger.info(
            f"Prepared round {round_number} for topic '{current_topic}' with order: {speaking_order}"
        )

        return RoundState(
            round_number=round_number,
            current_topic=current_topic,
            speaking_order=speaking_order,
            round_id=round_id,
            round_start_time=round_start_time,
            theme=theme,
        )

    def _prepare_speaking_order(
        self, state: VirtualAgoraState, round_number: int
    ) -> List[str]:
        """Prepare speaking order for the round with proper rotation.

        This method centralizes the speaking order logic from nodes_v13.py:1317-1330,
        handling both initial order creation and rotation between rounds.

        Args:
            state: Current state of the Virtual Agora session
            round_number: Current round number

        Returns:
            List of agent IDs in speaking order for this round
        """
        speaking_order = state.get("speaking_order", [])

        if not speaking_order:
            # Initialize speaking order if not set - this should come from agents
            # For now, we'll use the order from the state or empty list
            # The calling code should ensure agents are properly configured
            logger.warning(
                "No speaking order found in state - this should be initialized by the flow"
            )
            return []

        # Create a copy to avoid mutating the original state
        speaking_order = speaking_order.copy()

        if round_number > 1:
            # Rotate speaking order: [A,B,C] -> [B,C,A]
            speaking_order = speaking_order[1:] + [speaking_order[0]]
            logger.debug(
                f"Rotated speaking order for round {round_number}: {speaking_order}"
            )

        return speaking_order

    def apply_user_participation(
        self, user_input: str, state: VirtualAgoraState, **kwargs
    ) -> Dict[str, Any]:
        """Apply user participation to current state.

        This method centralizes the user participation logic that was previously
        scattered in nodes_v13.py:3172-3185, providing consistent user input
        handling across all flow scenarios.

        Args:
            user_input: User's participation message
            state: Current state of the Virtual Agora session
            **kwargs: Additional parameters:
                - current_round: Round number (optional, will use state if not provided)
                - topic: Current topic (optional, will use active_topic if not provided)
                - participation_type: Type of participation (default: "user_turn_participation")
                - use_next_round: Whether to apply to next round (default: True)

        Returns:
            Dictionary with state updates for user participation

        Note:
            This replaces the manual user message creation and state update
            logic that was previously duplicated across multiple nodes.
        """
        # Extract parameters with defaults
        current_round = kwargs.get(
            "current_round"
        ) or self.round_manager.get_current_round(state)
        current_topic = (
            kwargs.get("topic") or state.get("active_topic") or "Unknown Topic"
        )
        participation_type = kwargs.get("participation_type", "user_turn_participation")
        use_next_round = kwargs.get("use_next_round", True)

        logger.info(
            f"Applying user participation for round {current_round}: {user_input[:50]}..."
        )

        # Use MessageCoordinator to store user message with consistent round numbering
        user_message_updates = self.message_coordinator.store_user_message(
            content=user_input,
            round_num=current_round,
            state=state,
            topic=current_topic,
            participation_type=participation_type,
            use_next_round=use_next_round,
        )

        logger.debug(
            f"User participation applied with updates: {list(user_message_updates.keys())}"
        )

        return user_message_updates

    def finalize_round(
        self,
        state: VirtualAgoraState,
        round_state: RoundState,
        round_messages: List[Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Clean up and finalize round state.

        This method centralizes the round finalization logic that was previously
        scattered in nodes_v13.py:1600-1620, providing consistent round cleanup
        and state updates.

        Args:
            state: Current state of the Virtual Agora session
            round_state: The prepared round state from prepare_round_state()
            round_messages: Messages generated during the round
            **kwargs: Additional parameters for customization

        Returns:
            Dictionary with state updates for round finalization

        Note:
            This replaces the manual round info creation and counter update
            logic that was previously duplicated across multiple nodes.
        """
        current_topic = round_state.current_topic
        round_number = round_state.round_number
        speaking_order = round_state.speaking_order

        logger.info(
            f"Finalizing round {round_number} with {len(round_messages)} messages"
        )

        # Create round info with metadata
        round_info = {
            "round_id": round_state.round_id,
            "round_number": round_number,
            "topic": current_topic,
            "start_time": round_state.round_start_time,
            "end_time": datetime.now(),
            "participants": self._extract_participants(round_messages),
            "message_count": len(round_messages),
            "summary": None,  # Will be filled by summarization node
        }

        # Update rounds per topic counter
        rounds_per_topic = {
            **state.get("rounds_per_topic", {}),
            current_topic: state.get("rounds_per_topic", {}).get(current_topic, 0) + 1,
        }

        # Prepare state updates
        updates = {
            "current_round": round_number,
            "speaking_order": speaking_order,
            "messages": round_messages,  # Uses add_messages reducer (expects list)
            "round_history": round_info,  # Uses list.append reducer (expects single item)
            "turn_order_history": speaking_order,  # Uses list.append reducer (expects single item)
            "rounds_per_topic": rounds_per_topic,
        }

        logger.debug(
            f"Round {round_number} finalized with {len(updates)} state updates"
        )

        return updates

    def _extract_participants(self, round_messages: List[Any]) -> List[str]:
        """Extract participant IDs from round messages.

        Args:
            round_messages: Messages generated during the round

        Returns:
            List of participant IDs
        """
        participants = []

        for msg in round_messages:
            speaker_id = None
            speaker_role = None

            # Handle both LangChain messages and dict messages
            if hasattr(msg, "additional_kwargs"):
                speaker_id = msg.additional_kwargs.get("speaker_id", "unknown")
                speaker_role = msg.additional_kwargs.get("speaker_role", "participant")
            elif isinstance(msg, dict):
                speaker_id = msg.get("speaker_id", "unknown")
                speaker_role = msg.get("speaker_role", "participant")

            # Only include participants (not user messages)
            if speaker_role == "participant" and speaker_id not in participants:
                participants.append(speaker_id)

        return participants

    def create_immutable_state_update(
        self, state: VirtualAgoraState, updates: Dict[str, Any]
    ) -> VirtualAgoraState:
        """Create immutable state update by merging updates with existing state.

        This method ensures state operations are immutable by creating a new
        state dictionary rather than modifying the input state.

        Args:
            state: Current state of the Virtual Agora session
            updates: Updates to apply to the state

        Returns:
            New state dictionary with updates applied

        Note:
            This method is used internally to ensure immutable state operations
            as required by the Step 1.3 acceptance criteria.
        """
        # Create a deep copy to ensure immutability
        new_state = deepcopy(state)

        # Apply updates
        for key, value in updates.items():
            new_state[key] = value

        return new_state
