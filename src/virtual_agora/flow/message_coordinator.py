"""Message coordination abstraction for Virtual Agora.

This module provides centralized message assembly and routing functionality,
replacing scattered message coordination logic throughout the codebase.
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..state.schema import VirtualAgoraState
from ..flow.round_manager import RoundManager
from ..context.builders import get_context_builder
from ..context.message_processor import MessageProcessor, ProcessedMessage
from ..context.repository import ContextRepository

logger = logging.getLogger(__name__)


class MessageCoordinator:
    """Centralized coordinator for message assembly and routing.

    This class provides a single source of truth for message coordination,
    replacing the scattered message assembly logic that previously existed
    across multiple files.

    Key responsibilities:
    - Agent context assembly with proper round awareness
    - User message storage with consistent round numbering
    - Message retrieval with format standardization
    - Integration with existing context infrastructure
    """

    def __init__(self, round_manager: RoundManager):
        """Initialize MessageCoordinator with required dependencies.

        Args:
            round_manager: RoundManager instance for consistent round numbering
        """
        self.round_manager = round_manager
        self._context_builders = {}  # Cache for context builders

    def assemble_agent_context(
        self, agent_id: str, round_num: int, state: VirtualAgoraState, **kwargs
    ) -> Tuple[str, List[BaseMessage]]:
        """Assemble complete context for agent with centralized logic.

        This method replaces the scattered context assembly logic found in
        nodes_v13.py:1362-1400, providing a centralized interface for
        agent context building.

        Args:
            agent_id: Identifier for the agent
            round_num: Current round number
            state: Current state of the Virtual Agora session
            **kwargs: Additional context parameters:
                - topic: Current topic (optional)
                - agent_position: Agent's position in speaking order
                - speaking_order: Current speaking order
                - system_prompt: Agent's system prompt
                - current_round_messages: Messages from current round (optional)

        Returns:
            Tuple of (context_text, context_messages) for agent

        Note:
            This replaces the direct context builder calls and provides
            consistent parameter management for context building.
        """
        # Extract parameters with defaults
        topic = kwargs.get("topic") or state.get("active_topic", "Unknown Topic")
        agent_position = kwargs.get("agent_position", 0)
        speaking_order = kwargs.get("speaking_order") or state.get(
            "speaking_order", [agent_id]
        )
        system_prompt = kwargs.get("system_prompt", "")
        current_round_messages = kwargs.get("current_round_messages", [])

        logger.debug(
            f"Assembling context for agent {agent_id}, round {round_num}, topic '{topic}'"
        )

        # Get or cache context builder for discussion rounds
        if "discussion_round" not in self._context_builders:
            self._context_builders["discussion_round"] = get_context_builder(
                "discussion_round"
            )

        context_builder = self._context_builders["discussion_round"]

        # Build context using existing infrastructure
        context_data = context_builder.build_context(
            state=state,
            system_prompt=system_prompt,
            agent_id=agent_id,
            current_round=round_num,
            agent_position=agent_position,
            speaking_order=speaking_order,
            topic=topic,
            current_round_messages=current_round_messages,
        )

        # Extract formatted context messages
        context_messages = context_data.formatted_context_messages or []

        # Build enhanced context text with consistent formatting
        context_parts = []

        # Add theme if available
        if context_data.has_user_input:
            context_parts.append(f"Theme: {context_data.user_input}")

        # Add current topic and round info
        context_parts.append(f"Current Topic: {topic}")
        context_parts.append(f"Round: {round_num}")
        context_parts.append(
            f"Your speaking position: {agent_position + 1}/{len(speaking_order)}"
        )

        # Add context documents for complex topics
        if context_data.has_context_documents:
            context_parts.append(
                f"\nBackground Information:\n{context_data.context_documents}"
            )

        # Add previous topic conclusions
        previous_topic_summaries = state.get("topic_summaries", {})
        topic_conclusions = []
        for key, summary in previous_topic_summaries.items():
            if key.endswith("_conclusion") and not key.startswith(topic):
                topic_name = key.replace("_conclusion", "")
                topic_conclusions.append(f"{topic_name}: {summary}")

        if topic_conclusions:
            context_parts.append("\nPreviously Concluded Topics:")
            for conclusion in topic_conclusions:
                context_parts.append(f"- {conclusion}")

        # Add round summaries from previous rounds
        if context_data.has_round_summaries:
            context_parts.append("\nPrevious Round Summaries:")
            for summary in context_data.round_summaries[-3:]:  # Last 3 summaries
                summary_text = (
                    summary.get("summary_text", "")
                    if isinstance(summary, dict)
                    else getattr(summary, "summary_text", "")
                )
                if summary_text:
                    context_parts.append(f"- {summary_text}")

        # Combine context parts
        context_text = "\n".join(context_parts)

        logger.info(
            f"Assembled context for agent {agent_id}: {len(context_text)} chars, {len(context_messages)} context messages"
        )
        logger.debug(
            f"Context message types: {[type(msg).__name__ for msg in context_messages]}"
        )

        if context_messages:
            logger.debug(
                f"First context message: {context_messages[0].content[:100] if hasattr(context_messages[0], 'content') else str(context_messages[0])[:100]}..."
            )

        return context_text, context_messages

    def store_user_message(
        self, content: str, round_num: int, state: VirtualAgoraState, **kwargs
    ) -> Dict[str, Any]:
        """Store user participation message with correct round numbering.

        This method replaces the manual user message creation logic found in
        nodes_v13.py:3220-3245, providing consistent round numbering and
        metadata handling.

        Args:
            content: User message content
            round_num: Current round number
            state: Current state of the Virtual Agora session
            **kwargs: Additional parameters:
                - topic: Current topic (optional)
                - participation_type: Type of participation (default: "user_turn_participation")
                - use_next_round: Whether to use next round number (default: True)

        Returns:
            Dictionary with state updates including the user message

        Note:
            This replaces manual round calculation and message creation,
            coordinating with RoundManager for consistent round numbering.
        """
        # Extract parameters
        topic = kwargs.get("topic") or state.get("active_topic") or "Unknown Topic"
        participation_type = kwargs.get("participation_type", "user_turn_participation")
        use_next_round = kwargs.get("use_next_round", True)

        # Calculate target round number
        if use_next_round:
            # User participation typically applies to the next round
            target_round = round_num + 1
        else:
            # Use current round
            target_round = round_num

        logger.info(f"Storing user message for round {target_round}: {content[:50]}...")

        # Create user message with proper metadata
        user_msg = HumanMessage(
            content=content,
            additional_kwargs={
                "speaker_id": "user",
                "speaker_role": "user",
                "round": target_round,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "participation_type": participation_type,
            },
        )

        logger.debug(
            f"Created user message with metadata: {user_msg.additional_kwargs}"
        )

        # Return state updates
        updates = {
            "messages": [user_msg],  # Uses add_messages reducer
            "user_participation_message": content,
        }

        logger.debug(
            f"Created user message for round {target_round} with metadata: {user_msg.additional_kwargs}"
        )

        return updates

    def get_messages_for_round(
        self, round_num: int, state: VirtualAgoraState, topic: Optional[str] = None
    ) -> List[ProcessedMessage]:
        """Get all messages for specific round with consistent filtering.

        Args:
            round_num: Round number to filter by
            state: Current state of the Virtual Agora session
            topic: Topic to filter by (optional)

        Returns:
            List of ProcessedMessage objects for the specified round

        Note:
            This provides centralized message retrieval with format
            standardization using the existing MessageProcessor.
        """
        all_messages = state.get("messages", [])
        round_messages = []

        for msg in all_messages:
            # Standardize message format
            processed_msg = MessageProcessor.standardize_message(msg)

            # Filter by round
            if processed_msg.round_number == round_num:
                # Filter by topic if specified
                if topic is None or processed_msg.topic == topic:
                    round_messages.append(processed_msg)

        logger.debug(f"Found {len(round_messages)} messages for round {round_num}")
        return round_messages

    def get_user_messages_for_round(
        self, round_num: int, state: VirtualAgoraState, topic: Optional[str] = None
    ) -> List[ProcessedMessage]:
        """Get user messages for specific round.

        Args:
            round_num: Round number to filter by
            state: Current state of the Virtual Agora session
            topic: Topic to filter by (optional)

        Returns:
            List of ProcessedMessage objects from user for the specified round
        """
        round_messages = self.get_messages_for_round(round_num, state, topic)
        user_messages = [
            msg
            for msg in round_messages
            if msg.speaker_role == "user" or msg.speaker_id == "user"
        ]

        logger.debug(f"Found {len(user_messages)} user messages for round {round_num}")
        return user_messages

    def get_agent_messages_for_round(
        self,
        round_num: int,
        state: VirtualAgoraState,
        topic: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[ProcessedMessage]:
        """Get agent messages for specific round.

        Args:
            round_num: Round number to filter by
            state: Current state of the Virtual Agora session
            topic: Topic to filter by (optional)
            agent_id: Specific agent to filter by (optional)

        Returns:
            List of ProcessedMessage objects from agents for the specified round
        """
        round_messages = self.get_messages_for_round(round_num, state, topic)
        agent_messages = [
            msg
            for msg in round_messages
            if msg.speaker_role != "user" and msg.speaker_id != "user"
        ]

        # Filter by specific agent if requested
        if agent_id:
            agent_messages = [
                msg for msg in agent_messages if msg.speaker_id == agent_id
            ]

        logger.debug(
            f"Found {len(agent_messages)} agent messages for round {round_num}"
        )
        return agent_messages

    def clear_context_cache(self):
        """Clear cached context builders.

        This can be used for testing or when context builders need to be refreshed.
        """
        self._context_builders.clear()
        logger.debug("Cleared context builder cache")
