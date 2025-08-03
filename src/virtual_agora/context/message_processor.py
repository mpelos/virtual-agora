"""Message processing utilities for consistent context assembly.

This module provides centralized message processing functionality to handle
format conversions between Virtual Agora dict format and LangChain BaseMessage
objects, ensuring consistent context assembly across the system.
"""

from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from virtual_agora.state.schema import Message, VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedMessage:
    """Standardized message container for context assembly."""

    speaker_id: str
    speaker_role: str
    content: str
    round_number: int
    topic: str
    timestamp: datetime
    metadata: Dict[str, Any]
    original_format: str  # 'dict' or 'langchain'


class MessageProcessor:
    """Centralized processor for message format conversions and standardization.

    This class handles the complexity of working with different message formats
    in the Virtual Agora system and ensures consistent context assembly.
    """

    @staticmethod
    def extract_message_info(
        msg: Union[Dict, BaseMessage],
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """Extract standardized information from any message format.

        Args:
            msg: Message in either Virtual Agora dict or LangChain BaseMessage format

        Returns:
            Tuple of (speaker_id, speaker_role, content, metadata)
        """
        if isinstance(msg, dict):
            # Virtual Agora dict format
            speaker_id = msg.get("speaker_id", "unknown")
            speaker_role = msg.get("speaker_role", "participant")
            content = msg.get("content", "")
            metadata = {
                k: v
                for k, v in msg.items()
                if k not in ["speaker_id", "speaker_role", "content"]
            }
        elif hasattr(msg, "content"):
            # LangChain BaseMessage format
            additional_kwargs = getattr(msg, "additional_kwargs", {})
            speaker_id = additional_kwargs.get(
                "speaker_id", getattr(msg, "name", "unknown")
            )
            speaker_role = additional_kwargs.get("speaker_role", "participant")
            content = msg.content
            metadata = additional_kwargs.copy()
        else:
            logger.warning(f"Unknown message format: {type(msg)}")
            speaker_id = "unknown"
            speaker_role = "unknown"
            content = str(msg)
            metadata = {}

        return speaker_id, speaker_role, content, metadata

    @staticmethod
    def standardize_message(msg: Union[Dict, BaseMessage]) -> ProcessedMessage:
        """Convert any message format to standardized ProcessedMessage.

        Args:
            msg: Message in any format

        Returns:
            ProcessedMessage with standardized fields
        """
        speaker_id, speaker_role, content, metadata = (
            MessageProcessor.extract_message_info(msg)
        )

        # Extract common fields with defaults
        round_number = metadata.get("round", 0)
        topic = metadata.get("topic", "")

        # Handle timestamp conversion
        timestamp_raw = metadata.get("timestamp")
        if isinstance(timestamp_raw, datetime):
            timestamp = timestamp_raw
        elif isinstance(timestamp_raw, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Determine original format
        original_format = "dict" if isinstance(msg, dict) else "langchain"

        return ProcessedMessage(
            speaker_id=speaker_id,
            speaker_role=speaker_role,
            content=content,
            round_number=round_number,
            topic=topic,
            timestamp=timestamp,
            metadata=metadata,
            original_format=original_format,
        )

    @staticmethod
    def to_langchain_message(
        processed_msg: ProcessedMessage, force_role: Optional[str] = None
    ) -> BaseMessage:
        """Convert ProcessedMessage to appropriate LangChain BaseMessage.

        Args:
            processed_msg: Standardized message
            force_role: Force specific message type ('human', 'ai', 'system')

        Returns:
            Appropriate BaseMessage subclass
        """
        # Determine message type
        if force_role:
            message_type = force_role.lower()
        elif processed_msg.speaker_role == "user":
            message_type = "human"
        elif processed_msg.speaker_role in ["participant", "moderator"]:
            message_type = "ai"
        else:
            message_type = "ai"  # Default to AI message

        # Prepare additional kwargs with full metadata
        additional_kwargs = processed_msg.metadata.copy()
        additional_kwargs.update(
            {
                "speaker_id": processed_msg.speaker_id,
                "speaker_role": processed_msg.speaker_role,
                "round": processed_msg.round_number,
                "topic": processed_msg.topic,
                "timestamp": processed_msg.timestamp.isoformat(),
            }
        )

        # Create appropriate message type
        if message_type == "human":
            return HumanMessage(
                content=processed_msg.content,
                name=processed_msg.speaker_id,
                additional_kwargs=additional_kwargs,
            )
        elif message_type == "system":
            return SystemMessage(
                content=processed_msg.content, additional_kwargs=additional_kwargs
            )
        else:  # ai message
            return AIMessage(
                content=processed_msg.content,
                name=processed_msg.speaker_id,
                additional_kwargs=additional_kwargs,
            )

    @staticmethod
    def filter_messages_by_round(
        messages: List[Union[Dict, BaseMessage]],
        target_round: int,
        target_topic: Optional[str] = None,
    ) -> List[ProcessedMessage]:
        """Filter messages by round number and optionally topic.

        Args:
            messages: List of messages in any format
            target_round: Round number to filter by
            target_topic: Optional topic to filter by

        Returns:
            List of ProcessedMessage objects matching criteria
        """
        filtered = []

        for msg in messages:
            processed = MessageProcessor.standardize_message(msg)

            # Check round match
            if processed.round_number != target_round:
                continue

            # Check topic match if specified
            if target_topic and processed.topic != target_topic:
                continue

            filtered.append(processed)

        logger.debug(
            f"Filtered {len(filtered)} messages for round {target_round}, topic '{target_topic}'"
        )
        return filtered

    @staticmethod
    def filter_user_participation_messages(
        messages: List[Union[Dict, BaseMessage]],
        target_topic: str,
        exclude_round: Optional[int] = None,
    ) -> List[ProcessedMessage]:
        """Filter user participation messages for a specific topic.

        Args:
            messages: List of messages in any format
            target_topic: Topic to filter by
            exclude_round: Optional round to exclude (e.g., current round)

        Returns:
            List of ProcessedMessage objects for user participation
        """
        filtered = []

        for msg in messages:
            processed = MessageProcessor.standardize_message(msg)

            # Check if it's a user participation message
            if (
                processed.speaker_id == "user"
                and processed.topic == target_topic
                and processed.metadata.get("participation_type")
                == "user_turn_participation"
            ):

                # Exclude specific round if requested
                if exclude_round and processed.round_number == exclude_round:
                    continue

                filtered.append(processed)

        # Sort by round number for chronological order
        filtered.sort(key=lambda x: x.round_number)

        logger.debug(
            f"Found {len(filtered)} user participation messages for topic '{target_topic}'"
        )
        return filtered

    @staticmethod
    def create_context_messages_for_agent(
        round_messages: List[ProcessedMessage],
        user_messages: List[ProcessedMessage],
        agent_id: str,
    ) -> List[BaseMessage]:
        """Create properly formatted context messages for an agent.

        Args:
            round_messages: Messages from current round
            user_messages: User participation messages from previous rounds
            agent_id: ID of the agent receiving context

        Returns:
            List of BaseMessage objects for agent context
        """
        context_messages = []

        # Add user messages first (chronologically from previous rounds)
        for user_msg in user_messages:
            formatted_content = (
                f"[Round Moderator - Round {user_msg.round_number}]: {user_msg.content}"
            )
            human_msg = HumanMessage(
                content=formatted_content,
                name="user",
                additional_kwargs={
                    "speaker_id": "user",
                    "speaker_role": "user",
                    "round": user_msg.round_number,
                    "topic": user_msg.topic,
                    "timestamp": user_msg.timestamp.isoformat(),
                    "participation_type": "user_turn_participation",
                },
            )
            context_messages.append(human_msg)
            logger.debug(
                f"Added user participation message from round {user_msg.round_number} to {agent_id} context"
            )

        # Add current round messages (colleagues who spoke before this agent)
        for round_msg in round_messages:
            # Skip the agent's own previous messages
            if round_msg.speaker_id == agent_id:
                continue

            # Format colleague messages as HumanMessage for conversation flow
            colleague_content = f"[{round_msg.speaker_id}]: {round_msg.content}"
            colleague_msg = HumanMessage(
                content=colleague_content,
                name=round_msg.speaker_id,
                additional_kwargs={
                    "speaker_id": round_msg.speaker_id,
                    "speaker_role": round_msg.speaker_role,
                    "round": round_msg.round_number,
                    "topic": round_msg.topic,
                    "timestamp": round_msg.timestamp.isoformat(),
                },
            )
            context_messages.append(colleague_msg)
            logger.debug(
                f"Added colleague message from {round_msg.speaker_id} to {agent_id} context"
            )

        logger.info(
            f"Created {len(context_messages)} context messages for agent {agent_id}"
        )
        return context_messages

    @staticmethod
    def validate_message_consistency(
        messages: List[Union[Dict, BaseMessage]],
    ) -> Tuple[bool, List[str]]:
        """Validate message consistency and format compliance.

        Args:
            messages: List of messages to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        for i, msg in enumerate(messages):
            try:
                processed = MessageProcessor.standardize_message(msg)

                # Check required fields
                if not processed.speaker_id or processed.speaker_id == "unknown":
                    issues.append(f"Message {i}: Missing or invalid speaker_id")

                if not processed.content.strip():
                    issues.append(f"Message {i}: Empty content")

                if not processed.topic:
                    issues.append(f"Message {i}: Missing topic")

                if processed.round_number < 0:
                    issues.append(f"Message {i}: Invalid round number")

            except Exception as e:
                issues.append(f"Message {i}: Processing error - {str(e)}")

        is_valid = len(issues) == 0

        if not is_valid:
            logger.warning(f"Message validation failed with {len(issues)} issues")
        else:
            logger.debug(f"Message validation passed for {len(messages)} messages")

        return is_valid, issues
