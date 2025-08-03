"""Context Builder strategies for different agent types."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.context.types import ContextData
from virtual_agora.context.repository import ContextRepository
from virtual_agora.context.message_processor import MessageProcessor, ProcessedMessage
from virtual_agora.context.rules import ContextRules, ContextType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ContextBuilder(ABC):
    """Abstract base class for context builders.

    Each agent type gets its own context builder that knows how to
    assemble the appropriate context data for that agent's needs.
    """

    @abstractmethod
    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build context data appropriate for this agent type.

        Args:
            state: Application state
            system_prompt: Agent's system prompt
            **kwargs: Additional context-specific parameters

        Returns:
            ContextData configured for this agent type
        """
        pass


class DiscussionAgentContextBuilder(ContextBuilder):
    """Enhanced context builder for discussion/participant agents.

    Provides round-aware context assembly following business rules:
    - Round 0: Theme + topic + current round only
    - Round 1+: Theme + topic + round summaries + user messages + current round
    - Context documents for complex topics only
    - Proper message formatting and limits
    """

    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build round-aware context for discussion agents.

        Args:
            state: Application state
            system_prompt: Agent's system prompt
            **kwargs: Additional parameters including:
                - topic: Current topic (optional)
                - current_round: Round number (required for proper context)
                - agent_id: Agent ID (required for filtering)
                - agent_position: Agent's position in speaking order
        """
        # Extract parameters
        topic = kwargs.get("topic") or state.get("active_topic")
        current_round = kwargs.get("current_round", state.get("current_round", 0))
        agent_id = kwargs.get("agent_id", "unknown")
        agent_position = kwargs.get("agent_position", 0)

        logger.info(
            f"Building context for agent {agent_id}, round {current_round}, topic '{topic}'"
        )

        # Get context requirements based on round and rules
        rule_set, additional_params = ContextRules.get_context_requirements(
            ContextType.DISCUSSION_ROUND, current_round, state
        )

        # Initialize context data
        context_data = ContextData(system_prompt=system_prompt)

        # Add theme if required
        if rule_set.include_theme:
            context_data.user_input = ContextRepository.get_user_input(state)

        # Add context documents for complex topics
        if rule_set.include_context_documents:
            context_data.context_documents = ContextRepository.get_context_documents()
            logger.debug(f"Included context documents for complex topic")

        # Add round summaries from previous rounds
        if rule_set.include_round_summaries and current_round > 0:
            all_summaries = ContextRepository.get_round_summaries(state, topic)
            # Apply limits - summaries are not ProcessedMessage objects, so handle them directly
            if len(all_summaries) > rule_set.max_round_summaries:
                limited_summaries = all_summaries[
                    -rule_set.max_round_summaries :
                ]  # Recent summaries
            else:
                limited_summaries = all_summaries
            context_data.round_summaries = limited_summaries
            logger.debug(f"Included {len(limited_summaries)} round summaries")

        # Process current round and user messages using MessageProcessor
        all_messages = state.get("messages", [])

        # Get user participation messages from previous rounds
        if rule_set.include_user_messages and current_round > 0:
            user_messages = MessageProcessor.filter_user_participation_messages(
                all_messages, topic, exclude_round=current_round
            )
            # Apply limits and store for later use in context assembly
            limited_user_messages = ContextRules.enforce_message_limits(
                user_messages, rule_set.max_user_messages, strategy="recent"
            )
            # Store processed user messages in additional context
            context_data.user_participation_messages = limited_user_messages
            logger.debug(
                f"Included {len(limited_user_messages)} user participation messages"
            )

        # Get current round messages (colleagues who spoke before this agent)
        if rule_set.include_current_round:
            current_round_messages = MessageProcessor.filter_messages_by_round(
                all_messages, current_round, topic
            )
            # Limit to messages from agents who spoke before this agent
            colleague_messages = [
                msg
                for i, msg in enumerate(current_round_messages)
                if i < agent_position and msg.speaker_id != agent_id
            ]
            # Apply limits
            limited_colleague_messages = ContextRules.enforce_message_limits(
                colleague_messages,
                rule_set.max_current_round_messages,
                strategy="recent",
            )
            # Store processed colleague messages
            context_data.current_round_messages = limited_colleague_messages
            logger.debug(
                f"Included {len(limited_colleague_messages)} colleague messages from current round"
            )

        # Validate context assembly
        is_valid, issues, compliance_score = ContextRules.validate_context_assembly(
            ContextType.DISCUSSION_ROUND,
            current_round,
            context_data.user_input,
            topic,
            context_data.round_summaries or [],
            getattr(context_data, "user_participation_messages", []),
            getattr(context_data, "current_round_messages", []),
        )

        if not is_valid:
            logger.warning(f"Context validation issues for {agent_id}: {issues}")
        else:
            logger.info(
                f"Context validation passed for {agent_id} (compliance: {compliance_score:.2f})"
            )

        # Store additional metadata
        context_data.metadata = {
            "agent_id": agent_id,
            "current_round": current_round,
            "topic": topic,
            "agent_position": agent_position,
            "rule_set": rule_set,
            "compliance_score": compliance_score,
            "validation_issues": issues,
            **additional_params,
        }

        logger.info(
            f"Built context for {agent_id}: theme={bool(context_data.user_input)}, "
            f"docs={bool(context_data.context_documents)}, "
            f"summaries={len(context_data.round_summaries or [])}, "
            f"user_msgs={len(getattr(context_data, 'user_participation_messages', []))}, "
            f"colleague_msgs={len(getattr(context_data, 'current_round_messages', []))}"
        )

        return context_data


class ReportWriterContextBuilder(ContextBuilder):
    """Context builder for report writer agents.

    Report writers should ONLY get filtered discussion data:
    - For topic reports: topic messages + round summaries + final considerations
    - For session reports: topic reports only
    - NO context directory files or user input
    """

    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build filtered context for report writers."""
        report_type = kwargs.get("report_type", "topic")
        topic = kwargs.get("topic")

        if report_type == "topic":
            # Topic report: only discussion data for this topic
            return ContextData(
                system_prompt=system_prompt,
                # NO context_documents - this is the key isolation
                # NO user_input - report writers focus on discussion outcomes
                topic_messages=ContextRepository.get_topic_messages(state, topic),
                round_summaries=ContextRepository.get_round_summaries(state, topic),
            )
        elif report_type == "session":
            # Session report: only topic reports
            return ContextData(
                system_prompt=system_prompt,
                # NO context_documents
                # NO user_input
                # NO topic_messages - session reports synthesize topic reports
                topic_reports=ContextRepository.get_topic_reports(state),
            )
        else:
            logger.warning(
                f"Unknown report type: {report_type}, using topic report context"
            )
            return ContextData(
                system_prompt=system_prompt,
                topic_messages=ContextRepository.get_topic_messages(state, topic),
                round_summaries=ContextRepository.get_round_summaries(state, topic),
            )


class ModeratorContextBuilder(ContextBuilder):
    """Context builder for moderator agents.

    Moderators need process-specific context only:
    - Voting data when synthesizing votes
    - Agenda data when managing topics
    - Consensus data when tracking agreements
    - NO discussion content (they focus on process, not content)
    """

    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build process-focused context for moderators."""
        process_type = kwargs.get("process_type", "general")

        return ContextData(
            system_prompt=system_prompt,
            # NO context_documents - moderators focus on process
            # NO user_input - they work with structured data
            # NO topic_messages - they don't participate in content discussion
            process_context=ContextRepository.get_process_context(state, process_type),
        )


class SummarizerContextBuilder(ContextBuilder):
    """Context builder for summarizer agents.

    Summarizers need only the content to summarize:
    - Round messages when creating round summaries
    - Round summaries when creating topic conclusions
    - NO context directory files or extraneous context
    """

    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build content-focused context for summarizers."""
        content_type = kwargs.get("content_type", "round")
        topic = kwargs.get("topic")

        return ContextData(
            system_prompt=system_prompt,
            # NO context_documents - summarizers focus on specific content
            # NO user_input - they work with discussion outcomes
            content_to_summarize=ContextRepository.get_content_to_summarize(
                state, content_type, topic
            ),
        )


class DiscussionRoundContextBuilder(ContextBuilder):
    """Specialized context builder for discussion rounds.

    This builder focuses specifically on assembling context for agents
    participating in discussion rounds, with emphasis on proper message
    formatting and conversation flow.
    """

    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build context optimized for discussion round participation.

        Args:
            state: Application state
            system_prompt: Agent's system prompt
            **kwargs: Required parameters:
                - agent_id: Agent ID (required)
                - current_round: Round number (required)
                - agent_position: Position in speaking order (required)
                - speaking_order: Full speaking order list (required)
                - topic: Current topic (optional, uses active_topic)
                - current_round_messages: Messages from current round (optional)
        """
        # Extract required parameters
        agent_id = kwargs.get("agent_id")
        current_round = kwargs.get("current_round")
        agent_position = kwargs.get("agent_position")
        speaking_order = kwargs.get("speaking_order", [])
        topic = kwargs.get("topic") or state.get("active_topic")
        current_round_messages = kwargs.get("current_round_messages", [])

        if not agent_id:
            raise ValueError("agent_id is required for DiscussionRoundContextBuilder")
        if current_round is None:
            raise ValueError(
                "current_round is required for DiscussionRoundContextBuilder"
            )
        if agent_position is None:
            raise ValueError(
                "agent_position is required for DiscussionRoundContextBuilder"
            )

        logger.info(
            f"Building discussion round context for {agent_id} at position {agent_position + 1}/{len(speaking_order)}"
        )

        # Use the enhanced DiscussionAgentContextBuilder as base
        base_builder = DiscussionAgentContextBuilder()
        context_data = base_builder.build_context(state, system_prompt, **kwargs)

        # Add round-specific enhancements
        round_requirements = ContextRules.get_round_context_requirements(
            current_round, agent_position, len(speaking_order)
        )

        # Create formatted context messages using MessageProcessor
        all_messages = state.get("messages", [])

        # Get user messages from previous rounds
        user_messages = MessageProcessor.filter_user_participation_messages(
            all_messages, topic, exclude_round=current_round
        )
        logger.debug(
            f"Found {len(user_messages)} user participation messages for agent {agent_id}"
        )

        # Use provided current round messages or fall back to global state filtering
        if current_round_messages:
            logger.debug(
                f"Using {len(current_round_messages)} provided current round messages for agent {agent_id}"
            )
            # Use the provided messages from the current round execution
            colleague_messages = []
            # Convert LangChain messages to ProcessedMessage format for consistency
            for msg in current_round_messages:
                if hasattr(msg, "additional_kwargs"):
                    # Extract metadata from LangChain message
                    metadata = msg.additional_kwargs
                    speaker_id = metadata.get(
                        "speaker_id", getattr(msg, "name", "unknown")
                    )

                    # Only include messages from agents who spoke before this agent in speaking order
                    try:
                        speaker_position = speaking_order.index(speaker_id)
                        if speaker_position < agent_position and speaker_id != agent_id:
                            # Create ProcessedMessage-like object for consistency
                            processed_msg = MessageProcessor.standardize_message(msg)
                            colleague_messages.append(processed_msg)
                    except ValueError:
                        # Speaker not in speaking order, skip
                        continue
        else:
            # Fall back to global state filtering (for backward compatibility)
            round_messages_from_state = MessageProcessor.filter_messages_by_round(
                all_messages, current_round, topic
            )
            colleague_messages = []
            for i, msg in enumerate(round_messages_from_state):
                if i < agent_position and msg.speaker_id in speaking_order:
                    colleague_messages.append(msg)

        # Create properly formatted context messages
        formatted_context_messages = MessageProcessor.create_context_messages_for_agent(
            colleague_messages, user_messages, agent_id
        )

        # Store the formatted messages for immediate use
        context_data.formatted_context_messages = formatted_context_messages

        # Add round-specific metadata
        context_data.metadata.update(
            {
                "speaking_order": speaking_order,
                "round_requirements": round_requirements,
                "colleague_messages_count": len(colleague_messages),
                "user_messages_count": len(user_messages),
                "formatted_messages_count": len(formatted_context_messages),
                "context_type": "discussion_round",
            }
        )

        logger.info(
            f"Discussion round context for {agent_id}: "
            f"{len(formatted_context_messages)} formatted messages, "
            f"{len(colleague_messages)} colleagues, {len(user_messages)} user msgs"
        )

        if formatted_context_messages:
            logger.debug(
                f"Sample context message for {agent_id}: {formatted_context_messages[0].content[:100]}..."
            )

        return context_data


# Context builder factory for easy instantiation
CONTEXT_BUILDERS = {
    "discussion": DiscussionAgentContextBuilder,
    "discussion_round": DiscussionRoundContextBuilder,
    "report_writer": ReportWriterContextBuilder,
    "moderator": ModeratorContextBuilder,
    "summarizer": SummarizerContextBuilder,
}


def get_context_builder(agent_type: str) -> ContextBuilder:
    """Factory function to get appropriate context builder for agent type.

    Args:
        agent_type: Type of agent ("discussion", "report_writer", etc.)

    Returns:
        Appropriate context builder instance

    Raises:
        ValueError: If agent_type is not recognized
    """
    if agent_type not in CONTEXT_BUILDERS:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available types: {list(CONTEXT_BUILDERS.keys())}"
        )

    return CONTEXT_BUILDERS[agent_type]()
