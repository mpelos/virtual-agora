"""Context Builder strategies for different agent types."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.context.types import ContextData
from virtual_agora.context.repository import ContextRepository
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
    """Context builder for discussion/participant agents.

    Discussion agents need full context to participate meaningfully:
    - Context directory files (domain knowledge)
    - User's initial input/theme
    - Topic-specific discussion messages
    - Round summaries for awareness of discussion evolution
    """

    def build_context(
        self, state: VirtualAgoraState, system_prompt: str, **kwargs
    ) -> ContextData:
        """Build full context for discussion agents."""
        topic = kwargs.get("topic") or state.get("active_topic")

        return ContextData(
            system_prompt=system_prompt,
            context_documents=ContextRepository.get_context_documents(),
            user_input=ContextRepository.get_user_input(state),
            topic_messages=ContextRepository.get_topic_messages(state, topic),
            round_summaries=ContextRepository.get_round_summaries(state, topic),
        )


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


# Context builder factory for easy instantiation
CONTEXT_BUILDERS = {
    "discussion": DiscussionAgentContextBuilder,
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
