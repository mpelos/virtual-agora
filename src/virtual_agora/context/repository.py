"""Context Repository for centralized access to context data."""

from typing import Dict, List, Optional, Any

from virtual_agora.state.schema import Message, RoundSummary, VirtualAgoraState
from virtual_agora.utils.document_context import load_context_documents
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ContextRepository:
    """Centralized repository for accessing all types of context data.

    This class provides a clean interface for accessing different types
    of context data from the application state and external sources.
    It abstracts the complexity of data retrieval and filtering.
    """

    @staticmethod
    def get_context_documents() -> str:
        """Get all context directory files as a single string.

        Returns:
            Concatenated content of all context directory files
        """
        return load_context_documents()

    @staticmethod
    def get_user_input(state: VirtualAgoraState) -> Optional[str]:
        """Get the user's initial discussion theme/input.

        Args:
            state: Application state

        Returns:
            User's main topic/theme
        """
        return state.get("main_topic")

    @staticmethod
    def get_topic_messages(
        state: VirtualAgoraState, topic: Optional[str] = None
    ) -> List[Message]:
        """Get discussion messages, optionally filtered by topic.

        Args:
            state: Application state
            topic: Topic to filter by (if None, returns all messages)

        Returns:
            List of messages for the specified topic or all messages
        """
        all_messages = state.get("messages", [])

        if topic is None:
            return all_messages

        # Filter messages by topic (this is a simplified implementation)
        # In practice, you might need more sophisticated filtering logic
        # based on how topics are tracked in your state
        active_topic = state.get("active_topic")
        if active_topic == topic:
            # Return messages from when this topic was active
            # This is simplified - you might need topic-specific message tracking
            return all_messages

        return []

    @staticmethod
    def get_round_summaries(
        state: VirtualAgoraState, topic: Optional[str] = None
    ) -> List[RoundSummary]:
        """Get round summaries, optionally filtered by topic.

        Args:
            state: Application state
            topic: Topic to filter by (if None, returns all summaries)

        Returns:
            List of round summaries for the specified topic or all summaries
        """
        all_summaries = state.get("round_summaries", [])

        if topic is None:
            return all_summaries

        # Filter summaries by topic
        topic_summaries = []
        for summary in all_summaries:
            if isinstance(summary, dict) and summary.get("topic") == topic:
                topic_summaries.append(summary)
            elif hasattr(summary, "topic") and summary.topic == topic:
                topic_summaries.append(summary)

        return topic_summaries

    @staticmethod
    def get_topic_reports(state: VirtualAgoraState) -> Dict[str, str]:
        """Get all generated topic reports.

        Args:
            state: Application state

        Returns:
            Dictionary mapping topic names to their reports
        """
        return state.get("topic_summaries", {})

    @staticmethod
    def get_process_context(
        state: VirtualAgoraState, context_type: str
    ) -> Dict[str, Any]:
        """Get process-specific context for moderators.

        Args:
            state: Application state
            context_type: Type of process context needed

        Returns:
            Dictionary with process-specific context data
        """
        if context_type == "voting":
            return {
                "active_vote": state.get("active_vote"),
                "vote_history": state.get("vote_history", []),
                "votes": state.get("votes", []),
                "conclusion_vote": state.get("conclusion_vote"),
                "topic_conclusion_votes": state.get("topic_conclusion_votes", []),
            }
        elif context_type == "agenda":
            return {
                "proposed_topics": state.get("proposed_topics", []),
                "topic_queue": state.get("topic_queue", []),
                "proposed_agenda": state.get("proposed_agenda", []),
                "completed_topics": state.get("completed_topics", []),
                "active_topic": state.get("active_topic"),
            }
        elif context_type == "consensus":
            return {
                "consensus_proposals": state.get("consensus_proposals", {}),
                "consensus_reached": state.get("consensus_reached", {}),
                "consensus_summaries": state.get("consensus_summaries", {}),
            }
        else:
            logger.warning(f"Unknown process context type: {context_type}")
            return {}

    @staticmethod
    def get_content_to_summarize(
        state: VirtualAgoraState, content_type: str, topic: Optional[str] = None
    ) -> List[str]:
        """Get content that needs to be summarized.

        Args:
            state: Application state
            content_type: Type of content to summarize ("round", "topic", etc.)
            topic: Topic to filter by (if applicable)

        Returns:
            List of content strings to be summarized
        """
        if content_type == "round":
            # Get messages from current round for summarization
            messages = ContextRepository.get_topic_messages(state, topic)
            return [
                msg.get("content", "") if isinstance(msg, dict) else msg.content
                for msg in messages
            ]
        elif content_type == "topic":
            # Get round summaries for topic conclusion
            summaries = ContextRepository.get_round_summaries(state, topic)
            return [
                (
                    summary.get("summary_text", "")
                    if isinstance(summary, dict)
                    else summary.summary_text
                )
                for summary in summaries
            ]
        else:
            logger.warning(f"Unknown content type for summarization: {content_type}")
            return []
