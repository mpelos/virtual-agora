"""Agent mixin for clean context injection using the Context Strategy pattern."""

from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from virtual_agora.state.schema import VirtualAgoraState, Message
from virtual_agora.context.builders import ContextBuilder
from virtual_agora.context.types import ContextData
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ContextAwareMixin:
    """Mixin class that provides context-aware message formatting for agents.

    This mixin uses the Context Strategy pattern to provide clean context
    injection based on the agent's specific context builder. It replaces
    the monolithic context loading in the base LLMAgent class.
    """

    def __init__(self, context_builder: ContextBuilder, *args, **kwargs):
        """Initialize with a context builder strategy.

        Args:
            context_builder: Strategy for building context appropriate to this agent
        """
        self.context_builder = context_builder
        super().__init__(*args, **kwargs)

    def format_messages_with_context(
        self,
        state: VirtualAgoraState,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        include_system: bool = True,
        **context_kwargs,
    ) -> List[BaseMessage]:
        """Format messages with agent-specific context.

        This method replaces the problematic format_messages method in LLMAgent
        by using the injected context builder to provide appropriate context.

        Args:
            state: Application state for context building
            prompt: The current prompt/question
            context_messages: Previous messages for context
            include_system: Whether to include system prompt
            **context_kwargs: Additional context-specific parameters

        Returns:
            List of formatted messages with appropriate context
        """
        messages: List[BaseMessage] = []

        # Build context using the injected strategy
        try:
            context_data = self.context_builder.build_context(
                state,
                self.system_prompt if hasattr(self, "system_prompt") else "",
                **context_kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            # Fallback to minimal context
            context_data = ContextData(
                system_prompt=(
                    self.system_prompt if hasattr(self, "system_prompt") else ""
                )
            )

        # Add system prompt
        if include_system and context_data.system_prompt:
            messages.append(SystemMessage(content=context_data.system_prompt))

        # Add context messages (conversation history)
        if context_messages:
            for msg in context_messages:
                if msg["speaker_role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        # Build enhanced prompt with context data
        enhanced_prompt = self._build_enhanced_prompt(prompt, context_data)

        # Add the enhanced prompt
        messages.append(HumanMessage(content=enhanced_prompt))

        return messages

    def _build_enhanced_prompt(
        self, base_prompt: str, context_data: ContextData
    ) -> str:
        """Build enhanced prompt with context data.

        Args:
            base_prompt: The original prompt
            context_data: Context data to include

        Returns:
            Enhanced prompt with relevant context
        """
        prompt_parts = []

        # Add context documents if available (for discussion agents)
        if context_data.has_context_documents:
            prompt_parts.append(f"Context Documents:\n{context_data.context_documents}")

        # Add user input if available
        if context_data.has_user_input:
            prompt_parts.append(f"Discussion Theme: {context_data.user_input}")

        # Add topic messages if available
        if context_data.has_topic_messages:
            messages_text = "\n".join(
                [
                    self._format_message_for_context(msg)
                    for msg in context_data.topic_messages[-10:]  # Last 10 messages
                ]
            )
            prompt_parts.append(f"Recent Discussion Messages:\n{messages_text}")

        # Add round summaries if available
        if context_data.has_round_summaries:
            summaries_text = "\n".join(
                [
                    (
                        f"Round {summary.get('round_number', 'N/A')}: {summary.get('summary_text', '')}"
                        if isinstance(summary, dict)
                        else f"Round {summary.round_number}: {summary.summary_text}"
                    )
                    for summary in context_data.round_summaries[-5:]  # Last 5 summaries
                ]
            )
            prompt_parts.append(f"Previous Round Summaries:\n{summaries_text}")

        # Add topic reports if available (for session reports)
        if context_data.has_topic_reports:
            reports_text = "\n".join(
                [
                    (
                        f"**{topic}:**\n{report[:500]}..."
                        if len(report) > 500
                        else f"**{topic}:**\n{report}"
                    )
                    for topic, report in context_data.topic_reports.items()
                ]
            )
            prompt_parts.append(f"Topic Reports:\n{reports_text}")

        # Add process context if available (for moderators)
        if context_data.process_context:
            process_text = self._format_process_context(context_data.process_context)
            if process_text:
                prompt_parts.append(f"Process Context:\n{process_text}")

        # Add content to summarize if available (for summarizers)
        if context_data.content_to_summarize:
            content_text = "\n".join(
                [
                    f"- {content[:200]}..." if len(content) > 200 else f"- {content}"
                    for content in context_data.content_to_summarize[
                        -10:
                    ]  # Last 10 items
                ]
            )
            prompt_parts.append(f"Content to Summarize:\n{content_text}")

        # Combine all parts with the base prompt
        if prompt_parts:
            enhanced_prompt = (
                "\n\n".join(prompt_parts) + "\n\n" + "=" * 50 + "\n\n" + base_prompt
            )
        else:
            enhanced_prompt = base_prompt

        return enhanced_prompt

    def _format_process_context(self, process_context: Dict[str, Any]) -> str:
        """Format process context for moderators.

        Args:
            process_context: Process-specific context data

        Returns:
            Formatted process context string
        """
        if not process_context:
            return ""

        context_parts = []

        # Format voting context
        if "votes" in process_context:
            votes = process_context["votes"]
            if votes:
                votes_text = f"Recent votes: {len(votes)} votes recorded"
                context_parts.append(votes_text)

        # Format agenda context
        if "proposed_topics" in process_context:
            topics = process_context["proposed_topics"]
            if topics:
                topics_text = (
                    f"Proposed topics: {', '.join(topics[:5])}"  # First 5 topics
                )
                context_parts.append(topics_text)

        # Format consensus context
        if "consensus_proposals" in process_context:
            proposals = process_context["consensus_proposals"]
            if proposals:
                proposals_text = (
                    f"Active proposals: {len(proposals)} topics with proposals"
                )
                context_parts.append(proposals_text)

        return "\n".join(context_parts)

    def _format_message_for_context(self, msg) -> str:
        """Format a message for context display, handling different message formats.

        Args:
            msg: Message in various formats (dict, Virtual Agora Message, LangChain BaseMessage)

        Returns:
            Formatted message string
        """
        # Handle dictionary format (Virtual Agora Message)
        if isinstance(msg, dict):
            speaker_id = msg.get("speaker_id", "Unknown")
            content = msg.get("content", "")
            return f"- {speaker_id}: {content}"

        # Handle Virtual Agora Message objects with attributes
        elif hasattr(msg, "speaker_id") and hasattr(msg, "content"):
            return f"- {msg.speaker_id}: {msg.content}"

        # Handle LangChain BaseMessage objects (AIMessage, HumanMessage, etc.)
        elif hasattr(msg, "content") and hasattr(msg, "__class__"):
            # Extract speaker from message type or use class name
            message_type = msg.__class__.__name__
            if message_type == "HumanMessage":
                speaker = "User"
            elif message_type == "AIMessage":
                speaker = "AI"
            elif message_type == "SystemMessage":
                speaker = "System"
            else:
                speaker = message_type

            content = getattr(msg, "content", "")
            return f"- {speaker}: {content}"

        # Fallback for unknown formats
        else:
            return f"- Unknown: {str(msg)}"
