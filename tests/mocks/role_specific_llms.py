"""Role-specific deterministic LLM implementations.

This module contains specialized LLM classes for each Virtual Agora agent role,
providing realistic and consistent responses appropriate to each role's function.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.errors import GraphInterrupt
from pydantic import Field

from .deterministic_llms import BaseDeterministicLLM
from .llm_responses import (
    get_response_templates,
    get_context_patterns,
    get_interrupt_patterns,
    match_context_to_pattern,
    should_trigger_interrupt,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ModeratorDeterministicLLM(BaseDeterministicLLM):
    """Deterministic LLM for moderator agent behavior.

    Provides structured agenda proposals, round summaries, and voting decisions
    with consistent, realistic moderator behavior patterns.
    """

    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(
            model_name=model,
            role="moderator",
            provider=kwargs.get("provider", "openai"),
            temperature=float(kwargs.get("temperature") or 0.7),
            max_tokens=int(kwargs.get("max_tokens") or 4000),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["provider", "temperature", "max_tokens"] and v is not None
            },
        )

        # Load moderator-specific response patterns
        self.response_patterns = get_response_templates("moderator")

        # Set up context extractors
        self.context_extractors = {
            "is_agenda_request": self._extract_agenda_context,
            "is_summary_request": self._extract_summary_context,
            "is_voting_request": self._extract_voting_context,
            "active_topic": self._extract_active_topic,
            "discussion_phase": self._extract_discussion_phase,
        }

        # Set up interrupt triggers
        self.interrupt_triggers = {
            "agenda_approval": {
                "type": "agenda_approval",
                "condition": self._should_trigger_agenda_approval,
                "data": self._get_agenda_approval_data,
            }
        }

    def _extract_agenda_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is an agenda-related request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message
            for word in ["agenda", "structure", "propose", "topics"]
        )

    def _extract_summary_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a summary request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message
            for word in ["summary", "summarize", "recap", "progress"]
        )

    def _extract_voting_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a voting/decision request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message for word in ["vote", "decision", "continue", "proceed"]
        )

    def _extract_active_topic(self, messages: List[BaseMessage]) -> str:
        """Extract the current active topic from conversation."""
        # Look for topic mentions in recent messages
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                content = msg.content.lower()
                if "topic" in content:
                    # Simple extraction - in real implementation, this could be more sophisticated
                    return "Current Discussion Topic"
        return "General Discussion"

    def _extract_discussion_phase(self, messages: List[BaseMessage]) -> str:
        """Determine current phase of discussion."""
        if self.call_count <= 2:
            return "initialization"
        elif self.call_count <= 5:
            return "active_discussion"
        else:
            return "conclusion"

    def context_matches_pattern(
        self, context: Dict[str, Any], pattern_name: str
    ) -> bool:
        """Check if context matches moderator-specific patterns."""
        return match_context_to_pattern(context, "moderator", pattern_name)

    def _should_trigger_agenda_approval(self, context: Dict[str, Any]) -> bool:
        """Check if agenda approval interrupt should be triggered."""
        return (
            context.get("is_agenda_request", False)
            or "agenda" in context.get("last_user_message", "").lower()
            and context.get("call_count", 0) <= 3
        )

    def _get_agenda_approval_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for agenda approval interrupt."""
        return {
            "proposed_agenda": [
                "Core Concepts and Definitions",
                "Current State Analysis",
                "Key Challenges and Opportunities",
                "Potential Solutions and Approaches",
                "Implementation Considerations",
            ],
            "moderator_recommendation": "approve",
            "context": context.get("active_topic", "Discussion Topic"),
        }

    def should_trigger_interrupt(
        self, context: Dict[str, Any], trigger_name: str, trigger_config: Dict[str, Any]
    ) -> bool:
        """Check if moderator should trigger specific interrupts."""
        condition_func = trigger_config.get("condition")
        if condition_func and callable(condition_func):
            return condition_func(context)
        return False

    def trigger_interrupt(self, interrupt_info: Dict[str, Any]) -> None:
        """Trigger GraphInterrupt for moderator interactions."""
        interrupt_type = interrupt_info["type"]
        data_func = self.interrupt_triggers[interrupt_info["name"]].get("data")

        if data_func and callable(data_func):
            interrupt_data = data_func(interrupt_info["context"])
        else:
            interrupt_data = interrupt_info.get("data", {})

        logger.info(f"[Moderator] Triggering {interrupt_type} interrupt")

        # Note: In actual testing, this would trigger a real GraphInterrupt
        # For now, we just log the intent
        # raise GraphInterrupt({"type": interrupt_type, **interrupt_data})


class DiscussionAgentDeterministicLLM(BaseDeterministicLLM):
    """Deterministic LLM for discussion agent behavior.

    Provides thoughtful discussion contributions, responses to user input,
    and round conclusions with distinct agent personality.
    """

    agent_id: str = Field(default="agent_1")

    def __init__(
        self, model: str = "claude-3-opus-20240229", agent_id: str = "agent_1", **kwargs
    ):
        super().__init__(
            model_name=model,
            role="agent",
            provider=kwargs.get("provider", "anthropic"),
            temperature=float(kwargs.get("temperature") or 0.7),
            max_tokens=int(kwargs.get("max_tokens") or 3000),
            agent_id=agent_id,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["provider", "temperature", "max_tokens", "agent_id"]
                and v is not None
            },
        )

        # Load agent-specific response patterns
        self.response_patterns = get_response_templates("agent")

        # Set up context extractors
        self.context_extractors = {
            "is_discussion_contribution": self._extract_discussion_context,
            "is_user_response": self._extract_user_response_context,
            "is_conclusion_request": self._extract_conclusion_context,
            "discussion_depth": self._extract_discussion_depth,
            "agent_count": self._extract_agent_count,
        }

        # Set up interrupt triggers
        self.interrupt_triggers = {
            "user_turn_participation": {
                "type": "user_turn_participation",
                "condition": self._should_trigger_user_participation,
                "data": self._get_user_participation_data,
            }
        }

    def _extract_discussion_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a discussion contribution request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message
            for word in ["discuss", "perspective", "view", "opinion", "contribute"]
        )

    def _extract_user_response_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is responding to user input."""
        # Look for user input indicators in recent messages
        for msg in messages[-3:]:  # Check last 3 messages
            if hasattr(msg, "content") and msg.content:
                content = msg.content.lower()
                if any(
                    word in content
                    for word in ["user", "human", "input", "participant"]
                ):
                    return True
        return False

    def _extract_conclusion_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a round conclusion request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message
            for word in ["conclude", "summary", "position", "final"]
        )

    def _extract_discussion_depth(self, messages: List[BaseMessage]) -> str:
        """Determine depth of current discussion."""
        if len(messages) < 5:
            return "introductory"
        elif len(messages) < 15:
            return "developing"
        else:
            return "deep"

    def _extract_agent_count(self, messages: List[BaseMessage]) -> int:
        """Estimate number of participating agents."""
        # Simple heuristic based on conversation length
        return min(3, max(1, len(messages) // 5))

    def context_matches_pattern(
        self, context: Dict[str, Any], pattern_name: str
    ) -> bool:
        """Check if context matches agent-specific patterns."""
        return match_context_to_pattern(context, "agent", pattern_name)

    def _should_trigger_user_participation(self, context: Dict[str, Any]) -> bool:
        """Check if user participation interrupt should be triggered."""
        return (
            context.get("call_count", 0) >= 3
            and context.get("call_count", 0) % 3 == 0  # Every 3rd call
            and "user" in context.get("last_user_message", "").lower()
        )

    def _get_user_participation_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for user participation interrupt."""
        return {
            "current_round": context.get("call_count", 1),
            "current_topic": context.get("active_topic", "Discussion Topic"),
            "previous_summary": f"Agent {self.agent_id} has been contributing to the discussion",
            "agent_id": self.agent_id,
        }

    def should_trigger_interrupt(
        self, context: Dict[str, Any], trigger_name: str, trigger_config: Dict[str, Any]
    ) -> bool:
        """Check if agent should trigger specific interrupts."""
        condition_func = trigger_config.get("condition")
        if condition_func and callable(condition_func):
            return condition_func(context)
        return False


class SummarizerDeterministicLLM(BaseDeterministicLLM):
    """Deterministic LLM for summarizer agent behavior.

    Provides comprehensive round summaries and topic conclusions with
    structured analysis and consistent summarization patterns.
    """

    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(
            model_name=model,
            role="summarizer",
            provider=kwargs.get("provider", "openai"),
            temperature=float(kwargs.get("temperature") or 0.6),
            max_tokens=int(kwargs.get("max_tokens") or 3000),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["provider", "temperature", "max_tokens"] and v is not None
            },
        )

        # Load summarizer-specific response patterns
        self.response_patterns = get_response_templates("summarizer")

        # Set up context extractors
        self.context_extractors = {
            "is_round_summary": self._extract_round_summary_context,
            "is_topic_conclusion": self._extract_topic_conclusion_context,
            "total_rounds": self._extract_total_rounds,
            "discussion_quality": self._extract_discussion_quality,
            "active_topic": self._extract_active_topic,
        }

        # Summarizers typically don't trigger interrupts
        self.interrupt_triggers = {}

    def _extract_round_summary_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a round summary request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(word in last_message for word in ["summarize", "summary", "round"])

    def _extract_topic_conclusion_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a topic conclusion request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message
            for word in ["conclude", "conclusion", "final", "complete"]
        )

    def _extract_total_rounds(self, messages: List[BaseMessage]) -> int:
        """Estimate total rounds completed."""
        return max(1, self.call_count)

    def _extract_discussion_quality(self, messages: List[BaseMessage]) -> str:
        """Assess quality of discussion based on length and complexity."""
        total_length = sum(
            len(msg.content) for msg in messages if hasattr(msg, "content")
        )
        if total_length > 2000:
            return "comprehensive"
        elif total_length > 1000:
            return "adequate"
        else:
            return "developing"

    def _extract_active_topic(self, messages: List[BaseMessage]) -> str:
        """Extract current topic being discussed."""
        return "Current Discussion Topic"  # Simplified for testing

    def context_matches_pattern(
        self, context: Dict[str, Any], pattern_name: str
    ) -> bool:
        """Check if context matches summarizer-specific patterns."""
        return match_context_to_pattern(context, "summarizer", pattern_name)


class ReportWriterDeterministicLLM(BaseDeterministicLLM):
    """Deterministic LLM for report writer agent behavior.

    Provides comprehensive final reports and topic documentation with
    professional formatting and structured presentation.
    """

    def __init__(self, model: str = "claude-3-opus-20240229", **kwargs):
        super().__init__(
            model_name=model,
            role="report_writer",
            provider=kwargs.get("provider", "anthropic"),
            temperature=float(kwargs.get("temperature") or 0.5),
            max_tokens=int(kwargs.get("max_tokens") or 5000),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["provider", "temperature", "max_tokens"] and v is not None
            },
        )

        # Load report writer-specific response patterns
        self.response_patterns = get_response_templates("report_writer")

        # Set up context extractors
        self.context_extractors = {
            "is_final_report": self._extract_final_report_context,
            "is_topic_report": self._extract_topic_report_context,
            "session_metadata": self._extract_session_metadata,
            "discussion_duration": self._extract_discussion_duration,
            "main_topic": self._extract_main_topic,
        }

        # Report writers typically don't trigger interrupts
        self.interrupt_triggers = {}

    def _extract_final_report_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a final report request."""
        last_message = messages[-1].content.lower() if messages else ""
        return any(
            word in last_message for word in ["report", "final", "document", "complete"]
        )

    def _extract_topic_report_context(self, messages: List[BaseMessage]) -> bool:
        """Check if this is a topic-specific report request."""
        last_message = messages[-1].content.lower() if messages else ""
        return "topic" in last_message and "report" in last_message

    def _extract_session_metadata(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Extract session metadata for reporting."""
        return {
            "session_id": f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "agent_count": 3,  # Simplified for testing
            "total_rounds": max(1, self.call_count),
            "session_date": datetime.now().strftime("%Y-%m-%d"),
        }

    def _extract_discussion_duration(self, messages: List[BaseMessage]) -> str:
        """Calculate discussion duration."""
        # Simplified duration calculation for testing
        estimated_minutes = len(messages) * 2
        if estimated_minutes < 60:
            return f"{estimated_minutes} minutes"
        else:
            hours = estimated_minutes // 60
            minutes = estimated_minutes % 60
            return f"{hours}h {minutes}m"

    def _extract_main_topic(self, messages: List[BaseMessage]) -> str:
        """Extract main discussion topic."""
        # Look for topic mentions in early messages
        for msg in messages[:5]:
            if hasattr(msg, "content") and msg.content:
                # Simple extraction logic for testing
                if "topic" in msg.content.lower():
                    return "Main Discussion Topic"
        return "Virtual Agora Discussion"

    def context_matches_pattern(
        self, context: Dict[str, Any], pattern_name: str
    ) -> bool:
        """Check if context matches report writer-specific patterns."""
        return match_context_to_pattern(context, "report_writer", pattern_name)
