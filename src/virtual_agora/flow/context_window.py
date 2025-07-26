"""Context Window Management for Virtual Agora.

This module provides token counting and context compression functionality
to manage the context window size during discussion flows.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from virtual_agora.state.schema import VirtualAgoraState, Message
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ContextWindowManager:
    """Manages context window size through token counting and compression."""

    def __init__(self, window_limit: int = 8000):
        """Initialize context window manager.

        Args:
            window_limit: Maximum number of tokens in context window
        """
        self.window_limit = window_limit

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using simple heuristics.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token for English text
        # This is a rough estimate, actual tokenization varies by model
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages: List[Message]) -> int:
        """Count total tokens in message list.

        Args:
            messages: List of messages to count

        Returns:
            Total estimated token count
        """
        total_tokens = 0
        for message in messages:
            # Count tokens in content
            total_tokens += self.estimate_tokens(message["content"])

            # Add overhead for message metadata (speaker, timestamp, etc.)
            total_tokens += 20  # Estimated overhead per message

        return total_tokens

    def get_context_size(self, state: VirtualAgoraState) -> int:
        """Get current context size in tokens.

        Args:
            state: Virtual Agora state

        Returns:
            Current context size in tokens
        """
        context_size = 0

        # Count message tokens
        context_size += self.count_message_tokens(state["messages"])

        # Count tokens in summaries
        for summary in state["phase_summaries"].values():
            context_size += self.estimate_tokens(summary)

        for summary in state["topic_summaries"].values():
            context_size += self.estimate_tokens(summary)

        # Count tokens in consensus summaries
        for summary in state["consensus_summaries"].values():
            context_size += self.estimate_tokens(summary)

        # Count tokens in final report if present
        if state["final_report"]:
            context_size += self.estimate_tokens(state["final_report"])

        return context_size

    def needs_compression(self, state: VirtualAgoraState) -> bool:
        """Check if context window needs compression.

        Args:
            state: Virtual Agora state

        Returns:
            True if compression is needed
        """
        current_size = self.get_context_size(state)
        limit = state["flow_control"]["context_window_limit"]

        # Compress when we reach 80% of limit
        return current_size > (limit * 0.8)

    def compress_messages(
        self, messages: List[Message], target_size: int
    ) -> Tuple[List[Message], str]:
        """Compress message history to fit target size.

        Args:
            messages: Original message list
            target_size: Target token count after compression

        Returns:
            Tuple of (compressed_messages, compression_summary)
        """
        if not messages:
            return [], "No messages to compress"

        # Always keep the most recent messages
        recent_messages = []
        current_tokens = 0
        compression_summary = ""

        # Work backwards from newest messages
        for message in reversed(messages):
            message_tokens = self.estimate_tokens(message["content"]) + 20
            if current_tokens + message_tokens <= target_size:
                recent_messages.insert(0, message)
                current_tokens += message_tokens
            else:
                break

        # Create summary of compressed messages
        if len(recent_messages) < len(messages):
            compressed_count = len(messages) - len(recent_messages)
            earliest_kept = (
                recent_messages[0]["timestamp"] if recent_messages else datetime.now()
            )

            # Group compressed messages by phase and topic for summary
            phase_counts: Dict[int, int] = {}
            topic_counts: Dict[str, int] = {}

            for message in messages[:compressed_count]:
                phase = message["phase"]
                topic = message.get("topic")

                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if topic:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # Build compression summary
            summary_parts = [
                f"[COMPRESSED: {compressed_count} messages before {earliest_kept.strftime('%H:%M:%S')}]"
            ]

            if phase_counts:
                phase_summary = ", ".join(
                    [f"Phase {p}: {c} msgs" for p, c in phase_counts.items()]
                )
                summary_parts.append(f"By phase: {phase_summary}")

            if topic_counts:
                topic_summary = ", ".join(
                    [f"{t}: {c} msgs" for t, c in list(topic_counts.items())[:3]]
                )
                if len(topic_counts) > 3:
                    topic_summary += f", +{len(topic_counts)-3} more topics"
                summary_parts.append(f"By topic: {topic_summary}")

            compression_summary = " | ".join(summary_parts)

            logger.info(
                f"Compressed {compressed_count} messages, kept {len(recent_messages)} recent messages"
            )

        return recent_messages, compression_summary

    def compress_context(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Compress context to fit within window limit.

        Args:
            state: Virtual Agora state to compress

        Returns:
            Dictionary of updates to apply to state
        """
        limit = state["flow_control"]["context_window_limit"]
        current_size = self.get_context_size(state)

        if current_size <= limit:
            return {}  # No compression needed

        # Target size is 60% of limit to leave room for new content
        target_size = int(limit * 0.6)

        # Compress messages (main source of context bloat)
        compressed_messages, compression_summary = self.compress_messages(
            state["messages"], target_size
        )

        updates = {"messages": compressed_messages}

        # Add compression summary to warnings
        if compression_summary:
            updates["warnings"] = state["warnings"] + [
                f"Context compressed at {datetime.now().strftime('%H:%M:%S')}: {compression_summary}"
            ]

        new_size = self.get_context_size({**state, **updates})
        logger.info(
            f"Context compression: {current_size} -> {new_size} tokens ({target_size} target)"
        )

        return updates

    def get_context_stats(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Get detailed context window statistics.

        Args:
            state: Virtual Agora state

        Returns:
            Dictionary of context statistics
        """
        message_tokens = self.count_message_tokens(state["messages"])
        phase_summary_tokens = sum(
            self.estimate_tokens(summary)
            for summary in state["phase_summaries"].values()
        )
        topic_summary_tokens = sum(
            self.estimate_tokens(summary)
            for summary in state["topic_summaries"].values()
        )
        consensus_tokens = sum(
            self.estimate_tokens(summary)
            for summary in state["consensus_summaries"].values()
        )
        report_tokens = (
            self.estimate_tokens(state["final_report"]) if state["final_report"] else 0
        )

        total_tokens = (
            message_tokens
            + phase_summary_tokens
            + topic_summary_tokens
            + consensus_tokens
            + report_tokens
        )
        limit = state["flow_control"]["context_window_limit"]

        return {
            "total_tokens": total_tokens,
            "limit": limit,
            "usage_percent": (total_tokens / limit) * 100 if limit > 0 else 0,
            "needs_compression": total_tokens > (limit * 0.8),
            "breakdown": {
                "messages": message_tokens,
                "phase_summaries": phase_summary_tokens,
                "topic_summaries": topic_summary_tokens,
                "consensus_summaries": consensus_tokens,
                "final_report": report_tokens,
            },
            "message_count": len(state["messages"]),
        }


def create_context_manager(window_limit: int = 8000) -> ContextWindowManager:
    """Create a context window manager instance.

    Args:
        window_limit: Maximum number of tokens in context window

    Returns:
        ContextWindowManager instance
    """
    return ContextWindowManager(window_limit)
