"""Report metadata generation for Virtual Agora sessions.

This module provides functionality to generate comprehensive metadata
and analytics for discussion sessions and reports.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict

from ..state.schema import VirtualAgoraState, Message, Vote, VoteRound, AgentInfo
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportMetadataGenerator:
    """Generate metadata and analytics for reports."""

    def __init__(self):
        """Initialize ReportMetadataGenerator."""
        self.metadata_cache = {}

    def generate_metadata(
        self,
        state: VirtualAgoraState,
        topic_summaries: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for a session.

        Args:
            state: Current Virtual Agora state.
            topic_summaries: Optional topic summaries.

        Returns:
            Dictionary containing session metadata and analytics.
        """
        try:
            # Basic session information
            metadata = self._extract_session_info(state)

            # Agent participation metrics
            metadata["agent_metrics"] = self._calculate_agent_metrics(state)

            # Topic analytics
            metadata["topic_analytics"] = self._analyze_topics(state)

            # Voting statistics
            metadata["voting_statistics"] = self._calculate_voting_stats(state)

            # Discussion dynamics
            metadata["discussion_dynamics"] = self._analyze_discussion_dynamics(state)

            # Content statistics
            metadata["content_statistics"] = self._calculate_content_stats(
                state, topic_summaries
            )

            # Cache the metadata
            self.metadata_cache = metadata

            logger.info("Generated comprehensive session metadata")
            return metadata

        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return {}

    def _extract_session_info(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Extract basic session information.

        Args:
            state: Virtual Agora state.

        Returns:
            Dictionary of session information.
        """
        start_time = state.get("start_time")
        current_time = datetime.now()

        # Calculate duration
        if start_time:
            duration = (current_time - start_time).total_seconds() / 60
        else:
            duration = 0

        return {
            "session_id": state.get("session_id", "unknown"),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": current_time.isoformat(),
            "duration_minutes": round(duration, 2),
            "main_topic": state.get("main_topic"),
            "current_phase": state.get("current_phase", 0),
            "total_messages": state.get("total_messages", 0),
            "topics_discussed": len(state.get("completed_topics", [])),
            "current_round": state.get("current_round", 0),
        }

    def _calculate_agent_metrics(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Calculate agent participation metrics.

        Args:
            state: Virtual Agora state.

        Returns:
            Dictionary of agent metrics.
        """
        agents = state.get("agents", {})
        messages_by_agent = state.get("messages_by_agent", {})

        # Calculate metrics for each agent
        agent_metrics = {}
        for agent_id, agent_info in agents.items():
            message_count = messages_by_agent.get(agent_id, 0)

            # Calculate activity metrics
            metrics = {
                "agent_id": agent_id,
                "model": agent_info.get("model", "unknown"),
                "provider": agent_info.get("provider", "unknown"),
                "role": agent_info.get("role", "participant"),
                "message_count": message_count,
                "participation_rate": 0,
                "average_message_length": 0,
            }

            # Calculate participation rate
            total_messages = state.get("total_messages", 0)
            if total_messages > 0:
                metrics["participation_rate"] = round(
                    (message_count / total_messages) * 100, 2
                )

            # Calculate average message length
            agent_messages = [
                m for m in state.get("messages", []) if m.get("speaker_id") == agent_id
            ]
            if agent_messages:
                total_length = sum(len(m.get("content", "")) for m in agent_messages)
                metrics["average_message_length"] = round(
                    total_length / len(agent_messages), 0
                )

            agent_metrics[agent_id] = metrics

        # Summary statistics
        return {
            "total_agents": len(agents),
            "participating_agents": len(
                [a for a in agent_metrics.values() if a["message_count"] > 0]
            ),
            "moderator_count": len(
                [a for a in agent_metrics.values() if a["role"] == "moderator"]
            ),
            "agent_details": list(agent_metrics.values()),
            "most_active_agent": max(
                agent_metrics.values(), key=lambda x: x["message_count"], default=None
            ),
        }

    def _analyze_topics(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Analyze topic-related metrics.

        Args:
            state: Virtual Agora state.

        Returns:
            Dictionary of topic analytics.
        """
        topics_info = state.get("topics_info", {})
        completed_topics = state.get("completed_topics", [])
        messages_by_topic = state.get("messages_by_topic", {})

        topic_details = []
        for topic, info in topics_info.items():
            # Calculate topic duration
            start = info.get("start_time")
            end = info.get("end_time")
            duration = 0
            if start and end:
                duration = (end - start).total_seconds() / 60

            topic_data = {
                "topic": topic,
                "status": info.get("status", "unknown"),
                "proposed_by": info.get("proposed_by", "unknown"),
                "message_count": info.get("message_count", 0),
                "duration_minutes": round(duration, 2),
                "participation_count": 0,
            }

            # Count unique participants
            topic_messages = [
                m for m in state.get("messages", []) if m.get("topic") == topic
            ]
            participants = set(m.get("speaker_id") for m in topic_messages)
            topic_data["participation_count"] = len(participants)

            topic_details.append(topic_data)

        # Summary analytics
        total_duration = sum(t["duration_minutes"] for t in topic_details)

        return {
            "total_topics_proposed": len(state.get("proposed_topics", [])),
            "topics_discussed": len(completed_topics),
            "topics_skipped": len(
                [t for t in topics_info.values() if t.get("status") == "skipped"]
            ),
            "average_topic_duration": round(
                total_duration / len(topic_details) if topic_details else 0, 2
            ),
            "topic_details": topic_details,
        }

    def _calculate_voting_stats(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Calculate voting statistics.

        Args:
            state: Virtual Agora state.

        Returns:
            Dictionary of voting statistics.
        """
        vote_history = state.get("vote_history", [])
        votes = state.get("votes", [])

        # Analyze vote types
        vote_types = Counter(v.get("vote_type") for v in vote_history)

        # Calculate participation rates
        voting_rounds = []
        for round_info in vote_history:
            round_data = {
                "vote_type": round_info.get("vote_type"),
                "phase": round_info.get("phase"),
                "required_votes": round_info.get("required_votes", 0),
                "received_votes": round_info.get("received_votes", 0),
                "participation_rate": 0,
                "result": round_info.get("result"),
                "unanimous": False,
            }

            # Calculate participation rate
            if round_data["required_votes"] > 0:
                round_data["participation_rate"] = round(
                    (round_data["received_votes"] / round_data["required_votes"]) * 100,
                    2,
                )

            # Check for unanimous votes
            round_votes = [
                v
                for v in votes
                if v.get("vote_type") == round_info.get("vote_type")
                and v.get("phase") == round_info.get("phase")
            ]
            if round_votes:
                choices = set(v.get("choice") for v in round_votes)
                round_data["unanimous"] = len(choices) == 1

            voting_rounds.append(round_data)

        return {
            "total_voting_rounds": len(vote_history),
            "vote_type_distribution": dict(vote_types),
            "average_participation_rate": round(
                (
                    sum(r["participation_rate"] for r in voting_rounds)
                    / len(voting_rounds)
                    if voting_rounds
                    else 0
                ),
                2,
            ),
            "unanimous_votes": sum(1 for r in voting_rounds if r["unanimous"]),
            "voting_round_details": voting_rounds,
        }

    def _analyze_discussion_dynamics(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Analyze discussion dynamics and patterns.

        Args:
            state: Virtual Agora state.

        Returns:
            Dictionary of discussion dynamics.
        """
        messages = state.get("messages", [])
        round_history = state.get("round_history", [])

        # Message distribution over time
        message_timeline = defaultdict(int)
        if messages and messages[0].get("timestamp"):
            start_time = messages[0]["timestamp"]

            for msg in messages:
                if msg.get("timestamp"):
                    # Group by 5-minute intervals
                    elapsed = (msg["timestamp"] - start_time).total_seconds()
                    interval = int(elapsed // 300)  # 5-minute intervals
                    message_timeline[interval] += 1

        # Round analysis
        round_stats = []
        for round_info in round_history:
            round_stat = {
                "round_number": round_info.get("round_number"),
                "topic": round_info.get("topic"),
                "message_count": round_info.get("message_count", 0),
                "participants": len(round_info.get("participants", [])),
                "has_summary": bool(round_info.get("summary")),
            }
            round_stats.append(round_stat)

        # Turn order analysis
        turn_order_changes = len(state.get("turn_order_history", []))

        return {
            "message_distribution": dict(message_timeline),
            "rounds_completed": len(round_history),
            "average_messages_per_round": round(
                (
                    sum(r["message_count"] for r in round_stats) / len(round_stats)
                    if round_stats
                    else 0
                ),
                2,
            ),
            "turn_order_rotations": turn_order_changes,
            "round_statistics": round_stats,
        }

    def _calculate_content_stats(
        self,
        state: VirtualAgoraState,
        topic_summaries: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Calculate content-related statistics.

        Args:
            state: Virtual Agora state.
            topic_summaries: Optional topic summaries.

        Returns:
            Dictionary of content statistics.
        """
        messages = state.get("messages", [])

        # Calculate message lengths
        message_lengths = [len(m.get("content", "")) for m in messages]

        # Word and token estimates
        total_chars = sum(message_lengths)
        word_estimate = total_chars // 5  # Rough estimate
        token_estimate = total_chars // 4  # Rough token estimate

        # Summary statistics
        summary_stats = {}
        if topic_summaries:
            summary_lengths = [len(s) for s in topic_summaries.values()]
            summary_stats = {
                "total_summaries": len(topic_summaries),
                "total_summary_length": sum(summary_lengths),
                "average_summary_length": round(
                    (
                        sum(summary_lengths) / len(summary_lengths)
                        if summary_lengths
                        else 0
                    ),
                    0,
                ),
            }

        return {
            "total_characters": total_chars,
            "estimated_words": word_estimate,
            "estimated_tokens": token_estimate,
            "average_message_length": round(
                sum(message_lengths) / len(message_lengths) if message_lengths else 0, 0
            ),
            "longest_message": max(message_lengths, default=0),
            "shortest_message": min(message_lengths, default=0),
            "summary_statistics": summary_stats,
        }

    def export_metadata(
        self,
        output_path: Path,
        format: str = "json",
        include_raw_data: bool = False,
    ) -> Path:
        """Export metadata to file.

        Args:
            output_path: Path to save metadata.
            format: Export format (json, yaml).
            include_raw_data: Whether to include raw data.

        Returns:
            Path to the exported file.
        """
        if not self.metadata_cache:
            raise ValueError("No metadata generated yet")

        metadata = self.metadata_cache.copy()

        if not include_raw_data:
            # Remove detailed data for summary export
            if "agent_metrics" in metadata:
                metadata["agent_metrics"].pop("agent_details", None)
            if "topic_analytics" in metadata:
                metadata["topic_analytics"].pop("topic_details", None)
            if "voting_statistics" in metadata:
                metadata["voting_statistics"].pop("voting_round_details", None)

        if format == "json":
            output_path.write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        elif format == "yaml":
            # Would need to import yaml library
            # For now, just use JSON
            logger.warning("YAML format not implemented, using JSON")
            output_path = output_path.with_suffix(".json")
            output_path.write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )

        logger.info(f"Exported metadata to {output_path}")
        return output_path

    def generate_analytics_summary(self) -> str:
        """Generate a human-readable analytics summary.

        Returns:
            Formatted analytics summary string.
        """
        if not self.metadata_cache:
            return "No metadata available"

        m = self.metadata_cache
        lines = [
            "# Session Analytics Summary",
            "",
            f"**Session ID**: {m.get('session_id')}",
            f"**Duration**: {m.get('duration_minutes')} minutes",
            f"**Total Messages**: {m.get('total_messages')}",
            "",
            "## Participation",
            f"- Total Agents: {m.get('agent_metrics', {}).get('total_agents', 0)}",
            f"- Active Participants: {m.get('agent_metrics', {}).get('participating_agents', 0)}",
            "",
            "## Topics",
            f"- Topics Discussed: {m.get('topic_analytics', {}).get('topics_discussed', 0)}",
            f"- Average Topic Duration: {m.get('topic_analytics', {}).get('average_topic_duration', 0)} minutes",
            "",
            "## Voting",
            f"- Total Voting Rounds: {m.get('voting_statistics', {}).get('total_voting_rounds', 0)}",
            f"- Average Participation: {m.get('voting_statistics', {}).get('average_participation_rate', 0)}%",
            "",
            "## Content",
            f"- Estimated Words: {m.get('content_statistics', {}).get('estimated_words', 0):,}",
            f"- Average Message Length: {m.get('content_statistics', {}).get('average_message_length', 0)} characters",
        ]

        return "\n".join(lines)
