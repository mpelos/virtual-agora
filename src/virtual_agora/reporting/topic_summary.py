"""Topic summary generation for Virtual Agora discussions.

This module provides functionality to generate comprehensive summaries
for individual discussion topics, including metadata and key insights.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import re

from ..state.schema import VirtualAgoraState, Message, RoundInfo
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TopicSummaryGenerator:
    """Generate comprehensive summaries for discussion topics."""

    def __init__(self, output_dir: Path = Path("outputs/summaries")):
        """Initialize TopicSummaryGenerator.

        Args:
            output_dir: Directory to save topic summaries.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(
        self,
        topic: str,
        summary_content: str,
        state: VirtualAgoraState,
    ) -> Path:
        """Generate and save a comprehensive topic summary.

        Args:
            topic: The topic title.
            summary_content: The synthesized summary content from moderator.
            state: Current Virtual Agora state.

        Returns:
            Path to the saved summary file.
        """
        try:
            # Extract metadata
            metadata = self._extract_topic_metadata(topic, state)

            # Format the summary with metadata
            formatted_summary = self._format_summary(topic, summary_content, metadata)

            # Save to file
            file_path = self._save_summary(topic, formatted_summary)

            logger.info(f"Generated topic summary: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error generating topic summary: {e}")
            raise

    def _extract_topic_metadata(
        self, topic: str, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Extract metadata for the topic from state.

        Args:
            topic: The topic title.
            state: Current Virtual Agora state.

        Returns:
            Dictionary containing topic metadata.
        """
        topic_info = state.get("topics_info", {}).get(topic, {})

        # Calculate duration
        start_time = topic_info.get("start_time")
        end_time = topic_info.get("end_time", datetime.now())
        duration = (
            (end_time - start_time).total_seconds() / 60
            if start_time and end_time
            else 0
        )

        # Count rounds for this topic
        rounds = [r for r in state.get("round_history", []) if r.get("topic") == topic]

        # Extract key insights from messages
        topic_messages = [
            m for m in state.get("messages", []) if m.get("topic") == topic
        ]

        # Count participating agents
        participants = set(
            m.get("speaker_id")
            for m in topic_messages
            if m.get("speaker_role") == "participant"
        )

        metadata = {
            "topic": topic,
            "duration_minutes": round(duration, 2),
            "number_of_rounds": len(rounds),
            "message_count": topic_info.get("message_count", 0),
            "participants": list(participants),
            "participant_count": len(participants),
            "proposed_by": topic_info.get("proposed_by", "Unknown"),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "status": topic_info.get("status", "completed"),
        }

        # Extract voting statistics if available
        vote_rounds = [
            v
            for v in state.get("vote_history", [])
            if v.get("vote_type") == "continue_discussion"
            and any(
                m.get("topic") == topic
                for m in state.get("messages", [])
                if m.get("timestamp", datetime.min)
                > (v.get("start_time") or datetime.min)
            )
        ]

        if vote_rounds:
            metadata["voting_rounds"] = len(vote_rounds)
            metadata["final_vote_result"] = vote_rounds[-1].get("result")

        return metadata

    def _format_summary(
        self, topic: str, summary_content: str, metadata: Dict[str, Any]
    ) -> str:
        """Format the topic summary with metadata in Markdown.

        Args:
            topic: The topic title.
            summary_content: The synthesized summary content.
            metadata: Topic metadata dictionary.

        Returns:
            Formatted Markdown summary.
        """
        # Build the formatted summary
        lines = [
            f"# Topic Summary: {topic}",
            "",
            "## Metadata",
            "",
            f"- **Duration**: {metadata['duration_minutes']} minutes",
            f"- **Number of Rounds**: {metadata['number_of_rounds']}",
            f"- **Total Messages**: {metadata['message_count']}",
            f"- **Participants**: {metadata['participant_count']} agents",
            f"- **Proposed By**: {metadata['proposed_by']}",
        ]

        if metadata.get("start_time"):
            lines.append(f"- **Start Time**: {metadata['start_time']}")
        if metadata.get("end_time"):
            lines.append(f"- **End Time**: {metadata['end_time']}")

        if metadata.get("voting_rounds"):
            lines.append(f"- **Voting Rounds**: {metadata['voting_rounds']}")
            if metadata.get("final_vote_result"):
                lines.append(
                    f"- **Final Vote Result**: {metadata['final_vote_result']}"
                )

        lines.extend(
            [
                "",
                "## Summary",
                "",
                summary_content,
                "",
                "---",
                "",
                f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            ]
        )

        return "\n".join(lines)

    def _save_summary(self, topic: str, content: str) -> Path:
        """Save the topic summary to a file.

        Args:
            topic: The topic title.
            content: The formatted summary content.

        Returns:
            Path to the saved file.
        """
        # Create a filesystem-safe filename
        safe_topic = re.sub(r"[^\w\s-]", "", topic)
        safe_topic = re.sub(r"[-\s]+", "_", safe_topic)

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"topic_summary_{safe_topic}_{timestamp}.md"

        file_path = self.output_dir / filename
        file_path.write_text(content, encoding="utf-8")

        return file_path

    def get_all_summaries(self) -> List[Path]:
        """Get all topic summary files in the output directory.

        Returns:
            List of paths to summary files.
        """
        return sorted(
            self.output_dir.glob("topic_summary_*.md"), key=lambda p: p.stat().st_mtime
        )

    def load_summary(self, file_path: Path) -> Dict[str, Any]:
        """Load a topic summary from file.

        Args:
            file_path: Path to the summary file.

        Returns:
            Dictionary with topic and content.
        """
        content = file_path.read_text(encoding="utf-8")

        # Extract topic from first line
        first_line = content.split("\n")[0]
        topic = first_line.replace("# Topic Summary: ", "").strip()

        return {
            "topic": topic,
            "content": content,
            "file_path": file_path,
        }
