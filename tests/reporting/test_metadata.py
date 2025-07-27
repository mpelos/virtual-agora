"""Tests for report metadata generation."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from virtual_agora.reporting.metadata import ReportMetadataGenerator
from virtual_agora.state.schema import VirtualAgoraState


class TestReportMetadataGenerator:
    """Test ReportMetadataGenerator functionality."""

    def setup_method(self):
        """Set up test method."""
        self.generator = ReportMetadataGenerator()
        self.start_time = datetime.now() - timedelta(hours=1)

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.metadata_cache == {}

    def test_generate_metadata_basic(self):
        """Test basic metadata generation."""
        state = self._create_test_state()

        metadata = self.generator.generate_metadata(state)

        assert isinstance(metadata, dict)
        assert "session_id" in metadata
        assert "agent_metrics" in metadata
        assert "topic_analytics" in metadata
        assert "voting_statistics" in metadata
        assert "discussion_dynamics" in metadata
        assert "content_statistics" in metadata
        assert self.generator.metadata_cache == metadata

    def test_extract_session_info(self):
        """Test session information extraction."""
        state = {
            "session_id": "test-123",
            "start_time": self.start_time,
            "main_topic": "AI Development",
            "current_phase": 3,
            "total_messages": 100,
            "completed_topics": ["Topic 1", "Topic 2"],
            "current_round": 5,
        }

        info = self.generator._extract_session_info(state)

        assert info["session_id"] == "test-123"
        assert info["main_topic"] == "AI Development"
        assert info["current_phase"] == 3
        assert info["total_messages"] == 100
        assert info["topics_discussed"] == 2
        assert info["current_round"] == 5
        assert info["duration_minutes"] > 0
        assert info["start_time"] == self.start_time.isoformat()

    def test_calculate_agent_metrics(self):
        """Test agent metrics calculation."""
        state = {
            "agents": {
                "agent-1": {
                    "model": "gpt-4",
                    "provider": "OpenAI",
                    "role": "participant",
                },
                "agent-2": {
                    "model": "claude-3",
                    "provider": "Anthropic",
                    "role": "participant",
                },
                "moderator": {
                    "model": "gemini-pro",
                    "provider": "Google",
                    "role": "moderator",
                },
            },
            "messages_by_agent": {
                "agent-1": 30,
                "agent-2": 25,
                "moderator": 45,
            },
            "total_messages": 100,
            "messages": [
                {"speaker_id": "agent-1", "content": "A" * 100},
                {"speaker_id": "agent-1", "content": "B" * 200},
                {"speaker_id": "agent-2", "content": "C" * 150},
            ],
        }

        metrics = self.generator._calculate_agent_metrics(state)

        assert metrics["total_agents"] == 3
        assert metrics["participating_agents"] == 3
        assert metrics["moderator_count"] == 1
        assert len(metrics["agent_details"]) == 3

        # Check individual agent metrics
        agent1_metrics = next(
            a for a in metrics["agent_details"] if a["agent_id"] == "agent-1"
        )
        assert agent1_metrics["message_count"] == 30
        assert agent1_metrics["participation_rate"] == 30.0
        assert agent1_metrics["average_message_length"] == 150

    def test_analyze_topics(self):
        """Test topic analytics."""
        state = {
            "topics_info": {
                "Topic 1": {
                    "start_time": self.start_time,
                    "end_time": self.start_time + timedelta(minutes=30),
                    "message_count": 20,
                    "proposed_by": "agent-1",
                    "status": "completed",
                },
                "Topic 2": {
                    "start_time": self.start_time + timedelta(minutes=30),
                    "end_time": self.start_time + timedelta(minutes=45),
                    "message_count": 15,
                    "proposed_by": "agent-2",
                    "status": "completed",
                },
                "Topic 3": {
                    "proposed_by": "agent-3",
                    "status": "skipped",
                },
            },
            "completed_topics": ["Topic 1", "Topic 2"],
            "proposed_topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4"],
            "messages_by_topic": {
                "Topic 1": 20,
                "Topic 2": 15,
            },
            "messages": [
                {"topic": "Topic 1", "speaker_id": "agent-1"},
                {"topic": "Topic 1", "speaker_id": "agent-2"},
                {"topic": "Topic 2", "speaker_id": "agent-1"},
            ],
        }

        analytics = self.generator._analyze_topics(state)

        assert analytics["total_topics_proposed"] == 4
        assert analytics["topics_discussed"] == 2
        assert analytics["topics_skipped"] == 1
        # Average is calculated across all topics including skipped (30 + 15 + 0) / 3 = 15
        assert analytics["average_topic_duration"] == 15.0
        assert len(analytics["topic_details"]) == 3

    def test_calculate_voting_stats(self):
        """Test voting statistics calculation."""
        state = {
            "vote_history": [
                {
                    "vote_type": "topic_selection",
                    "phase": 1,
                    "required_votes": 5,
                    "received_votes": 5,
                    "result": "Topic A",
                },
                {
                    "vote_type": "continue_discussion",
                    "phase": 2,
                    "required_votes": 5,
                    "received_votes": 4,
                    "result": "continue",
                },
            ],
            "votes": [
                {"vote_type": "topic_selection", "phase": 1, "choice": "Topic A"},
                {"vote_type": "topic_selection", "phase": 1, "choice": "Topic A"},
                {"vote_type": "topic_selection", "phase": 1, "choice": "Topic A"},
                {"vote_type": "topic_selection", "phase": 1, "choice": "Topic A"},
                {"vote_type": "topic_selection", "phase": 1, "choice": "Topic A"},
                {"vote_type": "continue_discussion", "phase": 2, "choice": "yes"},
                {"vote_type": "continue_discussion", "phase": 2, "choice": "yes"},
                {"vote_type": "continue_discussion", "phase": 2, "choice": "yes"},
                {"vote_type": "continue_discussion", "phase": 2, "choice": "no"},
            ],
        }

        stats = self.generator._calculate_voting_stats(state)

        assert stats["total_voting_rounds"] == 2
        assert stats["vote_type_distribution"]["topic_selection"] == 1
        assert stats["vote_type_distribution"]["continue_discussion"] == 1
        assert stats["average_participation_rate"] == 90.0  # (100 + 80) / 2
        assert stats["unanimous_votes"] == 1
        assert len(stats["voting_round_details"]) == 2

    def test_analyze_discussion_dynamics(self):
        """Test discussion dynamics analysis."""
        base_time = datetime.now()
        state = {
            "messages": [
                {"timestamp": base_time},
                {"timestamp": base_time + timedelta(minutes=2)},
                {"timestamp": base_time + timedelta(minutes=7)},
                {"timestamp": base_time + timedelta(minutes=12)},
            ],
            "round_history": [
                {
                    "round_number": 1,
                    "topic": "Topic 1",
                    "message_count": 10,
                    "participants": ["agent-1", "agent-2"],
                    "summary": "Round 1 summary",
                },
                {
                    "round_number": 2,
                    "topic": "Topic 1",
                    "message_count": 8,
                    "participants": ["agent-1", "agent-2", "agent-3"],
                    "summary": None,
                },
            ],
            "turn_order_history": [
                ["agent-1", "agent-2", "agent-3"],
                ["agent-2", "agent-3", "agent-1"],
            ],
        }

        dynamics = self.generator._analyze_discussion_dynamics(state)

        assert dynamics["rounds_completed"] == 2
        assert dynamics["average_messages_per_round"] == 9.0
        assert dynamics["turn_order_rotations"] == 2
        assert len(dynamics["round_statistics"]) == 2
        assert dynamics["message_distribution"][0] == 2  # First 5-min interval
        assert dynamics["message_distribution"][1] == 1  # Second interval
        assert dynamics["message_distribution"][2] == 1  # Third interval

    def test_calculate_content_stats(self):
        """Test content statistics calculation."""
        state = {
            "messages": [
                {"content": "A" * 100},
                {"content": "B" * 200},
                {"content": "C" * 150},
            ],
        }

        topic_summaries = {
            "Topic 1": "X" * 500,
            "Topic 2": "Y" * 300,
        }

        stats = self.generator._calculate_content_stats(state, topic_summaries)

        assert stats["total_characters"] == 450
        assert stats["average_message_length"] == 150
        assert stats["longest_message"] == 200
        assert stats["shortest_message"] == 100
        assert stats["estimated_words"] == 90  # 450 / 5
        assert stats["estimated_tokens"] == 112  # 450 / 4

        assert stats["summary_statistics"]["total_summaries"] == 2
        assert stats["summary_statistics"]["total_summary_length"] == 800
        assert stats["summary_statistics"]["average_summary_length"] == 400

    def test_export_metadata_json(self, tmp_path):
        """Test metadata export to JSON."""
        # Generate metadata first
        state = self._create_test_state()
        self.generator.generate_metadata(state)

        # Export
        output_path = tmp_path / "metadata.json"
        exported_path = self.generator.export_metadata(output_path)

        assert exported_path == output_path
        assert output_path.exists()

        # Verify content
        loaded = json.loads(output_path.read_text())
        assert "session_id" in loaded
        assert "agent_metrics" in loaded

    def test_export_metadata_no_raw_data(self, tmp_path):
        """Test metadata export without raw data."""
        state = self._create_test_state()
        self.generator.generate_metadata(state)

        output_path = tmp_path / "metadata.json"
        self.generator.export_metadata(output_path, include_raw_data=False)

        loaded = json.loads(output_path.read_text())

        # Check that detailed data is removed
        assert "agent_details" not in loaded.get("agent_metrics", {})
        assert "topic_details" not in loaded.get("topic_analytics", {})
        assert "voting_round_details" not in loaded.get("voting_statistics", {})

    def test_export_metadata_no_cache(self):
        """Test export when no metadata is cached."""
        with pytest.raises(ValueError, match="No metadata generated"):
            self.generator.export_metadata(Path("test.json"))

    def test_generate_analytics_summary(self):
        """Test analytics summary generation."""
        # Generate metadata
        state = self._create_test_state()
        self.generator.generate_metadata(state)

        summary = self.generator.generate_analytics_summary()

        assert "# Session Analytics Summary" in summary
        assert "**Session ID**:" in summary
        assert "## Participation" in summary
        assert "## Topics" in summary
        assert "## Voting" in summary
        assert "## Content" in summary

    def test_generate_analytics_summary_no_metadata(self):
        """Test summary when no metadata available."""
        summary = self.generator.generate_analytics_summary()
        assert summary == "No metadata available"

    def test_error_handling(self):
        """Test error handling in metadata generation."""
        # Create state that will cause errors
        bad_state = {"invalid": "state"}

        metadata = self.generator.generate_metadata(bad_state)

        # Should return empty dict on error
        assert metadata == {}

    def _create_test_state(self) -> VirtualAgoraState:
        """Create a test state for metadata generation."""
        return {
            "session_id": "test-session",
            "start_time": self.start_time,
            "current_phase": 2,
            "total_messages": 50,
            "agents": {
                "agent-1": {
                    "model": "gpt-4",
                    "provider": "OpenAI",
                    "role": "participant",
                },
                "agent-2": {
                    "model": "claude",
                    "provider": "Anthropic",
                    "role": "participant",
                },
            },
            "messages_by_agent": {"agent-1": 25, "agent-2": 25},
            "messages": [
                {"speaker_id": "agent-1", "content": "Test", "topic": "Topic 1"},
                {"speaker_id": "agent-2", "content": "Response", "topic": "Topic 1"},
            ],
            "topics_info": {
                "Topic 1": {
                    "start_time": self.start_time,
                    "end_time": self.start_time + timedelta(minutes=30),
                    "message_count": 50,
                    "status": "completed",
                }
            },
            "completed_topics": ["Topic 1"],
            "proposed_topics": ["Topic 1", "Topic 2"],
            "vote_history": [
                {
                    "vote_type": "topic_selection",
                    "phase": 1,
                    "required_votes": 2,
                    "received_votes": 2,
                }
            ],
            "votes": [],
            "round_history": [],
            "turn_order_history": [],
        }
