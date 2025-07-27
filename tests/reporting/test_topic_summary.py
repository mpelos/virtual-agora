"""Tests for topic summary generation."""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

from virtual_agora.reporting.topic_summary import TopicSummaryGenerator
from virtual_agora.state.schema import VirtualAgoraState


class TestTopicSummaryGenerator:
    """Test TopicSummaryGenerator functionality."""

    def setup_method(self):
        """Set up test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "summaries"
        self.generator = TopicSummaryGenerator(self.output_dir)

    def teardown_method(self):
        """Clean up test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.output_dir == self.output_dir
        assert self.output_dir.exists()

    def test_generate_summary_basic(self):
        """Test basic summary generation."""
        # Create test state
        start_time = datetime.now()
        state = {
            "session_id": "test-session",
            "topics_info": {
                "Test Topic": {
                    "start_time": start_time,
                    "end_time": datetime.now(),
                    "message_count": 10,
                    "proposed_by": "agent-1",
                    "status": "completed",
                }
            },
            "messages": [
                {
                    "topic": "Test Topic",
                    "speaker_id": "agent-1",
                    "speaker_role": "participant",
                    "content": "Test message",
                }
            ],
            "round_history": [{"topic": "Test Topic", "round_number": 1}],
            "vote_history": [],
        }

        # Generate summary
        summary_path = self.generator.generate_summary(
            "Test Topic", "This is a test summary content.", state
        )

        # Verify file was created
        assert summary_path.exists()
        assert summary_path.suffix == ".md"

        # Verify content
        content = summary_path.read_text()
        assert "# Topic Summary: Test Topic" in content
        assert "This is a test summary content." in content
        assert "## Metadata" in content
        assert "Duration" in content
        assert "Number of Rounds" in content

    def test_extract_topic_metadata(self):
        """Test metadata extraction."""
        start_time = datetime.now()
        state = {
            "topics_info": {
                "Test Topic": {
                    "start_time": start_time,
                    "end_time": datetime.now(),
                    "message_count": 15,
                    "proposed_by": "agent-1",
                    "status": "completed",
                }
            },
            "messages": [
                {
                    "topic": "Test Topic",
                    "speaker_id": "agent-1",
                    "speaker_role": "participant",
                },
                {
                    "topic": "Test Topic",
                    "speaker_id": "agent-2",
                    "speaker_role": "participant",
                },
            ],
            "round_history": [
                {"topic": "Test Topic"},
                {"topic": "Test Topic"},
            ],
            "vote_history": [],
        }

        metadata = self.generator._extract_topic_metadata("Test Topic", state)

        assert metadata["topic"] == "Test Topic"
        assert metadata["message_count"] == 15
        assert metadata["number_of_rounds"] == 2
        assert metadata["participant_count"] == 2
        assert metadata["proposed_by"] == "agent-1"
        assert metadata["status"] == "completed"
        assert "duration_minutes" in metadata

    def test_format_summary(self):
        """Test summary formatting."""
        metadata = {
            "topic": "Test Topic",
            "duration_minutes": 30.5,
            "number_of_rounds": 5,
            "message_count": 25,
            "participant_count": 3,
            "proposed_by": "moderator",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "status": "completed",
        }

        formatted = self.generator._format_summary(
            "Test Topic", "Summary content here.", metadata
        )

        # Check formatting
        assert "# Topic Summary: Test Topic" in formatted
        assert "## Metadata" in formatted
        assert "- **Duration**: 30.5 minutes" in formatted
        assert "- **Number of Rounds**: 5" in formatted
        assert "- **Total Messages**: 25" in formatted
        assert "- **Participants**: 3 agents" in formatted
        assert "## Summary" in formatted
        assert "Summary content here." in formatted

    def test_save_summary_filename_sanitization(self):
        """Test filename sanitization."""
        # Topic with special characters
        topic = "Complex Topic: With Special Characters! & More?"

        summary_path = self.generator._save_summary(topic, "Test content")

        # Verify filename is safe
        assert summary_path.exists()
        filename = summary_path.name
        assert ":" not in filename
        assert "?" not in filename
        assert "!" not in filename
        assert "&" not in filename

    def test_get_all_summaries(self):
        """Test retrieving all summaries."""
        # Generate multiple summaries
        for i in range(3):
            self.generator._save_summary(f"Topic {i}", f"Content {i}")

        summaries = self.generator.get_all_summaries()

        assert len(summaries) == 3
        assert all(p.suffix == ".md" for p in summaries)
        assert all("topic_summary_" in p.name for p in summaries)

    def test_load_summary(self):
        """Test loading a summary file."""
        # Create a summary
        topic = "Test Loading"
        content = f"# Topic Summary: {topic}\n\nTest content"
        summary_path = self.generator._save_summary(topic, content)

        # Load it back
        loaded = self.generator.load_summary(summary_path)

        assert loaded["topic"] == topic
        assert loaded["content"] == content
        assert loaded["file_path"] == summary_path

    def test_empty_state_handling(self):
        """Test handling of empty or minimal state."""
        state = {
            "topics_info": {},
            "messages": [],
            "round_history": [],
            "vote_history": [],
        }

        metadata = self.generator._extract_topic_metadata("Unknown Topic", state)

        # Should handle gracefully
        assert metadata["topic"] == "Unknown Topic"
        assert metadata["duration_minutes"] == 0
        assert metadata["number_of_rounds"] == 0
        assert metadata["message_count"] == 0
        assert metadata["participant_count"] == 0

    def test_voting_statistics_extraction(self):
        """Test extraction of voting statistics."""
        # Create timestamps so message is after vote
        vote_time = datetime.now() - timedelta(minutes=20)
        message_time = datetime.now() - timedelta(minutes=15)  # After vote

        state = {
            "topics_info": {
                "Test Topic": {
                    "message_count": 10,
                }
            },
            "messages": [
                {
                    "topic": "Test Topic",
                    "timestamp": message_time,  # Message after vote
                    "speaker_id": "agent-1",
                }
            ],
            "vote_history": [
                {
                    "vote_type": "continue_discussion",
                    "start_time": vote_time,  # Vote before message
                    "result": "continue",
                }
            ],
            "round_history": [],
        }

        metadata = self.generator._extract_topic_metadata("Test Topic", state)

        assert "voting_rounds" in metadata
        assert metadata["voting_rounds"] == 1
        assert metadata["final_vote_result"] == "continue"
