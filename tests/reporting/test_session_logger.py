"""Tests for enhanced session logging."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from virtual_agora.reporting.session_logger import (
    EnhancedSessionLogger,
    EventType,
    LogLevel,
)


class TestEnhancedSessionLogger:
    """Test EnhancedSessionLogger functionality."""

    def setup_method(self):
        """Set up test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.session_id = "test-session-123"
        self.logger = EnhancedSessionLogger(
            self.session_id,
            self.log_dir,
            enable_compression=False,
            max_log_size_mb=1,
        )

    def teardown_method(self):
        """Clean up test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test logger initialization."""
        assert self.logger.session_id == self.session_id
        assert self.logger.log_dir == self.log_dir
        assert self.log_dir.exists()
        assert self.logger.structured_log_path.exists()
        assert not self.logger.enable_compression
        assert self.logger.max_log_size_bytes == 1024 * 1024

    def test_structured_log_initialization(self):
        """Test structured log file initialization."""
        # Read first event
        with open(self.logger.structured_log_path, "r") as f:
            first_line = f.readline()
            event = json.loads(first_line)

        assert event["event_type"] == "SESSION_START"
        assert event["session_id"] == self.session_id
        assert "timestamp" in event
        assert event["log_version"] == "1.0"

    def test_log_event_basic(self):
        """Test basic event logging."""
        self.logger.log_event(
            EventType.AGENT_RESPONSE,
            "agent-1",
            "This is a test response",
            metadata={"round": 1},
            level=LogLevel.INFO,
        )

        # Check event was logged
        assert self.logger.event_counts[EventType.AGENT_RESPONSE] == 1

        # Read from structured log
        events = self._read_structured_events()
        assert len(events) >= 2  # Session start + our event

        # Check our event
        agent_event = events[-1]
        assert agent_event["event_type"] == "AGENT_RESPONSE"
        assert agent_event["actor"] == "agent-1"
        assert agent_event["content"] == "This is a test response"
        assert agent_event["metadata"]["round"] == 1
        assert agent_event["level"] == LogLevel.INFO.value

    def test_log_event_with_dict_content(self):
        """Test event logging with dictionary content."""
        content_dict = {"action": "vote", "choice": "yes"}

        self.logger.log_event(
            EventType.VOTING_EVENT,
            "agent-2",
            content_dict,
        )

        events = self._read_structured_events()
        vote_event = events[-1]

        assert vote_event["event_type"] == "VOTING_EVENT"
        assert json.loads(vote_event["content"]) == content_dict

    def test_log_user_input(self):
        """Test user input logging."""
        self.logger.log_user_input("Enter topic:", "AI Ethics")

        assert self.logger.event_counts[EventType.USER_INPUT] == 1

        events = self._read_structured_events()
        user_event = next(e for e in events if e["event_type"] == "USER_INPUT")

        content = json.loads(user_event["content"])
        assert content["prompt"] == "Enter topic:"
        assert content["response"] == "AI Ethics"

    def test_log_agent_response(self):
        """Test agent response logging."""
        self.logger.log_agent_response("gpt-4", "I think we should discuss ethics.")

        assert self.logger.event_counts[EventType.AGENT_RESPONSE] == 1

    def test_log_system_event(self):
        """Test system event logging."""
        self.logger.log_system_event("Phase transition", "Moving to voting phase")

        assert (
            self.logger.event_counts[EventType.SYSTEM_EVENT] == 2
        )  # +1 for session start

    def test_log_voting_event(self):
        """Test voting event logging."""
        self.logger.log_voting_event(
            "topic_selection",
            "agent-1",
            "Topic A",
            metadata={"round": 1, "phase": 2},
        )

        assert self.logger.event_counts[EventType.VOTING_EVENT] == 1

        events = self._read_structured_events()
        vote_event = next(e for e in events if e["event_type"] == "VOTING_EVENT")

        content = json.loads(vote_event["content"])
        assert content["vote_type"] == "topic_selection"
        assert content["choice"] == "Topic A"
        assert vote_event["metadata"]["round"] == 1

    def test_log_phase_transition(self):
        """Test phase transition logging."""
        self.logger.log_phase_transition(1, 2, "Agenda approved")

        events = self._read_structured_events()
        transition_event = next(
            e for e in events if e["event_type"] == "PHASE_TRANSITION"
        )

        content = json.loads(transition_event["content"])
        assert content["from_phase"] == 1
        assert content["to_phase"] == 2
        assert content["reason"] == "Agenda approved"

    def test_log_topic_change(self):
        """Test topic change logging."""
        self.logger.log_topic_change(
            "Old Topic",
            "New Topic",
            metadata={"vote_result": "passed"},
        )

        events = self._read_structured_events()
        topic_event = next(e for e in events if e["event_type"] == "TOPIC_CHANGE")

        content = json.loads(topic_event["content"])
        assert content["old_topic"] == "Old Topic"
        assert content["new_topic"] == "New Topic"
        assert topic_event["metadata"]["vote_result"] == "passed"

    def test_log_error(self):
        """Test error logging."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            self.logger.log_error("Something went wrong", e)

        assert self.logger.event_counts[EventType.ERROR] == 1

        events = self._read_structured_events()
        error_event = next(e for e in events if e["event_type"] == "ERROR")

        content = json.loads(error_event["content"])
        assert content["error"] == "Something went wrong"
        assert "ValueError: Test error" in content["exception"]
        assert error_event["level"] == LogLevel.ERROR.value

    def test_log_state_snapshot(self):
        """Test state snapshot logging."""
        state = {
            "session_id": "test-123",
            "current_phase": 2,
            "active_topic": "AI Ethics",
            "current_round": 3,
            "total_messages": 50,
            "agents": {"agent-1": {}, "agent-2": {}},
            "completed_topics": ["Topic 1", "Topic 2"],
        }

        self.logger.log_state_snapshot(state)

        events = self._read_structured_events()
        snapshot_event = next(
            e
            for e in events
            if e["event_type"] == "SYSTEM_EVENT" and "state_snapshot" in e["content"]
        )

        content = json.loads(snapshot_event["content"])
        snapshot = content["snapshot"]
        assert snapshot["session_id"] == "test-123"
        assert snapshot["current_phase"] == 2
        assert snapshot["agent_count"] == 2
        assert snapshot["topics_completed"] == 2

    def test_get_session_analytics(self):
        """Test session analytics generation."""
        # Log various events
        self.logger.log_user_input("Topic?", "AI")
        self.logger.log_agent_response("agent-1", "Response 1")
        self.logger.log_agent_response("agent-2", "Response 2")
        self.logger.log_topic_change(None, "AI Ethics")
        self.logger.log_topic_change("AI Ethics", "AI Safety")
        self.logger.log_error("Test error")

        analytics = self.logger.get_session_analytics()

        assert analytics["session_id"] == self.session_id
        assert analytics["total_events"] >= 7
        assert analytics["event_counts"][EventType.AGENT_RESPONSE.value] == 2
        assert analytics["event_counts"][EventType.ERROR.value] == 1
        assert "agent-1" in analytics["agents_active"]
        assert "agent-2" in analytics["agents_active"]
        assert "AI Ethics" in analytics["topics_discussed"]
        assert "AI Safety" in analytics["topics_discussed"]
        assert analytics["error_count"] == 1
        assert analytics["warning_count"] == 0
        assert analytics["duration_seconds"] is not None

    def test_export_logs_json(self, tmp_path):
        """Test log export in JSON format."""
        # Log some events
        self.logger.log_agent_response("agent-1", "Test")
        self.logger.log_system_event("Test event")

        # Export
        exported = self.logger.export_logs(tmp_path, format="json")

        assert len(exported) == 2

        # Check structured log was copied
        events_file = tmp_path / f"session_{self.session_id}_events.json"
        assert events_file in exported
        assert events_file.exists()

        # Check analytics file
        analytics_file = tmp_path / f"session_{self.session_id}_analytics.json"
        assert analytics_file in exported
        assert analytics_file.exists()

        # Verify analytics content
        analytics = json.loads(analytics_file.read_text())
        assert analytics["session_id"] == self.session_id

    def test_export_logs_txt(self, tmp_path):
        """Test log export in text format."""
        # Create standard log file
        standard_log = self.logger.log_dir / f"session_{self.session_id}.log"
        standard_log.write_text("Test log content")

        exported = self.logger.export_logs(tmp_path, format="txt")

        assert len(exported) == 1
        assert exported[0].name == standard_log.name
        assert exported[0].read_text() == "Test log content"

    def test_close_logger(self):
        """Test logger closing."""
        # Log some events
        self.logger.log_agent_response("agent-1", "Test")

        # Close logger
        self.logger.close()

        # Check final event was logged
        events = self._read_structured_events()
        final_event = events[-1]

        assert final_event["event_type"] == "SYSTEM_EVENT"
        assert final_event["content"] == "SESSION_END"
        assert "session_id" in final_event["metadata"]
        assert "total_events" in final_event["metadata"]

    def test_event_numbering(self):
        """Test event numbering."""
        # Log multiple events
        for i in range(5):
            self.logger.log_event(
                EventType.AGENT_RESPONSE,
                f"agent-{i}",
                f"Response {i}",
            )

        events = self._read_structured_events()

        # Check event numbers are sequential
        event_numbers = [e.get("event_number") for e in events if "event_number" in e]
        assert event_numbers == list(range(1, len(event_numbers) + 1))

    def test_log_levels(self):
        """Test different log levels."""
        levels = [
            (LogLevel.DEBUG, "Debug message"),
            (LogLevel.INFO, "Info message"),
            (LogLevel.WARNING, "Warning message"),
            (LogLevel.ERROR, "Error message"),
            (LogLevel.CRITICAL, "Critical message"),
        ]

        for level, message in levels:
            self.logger.log_event(
                EventType.SYSTEM_EVENT,
                "system",
                message,
                level=level,
            )

        events = self._read_structured_events()

        # Check levels were recorded correctly
        for level, message in levels:
            event = next(e for e in events if e.get("content") == message)
            assert event["level"] == level.value

    def _read_structured_events(self):
        """Helper to read all structured events."""
        events = []
        with open(self.logger.structured_log_path, "r") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events
