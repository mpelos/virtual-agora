"""Tests for custom state reducers."""

import pytest
from datetime import datetime

from virtual_agora.state.reducers import (
    merge_hitl_state,
    merge_flow_control,
    update_rounds_per_topic,
    merge_topic_summaries,
    merge_phase_summaries,
    increment_counter,
    merge_statistics,
    merge_agent_info,
    update_topic_info,
)


class TestStateReducers:
    """Test custom state reducers."""

    def test_merge_hitl_state_none_existing(self):
        """Test merging HITL state with no existing state."""
        updates = {
            "awaiting_approval": True,
            "approval_type": "agenda",
            "prompt_message": "Please approve",
        }

        result = merge_hitl_state(None, updates)

        assert result["awaiting_approval"] == True
        assert result["approval_type"] == "agenda"
        assert result["prompt_message"] == "Please approve"
        assert result["approval_history"] == []

    def test_merge_hitl_state_with_existing(self):
        """Test merging HITL state with existing state."""
        existing = {
            "awaiting_approval": False,
            "approval_type": None,
            "prompt_message": None,
            "options": None,
            "approval_history": [{"old": "entry"}],
        }

        updates = {
            "awaiting_approval": True,
            "approval_type": "continuation",
            "approval_history": [{"new": "entry"}],
        }

        result = merge_hitl_state(existing, updates)

        assert result["awaiting_approval"] == True
        assert result["approval_type"] == "continuation"
        assert len(result["approval_history"]) == 2
        assert {"old": "entry"} in result["approval_history"]
        assert {"new": "entry"} in result["approval_history"]

    def test_merge_flow_control_none_existing(self):
        """Test merging flow control with no existing state."""
        updates = {"max_rounds_per_topic": 15, "auto_conclude_threshold": 5}

        result = merge_flow_control(None, updates)

        assert result["max_rounds_per_topic"] == 15
        assert result["auto_conclude_threshold"] == 5
        assert result["context_window_limit"] == 8000  # Default
        assert result["cycle_detection_enabled"] == True  # Default

    def test_merge_flow_control_with_existing(self):
        """Test merging flow control with existing state."""
        existing = {
            "max_rounds_per_topic": 10,
            "auto_conclude_threshold": 3,
            "context_window_limit": 8000,
            "cycle_detection_enabled": True,
            "max_iterations_per_phase": 5,
        }

        updates = {"max_rounds_per_topic": 12, "context_window_limit": 10000}

        result = merge_flow_control(existing, updates)

        assert result["max_rounds_per_topic"] == 12
        assert result["auto_conclude_threshold"] == 3  # Unchanged
        assert result["context_window_limit"] == 10000
        assert result["cycle_detection_enabled"] == True  # Unchanged

    def test_update_rounds_per_topic_none_existing(self):
        """Test updating rounds per topic with no existing data."""
        updates = {"Topic 1": 1, "Topic 2": 2}

        result = update_rounds_per_topic(None, updates)

        assert result == {"Topic 1": 1, "Topic 2": 2}

    def test_update_rounds_per_topic_with_existing(self):
        """Test updating rounds per topic with existing data."""
        existing = {"Topic 1": 2, "Topic 3": 1}
        updates = {"Topic 1": 1, "Topic 2": 3}

        result = update_rounds_per_topic(existing, updates)

        assert result["Topic 1"] == 3  # 2 + 1
        assert result["Topic 2"] == 3  # New topic
        assert result["Topic 3"] == 1  # Unchanged

    def test_merge_topic_summaries(self):
        """Test merging topic summaries."""
        existing = {"Topic 1": "Summary 1", "Topic 2": "Summary 2"}
        updates = {"Topic 2": "Updated Summary 2", "Topic 3": "Summary 3"}

        result = merge_topic_summaries(existing, updates)

        assert result["Topic 1"] == "Summary 1"  # Unchanged
        assert result["Topic 2"] == "Updated Summary 2"  # Updated
        assert result["Topic 3"] == "Summary 3"  # New

    def test_merge_phase_summaries(self):
        """Test merging phase summaries."""
        existing = {1: "Phase 1 summary", 2: "Phase 2 summary"}
        updates = {2: "Updated Phase 2 summary", 3: "Phase 3 summary"}

        result = merge_phase_summaries(existing, updates)

        assert result[1] == "Phase 1 summary"  # Unchanged
        assert result[2] == "Updated Phase 2 summary"  # Updated
        assert result[3] == "Phase 3 summary"  # New

    def test_increment_counter_none_existing(self):
        """Test incrementing counter with no existing value."""
        result = increment_counter(None, 5)
        assert result == 5

    def test_increment_counter_with_existing(self):
        """Test incrementing counter with existing value."""
        result = increment_counter(10, 3)
        assert result == 13

    def test_increment_counter_none_increment(self):
        """Test incrementing counter with no increment value."""
        result = increment_counter(10, None)
        assert result == 10

    def test_merge_statistics_none_existing(self):
        """Test merging statistics with no existing data."""
        updates = {"messages": 5, "votes": 2}

        result = merge_statistics(None, updates)

        assert result == {"messages": 5, "votes": 2}

    def test_merge_statistics_with_existing(self):
        """Test merging statistics with existing data."""
        existing = {"messages": 10, "rounds": 3}
        updates = {"messages": 5, "votes": 2}

        result = merge_statistics(existing, updates)

        assert result["messages"] == 15  # 10 + 5
        assert result["rounds"] == 3  # Unchanged
        assert result["votes"] == 2  # New

    def test_merge_agent_info_none_existing(self):
        """Test merging agent info with no existing data."""
        updates = {"agent1": {"id": "agent1", "model": "gpt-4o", "message_count": 5}}

        result = merge_agent_info(None, updates)

        assert result["agent1"]["id"] == "agent1"
        assert result["agent1"]["message_count"] == 5

    def test_merge_agent_info_with_existing(self):
        """Test merging agent info with existing data."""
        existing = {
            "agent1": {
                "id": "agent1",
                "model": "gpt-4o",
                "message_count": 3,
                "role": "participant",
            }
        }

        updates = {
            "agent1": {"message_count": 2, "last_active": datetime.now()},
            "agent2": {"id": "agent2", "model": "claude-3", "message_count": 1},
        }

        result = merge_agent_info(existing, updates)

        assert result["agent1"]["id"] == "agent1"
        assert result["agent1"]["model"] == "gpt-4o"
        assert result["agent1"]["role"] == "participant"
        assert result["agent1"]["message_count"] == 5  # 3 + 2
        assert "last_active" in result["agent1"]

        assert result["agent2"]["id"] == "agent2"
        assert result["agent2"]["message_count"] == 1

    def test_update_topic_info_none_existing(self):
        """Test updating topic info with no existing data."""
        updates = {
            "Topic 1": {"topic": "Topic 1", "status": "active", "message_count": 5}
        }

        result = update_topic_info(None, updates)

        assert result["Topic 1"]["topic"] == "Topic 1"
        assert result["Topic 1"]["status"] == "active"
        assert result["Topic 1"]["message_count"] == 5

    def test_update_topic_info_with_existing(self):
        """Test updating topic info with existing data."""
        existing = {
            "Topic 1": {
                "topic": "Topic 1",
                "status": "proposed",
                "message_count": 2,
                "proposed_by": "agent1",
            }
        }

        updates = {
            "Topic 1": {
                "status": "active",
                "message_count": 3,
                "start_time": datetime.now(),
            },
            "Topic 2": {"topic": "Topic 2", "status": "proposed", "message_count": 1},
        }

        result = update_topic_info(existing, updates)

        assert result["Topic 1"]["topic"] == "Topic 1"
        assert result["Topic 1"]["status"] == "active"  # Updated
        assert result["Topic 1"]["proposed_by"] == "agent1"  # Unchanged
        assert result["Topic 1"]["message_count"] == 5  # 2 + 3
        assert "start_time" in result["Topic 1"]

        assert result["Topic 2"]["topic"] == "Topic 2"
        assert result["Topic 2"]["message_count"] == 1

    def test_reducers_with_none_updates(self):
        """Test that all reducers handle None updates gracefully."""
        existing_data = {"test": "data"}

        # Test all reducers with None updates
        assert merge_hitl_state(existing_data, None) is not None
        assert merge_flow_control(existing_data, None) is not None
        assert update_rounds_per_topic(existing_data, None) == existing_data
        assert merge_topic_summaries(existing_data, None) == existing_data
        assert merge_phase_summaries(existing_data, None) == existing_data
        assert increment_counter(5, None) == 5
        assert merge_statistics(existing_data, None) == existing_data
        assert merge_agent_info(existing_data, None) == existing_data
        assert update_topic_info(existing_data, None) == existing_data
