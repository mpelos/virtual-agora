"""Tests for context window management."""

import pytest
from datetime import datetime

from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.state.schema import VirtualAgoraState, Message


class TestContextWindowManager:
    """Test context window management functionality."""

    def setup_method(self):
        """Set up test method."""
        self.manager = ContextWindowManager(window_limit=1000)

        # Create base state for testing
        self.base_state = {
            "session_id": "test-session",
            "start_time": datetime.now(),
            "config_hash": "test-hash",
            "current_phase": 2,
            "phase_history": [],
            "phase_start_time": datetime.now(),
            "current_round": 1,
            "round_history": [],
            "turn_order_history": [],
            "rounds_per_topic": {},
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": None,
                "prompt_message": None,
                "options": None,
                "approval_history": [],
            },
            "flow_control": {
                "max_rounds_per_topic": 10,
                "auto_conclude_threshold": 3,
                "context_window_limit": 1000,
                "cycle_detection_enabled": True,
                "max_iterations_per_phase": 5,
            },
            "active_topic": "Test Topic",
            "topic_queue": [],
            "proposed_topics": ["Test Topic"],
            "topics_info": {},
            "completed_topics": [],
            "agents": {},
            "moderator_id": "moderator",
            "current_speaker_id": "agent1",
            "speaking_order": ["agent1", "agent2"],
            "next_speaker_index": 0,
            "messages": [],
            "last_message_id": "msg_000001",
            "active_vote": None,
            "vote_history": [],
            "votes": [],
            "consensus_proposals": {},
            "consensus_reached": {},
            "phase_summaries": {},
            "topic_summaries": {},
            "consensus_summaries": {},
            "final_report": None,
            "total_messages": 0,
            "messages_by_phase": {},
            "messages_by_agent": {},
            "messages_by_topic": {},
            "vote_participation_rate": {},
            "tool_calls": [],
            "active_tool_calls": {},
            "tool_metrics": {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_execution_time_ms": 0.0,
                "calls_by_tool": {},
                "calls_by_agent": {},
                "errors_by_type": {},
            },
            "tools_enabled_agents": [],
            "last_error": None,
            "error_count": 0,
            "warnings": [],
        }

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Test empty string
        assert self.manager.estimate_tokens("") == 1

        # Test short text
        tokens = self.manager.estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens <= 5  # Should be around 2-3 tokens

        # Test longer text
        long_text = "This is a longer piece of text that should have more tokens than the short example"
        long_tokens = self.manager.estimate_tokens(long_text)
        assert long_tokens > tokens

    def test_count_message_tokens(self):
        """Test message token counting."""
        messages = [
            {
                "id": "msg_001",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Hello everyone!",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Test Topic",
            },
            {
                "id": "msg_002",
                "speaker_id": "agent2",
                "speaker_role": "participant",
                "content": "This is a longer message with more content to test token counting",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Test Topic",
            },
        ]

        total_tokens = self.manager.count_message_tokens(messages)

        # Should include content tokens plus overhead
        assert total_tokens > 40  # At least some tokens from content
        assert total_tokens < 200  # But not excessive

        # Each message should add at least 20 tokens overhead
        assert total_tokens >= len(messages) * 20

    def test_get_context_size(self):
        """Test context size calculation."""
        state = {
            **self.base_state,
            "messages": [
                {
                    "id": "msg_001",
                    "speaker_id": "agent1",
                    "speaker_role": "participant",
                    "content": "Test message",
                    "timestamp": datetime.now(),
                    "phase": 2,
                    "topic": "Test Topic",
                }
            ],
            "phase_summaries": {1: "Phase 1 summary"},
            "topic_summaries": {"Test Topic": "Topic summary"},
            "consensus_summaries": {"Test Topic": "Consensus summary"},
        }

        context_size = self.manager.get_context_size(state)
        assert context_size > 0
        assert context_size < 1000  # Should be reasonable for test data

    def test_needs_compression_below_threshold(self):
        """Test compression need detection below threshold."""
        state = {
            **self.base_state,
            "messages": [
                {
                    "id": "msg_001",
                    "speaker_id": "agent1",
                    "speaker_role": "participant",
                    "content": "Short message",
                    "timestamp": datetime.now(),
                    "phase": 2,
                    "topic": "Test Topic",
                }
            ],
        }

        assert not self.manager.needs_compression(state)

    def test_needs_compression_above_threshold(self):
        """Test compression need detection above threshold."""
        # Create many messages to exceed threshold
        messages = []
        for i in range(100):
            messages.append(
                {
                    "id": f"msg_{i:03d}",
                    "speaker_id": f"agent{i % 2 + 1}",
                    "speaker_role": "participant",
                    "content": f"This is message {i} with some content to increase token count "
                    * 10,
                    "timestamp": datetime.now(),
                    "phase": 2,
                    "topic": "Test Topic",
                }
            )

        state = {**self.base_state, "messages": messages}

        assert self.manager.needs_compression(state)

    def test_compress_messages(self):
        """Test message compression."""
        # Create messages to compress
        messages = []
        for i in range(20):
            messages.append(
                {
                    "id": f"msg_{i:03d}",
                    "speaker_id": f"agent{i % 2 + 1}",
                    "speaker_role": "participant",
                    "content": f"Message {i} content",
                    "timestamp": datetime.now(),
                    "phase": 2 if i < 10 else 3,
                    "topic": "Test Topic" if i < 15 else "Another Topic",
                }
            )

        compressed_messages, summary = self.manager.compress_messages(messages, 300)

        # Should keep recent messages and compress older ones
        assert len(compressed_messages) < len(messages)
        assert len(compressed_messages) > 0
        assert "COMPRESSED" in summary
        assert len(summary) > 0

    def test_compress_context(self):
        """Test full context compression."""
        # Create state that needs compression
        messages = []
        for i in range(50):
            messages.append(
                {
                    "id": f"msg_{i:03d}",
                    "speaker_id": f"agent{i % 2 + 1}",
                    "speaker_role": "participant",
                    "content": f"This is a longer message {i} with more content to trigger compression "
                    * 5,
                    "timestamp": datetime.now(),
                    "phase": 2,
                    "topic": "Test Topic",
                }
            )

        state = {
            **self.base_state,
            "messages": messages,
            "flow_control": {
                **self.base_state["flow_control"],
                "context_window_limit": 500,  # Low limit to force compression
            },
        }

        updates = self.manager.compress_context(state)

        # Should return compression updates
        assert "messages" in updates
        assert len(updates["messages"]) < len(messages)
        assert "warnings" in updates
        assert any("COMPRESSED" in warning for warning in updates["warnings"])

    def test_get_context_stats(self):
        """Test context statistics generation."""
        state = {
            **self.base_state,
            "messages": [
                {
                    "id": "msg_001",
                    "speaker_id": "agent1",
                    "speaker_role": "participant",
                    "content": "Test message",
                    "timestamp": datetime.now(),
                    "phase": 2,
                    "topic": "Test Topic",
                }
            ],
            "phase_summaries": {1: "Summary"},
            "topic_summaries": {"Test Topic": "Topic summary"},
        }

        stats = self.manager.get_context_stats(state)

        # Check required fields
        assert "total_tokens" in stats
        assert "limit" in stats
        assert "usage_percent" in stats
        assert "needs_compression" in stats
        assert "breakdown" in stats
        assert "message_count" in stats

        # Check values are reasonable
        assert stats["total_tokens"] > 0
        assert stats["limit"] == 1000
        assert 0 <= stats["usage_percent"] <= 100
        assert stats["message_count"] == 1

        # Check breakdown
        breakdown = stats["breakdown"]
        assert "messages" in breakdown
        assert "phase_summaries" in breakdown
        assert "topic_summaries" in breakdown
        assert breakdown["messages"] > 0

    def test_create_context_manager(self):
        """Test context manager factory function."""
        from virtual_agora.flow.context_window import create_context_manager

        manager = create_context_manager(2000)
        assert isinstance(manager, ContextWindowManager)
        assert manager.window_limit == 2000
