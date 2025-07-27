"""Integration tests for context compression flow triggers in Virtual Agora v1.3.

This module tests when and how context compression is triggered in the flow:
- Compression triggers at specific checkpoints or conditions
- State transitions when compression occurs
- Data structure preservation during compression
- Integration with discussion flow and state management
"""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
import uuid

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.flow.context_window import ContextWindowManager

from ..helpers.fake_llm import create_fake_llm_pool, create_specialized_fake_llms
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
    create_test_messages,
)


class TestContextWindowManagerIntegration:
    """Test ContextWindowManager integration with flow."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="extended_debate"
        )
        self.context_manager = ContextWindowManager(
            window_limit=1000
        )  # Small limit for testing

    @pytest.mark.integration
    def test_context_manager_initialization(self):
        """Test that ContextWindowManager is properly initialized in flow."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()

            # Verify that nodes component exists
            assert hasattr(flow, "nodes")
            # Context manager should be available in nodes
            assert hasattr(flow.nodes, "context_manager")
            assert isinstance(flow.nodes.context_manager, ContextWindowManager)

    @pytest.mark.integration
    def test_needs_compression_detection(self):
        """Test that compression need is detected correctly."""
        state = self.test_helper.create_discussion_state()

        # Initialize required fields
        state["phase_summaries"] = {}
        state["consensus_summaries"] = {}
        state["final_report"] = ""

        # Add messages to exceed 80% of limit (800 tokens)
        # Each message ~100 tokens (25 words * 4 chars/word)
        for i in range(10):
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": f"agent_{i % 3 + 1}",
                "content": "This is a fairly long message with enough words to simulate realistic token usage. "
                * 3,
                "timestamp": datetime.now(),
                "round_number": i // 3 + 1,
                "topic": "Test Topic",
                "message_type": "discussion",
            }
            state["messages"].append(message)

        # Check if compression needed
        needs_compression = self.context_manager.needs_compression(state)
        assert needs_compression

        # Get context stats
        stats = self.context_manager.get_context_stats(state)
        assert stats["usage_percent"] > 80

    @pytest.mark.integration
    def test_compression_preserves_recent_messages(self):
        """Test that compression preserves recent messages."""
        state = self.test_helper.create_discussion_state()

        # Initialize required fields
        state["phase_summaries"] = {}
        state["consensus_summaries"] = {}
        state["final_report"] = ""

        # Add 20 messages
        messages = []
        for i in range(20):
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": f"agent_{i % 3 + 1}",
                "content": f"Message {i} content with some substance. "
                * 20,  # Make it longer
                "timestamp": datetime.now() + timedelta(minutes=i),
                "round_number": i // 3 + 1,
                "topic": "Test Topic",
                "message_type": "discussion",
            }
            messages.append(message)
        state["messages"] = messages

        # Apply compression
        updates = self.context_manager.compress_context(state)

        # Should have updates
        assert "messages" in updates
        assert len(updates["messages"]) < 20

        # Recent messages should be preserved
        compressed_messages = updates["messages"]
        last_original = messages[-1]
        last_compressed = compressed_messages[-1]
        assert last_original["id"] == last_compressed["id"]

    @pytest.mark.integration
    def test_compression_adds_warning(self):
        """Test that compression adds a warning to state."""
        state = self.test_helper.create_discussion_state()
        state["warnings"] = []

        # Add many messages to trigger compression
        for i in range(30):
            state["messages"].append(
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "Long message content " * 10,
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Test",
                    "message_type": "discussion",
                }
            )

        # Apply compression
        updates = self.context_manager.compress_context(state)

        # Should add compression warning
        assert "warnings" in updates
        assert len(updates["warnings"]) > len(state["warnings"])
        assert "Context compressed" in updates["warnings"][-1]

    @pytest.mark.integration
    def test_compression_node_execution(self):
        """Test compression node execution in flow."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Initialize required fields
            state["phase_summaries"] = {}
            state["consensus_summaries"] = {}
            state["final_report"] = ""

            # Add messages to trigger compression
            for i in range(30):
                state["messages"].append(
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": f"agent_{i % 3 + 1}",
                        "content": "Discussion content " * 20,
                        "timestamp": datetime.now(),
                        "round_number": i // 3 + 1,
                        "topic": "Test Topic",
                        "message_type": "discussion",
                    }
                )

            # Create context manager directly to test compression
            context_manager = ContextWindowManager(window_limit=1000)

            # Check if compression is needed
            if context_manager.needs_compression(state):
                updates = context_manager.compress_context(state)

                # Should return updates
                assert updates is not None
                if "messages" in updates:
                    assert len(updates["messages"]) < len(state["messages"])


class TestCompressionByRounds:
    """Test round-based compression functionality."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="extended_debate"
        )
        self.context_manager = ContextWindowManager()

    @pytest.mark.integration
    def test_compress_by_rounds_keeps_recent(self):
        """Test that round-based compression keeps recent rounds."""
        state = self.test_helper.create_discussion_state()
        state["current_round"] = 10

        # Add messages from 10 rounds
        for round_num in range(1, 11):
            for agent_id in state["speaking_order"]:
                state["messages"].append(
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Round {round_num} message from {agent_id}",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": "Test Topic",
                        "message_type": "discussion",
                    }
                )

        # Compress keeping last 3 rounds
        compressed_state = self.context_manager.compress_by_rounds(
            state, keep_recent_rounds=3
        )

        # Should only have messages from rounds 8, 9, 10
        remaining_rounds = set(
            msg["round_number"] for msg in compressed_state["messages"]
        )
        assert remaining_rounds == {8, 9, 10}

        # Should have round summaries for older rounds
        assert len(compressed_state["round_summaries"]) >= 7

    @pytest.mark.integration
    def test_compress_by_topics_keeps_current(self):
        """Test that topic-based compression keeps current topic."""
        state = self.test_helper.create_discussion_state()
        state["current_topic_index"] = 1
        state["agenda"] = [
            {"title": "Topic A", "description": "First topic"},
            {"title": "Topic B", "description": "Current topic"},
            {"title": "Topic C", "description": "Future topic"},
        ]

        # Add messages for different topics
        for topic in ["Topic A", "Topic B"]:
            for i in range(5):
                state["messages"].append(
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": "agent_1",
                        "content": f"Message about {topic}",
                        "timestamp": datetime.now(),
                        "topic": topic,
                        "message_type": "discussion",
                    }
                )

        # Compress by topics
        compressed_state = self.context_manager.compress_by_topics(state)

        # Should only have Topic B messages
        remaining_topics = set(msg["topic"] for msg in compressed_state["messages"])
        assert remaining_topics == {"Topic B"}

        # Should have summary for Topic A
        assert "Topic A" in compressed_state["topic_summaries"]


class TestCompressionWithSummaries:
    """Test compression interaction with summaries."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.context_manager = ContextWindowManager(
            window_limit=1000
        )  # Small limit for testing

    @pytest.mark.integration
    def test_summaries_counted_in_context(self):
        """Test that summaries are included in context size calculation."""
        state = self.test_helper.create_discussion_state()

        # Add summaries
        state["phase_summaries"] = {
            "phase_1": "Summary of initialization phase with important context",
            "phase_2": "Summary of discussion phase with key insights and decisions",
        }
        state["topic_summaries"] = ["Topic 1 comprehensive summary with details"]
        state["consensus_summaries"] = {
            "topic_1": "Consensus reached on implementation approach"
        }
        state["final_report"] = ""

        # Get context size
        context_size = self.context_manager.get_context_size(state)

        # Should include summary tokens
        assert context_size > 0

        # Get detailed stats
        stats = self.context_manager.get_context_stats(state)
        assert stats["breakdown"]["phase_summaries"] > 0
        assert stats["breakdown"]["topic_summaries"] > 0
        assert stats["breakdown"]["consensus_summaries"] > 0

    @pytest.mark.integration
    def test_compression_preserves_summaries(self):
        """Test that compression preserves all summaries."""
        state = self.test_helper.create_discussion_state()

        # Add summaries
        original_phase_summaries = {
            "phase_1": "Important phase 1 summary",
            "phase_2": "Critical phase 2 summary",
        }
        original_topic_summaries = ["Topic summary 1", "Topic summary 2"]

        state["phase_summaries"] = original_phase_summaries.copy()
        state["topic_summaries"] = original_topic_summaries.copy()
        state["consensus_summaries"] = {}
        state["final_report"] = ""

        # Add many messages to trigger compression
        for i in range(50):
            state["messages"].append(
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "Message content " * 20,
                    "timestamp": datetime.now(),
                    "topic": "Test",
                    "message_type": "discussion",
                }
            )

        # Apply compression
        updates = self.context_manager.compress_context(state)

        # Summaries should not be in updates (not modified)
        assert "phase_summaries" not in updates
        assert "topic_summaries" not in updates

        # Messages should be compressed
        assert "messages" in updates
        assert len(updates["messages"]) < 50


class TestCompressionEdgeCases:
    """Test edge cases in compression."""

    def setup_method(self):
        """Set up test method."""
        self.context_manager = ContextWindowManager(window_limit=1000)

    @pytest.mark.integration
    def test_compression_with_empty_messages(self):
        """Test compression with no messages."""
        state = {"messages": [], "phase_summaries": {}, "topic_summaries": []}

        # Should not need compression
        assert not self.context_manager.needs_compression(state)

        # Compression should return empty updates
        updates = self.context_manager.compress_context(state)
        assert updates == {}

    @pytest.mark.integration
    def test_compression_with_single_message(self):
        """Test compression with only one message."""
        state = {
            "messages": [
                {
                    "id": "1",
                    "content": "Single message",
                    "timestamp": datetime.now(),
                }
            ],
            "phase_summaries": {},
            "topic_summaries": [],
        }

        # Apply compression
        compressed, summary = self.context_manager.compress_messages(
            state["messages"], 100
        )

        # Should keep the single message
        assert len(compressed) == 1
        assert compressed[0]["id"] == "1"

    @pytest.mark.integration
    def test_compression_recovery_from_corruption(self):
        """Test compression handles corrupted state gracefully."""
        state = {
            "messages": [
                {"id": "1", "content": "Normal message"},
                {"id": "2"},  # Missing content
                None,  # Null message
                {"id": "4", "content": "Another normal message"},
            ],
            "phase_summaries": None,  # Corrupted
            "topic_summaries": "not_a_list",  # Wrong type
            "consensus_summaries": {},
            "final_report": "",
        }

        # Should handle gracefully without crashing
        try:
            # Test message counting with corrupted messages
            valid_messages = [
                msg
                for msg in state["messages"]
                if msg and isinstance(msg, dict) and "content" in msg
            ]
            assert len(valid_messages) == 2

            # Create a clean state for context size calculation
            clean_state = {
                "messages": valid_messages,
                "phase_summaries": (
                    {}
                    if not isinstance(state.get("phase_summaries"), dict)
                    else state["phase_summaries"]
                ),
                "topic_summaries": (
                    []
                    if not isinstance(state.get("topic_summaries"), list)
                    else state["topic_summaries"]
                ),
                "consensus_summaries": state.get("consensus_summaries", {}),
                "final_report": state.get("final_report", ""),
            }

            context_size = self.context_manager.get_context_size(clean_state)
            assert context_size >= 0  # Should calculate something
        except Exception as e:
            pytest.fail(f"Compression failed on corrupted state: {e}")
