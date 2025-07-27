"""Integration tests for context compression flow triggers in Virtual Agora v1.3.

This module tests when and how context compression is triggered in the flow:
- Compression triggers at specific checkpoints or conditions
- State transitions when compression occurs
- Data structure preservation during compression
- Integration with discussion flow and state management
"""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime
import uuid

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow

from ..helpers.fake_llm import create_fake_llm_pool, create_specialized_fake_llms
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    patch_ui_components,
    create_test_messages,
)


class TestCompressionFlowTriggers:
    """Test when compression triggers in the discussion flow."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="extended_debate"
        )

    @pytest.mark.integration
    def test_compression_triggers_at_round_10(self):
        """Test that compression check happens at round 10."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state at round 10
            state["current_round"] = 10
            state["messages"] = create_test_messages(30)  # 10 rounds × 3 agents
            state["round_summaries"] = [f"Round {i} summary" for i in range(1, 11)]

            # Mock compression node
            with patch.object(flow, "_check_compression_needed") as mock_check:
                mock_check.return_value = True

                # Should check for compression at round 10
                needs_compression = flow._should_check_compression(state)
                assert needs_compression
                assert state["current_round"] % 10 == 0

    @pytest.mark.integration
    def test_compression_triggers_at_round_20(self):
        """Test that compression check happens at round 20."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state at round 20
            state["current_round"] = 20
            state["messages"] = create_test_messages(60)  # 20 rounds × 3 agents

            # Should check for compression at round 20
            needs_compression = flow._should_check_compression(state)
            assert needs_compression

    @pytest.mark.integration
    def test_no_compression_at_round_7(self):
        """Test that compression doesn't trigger at non-checkpoint rounds."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state at round 7
            state["current_round"] = 7
            state["messages"] = create_test_messages(21)  # 7 rounds × 3 agents

            # Should NOT check for compression at round 7
            needs_compression = flow._should_check_compression(state)
            assert not needs_compression

    @pytest.mark.integration
    def test_compression_state_transition(self):
        """Test state transitions when compression occurs."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up for compression
            state["current_round"] = 10
            state["messages"] = create_test_messages(30)
            state["compression_events"] = []

            # Mock compression execution
            compressed_state = flow._apply_compression(state)

            # Verify compression event recorded
            assert len(compressed_state["compression_events"]) == 1
            compression_event = compressed_state["compression_events"][0]
            assert compression_event["round"] == 10
            assert compression_event["messages_before"] == 30
            assert compression_event["messages_after"] < 30
            assert "timestamp" in compression_event

    @pytest.mark.integration
    def test_compression_preserves_state_integrity(self):
        """Test that compression preserves essential state fields."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up complex state
            original_state = {
                "current_round": 15,
                "current_phase": 2,
                "current_topic_index": 0,
                "current_speaker_index": 1,
                "speaking_order": ["agent_1", "agent_2", "agent_3"],
                "agenda": [{"title": "Test Topic", "description": "Test"}],
                "messages": create_test_messages(45),
                "round_summaries": [f"Round {i}" for i in range(1, 16)],
                "voting_rounds": [],
                "metadata": {"session_id": "test123"},
            }

            # Apply compression
            compressed_state = flow._apply_compression(original_state)

            # Verify essential fields preserved
            assert compressed_state["current_round"] == original_state["current_round"]
            assert compressed_state["current_phase"] == original_state["current_phase"]
            assert (
                compressed_state["current_topic_index"]
                == original_state["current_topic_index"]
            )
            assert (
                compressed_state["current_speaker_index"]
                == original_state["current_speaker_index"]
            )
            assert (
                compressed_state["speaking_order"] == original_state["speaking_order"]
            )
            assert compressed_state["agenda"] == original_state["agenda"]
            assert len(compressed_state["round_summaries"]) == len(
                original_state["round_summaries"]
            )
            assert compressed_state["metadata"] == original_state["metadata"]

    @pytest.mark.integration
    def test_compression_during_active_vote(self):
        """Test that compression doesn't interfere with active voting."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state with active vote at round 10
            state["current_round"] = 10
            state["active_vote"] = "vote_123"
            state["voting_rounds"] = [
                {
                    "id": "vote_123",
                    "status": "in_progress",
                    "vote_type": "conclusion",
                }
            ]

            # Compression should be deferred during active vote
            can_compress = flow._can_compress_now(state)
            assert not can_compress

    def _should_check_compression(self, state: VirtualAgoraState) -> bool:
        """Simulate compression check logic."""
        # Check every 10 rounds
        return state["current_round"] % 10 == 0 and state["current_round"] > 0

    def _apply_compression(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate compression application."""
        import copy

        compressed = copy.deepcopy(state)

        # Record compression event
        if "compression_events" not in compressed:
            compressed["compression_events"] = []

        compression_event = {
            "round": state["current_round"],
            "messages_before": len(state["messages"]),
            "messages_after": len(state["messages"]) // 2,  # Simulate 50% reduction
            "timestamp": datetime.now(),
        }
        compressed["compression_events"].append(compression_event)

        # Simulate message reduction (keep recent messages)
        if len(compressed["messages"]) > 20:
            compressed["messages"] = compressed["messages"][-20:]

        return compressed

    def _can_compress_now(self, state: VirtualAgoraState) -> bool:
        """Check if compression can happen now."""
        # Don't compress during active votes
        if state.get("active_vote"):
            active_vote = next(
                (
                    v
                    for v in state.get("voting_rounds", [])
                    if v["id"] == state["active_vote"]
                ),
                None,
            )
            if active_vote and active_vote["status"] == "in_progress":
                return False
        return True


class TestCompressionFlowIntegration:
    """Test compression integration with overall discussion flow."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="extended_debate"
        )

    @pytest.mark.integration
    def test_compression_node_in_flow(self):
        """Test that compression node is properly integrated in the flow."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()

            # Verify compression node exists in flow
            graph = flow.compile()
            assert "check_compression" in graph.nodes

            # Verify edges to/from compression node
            edges = graph.edges
            compression_edges = [e for e in edges if "compression" in str(e)]
            assert len(compression_edges) > 0

    @pytest.mark.integration
    def test_compression_with_summarizer_agent(self):
        """Test compression works with Summarizer Agent summaries."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add summaries from Summarizer Agent
            state["round_summaries"] = [
                "Round 1: Initial discussion on technical architecture.",
                "Round 2: Security considerations were thoroughly analyzed.",
                "Round 3: Performance optimization strategies discussed.",
            ]
            state["current_round"] = 10
            state["messages"] = create_test_messages(30)

            # Apply compression
            compressed_state = flow._apply_compression(state)

            # All summaries must be preserved
            assert len(compressed_state["round_summaries"]) == 3
            assert compressed_state["round_summaries"] == state["round_summaries"]

    @pytest.mark.integration
    def test_compression_events_tracking(self):
        """Test that compression events are properly tracked."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Initialize compression tracking
            state["compression_events"] = []

            # Simulate multiple compressions
            for round_num in [10, 20, 30]:
                state["current_round"] = round_num
                state["messages"] = create_test_messages(round_num * 3)

                if flow._should_check_compression(state):
                    state = flow._apply_compression(state)

            # Verify all compression events recorded
            assert len(state["compression_events"]) == 3
            assert [e["round"] for e in state["compression_events"]] == [10, 20, 30]

    @pytest.mark.integration
    def test_compression_with_final_considerations(self):
        """Test compression doesn't affect final considerations collection."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state near end of discussion
            state["current_round"] = 20
            state["messages"] = create_test_messages(60)
            state["final_considerations"] = {
                "agent_1": "Important closing thoughts on the topic.",
                "agent_2": "Final recommendations for implementation.",
            }

            # Apply compression
            compressed_state = flow._apply_compression(state)

            # Final considerations must be preserved
            assert (
                compressed_state["final_considerations"]
                == state["final_considerations"]
            )
            assert len(compressed_state["final_considerations"]) == 2


@pytest.mark.integration
class TestCompressionEdgeCases:
    """Test edge cases in compression flow."""

    def test_compression_with_no_messages(self):
        """Test compression handling when no messages exist."""
        helper = IntegrationTestHelper(num_agents=2, scenario="quick_consensus")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Empty state at round 10
            state["current_round"] = 10
            state["messages"] = []
            state["round_summaries"] = []

            # Should handle gracefully
            compressed_state = flow._apply_compression(state)
            assert compressed_state["messages"] == []
            assert "compression_events" in compressed_state

    def test_compression_at_round_100(self):
        """Test compression still works at very high round numbers."""
        helper = IntegrationTestHelper(num_agents=3, scenario="extended_debate")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # High round number
            state["current_round"] = 100
            state["messages"] = create_test_messages(50)  # Don't need 300 messages

            # Should still trigger compression check
            needs_compression = flow._should_check_compression(state)
            assert needs_compression
            assert state["current_round"] % 10 == 0

    def test_compression_recovery_after_error(self):
        """Test compression recovery after error."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Corrupted compression events
            state["current_round"] = 20
            state["compression_events"] = None  # Corrupted

            # Should recover gracefully
            compressed_state = flow._apply_compression(state)
            assert isinstance(compressed_state["compression_events"], list)
            assert len(compressed_state["compression_events"]) == 1

    def _should_check_compression(self, state: VirtualAgoraState) -> bool:
        """Check if compression should be checked."""
        return state["current_round"] % 10 == 0 and state["current_round"] > 0

    def _apply_compression(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Apply compression with error recovery."""
        import copy

        compressed = copy.deepcopy(state)

        # Initialize compression events if missing or corrupted
        if not isinstance(compressed.get("compression_events"), list):
            compressed["compression_events"] = []

        # Add compression event
        compressed["compression_events"].append(
            {
                "round": state["current_round"],
                "timestamp": datetime.now(),
            }
        )

        return compressed
