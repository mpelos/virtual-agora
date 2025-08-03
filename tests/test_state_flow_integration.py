"""Integration tests for state flow from nodes to LangGraph to reducers.

This test module verifies that state updates flow correctly through the
entire system, including node returns, LangGraph processing, and reducer handling.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from virtual_agora.state.schema import (
    VirtualAgoraState,
    PhaseTransition,
    RoundInfo,
    RoundSummary,
)
from virtual_agora.state.reducers import safe_list_append
from virtual_agora.flow.nodes_v13 import V13FlowNodes
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config


class TestStateFlowIntegration:
    """Test state flow from nodes through LangGraph to reducers."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Reset reducer state
        if hasattr(safe_list_append, "_empty_list_count"):
            delattr(safe_list_append, "_empty_list_count")

        # Mock config
        self.mock_config = Mock(spec=Config)

        # Mock agents
        self.mock_moderator = Mock(spec=ModeratorAgent)
        self.mock_summarizer = Mock(spec=SummarizerAgent)
        self.mock_discussion_agent = Mock(spec=DiscussionAgent)
        # Add required attributes for discussion agent
        self.mock_discussion_agent.agent_id = "agent1"
        self.mock_discussion_agent.name = "Test Agent"

        # Mock specialized agents dict
        self.specialized_agents = {
            "moderator": self.mock_moderator,
            "summarizer": self.mock_summarizer,
        }

        # Mock discussing agents list
        self.discussing_agents = [self.mock_discussion_agent]

        # Mock state manager
        self.mock_state_manager = Mock(spec=StateManager)

        # Create nodes instance
        self.nodes = V13FlowNodes(
            specialized_agents=self.specialized_agents,
            discussing_agents=self.discussing_agents,
            state_manager=self.mock_state_manager,
            checkpoint_interval=3,
        )

    def create_mock_state(self, **kwargs) -> Dict[str, Any]:
        """Create a mock state with default values."""
        default_state = {
            "session_id": "test-session",
            "main_topic": "Test Topic",
            "current_phase": 0,
            "current_round": 1,
            "active_topic": "Test Active Topic",
            "messages": [],
            "phase_history": [],
            "round_history": [],
            "turn_order_history": [],
            "round_summaries": [],
            "speaking_order": ["agent1"],
            "topic_queue": ["topic1", "topic2"],
            "completed_topics": [],
        }
        default_state.update(kwargs)
        return default_state

    def test_phase_transition_state_update(self):
        """Test that phase transition state updates work correctly."""
        state = self.create_mock_state()
        state["main_topic"] = "Test Topic"
        state["user_defines_topics"] = True

        # Mock the UI function to return topics
        with patch(
            "virtual_agora.ui.human_in_the_loop.get_user_defined_topics",
            return_value=["Topic 1", "Topic 2"],
        ):
            result = self.nodes.get_theme_node(state)

        # Verify the node returns correct phase_history structure
        assert "phase_history" in result
        phase_transition = result["phase_history"]
        assert isinstance(phase_transition, dict)
        assert "from_phase" in phase_transition
        assert "to_phase" in phase_transition
        assert "timestamp" in phase_transition
        assert "reason" in phase_transition
        assert "triggered_by" in phase_transition

        # Test that the reducer can handle this correctly
        existing_history = []
        processed_history = safe_list_append(existing_history, phase_transition)
        assert len(processed_history) == 1
        assert processed_history[0] == phase_transition

    def test_phase_transition_wrapped_handling(self):
        """Test that wrapped phase transitions are handled correctly."""
        state = self.create_mock_state()

        # Create a phase transition as it should be
        phase_transition = {
            "from_phase": 0,
            "to_phase": 1,
            "timestamp": datetime.now(),
            "reason": "Test transition",
            "triggered_by": "test",
        }

        # Simulate LangGraph wrapping it in a list (the bug we're fixing)
        wrapped_transition = [phase_transition]

        # Test that reducer unwraps it correctly
        existing_history = []
        processed_history = safe_list_append(existing_history, wrapped_transition)
        assert len(processed_history) == 1
        assert processed_history[0] == phase_transition  # Should be unwrapped

    def test_round_info_state_update(self):
        """Test that round info state updates work correctly."""
        state = self.create_mock_state()
        state["active_topic"] = "test topic"

        # Mock agent responses for discussion round - only one agent in speaking order
        mock_response = {"messages": [Mock(content="Response 1")]}

        with patch(
            "virtual_agora.flow.nodes_v13.retry_agent_call", return_value=mock_response
        ):
            with patch("virtual_agora.flow.nodes_v13.display_agent_response"):
                with patch.object(
                    self.mock_moderator,
                    "evaluate_message_relevance",
                    return_value={"is_relevant": True},
                ):
                    result = self.nodes.discussion_round_node(state)

        # Verify round_history structure
        assert "round_history" in result
        round_info = result["round_history"]
        assert isinstance(round_info, dict)
        assert "round_id" in round_info
        assert "round_number" in round_info
        assert "topic" in round_info
        assert "participants" in round_info

        # Test reducer handling
        existing_rounds = []
        processed_rounds = safe_list_append(existing_rounds, round_info)
        assert len(processed_rounds) == 1
        assert processed_rounds[0] == round_info

    def test_turn_order_history_state_update(self):
        """Test that turn order history state updates work correctly."""
        state = self.create_mock_state()
        state["active_topic"] = "test topic"

        # Mock agent responses for discussion round - only one agent in speaking order
        mock_response = {"messages": [Mock(content="Response 1")]}

        with patch(
            "virtual_agora.flow.nodes_v13.retry_agent_call", return_value=mock_response
        ):
            with patch("virtual_agora.flow.nodes_v13.display_agent_response"):
                with patch.object(
                    self.mock_moderator,
                    "evaluate_message_relevance",
                    return_value={"is_relevant": True},
                ):
                    result = self.nodes.discussion_round_node(state)

        # Verify turn_order_history structure
        assert "turn_order_history" in result
        turn_order = result["turn_order_history"]
        assert isinstance(turn_order, list)
        assert all(isinstance(agent_id, str) for agent_id in turn_order)

        # Test reducer handling (should accept list of agent IDs)
        existing_turn_orders = []
        processed_turn_orders = safe_list_append(existing_turn_orders, turn_order)
        assert len(processed_turn_orders) == 1
        assert processed_turn_orders[0] == turn_order

    def test_turn_order_nested_list_handling(self):
        """Test that nested turn order lists are unwrapped correctly."""
        turn_order = ["agent1", "agent2", "agent3"]

        # Simulate LangGraph double-wrapping (the bug we're fixing)
        nested_turn_order = [turn_order]

        # Test that reducer unwraps it correctly
        existing_turn_orders = []
        processed_turn_orders = safe_list_append(
            existing_turn_orders, nested_turn_order
        )
        assert len(processed_turn_orders) == 1
        assert processed_turn_orders[0] == turn_order  # Should be unwrapped

    def test_round_summary_state_update(self):
        """Test that round summary state updates work correctly."""
        # Test direct reducer behavior with round summary structure
        round_summary = {
            "round_number": 1,
            "topic": "test topic",
            "summary_text": "Test summary",
            "created_by": "summarizer",
            "timestamp": datetime.now(),
            "token_count": 10,
            "compression_ratio": 0.1,
        }

        # Test reducer handling
        existing_summaries = []
        processed_summaries = safe_list_append(existing_summaries, round_summary)
        assert len(processed_summaries) == 1
        assert processed_summaries[0] == round_summary

    def test_round_summary_wrapped_handling(self):
        """Test that wrapped round summaries are handled correctly."""
        round_summary = {
            "round_number": 1,
            "topic": "test topic",
            "summary_text": "Test summary",
            "created_by": "summarizer",
            "timestamp": datetime.now(),
            "token_count": 10,
            "compression_ratio": 0.1,
        }

        # Simulate LangGraph wrapping it in a list (the bug we're fixing)
        wrapped_summary = [round_summary]

        # Test that reducer unwraps it correctly
        existing_summaries = []
        processed_summaries = safe_list_append(existing_summaries, wrapped_summary)
        assert len(processed_summaries) == 1
        assert processed_summaries[0] == round_summary  # Should be unwrapped

    def test_empty_list_handling_in_flow(self):
        """Test that empty lists don't break the state flow."""
        # Test various empty list scenarios that might occur in real flow
        existing_list = ["existing_item"]

        # Test multiple empty list rejections
        for i in range(5):
            result = safe_list_append(existing_list, [])
            assert result == existing_list  # Should remain unchanged

        # Verify counter is working
        assert getattr(safe_list_append, "_empty_list_count", 0) >= 5

    def test_mixed_pattern_flow_stability(self):
        """Test that mixed correct/incorrect patterns maintain flow stability."""
        # Simulate a complete flow with mixed patterns

        # Phase history with mixed patterns
        phase_history = []

        # Correct pattern
        transition1 = {
            "from_phase": 0,
            "to_phase": 1,
            "timestamp": datetime.now(),
            "reason": "Test 1",
            "triggered_by": "system",
        }
        phase_history = safe_list_append(phase_history, transition1)

        # Wrapped pattern (bug simulation)
        transition2 = {
            "from_phase": 1,
            "to_phase": 2,
            "timestamp": datetime.now(),
            "reason": "Test 2",
            "triggered_by": "user",
        }
        phase_history = safe_list_append(phase_history, [transition2])

        # Both should be in the history correctly
        assert len(phase_history) == 2
        assert phase_history[0] == transition1
        assert phase_history[1] == transition2

    def test_concurrent_state_updates_simulation(self):
        """Test simulation of concurrent state updates that might cause LangGraph issues."""
        # Simulate rapid state updates to multiple fields
        phase_history = []
        round_history = []
        turn_order_history = []

        # Rapid updates with mixed patterns
        for i in range(5):
            # Phase transition
            transition = {
                "from_phase": i,
                "to_phase": i + 1,
                "timestamp": datetime.now(),
                "reason": f"Step {i}",
                "triggered_by": "test",
            }
            if i % 2 == 0:
                phase_history = safe_list_append(phase_history, [transition])  # Wrapped
            else:
                phase_history = safe_list_append(phase_history, transition)  # Correct

            # Round info
            round_info = {
                "round_id": f"round-{i}",
                "round_number": i + 1,
                "topic": f"topic-{i}",
                "participants": [f"agent{j}" for j in range(3)],
                "message_count": 3,
            }
            if i % 3 == 0:
                round_history = safe_list_append(round_history, [round_info])  # Wrapped
            else:
                round_history = safe_list_append(round_history, round_info)  # Correct

            # Turn order
            turn_order = [f"agent{j}" for j in range(i + 1, i + 4)]
            if i % 2 == 1:
                turn_order_history = safe_list_append(
                    turn_order_history, [turn_order]
                )  # Nested
            else:
                turn_order_history = safe_list_append(
                    turn_order_history, turn_order
                )  # Correct

        # Verify all updates were processed correctly
        assert len(phase_history) == 5
        assert len(round_history) == 5
        assert len(turn_order_history) == 5

        # Verify data integrity
        assert all(isinstance(t, dict) and "from_phase" in t for t in phase_history)
        assert all(isinstance(r, dict) and "round_id" in r for r in round_history)
        assert all(
            isinstance(to, list) and all(isinstance(agent, str) for agent in to)
            for to in turn_order_history
        )

    def test_error_recovery_in_flow(self):
        """Test that the flow can recover from various error conditions."""
        existing_data = ["valid_item"]

        # Test recovery from various malformed inputs
        malformed_inputs = [
            [],  # Empty list
            [[]],  # Nested empty list
            [None],  # List with None
            [{"incomplete": "data"}],  # Incomplete dict
            [[{"nested": "dict"}]],  # Deeply nested
        ]

        for malformed_input in malformed_inputs:
            result = safe_list_append(existing_data, malformed_input)
            # Should maintain stability
            assert isinstance(result, list)
            assert len(result) >= len(existing_data)
            # Original data should still be there
            assert "valid_item" in result

    def test_vote_batch_integration(self):
        """Test vote batch handling in realistic flow scenarios."""
        # Simulate agenda voting node returning vote batch
        votes = [
            {"voter_id": "agent1", "choice": "topic1", "timestamp": datetime.now()},
            {"voter_id": "agent2", "choice": "topic2", "timestamp": datetime.now()},
            {"voter_id": "agent3", "choice": "topic1", "timestamp": datetime.now()},
        ]

        # Test that vote batches are extended, not appended as single items
        existing_votes = []
        processed_votes = safe_list_append(existing_votes, votes)

        # Should have individual votes, not wrapped in another list
        assert len(processed_votes) == 3
        assert all("voter_id" in vote for vote in processed_votes)

    def test_state_schema_compliance(self):
        """Test that all processed data complies with state schema expectations."""
        # Test that processed data matches what the state schema expects

        # Phase history should be List[PhaseTransition]
        transition = {
            "from_phase": 0,
            "to_phase": 1,
            "timestamp": datetime.now(),
            "reason": "Test",
            "triggered_by": "system",
        }
        phase_history = safe_list_append([], transition)
        assert len(phase_history) == 1
        assert isinstance(phase_history[0], dict)

        # Round history should be List[RoundInfo]
        round_info = {
            "round_id": "test",
            "round_number": 1,
            "topic": "test",
            "participants": ["agent1"],
            "message_count": 1,
        }
        round_history = safe_list_append([], round_info)
        assert len(round_history) == 1
        assert isinstance(round_history[0], dict)

        # Turn order history should be List[List[str]]
        turn_order = ["agent1", "agent2"]
        turn_order_history = safe_list_append([], turn_order)
        assert len(turn_order_history) == 1
        assert isinstance(turn_order_history[0], list)
        assert all(isinstance(agent, str) for agent in turn_order_history[0])

    def test_performance_under_rapid_updates(self):
        """Test performance and stability under rapid state updates."""
        import time

        start_time = time.time()

        # Simulate rapid updates (like what might happen in a busy discussion)
        phase_history = []
        for i in range(100):
            transition = {
                "from_phase": i % 5,
                "to_phase": (i + 1) % 5,
                "timestamp": datetime.now(),
                "reason": f"Rapid transition {i}",
                "triggered_by": "performance_test",
            }

            # Mix patterns to test worst-case scenario
            if i % 4 == 0:
                phase_history = safe_list_append(phase_history, [transition])  # Wrapped
            else:
                phase_history = safe_list_append(phase_history, transition)  # Correct

        end_time = time.time()

        # Should complete quickly (under 1 second for 100 updates)
        assert end_time - start_time < 1.0

        # Should have all 100 transitions
        assert len(phase_history) == 100

        # All should be properly formatted
        assert all(isinstance(t, dict) and "from_phase" in t for t in phase_history)
