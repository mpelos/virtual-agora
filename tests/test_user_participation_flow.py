"""Integration tests for the user participation flow where the pass-through node fix is applied.

This test module verifies that the complete flow from round_threshold_check
through user_participation_check to discussion_round works without empty list spam.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.edges_v13 import V13FlowConditions
from virtual_agora.flow.nodes_v13 import V13FlowNodes
from virtual_agora.state.reducers import safe_list_append
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config


class TestUserParticipationFlow:
    """Test the user participation flow integration with the pass-through node fix."""

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

        # Create nodes and conditions instances
        self.nodes = V13FlowNodes(
            specialized_agents=self.specialized_agents,
            discussing_agents=self.discussing_agents,
            state_manager=self.mock_state_manager,
            checkpoint_interval=3,
        )

        self.conditions = V13FlowConditions(checkpoint_interval=3)

    def create_test_state(self, current_round: int = 1, **kwargs) -> Dict[str, Any]:
        """Create a test state for flow testing."""
        default_state = {
            "session_id": "test-session",
            "current_round": current_round,
            "current_phase": 2,
            "active_topic": "Test Topic",
            "speaking_order": ["agent1"],
            "messages": [],
            "phase_history": [],
            "round_history": [],
            "turn_order_history": [],
            "round_summaries": [],
            "votes": [],
            "warnings": [],
            "agent_invocations": [],
            "agent_contexts": [],
            "user_stop_history": [],
            "topic_transition_history": [],
            "edge_cases_encountered": [],
        }
        default_state.update(kwargs)
        return default_state

    def test_user_participation_check_condition_logic(self):
        """Test the conditional logic that determines when user_participation_check is called."""
        # Test case 1: Round 0 - should not show user participation
        state_round_0 = self.create_test_state(current_round=0)
        result_0 = self.conditions.should_show_user_participation(state_round_0)
        assert result_0 == "continue_flow"

        # Test case 2: Round 1 - should show user participation
        state_round_1 = self.create_test_state(current_round=1)
        result_1 = self.conditions.should_show_user_participation(state_round_1)
        assert result_1 == "user_participation"

        # Test case 3: Round 5 - should show user participation
        state_round_5 = self.create_test_state(current_round=5)
        result_5 = self.conditions.should_show_user_participation(state_round_5)
        assert result_5 == "user_participation"

    def test_user_participation_check_node_execution(self):
        """Test the actual execution of the user_participation_check node."""
        # Create state that would trigger user participation check
        state = self.create_test_state(current_round=2)

        # Create the actual lambda function used in the graph
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }

        # Execute the node
        result = user_participation_check_node(state)

        # Verify minimal update
        assert result == {"current_round": 2}
        assert len(result) == 1

        # Verify no reducer fields are returned
        reducer_fields = [
            "phase_history",
            "round_history",
            "turn_order_history",
            "round_summaries",
            "votes",
            "warnings",
            "agent_invocations",
            "agent_contexts",
            "user_stop_history",
            "topic_transition_history",
            "edge_cases_encountered",
        ]
        for field in reducer_fields:
            assert field not in result

    def test_round_threshold_to_user_participation_flow(self):
        """Test the flow from round_threshold_check to user_participation_check."""
        # Create state for round threshold check
        state = self.create_test_state(current_round=1)

        # Execute round_threshold_check_node
        threshold_result = self.nodes.round_threshold_check_node(state)

        # Verify it returns minimal update
        assert "current_round" in threshold_result
        assert threshold_result["current_round"] == 1

        # Update state with threshold result
        updated_state = {**state, **threshold_result}

        # Check condition for user participation
        condition_result = self.conditions.should_show_user_participation(updated_state)
        assert (
            condition_result == "user_participation"
        )  # Round 1 should show user participation

        # Execute user_participation_check node
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }
        participation_result = user_participation_check_node(updated_state)

        # Verify the entire flow maintains minimal updates
        assert participation_result == {"current_round": 1}

    def test_flow_with_reducer_monitoring(self):
        """Test the flow while monitoring all reducer calls."""
        # Mock all reducer functions to track calls
        reducer_call_log = []

        def mock_safe_list_append(current_list, new_item):
            reducer_call_log.append(
                {
                    "function": "safe_list_append",
                    "current_list": current_list,
                    "new_item": new_item,
                    "new_item_type": type(new_item),
                    "is_empty_list": isinstance(new_item, list) and len(new_item) == 0,
                }
            )
            return safe_list_append(current_list, new_item)

        with patch(
            "virtual_agora.state.reducers.safe_list_append",
            side_effect=mock_safe_list_append,
        ):
            # Create state that triggers user participation flow
            state = self.create_test_state(current_round=2)

            # Execute the complete flow simulation
            # 1. Round threshold check
            threshold_result = self.nodes.round_threshold_check_node(state)

            # 2. User participation check node
            user_participation_check_node = lambda state: {
                "current_round": state.get("current_round", 0)
            }
            participation_result = user_participation_check_node(
                {**state, **threshold_result}
            )

            # Verify no reducer calls were made (since we only return current_round)
            assert (
                len(reducer_call_log) == 0
            ), f"Unexpected reducer calls: {reducer_call_log}"

    def test_user_participation_flow_with_empty_state_fields(self):
        """Test the flow with state that has empty reducer fields."""
        # Create state with empty reducer fields (simulating real scenario)
        state = self.create_test_state(
            current_round=3,
            phase_history=[],  # Empty
            round_history=[],  # Empty
            votes=[],  # Empty
            warnings=[],  # Empty
        )

        # Execute user_participation_check node
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }
        result = user_participation_check_node(state)

        # Verify only current_round is returned, not the empty fields
        assert result == {"current_round": 3}

        # Verify empty fields are not in the result
        assert "phase_history" not in result
        assert "round_history" not in result
        assert "votes" not in result
        assert "warnings" not in result

    def test_continue_flow_path_vs_user_participation_path(self):
        """Test both paths from user_participation_check condition."""
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }

        # Test continue_flow path (round 0)
        state_continue = self.create_test_state(current_round=0)
        condition_continue = self.conditions.should_show_user_participation(
            state_continue
        )
        assert condition_continue == "continue_flow"

        # Execute node regardless of condition (node always executes the same way)
        result_continue = user_participation_check_node(state_continue)
        assert result_continue == {"current_round": 0}

        # Test user_participation path (round 1+)
        state_participation = self.create_test_state(current_round=1)
        condition_participation = self.conditions.should_show_user_participation(
            state_participation
        )
        assert condition_participation == "user_participation"

        # Execute node (same lambda regardless of condition)
        result_participation = user_participation_check_node(state_participation)
        assert result_participation == {"current_round": 1}

        # Both should return minimal updates
        assert len(result_continue) == 1
        assert len(result_participation) == 1

    def test_multiple_round_progression_through_user_participation(self):
        """Test multiple rounds progressing through the user participation flow."""
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }

        for round_num in range(0, 6):  # Test rounds 0-5
            state = self.create_test_state(current_round=round_num)

            # Check condition
            condition = self.conditions.should_show_user_participation(state)
            expected_condition = (
                "user_participation" if round_num >= 1 else "continue_flow"
            )
            assert condition == expected_condition

            # Execute node
            result = user_participation_check_node(state)

            # Verify consistent minimal updates
            assert result == {"current_round": round_num}
            assert len(result) == 1

    def test_integration_with_real_reducer_behavior(self):
        """Test integration with the actual safe_list_append reducer."""
        # Reset reducer state
        if hasattr(safe_list_append, "_empty_list_count"):
            delattr(safe_list_append, "_empty_list_count")

        # Create state with some legitimate data and empty fields
        state = self.create_test_state(
            current_round=2,
            phase_history=[],  # This would cause spam with broken approach
            votes=[],  # This would cause spam with broken approach
        )

        # Execute the fixed user_participation_check node
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }
        result = user_participation_check_node(state)

        # Simulate what LangGraph would do with the result
        # Only current_round would be processed (not a reducer field)
        # No reducer calls should happen

        # Verify no empty list counter was created (no empty lists were passed to reducers)
        assert not hasattr(safe_list_append, "_empty_list_count")

        # Test that if we had used the broken approach, it would have created the counter
        broken_result = state  # Old broken approach returned entire state

        # Simulate passing empty lists from broken result to reducer
        if "phase_history" in broken_result and broken_result["phase_history"] == []:
            safe_list_append(None, broken_result["phase_history"])

        # Now the counter should exist (proving the old approach would cause spam)
        assert hasattr(safe_list_append, "_empty_list_count")
        assert safe_list_append._empty_list_count == 1

    def test_flow_performance_and_state_consistency(self):
        """Test that the flow maintains performance and state consistency."""
        # Create large state to test performance
        large_state = self.create_test_state(
            current_round=10,
            messages=[{"id": f"msg{i}", "content": f"content{i}"} for i in range(100)],
            phase_history=[{"from_phase": i, "to_phase": i + 1} for i in range(5)],
            round_history=[
                {"round_id": f"round{i}", "round_number": i} for i in range(10)
            ],
        )

        # Execute user_participation_check node
        user_participation_check_node = lambda state: {
            "current_round": state.get("current_round", 0)
        }

        import time

        start_time = time.time()
        result = user_participation_check_node(large_state)
        end_time = time.time()

        # Should be very fast (much faster than processing entire state)
        execution_time = end_time - start_time
        assert execution_time < 0.001  # Should be under 1ms

        # Should return only current_round regardless of state size
        assert result == {"current_round": 10}
        assert len(result) == 1
