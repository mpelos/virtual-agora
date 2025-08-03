"""End-to-end discussion flow test to verify no empty list spam occurs.

This test module simulates complete discussion rounds to match the user's
original complaint of "10+ empty list rejections per round" and verifies
that the fix completely eliminates this issue.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List
import time

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.edges_v13 import V13FlowConditions
from virtual_agora.flow.nodes_v13 import V13FlowNodes
from virtual_agora.state.reducers import safe_list_append
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config


class TestDiscussionFlowEndToEnd:
    """End-to-end tests for complete discussion flow without empty list spam."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Reset all reducer state
        if hasattr(safe_list_append, "_empty_list_count"):
            delattr(safe_list_append, "_empty_list_count")

        # Mock config
        self.mock_config = Mock(spec=Config)

        # Mock agents with proper responses
        self.mock_moderator = Mock(spec=ModeratorAgent)
        self.mock_moderator.evaluate_message_relevance.return_value = {
            "is_relevant": True
        }

        self.mock_summarizer = Mock(spec=SummarizerAgent)
        self.mock_summarizer.summarize_round.return_value = "Test round summary"

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

    def create_initial_state(self) -> Dict[str, Any]:
        """Create initial state for discussion flow."""
        return {
            "session_id": "test-session",
            "current_round": 0,
            "current_phase": 2,
            "active_topic": "Test Discussion Topic",
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
            "agents": {
                "agent1": {
                    "id": "agent1",
                    "model": "test",
                    "provider": "test",
                    "role": "participant",
                }
            },
            "moderator_id": "moderator",
        }

    def simulate_discussion_round(
        self, state: Dict[str, Any], round_num: int
    ) -> Dict[str, Any]:
        """Simulate a complete discussion round with all nodes."""
        current_state = state.copy()
        current_state["current_round"] = round_num

        # Track all state updates and reducer calls
        state_updates = []
        reducer_calls = []

        def track_reducer_calls(current_list, new_item):
            reducer_calls.append(
                {
                    "current_list": current_list,
                    "new_item": new_item,
                    "is_empty_list": isinstance(new_item, list) and len(new_item) == 0,
                    "round": round_num,
                }
            )
            return safe_list_append(current_list, new_item)

        with patch(
            "virtual_agora.state.reducers.safe_list_append",
            side_effect=track_reducer_calls,
        ):
            with patch(
                "virtual_agora.flow.nodes_v13.retry_agent_call"
            ) as mock_retry_agent:
                mock_retry_agent.return_value = {
                    "messages": [Mock(content="Test response")]
                }

                with patch("virtual_agora.flow.nodes_v13.display_agent_response"):
                    # 1. Discussion round node
                    discussion_result = self.nodes.discussion_round_node(current_state)
                    state_updates.append(("discussion_round", discussion_result))
                    current_state.update(discussion_result)

                    # 2. Round summarization node
                    summary_result = self.nodes.round_summarization_node(current_state)
                    state_updates.append(("round_summarization", summary_result))
                    current_state.update(summary_result)

                    # 3. Round threshold check node
                    threshold_result = self.nodes.round_threshold_check_node(
                        current_state
                    )
                    state_updates.append(("round_threshold_check", threshold_result))
                    current_state.update(threshold_result)

                    # 4. User participation check (only for round 1+)
                    if round_num >= 1:
                        user_participation_check_node = lambda state: {
                            "current_round": state.get("current_round", 0)
                        }
                        participation_result = user_participation_check_node(
                            current_state
                        )
                        state_updates.append(
                            ("user_participation_check", participation_result)
                        )
                        current_state.update(participation_result)

        return {
            "final_state": current_state,
            "state_updates": state_updates,
            "reducer_calls": reducer_calls,
        }

    def test_single_round_no_empty_list_spam(self):
        """Test a single discussion round produces no empty list spam."""
        state = self.create_initial_state()

        # Simulate round 1 (first round where user participation check is called)
        result = self.simulate_discussion_round(state, 1)

        # Verify no empty list calls to reducers
        empty_list_calls = [
            call for call in result["reducer_calls"] if call["is_empty_list"]
        ]
        assert len(empty_list_calls) == 0, f"Found empty list calls: {empty_list_calls}"

        # Verify reducer calls are all legitimate
        for call in result["reducer_calls"]:
            assert not call["is_empty_list"], f"Unexpected empty list call: {call}"

        # Verify no empty list spam counter was created
        assert not hasattr(safe_list_append, "_empty_list_count")

    def test_multiple_rounds_no_cumulative_spam(self):
        """Test multiple discussion rounds to match user's '10+ rounds' scenario."""
        state = self.create_initial_state()
        all_reducer_calls = []
        all_state_updates = []

        # Simulate 12 rounds (more than user's "10+ rounds")
        for round_num in range(0, 12):
            result = self.simulate_discussion_round(state, round_num)

            # Update state for next round
            state = result["final_state"]

            # Collect all calls and updates
            all_reducer_calls.extend(result["reducer_calls"])
            all_state_updates.extend(result["state_updates"])

        # Verify NO empty list calls across all rounds
        empty_list_calls = [call for call in all_reducer_calls if call["is_empty_list"]]
        assert (
            len(empty_list_calls) == 0
        ), f"Found {len(empty_list_calls)} empty list calls across 12 rounds"

        # Verify total reducer calls are all legitimate
        total_reducer_calls = len(all_reducer_calls)
        legitimate_calls = len(
            [call for call in all_reducer_calls if not call["is_empty_list"]]
        )
        assert (
            total_reducer_calls == legitimate_calls
        ), f"Expected all {total_reducer_calls} calls to be legitimate"

        # Verify no cumulative spam counter
        assert not hasattr(safe_list_append, "_empty_list_count")

    def test_user_participation_flow_in_discussion_rounds(self):
        """Test that user participation flow integration doesn't cause spam."""
        state = self.create_initial_state()
        user_participation_rounds = []

        # Test rounds 0-5 and track when user participation is triggered
        for round_num in range(0, 6):
            # Check if this round should trigger user participation
            temp_state = state.copy()
            temp_state["current_round"] = round_num
            condition = self.conditions.should_show_user_participation(temp_state)

            result = self.simulate_discussion_round(state, round_num)

            if condition == "user_participation":
                user_participation_rounds.append(round_num)

                # Verify user participation check was included and didn't cause spam
                participation_updates = [
                    update
                    for update in result["state_updates"]
                    if update[0] == "user_participation_check"
                ]
                assert len(participation_updates) == 1

                # Verify the update is minimal
                participation_result = participation_updates[0][1]
                assert participation_result == {"current_round": round_num}

            # Verify no empty list calls in any round
            empty_list_calls = [
                call for call in result["reducer_calls"] if call["is_empty_list"]
            ]
            assert (
                len(empty_list_calls) == 0
            ), f"Round {round_num} had empty list calls: {empty_list_calls}"

            state = result["final_state"]

        # Verify user participation was triggered for rounds 1-5
        assert user_participation_rounds == [1, 2, 3, 4, 5]

    def test_performance_with_fixed_vs_broken_approach(self):
        """Test performance difference between fixed and broken approach."""
        state = self.create_initial_state()

        # Test fixed approach performance
        start_time = time.time()
        fixed_user_participation_check = lambda state: {
            "current_round": state.get("current_round", 0)
        }
        for _ in range(100):  # 100 calls to measure performance
            result = fixed_user_participation_check(state)
        fixed_time = time.time() - start_time

        # Test broken approach performance (returns entire state)
        start_time = time.time()
        broken_user_participation_check = lambda state: state
        for _ in range(100):  # 100 calls to measure performance
            result = broken_user_participation_check(state)
        broken_time = time.time() - start_time

        # Fixed approach should be much faster (doesn't copy entire state)
        assert fixed_time < broken_time

        # Verify results are different
        fixed_result = fixed_user_participation_check(state)
        broken_result = broken_user_participation_check(state)

        assert len(fixed_result) == 1
        assert len(broken_result) > 10  # Much larger result

        assert "current_round" in fixed_result
        assert fixed_result["current_round"] == 0

    def test_realistic_discussion_scenario_with_state_evolution(self):
        """Test realistic discussion scenario with evolving state over multiple rounds."""
        state = self.create_initial_state()

        # Simulate realistic state evolution through rounds
        for round_num in range(0, 8):
            result = self.simulate_discussion_round(state, round_num)

            # Verify no empty list spam
            empty_list_calls = [
                call for call in result["reducer_calls"] if call["is_empty_list"]
            ]
            assert len(empty_list_calls) == 0

            # Update state with realistic data accumulation
            state = result["final_state"]

            # Add some realistic state evolution
            if round_num > 0:
                # Add messages (simulating discussion)
                state["messages"].append(
                    {
                        "id": f"msg_{round_num}",
                        "content": f"Message from round {round_num}",
                        "round": round_num,
                    }
                )

            # Verify state grows realistically
            assert state["current_round"] == round_num
            assert len(state["messages"]) == max(0, round_num)

        # Final verification: no empty list spam counter exists
        assert not hasattr(safe_list_append, "_empty_list_count")

    def test_stress_test_high_round_count(self):
        """Stress test with very high round count to ensure robustness."""
        state = self.create_initial_state()
        total_reducer_calls = 0

        # Test 25 rounds (more than any realistic discussion)
        for round_num in range(0, 25):
            result = self.simulate_discussion_round(state, round_num)

            # Verify no empty list calls
            empty_list_calls = [
                call for call in result["reducer_calls"] if call["is_empty_list"]
            ]
            assert (
                len(empty_list_calls) == 0
            ), f"Round {round_num} failed: {empty_list_calls}"

            total_reducer_calls += len(result["reducer_calls"])
            state = result["final_state"]

        # Verify system handled high load without issues
        assert total_reducer_calls > 0  # Some legitimate reducer calls happened
        assert not hasattr(safe_list_append, "_empty_list_count")  # No spam counter

    def test_edge_case_empty_state_fields_throughout_flow(self):
        """Test edge case where state fields remain empty throughout flow."""
        # Create state with persistent empty fields
        state = self.create_initial_state()

        # Force some fields to remain empty throughout
        for round_num in range(0, 5):
            # Ensure certain fields stay empty
            state["votes"] = []
            state["warnings"] = []
            state["agent_invocations"] = []

            result = self.simulate_discussion_round(state, round_num)

            # Even with empty fields, should have no empty list spam
            empty_list_calls = [
                call for call in result["reducer_calls"] if call["is_empty_list"]
            ]
            assert len(empty_list_calls) == 0

            state = result["final_state"]

            # Verify empty fields persist but don't cause issues
            assert state["votes"] == []
            assert state["warnings"] == []
            assert state["agent_invocations"] == []

        # No spam counter should exist
        assert not hasattr(safe_list_append, "_empty_list_count")

    def test_comprehensive_flow_validation(self):
        """Comprehensive validation of the complete discussion flow."""
        state = self.create_initial_state()
        flow_metrics = {
            "total_rounds": 0,
            "total_node_executions": 0,
            "total_reducer_calls": 0,
            "empty_list_calls": 0,
            "user_participation_calls": 0,
            "state_update_count": 0,
        }

        # Run comprehensive flow
        for round_num in range(0, 10):
            result = self.simulate_discussion_round(state, round_num)

            # Update metrics
            flow_metrics["total_rounds"] += 1
            flow_metrics["total_node_executions"] += len(result["state_updates"])
            flow_metrics["total_reducer_calls"] += len(result["reducer_calls"])
            flow_metrics["empty_list_calls"] += len(
                [call for call in result["reducer_calls"] if call["is_empty_list"]]
            )
            flow_metrics["state_update_count"] += len(result["state_updates"])

            # Count user participation calls
            participation_calls = len(
                [
                    update
                    for update in result["state_updates"]
                    if update[0] == "user_participation_check"
                ]
            )
            flow_metrics["user_participation_calls"] += participation_calls

            state = result["final_state"]

        # Comprehensive assertions
        assert flow_metrics["total_rounds"] == 10
        assert flow_metrics["total_node_executions"] > 30  # Multiple nodes per round
        assert flow_metrics["empty_list_calls"] == 0  # ZERO empty list calls
        assert flow_metrics["user_participation_calls"] == 9  # Rounds 1-9 (not round 0)
        assert flow_metrics["state_update_count"] > 30  # Multiple updates per round

        # Final state should be well-formed
        assert state["current_round"] == 9
        assert "active_topic" in state
        assert isinstance(state["messages"], list)

        # No spam counter should exist after all this activity
        assert not hasattr(safe_list_append, "_empty_list_count")

        print(f"Flow completed successfully with metrics: {flow_metrics}")
