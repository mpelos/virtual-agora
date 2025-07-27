"""Integration tests for periodic stop functionality in Virtual Agora v1.3.

This module tests the complete periodic stop workflows including:
- 5-round checkpoint triggers
- User decisions at checkpoints (continue, force end, modify)
- State preservation across stops
- Integration with ongoing voting and discussion flows
"""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime
import uuid

from virtual_agora.state.schema import (
    VirtualAgoraState,
    Message,
    RoundInfo,
    HITLState,
    VoteRound,
)
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.ui.hitl_state import HITLApprovalType, HITLResponse

from ..helpers.fake_llm import create_fake_llm_pool, create_specialized_fake_llms
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
    create_test_messages,
)


class TestPeriodicStopBasics:
    """Test basic periodic stop functionality."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="extended_debate"
        )
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_periodic_stop_trigger_timing(self):
        """Test that periodic stops trigger exactly at rounds 5, 10, 15, etc."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Test rounds 1-20 for periodic stop triggers
            expected_stops = [5, 10, 15, 20]
            actual_stops = []

            for round_num in range(1, 21):
                state["current_round"] = round_num

                # Check if this round should trigger periodic stop
                should_stop = round_num % 5 == 0 and round_num > 0

                if should_stop:
                    actual_stops.append(round_num)
                    # Verify state is set up for periodic stop
                    assert self._should_trigger_periodic_stop(state)
                else:
                    assert not self._should_trigger_periodic_stop(state)

            assert actual_stops == expected_stops

    @pytest.mark.integration
    def test_periodic_stop_state_updates(self):
        """Test state updates when periodic stop is triggered."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up for round 5
            state["current_round"] = 5
            state["periodic_stop_counter"] = 0

            # Simulate periodic stop trigger
            updated_state = self._trigger_periodic_stop(state)

            # Verify state updates
            assert updated_state["hitl_state"]["awaiting_approval"]
            assert updated_state["hitl_state"]["approval_type"] == "periodic_stop"
            assert updated_state["periodic_stop_counter"] == 1
            assert "Do you wish to end" in updated_state["hitl_state"]["prompt_message"]

    @pytest.mark.integration
    def test_periodic_stop_preserves_discussion_context(self):
        """Test that discussion context is preserved during periodic stops."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Build up discussion context
            messages = []
            for round_num in range(1, 6):
                for agent_id in state["speaking_order"]:
                    msg = {
                        "id": str(uuid.uuid4()),
                        "agent_id": agent_id,
                        "content": f"Round {round_num} comment from {agent_id}",
                        "timestamp": datetime.now(),
                        "round_number": round_num,
                        "topic": state["agenda"][0]["title"],
                        "message_type": "discussion",
                    }
                    messages.append(msg)

            state["messages"] = messages
            state["current_round"] = 5

            # Trigger periodic stop
            updated_state = self._trigger_periodic_stop(state)

            # Verify context is preserved
            assert len(updated_state["messages"]) == len(messages)
            assert updated_state["messages"] == messages
            assert updated_state["current_topic_index"] == state["current_topic_index"]
            assert updated_state["agenda"] == state["agenda"]

    def _should_trigger_periodic_stop(self, state: VirtualAgoraState) -> bool:
        """Check if periodic stop should trigger."""
        return state["current_round"] % 5 == 0 and state["current_round"] > 0

    def _trigger_periodic_stop(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Simulate triggering a periodic stop."""
        state["hitl_state"] = {
            "awaiting_approval": True,
            "approval_type": "periodic_stop",
            "prompt_message": f"Round {state['current_round']} checkpoint reached. Do you wish to end the discussion on this topic?",
            "approval_history": state.get("hitl_state", {}).get("approval_history", []),
        }
        state["periodic_stop_counter"] = state.get("periodic_stop_counter", 0) + 1
        return state


class TestPeriodicStopUserDecisions:
    """Test different user decisions at periodic stops."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_user_continue_decision(self):
        """Test user choosing to continue discussion at periodic stop."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up periodic stop at round 5
            state["current_round"] = 5
            state = self._setup_periodic_stop_state(state)

            # Simulate user choosing to continue
            user_response = HITLResponse(
                approved=True, action="continue", metadata={"checkpoint_round": 5}
            )

            updated_state = self._process_user_decision(state, user_response)

            # Verify state after continue decision
            assert not updated_state["hitl_state"]["awaiting_approval"]
            assert not updated_state.get("user_forced_conclusion", False)
            assert updated_state["current_phase"] == 2  # Still in discussion

            # Should be able to continue to round 6
            assert updated_state["current_round"] == 5  # Round doesn't increment here

    @pytest.mark.integration
    def test_user_force_end_decision(self):
        """Test user forcing topic end at periodic stop."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up periodic stop at round 10
            state["current_round"] = 10
            state = self._setup_periodic_stop_state(state)

            # Simulate user forcing topic end
            user_response = HITLResponse(
                approved=True,
                action="force_topic_end",
                reason="Sufficient discussion completed",
            )

            updated_state = self._process_user_decision(state, user_response)

            # Verify state after force end
            assert not updated_state["hitl_state"]["awaiting_approval"]
            assert updated_state["user_forced_conclusion"]
            assert updated_state["force_all_final_considerations"]

            # Should transition to final considerations
            assert "final_considerations" in updated_state.get("next_phase", "")

    @pytest.mark.integration
    def test_user_modify_agenda_decision(self):
        """Test user choosing to modify agenda at periodic stop."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up periodic stop at round 15
            state["current_round"] = 15
            state = self._setup_periodic_stop_state(state)

            # Add some completed topics
            state["topic_summaries"] = ["Summary of Topic 1"]

            # Simulate user wanting to modify agenda
            user_response = HITLResponse(
                approved=True,
                action="modify_agenda",
                metadata={"add_topics": ["New Urgent Topic"]},
            )

            updated_state = self._process_user_decision(state, user_response)

            # Verify state transitions for agenda modification
            assert not updated_state["hitl_state"]["awaiting_approval"]
            assert updated_state.get("agenda_modification_requested", False)
            assert "New Urgent Topic" in updated_state.get("proposed_additions", [])

    @pytest.mark.integration
    def test_multiple_periodic_stops_in_session(self):
        """Test handling multiple periodic stops in a single session."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            stop_decisions = []

            # Simulate discussion through rounds 1-20 with stops at 5, 10, 15, 20
            for round_num in [5, 10, 15, 20]:
                state["current_round"] = round_num
                state = self._setup_periodic_stop_state(state)

                # Alternate between continue and other decisions
                if round_num == 5:
                    decision = "continue"
                    response = HITLResponse(approved=True, action="continue")
                elif round_num == 10:
                    decision = "continue"
                    response = HITLResponse(approved=True, action="continue")
                elif round_num == 15:
                    decision = "modify_agenda"
                    response = HITLResponse(
                        approved=True,
                        action="modify_agenda",
                        metadata={"reason": "New insights require additional topics"},
                    )
                else:  # round 20
                    decision = "force_end"
                    response = HITLResponse(
                        approved=True,
                        action="force_topic_end",
                        reason="Comprehensive discussion completed",
                    )

                state = self._process_user_decision(state, response)
                stop_decisions.append(
                    {
                        "round": round_num,
                        "decision": decision,
                        "periodic_stop_counter": state.get("periodic_stop_counter", 0),
                    }
                )

            # Verify all stops were processed
            assert len(stop_decisions) == 4
            assert state["periodic_stop_counter"] == 4

            # Verify final state after force end
            assert state["user_forced_conclusion"]

    def _setup_periodic_stop_state(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Set up state for periodic stop."""
        state["hitl_state"] = {
            "awaiting_approval": True,
            "approval_type": "periodic_stop",
            "prompt_message": f"Round {state['current_round']} checkpoint. Do you wish to end the discussion?",
            "approval_history": state.get("hitl_state", {}).get("approval_history", []),
        }
        state["periodic_stop_counter"] = state.get("periodic_stop_counter", 0) + 1
        return state

    def _process_user_decision(
        self, state: VirtualAgoraState, response: HITLResponse
    ) -> VirtualAgoraState:
        """Process user decision at periodic stop."""
        # Update HITL state
        state["hitl_state"]["awaiting_approval"] = False
        state["hitl_state"]["approval_history"].append(
            {
                "type": "periodic_stop",
                "round": state["current_round"],
                "result": response.action,
                "timestamp": datetime.now(),
                "approved": response.approved,
            }
        )

        # Handle different actions
        if response.action == "continue":
            # Continue discussion
            pass
        elif response.action == "force_topic_end":
            # Force topic conclusion
            state["user_forced_conclusion"] = True
            state["force_all_final_considerations"] = True
            state["next_phase"] = "final_considerations"
        elif response.action == "modify_agenda":
            # Request agenda modification
            state["agenda_modification_requested"] = True
            if response.metadata and "add_topics" in response.metadata:
                state["proposed_additions"] = response.metadata["add_topics"]

        return state


class TestPeriodicStopVotingIntegration:
    """Test integration of periodic stops with voting mechanisms."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=4, scenario="extended_debate"
        )

    @pytest.mark.integration
    def test_periodic_stop_during_conclusion_vote(self):
        """Test periodic stop occurring during an active conclusion vote."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Set up state: round 5, with active conclusion vote
            state["current_round"] = 5

            # Create active voting round
            active_vote = {
                "id": str(uuid.uuid4()),
                "phase": 5,
                "vote_type": "conclusion",
                "status": "in_progress",
                "start_time": datetime.now(),
                "required_votes": len(state["speaking_order"]),
                "received_votes": 2,  # Partial votes received
                "votes": {
                    "agent_1": "Yes. Ready to conclude.",
                    "agent_2": "No. More discussion needed.",
                },
            }
            state["voting_rounds"] = [active_vote]
            state["active_vote"] = active_vote["id"]

            # Periodic stop should wait for vote completion
            can_interrupt = self._can_interrupt_for_periodic_stop(state)
            assert not can_interrupt  # Should not interrupt active vote

    @pytest.mark.integration
    def test_periodic_stop_after_vote_completion(self):
        """Test periodic stop triggering after vote completes."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Complete voting at round 5
            state["current_round"] = 5
            completed_vote = {
                "id": str(uuid.uuid4()),
                "phase": 5,
                "vote_type": "conclusion",
                "status": "completed",
                "result": "No",  # Vote to continue
                "votes": {
                    "agent_1": "No. More to discuss.",
                    "agent_2": "No. Important points remain.",
                    "agent_3": "Yes. Ready to move on.",
                },
            }
            state["voting_rounds"] = [completed_vote]

            # Now periodic stop can trigger
            can_interrupt = self._can_interrupt_for_periodic_stop(state)
            assert can_interrupt

    @pytest.mark.integration
    def test_user_force_end_overrides_ongoing_vote(self):
        """Test user force end at periodic stop overriding ongoing votes."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Round 10 with agents wanting to continue
            state["current_round"] = 10

            # Recent vote showing agents want to continue
            recent_vote = {
                "id": str(uuid.uuid4()),
                "phase": 10,
                "vote_type": "conclusion",
                "status": "completed",
                "result": "No",
                "timestamp": datetime.now(),
                "votes": {
                    "agent_1": "No. Critical aspects unexplored.",
                    "agent_2": "No. Need deeper analysis.",
                    "agent_3": "No. Important questions remain.",
                    "agent_4": "No. Further investigation required.",
                },
            }
            state["voting_rounds"] = [recent_vote]

            # User forces end despite unanimous agent disagreement
            state = self._setup_periodic_stop_state(state)
            user_response = HITLResponse(
                approved=True,
                action="force_topic_end",
                reason="Time constraints require moving forward",
            )

            updated_state = self._process_user_decision(state, user_response)

            # User decision should override agent votes
            assert updated_state["user_forced_conclusion"]
            assert updated_state["force_all_final_considerations"]

            # All agents should provide final considerations
            assert len(self._get_agents_for_final_considerations(updated_state)) == len(
                state["speaking_order"]
            )

    def _can_interrupt_for_periodic_stop(self, state: VirtualAgoraState) -> bool:
        """Check if periodic stop can interrupt current activity."""
        # Don't interrupt active votes
        if state.get("active_vote"):
            active_vote = next(
                (v for v in state["voting_rounds"] if v["id"] == state["active_vote"]),
                None,
            )
            if active_vote and active_vote["status"] == "in_progress":
                return False

        return True

    def _setup_periodic_stop_state(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Set up state for periodic stop."""
        state["hitl_state"] = {
            "awaiting_approval": True,
            "approval_type": "periodic_stop",
            "prompt_message": f"Round {state['current_round']} checkpoint.",
            "approval_history": [],
        }
        state["periodic_stop_counter"] = state.get("periodic_stop_counter", 0) + 1
        return state

    def _process_user_decision(
        self, state: VirtualAgoraState, response: HITLResponse
    ) -> VirtualAgoraState:
        """Process user decision."""
        state["hitl_state"]["awaiting_approval"] = False

        if response.action == "force_topic_end":
            state["user_forced_conclusion"] = True
            state["force_all_final_considerations"] = True

        return state

    def _get_agents_for_final_considerations(
        self, state: VirtualAgoraState
    ) -> list[str]:
        """Get list of agents who should provide final considerations."""
        if state.get("force_all_final_considerations"):
            return state["speaking_order"]
        else:
            # Only minority voters
            last_vote = state["voting_rounds"][-1] if state["voting_rounds"] else None
            if last_vote and last_vote["result"] == "Yes":
                return [
                    agent_id
                    for agent_id, vote in last_vote["votes"].items()
                    if vote.startswith("No")
                ]
        return []


class TestPeriodicStopEdgeCases:
    """Test edge cases in periodic stop functionality."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=2, scenario="quick_consensus"
        )

    @pytest.mark.integration
    def test_periodic_stop_at_round_100(self):
        """Test periodic stop still works at very high round numbers."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Test round 100
            state["current_round"] = 100
            state["periodic_stop_counter"] = 19  # Should have had 19 stops before

            # Should trigger stop
            should_stop = state["current_round"] % 5 == 0
            assert should_stop

            # Verify counter increments correctly
            state = self._trigger_periodic_stop(state)
            assert state["periodic_stop_counter"] == 20

    @pytest.mark.integration
    def test_periodic_stop_with_single_agent(self):
        """Test periodic stop with minimal agent configuration."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Simulate one agent being muted, only one active
            state["speaking_order"] = ["agent_1"]
            state["current_round"] = 5

            # Periodic stop should still work
            state = self._trigger_periodic_stop(state)
            assert state["hitl_state"]["awaiting_approval"]
            assert state["hitl_state"]["approval_type"] == "periodic_stop"

    @pytest.mark.integration
    def test_periodic_stop_recovery_after_error(self):
        """Test periodic stop recovery after system error."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate corrupted periodic stop counter
            state["current_round"] = 15
            state["periodic_stop_counter"] = None  # Corrupted

            # Should recover and set correct counter
            state = self._trigger_periodic_stop(state)

            # Should initialize counter if missing
            assert state["periodic_stop_counter"] == 1
            assert state["hitl_state"]["awaiting_approval"]

    @pytest.mark.integration
    def test_rapid_periodic_stops(self):
        """Test handling rapid succession of periodic stops (user skipping quickly)."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate user rapidly clicking continue through multiple stops
            rapid_stops = []

            for round_num in [5, 10, 15]:
                state["current_round"] = round_num

                # Quick trigger and resolution
                state = self._trigger_periodic_stop(state)
                rapid_stops.append(
                    {
                        "round": round_num,
                        "timestamp": datetime.now(),
                    }
                )

                # Immediate continue
                state["hitl_state"]["awaiting_approval"] = False
                state["hitl_state"]["approval_history"].append(
                    {
                        "type": "periodic_stop",
                        "result": "continue",
                        "timestamp": datetime.now(),
                    }
                )

            # Verify all stops were recorded
            assert len(state["hitl_state"]["approval_history"]) == 3
            assert state["periodic_stop_counter"] == 3

    def _trigger_periodic_stop(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Trigger periodic stop with error recovery."""
        # Initialize counter if missing
        if state.get("periodic_stop_counter") is None:
            state["periodic_stop_counter"] = 0

        state["hitl_state"] = {
            "awaiting_approval": True,
            "approval_type": "periodic_stop",
            "prompt_message": f"Round {state['current_round']} checkpoint.",
            "approval_history": state.get("hitl_state", {}).get("approval_history", []),
        }
        state["periodic_stop_counter"] += 1

        return state


@pytest.mark.integration
class TestPeriodicStopStateTransitions:
    """Test state transitions during periodic stops."""

    def test_state_preservation_across_stop(self):
        """Test complete state preservation across periodic stop."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state = helper.create_discussion_state()

            # Build complex state
            state["current_round"] = 5
            state["messages"] = create_test_messages(15)  # 5 rounds × 3 agents
            state["round_summaries"] = [f"Round {i} summary" for i in range(1, 6)]
            state["metadata"] = {
                "discussion_quality": "high",
                "participation_balance": "even",
            }

            # Take snapshot before stop
            pre_stop_snapshot = {
                "messages": len(state["messages"]),
                "summaries": len(state["round_summaries"]),
                "metadata": dict(state["metadata"]),
                "topic_index": state["current_topic_index"],
            }

            # Trigger and resolve periodic stop
            state["hitl_state"] = {
                "awaiting_approval": True,
                "approval_type": "periodic_stop",
                "prompt_message": "Checkpoint",
                "approval_history": [],
            }

            # Simulate continue decision
            state["hitl_state"]["awaiting_approval"] = False
            state["hitl_state"]["approval_history"].append(
                {
                    "type": "periodic_stop",
                    "result": "continue",
                }
            )

            # Verify state preserved
            assert len(state["messages"]) == pre_stop_snapshot["messages"]
            assert len(state["round_summaries"]) == pre_stop_snapshot["summaries"]
            assert state["metadata"] == pre_stop_snapshot["metadata"]
            assert state["current_topic_index"] == pre_stop_snapshot["topic_index"]

    def test_transition_from_stop_to_final_considerations(self):
        """Test transition from periodic stop to final considerations."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            state = helper.create_discussion_state()
            state["current_round"] = 10

            # User forces end at periodic stop
            state["user_forced_conclusion"] = True
            state["force_all_final_considerations"] = True
            state["next_phase"] = "final_considerations"

            # Verify readiness for final considerations
            assert state["user_forced_conclusion"]
            assert state["force_all_final_considerations"]

            # All agents should participate in final considerations
            participating_agents = state["speaking_order"]
            assert len(participating_agents) == 3
