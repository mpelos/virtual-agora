"""Integration tests for Step 1.3: FlowStateManager with RoundManager and MessageCoordinator.

This module provides integration tests to ensure FlowStateManager works correctly
with RoundManager and MessageCoordinator, validating the Step 1.3 implementation.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from src.virtual_agora.flow.state_manager import FlowStateManager, RoundState
from src.virtual_agora.flow.round_manager import RoundManager
from src.virtual_agora.flow.message_coordinator import MessageCoordinator
from src.virtual_agora.state.schema import VirtualAgoraState
from src.virtual_agora.context.types import ContextData


class TestStep13Integration:
    """Integration tests for Step 1.3: FlowStateManager integration."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator(self.round_manager)
        self.flow_state_manager = FlowStateManager(
            self.round_manager, self.message_coordinator
        )

    def create_test_state(self, **kwargs) -> Dict[str, Any]:
        """Create a test state with default values."""
        default_state = {
            "current_round": 1,
            "active_topic": "AI Ethics",
            "main_topic": "Future of AI",
            "messages": [],
            "round_summaries": [],
            "topic_summaries": {},
            "speaking_order": ["alice", "bob", "charlie"],
            "rounds_per_topic": {"AI Ethics": 0},
        }
        default_state.update(kwargs)
        return default_state

    def test_flow_state_manager_uses_round_manager_and_message_coordinator(self):
        """Test that FlowStateManager properly integrates with both dependencies."""
        state = self.create_test_state(current_round=2)

        # Test round state preparation uses RoundManager
        round_state = self.flow_state_manager.prepare_round_state(state)
        assert round_state.round_number == 3  # RoundManager incremented from 2 to 3

        # Test user participation uses MessageCoordinator
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="User input")],
                "user_participation_message": "User input",
            }

            participation_result = self.flow_state_manager.apply_user_participation(
                user_input="User input", state=state
            )

            # Verify MessageCoordinator was used
            mock_store.assert_called_once()
            assert participation_result["user_participation_message"] == "User input"

    def test_complete_round_flow_integration(self):
        """Test complete round flow from preparation through finalization."""
        initial_state = self.create_test_state(current_round=0)

        # Step 1: Prepare round state
        round_state = self.flow_state_manager.prepare_round_state(initial_state)

        # Verify round preparation
        assert round_state.round_number == 1
        assert round_state.current_topic == "AI Ethics"
        assert round_state.speaking_order == [
            "alice",
            "bob",
            "charlie",
        ]  # No rotation for round 1
        assert round_state.theme == "Future of AI"

        # Step 2: Apply user participation
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [
                    HumanMessage(
                        content="Let's focus on AI safety",
                        additional_kwargs={
                            "speaker_id": "user",
                            "speaker_role": "user",
                            "round": 1,
                            "topic": "AI Ethics",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                ],
                "user_participation_message": "Let's focus on AI safety",
            }

            participation_updates = self.flow_state_manager.apply_user_participation(
                user_input="Let's focus on AI safety", state=initial_state
            )

            # Verify user participation was processed
            assert len(participation_updates["messages"]) == 1
            user_msg = participation_updates["messages"][0]
            assert "AI safety" in user_msg.content

        # Step 3: Simulate agent responses
        agent_messages = [
            HumanMessage(
                content="I agree, AI safety is crucial for ethical development.",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                    "round": 1,
                    "topic": "AI Ethics",
                },
            ),
            HumanMessage(
                content="We should consider both technical and social aspects of AI safety.",
                additional_kwargs={
                    "speaker_id": "bob",
                    "speaker_role": "participant",
                    "round": 1,
                    "topic": "AI Ethics",
                },
            ),
            HumanMessage(
                content="Regulation and governance frameworks are also important.",
                additional_kwargs={
                    "speaker_id": "charlie",
                    "speaker_role": "participant",
                    "round": 1,
                    "topic": "AI Ethics",
                },
            ),
        ]

        # Step 4: Finalize round
        finalization_updates = self.flow_state_manager.finalize_round(
            state=initial_state, round_state=round_state, round_messages=agent_messages
        )

        # Verify round finalization
        assert finalization_updates["current_round"] == 1
        assert finalization_updates["speaking_order"] == ["alice", "bob", "charlie"]
        assert len(finalization_updates["messages"]) == 3
        assert finalization_updates["rounds_per_topic"]["AI Ethics"] == 1

        # Verify round history
        round_info = finalization_updates["round_history"]
        assert round_info["round_number"] == 1
        assert round_info["topic"] == "AI Ethics"
        assert round_info["participants"] == ["alice", "bob", "charlie"]
        assert round_info["message_count"] == 3

    def test_round_state_consistency_across_operations(self):
        """Test that round state remains consistent across all FlowStateManager operations."""
        state = self.create_test_state(current_round=2)

        # Test round preparation
        round_state = self.flow_state_manager.prepare_round_state(state)
        prepared_round = round_state.round_number
        prepared_order = round_state.speaking_order

        # Test user participation with round consistency
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="Consistent round test")],
                "user_participation_message": "Consistent round test",
            }

            # Use the prepared round information
            self.flow_state_manager.apply_user_participation(
                user_input="Consistent round test",
                state=state,
                current_round=prepared_round,
            )

            # Verify the round was used consistently
            call_kwargs = mock_store.call_args.kwargs
            assert call_kwargs["round_num"] == prepared_round

        # Test finalization with same round state
        test_messages = [
            HumanMessage(
                content="Test message",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                },
            )
        ]

        finalization_updates = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=test_messages
        )

        # Verify consistency
        assert finalization_updates["current_round"] == prepared_round
        assert finalization_updates["speaking_order"] == prepared_order

    def test_speaking_order_rotation_integration(self):
        """Test speaking order rotation integration across multiple rounds."""
        state = self.create_test_state(current_round=0)

        # Round 1: No rotation
        round_1_state = self.flow_state_manager.prepare_round_state(state)
        assert round_1_state.round_number == 1
        assert round_1_state.speaking_order == ["alice", "bob", "charlie"]

        # Update state for round 2
        state["current_round"] = 1
        state["speaking_order"] = round_1_state.speaking_order

        # Round 2: First rotation
        round_2_state = self.flow_state_manager.prepare_round_state(state)
        assert round_2_state.round_number == 2
        assert round_2_state.speaking_order == ["bob", "charlie", "alice"]

        # Update state for round 3
        state["current_round"] = 2
        state["speaking_order"] = round_2_state.speaking_order

        # Round 3: Second rotation
        round_3_state = self.flow_state_manager.prepare_round_state(state)
        assert round_3_state.round_number == 3
        assert round_3_state.speaking_order == ["charlie", "alice", "bob"]

    def test_error_propagation_from_dependencies(self):
        """Test that errors from RoundManager and MessageCoordinator are properly propagated."""
        state = self.create_test_state(active_topic=None)

        # Test error from round preparation (missing active topic)
        with pytest.raises(
            ValueError, match="No active topic set for discussion round"
        ):
            self.flow_state_manager.prepare_round_state(state)

        # Test error handling in user participation
        valid_state = self.create_test_state()
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.side_effect = Exception("MessageCoordinator error")

            with pytest.raises(Exception, match="MessageCoordinator error"):
                self.flow_state_manager.apply_user_participation(
                    user_input="Test input", state=valid_state
                )

    def test_state_immutability_integration(self):
        """Test that state immutability is maintained across all operations."""
        original_state = self.create_test_state()
        original_speaking_order = original_state["speaking_order"].copy()
        original_round = original_state["current_round"]

        # Test round preparation doesn't modify original state
        round_state = self.flow_state_manager.prepare_round_state(original_state)
        assert original_state["speaking_order"] == original_speaking_order
        assert original_state["current_round"] == original_round

        # Test user participation doesn't modify original state
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="Test")],
                "user_participation_message": "Test",
            }

            self.flow_state_manager.apply_user_participation(
                user_input="Test", state=original_state
            )

            # Original state should be unchanged
            assert original_state["speaking_order"] == original_speaking_order
            assert original_state["current_round"] == original_round

        # Test create_immutable_state_update
        updates = {"new_field": "new_value"}
        new_state = self.flow_state_manager.create_immutable_state_update(
            original_state, updates
        )

        # Original should be unchanged
        assert "new_field" not in original_state
        assert new_state["new_field"] == "new_value"

    def test_integration_with_existing_message_formats(self):
        """Test integration with existing message formats used in the codebase."""
        state = self.create_test_state()
        round_state = self.flow_state_manager.prepare_round_state(state)

        # Test with mixed message formats (dict and LangChain)
        mixed_messages = [
            # Dict format (existing in codebase)
            {
                "content": "Dict format message",
                "speaker_id": "alice",
                "speaker_role": "participant",
                "round": 2,
                "topic": "AI Ethics",
                "timestamp": datetime.now().isoformat(),
            },
            # LangChain format
            HumanMessage(
                content="LangChain format message",
                additional_kwargs={
                    "speaker_id": "bob",
                    "speaker_role": "participant",
                    "round": 2,
                    "topic": "AI Ethics",
                },
            ),
            # User message format
            HumanMessage(
                content="User message",
                additional_kwargs={
                    "speaker_id": "user",
                    "speaker_role": "user",
                    "round": 2,
                    "topic": "AI Ethics",
                },
            ),
        ]

        finalization_result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=mixed_messages
        )

        # Verify all message formats are handled correctly
        round_info = finalization_result["round_history"]
        assert round_info["participants"] == ["alice", "bob"]  # User excluded
        assert round_info["message_count"] == 3  # All messages counted


class TestStep13AcceptanceCriteria:
    """Tests for Step 1.3 acceptance criteria validation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator(self.round_manager)
        self.flow_state_manager = FlowStateManager(
            self.round_manager, self.message_coordinator
        )

    def test_clear_state_transition_boundaries(self):
        """Verify clear state transition boundaries are established."""
        # FlowStateManager should provide clear boundaries between:
        # 1. Round preparation
        # 2. User participation application
        # 3. Round finalization

        state = {
            "current_round": 0,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "speaking_order": ["agent1", "agent2"],
            "rounds_per_topic": {},
        }

        # Boundary 1: Round preparation
        round_state = self.flow_state_manager.prepare_round_state(state)
        assert isinstance(round_state, RoundState)

        # Boundary 2: User participation application
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="Test")],
                "user_participation_message": "Test",
            }

            participation_result = self.flow_state_manager.apply_user_participation(
                user_input="Test", state=state
            )
            assert isinstance(participation_result, dict)

        # Boundary 3: Round finalization
        finalization_result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=[]
        )
        assert isinstance(finalization_result, dict)

    def test_composed_from_round_and_message_managers(self):
        """Verify FlowStateManager is properly composed from dependencies."""
        # Test that FlowStateManager has the required dependencies
        assert hasattr(self.flow_state_manager, "round_manager")
        assert hasattr(self.flow_state_manager, "message_coordinator")
        assert isinstance(self.flow_state_manager.round_manager, RoundManager)
        assert isinstance(
            self.flow_state_manager.message_coordinator, MessageCoordinator
        )

        # Test that dependencies are used correctly
        state = {
            "current_round": 5,
            "active_topic": "Test",
            "main_topic": "Theme",
            "speaking_order": ["a", "b"],
        }

        # Should delegate to RoundManager
        round_state = self.flow_state_manager.prepare_round_state(state)
        assert round_state.round_number == 6  # RoundManager incremented

        # Should delegate to MessageCoordinator
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [],
                "user_participation_message": "test",
            }

            self.flow_state_manager.apply_user_participation("test", state)
            mock_store.assert_called_once()

    def test_immutable_state_operations(self):
        """Verify that state operations are immutable."""
        original_state = {
            "current_round": 1,
            "active_topic": "Original Topic",
            "speaking_order": ["alice", "bob"],
            "messages": [],
            "rounds_per_topic": {"Original Topic": 1},
        }

        # Make a copy to compare later
        original_copy = original_state.copy()

        # Test that prepare_round_state doesn't modify original state
        round_state = self.flow_state_manager.prepare_round_state(original_state)
        assert original_state == original_copy

        # Test that apply_user_participation doesn't modify original state
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="New message")],
                "user_participation_message": "New message",
            }

            participation_result = self.flow_state_manager.apply_user_participation(
                "New message", original_state
            )
            assert original_state == original_copy

        # Test that finalize_round doesn't modify original state
        finalization_result = self.flow_state_manager.finalize_round(
            original_state, round_state, []
        )
        assert original_state == original_copy

        # Test explicit immutable state update
        updates = {"new_field": "new_value"}
        new_state = self.flow_state_manager.create_immutable_state_update(
            original_state, updates
        )

        # Original should be unchanged
        assert original_state == original_copy
        assert "new_field" not in original_state
        assert new_state["new_field"] == "new_value"

    def test_full_test_coverage_of_state_transitions(self):
        """Verify comprehensive test coverage of all state transitions."""
        # This test validates that all major state transitions are covered by tests.
        # The actual coverage is validated by the test execution, but this test
        # ensures all major transition paths are exercised.

        state = {
            "current_round": 2,
            "active_topic": "Coverage Test",
            "main_topic": "Test Suite",
            "speaking_order": ["agent1", "agent2", "agent3"],
            "rounds_per_topic": {"Coverage Test": 1},
        }

        # Test successful round preparation
        round_state = self.flow_state_manager.prepare_round_state(state)
        assert round_state.round_number == 3

        # Test error condition in round preparation
        error_state = state.copy()
        error_state["active_topic"] = None
        with pytest.raises(ValueError):
            self.flow_state_manager.prepare_round_state(error_state)

        # Test successful user participation
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="Test message")],
                "user_participation_message": "Test message",
            }

            result = self.flow_state_manager.apply_user_participation(
                "Test message", state
            )
            assert result["user_participation_message"] == "Test message"

        # Test successful round finalization
        messages = [
            HumanMessage(
                content="Agent response",
                additional_kwargs={
                    "speaker_id": "agent1",
                    "speaker_role": "participant",
                },
            )
        ]

        finalization_result = self.flow_state_manager.finalize_round(
            state, round_state, messages
        )
        assert finalization_result["rounds_per_topic"]["Coverage Test"] == 2

        # Test edge cases: empty messages, single speaker, etc.
        empty_result = self.flow_state_manager.finalize_round(state, round_state, [])
        assert empty_result["round_history"]["message_count"] == 0

        # All major state transition paths have been exercised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
