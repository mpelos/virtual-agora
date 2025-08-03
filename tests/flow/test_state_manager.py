"""Tests for FlowStateManager class.

This module provides comprehensive tests for the centralized flow state management
functionality, ensuring 100% coverage and proper integration with RoundManager
and MessageCoordinator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any
import uuid

from langchain_core.messages import HumanMessage, AIMessage

from src.virtual_agora.flow.state_manager import FlowStateManager, RoundState
from src.virtual_agora.flow.round_manager import RoundManager
from src.virtual_agora.flow.message_coordinator import MessageCoordinator
from src.virtual_agora.state.schema import VirtualAgoraState


class TestFlowStateManager:
    """Test suite for FlowStateManager class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator(self.round_manager)
        self.flow_state_manager = FlowStateManager(
            self.round_manager, self.message_coordinator
        )

    def create_test_state(self, **kwargs) -> Dict[str, Any]:
        """Create a test state with default values.

        Args:
            **kwargs: Override default state values

        Returns:
            Dictionary representing VirtualAgoraState for testing
        """
        default_state = {
            "current_round": 2,
            "active_topic": "AI Ethics",
            "main_topic": "Future of AI",
            "messages": [],
            "round_summaries": [],
            "topic_summaries": {},
            "speaking_order": ["alice", "bob", "charlie"],
            "rounds_per_topic": {"AI Ethics": 1},
        }
        default_state.update(kwargs)
        return default_state

    # ===== Tests for prepare_round_state =====

    def test_prepare_round_state_basic(self):
        """Test basic round state preparation."""
        state = self.create_test_state()

        result = self.flow_state_manager.prepare_round_state(state)

        # Verify RoundState structure
        assert isinstance(result, RoundState)
        assert result.round_number == 3  # round_manager.start_new_round() increments
        assert result.current_topic == "AI Ethics"
        assert result.speaking_order == [
            "bob",
            "charlie",
            "alice",
        ]  # Rotated for round 3
        assert result.theme == "Future of AI"
        assert isinstance(result.round_id, str)
        assert isinstance(result.round_start_time, datetime)

    def test_prepare_round_state_no_active_topic(self):
        """Test round state preparation with no active topic."""
        state = self.create_test_state(active_topic=None)

        with pytest.raises(
            ValueError, match="No active topic set for discussion round"
        ):
            self.flow_state_manager.prepare_round_state(state)

    def test_prepare_round_state_missing_active_topic(self):
        """Test round state preparation with missing active topic key."""
        state = self.create_test_state()
        del state["active_topic"]

        with pytest.raises(
            ValueError, match="No active topic set for discussion round"
        ):
            self.flow_state_manager.prepare_round_state(state)

    def test_prepare_round_state_no_theme(self):
        """Test round state preparation with no main topic."""
        state = self.create_test_state(main_topic=None)

        result = self.flow_state_manager.prepare_round_state(state)

        assert result.theme == "Unknown Topic"

    def test_prepare_round_state_first_round(self):
        """Test round state preparation for first round (no rotation)."""
        state = self.create_test_state(current_round=0)

        result = self.flow_state_manager.prepare_round_state(state)

        # First round (round 1) should not rotate speaking order
        assert result.round_number == 1
        assert result.speaking_order == ["alice", "bob", "charlie"]  # No rotation

    def test_prepare_round_state_empty_speaking_order(self):
        """Test round state preparation with empty speaking order."""
        state = self.create_test_state(speaking_order=[])

        result = self.flow_state_manager.prepare_round_state(state)

        assert result.speaking_order == []

    def test_prepare_round_state_missing_speaking_order(self):
        """Test round state preparation with missing speaking order."""
        state = self.create_test_state()
        del state["speaking_order"]

        result = self.flow_state_manager.prepare_round_state(state)

        assert result.speaking_order == []

    def test_prepare_round_state_single_speaker(self):
        """Test round state preparation with single speaker."""
        state = self.create_test_state(speaking_order=["alice"])

        result = self.flow_state_manager.prepare_round_state(state)

        # Single speaker rotation: [A] -> [A]
        assert result.speaking_order == ["alice"]

    def test_prepare_round_state_round_3_rotation(self):
        """Test speaking order rotation for round 3."""
        state = self.create_test_state(current_round=2)  # Will become round 3

        result = self.flow_state_manager.prepare_round_state(state)

        # Round 3: [A,B,C] -> [B,C,A] -> [C,A,B]
        assert result.round_number == 3
        assert result.speaking_order == ["bob", "charlie", "alice"]

    # ===== Tests for apply_user_participation =====

    @patch.object(MessageCoordinator, "store_user_message")
    def test_apply_user_participation_basic(self, mock_store_message):
        """Test basic user participation application."""
        state = self.create_test_state()
        mock_store_message.return_value = {
            "messages": [HumanMessage(content="Test message")],
            "user_participation_message": "Test input",
        }

        result = self.flow_state_manager.apply_user_participation(
            user_input="Test input", state=state
        )

        # Verify MessageCoordinator was called correctly
        mock_store_message.assert_called_once_with(
            content="Test input",
            round_num=2,
            state=state,
            topic="AI Ethics",
            participation_type="user_turn_participation",
            use_next_round=True,
        )

        # Verify result
        assert "messages" in result
        assert "user_participation_message" in result
        assert result["user_participation_message"] == "Test input"

    @patch.object(MessageCoordinator, "store_user_message")
    def test_apply_user_participation_with_parameters(self, mock_store_message):
        """Test user participation application with custom parameters."""
        state = self.create_test_state()
        mock_store_message.return_value = {
            "messages": [HumanMessage(content="Custom message")],
            "user_participation_message": "Custom input",
        }

        result = self.flow_state_manager.apply_user_participation(
            user_input="Custom input",
            state=state,
            current_round=5,
            topic="Custom Topic",
            participation_type="custom_participation",
            use_next_round=False,
        )

        # Verify MessageCoordinator was called with custom parameters
        mock_store_message.assert_called_once_with(
            content="Custom input",
            round_num=5,
            state=state,
            topic="Custom Topic",
            participation_type="custom_participation",
            use_next_round=False,
        )

    @patch.object(MessageCoordinator, "store_user_message")
    def test_apply_user_participation_no_active_topic(self, mock_store_message):
        """Test user participation application with no active topic."""
        state = self.create_test_state(active_topic=None)
        mock_store_message.return_value = {
            "messages": [HumanMessage(content="Test message")],
            "user_participation_message": "Test input",
        }

        self.flow_state_manager.apply_user_participation(
            user_input="Test input", state=state
        )

        # Should use "Unknown Topic" as default
        mock_store_message.assert_called_once()
        call_kwargs = mock_store_message.call_args.kwargs
        assert call_kwargs["topic"] == "Unknown Topic"

    @patch.object(MessageCoordinator, "store_user_message")
    def test_apply_user_participation_empty_input(self, mock_store_message):
        """Test user participation application with empty input."""
        state = self.create_test_state()
        mock_store_message.return_value = {
            "messages": [HumanMessage(content="")],
            "user_participation_message": "",
        }

        result = self.flow_state_manager.apply_user_participation(
            user_input="", state=state
        )

        # Verify empty input is handled
        mock_store_message.assert_called_once_with(
            content="",
            round_num=2,
            state=state,
            topic="AI Ethics",
            participation_type="user_turn_participation",
            use_next_round=True,
        )

    # ===== Tests for finalize_round =====

    def test_finalize_round_basic(self):
        """Test basic round finalization."""
        state = self.create_test_state()
        round_state = RoundState(
            round_number=3,
            current_topic="AI Ethics",
            speaking_order=["bob", "charlie", "alice"],
            round_id="test-round-id",
            round_start_time=datetime(2025, 1, 1, 12, 0, 0),
            theme="Future of AI",
        )

        # Create test messages with different formats
        round_messages = [
            HumanMessage(
                content="Agent message 1",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                },
            ),
            {
                "content": "Agent message 2",
                "speaker_id": "bob",
                "speaker_role": "participant",
            },
        ]

        result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=round_messages
        )

        # Verify state updates structure
        assert "current_round" in result
        assert "speaking_order" in result
        assert "messages" in result
        assert "round_history" in result
        assert "turn_order_history" in result
        assert "rounds_per_topic" in result

        # Verify values
        assert result["current_round"] == 3
        assert result["speaking_order"] == ["bob", "charlie", "alice"]
        assert result["messages"] == round_messages
        assert result["turn_order_history"] == ["bob", "charlie", "alice"]

        # Verify round history
        round_info = result["round_history"]
        assert round_info["round_id"] == "test-round-id"
        assert round_info["round_number"] == 3
        assert round_info["topic"] == "AI Ethics"
        assert round_info["start_time"] == datetime(2025, 1, 1, 12, 0, 0)
        assert isinstance(round_info["end_time"], datetime)
        assert round_info["participants"] == ["alice", "bob"]
        assert round_info["message_count"] == 2
        assert round_info["summary"] is None

        # Verify rounds per topic counter
        expected_rounds_per_topic = {"AI Ethics": 2}  # Was 1, now incremented to 2
        assert result["rounds_per_topic"] == expected_rounds_per_topic

    def test_finalize_round_empty_messages(self):
        """Test round finalization with no messages."""
        state = self.create_test_state()
        round_state = RoundState(
            round_number=1,
            current_topic="AI Ethics",
            speaking_order=["alice"],
            round_id="empty-round-id",
            round_start_time=datetime.now(),
            theme="Future of AI",
        )

        result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=[]
        )

        # Verify empty round handling
        round_info = result["round_history"]
        assert round_info["participants"] == []
        assert round_info["message_count"] == 0
        assert result["messages"] == []

    def test_finalize_round_new_topic(self):
        """Test round finalization for new topic."""
        state = self.create_test_state(rounds_per_topic={})
        round_state = RoundState(
            round_number=1,
            current_topic="New Topic",
            speaking_order=["alice"],
            round_id="new-topic-id",
            round_start_time=datetime.now(),
            theme="Future of AI",
        )

        result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=[]
        )

        # Verify new topic counter
        assert result["rounds_per_topic"] == {"New Topic": 1}

    def test_finalize_round_mixed_speaker_roles(self):
        """Test round finalization with mixed speaker roles."""
        state = self.create_test_state()
        round_state = RoundState(
            round_number=2,
            current_topic="AI Ethics",
            speaking_order=["alice", "bob"],
            round_id="mixed-round-id",
            round_start_time=datetime.now(),
            theme="Future of AI",
        )

        # Mix of participant and user messages
        round_messages = [
            HumanMessage(
                content="Participant message",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                },
            ),
            HumanMessage(
                content="User message",
                additional_kwargs={"speaker_id": "user", "speaker_role": "user"},
            ),
            {
                "content": "Another participant",
                "speaker_id": "bob",
                "speaker_role": "participant",
            },
        ]

        result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=round_messages
        )

        # Verify only participants are included (not user)
        round_info = result["round_history"]
        assert round_info["participants"] == ["alice", "bob"]
        assert round_info["message_count"] == 3  # All messages counted

    def test_finalize_round_duplicate_participants(self):
        """Test round finalization with duplicate participants."""
        state = self.create_test_state()
        round_state = RoundState(
            round_number=2,
            current_topic="AI Ethics",
            speaking_order=["alice"],
            round_id="duplicate-round-id",
            round_start_time=datetime.now(),
            theme="Future of AI",
        )

        # Multiple messages from same participant
        round_messages = [
            HumanMessage(
                content="First message",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                },
            ),
            HumanMessage(
                content="Second message",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                },
            ),
        ]

        result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=round_messages
        )

        # Verify participant is listed only once
        round_info = result["round_history"]
        assert round_info["participants"] == ["alice"]
        assert round_info["message_count"] == 2

    # ===== Tests for create_immutable_state_update =====

    def test_create_immutable_state_update_basic(self):
        """Test immutable state update creation."""
        original_state = self.create_test_state()
        updates = {"current_round": 5, "new_field": "new_value"}

        result = self.flow_state_manager.create_immutable_state_update(
            original_state, updates
        )

        # Verify immutability - original state unchanged
        assert original_state["current_round"] == 2
        assert "new_field" not in original_state

        # Verify new state has updates
        assert result["current_round"] == 5
        assert result["new_field"] == "new_value"

        # Verify other fields preserved
        assert result["active_topic"] == "AI Ethics"
        assert result["main_topic"] == "Future of AI"

    def test_create_immutable_state_update_empty_updates(self):
        """Test immutable state update with empty updates."""
        original_state = self.create_test_state()
        updates = {}

        result = self.flow_state_manager.create_immutable_state_update(
            original_state, updates
        )

        # Verify deep copy was made
        assert result is not original_state
        assert result == original_state

    def test_create_immutable_state_update_nested_data(self):
        """Test immutable state update with nested data structures."""
        original_state = self.create_test_state()
        original_state["nested"] = {"key": "value", "list": [1, 2, 3]}

        updates = {"nested": {"key": "updated", "new_key": "new_value"}}

        result = self.flow_state_manager.create_immutable_state_update(
            original_state, updates
        )

        # Verify original nested structure unchanged
        assert original_state["nested"]["key"] == "value"
        assert "new_key" not in original_state["nested"]

        # Verify updated state
        assert result["nested"]["key"] == "updated"
        assert result["nested"]["new_key"] == "new_value"

    # ===== Integration Tests =====

    def test_initialization_with_dependencies(self):
        """Test FlowStateManager initialization with dependencies."""
        round_manager = RoundManager()
        message_coordinator = MessageCoordinator(round_manager)
        flow_state_manager = FlowStateManager(round_manager, message_coordinator)

        assert flow_state_manager.round_manager is round_manager
        assert flow_state_manager.message_coordinator is message_coordinator

    @patch.object(MessageCoordinator, "store_user_message")
    def test_integration_with_message_coordinator(self, mock_store_message):
        """Test integration with MessageCoordinator."""
        state = self.create_test_state()
        mock_store_message.return_value = {
            "messages": [HumanMessage(content="Integration test")],
            "user_participation_message": "Integration test",
        }

        # Test that FlowStateManager properly uses MessageCoordinator
        result = self.flow_state_manager.apply_user_participation(
            user_input="Integration test", state=state
        )

        # Verify delegation to MessageCoordinator
        mock_store_message.assert_called_once()
        assert result["user_participation_message"] == "Integration test"

    def test_integration_with_round_manager(self):
        """Test integration with RoundManager."""
        state = self.create_test_state(current_round=0)

        # Test that FlowStateManager properly uses RoundManager
        round_state = self.flow_state_manager.prepare_round_state(state)

        # Verify delegation to RoundManager
        assert round_state.round_number == 1  # RoundManager incremented from 0 to 1

    def test_complete_round_workflow(self):
        """Test complete round workflow integration."""
        state = self.create_test_state(current_round=0)

        # Step 1: Prepare round
        round_state = self.flow_state_manager.prepare_round_state(state)
        assert round_state.round_number == 1
        assert round_state.speaking_order == ["alice", "bob", "charlie"]

        # Step 2: Apply user participation (mock MessageCoordinator)
        with patch.object(self.message_coordinator, "store_user_message") as mock_store:
            mock_store.return_value = {
                "messages": [HumanMessage(content="User guidance")],
                "user_participation_message": "User guidance",
            }

            participation_result = self.flow_state_manager.apply_user_participation(
                user_input="User guidance", state=state
            )

            assert participation_result["user_participation_message"] == "User guidance"

        # Step 3: Finalize round
        round_messages = [
            HumanMessage(
                content="Test response",
                additional_kwargs={
                    "speaker_id": "alice",
                    "speaker_role": "participant",
                },
            )
        ]

        finalization_result = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=round_messages
        )

        assert finalization_result["current_round"] == 1
        assert len(finalization_result["messages"]) == 1
        assert finalization_result["rounds_per_topic"]["AI Ethics"] == 2

    # ===== Error Handling Tests =====

    def test_prepare_round_state_error_handling(self):
        """Test error handling in prepare_round_state."""
        state = self.create_test_state(active_topic="")

        with pytest.raises(
            ValueError, match="No active topic set for discussion round"
        ):
            self.flow_state_manager.prepare_round_state(state)

    def test_extract_participants_malformed_messages(self):
        """Test participant extraction with malformed messages."""
        messages = [
            {"content": "No speaker info"},  # Missing speaker_id and speaker_role
            HumanMessage(content="No additional_kwargs"),  # Missing additional_kwargs
            None,  # Invalid message
        ]

        participants = self.flow_state_manager._extract_participants(messages)

        # Should handle malformed messages gracefully
        assert participants == [
            "unknown"
        ]  # Only the dict message with default speaker_id


class TestFlowStateManagerEdgeCases:
    """Test edge cases and boundary conditions for FlowStateManager."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator(self.round_manager)
        self.flow_state_manager = FlowStateManager(
            self.round_manager, self.message_coordinator
        )

    def test_round_state_namedtuple_immutability(self):
        """Test that RoundState namedtuple is immutable."""
        round_state = RoundState(
            round_number=1,
            current_topic="Test",
            speaking_order=["alice"],
            round_id="test-id",
            round_start_time=datetime.now(),
            theme="Theme",
        )

        # Verify we cannot modify namedtuple fields
        with pytest.raises(AttributeError):
            round_state.round_number = 2

    def test_large_speaking_order_rotation(self):
        """Test speaking order rotation with large number of speakers."""
        large_order = [f"agent_{i}" for i in range(100)]
        state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "speaking_order": large_order,
        }

        round_state = self.flow_state_manager.prepare_round_state(state)

        # Verify rotation works with large order
        expected_order = large_order[1:] + [large_order[0]]
        assert round_state.speaking_order == expected_order

    def test_extreme_round_numbers(self):
        """Test handling of extreme round numbers."""
        state = {
            "current_round": 999999,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "speaking_order": ["alice", "bob"],
        }

        round_state = self.flow_state_manager.prepare_round_state(state)

        # Should handle large round numbers
        assert round_state.round_number == 1000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
