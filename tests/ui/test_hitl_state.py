"""Tests for HITL state management in v1.3."""

import pytest
from datetime import datetime
from typing import Dict, Any

from virtual_agora.ui.hitl_state import (
    HITLApprovalType,
    HITLInteraction,
    HITLContext,
    HITLResponse,
    HITLStateTracker,
)


class TestHITLApprovalType:
    """Test the HITL approval type enum."""

    def test_all_types_defined(self):
        """Test that all HITL types are defined."""
        expected_types = [
            "theme_input",
            "agenda_approval",
            "periodic_stop",
            "topic_override",
            "topic_continuation",
            "agent_poll_override",
            "session_continuation",
            "final_report_approval",
        ]

        actual_types = [t.value for t in HITLApprovalType]
        assert set(actual_types) == set(expected_types)

    def test_type_descriptions(self):
        """Test that all types have descriptions."""
        for approval_type in HITLApprovalType:
            desc = approval_type.description
            assert desc is not None
            assert len(desc) > 0
            assert isinstance(desc, str)


class TestHITLContext:
    """Test the HITL context class."""

    def test_initialization_with_defaults(self):
        """Test context initialization with default values."""
        context = HITLContext()

        assert context.current_round is None
        assert context.active_topic is None
        assert context.completed_topics == []
        assert context.remaining_topics == []
        assert context.proposed_agenda == []
        assert context.session_stats == {}
        assert context.agent_vote_result is None
        assert context.custom_data == {}

    def test_initialization_with_values(self):
        """Test context initialization with provided values."""
        context = HITLContext(
            current_round=5,
            active_topic="Test Topic",
            completed_topics=["Topic 1", "Topic 2"],
            session_stats={"messages": 100},
            custom_data={"key": "value"},
        )

        assert context.current_round == 5
        assert context.active_topic == "Test Topic"
        assert len(context.completed_topics) == 2
        assert context.session_stats["messages"] == 100
        assert context.custom_data["key"] == "value"

    def test_to_dict(self):
        """Test converting context to dictionary."""
        context = HITLContext(
            current_round=10,
            active_topic="Active",
            agent_vote_result={"recommendation": "continue"},
        )

        context_dict = context.to_dict()

        assert isinstance(context_dict, dict)
        assert context_dict["current_round"] == 10
        assert context_dict["active_topic"] == "Active"
        assert context_dict["agent_vote_result"]["recommendation"] == "continue"

    def test_from_dict(self):
        """Test creating context from dictionary."""
        data = {
            "current_round": 15,
            "active_topic": "Discussion",
            "completed_topics": ["A", "B", "C"],
            "session_stats": {"duration": "1h"},
            "unknown_field": "ignored",
        }

        context = HITLContext.from_dict(data)

        assert context.current_round == 15
        assert context.active_topic == "Discussion"
        assert len(context.completed_topics) == 3
        assert context.session_stats["duration"] == "1h"
        # Unknown fields should be ignored
        assert not hasattr(context, "unknown_field")


class TestHITLInteraction:
    """Test the HITL interaction class."""

    def test_initialization(self):
        """Test interaction initialization."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.PERIODIC_STOP,
            prompt_message="5-round checkpoint",
            options=["continue", "end", "modify"],
            context={"round": 5},
        )

        assert interaction.approval_type == HITLApprovalType.PERIODIC_STOP
        assert interaction.prompt_message == "5-round checkpoint"
        assert len(interaction.options) == 3
        assert interaction.context["round"] == 5
        assert interaction.start_time is not None
        assert interaction.end_time is None
        assert interaction.response_time_seconds is None

    def test_default_values(self):
        """Test interaction with default values."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.THEME_INPUT, prompt_message="Enter theme"
        )

        assert interaction.options is None
        assert interaction.context == {}
        assert interaction.metadata == {}

    def test_metadata_handling(self):
        """Test metadata storage in interaction."""
        metadata = {
            "user_id": "test_user",
            "session_version": "1.3",
            "custom": {"nested": "data"},
        }

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Approve?",
            metadata=metadata,
        )

        assert interaction.metadata == metadata
        assert interaction.metadata["custom"]["nested"] == "data"


class TestHITLResponse:
    """Test the HITL response class."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = HITLResponse(
            approved=True, action="continue", reason="User wants to continue"
        )

        assert response.approved is True
        assert response.action == "continue"
        assert response.reason == "User wants to continue"
        assert response.modified_data is None
        assert response.metadata == {}

    def test_response_with_modified_data(self):
        """Test response with modified data."""
        new_agenda = ["Topic A", "Topic B", "Topic C"]

        response = HITLResponse(
            approved=True,
            action="edited",
            modified_data=new_agenda,
            metadata={"edit_count": 2},
        )

        assert response.modified_data == new_agenda
        assert response.metadata["edit_count"] == 2

    def test_rejection_response(self):
        """Test rejection response."""
        response = HITLResponse(
            approved=False, action="rejected", reason="User declined to continue"
        )

        assert response.approved is False
        assert response.action == "rejected"
        assert response.reason == "User declined to continue"

    def test_to_dict(self):
        """Test converting response to dictionary."""
        response = HITLResponse(
            approved=True,
            action="override",
            reason="User knows better",
            metadata={"confidence": 0.9},
        )

        response_dict = response.to_dict()

        assert response_dict["approved"] is True
        assert response_dict["action"] == "override"
        assert response_dict["reason"] == "User knows better"
        assert response_dict["metadata"]["confidence"] == 0.9


class TestHITLStateTracker:
    """Test the HITL state tracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = HITLStateTracker()

        assert tracker.interactions == []
        assert tracker.approval_history == []
        assert tracker.current_interaction is None

    def test_start_interaction(self):
        """Test starting an interaction."""
        tracker = HITLStateTracker()

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.THEME_INPUT, prompt_message="Enter theme"
        )

        tracker.start_interaction(interaction)

        assert tracker.current_interaction == interaction
        assert len(tracker.interactions) == 0  # Not added until completed

    def test_complete_interaction(self):
        """Test completing an interaction."""
        tracker = HITLStateTracker()

        # Start interaction
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Approve agenda?",
            context={"agenda_size": 5},
        )
        tracker.start_interaction(interaction)

        # Complete with response
        response = HITLResponse(approved=True, action="approved")
        tracker.complete_interaction(response)

        # Check results
        assert tracker.current_interaction is None
        assert len(tracker.interactions) == 1
        assert len(tracker.approval_history) == 1

        # Check stored interaction
        stored = tracker.interactions[0]
        assert stored.approval_type == HITLApprovalType.AGENDA_APPROVAL
        assert stored.end_time is not None
        assert stored.response_time_seconds > 0

        # Check approval history
        history = tracker.approval_history[0]
        assert history["type"] == "agenda_approval"
        assert history["approved"] is True
        assert history["action"] == "approved"
        assert history["context"]["agenda_size"] == 5

    def test_complete_without_start(self):
        """Test completing interaction without starting."""
        tracker = HITLStateTracker()

        response = HITLResponse(approved=True, action="test")

        # Should not raise error, just do nothing
        tracker.complete_interaction(response)

        assert len(tracker.interactions) == 0
        assert len(tracker.approval_history) == 0

    def test_get_stats(self):
        """Test getting statistics."""
        tracker = HITLStateTracker()

        # Process multiple interactions
        interaction_types = [
            HITLApprovalType.PERIODIC_STOP,
            HITLApprovalType.PERIODIC_STOP,
            HITLApprovalType.TOPIC_CONTINUATION,
            HITLApprovalType.AGENDA_APPROVAL,
        ]

        for approval_type in interaction_types:
            interaction = HITLInteraction(
                approval_type=approval_type, prompt_message="Test"
            )
            tracker.start_interaction(interaction)

            response = HITLResponse(
                approved=(
                    True
                    if approval_type != HITLApprovalType.TOPIC_CONTINUATION
                    else False
                ),
                action="test",
            )
            tracker.complete_interaction(response)

        stats = tracker.get_stats()

        assert stats["total_interactions"] == 4
        assert stats["approved_count"] == 3
        assert stats["rejected_count"] == 1
        assert stats["interactions_by_type"]["periodic_stop"] == 2
        assert stats["interactions_by_type"]["topic_continuation"] == 1
        assert stats["interactions_by_type"]["agenda_approval"] == 1
        assert stats["periodic_stops"] == 2
        assert stats["average_response_time"] > 0

    def test_get_recent_interactions(self):
        """Test getting recent interactions."""
        tracker = HITLStateTracker()

        # Add 5 interactions
        for i in range(5):
            interaction = HITLInteraction(
                approval_type=HITLApprovalType.PERIODIC_STOP,
                prompt_message=f"Interaction {i}",
            )
            tracker.start_interaction(interaction)
            response = HITLResponse(approved=True, action=f"action_{i}")
            tracker.complete_interaction(response)

        # Get last 3
        recent = tracker.get_recent_interactions(3)

        assert len(recent) == 3
        assert recent[0].prompt_message == "Interaction 4"  # Most recent
        assert recent[1].prompt_message == "Interaction 3"
        assert recent[2].prompt_message == "Interaction 2"

    def test_clear_history(self):
        """Test clearing history."""
        tracker = HITLStateTracker()

        # Add some interactions
        for _ in range(3):
            interaction = HITLInteraction(
                approval_type=HITLApprovalType.THEME_INPUT, prompt_message="Test"
            )
            tracker.start_interaction(interaction)
            response = HITLResponse(approved=True, action="test")
            tracker.complete_interaction(response)

        # Clear history
        tracker.clear_history()

        assert len(tracker.interactions) == 0
        assert len(tracker.approval_history) == 0
        assert tracker.current_interaction is None

    def test_interaction_timing_accuracy(self):
        """Test that interaction timing is accurate."""
        import time

        tracker = HITLStateTracker()

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.PERIODIC_STOP, prompt_message="Test timing"
        )

        tracker.start_interaction(interaction)

        # Simulate user thinking time
        time.sleep(0.5)

        response = HITLResponse(approved=True, action="continue")
        tracker.complete_interaction(response)

        # Check timing
        completed = tracker.interactions[0]
        assert completed.response_time_seconds >= 0.5
        assert completed.response_time_seconds < 1.0  # Reasonable upper bound

    def test_concurrent_interaction_handling(self):
        """Test handling of concurrent interaction attempts."""
        tracker = HITLStateTracker()

        # Start first interaction
        interaction1 = HITLInteraction(
            approval_type=HITLApprovalType.THEME_INPUT, prompt_message="First"
        )
        tracker.start_interaction(interaction1)

        # Try to start second interaction (should replace first)
        interaction2 = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL, prompt_message="Second"
        )
        tracker.start_interaction(interaction2)

        assert tracker.current_interaction == interaction2

        # Complete the second interaction
        response = HITLResponse(approved=True, action="test")
        tracker.complete_interaction(response)

        # Only the second interaction should be recorded
        assert len(tracker.interactions) == 1
        assert tracker.interactions[0].prompt_message == "Second"
