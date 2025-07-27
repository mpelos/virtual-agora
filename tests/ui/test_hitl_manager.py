"""Tests for the enhanced HITL manager in v1.3."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from rich.console import Console

from virtual_agora.ui.hitl_manager import EnhancedHITLManager
from virtual_agora.ui.hitl_state import (
    HITLApprovalType,
    HITLInteraction,
    HITLContext,
    HITLResponse,
)


@pytest.fixture
def console():
    """Mock console for testing."""
    return Mock(spec=Console)


@pytest.fixture
def hitl_manager(console):
    """Create HITL manager instance."""
    return EnhancedHITLManager(console)


class TestEnhancedHITLManager:
    """Test the enhanced HITL manager."""

    def test_initialization(self, hitl_manager):
        """Test HITL manager initialization."""
        assert hitl_manager.console is not None
        assert hitl_manager.state_tracker is not None
        assert len(hitl_manager.handlers) == 8  # All HITL types registered

    def test_periodic_stop_handler(self, hitl_manager):
        """Test periodic stop handler."""
        # Create interaction
        context = HITLContext(
            current_round=5,
            active_topic="AI Ethics",
            completed_topics=["Introduction"],
            remaining_topics=["Future Implications"],
        )

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.PERIODIC_STOP,
            prompt_message="5-round checkpoint reached",
            context=context.to_dict(),
        )

        # Mock user input
        with patch("rich.prompt.Prompt.ask", return_value="c"):
            response = hitl_manager._handle_periodic_stop(interaction)

        assert response.action == "continue"
        assert response.approved is True
        assert response.metadata["checkpoint_round"] == 5

    def test_periodic_stop_force_end(self, hitl_manager):
        """Test periodic stop with force end."""
        context = HITLContext(current_round=10, active_topic="Test Topic")
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.PERIODIC_STOP,
            prompt_message="Checkpoint",
            context=context.to_dict(),
        )

        # Mock user choosing to end topic
        with patch("rich.prompt.Prompt.ask", side_effect=["e", "User wants to end"]):
            response = hitl_manager._handle_periodic_stop(interaction)

        assert response.action == "force_topic_end"
        assert response.reason == "User wants to end"
        assert response.metadata["forced_by"] == "user"

    def test_agenda_approval_handler(self, hitl_manager):
        """Test agenda approval handler."""
        context = HITLContext(proposed_agenda=["Topic 1", "Topic 2", "Topic 3"])

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Approve agenda",
            context=context.to_dict(),
        )

        # Mock user approval
        with patch("rich.prompt.Prompt.ask", return_value="Approve"):
            response = hitl_manager._handle_agenda_approval(interaction)

        assert response.action == "approved"
        assert response.approved is True
        assert response.modified_data == ["Topic 1", "Topic 2", "Topic 3"]

    def test_agenda_editing(self, hitl_manager):
        """Test agenda editing functionality."""
        original_agenda = ["Topic A", "Topic B"]

        # Mock editing interaction
        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=[
                "Modified Topic A",  # Edit first topic
                "Topic B",  # Keep second topic
                "New Topic C",  # Add new topic
                "",  # Finish adding
            ],
        ):
            with patch("rich.prompt.Confirm.ask", return_value=True):
                edited_agenda = hitl_manager._edit_agenda(original_agenda)

        assert len(edited_agenda) == 3
        assert edited_agenda[0] == "Modified Topic A"
        assert edited_agenda[2] == "New Topic C"

    def test_topic_continuation_handler(self, hitl_manager):
        """Test topic continuation handler."""
        context = HITLContext(
            completed_topics=["Topic 1"], remaining_topics=["Topic 2", "Topic 3"]
        )

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.TOPIC_CONTINUATION,
            prompt_message="Continue?",
            context=context.to_dict(),
        )

        # Mock user continuing
        with patch("rich.prompt.Prompt.ask", return_value="c"):
            response = hitl_manager._handle_topic_continuation(interaction)

        assert response.action == "continue"
        assert response.approved is True

    def test_session_continuation_handler(self, hitl_manager):
        """Test session continuation handler."""
        context = HITLContext(agent_vote_result={"recommendation": "end"})

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.SESSION_CONTINUATION,
            prompt_message="Continue session?",
            context=context.to_dict(),
        )

        # Mock user overriding agent recommendation
        with patch("rich.prompt.Confirm.ask", return_value=True):
            response = hitl_manager._handle_session_continuation(interaction)

        assert response.action == "continue_session"
        assert response.approved is True

    def test_interrupt_handling(self, hitl_manager):
        """Test interrupt handling during interaction."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Test",
            context={},
        )

        # Mock interrupt and resume
        with patch("rich.prompt.Prompt.ask", side_effect=["r", "Approve"]):
            response = hitl_manager._handle_interrupt(interaction)

        assert response.action == "approved"

    def test_interaction_tracking(self, hitl_manager):
        """Test that interactions are properly tracked."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.THEME_INPUT,
            prompt_message="Enter theme",
            context={},
        )

        response = HITLResponse(
            approved=True, action="theme_provided", modified_data="AI Safety"
        )

        # Start and complete interaction
        hitl_manager.state_tracker.start_interaction(interaction)
        hitl_manager.state_tracker.complete_interaction(response)

        # Check tracking
        assert len(hitl_manager.state_tracker.interactions) == 1
        assert len(hitl_manager.state_tracker.approval_history) == 1
        assert hitl_manager.state_tracker.approval_history[0]["type"] == "theme_input"

    def test_process_interaction_error_handling(self, hitl_manager):
        """Test error handling in process_interaction."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Test",
            context={},
        )

        # Mock handler to raise exception
        def error_handler(interaction):
            raise ValueError("Test error")

        hitl_manager.handlers[HITLApprovalType.AGENDA_APPROVAL] = error_handler

        response = hitl_manager.process_interaction(interaction)

        assert response.action == "error"
        assert response.reason == "Test error"
        assert response.approved is False

    def test_stats_tracking(self, hitl_manager):
        """Test statistics tracking."""
        # Process several interactions
        for i in range(3):
            interaction = HITLInteraction(
                approval_type=HITLApprovalType.PERIODIC_STOP,
                prompt_message=f"Stop {i}",
                context={"current_round": i * 5},
            )

            with patch("rich.prompt.Prompt.ask", return_value="c"):
                hitl_manager.process_interaction(interaction)

        stats = hitl_manager.get_stats()

        assert stats["total_interactions"] == 3
        assert stats["interactions_by_type"]["periodic_stop"] == 3
        assert stats["periodic_stops"] == 3

    def test_theme_input_handler(self, hitl_manager):
        """Test theme input handler."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.THEME_INPUT,
            prompt_message="Enter discussion theme",
            context={},
        )

        # Mock user input
        with patch("rich.prompt.Prompt.ask", return_value="AI Ethics and Society"):
            response = hitl_manager._handle_theme_input(interaction)

        assert response.action == "theme_provided"
        assert response.approved is True
        assert response.modified_data == "AI Ethics and Society"

    def test_agent_poll_override_handler(self, hitl_manager):
        """Test agent poll override handler."""
        context = HITLContext(
            agent_vote_result={"yes_votes": 2, "no_votes": 3, "result": "end"}
        )

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENT_POLL_OVERRIDE,
            prompt_message="Agents recommend ending. Override?",
            context=context.to_dict(),
        )

        # Mock user overriding
        with patch("rich.prompt.Confirm.ask", return_value=True):
            with patch("rich.prompt.Prompt.ask", return_value="continue"):
                response = hitl_manager._handle_agent_poll_override(interaction)

        assert response.action == "override"
        assert response.approved is True
        assert response.modified_data["new_result"] == "continue"

    def test_final_report_approval_handler(self, hitl_manager):
        """Test final report approval handler."""
        context = HITLContext(
            session_stats={
                "total_messages": 150,
                "topics_completed": 5,
                "session_duration": "2h 15m",
            }
        )

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.FINAL_REPORT_APPROVAL,
            prompt_message="Generate final report?",
            context=context.to_dict(),
        )

        # Mock user approval
        with patch("rich.prompt.Confirm.ask", return_value=True):
            response = hitl_manager._handle_final_report_approval(interaction)

        assert response.action == "generate_report"
        assert response.approved is True

    def test_topic_override_handler(self, hitl_manager):
        """Test topic override handler."""
        context = HITLContext(active_topic="Current Topic", current_round=8)

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.TOPIC_OVERRIDE,
            prompt_message="Force topic end?",
            context=context.to_dict(),
        )

        # Mock user forcing end with reason
        with patch("rich.prompt.Confirm.ask", return_value=True):
            with patch("rich.prompt.Prompt.ask", return_value="Topic is repetitive"):
                response = hitl_manager._handle_topic_override(interaction)

        assert response.action == "force_conclusion"
        assert response.approved is True
        assert response.reason == "Topic is repetitive"

    def test_agenda_reordering(self, hitl_manager):
        """Test agenda reordering functionality."""
        original_agenda = ["Topic A", "Topic B", "Topic C"]

        # Mock Prompt.ask for reordering
        with patch(
            "rich.prompt.Prompt.ask",
            side_effect=[
                "2",  # Move Topic B from position 2
                "1",  # To position 1
                "3",  # Move Topic C from position 3
                "2",  # To position 2
                "done",  # Done reordering
            ],
        ):
            with patch("rich.prompt.Confirm.ask", return_value=True):
                reordered = hitl_manager._reorder_agenda(original_agenda)

        assert reordered == ["Topic B", "Topic C", "Topic A"]

    def test_interaction_timing(self, hitl_manager):
        """Test interaction timing tracking."""
        import time

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.THEME_INPUT,
            prompt_message="Test",
            context={},
        )

        # Manually test timing by starting and completing interaction
        hitl_manager.state_tracker.start_interaction(interaction)

        # Simulate some processing time
        time.sleep(0.05)

        # Complete the interaction
        response = HITLResponse(
            approved=True, action="theme_provided", modified_data="Test Theme"
        )
        hitl_manager.state_tracker.complete_interaction(response)

        # Check that interaction was tracked with timing
        assert len(hitl_manager.state_tracker.interactions) == 1
        completed_interaction = hitl_manager.state_tracker.interactions[-1]
        assert completed_interaction.duration_seconds >= 0.05
        assert completed_interaction.response_time > completed_interaction.timestamp

    def test_multiple_handler_types(self, hitl_manager):
        """Test processing different handler types in sequence."""
        handler_types = [
            (HITLApprovalType.THEME_INPUT, "rich.prompt.Prompt.ask", "Test Theme"),
            (HITLApprovalType.AGENDA_APPROVAL, "rich.prompt.Prompt.ask", "Approve"),
            (HITLApprovalType.PERIODIC_STOP, "rich.prompt.Prompt.ask", "c"),
            (HITLApprovalType.SESSION_CONTINUATION, "rich.prompt.Confirm.ask", True),
        ]

        for approval_type, mock_target, mock_value in handler_types:
            interaction = HITLInteraction(
                approval_type=approval_type,
                prompt_message=f"Test {approval_type.value}",
                context={"test": True},
            )

            with patch(mock_target, return_value=mock_value):
                response = hitl_manager.process_interaction(interaction)
                assert response.approved is True

        # Verify all interactions were tracked
        stats = hitl_manager.get_stats()
        assert stats["total_interactions"] == len(handler_types)
        assert len(stats["interactions_by_type"]) == len(handler_types)

    def test_context_preservation(self, hitl_manager):
        """Test that context is preserved through interactions."""
        test_context = {
            "current_round": 5,
            "active_topic": "Test Topic",
            "custom_data": {"key": "value"},
            "nested": {"deep": {"data": 123}},
        }

        interaction = HITLInteraction(
            approval_type=HITLApprovalType.PERIODIC_STOP,
            prompt_message="Test",
            context=test_context,
        )

        with patch("rich.prompt.Prompt.ask", return_value="c"):
            response = hitl_manager.process_interaction(interaction)

        # Check that interaction was tracked with context
        assert len(hitl_manager.state_tracker.interactions) == 1
        tracked_interaction = hitl_manager.state_tracker.interactions[0]
        assert tracked_interaction.context == test_context
        assert tracked_interaction.context["nested"]["deep"]["data"] == 123

    def test_error_recovery(self, hitl_manager):
        """Test error recovery in handlers."""
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Test",
            context={},  # Missing proposed_agenda will be handled
        )

        # Mock the approval to handle empty agenda
        with patch("rich.prompt.Prompt.ask", return_value="Approve"):
            response = hitl_manager.process_interaction(interaction)

        # Should handle empty agenda gracefully
        assert response.approved is True
        assert response.action == "approved"
        assert response.modified_data == []  # Empty agenda

    def test_handler_registration(self, hitl_manager):
        """Test that all handlers are properly registered."""
        expected_handlers = [
            HITLApprovalType.THEME_INPUT,
            HITLApprovalType.AGENDA_APPROVAL,
            HITLApprovalType.PERIODIC_STOP,
            HITLApprovalType.TOPIC_OVERRIDE,
            HITLApprovalType.TOPIC_CONTINUATION,
            HITLApprovalType.AGENT_POLL_OVERRIDE,
            HITLApprovalType.SESSION_CONTINUATION,
            HITLApprovalType.FINAL_REPORT_APPROVAL,
        ]

        for handler_type in expected_handlers:
            assert handler_type in hitl_manager.handlers
            assert callable(hitl_manager.handlers[handler_type])
