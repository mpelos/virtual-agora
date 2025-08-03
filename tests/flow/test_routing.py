"""Tests for the flow routing system.

This module tests the centralized routing logic that was extracted
from the graph definition for better separation of concerns.
"""

import pytest
from unittest.mock import Mock, patch

from virtual_agora.flow.routing import (
    FlowRouter,
    ConditionalRouter,
    create_flow_router,
    create_conditional_router,
)
from virtual_agora.flow.edges_v13 import V13FlowConditions
from virtual_agora.state.schema import VirtualAgoraState


@pytest.fixture
def mock_conditions():
    """Create mock V13FlowConditions."""
    conditions = Mock(spec=V13FlowConditions)
    conditions.evaluate_session_continuation.return_value = "continue"
    return conditions


@pytest.fixture
def flow_router(mock_conditions):
    """Create FlowRouter instance for testing."""
    return FlowRouter(mock_conditions)


@pytest.fixture
def conditional_router(mock_conditions):
    """Create ConditionalRouter instance for testing."""
    return ConditionalRouter(mock_conditions)


@pytest.fixture
def sample_state():
    """Create sample VirtualAgoraState for testing."""
    return VirtualAgoraState(
        session_id="test_session",
        theme="Test Theme",
        active_topic="Test Topic",
        current_round=1,
        topic_queue=["Topic 1", "Topic 2"],
        completed_topics=["Completed Topic"],
        user_approves_continuation=True,
        agents_vote_end_session=False,
        user_forced_conclusion=False,
        user_requested_modification=False,
        messages=[],
    )


class TestFlowRouter:
    """Test the FlowRouter class."""

    def test_initialization(self, flow_router, mock_conditions):
        """Test FlowRouter initialization."""
        assert flow_router.conditions == mock_conditions
        assert flow_router.routing_metrics["total_routing_decisions"] == 0
        assert flow_router.routing_metrics["user_approval_routes"] == 0
        assert flow_router.routing_metrics["corruption_cleanups"] == 0

    def test_user_approval_routing_session_end(self, flow_router, sample_state):
        """Test user approval routing when session should end."""
        flow_router.conditions.evaluate_session_continuation.return_value = (
            "end_session"
        )

        result = flow_router.evaluate_user_approval_routing(sample_state)

        assert result == "end_session"
        assert flow_router.routing_metrics["user_approval_routes"] == 1

    def test_user_approval_routing_modify_agenda(self, flow_router, sample_state):
        """Test user approval routing when user requests agenda modification."""
        sample_state["user_requested_modification"] = True

        result = flow_router.evaluate_user_approval_routing(sample_state)

        assert result == "modify_agenda"

    def test_user_approval_routing_no_items(self, flow_router, sample_state):
        """Test user approval routing when no topics remain."""
        sample_state["topic_queue"] = []

        result = flow_router.evaluate_user_approval_routing(sample_state)

        assert result == "no_items"

    def test_user_approval_routing_has_items(self, flow_router, sample_state):
        """Test user approval routing when topics remain."""
        sample_state["topic_queue"] = ["Topic 1", "Topic 2", "Topic 3"]

        with patch(
            "virtual_agora.flow.routing.get_interrupt_manager"
        ) as mock_get_interrupt:
            mock_interrupt_manager = Mock()
            mock_get_interrupt.return_value = mock_interrupt_manager

            result = flow_router.evaluate_user_approval_routing(sample_state)

        assert result == "has_items"

        # Verify interrupt manager was called for topic preparation
        mock_interrupt_manager.reset_for_new_topic.assert_called_once_with("Topic 1")

    def test_topic_queue_corruption_handling_nested_lists(self, flow_router):
        """Test handling of nested list corruption in topic queue."""
        corrupted_queue = [
            "Valid Topic",
            ["Nested Topic 1", "Nested Topic 2"],
            "Another Valid Topic",
            [["Deeply Nested"]],
        ]

        cleaned_queue, corruption_detected = flow_router.handle_topic_queue_corruption(
            corrupted_queue
        )

        assert corruption_detected is True
        assert "Valid Topic" in cleaned_queue
        assert "Nested Topic 1" in cleaned_queue
        assert "Nested Topic 2" in cleaned_queue
        assert "Another Valid Topic" in cleaned_queue
        assert len(cleaned_queue) == 4  # All valid topics extracted

    def test_topic_queue_corruption_handling_invalid_items(self, flow_router):
        """Test handling of invalid items in topic queue."""
        corrupted_queue = [
            "Valid Topic",
            None,
            123,
            "",
            "   ",  # Whitespace only
            "Another Valid Topic",
        ]

        cleaned_queue, corruption_detected = flow_router.handle_topic_queue_corruption(
            corrupted_queue
        )

        assert corruption_detected is True
        assert cleaned_queue == ["Valid Topic", "Another Valid Topic"]

    def test_topic_queue_corruption_handling_clean_queue(self, flow_router):
        """Test handling of already clean topic queue."""
        clean_queue = ["Topic 1", "Topic 2", "Topic 3"]

        cleaned_queue, corruption_detected = flow_router.handle_topic_queue_corruption(
            clean_queue
        )

        assert corruption_detected is False
        assert cleaned_queue == clean_queue

    def test_determine_session_continuation(self, flow_router, sample_state):
        """Test session continuation determination."""
        flow_router.conditions.evaluate_session_continuation.return_value = "continue"

        result = flow_router.determine_session_continuation(sample_state)

        assert result == "continue"
        assert flow_router.routing_metrics["session_continuations"] == 1
        flow_router.conditions.evaluate_session_continuation.assert_called_once_with(
            sample_state
        )

    def test_route_error_recovery_user_interaction(self, flow_router, sample_state):
        """Test error recovery routing for user interaction errors."""
        result = flow_router.route_error_recovery(
            "user_interaction_node", Exception("Test error"), sample_state
        )

        assert result == "continue_with_defaults"
        assert flow_router.routing_metrics["error_recoveries"] == 1

    def test_route_error_recovery_discussion(self, flow_router, sample_state):
        """Test error recovery routing for discussion errors."""
        result = flow_router.route_error_recovery(
            "discussion_round", Exception("Test error"), sample_state
        )

        assert result == "skip_to_summary"

    def test_route_error_recovery_report(self, flow_router, sample_state):
        """Test error recovery routing for report errors."""
        result = flow_router.route_error_recovery(
            "report_generation_node", Exception("Test error"), sample_state
        )

        assert result == "alternative_output"

    def test_route_error_recovery_generic(self, flow_router, sample_state):
        """Test error recovery routing for generic errors."""
        result = flow_router.route_error_recovery(
            "some_other_node", Exception("Test error"), sample_state
        )

        assert result == "continue_with_error"

    def test_get_routing_metrics(self, flow_router, sample_state):
        """Test getting routing metrics."""
        # Generate some routing activity
        flow_router.evaluate_user_approval_routing(
            sample_state
        )  # This internally calls handle_topic_queue_corruption
        flow_router.determine_session_continuation(sample_state)
        flow_router.handle_topic_queue_corruption(
            ["Topic 1"]
        )  # This is called again directly
        flow_router.route_error_recovery("test_node", Exception(), sample_state)

        metrics = flow_router.get_routing_metrics()

        assert metrics["total_routing_decisions"] == 1  # Only user approval routing
        assert metrics["user_approval_routes"] == 1
        assert metrics["session_continuations"] == 1
        assert (
            metrics["corruption_cleanups"] == 2
        )  # Called twice: once internally, once directly
        assert metrics["error_recoveries"] == 1

    def test_handle_topic_completion_cleanup_with_completed_topics(
        self, flow_router, sample_state
    ):
        """Test topic completion cleanup when topics are completed."""
        sample_state["completed_topics"] = ["Topic A", "Topic B"]

        with patch(
            "virtual_agora.flow.routing.get_interrupt_manager"
        ) as mock_get_interrupt:
            mock_interrupt_manager = Mock()
            mock_get_interrupt.return_value = mock_interrupt_manager

            flow_router._handle_topic_completion_cleanup(sample_state)

            # Should cleanup the last completed topic
            mock_interrupt_manager.clear_topic_context.assert_called_once_with(
                "Topic B"
            )

    def test_handle_topic_completion_cleanup_with_no_completed_topics(
        self, flow_router, sample_state
    ):
        """Test topic completion cleanup when no topics are completed."""
        sample_state["completed_topics"] = []

        with patch(
            "virtual_agora.flow.routing.get_interrupt_manager"
        ) as mock_get_interrupt:
            mock_interrupt_manager = Mock()
            mock_get_interrupt.return_value = mock_interrupt_manager

            flow_router._handle_topic_completion_cleanup(sample_state)

            # Should not call cleanup when no completed topics
            mock_interrupt_manager.clear_topic_context.assert_not_called()

    def test_prepare_next_topic_context(self, flow_router):
        """Test preparing context for next topic."""
        cleaned_queue = ["Next Topic", "Future Topic"]

        with patch(
            "virtual_agora.flow.routing.get_interrupt_manager"
        ) as mock_get_interrupt:
            mock_interrupt_manager = Mock()
            mock_get_interrupt.return_value = mock_interrupt_manager

            flow_router._prepare_next_topic_context(cleaned_queue)

            # Should setup context for the first topic in queue
            mock_interrupt_manager.reset_for_new_topic.assert_called_once_with(
                "Next Topic"
            )

    def test_truncate_topic_methods(self, flow_router):
        """Test topic truncation utility methods."""
        # Test single topic truncation
        short_topic = "Short"
        long_topic = (
            "This is a very long topic title that should be truncated for display"
        )

        assert flow_router._truncate_topic(short_topic) == "Short"
        truncated = flow_router._truncate_topic(long_topic)
        assert len(truncated) <= 63  # 60 + "..."
        assert truncated.endswith("...")

        # Test topic list truncation
        topics = [short_topic, long_topic, "Medium length topic"]
        truncated_list = flow_router._truncate_topic_list(topics)

        assert len(truncated_list) == 3
        assert truncated_list[0] == "Short"
        assert truncated_list[1].endswith("...")


class TestConditionalRouter:
    """Test the ConditionalRouter class."""

    def test_initialization(self, conditional_router, mock_conditions):
        """Test ConditionalRouter initialization."""
        assert conditional_router.conditions == mock_conditions
        assert conditional_router.condition_metrics["total_evaluations"] == 0
        assert conditional_router.condition_metrics["condition_failures"] == 0

    def test_evaluate_condition_success(self, conditional_router, sample_state):
        """Test successful condition evaluation."""
        # Mock a condition method
        conditional_router.conditions.test_condition = Mock(return_value="test_result")

        result = conditional_router.evaluate_condition("test_condition", sample_state)

        assert result == "test_result"
        assert conditional_router.condition_metrics["total_evaluations"] == 1
        assert conditional_router.condition_metrics["condition_failures"] == 0
        conditional_router.conditions.test_condition.assert_called_once_with(
            sample_state
        )

    def test_evaluate_condition_nonexistent(self, conditional_router, sample_state):
        """Test evaluation of non-existent condition."""
        with pytest.raises(AttributeError) as exc_info:
            conditional_router.evaluate_condition("nonexistent_condition", sample_state)

        assert "not found" in str(exc_info.value)
        assert conditional_router.condition_metrics["condition_failures"] == 1

    def test_evaluate_condition_with_error(self, conditional_router, sample_state):
        """Test condition evaluation with error."""
        # Mock a condition method that raises an error
        conditional_router.conditions.failing_condition = Mock(
            side_effect=Exception("Test error")
        )

        with pytest.raises(Exception) as exc_info:
            conditional_router.evaluate_condition("failing_condition", sample_state)

        assert "Test error" in str(exc_info.value)
        assert conditional_router.condition_metrics["condition_failures"] == 1

    def test_get_condition_metrics(self, conditional_router, sample_state):
        """Test getting condition metrics."""
        # Mock conditions and generate activity
        conditional_router.conditions.condition1 = Mock(return_value="result1")
        conditional_router.conditions.condition2 = Mock(return_value="result2")

        conditional_router.evaluate_condition("condition1", sample_state)
        conditional_router.evaluate_condition("condition2", sample_state)
        conditional_router.evaluate_condition("condition1", sample_state)  # Call again

        metrics = conditional_router.get_condition_metrics()

        assert metrics["total_evaluations"] == 3
        assert metrics["condition_failures"] == 0
        assert "average_evaluation_times" in metrics
        assert "condition1" in metrics["average_evaluation_times"]
        assert "condition2" in metrics["average_evaluation_times"]

    def test_condition_timing_tracking(self, conditional_router, sample_state):
        """Test that condition evaluation times are tracked."""
        conditional_router.conditions.timed_condition = Mock(return_value="result")

        conditional_router.evaluate_condition("timed_condition", sample_state)

        timing_data = conditional_router.condition_metrics["evaluation_times"]
        assert "timed_condition" in timing_data
        assert len(timing_data["timed_condition"]) == 1
        assert timing_data["timed_condition"][0] > 0  # Should have some execution time


class TestFactoryFunctions:
    """Test the factory functions for creating routers."""

    def test_create_flow_router(self, mock_conditions):
        """Test flow router factory function."""
        router = create_flow_router(mock_conditions)

        assert isinstance(router, FlowRouter)
        assert router.conditions == mock_conditions

    def test_create_conditional_router(self, mock_conditions):
        """Test conditional router factory function."""
        router = create_conditional_router(mock_conditions)

        assert isinstance(router, ConditionalRouter)
        assert router.conditions == mock_conditions


class TestRouterIntegration:
    """Test integration between different router components."""

    def test_flow_router_with_conditional_router(self, mock_conditions, sample_state):
        """Test that flow router properly delegates to conditions."""
        # Set up mock condition responses
        mock_conditions.evaluate_session_continuation.return_value = "continue"

        flow_router = FlowRouter(mock_conditions)

        # Test that flow router uses conditions correctly
        result = flow_router.determine_session_continuation(sample_state)

        assert result == "continue"
        mock_conditions.evaluate_session_continuation.assert_called_once_with(
            sample_state
        )

    def test_topic_queue_corruption_complex_scenario(self, flow_router):
        """Test complex topic queue corruption scenario."""
        complex_corrupted_queue = [
            "Valid Topic 1",
            ["Nested A", "Nested B", None, ""],
            None,
            "Valid Topic 2",
            [["Deeply Nested C"], "Nested D"],
            123,
            "   ",  # Whitespace only
            "Valid Topic 3",
        ]

        cleaned_queue, corruption_detected = flow_router.handle_topic_queue_corruption(
            complex_corrupted_queue
        )

        assert corruption_detected is True
        expected_clean = [
            "Valid Topic 1",
            "Nested A",
            "Nested B",
            "Valid Topic 2",
            "Nested D",
            "Valid Topic 3",
        ]
        assert cleaned_queue == expected_clean

        # Verify metrics are updated
        assert flow_router.routing_metrics["corruption_cleanups"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
