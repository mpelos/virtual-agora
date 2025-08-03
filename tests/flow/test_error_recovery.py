"""Tests for the error recovery system.

This module tests the comprehensive error recovery strategies that
provide resilience for flow node execution failures.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from virtual_agora.flow.error_recovery import (
    ErrorRecoveryManager,
    ErrorRecoveryResult,
    ErrorSeverity,
    RecoveryStrategy,
    create_error_recovery_manager,
)
from virtual_agora.state.schema import VirtualAgoraState


@pytest.fixture
def recovery_manager():
    """Create ErrorRecoveryManager instance for testing."""
    return ErrorRecoveryManager()


@pytest.fixture
def sample_state():
    """Create sample VirtualAgoraState for testing."""
    return VirtualAgoraState(
        session_id="test_session",
        theme="Test Theme",
        active_topic="Test Topic",
        current_round=1,
        topic_queue=["Topic 1", "Topic 2"],
        messages=[],
    )


class TestErrorRecoveryManager:
    """Test the ErrorRecoveryManager class."""

    def test_initialization(self, recovery_manager):
        """Test ErrorRecoveryManager initialization."""
        assert len(recovery_manager.recovery_strategies) == 0
        assert len(recovery_manager.generic_strategies) > 0  # Has default strategies
        assert len(recovery_manager.recovery_history) == 0
        assert recovery_manager.recovery_metrics["total_recoveries"] == 0

        # Check that default strategies are initialized
        assert TimeoutError in recovery_manager.generic_strategies
        assert ConnectionError in recovery_manager.generic_strategies
        assert ValueError in recovery_manager.generic_strategies
        assert KeyError in recovery_manager.generic_strategies

    def test_register_recovery_strategy(self, recovery_manager):
        """Test registering custom recovery strategy."""

        def custom_strategy(node_name, error, state):
            return {"custom_recovery": True}

        recovery_manager.register_recovery_strategy("test_node", custom_strategy)

        assert "test_node" in recovery_manager.recovery_strategies
        assert recovery_manager.recovery_strategies["test_node"] == custom_strategy

    def test_register_recovery_strategy_invalid_callable(self, recovery_manager):
        """Test registering invalid recovery strategy."""
        with pytest.raises(TypeError) as exc_info:
            recovery_manager.register_recovery_strategy("test_node", "not_callable")

        assert "must be callable" in str(exc_info.value)

    def test_register_generic_strategy(self, recovery_manager):
        """Test registering generic recovery strategy."""

        def timeout_strategy(node_name, error, state):
            return {"timeout_recovery": True}

        recovery_manager.register_generic_strategy(TimeoutError, timeout_strategy)

        assert TimeoutError in recovery_manager.generic_strategies
        assert recovery_manager.generic_strategies[TimeoutError] == timeout_strategy

    def test_register_generic_strategy_invalid_callable(self, recovery_manager):
        """Test registering invalid generic strategy."""
        with pytest.raises(TypeError) as exc_info:
            recovery_manager.register_generic_strategy(ValueError, "not_callable")

        assert "must be callable" in str(exc_info.value)

    def test_successful_recovery_with_custom_strategy(
        self, recovery_manager, sample_state
    ):
        """Test successful error recovery with custom strategy."""

        def custom_strategy(node_name, error, state):
            return {"recovered": True, "error_message": str(error)}

        recovery_manager.register_recovery_strategy("test_node", custom_strategy)

        result = recovery_manager.attempt_recovery(
            "test_node", Exception("Test error"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["recovered"] is True
        assert "Test error" in result.recovery_updates["error_message"]
        assert result.recovery_strategy == RecoveryStrategy.FALLBACK
        assert result.node_name == "test_node"
        assert recovery_manager.recovery_metrics["successful_recoveries"] == 1

    def test_successful_recovery_with_generic_strategy(
        self, recovery_manager, sample_state
    ):
        """Test successful error recovery with generic strategy."""

        def value_error_strategy(node_name, error, state):
            return {"value_error_recovery": True}

        recovery_manager.register_generic_strategy(ValueError, value_error_strategy)

        result = recovery_manager.attempt_recovery(
            "test_node", ValueError("Test value error"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["value_error_recovery"] is True
        assert result.recovery_strategy == RecoveryStrategy.FALLBACK

    def test_successful_recovery_with_inheritance(self, recovery_manager, sample_state):
        """Test recovery with error inheritance (parent class strategy)."""

        def exception_strategy(node_name, error, state):
            return {"exception_recovery": True}

        recovery_manager.register_generic_strategy(Exception, exception_strategy)

        # Use a custom error type to test inheritance (ValueError already has a default strategy)
        class CustomError(Exception):
            pass

        result = recovery_manager.attempt_recovery(
            "test_node", CustomError("Test custom error"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["exception_recovery"] is True

    def test_default_recovery_for_user_interaction_node(
        self, recovery_manager, sample_state
    ):
        """Test default recovery for user interaction nodes."""
        result = recovery_manager.attempt_recovery(
            "user_approval", Exception("User interaction failed"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["user_interaction_error"] is True
        assert result.recovery_updates["use_default_action"] is True
        assert result.recovery_strategy == RecoveryStrategy.CONTINUE

    def test_default_recovery_for_discussion_node(self, recovery_manager, sample_state):
        """Test default recovery for discussion nodes."""
        result = recovery_manager.attempt_recovery(
            "discussion_round", Exception("Discussion failed"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["discussion_round_error"] is True
        assert result.recovery_updates["skip_to_next_round"] is True
        assert result.recovery_strategy == RecoveryStrategy.SKIP

    def test_default_recovery_for_report_node(self, recovery_manager, sample_state):
        """Test default recovery for report generation nodes."""
        result = recovery_manager.attempt_recovery(
            "report_generation", Exception("Report failed"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["report_generation_error"] is True
        assert result.recovery_updates["use_simplified_report"] is True
        assert result.recovery_strategy == RecoveryStrategy.FALLBACK

    def test_default_recovery_for_generic_node(self, recovery_manager, sample_state):
        """Test default recovery for generic nodes."""
        result = recovery_manager.attempt_recovery(
            "some_other_node", Exception("Generic failure"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["error_occurred"] is True
        assert result.recovery_updates["continue_flow"] is True
        assert result.recovery_strategy == RecoveryStrategy.CONTINUE

    def test_failed_recovery_attempt(self, recovery_manager, sample_state):
        """Test failed recovery attempt."""

        def failing_strategy(node_name, error, state):
            raise Exception("Recovery also failed")

        recovery_manager.register_recovery_strategy("test_node", failing_strategy)

        result = recovery_manager.attempt_recovery(
            "test_node", Exception("Original error"), sample_state
        )

        assert result.success is False
        assert result.recovery_strategy == RecoveryStrategy.ABORT
        assert "Recovery also failed" in result.additional_info["recovery_error"]
        assert recovery_manager.recovery_metrics["failed_recoveries"] == 1

    def test_assess_error_severity_critical_nodes(self, recovery_manager, sample_state):
        """Test error severity assessment for critical nodes."""
        severity = recovery_manager._assess_error_severity(
            "config_and_keys", Exception("test"), sample_state
        )
        assert severity == ErrorSeverity.HIGH

        severity = recovery_manager._assess_error_severity(
            "discussion_round", Exception("test"), sample_state
        )
        assert severity == ErrorSeverity.HIGH

    def test_assess_error_severity_user_interaction_nodes(
        self, recovery_manager, sample_state
    ):
        """Test error severity assessment for user interaction nodes."""
        severity = recovery_manager._assess_error_severity(
            "user_approval", Exception("test"), sample_state
        )
        assert severity == ErrorSeverity.LOW

        severity = recovery_manager._assess_error_severity(
            "periodic_user_stop", Exception("test"), sample_state
        )
        assert severity == ErrorSeverity.LOW

    def test_assess_error_severity_critical_errors(
        self, recovery_manager, sample_state
    ):
        """Test error severity assessment for critical error types."""
        severity = recovery_manager._assess_error_severity(
            "test_node", SystemError("test"), sample_state
        )
        assert severity == ErrorSeverity.CRITICAL

        severity = recovery_manager._assess_error_severity(
            "test_node", MemoryError("test"), sample_state
        )
        assert severity == ErrorSeverity.CRITICAL

    def test_assess_error_severity_timeout_connection_errors(
        self, recovery_manager, sample_state
    ):
        """Test error severity assessment for timeout and connection errors."""
        severity = recovery_manager._assess_error_severity(
            "test_node", Exception("timeout occurred"), sample_state
        )
        assert severity == ErrorSeverity.MEDIUM

        severity = recovery_manager._assess_error_severity(
            "test_node", Exception("connection failed"), sample_state
        )
        assert severity == ErrorSeverity.MEDIUM

    def test_get_recovery_metrics(self, recovery_manager, sample_state):
        """Test getting recovery metrics."""
        # Generate some recovery activity
        recovery_manager.attempt_recovery(
            "test_node", Exception("error1"), sample_state
        )
        recovery_manager.attempt_recovery(
            "user_approval", Exception("error2"), sample_state
        )

        metrics = recovery_manager.get_recovery_metrics()

        assert metrics["total_recoveries"] == 2
        assert metrics["successful_recoveries"] == 2
        assert metrics["failed_recoveries"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["failure_rate"] == 0.0
        assert len(metrics["recent_recoveries"]) == 2

        # Check recent recoveries format
        recent = metrics["recent_recoveries"][0]
        assert "node_name" in recent
        assert "success" in recent
        assert "strategy" in recent
        assert "severity" in recent
        assert "recovery_time" in recent
        assert "timestamp" in recent

    def test_get_recovery_history(self, recovery_manager, sample_state):
        """Test getting recovery history."""
        # Generate recovery activity
        recovery_manager.attempt_recovery("node1", Exception("error1"), sample_state)
        recovery_manager.attempt_recovery("node2", Exception("error2"), sample_state)
        recovery_manager.attempt_recovery("node1", Exception("error3"), sample_state)

        # Get all history
        all_history = recovery_manager.get_recovery_history()
        assert len(all_history) == 3

        # Get filtered history
        node1_history = recovery_manager.get_recovery_history("node1")
        assert len(node1_history) == 2
        assert all(result.node_name == "node1" for result in node1_history)

    def test_clear_recovery_history(self, recovery_manager, sample_state):
        """Test clearing recovery history."""
        # Generate some history
        recovery_manager.attempt_recovery("test_node", Exception("error"), sample_state)
        assert len(recovery_manager.recovery_history) == 1

        recovery_manager.clear_recovery_history()
        assert len(recovery_manager.recovery_history) == 0

    def test_recovery_history_maintenance(self, recovery_manager, sample_state):
        """Test that recovery history is properly maintained."""
        # Generate more than 100 recovery attempts to test limit
        for i in range(105):
            try:
                recovery_manager.attempt_recovery(
                    f"node_{i}", Exception(f"error_{i}"), sample_state
                )
            except:
                pass  # Some might fail, that's ok for this test

        # History should be limited to 100 entries
        assert len(recovery_manager.recovery_history) <= 100

    def test_default_strategies_initialization(self, recovery_manager):
        """Test that default recovery strategies are properly initialized."""
        # Check that generic strategies are registered
        assert TimeoutError in recovery_manager.generic_strategies
        assert ConnectionError in recovery_manager.generic_strategies
        assert ValueError in recovery_manager.generic_strategies
        assert KeyError in recovery_manager.generic_strategies

    def test_timeout_recovery_strategy(self, recovery_manager, sample_state):
        """Test timeout error recovery strategy."""
        result = recovery_manager.attempt_recovery(
            "test_node", TimeoutError("Operation timed out"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["timeout_error"] is True
        assert result.recovery_updates["retry_with_longer_timeout"] is True

    def test_connection_recovery_strategy(self, recovery_manager, sample_state):
        """Test connection error recovery strategy."""
        result = recovery_manager.attempt_recovery(
            "test_node", ConnectionError("Connection failed"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["connection_error"] is True
        assert result.recovery_updates["retry_connection"] is True

    def test_value_error_recovery_strategy(self, recovery_manager, sample_state):
        """Test value error recovery strategy."""
        result = recovery_manager.attempt_recovery(
            "test_node", ValueError("Invalid value"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["value_error"] is True
        assert result.recovery_updates["use_default_values"] is True

    def test_key_error_recovery_strategy(self, recovery_manager, sample_state):
        """Test key error recovery strategy."""
        result = recovery_manager.attempt_recovery(
            "test_node", KeyError("Missing key"), sample_state
        )

        assert result.success is True
        assert result.recovery_updates["key_error"] is True
        assert result.recovery_updates["provide_default_key"] is True

    def test_recovery_metrics_tracking(self, recovery_manager, sample_state):
        """Test that recovery metrics are properly tracked."""
        # Successful recovery
        recovery_manager.attempt_recovery("node1", Exception("error1"), sample_state)

        # Failed recovery
        def failing_strategy(node_name, error, state):
            raise Exception("Recovery failed")

        recovery_manager.register_recovery_strategy("node2", failing_strategy)

        try:
            recovery_manager.attempt_recovery(
                "node2", Exception("error2"), sample_state
            )
        except:
            pass  # Expected to fail

        metrics = recovery_manager.recovery_metrics

        assert metrics["total_recoveries"] == 2
        assert metrics["successful_recoveries"] == 1
        assert metrics["failed_recoveries"] == 1
        # The following metrics are tracked in the orchestrator, not error recovery manager
        # assert metrics["recovery_attempts"] == 2
        # assert metrics["recovery_successes"] == 1


class TestErrorRecoveryResult:
    """Test the ErrorRecoveryResult class."""

    def test_initialization(self):
        """Test ErrorRecoveryResult initialization."""
        error = Exception("test error")
        result = ErrorRecoveryResult(
            node_name="test_node",
            original_error=error,
            recovery_strategy=RecoveryStrategy.CONTINUE,
            success=True,
            recovery_updates={"recovered": True},
            recovery_time=0.5,
            error_severity=ErrorSeverity.MEDIUM,
            additional_info={"test": "info"},
        )

        assert result.node_name == "test_node"
        assert result.original_error == error
        assert result.recovery_strategy == RecoveryStrategy.CONTINUE
        assert result.success is True
        assert result.recovery_updates == {"recovered": True}
        assert result.recovery_time == 0.5
        assert result.error_severity == ErrorSeverity.MEDIUM
        assert result.additional_info == {"test": "info"}
        assert result.timestamp is not None
        assert result.error_trace is not None


class TestFactoryFunction:
    """Test the factory function for creating error recovery manager."""

    def test_create_error_recovery_manager(self):
        """Test error recovery manager factory function."""
        manager = create_error_recovery_manager()

        assert isinstance(manager, ErrorRecoveryManager)
        assert len(manager.generic_strategies) > 0  # Should have default strategies


class TestComplexRecoveryScenarios:
    """Test complex error recovery scenarios."""

    def test_cascading_recovery_strategies(self, recovery_manager, sample_state):
        """Test recovery strategy selection with cascading priorities."""

        # Register both node-specific and generic strategies
        def node_specific_strategy(node_name, error, state):
            return {"node_specific": True}

        def generic_strategy(node_name, error, state):
            return {"generic": True}

        recovery_manager.register_recovery_strategy("test_node", node_specific_strategy)
        recovery_manager.register_generic_strategy(ValueError, generic_strategy)

        # Node-specific strategy should take precedence
        result = recovery_manager.attempt_recovery(
            "test_node", ValueError("test error"), sample_state
        )

        assert result.recovery_updates["node_specific"] is True
        assert "generic" not in result.recovery_updates

    def test_error_inheritance_strategy_selection(self, recovery_manager, sample_state):
        """Test that error inheritance works for strategy selection."""

        def base_exception_strategy(node_name, error, state):
            return {"base_exception": True}

        def value_error_strategy(node_name, error, state):
            return {"value_error": True}

        recovery_manager.register_generic_strategy(Exception, base_exception_strategy)
        recovery_manager.register_generic_strategy(ValueError, value_error_strategy)

        # More specific strategy should be used
        result = recovery_manager.attempt_recovery(
            "test_node", ValueError("test error"), sample_state
        )

        assert result.recovery_updates["value_error"] is True
        assert "base_exception" not in result.recovery_updates


if __name__ == "__main__":
    pytest.main([__file__])
