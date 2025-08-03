"""Tests for the discussion flow orchestrator.

This module tests the high-level orchestration functionality that separates
orchestration concerns from implementation details.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from virtual_agora.flow.orchestrator import (
    DiscussionFlowOrchestrator,
    ExecutionPhase,
    NodeExecutionResult,
)
from virtual_agora.flow.error_recovery import ErrorSeverity
from virtual_agora.flow.nodes.base import (
    FlowNode,
    NodeExecutionError,
    NodeValidationError,
)
from virtual_agora.flow.edges_v13 import V13FlowConditions
from virtual_agora.flow.state_manager import FlowStateManager
from virtual_agora.state.schema import VirtualAgoraState


class MockFlowNode(FlowNode):
    """Mock flow node for testing."""

    def __init__(
        self, name: str, should_fail: bool = False, validation_fail: bool = False
    ):
        super().__init__()
        self.name = name
        self.should_fail = should_fail
        self.validation_fail = validation_fail
        self.execution_count = 0

    def execute(self, state: VirtualAgoraState) -> dict:
        self.execution_count += 1
        if self.should_fail:
            raise NodeExecutionError(f"Mock failure in {self.name}")
        return {"executed_node": self.name, "execution_count": self.execution_count}

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        if self.validation_fail:
            self._validation_errors = [f"Mock validation failure in {self.name}"]
            return False
        return True

    def get_node_name(self) -> str:
        return f"MockNode({self.name})"


@pytest.fixture
def mock_conditions():
    """Create mock V13FlowConditions."""
    conditions = Mock(spec=V13FlowConditions)
    conditions.evaluate_session_continuation.return_value = "continue"
    return conditions


@pytest.fixture
def mock_flow_state_manager():
    """Create mock FlowStateManager."""
    flow_state_manager = Mock(spec=FlowStateManager)
    return flow_state_manager


@pytest.fixture
def sample_node_registry():
    """Create sample node registry for testing."""
    return {
        "test_node": MockFlowNode("test"),
        "failing_node": MockFlowNode("failing", should_fail=True),
        "validation_failing_node": MockFlowNode(
            "validation_failing", validation_fail=True
        ),
    }


@pytest.fixture
def orchestrator(mock_conditions, mock_flow_state_manager, sample_node_registry):
    """Create orchestrator instance for testing."""
    return DiscussionFlowOrchestrator(
        node_registry=sample_node_registry,
        conditions=mock_conditions,
        flow_state_manager=mock_flow_state_manager,
    )


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


class TestDiscussionFlowOrchestrator:
    """Test the DiscussionFlowOrchestrator class."""

    def test_initialization(self, orchestrator, sample_node_registry):
        """Test orchestrator initialization."""
        assert orchestrator.nodes == sample_node_registry
        assert orchestrator.conditions is not None
        assert orchestrator.flow_state_manager is not None
        assert len(orchestrator.execution_history) == 0
        assert orchestrator.execution_metrics["total_executions"] == 0

    def test_successful_node_execution(self, orchestrator, sample_state):
        """Test successful node execution with orchestration."""
        updates = orchestrator.execute_node_with_orchestration(
            "test_node", sample_state
        )

        assert updates["executed_node"] == "test"
        assert updates["execution_count"] == 1
        assert orchestrator.execution_metrics["total_executions"] == 1
        assert orchestrator.execution_metrics["successful_executions"] == 1
        assert len(orchestrator.execution_history) == 1

        result = orchestrator.execution_history[0]
        assert result.node_name == "test_node"
        assert result.success is True
        assert result.phase == ExecutionPhase.COMPLETED

    def test_node_execution_with_validation_failure(self, orchestrator, sample_state):
        """Test node execution with validation failure."""
        # The orchestrator should use error recovery instead of raising exceptions
        updates = orchestrator.execute_node_with_orchestration(
            "validation_failing_node", sample_state
        )

        # Check that error recovery was applied
        assert updates["error_occurred"] is True
        assert "validation failure" in updates["error_message"]
        assert updates["recovery_strategy"] == "default_continue"
        assert (
            orchestrator.execution_metrics["total_executions"] == 1
        )  # Counted as execution with recovery
        assert (
            orchestrator.execution_metrics["successful_executions"] == 1
        )  # Recovery was successful

        # Check execution history shows recovery
        result = orchestrator.execution_history[0]
        assert result.success is True  # Recovery was successful
        assert result.recovery_attempted is True

    def test_node_execution_with_error_and_recovery(self, orchestrator, sample_state):
        """Test node execution with error and successful recovery."""

        # Register a recovery strategy for the failing node
        def recovery_strategy(node_name, error, state):
            return {"recovered": True, "original_error": str(error)}

        orchestrator.register_error_recovery_strategy("failing_node", recovery_strategy)

        updates = orchestrator.execute_node_with_orchestration(
            "failing_node", sample_state
        )

        assert updates["recovered"] is True
        assert "Mock failure" in updates["original_error"]
        assert orchestrator.execution_metrics["total_executions"] == 1
        assert orchestrator.execution_metrics["successful_executions"] == 1
        assert orchestrator.execution_metrics["recovery_attempts"] == 1
        assert orchestrator.execution_metrics["recovery_successes"] == 1

        result = orchestrator.execution_history[0]
        assert result.success is True
        assert result.recovery_attempted is True
        assert result.phase == ExecutionPhase.ERROR_RECOVERY

    def test_node_execution_with_failed_recovery(self, orchestrator, sample_state):
        """Test node execution with error and failed recovery."""

        # Register a recovery strategy that also fails
        def failing_recovery_strategy(node_name, error, state):
            raise Exception("Recovery failed too")

        orchestrator.register_error_recovery_strategy(
            "failing_node", failing_recovery_strategy
        )

        with pytest.raises(NodeExecutionError) as exc_info:
            orchestrator.execute_node_with_orchestration("failing_node", sample_state)

        assert "execution failed and recovery unsuccessful" in str(exc_info.value)
        assert orchestrator.execution_metrics["total_executions"] == 1
        assert orchestrator.execution_metrics["failed_executions"] == 1
        assert orchestrator.execution_metrics["recovery_attempts"] == 1
        assert orchestrator.execution_metrics["recovery_successes"] == 0

        result = orchestrator.execution_history[0]
        assert result.success is False
        assert result.recovery_attempted is True

    def test_nonexistent_node_execution(self, orchestrator, sample_state):
        """Test execution of non-existent node."""
        # The orchestrator should use error recovery instead of raising exceptions
        updates = orchestrator.execute_node_with_orchestration(
            "nonexistent_node", sample_state
        )

        # Check that error recovery was applied
        assert updates["error_occurred"] is True
        assert "not found in registry" in updates["error_message"]
        assert updates["recovery_strategy"] == "default_continue"

    def test_register_error_recovery_strategy(self, orchestrator):
        """Test registering custom error recovery strategy."""

        def custom_strategy(node_name, error, state):
            return {"custom_recovery": True}

        orchestrator.register_error_recovery_strategy("test_node", custom_strategy)

        assert "test_node" in orchestrator.error_recovery_strategies
        assert orchestrator.error_recovery_strategies["test_node"] == custom_strategy

    def test_register_state_validator(self, orchestrator):
        """Test registering custom state validator."""

        def custom_validator(state):
            return state.get("valid", True)

        orchestrator.register_state_validator("test_node", custom_validator)

        assert "test_node" in orchestrator.state_validators
        assert orchestrator.state_validators["test_node"] == custom_validator

    def test_custom_state_validator_usage(self, orchestrator, sample_state):
        """Test that custom state validators are used during execution."""

        def failing_validator(state):
            return False

        orchestrator.register_state_validator("test_node", failing_validator)

        # The orchestrator should use error recovery instead of raising exceptions
        updates = orchestrator.execute_node_with_orchestration(
            "test_node", sample_state
        )

        # Check that error recovery was applied for validation failure
        assert updates["error_occurred"] is True
        assert "Custom validation failed" in updates["error_message"]
        assert updates["recovery_strategy"] == "default_continue"

    def test_get_execution_metrics(self, orchestrator, sample_state):
        """Test getting execution metrics."""
        # Execute some nodes to generate metrics
        orchestrator.execute_node_with_orchestration("test_node", sample_state)
        orchestrator.execute_node_with_orchestration("test_node", sample_state)

        metrics = orchestrator.get_execution_metrics()

        assert metrics["total_executions"] == 2
        assert metrics["successful_executions"] == 2
        assert metrics["failed_executions"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["failure_rate"] == 0.0
        assert len(metrics["recent_executions"]) == 2

        # Check recent executions format
        recent = metrics["recent_executions"][0]
        assert "node_name" in recent
        assert "success" in recent
        assert "execution_time" in recent
        assert "timestamp" in recent
        assert "phase" in recent

    def test_get_node_registry(self, orchestrator, sample_node_registry):
        """Test getting node registry."""
        registry = orchestrator.get_node_registry()

        assert registry == sample_node_registry
        assert registry is not orchestrator.nodes  # Should be a copy

    def test_validate_orchestrator_state(self, orchestrator):
        """Test orchestrator state validation."""
        validation_results = orchestrator.validate_orchestrator_state()

        assert validation_results["nodes_registered"] == 3
        assert validation_results["flow_state_manager"] is True
        assert validation_results["conditions"] is True
        assert validation_results["all_nodes_valid"] is True
        assert len(validation_results["invalid_nodes"]) == 0

    def test_default_error_recovery_strategies(self, orchestrator, sample_state):
        """Test that default error recovery strategies are initialized."""
        # Discussion round recovery
        updates = orchestrator._default_error_recovery(
            "discussion_round", Exception("test"), sample_state
        )
        assert updates["error_occurred"] is True
        assert updates["recovery_strategy"] == "default_continue"

        # The default strategies should be registered during initialization
        assert len(orchestrator.error_recovery_strategies) > 0

    def test_execution_history_maintenance(self, orchestrator, sample_state):
        """Test that execution history is properly maintained."""
        # Execute many nodes to test history limit
        for i in range(105):  # More than the 100 limit
            try:
                orchestrator.execute_node_with_orchestration("test_node", sample_state)
            except:
                pass  # Some might fail, that's ok for this test

        # History should be limited to 100 entries
        assert len(orchestrator.execution_history) <= 100

    def test_node_execution_timing(self, orchestrator, sample_state):
        """Test that execution timing is properly recorded."""
        updates = orchestrator.execute_node_with_orchestration(
            "test_node", sample_state
        )

        result = orchestrator.execution_history[0]
        assert result.execution_time > 0
        assert isinstance(result.execution_time, float)

    def test_execution_with_invalid_node_return(self, orchestrator, sample_state):
        """Test handling of nodes that return invalid data."""

        # Create a node that returns invalid data
        class InvalidReturnNode(FlowNode):
            def execute(self, state):
                return "invalid"  # Should return dict

            def validate_preconditions(self, state):
                return True

            def get_node_name(self):
                return "InvalidReturnNode"

        orchestrator.nodes["invalid_return_node"] = InvalidReturnNode()

        # The orchestrator should use error recovery instead of raising exceptions
        updates = orchestrator.execute_node_with_orchestration(
            "invalid_return_node", sample_state
        )

        # Check that error recovery was applied for invalid return type
        assert updates["error_occurred"] is True
        assert "invalid updates type" in updates["error_message"]
        assert updates["recovery_strategy"] == "default_continue"


class TestNodeExecutionResult:
    """Test the NodeExecutionResult class."""

    def test_initialization(self):
        """Test NodeExecutionResult initialization."""
        error = Exception("test error")
        result = NodeExecutionResult(
            node_name="test_node",
            success=True,
            updates={"test": "data"},
            execution_time=0.5,
            phase=ExecutionPhase.COMPLETED,
            error=error,
            recovery_attempted=True,
        )

        assert result.node_name == "test_node"
        assert result.success is True
        assert result.updates == {"test": "data"}
        assert result.execution_time == 0.5
        assert result.phase == ExecutionPhase.COMPLETED
        assert result.error == error
        assert result.recovery_attempted is True
        assert result.timestamp is not None
        assert result.execution_id is not None


class TestOrchestratorIntegration:
    """Test orchestrator integration with error recovery manager."""

    def test_error_recovery_manager_integration(self, orchestrator, sample_state):
        """Test that orchestrator properly integrates with error recovery manager."""
        # Create a custom error recovery manager to test integration
        from virtual_agora.flow.error_recovery import (
            ErrorRecoveryManager,
            ErrorSeverity,
        )

        error_manager = ErrorRecoveryManager()

        # Test that error severity assessment works through error recovery manager
        severity = error_manager._assess_error_severity(
            "discussion_round", Exception("test"), sample_state
        )
        assert severity == ErrorSeverity.HIGH

        severity = error_manager._assess_error_severity(
            "user_approval", Exception("test"), sample_state
        )
        assert severity == ErrorSeverity.LOW

        severity = error_manager._assess_error_severity(
            "test_node", SystemError("test"), sample_state
        )
        assert severity == ErrorSeverity.CRITICAL

        severity = error_manager._assess_error_severity(
            "test_node", Exception("timeout occurred"), sample_state
        )
        assert severity == ErrorSeverity.MEDIUM


if __name__ == "__main__":
    pytest.main([__file__])
