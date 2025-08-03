"""Flow orchestrator for Virtual Agora discussion flow.

This module implements the high-level orchestration layer that separates
orchestration concerns from implementation details. It provides:

- Centralized node execution management
- Error handling and recovery strategies
- Flow state validation and transitions
- Node registry management
- Comprehensive logging and monitoring

This addresses Step 2.3 of the architecture refactoring plan by creating
a clear separation between high-level flow orchestration and low-level
graph implementation details.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.state_manager import FlowStateManager
from virtual_agora.flow.edges_v13 import V13FlowConditions
from virtual_agora.flow.nodes.base import (
    FlowNode,
    NodeExecutionError,
    NodeValidationError,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionPhase(Enum):
    """Phases of node execution for orchestration tracking."""

    PRE_VALIDATION = "pre_validation"
    EXECUTION = "execution"
    POST_VALIDATION = "post_validation"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"


class NodeExecutionResult:
    """Result of node execution with orchestration metadata."""

    def __init__(
        self,
        node_name: str,
        success: bool,
        updates: Dict[str, Any],
        execution_time: float,
        phase: ExecutionPhase,
        error: Optional[Exception] = None,
        recovery_attempted: bool = False,
    ):
        self.node_name = node_name
        self.success = success
        self.updates = updates
        self.execution_time = execution_time
        self.phase = phase
        self.error = error
        self.recovery_attempted = recovery_attempted
        self.timestamp = datetime.now().isoformat()
        self.execution_id = str(uuid.uuid4())


class DiscussionFlowOrchestrator:
    """High-level orchestration of discussion flow.

    This class separates orchestration concerns from implementation details:
    - Node registry management and execution coordination
    - Centralized error handling and recovery strategies
    - State transition validation and management
    - Flow routing decisions and condition evaluation
    - Comprehensive logging and monitoring integration

    Key design principles:
    - Single responsibility: Only handles orchestration, not implementation
    - Pluggable architecture: Uses node registry for dynamic node management
    - Error resilience: Comprehensive error handling with recovery strategies
    - State safety: Validates state before/after each node execution
    - Monitoring: Detailed execution tracking and logging
    """

    def __init__(
        self,
        node_registry: Dict[str, FlowNode],
        conditions: V13FlowConditions,
        flow_state_manager: FlowStateManager,
    ):
        """Initialize the flow orchestrator.

        Args:
            node_registry: Dictionary mapping node names to FlowNode instances
            conditions: V13FlowConditions for flow routing decisions
            flow_state_manager: FlowStateManager for state operations
        """
        self.nodes = node_registry
        self.conditions = conditions
        self.flow_state_manager = flow_state_manager

        # Orchestration state tracking
        self.execution_history: List[NodeExecutionResult] = []
        self.error_recovery_strategies: Dict[str, Callable] = {}
        self.state_validators: Dict[str, Callable] = {}
        self.execution_metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "recovery_attempts": 0,
            "recovery_successes": 0,
        }

        # Initialize default error recovery strategies
        self._initialize_default_recovery_strategies()

        logger.info(
            f"Initialized DiscussionFlowOrchestrator with {len(self.nodes)} nodes"
        )

    def execute_node_with_orchestration(
        self, node_name: str, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Execute single node with full orchestration.

        This method provides the core orchestration functionality:
        1. Pre-execution validation
        2. Node execution with error handling
        3. Post-execution validation
        4. Error recovery if needed
        5. Execution tracking and metrics

        Args:
            node_name: Name of the node to execute
            state: Current Virtual Agora state

        Returns:
            Dictionary containing state updates from node execution

        Raises:
            NodeExecutionError: If node execution fails and recovery is unsuccessful
            NodeValidationError: If state validation fails
        """
        execution_start = datetime.now()
        execution_id = str(uuid.uuid4())

        logger.info(
            f"=== ORCHESTRATOR: Executing node '{node_name}' [ID: {execution_id}] ==="
        )

        try:
            # Phase 1: Pre-execution validation
            logger.debug(f"Phase 1: Pre-execution validation for '{node_name}'")
            self._validate_node_preconditions(node_name, state)

            # Phase 2: Node execution
            logger.debug(f"Phase 2: Executing node '{node_name}'")
            updates = self._execute_node_safely(node_name, state)

            # Phase 3: Post-execution validation
            logger.debug(f"Phase 3: Post-execution validation for '{node_name}'")
            self._validate_node_results(node_name, state, updates)

            # Phase 4: Success tracking
            execution_time = (datetime.now() - execution_start).total_seconds()
            result = NodeExecutionResult(
                node_name=node_name,
                success=True,
                updates=updates,
                execution_time=execution_time,
                phase=ExecutionPhase.COMPLETED,
            )

            self._record_execution_result(result)
            logger.info(f"Successfully executed '{node_name}' in {execution_time:.3f}s")

            return updates

        except Exception as error:
            # Phase 5: Error recovery
            logger.error(f"Error executing '{node_name}': {error}")
            execution_time = (datetime.now() - execution_start).total_seconds()

            try:
                logger.info(f"Attempting error recovery for '{node_name}'")
                recovery_updates = self._attempt_error_recovery(node_name, error, state)

                result = NodeExecutionResult(
                    node_name=node_name,
                    success=True,
                    updates=recovery_updates,
                    execution_time=execution_time,
                    phase=ExecutionPhase.ERROR_RECOVERY,
                    error=error,
                    recovery_attempted=True,
                )

                self._record_execution_result(result)
                logger.info(f"Successfully recovered from error in '{node_name}'")

                return recovery_updates

            except Exception as recovery_error:
                logger.error(
                    f"Error recovery failed for '{node_name}': {recovery_error}"
                )

                result = NodeExecutionResult(
                    node_name=node_name,
                    success=False,
                    updates={},
                    execution_time=execution_time,
                    phase=ExecutionPhase.ERROR_RECOVERY,
                    error=error,
                    recovery_attempted=True,
                )

                self._record_execution_result(result)

                # Re-raise the original error with context
                raise NodeExecutionError(
                    f"Node '{node_name}' execution failed and recovery unsuccessful. "
                    f"Original error: {error}. Recovery error: {recovery_error}"
                ) from error

    def register_error_recovery_strategy(
        self,
        node_name: str,
        strategy: Callable[[str, Exception, VirtualAgoraState], Dict[str, Any]],
    ) -> None:
        """Register error recovery strategy for specific node.

        Args:
            node_name: Name of the node to register strategy for
            strategy: Callable that takes (node_name, error, state) and returns state updates
        """
        self.error_recovery_strategies[node_name] = strategy
        logger.debug(f"Registered error recovery strategy for '{node_name}'")

    def register_state_validator(
        self, node_name: str, validator: Callable[[VirtualAgoraState], bool]
    ) -> None:
        """Register state validator for specific node.

        Args:
            node_name: Name of the node to register validator for
            validator: Callable that takes state and returns True if valid
        """
        self.state_validators[node_name] = validator
        logger.debug(f"Registered state validator for '{node_name}'")

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics and statistics.

        Returns:
            Dictionary containing execution metrics and performance data
        """
        metrics = self.execution_metrics.copy()

        # Calculate derived metrics
        total_executions = metrics["total_executions"]
        if total_executions > 0:
            metrics["success_rate"] = (
                metrics["successful_executions"] / total_executions
            )
            metrics["failure_rate"] = metrics["failed_executions"] / total_executions
            metrics["recovery_rate"] = metrics["recovery_successes"] / max(
                metrics["recovery_attempts"], 1
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
            metrics["recovery_rate"] = 0.0

        # Add recent execution history
        metrics["recent_executions"] = [
            {
                "node_name": result.node_name,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp,
                "phase": result.phase.value,
            }
            for result in self.execution_history[-10:]  # Last 10 executions
        ]

        return metrics

    def get_node_registry(self) -> Dict[str, FlowNode]:
        """Get the current node registry.

        Returns:
            Dictionary mapping node names to FlowNode instances
        """
        return self.nodes.copy()

    def validate_orchestrator_state(self) -> Dict[str, Any]:
        """Validate the orchestrator's internal state.

        Returns:
            Dictionary containing validation results and status
        """
        validation_results = {
            "nodes_registered": len(self.nodes),
            "recovery_strategies": len(self.error_recovery_strategies),
            "state_validators": len(self.state_validators),
            "execution_history_length": len(self.execution_history),
            "metrics_available": bool(self.execution_metrics),
            "flow_state_manager": self.flow_state_manager is not None,
            "conditions": self.conditions is not None,
            "all_nodes_valid": True,
            "invalid_nodes": [],
        }

        # Validate all registered nodes
        for node_name, node in self.nodes.items():
            try:
                if not isinstance(node, FlowNode):
                    validation_results["all_nodes_valid"] = False
                    validation_results["invalid_nodes"].append(
                        f"{node_name}: not a FlowNode instance"
                    )
            except Exception as e:
                validation_results["all_nodes_valid"] = False
                validation_results["invalid_nodes"].append(
                    f"{node_name}: validation error - {e}"
                )

        logger.debug(f"Orchestrator state validation: {validation_results}")
        return validation_results

    def _validate_node_preconditions(
        self, node_name: str, state: VirtualAgoraState
    ) -> None:
        """Validate node preconditions before execution.

        Args:
            node_name: Name of the node to validate
            state: Current state to validate

        Raises:
            NodeValidationError: If preconditions are not met
        """
        if node_name not in self.nodes:
            raise NodeValidationError(f"Node '{node_name}' not found in registry")

        node = self.nodes[node_name]

        # Use node's built-in validation if available
        if hasattr(node, "validate_preconditions"):
            try:
                if not node.validate_preconditions(state):
                    error_details = getattr(
                        node, "_validation_errors", ["Unknown validation error"]
                    )
                    raise NodeValidationError(
                        f"Precondition validation failed for '{node_name}': {error_details}"
                    )
            except Exception as e:
                raise NodeValidationError(
                    f"Precondition validation error for '{node_name}': {e}"
                )

        # Use custom validator if registered
        if node_name in self.state_validators:
            try:
                if not self.state_validators[node_name](state):
                    raise NodeValidationError(
                        f"Custom validation failed for '{node_name}'"
                    )
            except Exception as e:
                raise NodeValidationError(
                    f"Custom validation error for '{node_name}': {e}"
                )

    def _execute_node_safely(
        self, node_name: str, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Execute node with error handling.

        Args:
            node_name: Name of the node to execute
            state: Current state

        Returns:
            State updates from node execution

        Raises:
            NodeExecutionError: If node execution fails
        """
        node = self.nodes[node_name]

        try:
            updates = node.execute(state)

            if not isinstance(updates, dict):
                raise NodeExecutionError(
                    f"Node '{node_name}' returned invalid updates type: {type(updates)}"
                )

            return updates

        except Exception as e:
            raise NodeExecutionError(f"Execution failed for '{node_name}': {e}") from e

    def _validate_node_results(
        self, node_name: str, state: VirtualAgoraState, updates: Dict[str, Any]
    ) -> None:
        """Validate node execution results.

        Args:
            node_name: Name of the executed node
            state: Original state
            updates: Updates returned by node

        Raises:
            NodeValidationError: If results are invalid
        """
        # Basic validation
        if not isinstance(updates, dict):
            raise NodeValidationError(
                f"Node '{node_name}' returned invalid updates: {type(updates)}"
            )

        # Check for required update patterns
        # This could be extended based on specific node requirements
        logger.debug(f"Post-execution validation passed for '{node_name}'")

    def _attempt_error_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Attempt to recover from node execution error.

        Args:
            node_name: Name of the failed node
            error: The error that occurred
            state: Current state

        Returns:
            Recovery state updates

        Raises:
            Exception: If recovery fails or no strategy available
        """
        self.execution_metrics["recovery_attempts"] += 1

        if node_name in self.error_recovery_strategies:
            try:
                recovery_updates = self.error_recovery_strategies[node_name](
                    node_name, error, state
                )
                self.execution_metrics["recovery_successes"] += 1
                logger.info(
                    f"Successfully recovered from error in '{node_name}' using custom strategy"
                )
                return recovery_updates
            except Exception as recovery_error:
                logger.error(
                    f"Custom recovery strategy failed for '{node_name}': {recovery_error}"
                )
                raise
        else:
            # Use default recovery strategy
            try:
                recovery_updates = self._default_error_recovery(node_name, error, state)
                self.execution_metrics["recovery_successes"] += 1
                logger.info(
                    f"Successfully recovered from error in '{node_name}' using default strategy"
                )
                return recovery_updates
            except Exception as recovery_error:
                logger.error(
                    f"Default recovery strategy failed for '{node_name}': {recovery_error}"
                )
                raise

    def _default_error_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Default error recovery strategy.

        Args:
            node_name: Name of the failed node
            error: The error that occurred
            state: Current state

        Returns:
            Recovery state updates
        """
        # For most nodes, the default recovery is to return empty updates
        # and mark an error flag so the flow can continue gracefully
        recovery_updates = {
            "error_occurred": True,
            "error_node": node_name,
            "error_message": str(error),
            "error_timestamp": datetime.now().isoformat(),
            "recovery_strategy": "default_continue",
        }

        logger.warning(
            f"Applied default recovery for '{node_name}': continue with error flag"
        )
        return recovery_updates

    def _record_execution_result(self, result: NodeExecutionResult) -> None:
        """Record execution result in history and update metrics.

        Args:
            result: The execution result to record
        """
        self.execution_history.append(result)

        # Keep only the last 100 executions to prevent memory growth
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

        # Update metrics
        self.execution_metrics["total_executions"] += 1
        if result.success:
            self.execution_metrics["successful_executions"] += 1
        else:
            self.execution_metrics["failed_executions"] += 1

    def _initialize_default_recovery_strategies(self) -> None:
        """Initialize default error recovery strategies for common nodes."""
        # Default strategies for critical nodes that should never completely fail

        def discussion_round_recovery(
            node_name: str, error: Exception, state: VirtualAgoraState
        ) -> Dict[str, Any]:
            """Recovery strategy for discussion round failures."""
            return {
                "discussion_round_error": True,
                "error_message": f"Discussion round failed: {error}",
                "continue_with_summary": True,
                "round_completed": False,
            }

        def user_interaction_recovery(
            node_name: str, error: Exception, state: VirtualAgoraState
        ) -> Dict[str, Any]:
            """Recovery strategy for user interaction failures."""
            return {
                "user_interaction_error": True,
                "error_message": f"User interaction failed: {error}",
                "use_default_action": True,
                "user_forced_conclusion": False,
            }

        # Register default strategies
        self.error_recovery_strategies["discussion_round"] = discussion_round_recovery
        self.error_recovery_strategies["user_turn_participation"] = (
            user_interaction_recovery
        )
        self.error_recovery_strategies["periodic_user_stop"] = user_interaction_recovery
        self.error_recovery_strategies["user_approval"] = user_interaction_recovery

        logger.debug("Initialized default error recovery strategies")
