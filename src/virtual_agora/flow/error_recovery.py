"""Error recovery system for Virtual Agora discussion flow.

This module provides comprehensive error recovery strategies for flow nodes.
It implements:

- Pluggable recovery strategy registration
- Node-specific and generic recovery mechanisms
- Error categorization and handling
- Recovery attempt tracking and metrics

This supports Step 2.3 of the architecture refactoring by centralizing
error handling logic that was previously scattered throughout the system.
"""

import traceback
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from enum import Enum

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors in flow execution."""

    LOW = "low"  # Non-critical errors that don't affect flow
    MEDIUM = "medium"  # Errors that may affect current operation
    HIGH = "high"  # Errors that significantly impact flow
    CRITICAL = "critical"  # Errors that may cause flow failure


class RecoveryStrategy(Enum):
    """Types of recovery strategies available."""

    CONTINUE = "continue"  # Continue with default values
    RETRY = "retry"  # Retry the operation
    SKIP = "skip"  # Skip the current operation
    FALLBACK = "fallback"  # Use fallback mechanism
    ABORT = "abort"  # Abort current operation gracefully


class ErrorRecoveryResult:
    """Result of an error recovery attempt."""

    def __init__(
        self,
        node_name: str,
        original_error: Exception,
        recovery_strategy: RecoveryStrategy,
        success: bool,
        recovery_updates: Dict[str, Any],
        recovery_time: float,
        error_severity: ErrorSeverity,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        self.node_name = node_name
        self.original_error = original_error
        self.recovery_strategy = recovery_strategy
        self.success = success
        self.recovery_updates = recovery_updates
        self.recovery_time = recovery_time
        self.error_severity = error_severity
        self.additional_info = additional_info or {}
        self.timestamp = datetime.now().isoformat()
        self.error_trace = traceback.format_exc()


class ErrorRecoveryManager:
    """Manages error recovery strategies for flow nodes.

    This class provides a centralized system for handling errors that occur
    during node execution. It supports:

    - Registration of custom recovery strategies per node
    - Generic recovery strategies for common error types
    - Error categorization and severity assessment
    - Recovery attempt tracking and metrics
    - Fallback mechanisms for unhandled errors

    The manager enables graceful error handling and improves system
    resilience by providing multiple recovery options for different
    types of failures.
    """

    def __init__(self):
        """Initialize the error recovery manager."""
        self.recovery_strategies: Dict[str, Callable] = {}
        self.generic_strategies: Dict[type, Callable] = {}
        self.recovery_history: List[ErrorRecoveryResult] = []
        self.recovery_metrics = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_by_strategy": {},
            "recovery_by_node": {},
            "recovery_by_severity": {},
        }

        # Initialize default recovery strategies
        self._initialize_default_strategies()

        logger.debug("Initialized ErrorRecoveryManager")

    def register_recovery_strategy(
        self,
        node_name: str,
        strategy: Callable[[str, Exception, VirtualAgoraState], Dict[str, Any]],
    ) -> None:
        """Register error recovery strategy for specific node.

        Args:
            node_name: Name of the node to register strategy for
            strategy: Callable that takes (node_name, error, state) and returns state updates
        """
        if not callable(strategy):
            raise TypeError("Recovery strategy must be callable")

        self.recovery_strategies[node_name] = strategy
        logger.debug(f"Registered custom recovery strategy for '{node_name}'")

    def register_generic_strategy(
        self,
        error_type: type,
        strategy: Callable[[str, Exception, VirtualAgoraState], Dict[str, Any]],
    ) -> None:
        """Register generic recovery strategy for error type.

        Args:
            error_type: Type of exception to handle
            strategy: Callable that takes (node_name, error, state) and returns state updates
        """
        if not callable(strategy):
            raise TypeError("Recovery strategy must be callable")

        self.generic_strategies[error_type] = strategy
        logger.debug(f"Registered generic recovery strategy for {error_type.__name__}")

    def attempt_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> ErrorRecoveryResult:
        """Attempt recovery from node execution error.

        Args:
            node_name: Name of the failed node
            error: The error that occurred
            state: Current Virtual Agora state

        Returns:
            ErrorRecoveryResult containing recovery outcome
        """
        recovery_start = datetime.now()
        self.recovery_metrics["total_recoveries"] += 1

        logger.info(f"=== ERROR RECOVERY: Attempting recovery for '{node_name}' ===")
        logger.info(f"Error type: {type(error).__name__}")
        logger.info(f"Error message: {str(error)}")

        try:
            # Assess error severity
            severity = self._assess_error_severity(node_name, error, state)

            # Determine recovery strategy
            recovery_strategy, strategy_func = self._select_recovery_strategy(
                node_name, error, state
            )

            logger.info(f"Selected recovery strategy: {recovery_strategy.value}")
            logger.info(f"Error severity: {severity.value}")

            # Attempt recovery
            recovery_updates = strategy_func(node_name, error, state)

            recovery_time = (datetime.now() - recovery_start).total_seconds()

            result = ErrorRecoveryResult(
                node_name=node_name,
                original_error=error,
                recovery_strategy=recovery_strategy,
                success=True,
                recovery_updates=recovery_updates,
                recovery_time=recovery_time,
                error_severity=severity,
                additional_info={
                    "strategy_source": (
                        "custom" if node_name in self.recovery_strategies else "default"
                    )
                },
            )

            # Update metrics
            self._update_recovery_metrics(result)

            # Record in history
            self.recovery_history.append(result)
            self._maintain_history_size()

            logger.info(
                f"Recovery successful for '{node_name}' in {recovery_time:.3f}s"
            )
            return result

        except Exception as recovery_error:
            recovery_time = (datetime.now() - recovery_start).total_seconds()

            logger.error(f"Recovery failed for '{node_name}': {recovery_error}")

            result = ErrorRecoveryResult(
                node_name=node_name,
                original_error=error,
                recovery_strategy=RecoveryStrategy.ABORT,
                success=False,
                recovery_updates={},
                recovery_time=recovery_time,
                error_severity=ErrorSeverity.CRITICAL,
                additional_info={"recovery_error": str(recovery_error)},
            )

            # Update metrics
            self._update_recovery_metrics(result)

            # Record in history
            self.recovery_history.append(result)
            self._maintain_history_size()

            return result

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get error recovery metrics and statistics.

        Returns:
            Dictionary containing recovery metrics
        """
        metrics = self.recovery_metrics.copy()

        # Calculate derived metrics
        total_recoveries = metrics["total_recoveries"]
        if total_recoveries > 0:
            metrics["success_rate"] = (
                metrics["successful_recoveries"] / total_recoveries
            )
            metrics["failure_rate"] = metrics["failed_recoveries"] / total_recoveries
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0

        # Add recent recovery history
        metrics["recent_recoveries"] = [
            {
                "node_name": result.node_name,
                "success": result.success,
                "strategy": result.recovery_strategy.value,
                "severity": result.error_severity.value,
                "recovery_time": result.recovery_time,
                "timestamp": result.timestamp,
            }
            for result in self.recovery_history[-10:]  # Last 10 recoveries
        ]

        return metrics

    def get_recovery_history(
        self, node_name: Optional[str] = None
    ) -> List[ErrorRecoveryResult]:
        """Get recovery history, optionally filtered by node name.

        Args:
            node_name: Optional node name to filter by

        Returns:
            List of recovery results
        """
        if node_name:
            return [
                result
                for result in self.recovery_history
                if result.node_name == node_name
            ]
        return self.recovery_history.copy()

    def clear_recovery_history(self) -> None:
        """Clear recovery history."""
        cleared_count = len(self.recovery_history)
        self.recovery_history.clear()
        logger.info(f"Cleared {cleared_count} recovery records from history")

    def _assess_error_severity(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> ErrorSeverity:
        """Assess the severity of an error.

        Args:
            node_name: Name of the failed node
            error: The error that occurred
            state: Current state

        Returns:
            ErrorSeverity level
        """
        # Critical nodes that should never fail completely
        critical_nodes = {
            "config_and_keys",
            "agent_instantiation",
            "discussion_round",
            "final_report_generation",
            "multi_file_output",
        }

        # Error types that are always critical
        critical_errors = (SystemError, MemoryError, KeyboardInterrupt)

        # User interaction nodes are generally less critical
        user_interaction_nodes = {
            "user_approval",
            "user_turn_participation",
            "periodic_user_stop",
            "agenda_approval",
            "user_topic_conclusion_confirmation",
        }

        if isinstance(error, critical_errors):
            return ErrorSeverity.CRITICAL
        elif node_name in critical_nodes:
            return ErrorSeverity.HIGH
        elif node_name in user_interaction_nodes:
            return ErrorSeverity.LOW
        elif "timeout" in str(error).lower() or "connection" in str(error).lower():
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.MEDIUM

    def _select_recovery_strategy(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> tuple:
        """Select appropriate recovery strategy for the error.

        Args:
            node_name: Name of the failed node
            error: The error that occurred
            state: Current state

        Returns:
            Tuple of (RecoveryStrategy, strategy_function)
        """
        # Check for node-specific strategy
        if node_name in self.recovery_strategies:
            return RecoveryStrategy.FALLBACK, self.recovery_strategies[node_name]

        # Check for error-type-specific strategy
        error_type = type(error)
        if error_type in self.generic_strategies:
            return RecoveryStrategy.FALLBACK, self.generic_strategies[error_type]

        # Check for parent class strategies
        for registered_type, strategy in self.generic_strategies.items():
            if isinstance(error, registered_type):
                return RecoveryStrategy.FALLBACK, strategy

        # Use default strategy based on node type and error
        return self._select_default_strategy(node_name, error, state)

    def _select_default_strategy(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> tuple:
        """Select default recovery strategy.

        Args:
            node_name: Name of the failed node
            error: The error that occurred
            state: Current state

        Returns:
            Tuple of (RecoveryStrategy, strategy_function)
        """
        # User interaction nodes: continue with defaults
        if "user" in node_name.lower() and (
            "interaction" in node_name.lower()
            or "approval" in node_name.lower()
            or "participation" in node_name.lower()
        ):
            return RecoveryStrategy.CONTINUE, self._user_interaction_recovery

        # Discussion nodes: skip problematic round
        elif "discussion" in node_name.lower() or "round" in node_name.lower():
            return RecoveryStrategy.SKIP, self._discussion_recovery

        # Report generation nodes: use simplified output
        elif "report" in node_name.lower() or "output" in node_name.lower():
            return RecoveryStrategy.FALLBACK, self._report_generation_recovery

        # Default: continue with error flag
        else:
            return RecoveryStrategy.CONTINUE, self._default_continue_recovery

    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies."""

        # Register generic strategies for common exception types
        self.register_generic_strategy(TimeoutError, self._timeout_recovery)
        self.register_generic_strategy(ConnectionError, self._connection_recovery)
        self.register_generic_strategy(ValueError, self._value_error_recovery)
        self.register_generic_strategy(KeyError, self._key_error_recovery)

        logger.debug("Initialized default recovery strategies")

    def _user_interaction_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for user interaction failures."""
        return {
            "user_interaction_error": True,
            "error_node": node_name,
            "error_message": f"User interaction failed: {error}",
            "use_default_action": True,
            "user_forced_conclusion": False,
            "user_approves_continuation": True,  # Default to continuing
            "recovery_strategy": "default_user_action",
        }

    def _discussion_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for discussion round failures."""
        return {
            "discussion_round_error": True,
            "error_node": node_name,
            "error_message": f"Discussion round failed: {error}",
            "continue_with_summary": True,
            "round_completed": False,
            "skip_to_next_round": True,
            "recovery_strategy": "skip_problematic_round",
        }

    def _report_generation_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for report generation failures."""
        return {
            "report_generation_error": True,
            "error_node": node_name,
            "error_message": f"Report generation failed: {error}",
            "use_simplified_report": True,
            "report_content": f"Report generation failed due to: {error}",
            "recovery_strategy": "simplified_output",
        }

    def _default_continue_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Default recovery strategy: continue with error flag."""
        return {
            "error_occurred": True,
            "error_node": node_name,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "error_timestamp": datetime.now().isoformat(),
            "continue_flow": True,
            "recovery_strategy": "continue_with_error",
        }

    def _timeout_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for timeout errors."""
        return {
            "timeout_error": True,
            "error_node": node_name,
            "error_message": f"Operation timed out: {error}",
            "retry_with_longer_timeout": True,
            "use_cached_result": True,
            "recovery_strategy": "timeout_fallback",
        }

    def _connection_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for connection errors."""
        return {
            "connection_error": True,
            "error_node": node_name,
            "error_message": f"Connection failed: {error}",
            "retry_connection": True,
            "use_offline_mode": True,
            "recovery_strategy": "connection_fallback",
        }

    def _value_error_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for value errors."""
        return {
            "value_error": True,
            "error_node": node_name,
            "error_message": f"Invalid value: {error}",
            "use_default_values": True,
            "validate_inputs": True,
            "recovery_strategy": "value_correction",
        }

    def _key_error_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Recovery strategy for key errors."""
        return {
            "key_error": True,
            "error_node": node_name,
            "error_message": f"Missing key: {error}",
            "provide_default_key": True,
            "initialize_missing_state": True,
            "recovery_strategy": "key_initialization",
        }

    def _update_recovery_metrics(self, result: ErrorRecoveryResult) -> None:
        """Update recovery metrics with result.

        Args:
            result: Recovery result to record
        """
        if result.success:
            self.recovery_metrics["successful_recoveries"] += 1
        else:
            self.recovery_metrics["failed_recoveries"] += 1

        # Update strategy metrics
        strategy_key = result.recovery_strategy.value
        if strategy_key not in self.recovery_metrics["recovery_by_strategy"]:
            self.recovery_metrics["recovery_by_strategy"][strategy_key] = {
                "total": 0,
                "successful": 0,
            }
        self.recovery_metrics["recovery_by_strategy"][strategy_key]["total"] += 1
        if result.success:
            self.recovery_metrics["recovery_by_strategy"][strategy_key][
                "successful"
            ] += 1

        # Update node metrics
        node_key = result.node_name
        if node_key not in self.recovery_metrics["recovery_by_node"]:
            self.recovery_metrics["recovery_by_node"][node_key] = {
                "total": 0,
                "successful": 0,
            }
        self.recovery_metrics["recovery_by_node"][node_key]["total"] += 1
        if result.success:
            self.recovery_metrics["recovery_by_node"][node_key]["successful"] += 1

        # Update severity metrics
        severity_key = result.error_severity.value
        if severity_key not in self.recovery_metrics["recovery_by_severity"]:
            self.recovery_metrics["recovery_by_severity"][severity_key] = {
                "total": 0,
                "successful": 0,
            }
        self.recovery_metrics["recovery_by_severity"][severity_key]["total"] += 1
        if result.success:
            self.recovery_metrics["recovery_by_severity"][severity_key][
                "successful"
            ] += 1

    def _maintain_history_size(self, max_history: int = 100) -> None:
        """Maintain recovery history within size limits.

        Args:
            max_history: Maximum number of recovery records to keep
        """
        if len(self.recovery_history) > max_history:
            self.recovery_history = self.recovery_history[-max_history:]


def create_error_recovery_manager() -> ErrorRecoveryManager:
    """Factory function to create a configured ErrorRecoveryManager.

    Returns:
        Configured ErrorRecoveryManager instance
    """
    manager = ErrorRecoveryManager()
    logger.info("Created ErrorRecoveryManager with default strategies")
    return manager
