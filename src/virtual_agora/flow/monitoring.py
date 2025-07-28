"""Flow Monitoring and Debugging for Virtual Agora.

This module provides comprehensive monitoring, debugging, and metrics collection
capabilities for the LangGraph discussion flow.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import contextmanager

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FlowMetrics:
    """Metrics for a single flow operation."""

    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NodeExecutionInfo:
    """Information about node execution."""

    node_name: str
    execution_count: int
    total_duration_ms: float
    average_duration_ms: float
    success_count: int
    failure_count: int
    last_execution: datetime
    errors: List[str]


@dataclass
class PhaseTransitionInfo:
    """Information about phase transitions."""

    from_phase: int
    to_phase: int
    transition_count: int
    average_duration_ms: float
    last_transition: datetime
    success_rate: float


@dataclass
class DebugSnapshot:
    """A comprehensive debugging snapshot."""

    timestamp: datetime
    session_id: str
    current_phase: int
    current_node: Optional[str]
    state_summary: Dict[str, Any]
    recent_metrics: List[FlowMetrics]
    active_agents: List[str]
    message_count: int
    votes_cast: int
    errors: List[str]
    performance_issues: List[str]


class FlowMonitor:
    """Monitors Virtual Agora flow execution and collects metrics."""

    def __init__(self, max_metrics_history: int = 1000):
        """Initialize flow monitor.

        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
        """
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.node_stats: Dict[str, NodeExecutionInfo] = {}
        self.phase_transition_stats: Dict[str, PhaseTransitionInfo] = {}
        self.active_operations: Dict[str, FlowMetrics] = {}
        self.session_start_time: Optional[datetime] = None
        self.debug_snapshots: List[DebugSnapshot] = []

    def start_session(self, session_id: str):
        """Start monitoring a new session.

        Args:
            session_id: Session identifier
        """
        self.session_start_time = datetime.now()
        logger.debug(f"Started monitoring session: {session_id}")

    def start_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start monitoring an operation.

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata

        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        metric = FlowMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            metadata=metadata or {},
        )

        self.active_operations[operation_id] = metric
        logger.debug(f"Started operation: {operation_name} (ID: {operation_id})")

        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ):
        """End monitoring an operation.

        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            error_message: Optional error message if failed
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return

        metric = self.active_operations.pop(operation_id)
        metric.end_time = datetime.now()
        metric.duration_ms = (
            metric.end_time - metric.start_time
        ).total_seconds() * 1000
        metric.success = success
        metric.error_message = error_message

        self.metrics_history.append(metric)
        self._update_node_stats(metric)

        logger.debug(
            f"Ended operation: {metric.operation_name} "
            f"(Duration: {metric.duration_ms:.2f}ms, Success: {success})"
        )

    @contextmanager
    def monitor_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for monitoring operations.

        Args:
            operation_name: Name of the operation
            metadata: Optional metadata
        """
        operation_id = self.start_operation(operation_name, metadata)
        try:
            yield operation_id
            self.end_operation(operation_id, success=True)
        except Exception as e:
            self.end_operation(operation_id, success=False, error_message=str(e))
            raise

    def record_phase_transition(
        self, from_phase: int, to_phase: int, duration_ms: float, success: bool = True
    ):
        """Record a phase transition.

        Args:
            from_phase: Source phase
            to_phase: Target phase
            duration_ms: Transition duration in milliseconds
            success: Whether transition succeeded
        """
        transition_key = f"{from_phase}->{to_phase}"

        if transition_key not in self.phase_transition_stats:
            self.phase_transition_stats[transition_key] = PhaseTransitionInfo(
                from_phase=from_phase,
                to_phase=to_phase,
                transition_count=0,
                average_duration_ms=0.0,
                last_transition=datetime.now(),
                success_rate=0.0,
            )

        stats = self.phase_transition_stats[transition_key]
        stats.transition_count += 1
        stats.last_transition = datetime.now()

        # Update average duration
        total_duration = (
            stats.average_duration_ms * (stats.transition_count - 1) + duration_ms
        )
        stats.average_duration_ms = total_duration / stats.transition_count

        # Update success rate
        if success:
            stats.success_rate = (
                (stats.success_rate * (stats.transition_count - 1)) + 1.0
            ) / stats.transition_count
        else:
            stats.success_rate = (
                stats.success_rate * (stats.transition_count - 1)
            ) / stats.transition_count

        logger.info(
            f"Recorded phase transition: {transition_key} "
            f"(Duration: {duration_ms:.2f}ms, Success: {success})"
        )

    def create_debug_snapshot(
        self, state: VirtualAgoraState, current_node: Optional[str] = None
    ) -> DebugSnapshot:
        """Create a comprehensive debugging snapshot.

        Args:
            state: Current Virtual Agora state
            current_node: Currently executing node

        Returns:
            Debug snapshot
        """
        # Extract recent metrics (last 10)
        recent_metrics = list(self.metrics_history)[-10:]

        # Count active agents
        active_agents = list(state.get("agents", {}).keys())

        # Count messages and votes
        message_count = len(state.get("messages", []))
        votes_cast = len(state.get("vote_history", []))

        # Collect recent errors
        recent_errors = [
            m.error_message for m in recent_metrics if m.error_message is not None
        ]

        # Identify performance issues
        performance_issues = self._identify_performance_issues(recent_metrics)

        # Create state summary (avoid sensitive data)
        state_summary = {
            "session_id": state.get("session_id"),
            "current_phase": state.get("current_phase"),
            "phase_start_time": (
                state.get("phase_start_time", datetime.now()).isoformat()
                if state.get("phase_start_time")
                else None
            ),
            "agenda_topics": len(state.get("agenda_topics", [])),
            "active_topic": state.get("active_topic"),
            "message_count": message_count,
            "agent_count": len(active_agents),
            "current_round": state.get("current_round", 0),
            "flow_control": state.get("flow_control", {}),
            "hitl_state": {
                "awaiting_approval": state.get("hitl_state", {}).get(
                    "awaiting_approval", False
                ),
                "approval_type": state.get("hitl_state", {}).get("approval_type"),
            },
        }

        snapshot = DebugSnapshot(
            timestamp=datetime.now(),
            session_id=state.get("session_id", "unknown"),
            current_phase=state.get("current_phase", 0),
            current_node=current_node,
            state_summary=state_summary,
            recent_metrics=recent_metrics,
            active_agents=active_agents,
            message_count=message_count,
            votes_cast=votes_cast,
            errors=recent_errors,
            performance_issues=performance_issues,
        )

        self.debug_snapshots.append(snapshot)

        # Keep only last 50 snapshots
        if len(self.debug_snapshots) > 50:
            self.debug_snapshots = self.debug_snapshots[-50:]

        logger.debug(f"Created debug snapshot for session {snapshot.session_id}")
        return snapshot

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report.

        Returns:
            Performance report dictionary
        """
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}

        total_operations = len(self.metrics_history)
        successful_operations = sum(1 for m in self.metrics_history if m.success)
        failed_operations = total_operations - successful_operations

        # Calculate average durations by operation type
        operation_durations = defaultdict(list)
        for metric in self.metrics_history:
            if metric.duration_ms is not None:
                operation_durations[metric.operation_name].append(metric.duration_ms)

        operation_stats = {}
        for op_name, durations in operation_durations.items():
            operation_stats[op_name] = {
                "count": len(durations),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
            }

        # Recent performance trends (last 100 operations)
        recent_metrics = list(self.metrics_history)[-100:]
        recent_avg_duration = sum(
            m.duration_ms for m in recent_metrics if m.duration_ms
        ) / len(recent_metrics)
        recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(
            recent_metrics
        )

        return {
            "status": "active",
            "session_start_time": (
                self.session_start_time.isoformat() if self.session_start_time else None
            ),
            "uptime_minutes": (
                (datetime.now() - self.session_start_time).total_seconds() / 60
                if self.session_start_time
                else 0
            ),
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "overall_success_rate": successful_operations / total_operations,
            "recent_avg_duration_ms": recent_avg_duration,
            "recent_success_rate": recent_success_rate,
            "operation_stats": operation_stats,
            "node_stats": {
                name: asdict(stats) for name, stats in self.node_stats.items()
            },
            "phase_transition_stats": {
                key: asdict(stats) for key, stats in self.phase_transition_stats.items()
            },
            "active_operations": len(self.active_operations),
            "debug_snapshots_count": len(self.debug_snapshots),
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors and issues.

        Returns:
            Error summary dictionary
        """
        failed_metrics = [m for m in self.metrics_history if not m.success]

        if not failed_metrics:
            return {"status": "healthy", "total_errors": 0}

        # Group errors by type
        error_types = defaultdict(list)
        for metric in failed_metrics:
            error_key = metric.error_message or "unknown_error"
            error_types[error_key].append(
                {
                    "operation": metric.operation_name,
                    "timestamp": metric.start_time.isoformat(),
                    "duration_ms": metric.duration_ms,
                }
            )

        # Find most frequent errors
        error_frequency = {
            error: len(occurrences) for error, occurrences in error_types.items()
        }
        most_frequent_errors = sorted(
            error_frequency.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Recent error trend
        recent_hours = 1
        cutoff_time = datetime.now() - timedelta(hours=recent_hours)
        recent_errors = [m for m in failed_metrics if m.start_time >= cutoff_time]

        return {
            "status": "errors_detected",
            "total_errors": len(failed_metrics),
            "unique_error_types": len(error_types),
            "most_frequent_errors": most_frequent_errors,
            "recent_errors_count": len(recent_errors),
            "error_rate": len(failed_metrics) / len(self.metrics_history),
            "errors_by_type": {
                error: len(occurrences) for error, occurrences in error_types.items()
            },
        }

    def export_debug_data(self, include_snapshots: bool = True) -> Dict[str, Any]:
        """Export all debugging data for analysis.

        Args:
            include_snapshots: Whether to include debug snapshots

        Returns:
            Complete debug data export
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "session_start_time": (
                self.session_start_time.isoformat() if self.session_start_time else None
            ),
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "node_stats": {
                name: asdict(stats) for name, stats in self.node_stats.items()
            },
            "phase_transition_stats": {
                key: asdict(stats) for key, stats in self.phase_transition_stats.items()
            },
            "active_operations": [asdict(m) for m in self.active_operations.values()],
            "performance_report": self.get_performance_report(),
            "error_summary": self.get_error_summary(),
        }

        if include_snapshots:
            export_data["debug_snapshots"] = [asdict(s) for s in self.debug_snapshots]

        return export_data

    def _update_node_stats(self, metric: FlowMetrics):
        """Update node execution statistics.

        Args:
            metric: Completed flow metric
        """
        node_name = metric.operation_name

        if node_name not in self.node_stats:
            self.node_stats[node_name] = NodeExecutionInfo(
                node_name=node_name,
                execution_count=0,
                total_duration_ms=0.0,
                average_duration_ms=0.0,
                success_count=0,
                failure_count=0,
                last_execution=datetime.now(),
                errors=[],
            )

        stats = self.node_stats[node_name]
        stats.execution_count += 1
        stats.last_execution = datetime.now()

        if metric.duration_ms is not None:
            stats.total_duration_ms += metric.duration_ms
            stats.average_duration_ms = stats.total_duration_ms / stats.execution_count

        if metric.success:
            stats.success_count += 1
        else:
            stats.failure_count += 1
            if metric.error_message:
                stats.errors.append(metric.error_message)
                # Keep only last 10 errors
                if len(stats.errors) > 10:
                    stats.errors = stats.errors[-10:]

    def _identify_performance_issues(self, metrics: List[FlowMetrics]) -> List[str]:
        """Identify performance issues from metrics.

        Args:
            metrics: List of metrics to analyze

        Returns:
            List of performance issue descriptions
        """
        issues = []

        if not metrics:
            return issues

        # Check for slow operations (>5 seconds)
        slow_ops = [m for m in metrics if m.duration_ms and m.duration_ms > 5000]
        if slow_ops:
            issues.append(f"Found {len(slow_ops)} slow operations (>5s)")

        # Check for high failure rate
        failed_ops = [m for m in metrics if not m.success]
        if len(failed_ops) > len(metrics) * 0.1:  # >10% failure rate
            issues.append(
                f"High failure rate: {len(failed_ops)}/{len(metrics)} operations failed"
            )

        # Check for repeated failures of same operation
        operation_failures = defaultdict(int)
        for metric in failed_ops:
            operation_failures[metric.operation_name] += 1

        for op_name, failure_count in operation_failures.items():
            if failure_count >= 3:
                issues.append(f"Operation '{op_name}' failed {failure_count} times")

        return issues


class FlowDebugger:
    """Provides debugging capabilities for Virtual Agora flows."""

    def __init__(self, monitor: FlowMonitor):
        """Initialize debugger.

        Args:
            monitor: Flow monitor instance
        """
        self.monitor = monitor
        self.breakpoints: Dict[str, Callable] = {}
        self.watch_expressions: Dict[str, Callable] = {}
        self.debug_enabled = False

    def set_breakpoint(self, name: str, condition: Callable[[VirtualAgoraState], bool]):
        """Set a conditional breakpoint.

        Args:
            name: Breakpoint name
            condition: Function that returns True when breakpoint should trigger
        """
        self.breakpoints[name] = condition
        logger.debug(f"Set breakpoint: {name}")

    def remove_breakpoint(self, name: str):
        """Remove a breakpoint.

        Args:
            name: Breakpoint name
        """
        if name in self.breakpoints:
            del self.breakpoints[name]
            logger.debug(f"Removed breakpoint: {name}")

    def add_watch(self, name: str, expression: Callable[[VirtualAgoraState], Any]):
        """Add a watch expression.

        Args:
            name: Watch name
            expression: Function that extracts value from state
        """
        self.watch_expressions[name] = expression
        logger.debug(f"Added watch: {name}")

    def remove_watch(self, name: str):
        """Remove a watch expression.

        Args:
            name: Watch name
        """
        if name in self.watch_expressions:
            del self.watch_expressions[name]
            logger.debug(f"Removed watch: {name}")

    def check_breakpoints(
        self, state: VirtualAgoraState, current_node: Optional[str] = None
    ) -> List[str]:
        """Check if any breakpoints are triggered.

        Args:
            state: Current state
            current_node: Currently executing node

        Returns:
            List of triggered breakpoint names
        """
        if not self.debug_enabled:
            return []

        triggered = []

        for name, condition in self.breakpoints.items():
            try:
                if condition(state):
                    triggered.append(name)
                    logger.warning(f"Breakpoint triggered: {name}")

                    # Create debug snapshot when breakpoint triggers
                    self.monitor.create_debug_snapshot(state, current_node)

            except Exception as e:
                logger.error(f"Error evaluating breakpoint '{name}': {e}")

        return triggered

    def evaluate_watches(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Evaluate all watch expressions.

        Args:
            state: Current state

        Returns:
            Dictionary of watch names to values
        """
        if not self.debug_enabled:
            return {}

        results = {}

        for name, expression in self.watch_expressions.items():
            try:
                results[name] = expression(state)
            except Exception as e:
                results[name] = f"ERROR: {e}"
                logger.error(f"Error evaluating watch '{name}': {e}")

        return results

    def enable_debug_mode(self):
        """Enable debug mode."""
        self.debug_enabled = True
        logger.debug("Debug mode enabled")

    def disable_debug_mode(self):
        """Disable debug mode."""
        self.debug_enabled = False
        logger.debug("Debug mode disabled")

    def get_debug_status(self) -> Dict[str, Any]:
        """Get current debug status.

        Returns:
            Debug status dictionary
        """
        return {
            "debug_enabled": self.debug_enabled,
            "breakpoints": list(self.breakpoints.keys()),
            "watch_expressions": list(self.watch_expressions.keys()),
            "recent_snapshots": len(self.monitor.debug_snapshots),
        }


def create_flow_monitor(max_metrics_history: int = 1000) -> FlowMonitor:
    """Create a flow monitor instance.

    Args:
        max_metrics_history: Maximum metrics to keep in memory

    Returns:
        Flow monitor instance
    """
    return FlowMonitor(max_metrics_history)


def create_flow_debugger(monitor: FlowMonitor) -> FlowDebugger:
    """Create a flow debugger instance.

    Args:
        monitor: Flow monitor to use

    Returns:
        Flow debugger instance
    """
    return FlowDebugger(monitor)
