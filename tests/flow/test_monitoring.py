"""Tests for flow monitoring and debugging."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

from virtual_agora.flow.monitoring import (
    FlowMonitor,
    FlowDebugger,
    FlowMetrics,
    NodeExecutionInfo,
    PhaseTransitionInfo,
    DebugSnapshot,
    create_flow_monitor,
    create_flow_debugger,
)


class TestFlowMonitor:
    """Test flow monitoring functionality."""

    def setup_method(self):
        """Set up test method."""
        self.monitor = FlowMonitor(max_metrics_history=100)

        # Sample state for testing
        self.test_state = {
            "session_id": "test-session-123",
            "current_phase": 2,
            "phase_start_time": datetime.now(),
            "agents": {
                "agent1": {"id": "agent1", "model": "gpt-4o"},
                "agent2": {"id": "agent2", "model": "gpt-4o"},
            },
            "messages": [
                {
                    "id": "msg1",
                    "speaker_id": "agent1",
                    "content": "Test message 1",
                    "timestamp": datetime.now(),
                },
                {
                    "id": "msg2",
                    "speaker_id": "agent2",
                    "content": "Test message 2",
                    "timestamp": datetime.now(),
                },
            ],
            "agenda_topics": ["Topic 1", "Topic 2"],
            "active_topic": "Topic 1",
            "vote_history": [
                {
                    "vote_type": "continue",
                    "result": "passed",
                    "start_time": datetime.now(),
                }
            ],
            "current_round": 1,
            "flow_control": {"max_rounds_per_topic": 10},
            "hitl_state": {"awaiting_approval": False, "approval_type": None},
        }

    def test_start_session(self):
        """Test session start monitoring."""
        session_id = "test-session"
        self.monitor.start_session(session_id)

        assert self.monitor.session_start_time is not None
        assert isinstance(self.monitor.session_start_time, datetime)

    def test_operation_monitoring(self):
        """Test operation start and end monitoring."""
        operation_name = "test_operation"
        metadata = {"key": "value"}

        # Start operation
        operation_id = self.monitor.start_operation(operation_name, metadata)

        assert operation_id in self.monitor.active_operations
        assert (
            self.monitor.active_operations[operation_id].operation_name
            == operation_name
        )
        assert self.monitor.active_operations[operation_id].metadata == metadata

        # End operation successfully
        self.monitor.end_operation(operation_id, success=True)

        assert operation_id not in self.monitor.active_operations
        assert len(self.monitor.metrics_history) == 1

        metric = self.monitor.metrics_history[0]
        assert metric.operation_name == operation_name
        assert metric.success is True
        assert metric.duration_ms is not None
        assert metric.duration_ms >= 0

    def test_operation_monitoring_with_failure(self):
        """Test operation monitoring with failure."""
        operation_name = "failing_operation"
        error_message = "Test error"

        operation_id = self.monitor.start_operation(operation_name)
        self.monitor.end_operation(
            operation_id, success=False, error_message=error_message
        )

        metric = self.monitor.metrics_history[0]
        assert metric.success is False
        assert metric.error_message == error_message

    def test_operation_context_manager(self):
        """Test operation monitoring using context manager."""
        operation_name = "context_operation"

        with self.monitor.monitor_operation(operation_name) as operation_id:
            # Simulate some work
            time.sleep(0.01)
            assert operation_id in self.monitor.active_operations

        # Should be completed now
        assert operation_id not in self.monitor.active_operations
        assert len(self.monitor.metrics_history) == 1

        metric = self.monitor.metrics_history[0]
        assert metric.success is True
        assert metric.duration_ms >= 10  # At least 10ms from sleep

    def test_operation_context_manager_with_exception(self):
        """Test operation monitoring context manager with exception."""
        operation_name = "failing_context_operation"

        with pytest.raises(ValueError):
            with self.monitor.monitor_operation(operation_name):
                raise ValueError("Test exception")

        # Should record the failure
        assert len(self.monitor.metrics_history) == 1

        metric = self.monitor.metrics_history[0]
        assert metric.success is False
        assert "Test exception" in metric.error_message

    def test_phase_transition_recording(self):
        """Test phase transition recording."""
        from_phase, to_phase = 1, 2
        duration_ms = 150.0

        # Record first transition
        self.monitor.record_phase_transition(from_phase, to_phase, duration_ms)

        transition_key = f"{from_phase}->{to_phase}"
        assert transition_key in self.monitor.phase_transition_stats

        stats = self.monitor.phase_transition_stats[transition_key]
        assert stats.transition_count == 1
        assert stats.average_duration_ms == duration_ms
        assert stats.success_rate == 1.0

        # Record second transition with different duration
        duration_ms2 = 200.0
        self.monitor.record_phase_transition(
            from_phase, to_phase, duration_ms2, success=False
        )

        stats = self.monitor.phase_transition_stats[transition_key]
        assert stats.transition_count == 2
        assert stats.average_duration_ms == (duration_ms + duration_ms2) / 2
        assert stats.success_rate == 0.5  # One success, one failure

    def test_debug_snapshot_creation(self):
        """Test debug snapshot creation."""
        current_node = "test_node"

        # Add some metrics first
        operation_id = self.monitor.start_operation("test_op")
        self.monitor.end_operation(operation_id, success=True)

        snapshot = self.monitor.create_debug_snapshot(self.test_state, current_node)

        assert isinstance(snapshot, DebugSnapshot)
        assert snapshot.session_id == "test-session-123"
        assert snapshot.current_phase == 2
        assert snapshot.current_node == current_node
        assert snapshot.message_count == 2
        assert snapshot.votes_cast == 1
        assert len(snapshot.active_agents) == 2
        assert "agent1" in snapshot.active_agents
        assert "agent2" in snapshot.active_agents

        # Check state summary
        assert snapshot.state_summary["agenda_topics"] == 2
        assert snapshot.state_summary["active_topic"] == "Topic 1"
        assert snapshot.state_summary["current_round"] == 1

        # Should be stored in monitor
        assert len(self.monitor.debug_snapshots) == 1

    def test_performance_report_no_data(self):
        """Test performance report with no data."""
        report = self.monitor.get_performance_report()

        assert report["status"] == "no_data"
        assert "message" in report

    def test_performance_report_with_data(self):
        """Test performance report with metrics data."""
        self.monitor.start_session("test-session")

        # Add some operations
        for i in range(5):
            with self.monitor.monitor_operation(f"operation_{i % 2}"):
                time.sleep(0.001)  # Small delay

        # Add one failure
        op_id = self.monitor.start_operation("failing_op")
        self.monitor.end_operation(op_id, success=False, error_message="Test error")

        report = self.monitor.get_performance_report()

        assert report["status"] == "active"
        assert report["total_operations"] == 6
        assert report["successful_operations"] == 5
        assert report["failed_operations"] == 1
        assert report["overall_success_rate"] == 5 / 6
        assert report["uptime_minutes"] > 0

        # Check operation stats
        assert "operation_0" in report["operation_stats"]
        assert "operation_1" in report["operation_stats"]

        # Check node stats
        assert len(report["node_stats"]) > 0

    def test_error_summary_no_errors(self):
        """Test error summary with no errors."""
        # Add successful operations
        with self.monitor.monitor_operation("success_op"):
            pass

        summary = self.monitor.get_error_summary()

        assert summary["status"] == "healthy"
        assert summary["total_errors"] == 0

    def test_error_summary_with_errors(self):
        """Test error summary with errors."""
        # Add some failures
        for i in range(3):
            op_id = self.monitor.start_operation(f"failing_op_{i}")
            self.monitor.end_operation(op_id, success=False, error_message=f"Error {i}")

        # Add repeated error
        for i in range(2):
            op_id = self.monitor.start_operation("repeated_fail")
            self.monitor.end_operation(
                op_id, success=False, error_message="Repeated error"
            )

        summary = self.monitor.get_error_summary()

        assert summary["status"] == "errors_detected"
        assert summary["total_errors"] == 5
        assert summary["unique_error_types"] >= 3
        assert summary["error_rate"] == 1.0  # All operations failed

        # Check most frequent errors
        most_frequent = summary["most_frequent_errors"]
        assert len(most_frequent) > 0
        assert most_frequent[0][0] == "Repeated error"  # Most frequent
        assert most_frequent[0][1] == 2  # Occurred twice

    def test_export_debug_data(self):
        """Test debug data export."""
        self.monitor.start_session("test-session")

        # Add some data
        with self.monitor.monitor_operation("test_op"):
            pass

        self.monitor.record_phase_transition(1, 2, 100.0)
        self.monitor.create_debug_snapshot(self.test_state)

        # Export with snapshots
        export_data = self.monitor.export_debug_data(include_snapshots=True)

        assert "export_timestamp" in export_data
        assert "session_start_time" in export_data
        assert "metrics_history" in export_data
        assert "node_stats" in export_data
        assert "phase_transition_stats" in export_data
        assert "debug_snapshots" in export_data
        assert "performance_report" in export_data
        assert "error_summary" in export_data

        assert len(export_data["metrics_history"]) == 1
        assert len(export_data["debug_snapshots"]) == 1

        # Export without snapshots
        export_data_no_snapshots = self.monitor.export_debug_data(
            include_snapshots=False
        )
        assert "debug_snapshots" not in export_data_no_snapshots

    def test_node_stats_update(self):
        """Test node statistics updates."""
        node_name = "test_node"

        # Execute same node multiple times
        for i in range(3):
            with self.monitor.monitor_operation(node_name):
                time.sleep(0.001)

        # Add one failure
        op_id = self.monitor.start_operation(node_name)
        self.monitor.end_operation(op_id, success=False, error_message="Node error")

        assert node_name in self.monitor.node_stats
        stats = self.monitor.node_stats[node_name]

        assert stats.execution_count == 4
        assert stats.success_count == 3
        assert stats.failure_count == 1
        assert stats.average_duration_ms > 0
        assert len(stats.errors) == 1
        assert "Node error" in stats.errors

    def test_max_metrics_history(self):
        """Test metrics history size limit."""
        monitor = FlowMonitor(max_metrics_history=3)

        # Add more metrics than the limit
        for i in range(5):
            with monitor.monitor_operation(f"op_{i}"):
                pass

        # Should only keep the last 3
        assert len(monitor.metrics_history) == 3

        # Should be the most recent operations
        operation_names = [m.operation_name for m in monitor.metrics_history]
        assert "op_2" in operation_names
        assert "op_3" in operation_names
        assert "op_4" in operation_names
        assert "op_0" not in operation_names
        assert "op_1" not in operation_names


class TestFlowDebugger:
    """Test flow debugging functionality."""

    def setup_method(self):
        """Set up test method."""
        self.monitor = FlowMonitor()
        self.debugger = FlowDebugger(self.monitor)

        self.test_state = {
            "session_id": "debug-session",
            "current_phase": 2,
            "agents": {"agent1": {}},
            "messages": [{"speaker_id": "agent1"}],
            "current_round": 5,
        }

    def test_set_and_remove_breakpoint(self):
        """Test breakpoint management."""
        breakpoint_name = "phase_breakpoint"
        condition = lambda state: state["current_phase"] == 2

        # Set breakpoint
        self.debugger.set_breakpoint(breakpoint_name, condition)
        assert breakpoint_name in self.debugger.breakpoints

        # Remove breakpoint
        self.debugger.remove_breakpoint(breakpoint_name)
        assert breakpoint_name not in self.debugger.breakpoints

    def test_add_and_remove_watch(self):
        """Test watch expression management."""
        watch_name = "round_watch"
        expression = lambda state: state.get("current_round", 0)

        # Add watch
        self.debugger.add_watch(watch_name, expression)
        assert watch_name in self.debugger.watch_expressions

        # Remove watch
        self.debugger.remove_watch(watch_name)
        assert watch_name not in self.debugger.watch_expressions

    def test_check_breakpoints_disabled(self):
        """Test breakpoint checking when debug is disabled."""
        condition = lambda state: True  # Always trigger
        self.debugger.set_breakpoint("always_trigger", condition)

        # Debug is disabled by default
        triggered = self.debugger.check_breakpoints(self.test_state)
        assert len(triggered) == 0

    def test_check_breakpoints_enabled(self):
        """Test breakpoint checking when debug is enabled."""
        self.debugger.enable_debug_mode()

        # Set breakpoints
        always_trigger = lambda state: True
        never_trigger = lambda state: False
        phase_trigger = lambda state: state["current_phase"] == 2

        self.debugger.set_breakpoint("always", always_trigger)
        self.debugger.set_breakpoint("never", never_trigger)
        self.debugger.set_breakpoint("phase", phase_trigger)

        triggered = self.debugger.check_breakpoints(self.test_state)

        assert "always" in triggered
        assert "phase" in triggered
        assert "never" not in triggered
        assert len(triggered) == 2

        # Should create debug snapshots (one per triggered breakpoint)
        assert len(self.monitor.debug_snapshots) == 2

    def test_check_breakpoints_with_error(self):
        """Test breakpoint checking with error in condition."""
        self.debugger.enable_debug_mode()

        # Condition that will raise an exception
        error_condition = lambda state: state["nonexistent_key"]

        self.debugger.set_breakpoint("error_bp", error_condition)

        # Should not crash, should handle error gracefully
        triggered = self.debugger.check_breakpoints(self.test_state)
        assert len(triggered) == 0

    def test_evaluate_watches_disabled(self):
        """Test watch evaluation when debug is disabled."""
        expression = lambda state: state.get("current_round", 0)
        self.debugger.add_watch("round", expression)

        # Debug is disabled by default
        results = self.debugger.evaluate_watches(self.test_state)
        assert len(results) == 0

    def test_evaluate_watches_enabled(self):
        """Test watch evaluation when debug is enabled."""
        self.debugger.enable_debug_mode()

        # Add watch expressions
        round_expr = lambda state: state.get("current_round", 0)
        phase_expr = lambda state: f"Phase: {state['current_phase']}"
        agent_count_expr = lambda state: len(state.get("agents", {}))

        self.debugger.add_watch("round", round_expr)
        self.debugger.add_watch("phase", phase_expr)
        self.debugger.add_watch("agent_count", agent_count_expr)

        results = self.debugger.evaluate_watches(self.test_state)

        assert results["round"] == 5
        assert results["phase"] == "Phase: 2"
        assert results["agent_count"] == 1

    def test_evaluate_watches_with_error(self):
        """Test watch evaluation with error in expression."""
        self.debugger.enable_debug_mode()

        # Expression that will raise an exception
        error_expr = lambda state: state["nonexistent_key"]["nested"]

        self.debugger.add_watch("error_watch", error_expr)

        results = self.debugger.evaluate_watches(self.test_state)

        assert "error_watch" in results
        assert "ERROR:" in str(results["error_watch"])

    def test_debug_mode_toggle(self):
        """Test enabling and disabling debug mode."""
        # Initially disabled
        assert not self.debugger.debug_enabled

        # Enable
        self.debugger.enable_debug_mode()
        assert self.debugger.debug_enabled

        # Disable
        self.debugger.disable_debug_mode()
        assert not self.debugger.debug_enabled

    def test_get_debug_status(self):
        """Test debug status reporting."""
        self.debugger.enable_debug_mode()

        # Add some breakpoints and watches
        self.debugger.set_breakpoint("bp1", lambda s: True)
        self.debugger.set_breakpoint("bp2", lambda s: False)
        self.debugger.add_watch("watch1", lambda s: s.get("phase", 0))

        # Create some snapshots
        self.monitor.create_debug_snapshot(self.test_state)
        self.monitor.create_debug_snapshot(self.test_state)

        status = self.debugger.get_debug_status()

        assert status["debug_enabled"] is True
        assert len(status["breakpoints"]) == 2
        assert "bp1" in status["breakpoints"]
        assert "bp2" in status["breakpoints"]
        assert len(status["watch_expressions"]) == 1
        assert "watch1" in status["watch_expressions"]
        assert status["recent_snapshots"] == 2


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_flow_monitor(self):
        """Test flow monitor factory function."""
        monitor = create_flow_monitor()
        assert isinstance(monitor, FlowMonitor)
        assert monitor.max_metrics_history == 1000

        monitor_custom = create_flow_monitor(max_metrics_history=500)
        assert monitor_custom.max_metrics_history == 500

    def test_create_flow_debugger(self):
        """Test flow debugger factory function."""
        monitor = create_flow_monitor()
        debugger = create_flow_debugger(monitor)

        assert isinstance(debugger, FlowDebugger)
        assert debugger.monitor is monitor
