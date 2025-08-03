"""V13NodeWrapper Exception Handling Tests.

This module tests the critical fix for V13NodeWrapper GraphInterrupt propagation.
Without this fix, GraphInterrupt exceptions are caught and wrapped, breaking
the user input flow.

CRITICAL: This validates the fix implemented in node_registry.py that allows
GraphInterrupt to propagate while still wrapping other exceptions.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from virtual_agora.flow.node_registry import V13NodeWrapper
from virtual_agora.flow.nodes.base import NodeExecutionError
from virtual_agora.state.schema import VirtualAgoraState
from datetime import datetime


class TestV13NodeWrapperExceptionHandling:
    """Test V13NodeWrapper exception handling patterns."""

    def create_test_state(self) -> VirtualAgoraState:
        """Create minimal test state."""
        return {
            "session_id": "test_session",
            "current_phase": 0,
            "current_round": 0,
            "start_time": datetime.now(),
        }

    def test_graphinterrupt_propagation(self):
        """Test that GraphInterrupt exceptions propagate through wrapper."""

        # Create test interrupt
        interrupt_data = {
            "type": "agenda_approval",
            "proposed_agenda": ["Topic 1", "Topic 2"],
            "message": "Please approve the agenda",
        }

        test_interrupt = GraphInterrupt(
            (
                Interrupt(
                    value=interrupt_data,
                    resumable=True,
                    ns=["agenda_approval:test"],
                    when="during",
                ),
            )
        )

        # Create mock node function that raises GraphInterrupt
        def mock_node_function(state):
            raise test_interrupt

        # Create wrapper
        wrapper = V13NodeWrapper(mock_node_function, "test_node")
        test_state = self.create_test_state()

        # GraphInterrupt should propagate, not be wrapped
        with pytest.raises(GraphInterrupt) as exc_info:
            wrapper.execute(test_state)

        # Validate the exact same GraphInterrupt was propagated
        assert exc_info.value == test_interrupt
        assert exc_info.value.args[0][0].value == interrupt_data

    def test_regular_exception_wrapping(self):
        """Test that regular exceptions are still wrapped in NodeExecutionError."""

        # Create mock node function that raises regular exception
        def mock_node_function(state):
            raise ValueError("Test error message")

        # Create wrapper
        wrapper = V13NodeWrapper(mock_node_function, "test_node")
        test_state = self.create_test_state()

        # Regular exception should be wrapped
        with pytest.raises(NodeExecutionError) as exc_info:
            wrapper.execute(test_state)

        # Validate wrapping
        assert "Wrapped node 'test_node' execution failed" in str(exc_info.value)
        assert "Test error message" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_successful_execution_passthrough(self):
        """Test that successful execution passes through normally."""

        expected_result = {
            "messages": [{"content": "Test message", "speaker_id": "test_agent"}],
            "current_phase": 1,
        }

        # Create mock node function that succeeds
        def mock_node_function(state):
            return expected_result

        # Create wrapper
        wrapper = V13NodeWrapper(mock_node_function, "test_node")
        test_state = self.create_test_state()

        # Should return result normally
        result = wrapper.execute(test_state)

        assert result == expected_result

    def test_different_graphinterrupt_types(self):
        """Test GraphInterrupt propagation for different interrupt types."""

        interrupt_types = [
            {"type": "agenda_approval", "data": {"proposed_agenda": ["Topic 1"]}},
            {"type": "topic_conclusion", "data": {"current_topic": "Test Topic"}},
            {"type": "periodic_stop", "data": {"round_number": 5}},
        ]

        for interrupt_config in interrupt_types:
            # Create specific interrupt
            test_interrupt = GraphInterrupt(
                (
                    Interrupt(
                        value=interrupt_config,
                        resumable=True,
                        ns=[f"{interrupt_config['type']}:test"],
                        when="during",
                    ),
                )
            )

            # Create mock node function that raises this interrupt
            def mock_node_function(state):
                raise test_interrupt

            # Create wrapper
            wrapper = V13NodeWrapper(
                mock_node_function, f"test_node_{interrupt_config['type']}"
            )
            test_state = self.create_test_state()

            # GraphInterrupt should propagate
            with pytest.raises(GraphInterrupt) as exc_info:
                wrapper.execute(test_state)

            # Validate correct interrupt was propagated
            assert exc_info.value == test_interrupt
            assert exc_info.value.args[0][0].value["type"] == interrupt_config["type"]

    def test_nested_exception_handling(self):
        """Test exception handling with nested exceptions."""

        # Create mock node function that raises nested exception
        def mock_node_function(state):
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise RuntimeError("Outer error") from e

        # Create wrapper
        wrapper = V13NodeWrapper(mock_node_function, "test_node")
        test_state = self.create_test_state()

        # Should wrap the outer exception
        with pytest.raises(NodeExecutionError) as exc_info:
            wrapper.execute(test_state)

        # Validate proper exception chaining
        assert "Wrapped node 'test_node' execution failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert isinstance(exc_info.value.__cause__.__cause__, ValueError)

    def test_wrapper_callable_interface(self):
        """Test that wrapper is callable (for LangGraph integration)."""

        expected_result = {"test": "result"}

        # Create mock node function
        def mock_node_function(state):
            return expected_result

        # Create wrapper
        wrapper = V13NodeWrapper(mock_node_function, "test_node")
        test_state = self.create_test_state()

        # Should be callable directly
        result = wrapper(test_state)
        assert result == expected_result

        # Callable should have same exception behavior
        def mock_interrupt_function(state):
            raise GraphInterrupt(
                (
                    Interrupt(
                        value={"type": "test"},
                        resumable=True,
                        ns=["test:ns"],
                        when="during",
                    ),
                )
            )

        interrupt_wrapper = V13NodeWrapper(mock_interrupt_function, "interrupt_node")

        with pytest.raises(GraphInterrupt):
            interrupt_wrapper(test_state)

    def test_validation_function_integration(self):
        """Test wrapper with validation function."""

        def mock_node_function(state):
            return {"result": "success"}

        def mock_validation_function(state):
            return state.get("current_phase", 0) >= 0

        # Create wrapper with validation
        wrapper = V13NodeWrapper(
            mock_node_function,
            "test_node",
            validation_function=mock_validation_function,
        )

        test_state = self.create_test_state()

        # Validation should work normally
        assert wrapper.validate_preconditions(test_state) == True

        # Execution should work normally
        result = wrapper.execute(test_state)
        assert result == {"result": "success"}

        # GraphInterrupt should still propagate even with validation
        def mock_interrupt_function(state):
            raise GraphInterrupt(
                (
                    Interrupt(
                        value={"type": "test"},
                        resumable=True,
                        ns=["test:ns"],
                        when="during",
                    ),
                )
            )

        interrupt_wrapper = V13NodeWrapper(
            mock_interrupt_function,
            "interrupt_node",
            validation_function=mock_validation_function,
        )

        with pytest.raises(GraphInterrupt):
            interrupt_wrapper.execute(test_state)

    def test_node_name_reporting(self):
        """Test that node names are correctly reported in errors."""

        def mock_node_function(state):
            raise ValueError("Test error")

        wrapper = V13NodeWrapper(mock_node_function, "specific_node_name")
        test_state = self.create_test_state()

        with pytest.raises(NodeExecutionError) as exc_info:
            wrapper.execute(test_state)

        # Node name should be in error message
        assert "specific_node_name" in str(exc_info.value)

        # get_node_name should return descriptive name
        assert "V13Wrapper(specific_node_name)" == wrapper.get_node_name()

    def test_production_scenario_simulation(self):
        """Test actual production scenario where user input is required."""

        # Simulate agenda approval node that needs user input
        def agenda_approval_node(state):
            # This simulates the actual pattern used in nodes_v13.py
            interrupt_data = {
                "type": "agenda_approval",
                "proposed_agenda": state.get("proposed_topics", ["Default Topic"]),
                "message": "Please review and approve the proposed discussion agenda.",
                "options": ["approve", "edit", "reorder", "reject"],
            }

            # This would normally call interrupt() function from LangGraph
            raise GraphInterrupt(
                (
                    Interrupt(
                        value=interrupt_data,
                        resumable=True,
                        ns=["agenda_approval:production_test"],
                        when="during",
                    ),
                )
            )

        # Create wrapper (this is the critical component being tested)
        wrapper = V13NodeWrapper(agenda_approval_node, "agenda_approval")

        test_state = self.create_test_state()
        test_state["proposed_topics"] = ["AI Ethics", "Climate Policy"]

        # GraphInterrupt should propagate for HITL processing
        with pytest.raises(GraphInterrupt) as exc_info:
            wrapper.execute(test_state)

        # Validate interrupt data is preserved
        interrupt = exc_info.value.args[0][0]
        assert interrupt.value["type"] == "agenda_approval"
        assert interrupt.value["proposed_agenda"] == ["AI Ethics", "Climate Policy"]
        assert "approve" in interrupt.value["options"]

        # This GraphInterrupt will be caught by StreamCoordinator and processed
        # by process_interrupt_recursive in main.py, then execution will resume
