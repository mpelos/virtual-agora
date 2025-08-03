"""Tests for Virtual Agora flow node base classes.

This module provides comprehensive tests for the base node architecture
introduced in Step 2.1, ensuring 100% code coverage and proper functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

from src.virtual_agora.flow.nodes.base import (
    FlowNode,
    HITLNode,
    AgentOrchestratorNode,
    NodeDependencies,
    NodeExecutionContext,
    NodeValidationError,
    NodeExecutionError,
)
from src.virtual_agora.state.schema import VirtualAgoraState
from src.virtual_agora.agents.llm_agent import LLMAgent


class TestFlowNode:
    """Test suite for FlowNode abstract base class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""

        # Create concrete implementation of FlowNode for testing
        class ConcreteFlowNode(FlowNode):
            def __init__(self, should_fail=False, validation_should_fail=False):
                super().__init__()
                self.should_fail = should_fail
                self.validation_should_fail = validation_should_fail
                self.execute_called = False
                self.validate_called = False

            def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
                self.execute_called = True
                if self.should_fail:
                    raise RuntimeError("Test execution failure")
                return {"test_key": "test_value", "updated": True}

            def validate_preconditions(self, state: VirtualAgoraState) -> bool:
                self.validate_called = True
                if self.validation_should_fail:
                    self._validation_errors.append("Test validation error")
                    return False
                return True

        self.ConcreteFlowNode = ConcreteFlowNode
        self.test_state = {
            "session_id": "test_session",
            "current_round": 1,
            "active_topic": "Test Topic",
            "messages": [],
            "error_count": 0,
        }

    def test_flow_node_initialization(self):
        """Test FlowNode initialization."""
        dependencies = Mock()
        node = self.ConcreteFlowNode()

        assert node.dependencies is None
        assert node._validation_errors == []
        assert node._execution_metadata == {}

        # Test with dependencies
        node_with_deps = self.ConcreteFlowNode()
        node_with_deps.dependencies = dependencies
        assert node_with_deps.dependencies is dependencies

    def test_get_node_name(self):
        """Test get_node_name method."""
        node = self.ConcreteFlowNode()
        assert node.get_node_name() == "ConcreteFlowNode"

    def test_get_validation_errors(self):
        """Test get_validation_errors method."""
        node = self.ConcreteFlowNode()

        # Initially empty
        assert node.get_validation_errors() == []

        # Add some errors
        node._validation_errors = ["Error 1", "Error 2"]
        errors = node.get_validation_errors()
        assert errors == ["Error 1", "Error 2"]

        # Ensure it returns a copy
        errors.append("Error 3")
        assert node._validation_errors == ["Error 1", "Error 2"]

    def test_handle_error(self):
        """Test error handling functionality."""
        node = self.ConcreteFlowNode()
        test_error = RuntimeError("Test error")

        result = node.handle_error(test_error, self.test_state)

        # Check returned error state
        assert result["last_error"] == "Test error"
        assert result["error_count"] == 1
        assert result["error_source"] == "ConcreteFlowNode"
        assert "error_timestamp" in result

        # Check execution metadata
        assert "last_error" in node._execution_metadata
        error_info = node._execution_metadata["last_error"]
        assert error_info["error_type"] == "RuntimeError"
        assert error_info["error_message"] == "Test error"
        assert error_info["node_name"] == "ConcreteFlowNode"

    def test_safe_execute_success(self):
        """Test successful safe execution."""
        node = self.ConcreteFlowNode()

        result = node.safe_execute(self.test_state)

        # Verify execution occurred
        assert node.validate_called
        assert node.execute_called

        # Check result
        assert result["test_key"] == "test_value"
        assert result["updated"] is True

        # Check execution metadata
        assert "last_execution" in node._execution_metadata
        exec_info = node._execution_metadata["last_execution"]
        assert exec_info["success"] is True
        assert exec_info["execution_time"] >= 0

    def test_safe_execute_validation_failure(self):
        """Test safe execution with validation failure."""
        node = self.ConcreteFlowNode(validation_should_fail=True)

        result = node.safe_execute(self.test_state)

        # Verify validation was called but execution was not
        assert node.validate_called
        assert not node.execute_called

        # Check error result
        assert "last_error" in result
        assert "Precondition validation failed" in result["last_error"]
        assert result["error_count"] == 1
        assert result["error_source"] == "ConcreteFlowNode"

    def test_safe_execute_execution_failure(self):
        """Test safe execution with execution failure."""
        node = self.ConcreteFlowNode(should_fail=True)

        result = node.safe_execute(self.test_state)

        # Verify both validation and execution were called
        assert node.validate_called
        assert node.execute_called

        # Check error result
        assert "last_error" in result
        assert "Test execution failure" in result["last_error"]
        assert result["error_count"] == 1
        assert result["error_source"] == "ConcreteFlowNode"

    def test_validate_required_keys_success(self):
        """Test successful required keys validation."""
        node = self.ConcreteFlowNode()

        result = node._validate_required_keys(
            self.test_state, ["session_id", "current_round"]
        )

        assert result is True
        assert node._validation_errors == []

    def test_validate_required_keys_failure(self):
        """Test required keys validation with missing keys."""
        node = self.ConcreteFlowNode()

        result = node._validate_required_keys(
            self.test_state, ["session_id", "missing_key", "another_missing"]
        )

        assert result is False
        assert len(node._validation_errors) == 2
        assert "Missing required key: missing_key" in node._validation_errors
        assert "Missing required key: another_missing" in node._validation_errors

    def test_validate_state_type_success(self):
        """Test successful state type validation."""
        node = self.ConcreteFlowNode()

        result = node._validate_state_type(self.test_state, "current_round", int)

        assert result is True
        assert node._validation_errors == []

    def test_validate_state_type_failure(self):
        """Test state type validation with wrong type."""
        node = self.ConcreteFlowNode()

        result = node._validate_state_type(self.test_state, "session_id", int)

        assert result is False
        assert len(node._validation_errors) == 1
        assert "expected type int, got str" in node._validation_errors[0]

    def test_validate_state_type_missing_key(self):
        """Test state type validation with missing key."""
        node = self.ConcreteFlowNode()

        result = node._validate_state_type(self.test_state, "missing_key", str)

        assert result is True  # Missing keys are allowed
        assert node._validation_errors == []


class TestHITLNode:
    """Test suite for HITLNode specialized base class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""

        # Create concrete implementation of HITLNode for testing
        class ConcreteHITLNode(HITLNode):
            def __init__(self, interrupt_should_fail=False, process_should_fail=False):
                super().__init__()
                self.interrupt_should_fail = interrupt_should_fail
                self.process_should_fail = process_should_fail
                self.interrupt_payload_created = False
                self.user_input_processed = False

            def create_interrupt_payload(
                self, state: VirtualAgoraState
            ) -> Dict[str, Any]:
                self.interrupt_payload_created = True
                if self.interrupt_should_fail:
                    raise RuntimeError("Test interrupt payload failure")
                return {
                    "type": "test_interrupt",
                    "message": "Test interrupt message",
                    "current_round": state.get("current_round", 0),
                }

            def process_user_input(
                self, user_input: Dict[str, Any], state: VirtualAgoraState
            ) -> Dict[str, Any]:
                self.user_input_processed = True
                if self.process_should_fail:
                    raise RuntimeError("Test process user input failure")
                return {
                    "user_choice": user_input.get("choice", "unknown"),
                    "processed": True,
                }

            def validate_preconditions(self, state: VirtualAgoraState) -> bool:
                return isinstance(state, dict) and "session_id" in state

        self.ConcreteHITLNode = ConcreteHITLNode
        self.test_state = {
            "session_id": "test_session",
            "current_round": 2,
            "active_topic": "Test Topic",
            "current_phase": 1,
            "completed_topics": ["Topic 1"],
            "messages": [{"content": "test"}],
        }

    @patch("src.virtual_agora.flow.nodes.base.interrupt")
    def test_hitl_node_execute_success(self, mock_interrupt):
        """Test successful HITL node execution."""
        # Setup mock interrupt response
        mock_user_input = {"choice": "continue", "additional": "data"}
        mock_interrupt.return_value = mock_user_input

        node = self.ConcreteHITLNode()

        result = node.execute(self.test_state)

        # Verify interrupt was called with proper payload
        assert mock_interrupt.called
        interrupt_payload = mock_interrupt.call_args[0][0]

        # Check interrupt payload structure
        assert interrupt_payload["type"] == "test_interrupt"
        assert interrupt_payload["message"] == "Test interrupt message"
        assert interrupt_payload["current_round"] == 2
        assert interrupt_payload["node_name"] == "ConcreteHITLNode"
        assert "timestamp" in interrupt_payload
        assert "state_summary" in interrupt_payload

        # Check state summary
        state_summary = interrupt_payload["state_summary"]
        assert state_summary["current_round"] == 2
        assert state_summary["active_topic"] == "Test Topic"
        assert state_summary["current_phase"] == 1
        assert state_summary["session_id"] == "test_session"
        assert state_summary["completed_topics"] == 1
        assert state_summary["total_messages"] == 1

        # Check processing result
        assert node.interrupt_payload_created
        assert node.user_input_processed
        assert result["user_choice"] == "continue"
        assert result["processed"] is True

    @patch("src.virtual_agora.flow.nodes.base.interrupt")
    def test_hitl_node_execute_interrupt_failure(self, mock_interrupt):
        """Test HITL node execution with interrupt failure."""
        mock_interrupt.side_effect = RuntimeError("Interrupt failed")

        node = self.ConcreteHITLNode()

        result = node.execute(self.test_state)

        # Check error handling
        assert "last_error" in result
        assert "Interrupt failed" in result["last_error"]
        assert result["error_source"] == "ConcreteHITLNode"
        assert node.interrupt_payload_created
        assert not node.user_input_processed

    def test_hitl_node_execute_payload_creation_failure(self):
        """Test HITL node execution with payload creation failure."""
        node = self.ConcreteHITLNode(interrupt_should_fail=True)

        result = node.safe_execute(self.test_state)

        # Check error handling
        assert "last_error" in result
        assert "Test interrupt payload failure" in result["last_error"]
        assert not node.user_input_processed

    @patch("src.virtual_agora.flow.nodes.base.interrupt")
    def test_hitl_node_execute_process_failure(self, mock_interrupt):
        """Test HITL node execution with user input processing failure."""
        mock_interrupt.return_value = {"choice": "invalid"}
        node = self.ConcreteHITLNode(process_should_fail=True)

        result = node.execute(self.test_state)

        # Check error handling
        assert "last_error" in result
        assert "Test process user input failure" in result["last_error"]
        assert node.interrupt_payload_created
        assert node.user_input_processed

    def test_create_state_summary(self):
        """Test state summary creation."""
        node = self.ConcreteHITLNode()

        summary = node._create_state_summary(self.test_state)

        assert summary["current_round"] == 2
        assert summary["active_topic"] == "Test Topic"
        assert summary["current_phase"] == 1
        assert summary["session_id"] == "test_session"
        assert summary["completed_topics"] == 1
        assert summary["total_messages"] == 1

    def test_create_state_summary_with_defaults(self):
        """Test state summary creation with missing values."""
        minimal_state = {"session_id": "test"}
        node = self.ConcreteHITLNode()

        summary = node._create_state_summary(minimal_state)

        assert summary["current_round"] == 0
        assert summary["active_topic"] == "Unknown"
        assert summary["current_phase"] == 0
        assert summary["session_id"] == "test"
        assert summary["completed_topics"] == 0
        assert summary["total_messages"] == 0

    def test_validate_interrupt_payload_success(self):
        """Test successful interrupt payload validation."""
        node = self.ConcreteHITLNode()

        valid_payload = {
            "type": "test_interrupt",
            "message": "Test message",
            "additional": "data",
        }

        result = node.validate_interrupt_payload(valid_payload)

        assert result is True
        assert node._validation_errors == []

    def test_validate_interrupt_payload_missing_fields(self):
        """Test interrupt payload validation with missing required fields."""
        node = self.ConcreteHITLNode()

        invalid_payload = {
            "message": "Test message"
            # Missing 'type' field
        }

        result = node.validate_interrupt_payload(invalid_payload)

        assert result is False
        assert len(node._validation_errors) == 1
        assert "Missing interrupt field: type" in node._validation_errors


class TestAgentOrchestratorNode:
    """Test suite for AgentOrchestratorNode base class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock agents
        self.mock_agent1 = Mock()
        self.mock_agent1.agent_id = "test_agent_1"
        self.mock_agent1.return_value = {"content": "Agent 1 response"}

        self.mock_agent2 = Mock()
        self.mock_agent2.agent_id = "test_agent_2"
        self.mock_agent2.return_value = {"content": "Agent 2 response"}

        self.specialized_agents = {
            "agent1": self.mock_agent1,
            "agent2": self.mock_agent2,
        }

        self.discussing_agents = [self.mock_agent1, self.mock_agent2]

        # Create concrete implementation for testing
        class ConcreteAgentOrchestratorNode(AgentOrchestratorNode):
            def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
                return {"orchestrator_executed": True}

            def validate_preconditions(self, state: VirtualAgoraState) -> bool:
                return isinstance(state, dict)

        self.ConcreteAgentOrchestratorNode = ConcreteAgentOrchestratorNode
        self.test_state = {
            "session_id": "test_session",
            "current_round": 1,
            "messages": [],
        }

    def test_agent_orchestrator_initialization(self):
        """Test AgentOrchestratorNode initialization."""
        dependencies = Mock()

        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents, dependencies
        )

        assert node.specialized_agents is self.specialized_agents
        assert node.discussing_agents is self.discussing_agents
        assert node.dependencies is dependencies

    def test_validate_agent_availability_success(self):
        """Test successful agent availability validation."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        result = node.validate_agent_availability("agent1")

        assert result is True
        assert node._validation_errors == []

    def test_validate_agent_availability_missing_agent(self):
        """Test agent availability validation with missing agent."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        result = node.validate_agent_availability("missing_agent")

        assert result is False
        assert len(node._validation_errors) == 1
        assert "Required agent not available: missing_agent" in node._validation_errors

    def test_validate_agent_availability_non_callable(self):
        """Test agent availability validation with non-callable agent."""
        non_callable_agents = {"bad_agent": "not_an_agent"}
        node = self.ConcreteAgentOrchestratorNode(non_callable_agents, [])

        result = node.validate_agent_availability("bad_agent")

        assert result is False
        assert len(node._validation_errors) == 1
        assert "Agent bad_agent is not callable" in node._validation_errors

    def test_call_agent_with_retry_success(self):
        """Test successful agent call with retry."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        result = node.call_agent_with_retry(
            self.mock_agent1, self.test_state, "Test prompt"
        )

        assert result == {"content": "Agent 1 response"}
        assert self.mock_agent1.call_count == 1
        self.mock_agent1.assert_called_with(
            self.test_state, prompt="Test prompt", context_messages=None
        )

    def test_call_agent_with_retry_failure_then_success(self):
        """Test agent call with retry after initial failure."""
        # First call fails, second succeeds
        self.mock_agent1.side_effect = [
            RuntimeError("First attempt fails"),
            {"content": "Second attempt succeeds"},
        ]

        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = node.call_agent_with_retry(
                self.mock_agent1,
                self.test_state,
                "Test prompt",
                max_attempts=2,
                base_delay=0.1,
            )

        assert result == {"content": "Second attempt succeeds"}
        assert self.mock_agent1.call_count == 2

    def test_call_agent_with_retry_all_attempts_fail(self):
        """Test agent call when all retry attempts fail."""
        self.mock_agent1.side_effect = RuntimeError("All attempts fail")

        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = node.call_agent_with_retry(
                self.mock_agent1,
                self.test_state,
                "Test prompt",
                max_attempts=3,
                base_delay=0.1,
            )

        assert result is None
        assert self.mock_agent1.call_count == 3

    def test_call_agent_with_retry_returns_none(self):
        """Test agent call when agent returns None."""
        self.mock_agent1.return_value = None

        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        with patch("time.sleep"):
            result = node.call_agent_with_retry(
                self.mock_agent1, self.test_state, "Test prompt", max_attempts=2
            )

        assert result is None
        assert self.mock_agent1.call_count == 2

    def test_call_agent_with_context_messages(self):
        """Test agent call with context messages."""
        context_messages = [
            HumanMessage(content="Previous message"),
            AIMessage(content="Previous response"),
        ]

        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        result = node.call_agent_with_retry(
            self.mock_agent1,
            self.test_state,
            "Test prompt",
            context_messages=context_messages,
        )

        assert result == {"content": "Agent 1 response"}
        self.mock_agent1.assert_called_with(
            self.test_state, prompt="Test prompt", context_messages=context_messages
        )

    def test_validate_agent_response_success(self):
        """Test successful agent response validation."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        valid_response = {"content": "Valid response"}
        result = node.validate_agent_response(valid_response, "test_agent")

        assert result is True
        assert node._validation_errors == []

    def test_validate_agent_response_with_message_key(self):
        """Test agent response validation with message key."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        valid_response = {"message": "Valid response"}
        result = node.validate_agent_response(valid_response, "test_agent")

        assert result is True
        assert node._validation_errors == []

    def test_validate_agent_response_not_dict(self):
        """Test agent response validation with non-dict response."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        invalid_response = "not a dict"
        result = node.validate_agent_response(invalid_response, "test_agent")

        assert result is False
        assert len(node._validation_errors) == 1
        assert "response must be a dictionary" in node._validation_errors[0]

    def test_validate_agent_response_missing_content(self):
        """Test agent response validation with missing content/message."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        invalid_response = {"other_key": "value"}
        result = node.validate_agent_response(invalid_response, "test_agent")

        assert result is False
        assert len(node._validation_errors) == 1
        assert "response missing content or message" in node._validation_errors[0]

    def test_get_available_agents(self):
        """Test getting available agents list."""
        node = self.ConcreteAgentOrchestratorNode(
            self.specialized_agents, self.discussing_agents
        )

        agents = node.get_available_agents()

        # Check specialized agents
        assert "agent1" in agents
        assert "agent2" in agents
        assert agents["agent1"] == "Mock"  # Mock class name
        assert agents["agent2"] == "Mock"

        # Check discussing agents (should be included with their agent_id)
        assert "test_agent_1" in agents
        assert "test_agent_2" in agents


class TestNodeDependencies:
    """Test suite for NodeDependencies dependency injection system."""

    def test_node_dependencies_initialization(self):
        """Test NodeDependencies initialization."""
        mock_state_manager = Mock()
        mock_round_manager = Mock()
        mock_message_coordinator = Mock()
        mock_flow_state_manager = Mock()

        deps = NodeDependencies(
            state_manager=mock_state_manager,
            round_manager=mock_round_manager,
            message_coordinator=mock_message_coordinator,
            flow_state_manager=mock_flow_state_manager,
            checkpoint_interval=5,
        )

        assert deps.state_manager is mock_state_manager
        assert deps.round_manager is mock_round_manager
        assert deps.message_coordinator is mock_message_coordinator
        assert deps.flow_state_manager is mock_flow_state_manager
        assert deps.checkpoint_interval == 5
        assert not deps._validated
        assert deps._validation_errors == []

    def test_validate_success(self):
        """Test successful dependency validation."""
        deps = NodeDependencies(
            round_manager=Mock(), message_coordinator=Mock(), flow_state_manager=Mock()
        )

        result = deps.validate()

        assert result is True
        assert deps.is_validated()
        assert deps.get_validation_errors() == []

    def test_validate_missing_dependencies(self):
        """Test dependency validation with missing required dependencies."""
        deps = NodeDependencies(
            round_manager=Mock(),
            # Missing message_coordinator and flow_state_manager
        )

        result = deps.validate()

        assert result is False
        assert not deps.is_validated()

        errors = deps.get_validation_errors()
        assert len(errors) == 2
        assert "Missing required dependency: message_coordinator" in errors
        assert "Missing required dependency: flow_state_manager" in errors

    def test_get_validation_errors_returns_copy(self):
        """Test that get_validation_errors returns a copy."""
        deps = NodeDependencies()
        deps._validation_errors = ["Error 1", "Error 2"]

        errors = deps.get_validation_errors()
        errors.append("Error 3")

        assert deps._validation_errors == ["Error 1", "Error 2"]


class TestNodeExecutionContext:
    """Test suite for NodeExecutionContext tracking system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""

        class TestNode(FlowNode):
            def execute(self, state):
                return {"test": "result"}

            def validate_preconditions(self, state):
                return True

        self.test_node = TestNode()
        self.test_state = {"session_id": "test"}

    def test_node_execution_context_initialization(self):
        """Test NodeExecutionContext initialization."""
        context = NodeExecutionContext(self.test_node, self.test_state)

        assert context.node is self.test_node
        assert context.state is self.test_state
        assert isinstance(context.start_time, datetime)
        assert context.end_time is None
        assert context.success is False
        assert context.result is None
        assert context.error is None
        assert context.validation_errors == []

    def test_mark_completed(self):
        """Test marking execution context as completed."""
        context = NodeExecutionContext(self.test_node, self.test_state)
        test_result = {"test": "result"}

        context.mark_completed(test_result)

        assert context.success is True
        assert context.result is test_result
        assert isinstance(context.end_time, datetime)
        assert context.end_time >= context.start_time

    def test_mark_failed(self):
        """Test marking execution context as failed."""
        context = NodeExecutionContext(self.test_node, self.test_state)
        test_error = RuntimeError("Test error")

        context.mark_failed(test_error)

        assert context.success is False
        assert context.error is test_error
        assert isinstance(context.end_time, datetime)
        assert context.end_time >= context.start_time

    def test_get_execution_time_not_completed(self):
        """Test getting execution time when not completed."""
        context = NodeExecutionContext(self.test_node, self.test_state)

        execution_time = context.get_execution_time()

        assert execution_time == 0.0

    def test_get_execution_time_completed(self):
        """Test getting execution time when completed."""
        context = NodeExecutionContext(self.test_node, self.test_state)

        # Mark as completed after a small delay
        import time

        time.sleep(0.01)
        context.mark_completed({"test": "result"})

        execution_time = context.get_execution_time()

        assert execution_time > 0.0
        assert execution_time < 1.0  # Should be very small

    def test_to_dict(self):
        """Test converting execution context to dictionary."""
        context = NodeExecutionContext(self.test_node, self.test_state)
        test_error = RuntimeError("Test error")
        context.validation_errors = ["Validation error 1"]
        context.mark_failed(test_error)

        result_dict = context.to_dict()

        assert result_dict["node_name"] == "TestNode"
        assert result_dict["start_time"] == context.start_time.isoformat()
        assert result_dict["end_time"] == context.end_time.isoformat()
        assert result_dict["execution_time"] > 0
        assert result_dict["success"] is False
        assert result_dict["error"] == "Test error"
        assert result_dict["validation_errors"] == ["Validation error 1"]
        assert result_dict["state_keys"] == ["session_id"]


class TestNodeValidationError:
    """Test suite for NodeValidationError exception."""

    def test_node_validation_error(self):
        """Test NodeValidationError can be raised and caught."""
        with pytest.raises(NodeValidationError) as exc_info:
            raise NodeValidationError("Test validation error")

        assert str(exc_info.value) == "Test validation error"


class TestNodeExecutionError:
    """Test suite for NodeExecutionError exception."""

    def test_node_execution_error(self):
        """Test NodeExecutionError can be raised and caught."""
        with pytest.raises(NodeExecutionError) as exc_info:
            raise NodeExecutionError("Test execution error")

        assert str(exc_info.value) == "Test execution error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
