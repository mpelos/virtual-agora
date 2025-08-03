"""Tests for Virtual Agora node registry system.

This module provides comprehensive tests for the NodeRegistry and related
classes, ensuring proper node management and dependency injection.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.virtual_agora.flow.nodes.registry import (
    NodeRegistry,
    CompatibilityNodeRegistry,
    V13CompatibilityWrapper,
    NodeRegistryError,
)
from src.virtual_agora.flow.nodes.base import (
    FlowNode,
    HITLNode,
    AgentOrchestratorNode,
    NodeDependencies,
)
from src.virtual_agora.state.schema import VirtualAgoraState


class TestNodeRegistry:
    """Test suite for NodeRegistry class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create test dependencies
        self.dependencies = NodeDependencies(
            round_manager=Mock(), message_coordinator=Mock(), flow_state_manager=Mock()
        )
        self.dependencies.validate()

        # Create test node classes
        class TestFlowNode(FlowNode):
            def __init__(self, should_fail=False, node_dependencies=None):
                super().__init__()
                self.should_fail = should_fail
                if node_dependencies:
                    self.dependencies = node_dependencies

            def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
                if self.should_fail:
                    raise RuntimeError("Test execution failure")
                return {"test_key": "test_value"}

            def validate_preconditions(self, state: VirtualAgoraState) -> bool:
                return isinstance(state, dict) and "session_id" in state

        class TestHITLNode(HITLNode):
            def __init__(self, node_dependencies=None):
                super().__init__()
                if node_dependencies:
                    self.dependencies = node_dependencies

            def create_interrupt_payload(
                self, state: VirtualAgoraState
            ) -> Dict[str, Any]:
                return {"type": "test", "message": "Test interrupt"}

            def process_user_input(
                self, user_input: Dict[str, Any], state: VirtualAgoraState
            ) -> Dict[str, Any]:
                return {"processed": True}

            def validate_preconditions(self, state: VirtualAgoraState) -> bool:
                return True

        self.TestFlowNode = TestFlowNode
        self.TestHITLNode = TestHITLNode

        self.test_state = {
            "session_id": "test_session",
            "current_round": 1,
            "messages": [],
        }

    def test_registry_initialization(self):
        """Test NodeRegistry initialization."""
        registry = NodeRegistry(self.dependencies)

        assert registry.dependencies is self.dependencies
        assert registry._nodes == {}
        assert registry._node_metadata == {}
        assert registry._execution_stats == {}
        assert registry._registration_history == []

    def test_register_node_success(self):
        """Test successful node registration."""
        registry = NodeRegistry(self.dependencies)
        test_node = self.TestFlowNode()
        metadata = {"description": "Test node"}

        registry.register_node("test_node", test_node, metadata)

        # Check node was registered
        assert "test_node" in registry._nodes
        assert registry._nodes["test_node"] is test_node

        # Check metadata was stored
        node_metadata = registry._node_metadata["test_node"]
        assert node_metadata["node_class"] == "TestFlowNode"
        assert node_metadata["node_type"] == "flow"
        assert node_metadata["custom_metadata"] == metadata
        assert "registration_time" in node_metadata

        # Check execution stats were initialized
        stats = registry._execution_stats["test_node"]
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0

        # Check registration history
        assert len(registry._registration_history) == 1
        history_entry = registry._registration_history[0]
        assert history_entry["action"] == "register"
        assert history_entry["node_name"] == "test_node"
        assert history_entry["node_class"] == "TestFlowNode"

    def test_register_node_with_dependencies_injection(self):
        """Test node registration with dependency injection."""
        registry = NodeRegistry(self.dependencies)
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        # Check dependencies were injected
        assert test_node.dependencies is self.dependencies

    def test_register_node_duplicate_without_override(self):
        """Test registering duplicate node without override."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        # Attempt to register again should fail
        with pytest.raises(NodeRegistryError) as exc_info:
            registry.register_node("test_node", self.TestFlowNode())

        assert "already registered" in str(exc_info.value)

    def test_register_node_duplicate_with_override(self):
        """Test registering duplicate node with override."""
        registry = NodeRegistry()
        first_node = self.TestFlowNode()
        second_node = self.TestFlowNode()

        registry.register_node("test_node", first_node)
        registry.register_node("test_node", second_node, override=True)

        # Check second node replaced first
        assert registry._nodes["test_node"] is second_node
        assert len(registry._registration_history) == 2
        assert registry._registration_history[1]["override"] is True

    def test_register_node_invalid_type(self):
        """Test registering invalid node type."""
        registry = NodeRegistry()

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.register_node("invalid_node", "not_a_node")

        assert "must be an instance of FlowNode" in str(exc_info.value)

    def test_register_node_class_success(self):
        """Test registering node by class."""
        registry = NodeRegistry(self.dependencies)

        registry.register_node_class(
            "test_node",
            self.TestFlowNode,
            init_kwargs={"should_fail": False},
            metadata={"created_from_class": True},
        )

        # Check node was instantiated and registered
        assert "test_node" in registry._nodes
        node = registry._nodes["test_node"]
        assert isinstance(node, self.TestFlowNode)
        assert node.should_fail is False

        # Check metadata
        metadata = registry._node_metadata["test_node"]
        assert metadata["custom_metadata"]["created_from_class"] is True

    def test_register_node_class_failure(self):
        """Test registering node class with instantiation failure."""
        registry = NodeRegistry()

        class FailingNode(FlowNode):
            def __init__(self):
                raise RuntimeError("Instantiation failed")

            def execute(self, state):
                return {}

            def validate_preconditions(self, state):
                return True

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.register_node_class("failing_node", FailingNode)

        assert "Failed to register node class" in str(exc_info.value)

    def test_get_node_success(self):
        """Test successful node retrieval."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        retrieved_node = registry.get_node("test_node")

        assert retrieved_node is test_node

    def test_get_node_not_found(self):
        """Test getting non-existent node."""
        registry = NodeRegistry()
        registry.register_node("existing_node", self.TestFlowNode())

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.get_node("missing_node")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "existing_node" in error_msg

    def test_execute_node_success(self):
        """Test successful node execution through registry."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        result = registry.execute_node("test_node", self.test_state)

        assert result["test_key"] == "test_value"

        # Check execution stats were updated
        stats = registry._execution_stats["test_node"]
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 0
        assert stats["average_execution_time"] > 0

    def test_execute_node_failure(self):
        """Test node execution failure through registry."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode(should_fail=True)

        registry.register_node("test_node", test_node)

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.execute_node("test_node", self.test_state)

        assert "execution failed" in str(exc_info.value)

        # Check execution stats were updated
        stats = registry._execution_stats["test_node"]
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 1
        assert stats["last_error"] == "Test execution failure"

    def test_execute_node_not_found(self):
        """Test executing non-existent node."""
        registry = NodeRegistry()

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.execute_node("missing_node", self.test_state)

        assert "not found" in str(exc_info.value)

    def test_unregister_node_success(self):
        """Test successful node unregistration."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        result = registry.unregister_node("test_node")

        assert result is True
        assert "test_node" not in registry._nodes
        assert "test_node" not in registry._node_metadata
        assert "test_node" not in registry._execution_stats

        # Check unregistration was recorded
        history = registry._registration_history
        assert len(history) == 2  # Register + unregister
        assert history[1]["action"] == "unregister"
        assert history[1]["node_name"] == "test_node"

    def test_unregister_node_not_found(self):
        """Test unregistering non-existent node."""
        registry = NodeRegistry()

        result = registry.unregister_node("missing_node")

        assert result is False

    def test_list_nodes(self):
        """Test listing registered nodes."""
        registry = NodeRegistry()

        registry.register_node("node1", self.TestFlowNode())
        registry.register_node("node2", self.TestHITLNode())

        nodes = registry.list_nodes()

        assert set(nodes) == {"node1", "node2"}

    def test_get_node_info(self):
        """Test getting detailed node information."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()
        metadata = {"description": "Test node"}

        registry.register_node("test_node", test_node, metadata)

        info = registry.get_node_info("test_node")

        assert info["name"] == "test_node"
        assert info["class"] == "TestFlowNode"
        assert info["metadata"]["custom_metadata"] == metadata
        assert info["is_available"] is True
        assert "execution_stats" in info

    def test_get_node_info_not_found(self):
        """Test getting info for non-existent node."""
        registry = NodeRegistry()

        with pytest.raises(NodeRegistryError):
            registry.get_node_info("missing_node")

    def test_validate_all_nodes(self):
        """Test validating all registered nodes."""
        registry = NodeRegistry()

        # Register nodes with different validation behaviors
        valid_node = self.TestFlowNode()
        invalid_node = self.TestFlowNode()

        registry.register_node("valid_node", valid_node)
        registry.register_node("invalid_node", invalid_node)

        # Mock validation to return different results
        with (
            patch.object(
                valid_node, "validate_preconditions", return_value=True
            ) as mock_valid,
            patch.object(
                invalid_node, "validate_preconditions", return_value=False
            ) as mock_invalid,
        ):

            # Set up validation errors for invalid node - simulate errors being added during validation
            def add_error_and_return_false(state):
                invalid_node._validation_errors.append("Test validation error")
                return False

            mock_invalid.side_effect = add_error_and_return_false

            results = registry.validate_all_nodes(self.test_state)

        assert results["valid_node"]["valid"] is True
        assert results["valid_node"]["errors"] == []

        assert results["invalid_node"]["valid"] is False
        assert "Test validation error" in results["invalid_node"]["errors"]

    def test_validate_all_nodes_with_exception(self):
        """Test validation when node raises exception."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        # Mock validation to raise exception
        with patch.object(
            test_node,
            "validate_preconditions",
            side_effect=RuntimeError("Validation error"),
        ):
            results = registry.validate_all_nodes(self.test_state)

        assert results["test_node"]["valid"] is False
        assert (
            "Validation exception: Validation error" in results["test_node"]["errors"]
        )

    def test_get_nodes_by_type(self):
        """Test getting nodes by type."""
        registry = NodeRegistry()

        flow_node = self.TestFlowNode()
        hitl_node = self.TestHITLNode()

        registry.register_node("flow_node", flow_node)
        registry.register_node("hitl_node", hitl_node)

        flow_nodes = registry.get_nodes_by_type("flow")
        hitl_nodes = registry.get_nodes_by_type("hitl")

        assert "flow_node" in flow_nodes
        assert flow_nodes["flow_node"] is flow_node

        assert "hitl_node" in hitl_nodes
        assert hitl_nodes["hitl_node"] is hitl_node

    def test_get_execution_statistics(self):
        """Test getting execution statistics."""
        registry = NodeRegistry()
        test_node = self.TestFlowNode()

        registry.register_node("test_node", test_node)

        # Execute node to generate stats
        registry.execute_node("test_node", self.test_state)

        stats = registry.get_execution_statistics()

        assert "test_node" in stats
        node_stats = stats["test_node"]
        assert node_stats["total_executions"] == 1
        assert node_stats["successful_executions"] == 1
        assert node_stats["failed_executions"] == 0

    def test_get_registry_summary(self):
        """Test getting registry summary."""
        registry = NodeRegistry(self.dependencies)

        registry.register_node("flow_node", self.TestFlowNode())
        registry.register_node("hitl_node", self.TestHITLNode())

        # Execute some nodes to generate stats
        registry.execute_node("flow_node", self.test_state)

        summary = registry.get_registry_summary()

        assert summary["total_nodes"] == 2
        assert summary["node_types"]["flow"] == 1
        assert summary["node_types"]["hitl"] == 1
        assert summary["total_executions"] == 1
        assert summary["total_failures"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["dependencies_available"] is True

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = NodeRegistry()

        registry.register_node("node1", self.TestFlowNode())
        registry.register_node("node2", self.TestHITLNode())

        registry.clear_registry()

        assert len(registry._nodes) == 0
        assert len(registry._node_metadata) == 0
        assert len(registry._execution_stats) == 0

        # Check clearing was recorded
        history = registry._registration_history
        assert history[-1]["action"] == "clear_all"
        assert history[-1]["nodes_removed"] == 2


class TestCompatibilityNodeRegistry:
    """Test suite for CompatibilityNodeRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""

        # Create a proper mock that only has specific methods
        class FakeV13Nodes:
            def test_method(self, state):
                return {"v13_result": True}

            def another_method(self, state):
                return {"another_result": True}

        self.mock_v13_nodes = FakeV13Nodes()
        self.test_state = {"session_id": "test_session"}

    def test_compatibility_registry_initialization(self):
        """Test CompatibilityNodeRegistry initialization."""
        registry = CompatibilityNodeRegistry(v13_nodes=self.mock_v13_nodes)

        assert registry.v13_nodes is self.mock_v13_nodes
        assert registry._compatibility_mode is True

    def test_compatibility_registry_without_v13(self):
        """Test CompatibilityNodeRegistry without V13 nodes."""
        registry = CompatibilityNodeRegistry()

        assert registry.v13_nodes is None
        assert registry._compatibility_mode is False

    def test_register_v13_compatibility_wrapper_success(self):
        """Test successful V13 compatibility wrapper registration."""
        registry = CompatibilityNodeRegistry(v13_nodes=self.mock_v13_nodes)

        # test_method is already available on the fake V13 nodes
        registry.register_v13_compatibility_wrapper("test_node", "test_method")

        # Check wrapper was registered
        assert "test_node" in registry._nodes
        wrapper = registry._nodes["test_node"]
        assert isinstance(wrapper, V13CompatibilityWrapper)

        # Check metadata
        metadata = registry._node_metadata["test_node"]
        assert metadata["custom_metadata"]["compatibility_wrapper"] is True
        assert metadata["custom_metadata"]["v13_method"] == "test_method"

    def test_register_v13_compatibility_wrapper_no_compatibility_mode(self):
        """Test V13 wrapper registration without compatibility mode."""
        registry = CompatibilityNodeRegistry()

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.register_v13_compatibility_wrapper("test_node", "test_method")

        assert "V13 compatibility mode not enabled" in str(exc_info.value)

    def test_register_v13_compatibility_wrapper_missing_method(self):
        """Test V13 wrapper registration with missing method."""
        registry = CompatibilityNodeRegistry(v13_nodes=self.mock_v13_nodes)

        # Don't add the missing_method to the mock, so hasattr will return False
        with pytest.raises(NodeRegistryError) as exc_info:
            registry.register_v13_compatibility_wrapper("test_node", "missing_method")

        assert "has no method 'missing_method'" in str(exc_info.value)

    def test_migrate_from_v13(self):
        """Test migrating from V13 nodes."""
        registry = CompatibilityNodeRegistry(v13_nodes=self.mock_v13_nodes)

        node_mappings = {"test_node": "test_method", "another_node": "another_method"}

        registry.migrate_from_v13(node_mappings)

        # Check both nodes were registered
        assert "test_node" in registry._nodes
        assert "another_node" in registry._nodes
        assert len(registry._registration_history) == 2

    def test_migrate_from_v13_no_compatibility_mode(self):
        """Test migration without compatibility mode."""
        registry = CompatibilityNodeRegistry()

        with pytest.raises(NodeRegistryError) as exc_info:
            registry.migrate_from_v13({"test_node": "test_method"})

        assert "V13 compatibility mode not enabled" in str(exc_info.value)


class TestV13CompatibilityWrapper:
    """Test suite for V13CompatibilityWrapper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_v13_nodes = Mock()
        self.mock_v13_nodes.test_method = Mock(return_value={"v13_result": True})

        self.test_state = {"session_id": "test_session"}

    def test_wrapper_initialization(self):
        """Test V13CompatibilityWrapper initialization."""
        wrapper = V13CompatibilityWrapper(self.mock_v13_nodes, "test_method")

        assert wrapper.v13_nodes is self.mock_v13_nodes
        assert wrapper.method_name == "test_method"
        assert wrapper.method is self.mock_v13_nodes.test_method

    def test_wrapper_execute(self):
        """Test wrapper execution."""
        wrapper = V13CompatibilityWrapper(self.mock_v13_nodes, "test_method")

        result = wrapper.execute(self.test_state)

        assert result == {"v13_result": True}
        self.mock_v13_nodes.test_method.assert_called_once_with(self.test_state)

    def test_wrapper_validate_preconditions_success(self):
        """Test wrapper validation with valid state."""
        wrapper = V13CompatibilityWrapper(self.mock_v13_nodes, "test_method")

        result = wrapper.validate_preconditions(self.test_state)

        assert result is True
        assert wrapper._validation_errors == []

    def test_wrapper_validate_preconditions_invalid_state(self):
        """Test wrapper validation with invalid state."""
        wrapper = V13CompatibilityWrapper(self.mock_v13_nodes, "test_method")

        result = wrapper.validate_preconditions("not_a_dict")

        assert result is False
        assert len(wrapper._validation_errors) == 1
        assert "State must be a dictionary" in wrapper._validation_errors

    def test_wrapper_get_node_name(self):
        """Test wrapper node name."""
        wrapper = V13CompatibilityWrapper(self.mock_v13_nodes, "test_method")

        name = wrapper.get_node_name()

        assert name == "V13_test_method"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
