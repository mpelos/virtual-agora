"""Tests for the node registry system.

This module tests the pluggable node architecture that allows dynamic
registration and management of flow nodes.
"""

import pytest
from unittest.mock import Mock

from virtual_agora.flow.node_registry import (
    NodeRegistry,
    V13NodeWrapper,
    create_default_v13_registry,
)
from virtual_agora.flow.nodes.base import (
    FlowNode,
    NodeValidationError,
    NodeExecutionError,
)
from virtual_agora.state.schema import VirtualAgoraState


class MockFlowNode(FlowNode):
    """Mock flow node for testing."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def execute(self, state: VirtualAgoraState) -> dict:
        return {"executed": self.name}

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        return True

    def get_node_name(self) -> str:
        return f"MockNode({self.name})"


class InvalidNode:
    """Invalid node that doesn't inherit from FlowNode."""

    pass


@pytest.fixture
def registry():
    """Create a fresh NodeRegistry for testing."""
    return NodeRegistry()


@pytest.fixture
def sample_node():
    """Create a sample FlowNode for testing."""
    return MockFlowNode("test")


@pytest.fixture
def sample_state():
    """Create sample VirtualAgoraState for testing."""
    return VirtualAgoraState(
        session_id="test_session",
        theme="Test Theme",
        active_topic="Test Topic",
        current_round=1,
        messages=[],
    )


class TestNodeRegistry:
    """Test the NodeRegistry class."""

    def test_initialization(self, registry):
        """Test NodeRegistry initialization."""
        assert len(registry._nodes) == 0
        assert len(registry._node_metadata) == 0
        assert len(registry._registration_order) == 0
        assert len(registry.get_node_names()) == 0

    def test_register_node_success(self, registry, sample_node):
        """Test successful node registration."""
        metadata = {"phase": 1, "type": "test"}
        registry.register_node("test_node", sample_node, metadata)

        assert registry.has_node("test_node")
        assert registry.get_node("test_node") == sample_node
        assert registry.get_node_metadata("test_node") == metadata
        assert "test_node" in registry.get_node_names()
        assert len(registry.get_all_nodes()) == 1

    def test_register_node_with_empty_name(self, registry, sample_node):
        """Test registration with empty node name."""
        with pytest.raises(ValueError) as exc_info:
            registry.register_node("", sample_node)

        assert "non-empty string" in str(exc_info.value)

    def test_register_node_with_invalid_type(self, registry):
        """Test registration with invalid node type."""
        invalid_node = InvalidNode()

        with pytest.raises(TypeError) as exc_info:
            registry.register_node("invalid", invalid_node)

        assert "FlowNode instance" in str(exc_info.value)

    def test_register_duplicate_node(self, registry, sample_node):
        """Test registration of duplicate node name."""
        registry.register_node("test_node", sample_node)

        with pytest.raises(ValueError) as exc_info:
            registry.register_node("test_node", MockFlowNode("another"))

        assert "already registered" in str(exc_info.value)

    def test_get_nonexistent_node(self, registry):
        """Test getting non-existent node."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_node("nonexistent")

        assert "not found" in str(exc_info.value)
        assert "Available nodes" in str(exc_info.value)

    def test_unregister_node(self, registry, sample_node):
        """Test node unregistration."""
        registry.register_node("test_node", sample_node)
        assert registry.has_node("test_node")

        unregistered_node = registry.unregister_node("test_node")

        assert unregistered_node == sample_node
        assert not registry.has_node("test_node")
        assert "test_node" not in registry.get_node_names()
        assert len(registry.get_all_nodes()) == 0

    def test_unregister_nonexistent_node(self, registry):
        """Test unregistering non-existent node."""
        with pytest.raises(KeyError) as exc_info:
            registry.unregister_node("nonexistent")

        assert "not found" in str(exc_info.value)

    def test_update_node_metadata(self, registry, sample_node):
        """Test updating node metadata."""
        registry.register_node("test_node", sample_node, {"initial": "data"})

        registry.update_node_metadata(
            "test_node", {"updated": "data", "initial": "modified"}
        )

        metadata = registry.get_node_metadata("test_node")
        assert metadata["updated"] == "data"
        assert metadata["initial"] == "modified"

    def test_update_metadata_nonexistent_node(self, registry):
        """Test updating metadata for non-existent node."""
        with pytest.raises(KeyError) as exc_info:
            registry.update_node_metadata("nonexistent", {"test": "data"})

        assert "not found" in str(exc_info.value)

    def test_validate_registry_success(self, registry, sample_node):
        """Test registry validation with valid nodes."""
        registry.register_node("test_node", sample_node, {"type": "test"})
        registry.register_node("another_node", MockFlowNode("another"))

        validation_results = registry.validate_registry()

        assert validation_results["total_nodes"] == 2
        assert validation_results["valid_nodes"] == 2
        assert len(validation_results["invalid_nodes"]) == 0
        assert len(validation_results["validation_errors"]) == 0

    def test_validate_registry_with_invalid_nodes(self, registry):
        """Test registry validation with invalid nodes."""
        # Manually insert invalid node to test validation
        registry._nodes["invalid"] = InvalidNode()
        registry._registration_order.append("invalid")

        validation_results = registry.validate_registry()

        assert validation_results["total_nodes"] == 1
        assert validation_results["valid_nodes"] == 0
        assert len(validation_results["invalid_nodes"]) == 1
        assert "not a FlowNode instance" in validation_results["invalid_nodes"][0]

    def test_clear_registry(self, registry, sample_node):
        """Test clearing the registry."""
        registry.register_node("test_node", sample_node)
        registry.register_node("another_node", MockFlowNode("another"))

        assert len(registry.get_all_nodes()) == 2

        registry.clear()

        assert len(registry.get_all_nodes()) == 0
        assert len(registry.get_node_names()) == 0

    def test_registration_order_maintained(self, registry):
        """Test that registration order is maintained."""
        node1 = MockFlowNode("first")
        node2 = MockFlowNode("second")
        node3 = MockFlowNode("third")

        registry.register_node("first", node1)
        registry.register_node("second", node2)
        registry.register_node("third", node3)

        node_names = registry.get_node_names()
        assert node_names == ["first", "second", "third"]


class TestV13NodeWrapper:
    """Test the V13NodeWrapper class."""

    def test_wrapper_initialization(self):
        """Test V13NodeWrapper initialization."""

        def mock_node_function(state):
            return {"result": "test"}

        wrapper = V13NodeWrapper(mock_node_function, "test_node")

        assert wrapper.node_function == mock_node_function
        assert wrapper.node_name == "test_node"
        assert wrapper.validation_function is None

    def test_wrapper_execution(self, sample_state):
        """Test wrapped node execution."""

        def mock_node_function(state):
            return {"node_executed": True, "session_id": state.get("session_id")}

        wrapper = V13NodeWrapper(mock_node_function, "test_node")
        result = wrapper.execute(sample_state)

        assert result["node_executed"] is True
        assert result["session_id"] == "test_session"

    def test_wrapper_execution_with_error(self, sample_state):
        """Test wrapped node execution with error."""

        def failing_node_function(state):
            raise ValueError("Test error")

        wrapper = V13NodeWrapper(failing_node_function, "failing_node")

        with pytest.raises(NodeExecutionError) as exc_info:
            wrapper.execute(sample_state)

        assert "Wrapped node 'failing_node' execution failed" in str(exc_info.value)
        assert "Test error" in str(exc_info.value)

    def test_wrapper_validation_without_validator(self, sample_state):
        """Test wrapper validation without custom validator."""

        def mock_node_function(state):
            return {"result": "test"}

        wrapper = V13NodeWrapper(mock_node_function, "test_node")

        assert wrapper.validate_preconditions(sample_state) is True

    def test_wrapper_validation_with_custom_validator(self, sample_state):
        """Test wrapper validation with custom validator."""

        def mock_node_function(state):
            return {"result": "test"}

        def custom_validator(state):
            return state.get("session_id") == "test_session"

        wrapper = V13NodeWrapper(mock_node_function, "test_node", custom_validator)

        assert wrapper.validate_preconditions(sample_state) is True

        # Test with invalid state
        invalid_state = VirtualAgoraState(session_id="different_session")
        assert wrapper.validate_preconditions(invalid_state) is False

    def test_wrapper_validation_with_failing_validator(self, sample_state):
        """Test wrapper validation with failing validator."""

        def mock_node_function(state):
            return {"result": "test"}

        def failing_validator(state):
            raise Exception("Validation error")

        wrapper = V13NodeWrapper(mock_node_function, "test_node", failing_validator)

        assert wrapper.validate_preconditions(sample_state) is False
        assert "Validation function error" in wrapper._validation_errors[0]

    def test_wrapper_get_node_name(self):
        """Test wrapper get_node_name method."""

        def mock_node_function(state):
            return {"result": "test"}

        wrapper = V13NodeWrapper(mock_node_function, "test_node")

        assert wrapper.get_node_name() == "V13Wrapper(test_node)"


class TestCreateDefaultV13Registry:
    """Test the create_default_v13_registry function."""

    def test_create_default_registry(self):
        """Test creating default V13 registry."""
        # Create mock V13FlowNodes
        mock_v13_nodes = Mock()

        # Mock all the node methods
        mock_v13_nodes.config_and_keys_node = Mock(return_value={"config": "loaded"})
        mock_v13_nodes.agent_instantiation_node = Mock(
            return_value={"agents": "created"}
        )
        mock_v13_nodes.get_theme_node = Mock(return_value={"theme": "set"})
        mock_v13_nodes.agenda_proposal_node = Mock(return_value={"agenda": "proposed"})
        mock_v13_nodes.topic_refinement_node = Mock(return_value={"topics": "refined"})
        mock_v13_nodes.collate_proposals_node = Mock(
            return_value={"proposals": "collated"}
        )
        mock_v13_nodes.agenda_voting_node = Mock(return_value={"votes": "cast"})
        mock_v13_nodes.synthesize_agenda_node = Mock(
            return_value={"agenda": "synthesized"}
        )
        mock_v13_nodes.agenda_approval_node = Mock(return_value={"agenda": "approved"})
        mock_v13_nodes.announce_item_node = Mock(return_value={"item": "announced"})
        mock_v13_nodes.discussion_round_node = Mock(
            return_value={"discussion": "completed"}
        )
        mock_v13_nodes.round_summarization_node = Mock(
            return_value={"round": "summarized"}
        )
        mock_v13_nodes.round_threshold_check_node = Mock(
            return_value={"threshold": "checked"}
        )
        mock_v13_nodes.end_topic_poll_node = Mock(return_value={"poll": "completed"})
        mock_v13_nodes.vote_evaluation_node = Mock(return_value={"votes": "evaluated"})
        mock_v13_nodes.periodic_user_stop_node = Mock(return_value={"user": "stopped"})
        mock_v13_nodes.user_topic_conclusion_confirmation_node = Mock(
            return_value={"conclusion": "confirmed"}
        )
        mock_v13_nodes.final_considerations_node = Mock(
            return_value={"considerations": "final"}
        )
        mock_v13_nodes.topic_report_generation_node = Mock(
            return_value={"report": "generated"}
        )
        mock_v13_nodes.topic_summary_generation_node = Mock(
            return_value={"summary": "generated"}
        )
        mock_v13_nodes.file_output_node = Mock(return_value={"file": "written"})
        mock_v13_nodes.agent_poll_node = Mock(return_value={"agents": "polled"})
        mock_v13_nodes.user_approval_node = Mock(return_value={"user": "approved"})
        mock_v13_nodes.agenda_modification_node = Mock(
            return_value={"agenda": "modified"}
        )
        mock_v13_nodes.final_report_node = Mock(return_value={"report": "final"})
        mock_v13_nodes.multi_file_output_node = Mock(return_value={"files": "written"})
        mock_v13_nodes.user_turn_participation_node = Mock(
            return_value={"user": "participated"}
        )

        registry = create_default_v13_registry(mock_v13_nodes)

        # Verify all expected nodes are registered
        expected_nodes = [
            "config_and_keys",
            "agent_instantiation",
            "get_theme",
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
            "announce_item",
            "discussion_round",
            "round_summarization",
            "round_threshold_check",
            "end_topic_poll",
            "vote_evaluation",
            "periodic_user_stop",
            "user_topic_conclusion_confirmation",
            "final_considerations",
            "topic_report_generation",
            "topic_summary_generation",
            "file_output",
            "agent_poll",
            "user_approval",
            "agenda_modification",
            "final_report_generation",
            "multi_file_output",
            "user_turn_participation",
        ]

        all_nodes = registry.get_all_nodes()
        registered_names = set(all_nodes.keys())

        for node_name in expected_nodes:
            assert node_name in registered_names, f"Node {node_name} not registered"

        # Verify nodes are wrapped properly
        for node_name, node in all_nodes.items():
            if node_name != "discussion_round":  # Might be replaced with unified node
                assert isinstance(
                    node, V13NodeWrapper
                ), f"Node {node_name} not properly wrapped"

    def test_create_default_registry_with_unified_discussion_node(self):
        """Test creating default registry with unified discussion node."""
        mock_v13_nodes = Mock()

        # Mock all required node methods (abbreviated for brevity)
        for attr in [
            "config_and_keys_node",
            "agent_instantiation_node",
            "get_theme_node",
            "agenda_proposal_node",
            "topic_refinement_node",
            "collate_proposals_node",
            "agenda_voting_node",
            "synthesize_agenda_node",
            "agenda_approval_node",
            "announce_item_node",
            "discussion_round_node",
            "round_summarization_node",
            "round_threshold_check_node",
            "end_topic_poll_node",
            "vote_evaluation_node",
            "periodic_user_stop_node",
            "user_topic_conclusion_confirmation_node",
            "final_considerations_node",
            "topic_report_generation_node",
            "topic_summary_generation_node",
            "file_output_node",
            "agent_poll_node",
            "user_approval_node",
            "agenda_modification_node",
            "final_report_node",
            "multi_file_output_node",
            "user_turn_participation_node",
        ]:
            setattr(mock_v13_nodes, attr, Mock(return_value={"test": "result"}))

        unified_discussion_node = MockFlowNode("unified_discussion")

        registry = create_default_v13_registry(mock_v13_nodes, unified_discussion_node)

        all_nodes = registry.get_all_nodes()

        # Verify unified discussion node is used
        assert all_nodes["discussion_round"] == unified_discussion_node

        # Verify metadata indicates it's unified
        discussion_metadata = registry.get_node_metadata("discussion_round")
        assert discussion_metadata.get("unified") is True

    def test_registry_validation_with_default_nodes(self):
        """Test that default registry passes validation."""
        mock_v13_nodes = Mock()

        # Mock all required node methods
        for attr in [
            "config_and_keys_node",
            "agent_instantiation_node",
            "get_theme_node",
            "agenda_proposal_node",
            "topic_refinement_node",
            "collate_proposals_node",
            "agenda_voting_node",
            "synthesize_agenda_node",
            "agenda_approval_node",
            "announce_item_node",
            "discussion_round_node",
            "round_summarization_node",
            "round_threshold_check_node",
            "end_topic_poll_node",
            "vote_evaluation_node",
            "periodic_user_stop_node",
            "user_topic_conclusion_confirmation_node",
            "final_considerations_node",
            "topic_report_generation_node",
            "topic_summary_generation_node",
            "file_output_node",
            "agent_poll_node",
            "user_approval_node",
            "agenda_modification_node",
            "final_report_node",
            "multi_file_output_node",
            "user_turn_participation_node",
        ]:
            setattr(mock_v13_nodes, attr, Mock(return_value={"test": "result"}))

        registry = create_default_v13_registry(mock_v13_nodes)
        validation_results = registry.validate_registry()

        # Check that all nodes are valid
        assert len(validation_results["invalid_nodes"]) == 0
        assert len(validation_results["validation_errors"]) == 0
        assert validation_results["valid_nodes"] == validation_results["total_nodes"]


if __name__ == "__main__":
    pytest.main([__file__])
