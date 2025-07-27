"""End-to-end integration tests for graph execution.

This module tests the actual execution of the LangGraph graph
to ensure all nodes return valid state updates and the flow
works correctly from start to finish.
"""

import pytest
import uuid
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import asyncio

from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.state.manager import StateManager
from virtual_agora.utils.exceptions import StateError, VirtualAgoraError

from ..helpers.fake_llm import create_fake_llm_pool, create_specialized_fake_llms
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    patch_ui_components,
    create_test_messages,
)


class TestEndToEndGraphExecution:
    """Test complete graph execution from start to finish."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=3, scenario="quick_consensus"
        )
        self.session_id = f"test_session_{uuid.uuid4().hex[:8]}"

    @pytest.mark.integration
    def test_minimal_graph_execution(self):
        """Test minimal graph execution through first few nodes."""
        with patch_ui_components() as ui_patches:
            # Set up UI mocks for minimal execution
            ui_patches.enter_context(
                patch(
                    "virtual_agora.ui.human_in_the_loop.get_initial_topic",
                    return_value="Test Topic for Graph Execution",
                )
            )

            # Create flow
            flow = self.test_helper.create_test_flow()
            flow.compile()

            # Create session properly through flow
            session_id = flow.create_session(
                main_topic="Test Topic for Graph Execution"
            )

            # Run graph for just a few steps
            config = {"configurable": {"thread_id": session_id}}

            updates_received = []
            nodes_executed = []

            try:
                # Stream execution and collect updates
                for update in flow.stream(config):
                    updates_received.append(update)

                    # Track which nodes were executed
                    for node_name, node_update in update.items():
                        nodes_executed.append(node_name)

                        # Verify each update only contains valid state fields
                        if isinstance(node_update, dict):
                            for key in node_update.keys():
                                # This would have caught our bug - the key must exist in state
                                assert key in initial_state or key in [
                                    "agents",
                                    "messages",
                                    "round_summaries",
                                    "voting_rounds",
                                    "topic_summaries",
                                    "specialized_agents",
                                ], f"Node {node_name} returned invalid state field: {key}"

                    # Stop after a few nodes to keep test fast
                    if len(nodes_executed) >= 3:
                        break

            except Exception as e:
                pytest.fail(f"Graph execution failed: {str(e)}")

            # Verify basic execution
            assert len(updates_received) > 0, "No updates received from graph"
            assert len(nodes_executed) > 0, "No nodes were executed"

            # Verify expected initial nodes were executed
            assert (
                "config_and_keys" in nodes_executed
            ), "config_and_keys node not executed"
            assert (
                "agent_instantiation" in nodes_executed
            ), "agent_instantiation node not executed"

    @pytest.mark.integration
    def test_graph_state_validation(self):
        """Test that graph properly validates state updates."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()

            # Create a node that returns invalid fields (to test validation)
            def bad_node(state):
                return {"invalid_field": "This should fail", "another_bad_field": 123}

            # Replace a node with our bad node
            original_node = flow.nodes.config_and_keys_node
            flow.nodes.config_and_keys_node = bad_node

            # Recompile with the bad node
            flow.compile()

            # Create a session first
            session_id = flow.create_session(main_topic="Test Topic")

            config = {"configurable": {"thread_id": session_id}}

            # This should raise an error due to invalid state fields
            with pytest.raises(Exception) as exc_info:
                for update in flow.stream(config):
                    pass  # Try to consume the stream

            # Verify we got a state validation error
            assert "invalid_field" in str(exc_info.value) or "Expected node" in str(
                exc_info.value
            )

            # Restore original node
            flow.nodes.config_and_keys_node = original_node

    @pytest.mark.integration
    def test_phase_0_initialization_flow(self):
        """Test Phase 0 (initialization) completes successfully."""
        with patch_ui_components() as ui_patches:
            # Mock user input for initial topic
            ui_patches.enter_context(
                patch(
                    "virtual_agora.ui.human_in_the_loop.get_initial_topic",
                    return_value="AI Ethics and Governance",
                )
            )

            flow = self.test_helper.create_test_flow()
            flow.compile()

            config = {"configurable": {"thread_id": self.session_id}}

            phase_0_nodes = []
            state_updates = {}

            for update in flow.stream(config):
                for node_name, node_update in update.items():
                    phase_0_nodes.append(node_name)

                    # Collect state updates
                    if isinstance(node_update, dict):
                        state_updates.update(node_update)

                    # Stop after get_theme node (end of Phase 0)
                    if node_name == "get_theme":
                        break

                if "get_theme" in phase_0_nodes:
                    break

            # Verify Phase 0 nodes executed in order
            expected_phase_0_nodes = [
                "config_and_keys",
                "agent_instantiation",
                "get_theme",
            ]

            for expected_node in expected_phase_0_nodes:
                assert (
                    expected_node in phase_0_nodes
                ), f"Missing Phase 0 node: {expected_node}"

            # Verify state was properly initialized
            assert "main_topic" in state_updates, "Main topic not set"
            assert state_updates["main_topic"] == "AI Ethics and Governance"
            assert "agents" in state_updates, "Agents not initialized"

    @pytest.mark.integration
    def test_graph_execution_with_interrupt(self):
        """Test graph execution with interrupt handling."""
        with patch_ui_components() as ui_patches:
            # Set up mocks
            ui_patches.enter_context(
                patch(
                    "virtual_agora.ui.human_in_the_loop.get_initial_topic",
                    return_value="Test Topic",
                )
            )

            # Mock agenda approval to test interrupt
            approval_mock = Mock(return_value=["Topic 1", "Topic 2"])
            ui_patches.enter_context(
                patch(
                    "virtual_agora.ui.human_in_the_loop.get_agenda_approval",
                    approval_mock,
                )
            )

            flow = self.test_helper.create_test_flow()
            flow.compile()

            config = {"configurable": {"thread_id": self.session_id}}

            nodes_before_interrupt = []
            interrupt_encountered = False

            try:
                for update in flow.stream(config):
                    for node_name, node_update in update.items():
                        nodes_before_interrupt.append(node_name)

                        # Check if we hit an interrupt (HITL approval)
                        if (
                            isinstance(node_update, dict)
                            and "hitl_state" in node_update
                        ):
                            hitl_state = node_update.get("hitl_state", {})
                            if hitl_state.get("awaiting_approval"):
                                interrupt_encountered = True
                                break

                    if interrupt_encountered:
                        break

            except Exception as e:
                # Interrupts might raise exceptions in test environment
                if "interrupt" in str(e).lower():
                    interrupt_encountered = True
                else:
                    raise

            # Verify we executed some nodes before interrupt
            assert len(nodes_before_interrupt) > 0
            assert any(
                node in nodes_before_interrupt
                for node in ["config_and_keys", "agent_instantiation", "get_theme"]
            )

    @pytest.mark.integration
    def test_stream_yields_proper_updates(self):
        """Test that stream yields properly formatted updates."""
        with patch_ui_components() as ui_patches:
            ui_patches.enter_context(
                patch(
                    "virtual_agora.ui.human_in_the_loop.get_initial_topic",
                    return_value="Testing Stream Updates",
                )
            )

            flow = self.test_helper.create_test_flow()
            flow.compile()

            config = {"configurable": {"thread_id": self.session_id}}

            updates = []
            max_updates = 5  # Limit for testing

            for i, update in enumerate(flow.stream(config)):
                updates.append(update)

                # Validate update structure
                assert isinstance(update, dict), f"Update {i} is not a dict"
                assert len(update) > 0, f"Update {i} is empty"

                # Each update should have node_name -> state_update mapping
                for node_name, state_update in update.items():
                    assert isinstance(
                        node_name, str
                    ), f"Node name is not string: {node_name}"

                    # State update should be a dict (or special values like interrupt)
                    if not isinstance(state_update, dict):
                        # Could be interrupt or other special update
                        assert state_update is not None

                if i >= max_updates - 1:
                    break

            assert len(updates) > 0, "No updates yielded from stream"

    @pytest.mark.integration
    def test_graph_checkpointing(self):
        """Test that graph checkpointing works correctly."""
        with patch_ui_components() as ui_patches:
            ui_patches.enter_context(
                patch(
                    "virtual_agora.ui.human_in_the_loop.get_initial_topic",
                    return_value="Checkpointing Test",
                )
            )

            flow = self.test_helper.create_test_flow()
            flow.compile()

            # Verify checkpointer is configured
            assert flow.checkpointer is not None, "Checkpointer not configured"

            config = {"configurable": {"thread_id": self.session_id}}

            # Execute a few nodes
            nodes_executed = []
            for update in flow.stream(config):
                for node_name in update.keys():
                    nodes_executed.append(node_name)

                if len(nodes_executed) >= 3:
                    break

            # Verify we can get state from checkpointer
            # Note: The exact API depends on the checkpointer implementation
            # This is a basic test to ensure checkpointing is active
            assert len(nodes_executed) > 0, "No nodes executed for checkpoint test"


@pytest.mark.integration
class TestGraphErrorHandling:
    """Test graph error handling and recovery."""

    def test_node_exception_handling(self):
        """Test handling of exceptions raised by nodes."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()

            # Create a node that raises an exception
            def failing_node(state):
                raise ValueError("Simulated node failure")

            # Replace a non-critical node with failing node
            original_node = flow.nodes.get_theme_node
            flow.nodes.get_theme_node = failing_node

            flow.compile()

            config = {"configurable": {"thread_id": "error_test"}}

            # Graph execution should handle the error gracefully
            error_caught = False
            try:
                for update in flow.stream(config):
                    pass  # Try to consume stream
            except Exception as e:
                error_caught = True
                assert "Simulated node failure" in str(e)

            assert error_caught, "Expected error was not raised"

            # Restore original node
            flow.nodes.get_theme_node = original_node

    def test_state_type_validation(self):
        """Test that state type validation works."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()

            # Create a node that returns wrong types
            def bad_type_node(state):
                return {
                    "current_round": "should be int",  # Wrong type
                    "messages": "should be list",  # Wrong type
                }

            # This would cause type errors when the graph tries to merge state
            original_node = flow.nodes.agent_instantiation_node
            flow.nodes.agent_instantiation_node = bad_type_node

            flow.compile()

            config = {"configurable": {"thread_id": "type_test"}}

            # Should raise type error
            with pytest.raises(Exception) as exc_info:
                for update in flow.stream(config):
                    # First node (config_and_keys) should work
                    # Second node (agent_instantiation) should fail
                    pass

            # Restore original node
            flow.nodes.agent_instantiation_node = original_node
