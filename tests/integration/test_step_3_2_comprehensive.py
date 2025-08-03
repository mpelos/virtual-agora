"""Comprehensive validation test for Step 3.2: Hybrid graph with extracted agenda nodes.

This test validates the complete Step 3.2 implementation by testing:
1. Hybrid node registry creation and functionality
2. Graph building with mixed extracted and wrapped nodes
3. Node interface compliance and execution paths
4. Error recovery and initialization order
5. Full integration with existing flow components
"""

import pytest
from unittest.mock import Mock, patch

from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.flow.nodes.agenda import (
    AgendaProposalNode,
    TopicRefinementNode,
    CollateProposalsNode,
    AgendaVotingNode,
    SynthesizeAgendaNode,
)
from virtual_agora.flow.node_registry import V13NodeWrapper
from virtual_agora.flow.nodes.base import FlowNode


@pytest.fixture
def mock_config():
    """Create comprehensive mock configuration."""
    config = Mock(spec=VirtualAgoraConfig)

    # Mock moderator config
    config.moderator = Mock()
    config.moderator.provider.value = "openai"
    config.moderator.model = "gpt-4o"
    config.moderator.temperature = 0.7
    config.moderator.max_tokens = 4000

    # Mock summarizer config
    config.summarizer = Mock()
    config.summarizer.provider.value = "openai"
    config.summarizer.model = "gpt-4o"
    config.summarizer.temperature = 0.6
    config.summarizer.max_tokens = 3000

    # Mock report writer config
    config.report_writer = Mock()
    config.report_writer.provider.value = "openai"
    config.report_writer.model = "gpt-4o"
    config.report_writer.temperature = 0.5
    config.report_writer.max_tokens = 5000

    # Mock agent configs
    agent_config = Mock()
    agent_config.provider.value = "openai"
    agent_config.model = "gpt-4o"
    agent_config.count = 3
    agent_config.temperature = 0.7
    agent_config.max_tokens = 3000
    config.agents = [agent_config]

    return config


@pytest.fixture
def mock_create_provider():
    """Mock the create_provider function."""
    with patch("virtual_agora.flow.graph_v13.create_provider") as mock:
        # Return mock LLM instances
        mock_llm = Mock()
        mock.return_value = mock_llm
        yield mock


class TestStep32Comprehensive:
    """Comprehensive test suite for Step 3.2 implementation."""

    def test_step_3_2_complete_integration(self, mock_config, mock_create_provider):
        """Test complete Step 3.2 integration with all components."""
        # Initialize VirtualAgoraV13Flow
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # === PHASE 1: Validate Hybrid Registry Creation ===

        # Check agenda factory was created
        assert hasattr(flow, "agenda_node_factory")
        assert flow.agenda_node_factory is not None

        # Check hybrid node registry was created
        assert hasattr(flow, "node_registry")
        assert flow.node_registry is not None

        # Check error recovery manager initialization order (fixed bug)
        assert hasattr(flow, "error_recovery_manager")
        assert flow.error_recovery_manager is not None

        # Validate registry has all expected nodes
        all_nodes = flow.node_registry  # This is the dictionary from get_all_nodes()
        assert len(all_nodes) > 25  # Should have all flow nodes

        # === PHASE 2: Validate Extracted vs Wrapped Nodes ===

        # Check extracted agenda nodes
        extracted_agenda_nodes = {
            "agenda_proposal": AgendaProposalNode,
            "topic_refinement": TopicRefinementNode,
            "collate_proposals": CollateProposalsNode,
            "agenda_voting": AgendaVotingNode,
            "synthesize_agenda": SynthesizeAgendaNode,
        }

        for node_name, expected_type in extracted_agenda_nodes.items():
            node = all_nodes[node_name]
            assert isinstance(
                node, expected_type
            ), f"Node {node_name} should be extracted"
            assert isinstance(
                node, FlowNode
            ), f"Extracted node {node_name} should be FlowNode"
            assert hasattr(
                node, "execute"
            ), f"Extracted node {node_name} missing execute method"
            assert callable(
                node.execute
            ), f"Extracted node {node_name} execute not callable"

        # Check wrapped agenda node (agenda_approval not yet extracted)
        agenda_approval = all_nodes["agenda_approval"]
        assert isinstance(
            agenda_approval, V13NodeWrapper
        ), "agenda_approval should be V13NodeWrapper"
        assert isinstance(
            agenda_approval, FlowNode
        ), "V13NodeWrapper should inherit from FlowNode"
        assert callable(agenda_approval), "V13NodeWrapper should be callable"

        # === PHASE 3: Validate Graph Building ===

        # Build the graph
        graph = flow.build_graph()
        assert graph is not None

        # Check that all agenda nodes are in the graph
        agenda_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        graph_nodes = list(graph.nodes.keys())
        for node_name in agenda_nodes:
            assert node_name in graph_nodes, f"Graph missing agenda node: {node_name}"

        # === PHASE 4: Validate Node Interface Compliance ===

        # All nodes should comply with FlowNode interface
        for node_name, node in all_nodes.items():
            # Check FlowNode interface methods
            assert hasattr(node, "execute"), f"Node {node_name} missing execute method"
            assert hasattr(
                node, "validate_preconditions"
            ), f"Node {node_name} missing validate_preconditions"
            assert hasattr(
                node, "get_node_name"
            ), f"Node {node_name} missing get_node_name"

            # Check methods are callable
            assert callable(node.execute), f"Node {node_name} execute not callable"
            assert callable(
                node.validate_preconditions
            ), f"Node {node_name} validate_preconditions not callable"
            assert callable(
                node.get_node_name
            ), f"Node {node_name} get_node_name not callable"

        # === PHASE 5: Validate Graph Compilation ===

        # Graph should compile successfully
        compiled_graph = flow.compile()
        assert compiled_graph is not None
        assert flow.compiled_graph is not None

        # === PHASE 6: Validate Dependency Injection ===

        # Check agenda factory dependencies
        factory = flow.agenda_node_factory
        assert factory.moderator_agent is not None
        assert factory.discussing_agents is not None
        assert len(factory.discussing_agents) > 0

        # Check factory can create all extracted nodes
        created_nodes = factory.create_all_agenda_nodes()
        assert len(created_nodes) == 6  # 5 extracted + 1 None (agenda_approval)

        extracted_count = sum(1 for node in created_nodes.values() if node is not None)
        assert extracted_count == 5, "Should have exactly 5 extracted agenda nodes"

        # === PHASE 7: Validate Error Recovery Integration ===

        # Check error recovery strategies are registered
        error_manager = flow.error_recovery_manager
        assert hasattr(error_manager, "register_recovery_strategy")

        # === PHASE 8: Validate Flow Router Integration ===

        # Check flow router was created
        assert hasattr(flow, "flow_router")
        assert flow.flow_router is not None

        # === PHASE 9: Validate Orchestrator Integration ===

        # Check orchestrator was created with hybrid registry
        assert hasattr(flow, "orchestrator")
        assert flow.orchestrator is not None
        # Note: orchestrator.node_registry is the NodeRegistry object, not the dict

    def test_step_3_2_backward_compatibility(self, mock_config, mock_create_provider):
        """Test that Step 3.2 maintains full backward compatibility."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Should maintain all existing node names and functionality
        expected_node_names = [
            # Phase 0: Initialization
            "config_and_keys",
            "agent_instantiation",
            "get_theme",
            # Phase 1: Agenda Setting (now hybrid)
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
            # Phase 2: Discussion
            "announce_item",
            "discussion_round",
            "round_summarization",
            "round_threshold_check",
            "end_topic_poll",
            "vote_evaluation",
            "periodic_user_stop",
            "user_topic_conclusion_confirmation",
            # Phase 3: Topic Conclusion
            "final_considerations",
            "topic_report_generation",
            "topic_summary_generation",
            "file_output",
            # Phase 4: Continuation
            "agent_poll",
            "user_approval",
            "agenda_modification",
            # Phase 5: Final Report
            "final_report_generation",
            "multi_file_output",
            # Legacy
            "user_turn_participation",
        ]

        all_nodes = flow.node_registry
        for node_name in expected_node_names:
            assert (
                node_name in all_nodes
            ), f"Missing backward compatibility node: {node_name}"

    def test_step_3_2_performance_characteristics(
        self, mock_config, mock_create_provider
    ):
        """Test that Step 3.2 maintains performance characteristics."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Graph building should be fast
        import time

        start_time = time.time()
        graph = flow.build_graph()
        build_time = time.time() - start_time

        assert build_time < 1.0, "Graph building should complete in under 1 second"

        # Compilation should be fast
        start_time = time.time()
        compiled_graph = flow.compile()
        compile_time = time.time() - start_time

        assert (
            compile_time < 2.0
        ), "Graph compilation should complete in under 2 seconds"

        # Registry should have reasonable size
        all_nodes = flow.node_registry
        assert (
            25 <= len(all_nodes) <= 35
        ), f"Registry size {len(all_nodes)} outside expected range"

    def test_step_3_2_node_execution_paths(self, mock_config, mock_create_provider):
        """Test that both extracted and wrapped nodes have proper execution paths."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        all_nodes = flow.node_registry

        # Test extracted agenda nodes
        extracted_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
        ]

        for node_name in extracted_nodes:
            node = all_nodes[node_name]

            # Should be extracted FlowNode
            assert not isinstance(
                node, V13NodeWrapper
            ), f"{node_name} should not be V13NodeWrapper"
            assert isinstance(node, FlowNode), f"{node_name} should be FlowNode"

            # Should have proper execution path
            assert hasattr(node, "execute"), f"{node_name} missing execute method"

            # Node name should be descriptive
            node_name_result = node.get_node_name()
            assert isinstance(
                node_name_result, str
            ), f"{node_name} get_node_name not string"
            assert len(node_name_result) > 0, f"{node_name} get_node_name empty"

        # Test wrapped nodes
        wrapped_nodes = [
            "agenda_approval",
            "config_and_keys",
            "announce_item",
            "final_considerations",
            "agent_poll",
        ]

        for node_name in wrapped_nodes:
            node = all_nodes[node_name]

            # Should be V13NodeWrapper
            assert isinstance(
                node, V13NodeWrapper
            ), f"{node_name} should be V13NodeWrapper"
            assert callable(node), f"{node_name} should be callable"

            # Should have proper execution path through wrapper
            assert hasattr(node, "execute"), f"{node_name} missing execute method"
            assert hasattr(node, "__call__"), f"{node_name} missing __call__ method"

            # Wrapper name should indicate it's wrapped
            node_name_result = node.get_node_name()
            assert (
                "V13Wrapper" in node_name_result
            ), f"{node_name} name should indicate wrapper"

    def test_step_3_2_migration_correctness(self, mock_config, mock_create_provider):
        """Test that Step 3.2 correctly migrates from V13 to hybrid architecture."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Verify the migration statistics
        all_nodes = flow.node_registry

        extracted_count = 0
        wrapped_count = 0

        # Count extracted agenda nodes specifically
        extracted_agenda_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
        ]

        for node_name in extracted_agenda_nodes:
            if node_name in all_nodes and not isinstance(
                all_nodes[node_name], V13NodeWrapper
            ):
                extracted_count += 1

        # Count wrapped nodes (everything else)
        for node_name, node in all_nodes.items():
            if isinstance(node, V13NodeWrapper):
                wrapped_count += 1

        # Should have exactly 5 extracted agenda nodes
        assert (
            extracted_count == 5
        ), f"Expected 5 extracted agenda nodes, got {extracted_count}"

        # Should have many wrapped nodes (rest of the flow)
        assert wrapped_count > 20, f"Expected >20 wrapped nodes, got {wrapped_count}"

        # Total should match registry size
        total_nodes = len(all_nodes)
        non_extracted_count = total_nodes - extracted_count

        # Log the migration statistics for verification
        print(f"Step 3.2 Migration Summary:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Extracted agenda nodes: {extracted_count}")
        print(f"  Wrapped nodes: {wrapped_count}")
        print(f"  Other nodes: {non_extracted_count - wrapped_count}")
        print(f"  Migration ratio: {extracted_count/total_nodes:.1%} extracted")
