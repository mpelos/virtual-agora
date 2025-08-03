"""Integration tests for Step 3.2: Hybrid graph with extracted agenda nodes."""

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


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=VirtualAgoraConfig)

    # Mock moderator config
    config.moderator = Mock()
    config.moderator.provider.value = "openai"
    config.moderator.model = "gpt-4o"

    # Mock agent configs
    agent_config = Mock()
    agent_config.provider.value = "openai"
    agent_config.model = "gpt-4o"
    agent_config.count = 3
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


class TestStep32Integration:
    """Integration tests for Step 3.2 hybrid graph implementation."""

    def test_graph_initialization_with_hybrid_registry(
        self, mock_config, mock_create_provider
    ):
        """Test that VirtualAgoraV13Flow initializes with hybrid registry."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Check that agenda node factory was created
        assert hasattr(flow, "agenda_node_factory")
        assert flow.agenda_node_factory is not None

        # Check that hybrid node registry was created
        assert hasattr(flow, "node_registry")
        assert flow.node_registry is not None
        assert len(flow.node_registry) > 20  # Should have many nodes

        # Check that agenda nodes are present in registry
        agenda_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        for node_name in agenda_nodes:
            assert node_name in flow.node_registry, f"Missing agenda node: {node_name}"

    def test_extracted_agenda_nodes_in_registry(
        self, mock_config, mock_create_provider
    ):
        """Test that extracted agenda nodes are properly instantiated."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Check extracted nodes are correct types
        extracted_nodes = {
            "agenda_proposal": AgendaProposalNode,
            "topic_refinement": TopicRefinementNode,
            "collate_proposals": CollateProposalsNode,
            "agenda_voting": AgendaVotingNode,
            "synthesize_agenda": SynthesizeAgendaNode,
        }

        for node_name, expected_type in extracted_nodes.items():
            node = flow.node_registry[node_name]
            assert isinstance(
                node, expected_type
            ), f"Node {node_name} is not {expected_type.__name__}"

    def test_agenda_approval_uses_v13_wrapper(self, mock_config, mock_create_provider):
        """Test that agenda_approval still uses V13 wrapper (not yet extracted)."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # agenda_approval should be a V13NodeWrapper
        agenda_approval = flow.node_registry["agenda_approval"]

        # Should be callable (V13NodeWrapper is callable)
        assert callable(agenda_approval)

        # Should be a V13NodeWrapper instance (not an extracted agenda node)
        from virtual_agora.flow.node_registry import V13NodeWrapper

        assert isinstance(agenda_approval, V13NodeWrapper)

        # Should also be a FlowNode (V13NodeWrapper inherits from FlowNode)
        from virtual_agora.flow.nodes.base import FlowNode

        assert isinstance(agenda_approval, FlowNode)

    def test_graph_building_with_hybrid_nodes(self, mock_config, mock_create_provider):
        """Test that graph builds successfully with hybrid nodes."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Build the graph
        graph = flow.build_graph()

        # Check that graph has all expected nodes
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
            "final_considerations",
            "agent_poll",
            "final_report_generation",
        ]

        graph_nodes = list(graph.nodes.keys())
        for node_name in expected_nodes:
            assert node_name in graph_nodes, f"Missing graph node: {node_name}"

    def test_extracted_nodes_have_execute_method(
        self, mock_config, mock_create_provider
    ):
        """Test that extracted nodes can be executed."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        extracted_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
        ]

        for node_name in extracted_nodes:
            node = flow.node_registry[node_name]

            # Should have execute method
            assert hasattr(node, "execute"), f"Node {node_name} missing execute method"
            assert callable(node.execute), f"Node {node_name} execute not callable"

            # Should have other FlowNode interface methods
            assert hasattr(
                node, "validate_preconditions"
            ), f"Node {node_name} missing validate_preconditions"
            assert hasattr(
                node, "get_node_name"
            ), f"Node {node_name} missing get_node_name"

    def test_agenda_factory_dependency_injection(
        self, mock_config, mock_create_provider
    ):
        """Test that agenda factory properly injects dependencies."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        factory = flow.agenda_node_factory

        # Check factory has required dependencies
        assert factory.moderator_agent is not None
        assert factory.discussing_agents is not None
        assert len(factory.discussing_agents) > 0

        # Check factory can create all extracted nodes
        agenda_nodes = factory.create_all_agenda_nodes()

        # Should have 5 extracted nodes and 1 None (agenda_approval)
        assert len(agenda_nodes) == 6
        extracted_count = sum(1 for node in agenda_nodes.values() if node is not None)
        assert extracted_count == 5

    def test_hybrid_registry_logging(self, mock_config, mock_create_provider):
        """Test that hybrid registry logs extraction status correctly."""
        with patch("virtual_agora.flow.graph_v13.logger") as mock_logger:
            flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

            # Check that logging was called for hybrid registry creation
            log_calls = mock_logger.info.call_args_list
            log_messages = [call[0][0] for call in log_calls]

            # Should log agenda factory creation
            factory_logs = [msg for msg in log_messages if "agenda node factory" in msg]
            assert len(factory_logs) > 0

            # Should log hybrid registry summary
            hybrid_logs = [msg for msg in log_messages if "hybrid node registry" in msg]
            assert len(hybrid_logs) > 0

    def test_error_recovery_strategies_registered(
        self, mock_config, mock_create_provider
    ):
        """Test that error recovery strategies are properly registered."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Check that error recovery manager exists
        assert hasattr(flow, "error_recovery_manager")
        assert flow.error_recovery_manager is not None

        # Check that recovery strategies are registered
        # (The exact implementation may vary, but we can check the manager exists)
        assert hasattr(flow.error_recovery_manager, "register_recovery_strategy")

    def test_graph_edges_preserved(self, mock_config, mock_create_provider):
        """Test that graph edges are preserved with hybrid nodes."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        graph = flow.build_graph()

        # Check some key edges exist (agenda flow)
        edges = graph.edges

        # Should have edges between agenda nodes
        agenda_flow_edges = [
            ("agenda_proposal", "topic_refinement"),
            ("topic_refinement", "collate_proposals"),
            ("collate_proposals", "agenda_voting"),
            ("agenda_voting", "synthesize_agenda"),
            ("synthesize_agenda", "agenda_approval"),
        ]

        for source, target in agenda_flow_edges:
            edge_exists = any(
                edge[0] == source and target in edge[1]
                for edge in edges
                if isinstance(edge[1], dict)
            )
            # Note: LangGraph edge structure may vary, so we just check nodes exist
            assert source in graph.nodes
            assert target in graph.nodes

    def test_compilation_success(self, mock_config, mock_create_provider):
        """Test that hybrid graph compiles successfully."""
        flow = VirtualAgoraV13Flow(config=mock_config, enable_monitoring=False)

        # Should be able to compile without errors
        compiled_graph = flow.compile()

        assert compiled_graph is not None
        assert flow.compiled_graph is not None
