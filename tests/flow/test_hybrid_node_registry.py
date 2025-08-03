"""Tests for hybrid node registry functionality."""

import pytest
from unittest.mock import Mock

from virtual_agora.flow.node_registry import (
    NodeRegistry,
    create_hybrid_v13_registry,
    _register_v13_agenda_node,
    _register_all_v13_agenda_nodes,
    _register_remaining_v13_nodes,
)
from virtual_agora.flow.nodes.agenda.factory import AgendaNodeFactory
from virtual_agora.flow.nodes.agenda import (
    AgendaProposalNode,
    TopicRefinementNode,
    CollateProposalsNode,
    AgendaVotingNode,
    SynthesizeAgendaNode,
)
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.llm_agent import LLMAgent


@pytest.fixture
def mock_moderator_agent():
    """Create mock moderator agent."""
    agent = Mock(spec=ModeratorAgent)
    agent.agent_id = "test_moderator"
    return agent


@pytest.fixture
def mock_discussing_agents():
    """Create mock discussing agents."""
    agents = []
    for i in range(3):
        agent = Mock(spec=LLMAgent)
        agent.agent_id = f"test_agent_{i+1}"
        agents.append(agent)
    return agents


@pytest.fixture
def agenda_factory(mock_moderator_agent, mock_discussing_agents):
    """Create AgendaNodeFactory with mock dependencies."""
    return AgendaNodeFactory(
        moderator_agent=mock_moderator_agent, discussing_agents=mock_discussing_agents
    )


@pytest.fixture
def mock_v13_nodes():
    """Create mock V13FlowNodes instance."""
    v13_nodes = Mock()

    # Mock agenda node methods
    v13_nodes.agenda_proposal_node = Mock()
    v13_nodes.topic_refinement_node = Mock()
    v13_nodes.collate_proposals_node = Mock()
    v13_nodes.agenda_voting_node = Mock()
    v13_nodes.synthesize_agenda_node = Mock()
    v13_nodes.agenda_approval_node = Mock()

    # Mock initialization node methods
    v13_nodes.config_and_keys_node = Mock()
    v13_nodes.agent_instantiation_node = Mock()
    v13_nodes.get_theme_node = Mock()

    # Mock discussion node methods
    v13_nodes.announce_item_node = Mock()
    v13_nodes.discussion_round_node = Mock()
    v13_nodes.round_summarization_node = Mock()
    v13_nodes.round_threshold_check_node = Mock()
    v13_nodes.end_topic_poll_node = Mock()
    v13_nodes.vote_evaluation_node = Mock()
    v13_nodes.periodic_user_stop_node = Mock()
    v13_nodes.user_topic_conclusion_confirmation_node = Mock()

    # Mock conclusion node methods
    v13_nodes.final_considerations_node = Mock()
    v13_nodes.topic_report_generation_node = Mock()
    v13_nodes.topic_summary_generation_node = Mock()
    v13_nodes.file_output_node = Mock()

    # Mock continuation node methods
    v13_nodes.agent_poll_node = Mock()
    v13_nodes.user_approval_node = Mock()
    v13_nodes.agenda_modification_node = Mock()

    # Mock final report node methods
    v13_nodes.final_report_node = Mock()
    v13_nodes.multi_file_output_node = Mock()

    # Mock legacy node methods
    v13_nodes.user_turn_participation_node = Mock()

    return v13_nodes


@pytest.fixture
def mock_unified_discussion_node():
    """Create mock unified discussion node."""
    from virtual_agora.flow.nodes.base import FlowNode

    node = Mock(spec=FlowNode)
    node.execute = Mock()
    node.validate_preconditions = Mock()
    node.get_node_name = Mock(return_value="UnifiedDiscussionRound")
    return node


class TestHybridNodeRegistry:
    """Test hybrid node registry functionality."""

    def test_create_hybrid_registry_with_factory(self, mock_v13_nodes, agenda_factory):
        """Test creating hybrid registry with agenda factory."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=agenda_factory
        )

        # Check total node count
        all_nodes = registry.get_all_nodes()
        assert len(all_nodes) > 20  # Should have many nodes

        # Check extracted agenda nodes
        agenda_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
        ]

        for node_name in agenda_nodes:
            assert node_name in all_nodes
            node = all_nodes[node_name]
            metadata = registry._node_metadata[node_name]

            # Check it's an extracted node
            assert metadata.get("extracted") is True
            assert metadata.get("phase") == 1
            assert metadata.get("type") == "agenda_setting"

            # Check it's the correct type
            expected_types = {
                "agenda_proposal": AgendaProposalNode,
                "topic_refinement": TopicRefinementNode,
                "collate_proposals": CollateProposalsNode,
                "agenda_voting": AgendaVotingNode,
                "synthesize_agenda": SynthesizeAgendaNode,
            }
            assert isinstance(node, expected_types[node_name])

        # Check agenda_approval uses V13 wrapper (not yet extracted)
        assert "agenda_approval" in all_nodes
        approval_metadata = registry._node_metadata["agenda_approval"]
        assert approval_metadata.get("extracted") is False

    def test_create_hybrid_registry_without_factory(self, mock_v13_nodes):
        """Test creating hybrid registry without agenda factory."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=None
        )

        # All agenda nodes should use V13 wrappers
        agenda_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        for node_name in agenda_nodes:
            metadata = registry._node_metadata[node_name]
            assert metadata.get("extracted") is False
            assert metadata.get("phase") == 1
            assert metadata.get("type") == "agenda_setting"

    def test_create_hybrid_registry_with_unified_discussion(
        self, mock_v13_nodes, agenda_factory, mock_unified_discussion_node
    ):
        """Test creating hybrid registry with unified discussion node."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes,
            agenda_node_factory=agenda_factory,
            unified_discussion_node=mock_unified_discussion_node,
        )

        # Check discussion_round node
        all_nodes = registry.get_all_nodes()
        assert "discussion_round" in all_nodes
        assert all_nodes["discussion_round"] == mock_unified_discussion_node

        metadata = registry._node_metadata["discussion_round"]
        assert metadata.get("extracted") is True
        assert metadata.get("unified") is True
        assert metadata.get("phase") == 2

    def test_create_hybrid_registry_without_unified_discussion(
        self, mock_v13_nodes, agenda_factory
    ):
        """Test creating hybrid registry without unified discussion node."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes,
            agenda_node_factory=agenda_factory,
            unified_discussion_node=None,
        )

        # Check discussion_round uses V13 wrapper
        metadata = registry._node_metadata["discussion_round"]
        assert metadata.get("extracted") is False
        assert metadata.get("legacy") is True
        assert metadata.get("phase") == 2

    def test_hybrid_registry_initialization_nodes(self, mock_v13_nodes, agenda_factory):
        """Test that initialization nodes are properly wrapped."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=agenda_factory
        )

        init_nodes = ["config_and_keys", "agent_instantiation", "get_theme"]

        for node_name in init_nodes:
            metadata = registry._node_metadata[node_name]
            assert metadata.get("extracted") is False
            assert metadata.get("phase") == 0
            assert metadata.get("type") == "initialization"

    def test_hybrid_registry_discussion_nodes(self, mock_v13_nodes, agenda_factory):
        """Test that discussion nodes are properly wrapped."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=agenda_factory
        )

        discussion_nodes = [
            "announce_item",
            "round_summarization",
            "round_threshold_check",
            "end_topic_poll",
            "vote_evaluation",
        ]

        for node_name in discussion_nodes:
            metadata = registry._node_metadata[node_name]
            assert metadata.get("extracted") is False
            assert metadata.get("phase") == 2

    def test_hybrid_registry_all_phases_present(self, mock_v13_nodes, agenda_factory):
        """Test that all phases are represented in hybrid registry."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=agenda_factory
        )

        # Check that all phases 0-5 are present
        phases = set()
        for metadata in registry._node_metadata.values():
            phases.add(metadata.get("phase"))

        assert phases == {0, 1, 2, 3, 4, 5}

    def test_register_v13_agenda_node(self, mock_v13_nodes):
        """Test registering individual V13 agenda node."""
        registry = NodeRegistry()

        _register_v13_agenda_node(registry, mock_v13_nodes, "agenda_proposal")

        all_nodes = registry.get_all_nodes()
        assert "agenda_proposal" in all_nodes

        metadata = registry._node_metadata["agenda_proposal"]
        assert metadata.get("extracted") is False
        assert metadata.get("phase") == 1
        assert metadata.get("type") == "agenda_setting"

    def test_register_v13_agenda_node_invalid_name(self, mock_v13_nodes):
        """Test registering V13 agenda node with invalid name."""
        registry = NodeRegistry()

        # Should handle unknown node gracefully
        _register_v13_agenda_node(registry, mock_v13_nodes, "invalid_node")

        all_nodes = registry.get_all_nodes()
        assert "invalid_node" not in all_nodes

    def test_register_all_v13_agenda_nodes(self, mock_v13_nodes):
        """Test registering all V13 agenda nodes."""
        registry = NodeRegistry()

        _register_all_v13_agenda_nodes(registry, mock_v13_nodes)

        agenda_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        all_nodes = registry.get_all_nodes()
        for node_name in agenda_nodes:
            assert node_name in all_nodes
            metadata = registry._node_metadata[node_name]
            assert metadata.get("extracted") is False

    def test_register_remaining_v13_nodes(self, mock_v13_nodes):
        """Test registering remaining V13 nodes."""
        registry = NodeRegistry()

        _register_remaining_v13_nodes(registry, mock_v13_nodes)

        # Check some key nodes are present
        expected_nodes = [
            "announce_item",
            "discussion_round",
            "final_considerations",
            "agent_poll",
            "final_report_generation",
        ]

        all_nodes = registry.get_all_nodes()
        for node_name in expected_nodes:
            assert node_name in all_nodes
            metadata = registry._node_metadata[node_name]
            assert metadata.get("extracted") is False

    def test_hybrid_registry_extracted_count(self, mock_v13_nodes, agenda_factory):
        """Test counting extracted vs wrapped nodes."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=agenda_factory
        )

        extracted_count = 0
        wrapped_count = 0

        for metadata in registry._node_metadata.values():
            if metadata.get("extracted", False):
                extracted_count += 1
            else:
                wrapped_count += 1

        # Should have exactly 5 extracted agenda nodes
        assert extracted_count == 5

        # Should have many more wrapped nodes
        assert wrapped_count > 15

        # Total should match registry size
        assert extracted_count + wrapped_count == len(registry.get_all_nodes())

    def test_hybrid_registry_node_interface_compliance(
        self, mock_v13_nodes, agenda_factory
    ):
        """Test that all nodes in hybrid registry comply with FlowNode interface."""
        registry = create_hybrid_v13_registry(
            v13_nodes=mock_v13_nodes, agenda_node_factory=agenda_factory
        )

        all_nodes = registry.get_all_nodes()

        for node_name, node in all_nodes.items():
            # All nodes should have required methods
            assert hasattr(node, "execute"), f"Node {node_name} missing execute method"
            assert hasattr(
                node, "validate_preconditions"
            ), f"Node {node_name} missing validate_preconditions method"
            assert hasattr(
                node, "get_node_name"
            ), f"Node {node_name} missing get_node_name method"

            # Methods should be callable
            assert callable(node.execute), f"Node {node_name} execute not callable"
            assert callable(
                node.validate_preconditions
            ), f"Node {node_name} validate_preconditions not callable"
            assert callable(
                node.get_node_name
            ), f"Node {node_name} get_node_name not callable"
