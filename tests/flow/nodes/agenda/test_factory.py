"""Tests for AgendaNodeFactory."""

import pytest
from unittest.mock import Mock

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


class TestAgendaNodeFactory:
    """Test the AgendaNodeFactory class."""

    def test_initialization_success(self, mock_moderator_agent, mock_discussing_agents):
        """Test successful factory initialization."""
        factory = AgendaNodeFactory(
            moderator_agent=mock_moderator_agent,
            discussing_agents=mock_discussing_agents,
        )

        assert factory.moderator_agent == mock_moderator_agent
        assert factory.discussing_agents == mock_discussing_agents

    def test_initialization_no_moderator(self, mock_discussing_agents):
        """Test initialization fails without moderator agent."""
        with pytest.raises(ValueError, match="Moderator agent is required"):
            AgendaNodeFactory(
                moderator_agent=None, discussing_agents=mock_discussing_agents
            )

    def test_initialization_no_discussing_agents(self, mock_moderator_agent):
        """Test initialization fails without discussing agents."""
        with pytest.raises(
            ValueError, match="At least one discussing agent is required"
        ):
            AgendaNodeFactory(
                moderator_agent=mock_moderator_agent, discussing_agents=[]
            )

        with pytest.raises(
            ValueError, match="At least one discussing agent is required"
        ):
            AgendaNodeFactory(
                moderator_agent=mock_moderator_agent, discussing_agents=None
            )

    def test_initialization_invalid_moderator_type(self, mock_discussing_agents):
        """Test initialization fails with invalid moderator type."""
        with pytest.raises(ValueError, match="Expected ModeratorAgent"):
            AgendaNodeFactory(
                moderator_agent="not_an_agent", discussing_agents=mock_discussing_agents
            )

    def test_initialization_invalid_discussing_agent_type(self, mock_moderator_agent):
        """Test initialization fails with invalid discussing agent type."""
        invalid_agents = [Mock(spec=LLMAgent), "not_an_agent", Mock(spec=LLMAgent)]

        with pytest.raises(ValueError, match="Expected LLMAgent at index 1"):
            AgendaNodeFactory(
                moderator_agent=mock_moderator_agent, discussing_agents=invalid_agents
            )

    def test_create_agenda_proposal_node(self, agenda_factory):
        """Test creation of agenda proposal node."""
        node = agenda_factory.create_agenda_proposal_node()

        assert isinstance(node, AgendaProposalNode)
        assert node.discussing_agents == agenda_factory.discussing_agents

    def test_create_topic_refinement_node(self, agenda_factory):
        """Test creation of topic refinement node."""
        node = agenda_factory.create_topic_refinement_node()

        assert isinstance(node, TopicRefinementNode)
        assert node.discussing_agents == agenda_factory.discussing_agents

    def test_create_collate_proposals_node(self, agenda_factory):
        """Test creation of collate proposals node."""
        node = agenda_factory.create_collate_proposals_node()

        assert isinstance(node, CollateProposalsNode)
        assert node.moderator_agent == agenda_factory.moderator_agent

    def test_create_agenda_voting_node(self, agenda_factory):
        """Test creation of agenda voting node."""
        node = agenda_factory.create_agenda_voting_node()

        assert isinstance(node, AgendaVotingNode)
        assert node.discussing_agents == agenda_factory.discussing_agents

    def test_create_synthesize_agenda_node(self, agenda_factory):
        """Test creation of synthesize agenda node."""
        node = agenda_factory.create_synthesize_agenda_node()

        assert isinstance(node, SynthesizeAgendaNode)
        assert node.moderator_agent == agenda_factory.moderator_agent

    def test_create_agenda_approval_node(self, agenda_factory):
        """Test creation of agenda approval node returns None (not yet extracted)."""
        node = agenda_factory.create_agenda_approval_node()

        # Should return None since AgendaApprovalNode is not yet extracted
        assert node is None

    def test_create_all_agenda_nodes(self, agenda_factory):
        """Test creation of all agenda nodes."""
        nodes = agenda_factory.create_all_agenda_nodes()

        # Check all expected nodes are present
        expected_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        assert set(nodes.keys()) == set(expected_nodes)

        # Check extracted nodes are properly instantiated
        assert isinstance(nodes["agenda_proposal"], AgendaProposalNode)
        assert isinstance(nodes["topic_refinement"], TopicRefinementNode)
        assert isinstance(nodes["collate_proposals"], CollateProposalsNode)
        assert isinstance(nodes["agenda_voting"], AgendaVotingNode)
        assert isinstance(nodes["synthesize_agenda"], SynthesizeAgendaNode)

        # Check agenda_approval returns None (not yet extracted)
        assert nodes["agenda_approval"] is None

    def test_get_node_metadata(self, agenda_factory):
        """Test retrieval of node metadata."""
        metadata = agenda_factory.get_node_metadata()

        # Check all expected nodes have metadata
        expected_nodes = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
            "agenda_approval",
        ]

        assert set(metadata.keys()) == set(expected_nodes)

        # Check extracted nodes have correct metadata
        for node_name in [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
        ]:
            node_meta = metadata[node_name]
            assert node_meta["extracted"] is True
            assert node_meta["phase"] == 1
            assert "dependencies" in node_meta
            assert "description" in node_meta

        # Check agenda_approval has fallback metadata
        approval_meta = metadata["agenda_approval"]
        assert approval_meta["extracted"] is False
        assert approval_meta["dependencies"] == ["v13_wrapper"]
        assert "V13 wrapper" in approval_meta["description"]

    def test_validate_dependencies(self, agenda_factory):
        """Test dependency validation."""
        deps = agenda_factory.validate_dependencies()

        # All dependencies should be valid with mocked agents
        assert deps["moderator_agent_available"] is True
        assert deps["moderator_agent_valid"] is True
        assert deps["discussing_agents_available"] is True
        assert deps["discussing_agents_count"] == 3
        assert deps["all_agents_valid"] is True

    def test_validate_dependencies_invalid_moderator(self, mock_discussing_agents):
        """Test dependency validation with invalid moderator."""
        # Create factory with invalid moderator type (bypass __init__ validation)
        factory = AgendaNodeFactory.__new__(AgendaNodeFactory)
        factory.moderator_agent = "not_an_agent"
        factory.discussing_agents = mock_discussing_agents

        deps = factory.validate_dependencies()

        assert deps["moderator_agent_available"] is True  # Not None
        assert deps["moderator_agent_valid"] is False  # Wrong type
        assert deps["discussing_agents_available"] is True
        assert deps["all_agents_valid"] is True

    def test_validate_dependencies_invalid_discussing_agents(
        self, mock_moderator_agent
    ):
        """Test dependency validation with invalid discussing agents."""
        # Create factory with mixed valid/invalid agents (bypass __init__ validation)
        factory = AgendaNodeFactory.__new__(AgendaNodeFactory)
        factory.moderator_agent = mock_moderator_agent
        factory.discussing_agents = [Mock(spec=LLMAgent), "not_an_agent"]

        deps = factory.validate_dependencies()

        assert deps["moderator_agent_available"] is True
        assert deps["moderator_agent_valid"] is True
        assert deps["discussing_agents_available"] is True
        assert deps["discussing_agents_count"] == 2
        assert deps["all_agents_valid"] is False  # One invalid agent

    def test_get_factory_summary(self, agenda_factory):
        """Test factory summary generation."""
        summary = agenda_factory.get_factory_summary()

        # Check summary structure
        assert "total_agenda_nodes" in summary
        assert "extracted_nodes" in summary
        assert "extracted_count" in summary
        assert "fallback_nodes" in summary
        assert "fallback_count" in summary
        assert "dependencies_valid" in summary
        assert "dependency_status" in summary
        assert "moderator_agent_id" in summary
        assert "discussing_agent_ids" in summary

        # Check summary content
        assert summary["total_agenda_nodes"] == 6
        assert summary["extracted_count"] == 5
        assert summary["fallback_count"] == 1
        assert summary["fallback_nodes"] == ["agenda_approval"]
        assert summary["dependencies_valid"] is True
        assert summary["moderator_agent_id"] == "test_moderator"
        assert len(summary["discussing_agent_ids"]) == 3

        # Check extracted nodes list
        expected_extracted = [
            "agenda_proposal",
            "topic_refinement",
            "collate_proposals",
            "agenda_voting",
            "synthesize_agenda",
        ]
        assert set(summary["extracted_nodes"]) == set(expected_extracted)

    def test_factory_integration_with_real_node_creation(self, agenda_factory):
        """Integration test: verify created nodes can be executed."""
        nodes = agenda_factory.create_all_agenda_nodes()

        # Check that extracted nodes have proper interfaces
        for node_name, node in nodes.items():
            if node is not None:  # Skip None values (fallback nodes)
                # Verify FlowNode interface
                assert hasattr(node, "execute")
                assert hasattr(node, "validate_preconditions")
                assert hasattr(node, "get_node_name")

                # Verify methods are callable
                assert callable(node.execute)
                assert callable(node.validate_preconditions)
                assert callable(node.get_node_name)

                # Check node name is reasonable
                node_name_result = node.get_node_name()
                assert isinstance(node_name_result, str)
                assert len(node_name_result) > 0

    def test_factory_thread_safety(self, agenda_factory):
        """Test that factory can be used safely from multiple contexts."""
        import concurrent.futures

        def create_node():
            return agenda_factory.create_agenda_proposal_node()

        # Create nodes concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_node) for _ in range(5)]
            nodes = [future.result() for future in futures]

        # All nodes should be created successfully
        assert len(nodes) == 5
        assert all(isinstance(node, AgendaProposalNode) for node in nodes)

        # Each node should be a separate instance
        assert len(set(id(node) for node in nodes)) == 5
