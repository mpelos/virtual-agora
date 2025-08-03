"""Tests for TopicRefinementNode."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from virtual_agora.flow.nodes.agenda.refinement import TopicRefinementNode
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent


@pytest.fixture
def mock_discussing_agents():
    """Create mock discussing agents."""
    agents = []
    for i in range(3):
        agent = Mock(spec=LLMAgent)
        agent.agent_id = f"gpt-4o-{i+1}"
        agent.return_value = {
            "messages": [Mock(content=f"Refined proposal content from agent {i+1}")]
        }
        agents.append(agent)
    return agents


@pytest.fixture
def sample_state_with_proposals():
    """Create sample VirtualAgoraState with initial proposals."""
    return VirtualAgoraState(
        session_id="test_session",
        main_topic="AI Ethics",
        theme="AI Ethics",
        current_phase=1,
        proposed_topics=[
            {"agent_id": "gpt-4o-1", "proposals": "Initial proposal 1"},
            {"agent_id": "gpt-4o-2", "proposals": "Initial proposal 2"},
            {"agent_id": "gpt-4o-3", "proposals": "Initial proposal 3"},
        ],
        messages=[],
    )


@pytest.fixture
def refinement_node(mock_discussing_agents):
    """Create TopicRefinementNode instance."""
    return TopicRefinementNode(mock_discussing_agents)


class TestTopicRefinementNode:
    """Test the TopicRefinementNode class."""

    def test_initialization(self, mock_discussing_agents):
        """Test node initialization."""
        node = TopicRefinementNode(mock_discussing_agents)
        assert node.discussing_agents == mock_discussing_agents
        assert node.get_node_name() == "TopicRefinement"

    def test_validate_preconditions_success(
        self, refinement_node, sample_state_with_proposals
    ):
        """Test successful precondition validation."""
        assert (
            refinement_node.validate_preconditions(sample_state_with_proposals) is True
        )

    def test_validate_preconditions_no_proposals(self, refinement_node):
        """Test precondition validation fails without proposals."""
        state = VirtualAgoraState(
            session_id="test", main_topic="AI Ethics", messages=[]
        )
        assert refinement_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_main_topic(self, refinement_node):
        """Test precondition validation fails without main topic."""
        state = VirtualAgoraState(
            session_id="test",
            proposed_topics=[{"agent_id": "test", "proposals": "test"}],
            messages=[],
        )
        assert refinement_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_agents(self, sample_state_with_proposals):
        """Test precondition validation fails without agents."""
        node = TopicRefinementNode([])
        assert node.validate_preconditions(sample_state_with_proposals) is False

    @patch("virtual_agora.flow.nodes.agenda.refinement.display_agent_response")
    def test_execute_success(
        self, mock_display, refinement_node, sample_state_with_proposals
    ):
        """Test successful topic refinement."""
        result = refinement_node.execute(sample_state_with_proposals)

        # Check state updates
        assert "proposed_topics" in result
        assert "initial_proposals" in result
        assert "refinement_completed" in result
        assert result["refinement_completed"] is True

        # Check that initial proposals are preserved
        assert (
            result["initial_proposals"]
            == sample_state_with_proposals["proposed_topics"]
        )

        # Check refined proposals
        refined_proposals = result["proposed_topics"]
        assert len(refined_proposals) == 3

        for i, proposal in enumerate(refined_proposals):
            assert proposal["agent_id"] == f"gpt-4o-{i+1}"
            assert "proposals" in proposal
            assert f"Refined proposal content from agent {i+1}" in proposal["proposals"]

        # Check display was called for each agent
        assert mock_display.call_count == 3

    @patch("virtual_agora.flow.nodes.agenda.refinement.display_agent_response")
    def test_execute_with_agent_failure(
        self, mock_display, mock_discussing_agents, sample_state_with_proposals
    ):
        """Test refinement with one agent failure."""
        # Make second agent fail
        mock_discussing_agents[1].side_effect = Exception("Agent failure")

        node = TopicRefinementNode(mock_discussing_agents)
        result = node.execute(sample_state_with_proposals)

        # Check state updates
        assert "proposed_topics" in result
        assert "refinement_completed" in result

        refined_proposals = result["proposed_topics"]
        assert len(refined_proposals) == 3

        # Check successful agents
        assert refined_proposals[0]["agent_id"] == "gpt-4o-1"
        assert "Refined proposal content" in refined_proposals[0]["proposals"]

        # Check failed agent falls back to original proposal
        assert refined_proposals[1]["agent_id"] == "gpt-4o-2"
        assert refined_proposals[1]["proposals"] == "Initial proposal 2"

        # Check third agent still works
        assert refined_proposals[2]["agent_id"] == "gpt-4o-3"
        assert "Refined proposal content" in refined_proposals[2]["proposals"]

        # Check display was called for all agents (including error display)
        assert mock_display.call_count == 3

    def test_execute_creates_proper_prompt(
        self, refinement_node, sample_state_with_proposals
    ):
        """Test that the prompt includes all proposals correctly."""
        result = refinement_node.execute(sample_state_with_proposals)

        # Check that each agent was called
        for i, agent in enumerate(refinement_node.discussing_agents):
            assert agent.called
            args, kwargs = agent.call_args
            assert args[0] == sample_state_with_proposals  # state argument
            assert "prompt" in kwargs
            prompt = kwargs["prompt"]

            # Check that prompt includes theme and all proposals
            assert "AI Ethics" in prompt
            assert "ALL INITIAL PROPOSALS:" in prompt
            assert "Initial proposal 1" in prompt
            assert "Initial proposal 2" in prompt
            assert "Initial proposal 3" in prompt
            assert f"Initial proposal {i+1}" in prompt  # Agent's own proposal
            assert "COLLABORATIVE REFINEMENT TASK:" in prompt

    @patch("virtual_agora.flow.nodes.agenda.refinement.display_agent_response")
    def test_execute_with_empty_messages(
        self, mock_display, mock_discussing_agents, sample_state_with_proposals
    ):
        """Test refinement with empty response messages."""
        # Make first agent return empty messages
        mock_discussing_agents[0].return_value = {"messages": []}

        node = TopicRefinementNode(mock_discussing_agents)
        result = node.execute(sample_state_with_proposals)

        # Check that refinement continues
        assert "proposed_topics" in result
        refined_proposals = result["proposed_topics"]

        # First agent should fall back to original proposal due to empty messages
        # Other agents should still work
        assert len(refined_proposals) == 2  # Only agents with content are added

    def test_execute_preserves_initial_proposals(
        self, refinement_node, sample_state_with_proposals
    ):
        """Test that initial proposals are preserved in state."""
        original_proposals = sample_state_with_proposals["proposed_topics"]
        result = refinement_node.execute(sample_state_with_proposals)

        # Check that initial proposals are preserved exactly
        assert result["initial_proposals"] == original_proposals

        # Check that proposed_topics now contains refined versions
        assert result["proposed_topics"] != original_proposals
        assert len(result["proposed_topics"]) == len(original_proposals)

    def test_agent_finds_own_proposal(
        self, refinement_node, sample_state_with_proposals
    ):
        """Test that each agent can find its own initial proposal."""
        result = refinement_node.execute(sample_state_with_proposals)

        # Verify each agent was called with its own proposal in the prompt
        for i, agent in enumerate(refinement_node.discussing_agents):
            args, kwargs = agent.call_args
            prompt = kwargs["prompt"]
            expected_proposal = f"Initial proposal {i+1}"
            assert f"YOUR INITIAL PROPOSAL:\n{expected_proposal}" in prompt
