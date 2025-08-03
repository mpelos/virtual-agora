"""Tests for AgendaProposalNode."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from virtual_agora.flow.nodes.agenda.proposal import (
    AgendaProposalNode,
    get_provider_type_from_agent_id,
)
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.providers.config import ProviderType


@pytest.fixture
def mock_discussing_agents():
    """Create mock discussing agents."""
    agents = []
    for i in range(3):
        agent = Mock(spec=LLMAgent)
        agent.agent_id = f"gpt-4o-{i+1}"
        agent.return_value = {
            "messages": [Mock(content=f"Mock proposal content from agent {i+1}")]
        }
        agents.append(agent)
    return agents


@pytest.fixture
def sample_state():
    """Create sample VirtualAgoraState."""
    return VirtualAgoraState(
        session_id="test_session",
        main_topic="AI Ethics",
        theme="AI Ethics",
        current_phase=0,
        proposed_topics=[],
        messages=[],
    )


@pytest.fixture
def proposal_node(mock_discussing_agents):
    """Create AgendaProposalNode instance."""
    return AgendaProposalNode(mock_discussing_agents)


class TestGetProviderTypeFromAgentId:
    """Test provider type extraction from agent ID."""

    def test_openai_detection(self):
        assert get_provider_type_from_agent_id("gpt-4o-1") == ProviderType.OPENAI
        assert get_provider_type_from_agent_id("openai-model-1") == ProviderType.OPENAI

    def test_anthropic_detection(self):
        assert (
            get_provider_type_from_agent_id("claude-3-opus-1") == ProviderType.ANTHROPIC
        )
        assert (
            get_provider_type_from_agent_id("anthropic-claude-1")
            == ProviderType.ANTHROPIC
        )

    def test_google_detection(self):
        assert (
            get_provider_type_from_agent_id("gemini-2.5-pro-1") == ProviderType.GOOGLE
        )
        assert get_provider_type_from_agent_id("google-model-1") == ProviderType.GOOGLE

    def test_grok_detection(self):
        assert get_provider_type_from_agent_id("grok-beta-1") == ProviderType.GROK

    def test_unknown_fallback(self):
        assert get_provider_type_from_agent_id("unknown-model-1") == ProviderType.OPENAI


class TestAgendaProposalNode:
    """Test the AgendaProposalNode class."""

    def test_initialization(self, mock_discussing_agents):
        """Test node initialization."""
        node = AgendaProposalNode(mock_discussing_agents)
        assert node.discussing_agents == mock_discussing_agents
        assert node.get_node_name() == "AgendaProposal"

    def test_validate_preconditions_success(self, proposal_node, sample_state):
        """Test successful precondition validation."""
        assert proposal_node.validate_preconditions(sample_state) is True

    def test_validate_preconditions_no_main_topic(self, proposal_node):
        """Test precondition validation fails without main topic."""
        state = VirtualAgoraState(session_id="test", messages=[])
        assert proposal_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_agents(self, sample_state):
        """Test precondition validation fails without agents."""
        node = AgendaProposalNode([])
        assert node.validate_preconditions(sample_state) is False

    @patch("virtual_agora.flow.nodes.agenda.proposal.display_agent_response")
    def test_execute_success(self, mock_display, proposal_node, sample_state):
        """Test successful proposal collection."""
        result = proposal_node.execute(sample_state)

        # Check state updates
        assert "proposed_topics" in result
        assert "current_phase" in result
        assert "phase_start_time" in result
        assert result["current_phase"] == 1

        # Check proposals collected
        proposals = result["proposed_topics"]
        assert len(proposals) == 3

        for i, proposal in enumerate(proposals):
            assert proposal["agent_id"] == f"gpt-4o-{i+1}"
            assert "proposals" in proposal
            assert f"Mock proposal content from agent {i+1}" in proposal["proposals"]

        # Check display was called for each agent
        assert mock_display.call_count == 3

    @patch("virtual_agora.flow.nodes.agenda.proposal.display_agent_response")
    def test_execute_with_agent_failure(
        self, mock_display, mock_discussing_agents, sample_state
    ):
        """Test proposal collection with one agent failure."""
        # Make second agent fail
        mock_discussing_agents[1].side_effect = Exception("Agent failure")

        node = AgendaProposalNode(mock_discussing_agents)
        result = node.execute(sample_state)

        # Check state updates
        assert "proposed_topics" in result
        proposals = result["proposed_topics"]
        assert len(proposals) == 3

        # Check successful agents
        assert proposals[0]["agent_id"] == "gpt-4o-1"
        assert "Mock proposal content" in proposals[0]["proposals"]

        # Check failed agent
        assert proposals[1]["agent_id"] == "gpt-4o-2"
        assert proposals[1]["proposals"] == "Failed to provide proposals"
        assert "error" in proposals[1]

        # Check third agent still works
        assert proposals[2]["agent_id"] == "gpt-4o-3"
        assert "Mock proposal content" in proposals[2]["proposals"]

        # Check display was called for all agents (including error display)
        assert mock_display.call_count == 3

    @patch("virtual_agora.flow.nodes.agenda.proposal.display_agent_response")
    def test_execute_with_empty_messages(
        self, mock_display, mock_discussing_agents, sample_state
    ):
        """Test proposal collection with empty response messages."""
        # Make first agent return empty messages
        mock_discussing_agents[0].return_value = {"messages": []}

        node = AgendaProposalNode(mock_discussing_agents)
        result = node.execute(sample_state)

        # Check that proposal collection continues
        assert "proposed_topics" in result
        proposals = result["proposed_topics"]

        # First agent should have no proposals due to empty messages
        # Other agents should still work
        assert len(proposals) == 2  # Only agents with content are added

    def test_execute_creates_proper_prompt(self, proposal_node, sample_state):
        """Test that the prompt includes the theme correctly."""
        result = proposal_node.execute(sample_state)

        # Check that each agent was called
        for agent in proposal_node.discussing_agents:
            assert agent.called
            args, kwargs = agent.call_args
            assert args[0] == sample_state  # state argument
            assert "prompt" in kwargs
            assert "AI Ethics" in kwargs["prompt"]  # theme should be in prompt
            assert "3-5 strategic topics" in kwargs["prompt"]
