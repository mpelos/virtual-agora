"""Tests for AgendaVotingNode."""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from virtual_agora.flow.nodes.agenda.voting import AgendaVotingNode
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
            "messages": [Mock(content=f"My vote preference from agent {i+1}")]
        }
        agents.append(agent)
    return agents


@pytest.fixture
def sample_state_with_topics():
    """Create sample VirtualAgoraState with topic queue."""
    return VirtualAgoraState(
        session_id="test_session",
        main_topic="AI Ethics",
        current_phase=1,
        topic_queue=[
            "Ethics foundations",
            "Privacy concerns",
            "Fairness in AI",
            "Transparency issues",
        ],
        votes=[],
        messages=[],
    )


@pytest.fixture
def voting_node(mock_discussing_agents):
    """Create AgendaVotingNode instance."""
    return AgendaVotingNode(mock_discussing_agents)


class TestAgendaVotingNode:
    """Test the AgendaVotingNode class."""

    def test_initialization(self, mock_discussing_agents):
        """Test node initialization."""
        node = AgendaVotingNode(mock_discussing_agents)
        assert node.discussing_agents == mock_discussing_agents
        assert node.get_node_name() == "AgendaVoting"

    def test_validate_preconditions_success(
        self, voting_node, sample_state_with_topics
    ):
        """Test successful precondition validation."""
        assert voting_node.validate_preconditions(sample_state_with_topics) is True

    def test_validate_preconditions_no_topics(self, voting_node):
        """Test precondition validation fails without topics."""
        state = VirtualAgoraState(session_id="test", messages=[])
        assert voting_node.validate_preconditions(state) is False

    def test_validate_preconditions_empty_topic_queue(self, voting_node):
        """Test precondition validation fails with empty topic queue."""
        state = VirtualAgoraState(session_id="test", topic_queue=[], messages=[])
        assert voting_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_agents(self, sample_state_with_topics):
        """Test precondition validation fails without agents."""
        node = AgendaVotingNode([])
        assert node.validate_preconditions(sample_state_with_topics) is False

    @patch("uuid.uuid4")
    @patch("virtual_agora.flow.nodes.agenda.voting.datetime")
    def test_execute_success(
        self, mock_datetime, mock_uuid, voting_node, sample_state_with_topics
    ):
        """Test successful agenda voting."""
        # Mock datetime and uuid for consistent test output
        mock_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_time

        mock_uuids = [Mock() for _ in range(3)]
        for i, mock_uuid_obj in enumerate(mock_uuids):
            mock_uuid_obj.hex = f"abcd123{i}"
        mock_uuid.return_value.__getitem__ = lambda self, key: (
            mock_uuids[0] if key == slice(None, 8, None) else None
        )
        mock_uuid.side_effect = mock_uuids

        result = voting_node.execute(sample_state_with_topics)

        # Check state updates
        assert "votes" in result
        assert "current_phase" in result
        assert result["current_phase"] == 2

        # Check votes structure
        votes = result["votes"]
        assert len(votes) == 3

        for i, vote in enumerate(votes):
            assert "id" in vote
            assert vote["voter_id"] == f"gpt-4o-{i+1}"
            assert vote["phase"] == 1
            assert vote["vote_type"] == "topic_selection"
            assert vote["timestamp"] == mock_time
            assert f"My vote preference from agent {i+1}" in vote["choice"]

    def test_execute_with_agent_failure(
        self, mock_discussing_agents, sample_state_with_topics
    ):
        """Test voting with one agent failure."""
        # Make second agent fail
        mock_discussing_agents[1].side_effect = Exception("Agent failure")

        node = AgendaVotingNode(mock_discussing_agents)
        result = node.execute(sample_state_with_topics)

        # Check state updates
        assert "votes" in result
        votes = result["votes"]
        assert len(votes) == 3

        # Check successful agents
        assert votes[0]["voter_id"] == "gpt-4o-1"
        assert "My vote preference" in votes[0]["choice"]

        # Check failed agent gets fallback vote
        assert votes[1]["voter_id"] == "gpt-4o-2"
        assert votes[1]["choice"] == "No preference"
        assert "metadata" in votes[1]
        assert "error" in votes[1]["metadata"]

        # Check third agent still works
        assert votes[2]["voter_id"] == "gpt-4o-3"
        assert "My vote preference" in votes[2]["choice"]

    def test_execute_formats_topics_correctly(
        self, voting_node, sample_state_with_topics
    ):
        """Test that topics are formatted correctly in the prompt."""
        result = voting_node.execute(sample_state_with_topics)

        # Check that each agent was called with formatted topics
        for i, agent in enumerate(voting_node.discussing_agents):
            assert agent.called
            args, kwargs = agent.call_args
            assert args[0] == sample_state_with_topics  # state argument
            assert "prompt" in kwargs
            prompt = kwargs["prompt"]

            # Check that prompt includes all topics formatted correctly
            assert "1. Ethics foundations" in prompt
            assert "2. Privacy concerns" in prompt
            assert "3. Fairness in AI" in prompt
            assert "4. Transparency issues" in prompt
            assert "Vote on your preferred discussion order" in prompt
            assert "Express your preferences in natural language" in prompt

    def test_execute_with_empty_messages(
        self, mock_discussing_agents, sample_state_with_topics
    ):
        """Test voting with empty response messages."""
        # Make first agent return empty messages
        mock_discussing_agents[0].return_value = {"messages": []}

        node = AgendaVotingNode(mock_discussing_agents)
        result = node.execute(sample_state_with_topics)

        # Check that voting continues, but agent with empty messages doesn't get a vote
        assert "votes" in result
        votes = result["votes"]
        assert len(votes) == 2  # Only agents with content are included

        # Check that votes are from agents 2 and 3
        voter_ids = [vote["voter_id"] for vote in votes]
        assert "gpt-4o-2" in voter_ids
        assert "gpt-4o-3" in voter_ids
        assert "gpt-4o-1" not in voter_ids

    def test_vote_object_structure(self, voting_node, sample_state_with_topics):
        """Test that vote objects have correct structure."""
        result = voting_node.execute(sample_state_with_topics)

        votes = result["votes"]
        for vote in votes:
            # Check required fields
            assert "id" in vote
            assert "voter_id" in vote
            assert "phase" in vote
            assert "vote_type" in vote
            assert "choice" in vote
            assert "timestamp" in vote

            # Check field types and values
            assert isinstance(vote["id"], str)
            assert vote["id"].startswith("vote_")
            assert vote["phase"] == 1
            assert vote["vote_type"] == "topic_selection"
            assert isinstance(vote["timestamp"], datetime)

    def test_execute_with_message_content_attribute(
        self, mock_discussing_agents, sample_state_with_topics
    ):
        """Test voting with messages that have content attribute."""
        # Set up agent with message object having content attribute
        message_obj = Mock()
        message_obj.content = "I prefer topic 1 first, then topic 3"
        mock_discussing_agents[0].return_value = {"messages": [message_obj]}

        node = AgendaVotingNode(mock_discussing_agents)
        result = node.execute(sample_state_with_topics)

        votes = result["votes"]
        first_vote = votes[0]
        assert first_vote["choice"] == "I prefer topic 1 first, then topic 3"

    def test_execute_with_string_messages(
        self, mock_discussing_agents, sample_state_with_topics
    ):
        """Test voting with string messages (no content attribute)."""
        # Set up agent with string message (no content attribute)
        mock_discussing_agents[0].return_value = {
            "messages": ["String vote preference"]
        }

        node = AgendaVotingNode(mock_discussing_agents)
        result = node.execute(sample_state_with_topics)

        votes = result["votes"]
        first_vote = votes[0]
        assert first_vote["choice"] == "String vote preference"

    def test_vote_id_uniqueness(self, voting_node, sample_state_with_topics):
        """Test that vote IDs are unique."""
        result = voting_node.execute(sample_state_with_topics)

        votes = result["votes"]
        vote_ids = [vote["id"] for vote in votes]

        # Check that all IDs are unique
        assert len(vote_ids) == len(set(vote_ids))

        # Check that all IDs follow the expected format
        for vote_id in vote_ids:
            assert vote_id.startswith("vote_")
            assert len(vote_id) == 13  # "vote_" + 8 hex characters
