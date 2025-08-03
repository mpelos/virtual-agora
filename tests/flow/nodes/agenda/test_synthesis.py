"""Tests for SynthesizeAgendaNode."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from virtual_agora.flow.nodes.agenda.synthesis import SynthesizeAgendaNode
from virtual_agora.state.schema import VirtualAgoraState, Agenda
from virtual_agora.agents.moderator import ModeratorAgent


@pytest.fixture
def mock_moderator_agent():
    """Create mock moderator agent."""
    agent = Mock(spec=ModeratorAgent)
    agent.synthesize_agenda.return_value = {
        "proposed_agenda": ["Ethics foundations", "Fairness in AI", "Privacy concerns"]
    }
    return agent


@pytest.fixture
def sample_state_with_votes():
    """Create sample VirtualAgoraState with votes."""
    return VirtualAgoraState(
        session_id="test_session",
        main_topic="AI Ethics",
        current_phase=2,
        topic_queue=[
            "Ethics foundations",
            "Privacy concerns",
            "Fairness in AI",
            "Transparency issues",
            "Accountability",
        ],
        votes=[
            {
                "id": "vote_12345678",
                "voter_id": "gpt-4o-1",
                "phase": 1,
                "vote_type": "topic_selection",
                "choice": "I prefer Ethics foundations first, then Fairness in AI",
                "timestamp": datetime.now(),
            },
            {
                "id": "vote_87654321",
                "voter_id": "gpt-4o-2",
                "phase": 1,
                "vote_type": "topic_selection",
                "choice": "Fairness in AI should be priority, then Privacy concerns",
                "timestamp": datetime.now(),
            },
            {
                "id": "vote_11223344",
                "voter_id": "gpt-4o-3",
                "phase": 1,
                "vote_type": "topic_selection",
                "choice": "Ethics foundations, Privacy concerns, Fairness in AI",
                "timestamp": datetime.now(),
            },
        ],
        messages=[],
    )


@pytest.fixture
def synthesis_node(mock_moderator_agent):
    """Create SynthesizeAgendaNode instance."""
    return SynthesizeAgendaNode(mock_moderator_agent)


class TestSynthesizeAgendaNode:
    """Test the SynthesizeAgendaNode class."""

    def test_initialization(self, mock_moderator_agent):
        """Test node initialization."""
        node = SynthesizeAgendaNode(mock_moderator_agent)
        assert node.moderator_agent == mock_moderator_agent
        assert node.get_node_name() == "SynthesizeAgenda"

    def test_validate_preconditions_success(
        self, synthesis_node, sample_state_with_votes
    ):
        """Test successful precondition validation."""
        assert synthesis_node.validate_preconditions(sample_state_with_votes) is True

    def test_validate_preconditions_no_votes(self, synthesis_node):
        """Test precondition validation fails without votes."""
        state = VirtualAgoraState(
            session_id="test", topic_queue=["Topic 1", "Topic 2"], messages=[]
        )
        assert synthesis_node.validate_preconditions(state) is False

    def test_validate_preconditions_empty_votes(self, synthesis_node):
        """Test precondition validation fails with empty votes."""
        state = VirtualAgoraState(
            session_id="test", topic_queue=["Topic 1", "Topic 2"], votes=[], messages=[]
        )
        assert synthesis_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_topics(self, synthesis_node):
        """Test precondition validation fails without topic queue."""
        state = VirtualAgoraState(
            session_id="test",
            votes=[{"id": "vote1", "vote_type": "topic_selection"}],
            messages=[],
        )
        assert synthesis_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_moderator(self, sample_state_with_votes):
        """Test precondition validation fails without moderator."""
        node = SynthesizeAgendaNode(None)
        assert node.validate_preconditions(sample_state_with_votes) is False

    def test_execute_success(self, synthesis_node, sample_state_with_votes):
        """Test successful agenda synthesis."""
        result = synthesis_node.execute(sample_state_with_votes)

        # Check state updates
        assert "agenda" in result
        assert "proposed_agenda" in result
        assert "current_phase" in result
        assert result["current_phase"] == 3

        # Check agenda object
        agenda = result["agenda"]
        assert isinstance(agenda, dict)
        assert agenda["topics"] == [
            "Ethics foundations",
            "Fairness in AI",
            "Privacy concerns",
        ]
        assert agenda["current_topic_index"] == 0
        assert agenda["completed_topics"] == []

        # Check proposed agenda
        proposed_agenda = result["proposed_agenda"]
        assert proposed_agenda == [
            "Ethics foundations",
            "Fairness in AI",
            "Privacy concerns",
        ]

        # Check moderator was called with correct format
        synthesis_node.moderator_agent.synthesize_agenda.assert_called_once()
        call_args = synthesis_node.moderator_agent.synthesize_agenda.call_args[0][0]

        # Should have 3 agent votes
        assert len(call_args) == 3

        # Check vote format
        for vote_data in call_args:
            assert "agent_id" in vote_data
            assert "vote" in vote_data
            assert vote_data["agent_id"].startswith("gpt-4o-")

    def test_execute_with_moderator_failure(
        self, mock_moderator_agent, sample_state_with_votes
    ):
        """Test synthesis with moderator failure - uses fallback."""
        # Make moderator fail
        mock_moderator_agent.synthesize_agenda.side_effect = Exception(
            "Moderator failed"
        )

        node = SynthesizeAgendaNode(mock_moderator_agent)
        result = node.execute(sample_state_with_votes)

        # Check that fallback was used (first 5 topics)
        assert "agenda" in result
        agenda = result["agenda"]
        expected_fallback = sample_state_with_votes["topic_queue"][:5]
        assert agenda["topics"] == expected_fallback
        assert result["proposed_agenda"] == expected_fallback

    def test_execute_with_invalid_moderator_response(
        self, mock_moderator_agent, sample_state_with_votes
    ):
        """Test synthesis with invalid moderator response format."""
        # Make moderator return invalid format
        mock_moderator_agent.synthesize_agenda.return_value = "Invalid response"

        node = SynthesizeAgendaNode(mock_moderator_agent)
        result = node.execute(sample_state_with_votes)

        # Check that fallback was used
        assert "agenda" in result
        agenda = result["agenda"]
        expected_fallback = sample_state_with_votes["topic_queue"][:5]
        assert agenda["topics"] == expected_fallback

    def test_execute_with_missing_proposed_agenda_key(
        self, mock_moderator_agent, sample_state_with_votes
    ):
        """Test synthesis when moderator response lacks proposed_agenda key."""
        # Make moderator return dict without proposed_agenda key
        mock_moderator_agent.synthesize_agenda.return_value = {"other_key": "value"}

        node = SynthesizeAgendaNode(mock_moderator_agent)
        result = node.execute(sample_state_with_votes)

        # Check that fallback was used
        assert "agenda" in result
        agenda = result["agenda"]
        expected_fallback = sample_state_with_votes["topic_queue"][:5]
        assert agenda["topics"] == expected_fallback

    def test_execute_handles_nested_vote_lists(
        self, synthesis_node, sample_state_with_votes
    ):
        """Test that synthesis handles nested vote lists from reducer."""
        # Modify state to have nested vote lists
        nested_votes = [
            [sample_state_with_votes["votes"][0]],  # Wrapped in list
            sample_state_with_votes["votes"][1],  # Normal vote
            [sample_state_with_votes["votes"][2]],  # Wrapped in list
        ]
        sample_state_with_votes["votes"] = nested_votes

        result = synthesis_node.execute(sample_state_with_votes)

        # Should still work correctly
        assert "agenda" in result
        agenda = result["agenda"]
        assert len(agenda["topics"]) == 3

        # Check moderator was called with flattened votes
        synthesis_node.moderator_agent.synthesize_agenda.assert_called_once()
        call_args = synthesis_node.moderator_agent.synthesize_agenda.call_args[0][0]
        assert len(call_args) == 3

    def test_execute_filters_non_topic_votes(
        self, synthesis_node, sample_state_with_votes
    ):
        """Test that synthesis only uses topic_selection votes."""
        # Add non-topic votes to the state
        additional_votes = [
            {
                "id": "vote_99999999",
                "voter_id": "gpt-4o-4",
                "phase": 1,
                "vote_type": "other_vote_type",
                "choice": "Some other choice",
                "timestamp": datetime.now(),
            }
        ]
        sample_state_with_votes["votes"].extend(additional_votes)

        result = synthesis_node.execute(sample_state_with_votes)

        # Should still work with only topic_selection votes
        synthesis_node.moderator_agent.synthesize_agenda.assert_called_once()
        call_args = synthesis_node.moderator_agent.synthesize_agenda.call_args[0][0]

        # Should only have 3 topic_selection votes, not the other vote
        assert len(call_args) == 3
        for vote_data in call_args:
            assert vote_data["agent_id"] in ["gpt-4o-1", "gpt-4o-2", "gpt-4o-3"]

    def test_execute_handles_unexpected_vote_types(
        self, synthesis_node, sample_state_with_votes
    ):
        """Test that synthesis handles unexpected vote object types."""
        # Add unexpected vote types
        unexpected_votes = [
            "string_vote",  # String instead of dict
            123,  # Number instead of dict
            None,  # None value
        ]
        sample_state_with_votes["votes"].extend(unexpected_votes)

        result = synthesis_node.execute(sample_state_with_votes)

        # Should still work, ignoring unexpected types
        assert "agenda" in result
        agenda = result["agenda"]
        assert len(agenda["topics"]) == 3

    def test_execute_with_no_topic_selection_votes(
        self, synthesis_node, sample_state_with_votes
    ):
        """Test synthesis when no votes are topic_selection type."""
        # Change all votes to different type
        for vote in sample_state_with_votes["votes"]:
            vote["vote_type"] = "other_type"

        result = synthesis_node.execute(sample_state_with_votes)

        # Since moderator gets called with empty list, it returns the configured response
        # The moderator was configured to return a valid agenda, so it should succeed
        assert "agenda" in result
        agenda = result["agenda"]
        # The moderator returns the configured agenda: ["Ethics foundations", "Fairness in AI", "Privacy concerns"]
        assert agenda["topics"] == [
            "Ethics foundations",
            "Fairness in AI",
            "Privacy concerns",
        ]

    def test_vote_data_format_to_moderator(
        self, synthesis_node, sample_state_with_votes
    ):
        """Test that vote data is correctly formatted for moderator."""
        result = synthesis_node.execute(sample_state_with_votes)

        # Check the format sent to moderator
        synthesis_node.moderator_agent.synthesize_agenda.assert_called_once()
        call_args = synthesis_node.moderator_agent.synthesize_agenda.call_args[0][0]

        expected_data = [
            {
                "agent_id": "gpt-4o-1",
                "vote": "I prefer Ethics foundations first, then Fairness in AI",
            },
            {
                "agent_id": "gpt-4o-2",
                "vote": "Fairness in AI should be priority, then Privacy concerns",
            },
            {
                "agent_id": "gpt-4o-3",
                "vote": "Ethics foundations, Privacy concerns, Fairness in AI",
            },
        ]

        assert call_args == expected_data

    def test_fallback_limits_to_five_topics(self, synthesis_node):
        """Test that fallback limits agenda to 5 topics maximum."""
        # Create state with more than 5 topics
        many_topics = [f"Topic {i+1}" for i in range(10)]
        state = VirtualAgoraState(
            session_id="test_session",
            topic_queue=many_topics,
            votes=[
                {
                    "id": "vote_12345678",
                    "voter_id": "gpt-4o-1",
                    "vote_type": "topic_selection",
                    "choice": "Some vote",
                    "timestamp": datetime.now(),
                }
            ],
            messages=[],
        )

        # Make moderator fail to trigger fallback
        synthesis_node.moderator_agent.synthesize_agenda.side_effect = Exception("Fail")

        result = synthesis_node.execute(state)

        # Check that only first 5 topics are used
        agenda = result["agenda"]
        assert len(agenda["topics"]) == 5
        assert agenda["topics"] == [
            "Topic 1",
            "Topic 2",
            "Topic 3",
            "Topic 4",
            "Topic 5",
        ]
