"""Tests for agenda management models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from virtual_agora.agenda.models import (
    Proposal,
    Vote,
    VoteType,
    AgendaItem,
    AgendaState,
    ProposalCollection,
    VoteCollection,
    AgendaModification,
    TopicTransition,
    AgendaAnalytics,
    AgendaSynthesisResult,
    EdgeCaseEvent,
    ProposalStatus,
    VoteStatus,
    AgendaStatus,
)


class TestProposal:
    """Test Proposal model."""

    def test_proposal_creation(self):
        """Test creating a valid proposal."""
        proposal = Proposal(
            agent_id="agent_1", topic="Test Topic", description="Test description"
        )

        assert proposal.agent_id == "agent_1"
        assert proposal.topic == "Test Topic"
        assert proposal.description == "Test description"
        assert proposal.status == ProposalStatus.PENDING
        assert isinstance(proposal.timestamp, datetime)
        assert proposal.id is not None

    def test_proposal_empty_topic_validation(self):
        """Test that empty topics are rejected."""
        with pytest.raises(ValidationError):
            Proposal(agent_id="agent_1", topic="")

        with pytest.raises(ValidationError):
            Proposal(agent_id="agent_1", topic="   ")

    def test_proposal_topic_trimming(self):
        """Test that topics are trimmed."""
        proposal = Proposal(agent_id="agent_1", topic="  Test Topic  ")
        assert proposal.topic == "Test Topic"


class TestVote:
    """Test Vote model."""

    def test_vote_creation(self):
        """Test creating a valid vote."""
        vote = Vote(
            agent_id="agent_1",
            vote_type=VoteType.INITIAL_AGENDA,
            vote_content="I prefer topic A first, then topic B",
        )

        assert vote.agent_id == "agent_1"
        assert vote.vote_type == VoteType.INITIAL_AGENDA
        assert vote.vote_content == "I prefer topic A first, then topic B"
        assert vote.status == VoteStatus.PENDING
        assert isinstance(vote.timestamp, datetime)

    def test_vote_empty_content_validation(self):
        """Test that empty vote content is rejected."""
        with pytest.raises(ValidationError):
            Vote(agent_id="agent_1", vote_type=VoteType.INITIAL_AGENDA, vote_content="")

    def test_vote_content_trimming(self):
        """Test that vote content is trimmed."""
        vote = Vote(
            agent_id="agent_1",
            vote_type=VoteType.INITIAL_AGENDA,
            vote_content="  Test vote content  ",
        )
        assert vote.vote_content == "Test vote content"


class TestAgendaItem:
    """Test AgendaItem model."""

    def test_agenda_item_creation(self):
        """Test creating a valid agenda item."""
        item = AgendaItem(
            topic="Discussion Topic", rank=1, proposed_by=["agent_1", "agent_2"]
        )

        assert item.topic == "Discussion Topic"
        assert item.rank == 1
        assert item.proposed_by == ["agent_1", "agent_2"]
        assert item.status == "pending"

    def test_agenda_item_rank_validation(self):
        """Test that rank must be positive."""
        with pytest.raises(ValidationError):
            AgendaItem(topic="Test", rank=0)

        with pytest.raises(ValidationError):
            AgendaItem(topic="Test", rank=-1)

    def test_agenda_item_empty_topic_validation(self):
        """Test that empty topics are rejected."""
        with pytest.raises(ValidationError):
            AgendaItem(topic="", rank=1)


class TestProposalCollection:
    """Test ProposalCollection model."""

    def test_proposal_collection_creation(self):
        """Test creating a proposal collection."""
        collection = ProposalCollection(
            session_id="session_1",
            requested_agents=["agent_1", "agent_2"],
            timeout_seconds=300,
        )

        assert collection.session_id == "session_1"
        assert collection.requested_agents == ["agent_1", "agent_2"]
        assert collection.timeout_seconds == 300
        assert collection.status == ProposalStatus.PENDING
        assert collection.completion_rate == 0.0

    def test_completion_rate_calculation(self):
        """Test completion rate calculation."""
        collection = ProposalCollection(
            session_id="session_1",
            requested_agents=["agent_1", "agent_2", "agent_3"],
            responding_agents=["agent_1", "agent_2"],
        )

        assert collection.completion_rate == 2 / 3


class TestVoteCollection:
    """Test VoteCollection model."""

    def test_vote_collection_creation(self):
        """Test creating a vote collection."""
        collection = VoteCollection(
            session_id="session_1",
            vote_type=VoteType.INITIAL_AGENDA,
            topic_options=["topic_1", "topic_2"],
        )

        assert collection.session_id == "session_1"
        assert collection.vote_type == VoteType.INITIAL_AGENDA
        assert collection.topic_options == ["topic_1", "topic_2"]
        assert collection.participation_rate == 0.0

    def test_participation_rate_calculation(self):
        """Test participation rate calculation."""
        collection = VoteCollection(
            session_id="session_1",
            vote_type=VoteType.INITIAL_AGENDA,
            requested_agents=["agent_1", "agent_2", "agent_3", "agent_4"],
            responding_agents=["agent_1", "agent_3"],
        )

        assert collection.participation_rate == 0.5


class TestAgendaState:
    """Test AgendaState model."""

    def test_agenda_state_creation(self):
        """Test creating an agenda state."""
        state = AgendaState(session_id="session_1")

        assert state.session_id == "session_1"
        assert state.version == 1
        assert state.status == AgendaStatus.PENDING
        assert state.current_agenda == []
        assert state.current_topic_index == 0
        assert isinstance(state.created_at, datetime)

    def test_remaining_topics_property(self):
        """Test remaining topics property."""
        item1 = AgendaItem(topic="Topic 1", rank=1)
        item2 = AgendaItem(topic="Topic 2", rank=2)
        item3 = AgendaItem(topic="Topic 3", rank=3)

        state = AgendaState(
            session_id="session_1",
            current_agenda=[item1, item2, item3],
            completed_topics=["Topic 1"],
        )

        remaining = state.remaining_topics
        assert "Topic 1" not in remaining
        assert "Topic 2" in remaining
        assert "Topic 3" in remaining
        assert len(remaining) == 2

    def test_current_topic_property(self):
        """Test current topic property."""
        item1 = AgendaItem(topic="Active Topic", rank=1)
        item2 = AgendaItem(topic="Other Topic", rank=2)

        state = AgendaState(
            session_id="session_1",
            current_agenda=[item1, item2],
            active_topic="Active Topic",
        )

        current = state.current_topic
        assert current is not None
        assert current.topic == "Active Topic"
        assert current.rank == 1


class TestAgendaModification:
    """Test AgendaModification model."""

    def test_modification_creation(self):
        """Test creating an agenda modification."""
        mod = AgendaModification(
            agent_id="agent_1", modification_type="add", new_topic="New Topic"
        )

        assert mod.agent_id == "agent_1"
        assert mod.modification_type == "add"
        assert mod.new_topic == "New Topic"
        assert mod.applied is False

    def test_modification_type_validation(self):
        """Test modification type validation."""
        with pytest.raises(ValidationError):
            AgendaModification(agent_id="agent_1", modification_type="invalid")


class TestTopicTransition:
    """Test TopicTransition model."""

    def test_transition_creation(self):
        """Test creating a topic transition."""
        transition = TopicTransition(
            session_id="session_1",
            from_topic="Topic A",
            to_topic="Topic B",
            transition_type="start",
        )

        assert transition.session_id == "session_1"
        assert transition.from_topic == "Topic A"
        assert transition.to_topic == "Topic B"
        assert transition.transition_type == "start"
        assert isinstance(transition.timestamp, datetime)

    def test_transition_type_validation(self):
        """Test transition type validation."""
        with pytest.raises(ValidationError):
            TopicTransition(session_id="session_1", transition_type="invalid")


class TestAgendaSynthesisResult:
    """Test AgendaSynthesisResult model."""

    def test_synthesis_result_creation(self):
        """Test creating a synthesis result."""
        result = AgendaSynthesisResult(
            proposed_agenda=["Topic 1", "Topic 2", "Topic 3"]
        )

        assert result.proposed_agenda == ["Topic 1", "Topic 2", "Topic 3"]
        assert result.synthesis_attempts == 1
        assert isinstance(result.timestamp, datetime)

    def test_empty_agenda_validation(self):
        """Test that empty agendas are rejected."""
        with pytest.raises(ValidationError):
            AgendaSynthesisResult(proposed_agenda=[])


class TestEdgeCaseEvent:
    """Test EdgeCaseEvent model."""

    def test_edge_case_creation(self):
        """Test creating an edge case event."""
        edge_case = EdgeCaseEvent(
            session_id="session_1",
            event_type="empty_proposals",
            description="No proposals were collected",
            resolution_strategy="fallback_agenda",
            system_response="Created fallback agenda",
        )

        assert edge_case.session_id == "session_1"
        assert edge_case.event_type == "empty_proposals"
        assert edge_case.description == "No proposals were collected"
        assert edge_case.recovered_successfully is False
        assert isinstance(edge_case.timestamp, datetime)


class TestAgendaAnalytics:
    """Test AgendaAnalytics model."""

    def test_analytics_creation(self):
        """Test creating agenda analytics."""
        analytics = AgendaAnalytics(session_id="session_1")

        assert analytics.session_id == "session_1"
        assert analytics.total_proposals == 0
        assert analytics.unique_topics_proposed == 0
        assert analytics.proposal_acceptance_rate == 0.0
        assert isinstance(analytics.generated_at, datetime)
