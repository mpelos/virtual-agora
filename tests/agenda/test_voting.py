"""Tests for agenda voting orchestration."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from virtual_agora.agenda.voting import VotingOrchestrator, VoteParser
from virtual_agora.agenda.models import Vote, VoteType, VoteCollection, VoteStatus
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent


class TestVoteParser:
    """Test VoteParser functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = VoteParser()
        self.topics = ["Climate Change", "Economic Policy", "Technology Ethics"]

    def test_parse_numbered_vote(self):
        """Test parsing numbered vote."""
        vote_content = """
        1. Climate Change
        2. Technology Ethics
        3. Economic Policy
        """

        preferences = self.parser.parse_vote(vote_content, self.topics)

        assert preferences == ["Climate Change", "Technology Ethics", "Economic Policy"]

    def test_parse_explicit_preferences(self):
        """Test parsing explicit preference statements."""
        vote_content = """
        First: Technology Ethics
        Second: Climate Change
        Third: Economic Policy
        """

        preferences = self.parser.parse_vote(vote_content, self.topics)

        assert "Technology Ethics" in preferences
        assert "Climate Change" in preferences
        assert "Economic Policy" in preferences

    def test_parse_order_words(self):
        """Test parsing with order words."""
        vote_content = "I prefer first Technology Ethics, then Climate Change"

        preferences = self.parser.parse_vote(vote_content, self.topics)

        assert len(preferences) >= 2
        assert "Technology Ethics" in preferences
        assert "Climate Change" in preferences

    def test_parse_contextual_vote(self):
        """Test parsing vote with contextual clues."""
        vote_content = """
        I think Climate Change is most important,
        but we should also discuss Technology Ethics.
        Economic Policy is less urgent.
        """

        preferences = self.parser.parse_vote(vote_content, self.topics)

        assert "Climate Change" in preferences
        assert len(preferences) >= 1

    def test_parse_empty_vote(self):
        """Test parsing empty or invalid vote."""
        preferences = self.parser.parse_vote("", self.topics)
        assert preferences == []

        preferences = self.parser.parse_vote(
            "This doesn't mention any topics", self.topics
        )
        assert preferences == []

    def test_find_best_topic_match(self):
        """Test topic matching logic."""
        # Exact match
        match = self.parser._find_best_topic_match("Climate Change", self.topics)
        assert match == "Climate Change"

        # Partial match
        match = self.parser._find_best_topic_match("climate", self.topics)
        assert match == "Climate Change"

        # Word match
        match = self.parser._find_best_topic_match("ethics technology", self.topics)
        assert match == "Technology Ethics"

        # No match
        match = self.parser._find_best_topic_match("unrelated topic", self.topics)
        assert match is None


class TestVotingOrchestrator:
    """Test VotingOrchestrator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.moderator = Mock(spec=ModeratorAgent)
        self.orchestrator = VotingOrchestrator(self.moderator, timeout_seconds=30)

        # Create mock agents
        self.agents = {}
        for i in range(3):
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = f"agent_{i+1}"
            self.agents[f"agent_{i+1}"] = agent

    def test_create_vote_prompt_initial_agenda(self):
        """Test creating vote prompt for initial agenda."""
        topics = ["Topic A", "Topic B", "Topic C"]
        prompt = self.orchestrator._create_vote_prompt(topics, VoteType.INITIAL_AGENDA)

        assert "rank the following topics" in prompt.lower()
        assert "Topic A" in prompt
        assert "Topic B" in prompt
        assert "Topic C" in prompt

    def test_create_vote_prompt_modification(self):
        """Test creating vote prompt for modification."""
        topics = ["Topic A", "Topic B"]
        prompt = self.orchestrator._create_vote_prompt(
            topics, VoteType.AGENDA_MODIFICATION
        )

        assert "modified agenda" in prompt.lower()
        assert "Topic A" in prompt
        assert "Topic B" in prompt

    def test_collect_single_agent_vote_success(self):
        """Test collecting vote from single agent successfully."""
        agent = Mock(spec=DiscussionAgent)
        agent.vote_on_agenda.return_value = "1. Topic A\n2. Topic B"

        vote = self.orchestrator._collect_single_agent_vote(
            "agent_1", agent, ["Topic A", "Topic B"], VoteType.INITIAL_AGENDA
        )

        assert vote is not None
        assert vote.agent_id == "agent_1"
        assert vote.vote_type == VoteType.INITIAL_AGENDA
        assert vote.vote_content == "1. Topic A\n2. Topic B"
        assert vote.status == VoteStatus.SUBMITTED

    def test_collect_single_agent_vote_fallback(self):
        """Test collecting vote with fallback method."""
        agent = Mock(spec=DiscussionAgent)
        # Agent doesn't have vote_on_agenda method
        del agent.vote_on_agenda
        agent.generate_response.return_value = "I prefer Topic A first"

        vote = self.orchestrator._collect_single_agent_vote(
            "agent_1", agent, ["Topic A", "Topic B"], VoteType.INITIAL_AGENDA
        )

        assert vote is not None
        assert vote.vote_content == "I prefer Topic A first"

    def test_collect_single_agent_vote_failure(self):
        """Test handling agent vote failure."""
        agent = Mock(spec=DiscussionAgent)
        agent.vote_on_agenda.side_effect = Exception("Agent error")

        vote = self.orchestrator._collect_single_agent_vote(
            "agent_1", agent, ["Topic A", "Topic B"], VoteType.INITIAL_AGENDA
        )

        assert vote is None

    def test_validate_and_parse_votes(self):
        """Test vote validation and parsing."""
        votes = [
            Vote(
                agent_id="agent_1",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Topic A\n2. Topic B",
            ),
            Vote(
                agent_id="agent_2",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="I prefer Topic B",
            ),
            Vote(
                agent_id="agent_3",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="Invalid vote content that doesn't match any topics",  # Invalid vote
            ),
        ]

        valid_votes = self.orchestrator._validate_and_parse_votes(
            votes, ["Topic A", "Topic B"]
        )

        # Should have 2 valid votes (empty vote filtered out)
        assert len(valid_votes) == 2

        # Check that preferences were parsed
        for vote in valid_votes:
            assert vote.parsed_preferences is not None
            assert len(vote.parsed_preferences) > 0
            assert vote.confidence_score is not None

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        # High confidence vote (numbered, complete)
        score = self.orchestrator._calculate_confidence_score(
            "1. Topic A\n2. Topic B\n3. Topic C",
            ["Topic A", "Topic B", "Topic C"],
            ["Topic A", "Topic B", "Topic C"],
        )
        assert score > 0.8

        # Medium confidence vote (some structure)
        score = self.orchestrator._calculate_confidence_score(
            "I prefer Topic A and Topic B",
            ["Topic A", "Topic B"],
            ["Topic A", "Topic B", "Topic C"],
        )
        assert 0.3 < score < 0.8

        # Low confidence vote (minimal structure)
        score = self.orchestrator._calculate_confidence_score(
            "Topic A", ["Topic A"], ["Topic A", "Topic B", "Topic C"]
        )
        assert score < 0.7

    @pytest.mark.asyncio
    async def test_collect_votes_success(self):
        """Test successful vote collection from all agents."""
        # Mock agent responses
        for i, agent in enumerate(self.agents.values()):
            agent.vote_on_agenda.return_value = (
                f"1. Topic {chr(65+i)}\n2. Topic {chr(66+i)}"
            )

        # Mock the collect_single_agent_vote to avoid threading issues in tests
        def mock_collect_vote(agent_id, agent, topics, vote_type):
            return Vote(
                agent_id=agent_id,
                vote_type=vote_type,
                vote_content=agent.vote_on_agenda.return_value,
                status=VoteStatus.SUBMITTED,
            )

        with patch.object(
            self.orchestrator,
            "_collect_single_agent_vote",
            side_effect=mock_collect_vote,
        ):
            collection = await self.orchestrator.collect_votes(
                self.agents,
                ["Topic A", "Topic B", "Topic C"],
                VoteType.INITIAL_AGENDA,
                "session_1",
            )

        assert collection.session_id == "session_1"
        assert collection.vote_type == VoteType.INITIAL_AGENDA
        assert len(collection.votes) == 3
        assert len(collection.responding_agents) == 3
        assert collection.participation_rate == 1.0
        assert collection.status == VoteStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_collect_votes_with_timeouts(self):
        """Test vote collection with some agents timing out."""
        # Mock one agent to succeed, others to fail/timeout
        self.agents["agent_1"].vote_on_agenda.return_value = "1. Topic A\n2. Topic B"

        def mock_collect_vote(agent_id, agent, topics, vote_type):
            if agent_id == "agent_1":
                return Vote(
                    agent_id=agent_id,
                    vote_type=vote_type,
                    vote_content="1. Topic A\n2. Topic B",
                    status=VoteStatus.SUBMITTED,
                )
            return None  # Simulate timeout/failure

        with patch.object(
            self.orchestrator,
            "_collect_single_agent_vote",
            side_effect=mock_collect_vote,
        ):
            collection = await self.orchestrator.collect_votes(
                self.agents, ["Topic A", "Topic B"], VoteType.INITIAL_AGENDA
            )

        assert len(collection.votes) == 1
        assert len(collection.responding_agents) == 1
        assert collection.participation_rate == 1 / 3

    def test_analyze_vote_distribution(self):
        """Test vote distribution analysis."""
        votes = [
            Vote(
                agent_id="agent_1",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Topic A\n2. Topic B",
                parsed_preferences=["Topic A", "Topic B"],
                confidence_score=0.9,
                status=VoteStatus.SUBMITTED,
            ),
            Vote(
                agent_id="agent_2",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Topic B\n2. Topic A",
                parsed_preferences=["Topic B", "Topic A"],
                confidence_score=0.8,
                status=VoteStatus.SUBMITTED,
            ),
            Vote(
                agent_id="agent_3",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="Invalid vote",
                status=VoteStatus.INVALID,
            ),
        ]

        analysis = self.orchestrator.analyze_vote_distribution(votes)

        assert analysis["total_votes"] == 3
        assert analysis["valid_votes"] == 2
        assert analysis["invalid_votes"] == 1
        assert analysis["average_confidence"] == pytest.approx(0.85)

        # Check topic mentions
        assert "Topic A" in analysis["topic_mentions"]
        assert "Topic B" in analysis["topic_mentions"]
        assert analysis["topic_mentions"]["Topic A"] == 2
        assert analysis["topic_mentions"]["Topic B"] == 2

        # Check preference patterns
        assert 1 in analysis["preference_patterns"]  # First choices
        assert 2 in analysis["preference_patterns"]  # Second choices

        # Check consensus indicators
        consensus = analysis["consensus_indicators"]
        assert consensus["agreement_level"] == 1.0  # Both topics got equal support
        assert consensus["topic_diversity"] == 2
