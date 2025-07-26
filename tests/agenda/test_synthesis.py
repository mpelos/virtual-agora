"""Tests for agenda synthesis and ranking."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json
from datetime import datetime

from virtual_agora.agenda.synthesis import AgendaSynthesizer, TopicScore
from virtual_agora.agenda.models import (
    Proposal,
    Vote,
    VoteType,
    AgendaItem,
    AgendaSynthesisResult,
    AgendaModification,
    ProposalStatus,
    VoteStatus,
)
from virtual_agora.agents.moderator import ModeratorAgent


class TestTopicScore:
    """Test TopicScore dataclass."""

    def test_topic_score_creation(self):
        """Test creating a TopicScore."""
        score = TopicScore(
            topic="Test Topic",
            raw_score=5.0,
            proposal_count=2,
            vote_positions=[1, 2, 1],
            first_choice_count=2,
            total_mentions=3,
            proposing_agents=["agent_1", "agent_2"],
        )

        assert score.topic == "Test Topic"
        assert score.raw_score == 5.0
        assert score.proposal_count == 2
        assert score.first_choice_count == 2
        assert score.total_mentions == 3
        assert len(score.proposing_agents) == 2


class TestAgendaSynthesizer:
    """Test AgendaSynthesizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.moderator = Mock(spec=ModeratorAgent)
        self.synthesizer = AgendaSynthesizer(self.moderator, max_retries=3)

    def create_sample_proposals(self):
        """Create sample proposals for testing."""
        return [
            Proposal(
                agent_id="agent_1",
                topic="Climate Change",
                description="Discussion about climate change",
                status=ProposalStatus.COLLECTED,
            ),
            Proposal(
                agent_id="agent_2",
                topic="Technology Ethics",
                description="Discussion about technology ethics",
                status=ProposalStatus.COLLECTED,
            ),
            Proposal(
                agent_id="agent_3",
                topic="Climate Change",  # Duplicate
                description="Another climate proposal",
                status=ProposalStatus.COLLECTED,
            ),
            Proposal(
                agent_id="agent_1",
                topic="Economic Policy",
                description="Discussion about economic policy",
                status=ProposalStatus.COLLECTED,
            ),
        ]

    def create_sample_votes(self):
        """Create sample votes for testing."""
        return [
            Vote(
                agent_id="agent_1",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Climate Change\n2. Technology Ethics",
                parsed_preferences=["Climate Change", "Technology Ethics"],
                confidence_score=0.9,
                status=VoteStatus.SUBMITTED,
            ),
            Vote(
                agent_id="agent_2",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Technology Ethics\n2. Economic Policy",
                parsed_preferences=["Technology Ethics", "Economic Policy"],
                confidence_score=0.8,
                status=VoteStatus.SUBMITTED,
            ),
            Vote(
                agent_id="agent_3",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Climate Change\n2. Economic Policy",
                parsed_preferences=["Climate Change", "Economic Policy"],
                confidence_score=0.7,
                status=VoteStatus.SUBMITTED,
            ),
        ]

    def test_calculate_topic_scores(self):
        """Test topic score calculation."""
        proposals = self.create_sample_proposals()
        votes = self.create_sample_votes()

        topic_scores = self.synthesizer._calculate_topic_scores(proposals, votes)

        # Check that all topics are present
        assert "climate change" in topic_scores
        assert "technology ethics" in topic_scores
        assert "economic policy" in topic_scores

        # Climate Change should have high score (2 proposals + 2 first choices)
        climate_score = topic_scores["climate change"]
        assert climate_score.proposal_count == 2
        assert climate_score.first_choice_count == 2
        assert climate_score.total_mentions == 2
        assert len(climate_score.proposing_agents) == 2

        # Technology Ethics should have medium score (1 proposal + 1 first choice)
        tech_score = topic_scores["technology ethics"]
        assert tech_score.proposal_count == 1
        assert tech_score.first_choice_count == 1
        assert tech_score.total_mentions == 2

        # Economic Policy should have lower score (1 proposal + 0 first choices)
        econ_score = topic_scores["economic policy"]
        assert econ_score.proposal_count == 1
        assert econ_score.first_choice_count == 0
        assert econ_score.total_mentions == 2

    def test_prepare_synthesis_context(self):
        """Test synthesis context preparation."""
        proposals = self.create_sample_proposals()
        votes = self.create_sample_votes()
        topic_scores = self.synthesizer._calculate_topic_scores(proposals, votes)

        context = self.synthesizer._prepare_synthesis_context(
            proposals, votes, topic_scores
        )

        assert "proposals_summary" in context
        assert "votes_summary" in context
        assert "scoring_data" in context

        # Check proposals summary
        proposals_summary = context["proposals_summary"]
        assert len(proposals_summary) == 4
        assert proposals_summary[0]["topic"] == "Climate Change"
        assert proposals_summary[0]["agent"] == "agent_1"

        # Check votes summary
        votes_summary = context["votes_summary"]
        assert len(votes_summary) == 3
        assert votes_summary[0]["agent"] == "agent_1"
        assert votes_summary[0]["preferences"] == [
            "Climate Change",
            "Technology Ethics",
        ]

        # Check scoring data
        scoring_data = context["scoring_data"]
        assert len(scoring_data) == 3
        # Should be sorted by score (highest first)
        assert scoring_data[0]["topic"] in ["Climate Change", "Technology Ethics"]

    def test_parse_agenda_response_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"proposed_agenda": ["Topic A", "Topic B", "Topic C"]}'

        agenda = self.synthesizer._parse_agenda_response(response)

        assert agenda == ["Topic A", "Topic B", "Topic C"]

    def test_parse_agenda_response_embedded_json(self):
        """Test parsing JSON embedded in text response."""
        response = """
        Based on the analysis, here is the proposed agenda:
        {"proposed_agenda": ["Topic A", "Topic B", "Topic C"]}
        This agenda reflects the voting preferences.
        """

        agenda = self.synthesizer._parse_agenda_response(response)

        assert agenda == ["Topic A", "Topic B", "Topic C"]

    def test_parse_agenda_response_invalid(self):
        """Test parsing invalid response."""
        response = "This is not JSON"

        agenda = self.synthesizer._parse_agenda_response(response)

        assert agenda is None

    def test_apply_tie_breaking(self):
        """Test tie-breaking logic."""
        # Create topics with similar scores
        topic_scores = {
            "topic a": TopicScore(
                topic="Topic A",
                raw_score=5.0,
                proposal_count=1,
                vote_positions=[1, 2],
                first_choice_count=1,
                total_mentions=2,
                proposing_agents=["agent_1"],
            ),
            "topic b": TopicScore(
                topic="Topic B",
                raw_score=5.0,  # Same score as Topic A
                proposal_count=1,
                vote_positions=[2, 1],
                first_choice_count=2,  # More first choices - should win
                total_mentions=2,
                proposing_agents=["agent_2"],
            ),
        }

        agenda = ["Topic A", "Topic B"]

        final_agenda, tie_breaks = self.synthesizer._apply_tie_breaking(
            agenda, topic_scores
        )

        # Topic B should come first due to more first choices
        assert final_agenda[0] == "Topic B"
        assert final_agenda[1] == "Topic A"
        assert len(tie_breaks) > 0

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        agenda = ["Topic A", "Topic B", "Topic C"]
        topic_scores = {
            "topic a": TopicScore(
                topic="Topic A",
                raw_score=10.0,
                proposal_count=2,
                vote_positions=[1, 1, 1],
                first_choice_count=3,
                total_mentions=3,
                proposing_agents=["agent_1", "agent_2"],
            ),
            "topic b": TopicScore(
                topic="Topic B",
                raw_score=7.0,
                proposal_count=1,
                vote_positions=[2, 2],
                first_choice_count=0,
                total_mentions=2,
                proposing_agents=["agent_3"],
            ),
            "topic c": TopicScore(
                topic="Topic C",
                raw_score=3.0,
                proposal_count=1,
                vote_positions=[3],
                first_choice_count=0,
                total_mentions=1,
                proposing_agents=["agent_1"],
            ),
        }

        confidence = self.synthesizer._calculate_confidence_score(agenda, topic_scores)

        assert 0.0 <= confidence <= 1.0
        # Should be higher confidence due to strong Topic A
        assert confidence > 0.5

    def test_generate_explanation(self):
        """Test explanation generation."""
        agenda = ["Topic A", "Topic B"]
        topic_scores = {
            "topic a": TopicScore(
                topic="Topic A",
                raw_score=10.0,
                proposal_count=2,
                vote_positions=[1, 1],
                first_choice_count=2,
                total_mentions=2,
                proposing_agents=["agent_1", "agent_2"],
            ),
            "topic b": TopicScore(
                topic="Topic B",
                raw_score=7.0,
                proposal_count=1,
                vote_positions=[2, 2],
                first_choice_count=0,
                total_mentions=2,
                proposing_agents=["agent_3"],
            ),
        }
        tie_breaks = ["Tie resolved by first choice count"]

        explanation = self.synthesizer._generate_explanation(
            agenda, topic_scores, tie_breaks
        )

        assert "Topic A" in explanation
        assert "Topic B" in explanation
        assert "Score:" in explanation
        assert "First choices:" in explanation
        assert "Tie-breaking applied:" in explanation
        assert tie_breaks[0] in explanation

    def test_synthesize_agenda_success(self):
        """Test successful agenda synthesis."""
        proposals = self.create_sample_proposals()
        votes = self.create_sample_votes()

        # Mock moderator response - synthesize_agenda returns a list directly
        self.moderator.synthesize_agenda.return_value = [
            "Climate Change",
            "Technology Ethics",
            "Economic Policy",
        ]

        result = self.synthesizer.synthesize_agenda(proposals, votes)

        assert isinstance(result, AgendaSynthesisResult)
        assert result.proposed_agenda == [
            "Climate Change",
            "Technology Ethics",
            "Economic Policy",
        ]
        assert result.synthesis_attempts == 1
        assert result.confidence_score > 0.0
        assert result.synthesis_explanation is not None

    def test_synthesize_agenda_with_retries(self):
        """Test agenda synthesis with retries after failures."""
        proposals = self.create_sample_proposals()
        votes = self.create_sample_votes()

        # Mock moderator to fail twice, then succeed
        self.moderator.synthesize_agenda.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            [
                "Climate Change",
                "Technology Ethics",
                "Economic Policy",
            ],  # Use actual topics from proposals
        ]

        result = self.synthesizer.synthesize_agenda(proposals, votes)

        assert isinstance(result, AgendaSynthesisResult)
        assert len(result.proposed_agenda) > 0  # Should have valid agenda
        assert result.synthesis_attempts == 3

    def test_synthesize_agenda_fallback(self):
        """Test fallback synthesis when all attempts fail."""
        proposals = self.create_sample_proposals()
        votes = self.create_sample_votes()

        # Mock moderator to always fail
        self.moderator.synthesize_agenda.side_effect = Exception("Persistent failure")

        result = self.synthesizer.synthesize_agenda(proposals, votes)

        assert isinstance(result, AgendaSynthesisResult)
        assert len(result.proposed_agenda) > 0
        assert result.synthesis_attempts == self.synthesizer.max_retries
        assert result.confidence_score == 0.7  # Fallback confidence
        assert "Fallback synthesis" in result.synthesis_explanation

    def test_synthesize_modified_agenda(self):
        """Test synthesis of modified agenda."""
        # Create current agenda items
        current_agenda = [
            AgendaItem(topic="Existing Topic 1", rank=1, status="pending"),
            AgendaItem(topic="Existing Topic 2", rank=2, status="completed"),
            AgendaItem(topic="Existing Topic 3", rank=3, status="pending"),
        ]

        # Create modifications
        modifications = [
            AgendaModification(
                agent_id="agent_1",
                modification_type="add",
                new_topic="New Topic",
                new_description="A new topic to discuss",
            ),
            AgendaModification(
                agent_id="agent_2",
                modification_type="remove",
                target_topic="Existing Topic 3",
            ),
        ]

        # Create votes on modified topics
        votes = [
            Vote(
                agent_id="agent_1",
                vote_type=VoteType.AGENDA_MODIFICATION,
                vote_content="1. New Topic\n2. Existing Topic 1",
                parsed_preferences=["New Topic", "Existing Topic 1"],
                status=VoteStatus.SUBMITTED,
            )
        ]

        # Mock moderator response - synthesize_agenda returns a list directly
        self.moderator.synthesize_agenda.return_value = [
            "New Topic",
            "Existing Topic 1",
        ]

        result = self.synthesizer.synthesize_modified_agenda(
            current_agenda, modifications, votes
        )

        assert isinstance(result, AgendaSynthesisResult)
        assert "New Topic" in result.proposed_agenda
        assert "Existing Topic 1" in result.proposed_agenda
        # Completed topic should not be included
        assert "Existing Topic 2" not in result.proposed_agenda

    def test_normalize_topic(self):
        """Test topic normalization."""
        assert self.synthesizer._normalize_topic("Climate Change") == "climate change"
        assert (
            self.synthesizer._normalize_topic("  Technology Ethics  ")
            == "technology ethics"
        )
        assert (
            self.synthesizer._normalize_topic("Economic  Policy") == "economic policy"
        )

    def test_get_synthesis_metrics(self):
        """Test synthesis metrics generation."""
        result = AgendaSynthesisResult(
            proposed_agenda=["Topic A", "Topic B", "Topic C"],
            synthesis_explanation="Test explanation",
            tie_breaks_applied=["Tie break 1", "Tie break 2"],
            confidence_score=0.85,
            synthesis_attempts=2,
        )

        metrics = self.synthesizer.get_synthesis_metrics(result)

        assert metrics["agenda_length"] == 3
        assert metrics["synthesis_attempts"] == 2
        assert metrics["confidence_score"] == 0.85
        assert metrics["tie_breaks_count"] == 2
        assert metrics["has_explanation"] is True
        assert "synthesis_timestamp" in metrics
