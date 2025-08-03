"""Tests for CollateProposalsNode."""

import pytest
from unittest.mock import Mock

from virtual_agora.flow.nodes.agenda.collation import CollateProposalsNode
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.moderator import ModeratorAgent


@pytest.fixture
def mock_moderator_agent():
    """Create mock moderator agent."""
    agent = Mock(spec=ModeratorAgent)
    agent.collect_proposals.return_value = [
        "Unified Topic 1",
        "Unified Topic 2",
        "Unified Topic 3",
    ]
    return agent


@pytest.fixture
def sample_state_with_proposals():
    """Create sample VirtualAgoraState with proposals."""
    return VirtualAgoraState(
        session_id="test_session",
        main_topic="AI Ethics",
        current_phase=1,
        proposed_topics=[
            {
                "agent_id": "gpt-4o-1",
                "proposals": "1. Ethics foundations\n2. Privacy concerns",
            },
            {
                "agent_id": "gpt-4o-2",
                "proposals": "- Fairness in AI\n- Transparency issues",
            },
            {
                "agent_id": "gpt-4o-3",
                "proposals": "1) Accountability\n2) Bias mitigation",
            },
        ],
        messages=[],
    )


@pytest.fixture
def collation_node(mock_moderator_agent):
    """Create CollateProposalsNode instance."""
    return CollateProposalsNode(mock_moderator_agent)


class TestCollateProposalsNode:
    """Test the CollateProposalsNode class."""

    def test_initialization(self, mock_moderator_agent):
        """Test node initialization."""
        node = CollateProposalsNode(mock_moderator_agent)
        assert node.moderator_agent == mock_moderator_agent
        assert node.get_node_name() == "CollateProposals"

    def test_validate_preconditions_success(
        self, collation_node, sample_state_with_proposals
    ):
        """Test successful precondition validation."""
        assert (
            collation_node.validate_preconditions(sample_state_with_proposals) is True
        )

    def test_validate_preconditions_no_proposals(self, collation_node):
        """Test precondition validation fails without proposals."""
        state = VirtualAgoraState(session_id="test", messages=[])
        assert collation_node.validate_preconditions(state) is False

    def test_validate_preconditions_no_moderator(self, sample_state_with_proposals):
        """Test precondition validation fails without moderator."""
        node = CollateProposalsNode(None)
        assert node.validate_preconditions(sample_state_with_proposals) is False

    def test_execute_success(self, collation_node, sample_state_with_proposals):
        """Test successful proposal collation."""
        result = collation_node.execute(sample_state_with_proposals)

        # Check state updates
        assert "topic_queue" in result
        assert "current_phase" in result
        assert result["current_phase"] == 1

        # Check unified topics
        topic_queue = result["topic_queue"]
        assert topic_queue == ["Unified Topic 1", "Unified Topic 2", "Unified Topic 3"]

        # Check moderator was called with proposals
        collation_node.moderator_agent.collect_proposals.assert_called_once_with(
            sample_state_with_proposals["proposed_topics"]
        )

    def test_execute_with_moderator_failure(
        self, mock_moderator_agent, sample_state_with_proposals
    ):
        """Test collation with moderator failure - uses fallback."""
        # Make moderator fail
        mock_moderator_agent.collect_proposals.side_effect = Exception(
            "Moderator failed"
        )

        node = CollateProposalsNode(mock_moderator_agent)
        result = node.execute(sample_state_with_proposals)

        # Check that fallback was used
        assert "topic_queue" in result
        topic_queue = result["topic_queue"]

        # Should extract topics from numbered/bulleted lists
        expected_topics = [
            "Ethics foundations",
            "Privacy concerns",
            "Fairness in AI",
            "Transparency issues",
            "Accountability",
            "Bias mitigation",
        ]
        assert topic_queue == expected_topics

    def test_fallback_collation_numbered_lists(self, collation_node):
        """Test fallback collation with numbered lists."""
        proposals = [
            {
                "agent_id": "agent1",
                "proposals": "1. First topic\n2. Second topic\nNot numbered",
            },
            {"agent_id": "agent2", "proposals": "3. Third topic\n4. Fourth topic"},
        ]

        result = collation_node._fallback_collation(proposals)

        expected = ["First topic", "Second topic", "Third topic", "Fourth topic"]
        assert result == expected

    def test_fallback_collation_bulleted_lists(self, collation_node):
        """Test fallback collation with bulleted lists."""
        proposals = [
            {
                "agent_id": "agent1",
                "proposals": "- First topic\n- Second topic\nNot bulleted",
            },
            {"agent_id": "agent2", "proposals": "- Third topic\n- Fourth topic"},
        ]

        result = collation_node._fallback_collation(proposals)

        expected = ["First topic", "Second topic", "Third topic", "Fourth topic"]
        assert result == expected

    def test_fallback_collation_deduplication(self, collation_node):
        """Test that fallback collation removes duplicates."""
        proposals = [
            {
                "agent_id": "agent1",
                "proposals": "1. Duplicate topic\n2. Unique topic 1",
            },
            {
                "agent_id": "agent2",
                "proposals": "1. Duplicate topic\n2. Unique topic 2",
            },
        ]

        result = collation_node._fallback_collation(proposals)

        expected = ["Duplicate topic", "Unique topic 1", "Unique topic 2"]
        assert result == expected

    def test_fallback_collation_limit_topics(self, collation_node):
        """Test that fallback collation limits to 10 topics."""
        proposals = [
            {
                "agent_id": "agent1",
                "proposals": "\n".join([f"{i+1}. Topic {i+1}" for i in range(15)]),
            }
        ]

        result = collation_node._fallback_collation(proposals)

        assert len(result) == 10
        assert result == [f"Topic {i+1}" for i in range(10)]

    def test_fallback_collation_mixed_formats(self, collation_node):
        """Test fallback collation with mixed numbering formats."""
        proposals = [
            {
                "agent_id": "agent1",
                "proposals": "1. Format one\n2) Format two\n3 Format three",
            },
            {"agent_id": "agent2", "proposals": "- Bullet one\n4. Number four"},
        ]

        result = collation_node._fallback_collation(proposals)

        expected = [
            "Format one",
            "Format two",
            "Format three",
            "Bullet one",
            "Number four",
        ]
        assert result == expected

    def test_fallback_collation_empty_proposals(self, collation_node):
        """Test fallback collation with empty proposals."""
        proposals = []
        result = collation_node._fallback_collation(proposals)
        assert result == []

    def test_fallback_collation_no_formatted_topics(self, collation_node):
        """Test fallback collation with no properly formatted topics."""
        proposals = [
            {
                "agent_id": "agent1",
                "proposals": "Just some unformatted text\nAnother line of text",
            },
        ]

        result = collation_node._fallback_collation(proposals)
        assert result == []
