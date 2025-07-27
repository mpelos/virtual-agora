"""Integration tests for ModeratorAgent workflows in Virtual Agora graph context.

These tests verify that the ModeratorAgent properly integrates with the
Virtual Agora state management and LangGraph execution environment.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.base import BaseLanguageModel

from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.state.schema import (
    VirtualAgoraState,
    Message,
    Vote,
    TopicInfo,
    AgentInfo,
)
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.utils.exceptions import StateError


class TestModeratorAgentIntegration:
    """Integration tests for ModeratorAgent with Virtual Agora state."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock(spec=BaseLanguageModel)
        llm.invoke = Mock(return_value=AIMessage(content="Test response"))
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test async response"))
        return llm

    @pytest.fixture
    def sample_config(self) -> VirtualAgoraConfig:
        """Create a sample configuration for testing."""
        config_dict = {
            "session": {"max_duration_minutes": 60, "max_messages": 100},
            "moderator": {"model": "gpt-4", "provider": "openai", "temperature": 0.7},
            "agents": [
                {
                    "model": "gpt-3.5-turbo",
                    "provider": "openai",
                    "count": 2,
                    "temperature": 0.8,
                }
            ],
        }
        return VirtualAgoraConfig(**config_dict)

    @pytest.fixture
    def state_manager(self, sample_config):
        """Create a state manager for testing."""
        return StateManager(sample_config)

    @pytest.fixture
    def moderator_agent(self, mock_llm):
        """Create a ModeratorAgent for testing."""
        return ModeratorAgent(
            agent_id="moderator",
            llm=mock_llm,
            mode="facilitation",
            turn_timeout_seconds=300,
            relevance_threshold=0.7,
            warning_threshold=2,
            mute_duration_minutes=5,
        )

    def test_moderator_agent_initialization(self, moderator_agent):
        """Test that ModeratorAgent initializes correctly with all parameters."""
        assert moderator_agent.agent_id == "moderator"
        # In v1.3, moderator no longer has modes
        assert moderator_agent.turn_timeout_seconds == 300
        assert moderator_agent.relevance_threshold == 0.7
        assert moderator_agent.warning_threshold == 2
        assert moderator_agent.mute_duration_minutes == 5

        # Check that collections are initialized
        assert hasattr(moderator_agent, "relevance_violations")
        assert hasattr(moderator_agent, "muted_agents")

    def test_round_management_workflow(self, moderator_agent):
        """Test complete round management workflow."""
        topic = "AI Ethics"
        round_number = 1

        # Set up some participants
        moderator_agent.speaking_order = ["agent1", "agent2", "agent3"]
        moderator_agent.participation_metrics = {
            "agent1": {"turns_taken": 2},
            "agent2": {"turns_taken": 1},
            "agent3": {"turns_taken": 0},
        }

        # Test topic announcement
        announcement = moderator_agent.announce_topic(topic, round_number)
        assert f"Round {round_number}" in announcement
        assert topic in announcement

        # Test round completion signal
        completion_signal = moderator_agent.signal_round_completion(round_number, topic)
        assert f"Round {round_number}" in completion_signal
        assert topic in completion_signal

    def test_relevance_enforcement_integration(self, moderator_agent):
        """Test complete relevance enforcement workflow."""
        agent_id = "participant1"
        topic = "AI Ethics"
        moderator_agent.current_topic_context = topic

        # Test relevance scoring
        relevance_assessment = {
            "reason": "Message is off-topic",
            "suggestions": "Please focus on AI ethics",
            "relevance_score": 0.3,
            "is_relevant": False,
            "topic": topic,
        }

        # First track the violation
        moderator_agent.track_relevance_violation(
            agent_id, "This is an off-topic message", relevance_assessment
        )

        # Issue first warning
        warning_msg = moderator_agent.issue_relevance_warning(
            agent_id, relevance_assessment
        )
        assert "Relevance Warning" in warning_msg
        assert agent_id in warning_msg

        # Check violation tracking
        assert agent_id in moderator_agent.relevance_violations
        assert moderator_agent.relevance_violations[agent_id]["warnings_issued"] == 1

        # Issue second warning (should trigger muting)
        second_warning = moderator_agent.issue_relevance_warning(
            agent_id, relevance_assessment
        )
        assert "Final Warning" in second_warning

        # Next violation should mute
        mute_msg = moderator_agent.mute_agent(agent_id, "Multiple relevance violations")
        assert "has been temporarily muted" in mute_msg
        assert agent_id in moderator_agent.muted_agents

    def test_polling_workflow_integration(self, moderator_agent):
        """Test complete polling workflow."""
        topic = "AI Ethics"
        eligible_voters = ["participant1", "participant2", "participant3"]

        # Initiate poll
        poll_result = moderator_agent.initiate_conclusion_poll(
            topic, eligible_voters, 10
        )
        assert poll_result["status"] == "initiated"
        poll_id = poll_result["poll_id"]

        # Cast votes
        vote1_result = moderator_agent.cast_vote(
            poll_id, "participant1", "conclude", "We've covered enough"
        )
        assert vote1_result["success"] is True

        vote2_result = moderator_agent.cast_vote(
            poll_id, "participant2", "conclude", "Time to move on"
        )
        assert vote2_result["success"] is True

        vote3_result = moderator_agent.cast_vote(
            poll_id, "participant3", "continue", "Need more discussion"
        )
        assert vote3_result["success"] is True

        # Check poll status instead of tallying (tally_poll_results was removed)
        poll_status = moderator_agent.check_poll_status(poll_id)
        assert poll_status["received_votes"] == 3
        # In v1.3, the ModeratorAgent no longer tallies votes or handles minority considerations
        # Those responsibilities moved to specialized agents

    def test_timeout_handling_integration(self, moderator_agent):
        """Test timeout detection and handling."""
        agent_id = "participant1"

        # Test timeout handling - the handle_agent_timeout method handles all timeout logic
        timeout_msg = moderator_agent.handle_agent_timeout(agent_id)
        assert "did not respond" in timeout_msg
        assert agent_id in timeout_msg
        assert str(moderator_agent.turn_timeout_seconds) in timeout_msg

    def test_error_handling_integration(self, moderator_agent):
        """Test error handling in moderator operations."""
        # Test invalid poll ID
        result = moderator_agent.cast_vote(
            "invalid_poll_id", "participant1", "yes", "reasoning"
        )
        assert result["success"] is False
        assert result["error"] == "Poll invalid_poll_id not found"

        # Test invalid choice in valid poll
        poll_result = moderator_agent.initiate_conclusion_poll(
            "Test Topic", ["participant1"], 5
        )
        poll_id = poll_result["poll_id"]

        result = moderator_agent.cast_vote(
            poll_id, "participant1", "invalid_choice", "reasoning"
        )
        assert result["success"] is False
        assert result["error"] == "invalid_choice"

    def test_comprehensive_discussion_cycle(self, moderator_agent):
        """Test a complete discussion cycle with multiple operations."""
        topic = "AI Ethics"
        participants = ["participant1", "participant2", "participant3"]

        # Set up moderator state
        moderator_agent.speaking_order = participants
        moderator_agent.participation_metrics = {
            "participant1": {"turns_taken": 2},
            "participant2": {"turns_taken": 1},
            "participant3": {"turns_taken": 1},
        }

        # 1. Start discussion round
        announcement = moderator_agent.announce_topic(topic, 1)
        assert topic in announcement

        # 2. Simulate some relevance issues
        moderator_agent.current_topic_context = topic

        # First track a violation (required before issuing a warning)
        relevance_assessment = {
            "reason": "Off-topic",
            "suggestions": "Stay focused",
            "relevance_score": 0.3,
            "is_relevant": False,
            "topic": topic,
        }
        moderator_agent.track_relevance_violation(
            "participant1", "Some off-topic message", relevance_assessment
        )

        # Now issue the warning
        warning_msg = moderator_agent.issue_relevance_warning(
            "participant1", relevance_assessment
        )
        assert "Warning" in warning_msg

        # 3. Conduct conclusion poll
        poll_result = moderator_agent.initiate_conclusion_poll(topic, participants, 10)
        poll_id = poll_result["poll_id"]

        # Cast votes
        moderator_agent.cast_vote(
            poll_id, "participant1", "conclude", "Enough discussion"
        )
        moderator_agent.cast_vote(
            poll_id, "participant2", "conclude", "Time to move on"
        )
        moderator_agent.cast_vote(poll_id, "participant3", "continue", "Need more time")

        # Check poll status instead of tallying
        poll_status = moderator_agent.check_poll_status(poll_id)
        assert poll_status["received_votes"] == 3
        # In v1.3, vote tallying and summary generation moved to specialized agents

        # 5. Signal completion
        completion_msg = moderator_agent.signal_round_completion(1, topic)
        assert topic in completion_msg

    def test_state_consistency_tracking(self, moderator_agent):
        """Test that moderator maintains consistent internal state."""
        # Track initial state
        initial_violations = len(moderator_agent.relevance_violations)
        initial_muted = len(moderator_agent.muted_agents)
        initial_polls = len(moderator_agent.get_active_polls())

        # Perform operations that modify state
        # First track a violation before issuing warning
        moderator_agent.track_relevance_violation(
            "agent1",
            "test message",
            {
                "reason": "test",
                "suggestions": "test",
                "relevance_score": 0.3,
                "is_relevant": False,
                "topic": "test",
            },
        )
        moderator_agent.issue_relevance_warning(
            "agent1", {"reason": "test", "suggestions": "test"}
        )
        moderator_agent.mute_agent("agent2", "test mute")
        poll_result = moderator_agent.initiate_conclusion_poll("Test", ["agent1"], 5)

        # Verify state changes
        assert len(moderator_agent.relevance_violations) == initial_violations + 1
        assert len(moderator_agent.muted_agents) == initial_muted + 1
        assert len(moderator_agent.get_active_polls()) == initial_polls + 1

        # Complete poll to clean up state
        moderator_agent.cast_vote(poll_result["poll_id"], "agent1", "conclude", "done")
        # In v1.3, poll tallying is handled by the flow, not the moderator

        # Verify cleanup
        assert (
            len(moderator_agent.get_active_polls()) == initial_polls
        )  # Poll completed, removed from active
