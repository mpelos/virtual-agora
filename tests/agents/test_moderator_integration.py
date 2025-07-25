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

from virtual_agora.agents.moderator import ModeratorAgent, ModeratorMode
from virtual_agora.state.schema import VirtualAgoraState, Message, Vote, TopicInfo, AgentInfo
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
            "session": {
                "max_duration_minutes": 60,
                "max_messages": 100
            },
            "moderator": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": 0.7
            },
            "agents": [
                {
                    "model": "gpt-3.5-turbo",
                    "provider": "openai",
                    "count": 2,
                    "temperature": 0.8
                }
            ]
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
            mute_duration_minutes=5
        )

    def test_moderator_agent_initialization(self, moderator_agent):
        """Test that ModeratorAgent initializes correctly with all parameters."""
        assert moderator_agent.agent_id == "moderator"
        assert moderator_agent.mode == "facilitation"
        assert moderator_agent.turn_timeout_seconds == 300
        assert moderator_agent.relevance_threshold == 0.7
        assert moderator_agent.warning_threshold == 2
        assert moderator_agent.mute_duration_minutes == 5
        
        # Check that collections are initialized
        assert hasattr(moderator_agent, 'relevance_violations')
        assert hasattr(moderator_agent, 'muted_agents')
        assert hasattr(moderator_agent, 'active_polls')

    def test_round_management_workflow(self, moderator_agent):
        """Test complete round management workflow."""
        topic = "AI Ethics"
        round_number = 1
        
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
        with patch.object(moderator_agent, '_calculate_relevance_score', return_value=0.3):
            relevance_assessment = {
                "reason": "Message is off-topic",
                "suggestions": "Please focus on AI ethics"
            }
            
            # Issue first warning
            warning_msg = moderator_agent.issue_relevance_warning(agent_id, relevance_assessment)
            assert "Relevance Warning" in warning_msg
            assert agent_id in warning_msg
            
            # Check violation tracking
            assert agent_id in moderator_agent.relevance_violations
            assert moderator_agent.relevance_violations[agent_id]["warnings_issued"] == 1
            
            # Issue second warning (should trigger muting)
            second_warning = moderator_agent.issue_relevance_warning(agent_id, relevance_assessment)
            assert "Final Warning" in second_warning
            
            # Next violation should mute
            mute_msg = moderator_agent.mute_agent(agent_id, "Multiple relevance violations")
            assert "has been temporarily muted" in mute_msg
            assert agent_id in moderator_agent.muted_agents

    def test_summarization_integration(self, moderator_agent):
        """Test summarization workflow with mock LLM responses."""
        with patch.object(moderator_agent.llm, 'invoke') as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Test summary generated")
            
            # Test round summary generation
            messages = [
                Message(
                    id="msg1",
                    speaker_id="participant1",
                    speaker_role="participant",
                    content="Discussion point 1",
                    timestamp=datetime.now(),
                    phase=2,
                    topic="AI Ethics"
                )
            ]
            
            round_summary = moderator_agent.generate_round_summary(1, "AI Ethics", messages)
            assert "Test summary generated" in round_summary
            
            # Test topic summary generation
            topic_summary = moderator_agent.generate_topic_summary("AI Ethics", messages)
            assert "Test summary generated" in topic_summary

    def test_polling_workflow_integration(self, moderator_agent):
        """Test complete polling workflow."""
        topic = "AI Ethics"
        eligible_voters = ["participant1", "participant2", "participant3"]
        
        # Initiate poll
        poll_result = moderator_agent.initiate_conclusion_poll(topic, eligible_voters, 10)
        assert poll_result["status"] == "initiated"
        poll_id = poll_result["poll_id"]
        
        # Cast votes
        vote1_result = moderator_agent.cast_vote(poll_id, "participant1", "conclude", "We've covered enough")
        assert vote1_result["status"] == "recorded"
        
        vote2_result = moderator_agent.cast_vote(poll_id, "participant2", "conclude", "Time to move on")
        assert vote2_result["status"] == "recorded"
        
        vote3_result = moderator_agent.cast_vote(poll_id, "participant3", "continue", "Need more discussion")
        assert vote3_result["status"] == "recorded"
        
        # Tally results
        tally_result = moderator_agent.tally_poll_results(poll_id)
        assert tally_result["success"] is True
        assert tally_result["decision"] == "conclude"  # 2 vs 1
        assert tally_result["vote_counts"]["conclude"] == 2
        assert tally_result["vote_counts"]["continue"] == 1
        
        # Handle minority considerations
        minority_msg = moderator_agent.handle_minority_considerations(tally_result, topic)
        assert "Minority Final Considerations" in minority_msg
        assert "participant3" in minority_msg  # The minority voter

    def test_mode_switching_integration(self, moderator_agent):
        """Test mode switching during operations."""
        original_mode = moderator_agent.mode
        assert original_mode == "facilitation"
        
        # Switch to synthesis mode
        moderator_agent.set_mode("synthesis")
        assert moderator_agent.mode == "synthesis"
        
        # Perform synthesis operation
        with patch.object(moderator_agent.llm, 'invoke') as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Synthesis complete")
            
            messages = [
                Message(
                    id="msg1",
                    speaker_id="participant1",
                    speaker_role="participant",
                    content="Test message",
                    timestamp=datetime.now(),
                    phase=2,
                    topic="AI Ethics"
                )
            ]
            
            summary = moderator_agent.generate_topic_summary("AI Ethics", messages)
            assert "Synthesis complete" in summary
        
        # Switch back to facilitation
        moderator_agent.set_mode("facilitation")
        assert moderator_agent.mode == "facilitation"

    def test_timeout_handling_integration(self, moderator_agent):
        """Test timeout detection and handling."""
        agent_id = "participant1"
        
        # Mock a timeout situation
        past_time = datetime.now() - timedelta(seconds=400)  # 400 seconds ago
        
        # Check timeout detection
        is_timeout = moderator_agent._is_turn_timeout(past_time)
        assert is_timeout
        
        # Test timeout announcement
        timeout_msg = moderator_agent.announce_turn_timeout(agent_id)
        assert "timeout" in timeout_msg.lower()
        assert agent_id in timeout_msg

    def test_error_handling_integration(self, moderator_agent):
        """Test error handling in moderator operations."""
        # Test invalid poll ID
        with pytest.raises(ValueError, match="Poll .* not found"):
            moderator_agent.cast_vote("invalid_poll_id", "participant1", "yes", "reasoning")
        
        # Test invalid choice in valid poll
        poll_result = moderator_agent.initiate_conclusion_poll("Test Topic", ["participant1"], 5)
        poll_id = poll_result["poll_id"]
        
        with pytest.raises(ValueError, match="Invalid choice"):
            moderator_agent.cast_vote(poll_id, "participant1", "invalid_choice", "reasoning")

    def test_comprehensive_discussion_cycle(self, moderator_agent):
        """Test a complete discussion cycle with multiple operations."""
        topic = "AI Ethics"
        participants = ["participant1", "participant2", "participant3"]
        
        # 1. Start discussion round
        announcement = moderator_agent.announce_topic(topic, 1)
        assert topic in announcement
        
        # 2. Simulate some relevance issues
        moderator_agent.current_topic_context = topic
        with patch.object(moderator_agent, '_calculate_relevance_score', return_value=0.3):
            relevance_assessment = {"reason": "Off-topic", "suggestions": "Stay focused"}
            warning_msg = moderator_agent.issue_relevance_warning("participant1", relevance_assessment)
            assert "Warning" in warning_msg
        
        # 3. Conduct conclusion poll
        poll_result = moderator_agent.initiate_conclusion_poll(topic, participants, 10)
        poll_id = poll_result["poll_id"]
        
        # Cast votes
        moderator_agent.cast_vote(poll_id, "participant1", "conclude", "Enough discussion")
        moderator_agent.cast_vote(poll_id, "participant2", "conclude", "Time to move on")
        moderator_agent.cast_vote(poll_id, "participant3", "continue", "Need more time")
        
        # Tally results
        tally_result = moderator_agent.tally_poll_results(poll_id)
        assert tally_result["decision"] == "conclude"
        
        # 4. Generate summaries
        with patch.object(moderator_agent.llm, 'invoke') as mock_invoke:
            mock_invoke.return_value = AIMessage(content="Summary of discussion")
            
            messages = [
                Message(
                    id="msg1",
                    speaker_id="participant1",
                    speaker_role="participant",
                    content="Point about AI safety",
                    timestamp=datetime.now(),
                    phase=2,
                    topic=topic
                )
            ]
            
            round_summary = moderator_agent.generate_round_summary(1, topic, messages)
            topic_summary = moderator_agent.generate_topic_summary(topic, messages)
            
            assert "Summary of discussion" in round_summary
            assert "Summary of discussion" in topic_summary
        
        # 5. Signal completion
        completion_msg = moderator_agent.signal_round_completion(1, topic)
        assert topic in completion_msg

    def test_state_consistency_tracking(self, moderator_agent):
        """Test that moderator maintains consistent internal state."""
        # Track initial state
        initial_violations = len(moderator_agent.relevance_violations)
        initial_muted = len(moderator_agent.muted_agents)
        initial_polls = len(moderator_agent.active_polls)
        
        # Perform operations that modify state
        moderator_agent.issue_relevance_warning("agent1", {"reason": "test", "suggestions": "test"})
        moderator_agent.mute_agent("agent2", "test mute")
        poll_result = moderator_agent.initiate_conclusion_poll("Test", ["agent1"], 5)
        
        # Verify state changes
        assert len(moderator_agent.relevance_violations) == initial_violations + 1
        assert len(moderator_agent.muted_agents) == initial_muted + 1
        assert len(moderator_agent.active_polls) == initial_polls + 1
        
        # Complete poll to clean up state
        moderator_agent.cast_vote(poll_result["poll_id"], "agent1", "conclude", "done")
        moderator_agent.tally_poll_results(poll_result["poll_id"])
        
        # Verify cleanup
        assert len(moderator_agent.active_polls) == initial_polls  # Poll completed, removed from active