"""Tests for DiscussionAgent implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from virtual_agora.agents.discussion_agent import DiscussionAgent, AgentState, VoteType
from virtual_agora.state.schema import Message, Vote


class TestDiscussionAgent:
    """Test DiscussionAgent class."""

    def setup_method(self):
        """Set up test method."""
        # Create mock LLM
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = (
            self.mock_llm
        )  # For error handling chain

        # Create agent with error handling disabled for testing
        self.agent = DiscussionAgent(
            agent_id="test-agent",
            llm=self.mock_llm,
            max_context_messages=5,
            warning_threshold=2,
            enable_error_handling=False,
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == "test-agent"
        assert self.agent.llm == self.mock_llm
        assert self.agent.role == "participant"
        assert self.agent.max_context_messages == 5
        assert self.agent.warning_threshold == 2
        assert self.agent.current_state == AgentState.ACTIVE
        assert self.agent.warning_count == 0
        assert self.agent.current_topic is None
        assert len(self.agent.vote_history) == 0
        assert "thoughtful participant" in self.agent.system_prompt

    def test_reset_for_new_topic(self):
        """Test resetting agent for new topic."""
        # Set some initial state
        self.agent.warning_count = 1
        self.agent.current_state = AgentState.WARNED
        self.agent.current_topic = "old_topic"

        # Reset for new topic
        self.agent.reset_for_new_topic("new_topic")

        assert self.agent.current_topic == "new_topic"
        assert self.agent.warning_count == 0
        assert self.agent.current_state == AgentState.ACTIVE

    def test_add_warning(self):
        """Test adding warnings to agent."""
        # First warning
        should_mute = self.agent.add_warning("Test warning 1")
        assert not should_mute
        assert self.agent.warning_count == 1
        assert self.agent.current_state == AgentState.WARNED

        # Second warning (should mute)
        should_mute = self.agent.add_warning("Test warning 2")
        assert should_mute
        assert self.agent.warning_count == 2
        assert self.agent.current_state == AgentState.MUTED

    def test_is_muted(self):
        """Test muted state checking."""
        assert not self.agent.is_muted()

        # Add warnings to mute
        self.agent.add_warning("Warning 1")
        self.agent.add_warning("Warning 2")

        assert self.agent.is_muted()

    def test_get_agent_state_info(self):
        """Test getting agent state info."""
        self.agent.current_topic = "Test Topic"
        self.agent.warning_count = 1
        self.agent.message_count = 5

        info = self.agent.get_agent_state_info()

        assert info["agent_id"] == "test-agent"
        assert info["state"] == AgentState.ACTIVE.value
        assert info["warning_count"] == 1
        assert info["current_topic"] == "Test Topic"
        assert info["vote_count"] == 0
        assert info["message_count"] == 5


class TestDiscussionAgentTopicProposals:
    """Test topic proposal functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "test-agent", self.mock_llm, enable_error_handling=False
        )

    def test_propose_topics_success(self):
        """Test successful topic proposal."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """1. Topic one for discussion
2. Topic two with details
3. Topic three analysis
4. Topic four exploration"""
        self.mock_llm.invoke.return_value = mock_response

        topics = self.agent.propose_topics("Main Topic")

        assert len(topics) == 4
        assert "Topic one for discussion" in topics
        assert "Topic two with details" in topics
        assert "Topic three analysis" in topics
        assert "Topic four exploration" in topics

    def test_propose_topics_with_context(self):
        """Test topic proposal with context messages."""
        mock_response = Mock()
        mock_response.content = """1. Context-aware topic one
2. Context-aware topic two
3. Context-aware topic three"""
        self.mock_llm.invoke.return_value = mock_response

        context = [
            {
                "id": "1",
                "speaker_id": "user",
                "speaker_role": "user",
                "content": "Previous discussion about AI ethics",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": "AI",
            }
        ]

        topics = self.agent.propose_topics("AI Ethics", context)

        assert len(topics) == 3
        self.mock_llm.invoke.assert_called_once()

    def test_propose_topics_muted_agent(self):
        """Test topic proposal when agent is muted."""
        # Mute the agent
        self.agent.add_warning("Warning 1")
        self.agent.add_warning("Warning 2")

        topics = self.agent.propose_topics("Test Topic")

        assert topics == []
        self.mock_llm.invoke.assert_not_called()

    def test_parse_topic_proposals_numbered(self):
        """Test parsing numbered topic proposals."""
        response = """1. First topic for discussion
2. Second topic analysis
3. Third topic exploration"""

        topics = self.agent._parse_topic_proposals(response)

        assert len(topics) == 3
        assert "First topic for discussion" in topics
        assert "Second topic analysis" in topics
        assert "Third topic exploration" in topics

    def test_parse_topic_proposals_bullet_points(self):
        """Test parsing bullet point topic proposals."""
        response = """- First bullet topic
* Second bullet topic
• Third bullet topic"""

        topics = self.agent._parse_topic_proposals(response)

        assert len(topics) == 3
        assert "First bullet topic" in topics
        assert "Second bullet topic" in topics
        assert "Third bullet topic" in topics

    def test_parse_topic_proposals_mixed_format(self):
        """Test parsing mixed format proposals."""
        response = """1. Numbered topic one
- Bullet topic two
3. Numbered topic three"""

        topics = self.agent._parse_topic_proposals(response)

        # Should parse numbered format first
        assert len(topics) == 2
        assert "Numbered topic one" in topics
        assert "Numbered topic three" in topics

    def test_parse_topic_proposals_max_five(self):
        """Test that parsing limits to max 5 topics."""
        response = """1. Topic one
2. Topic two  
3. Topic three
4. Topic four
5. Topic five
6. Topic six
7. Topic seven"""

        topics = self.agent._parse_topic_proposals(response)

        assert len(topics) == 5
        assert "Topic six" not in topics
        assert "Topic seven" not in topics


class TestDiscussionAgentVoting:
    """Test voting functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "test-agent", self.mock_llm, enable_error_handling=False
        )

    def test_vote_on_agenda_success(self):
        """Test successful agenda voting."""
        mock_response = Mock()
        mock_response.content = "I prefer topics 1, 3, 2 in that order because..."
        self.mock_llm.invoke.return_value = mock_response

        topics = ["Topic A", "Topic B", "Topic C"]

        vote_response = self.agent.vote_on_agenda(topics)

        assert vote_response == "I prefer topics 1, 3, 2 in that order because..."
        assert len(self.agent.vote_history) == 1

        vote = self.agent.vote_history[0]
        assert vote["voter_id"] == "test-agent"
        assert vote["vote_type"] == VoteType.AGENDA_SELECTION.value
        assert vote["phase"] == 1

    def test_vote_on_agenda_muted_agent(self):
        """Test agenda voting when agent is muted."""
        # Mute the agent
        self.agent.add_warning("Warning 1")
        self.agent.add_warning("Warning 2")

        topics = ["Topic A", "Topic B"]
        vote_response = self.agent.vote_on_agenda(topics)

        assert "muted" in vote_response.lower()
        assert len(self.agent.vote_history) == 0
        self.mock_llm.invoke.assert_not_called()

    def test_vote_on_topic_conclusion_yes(self):
        """Test voting yes on topic conclusion."""
        mock_response = Mock()
        mock_response.content = (
            "Yes, I think we have covered the main points adequately."
        )
        self.mock_llm.invoke.return_value = mock_response

        vote_result, reasoning = self.agent.vote_on_topic_conclusion("Test Topic")

        assert vote_result is True
        assert "covered the main points adequately" in reasoning
        assert len(self.agent.vote_history) == 1

        vote = self.agent.vote_history[0]
        assert vote["choice"] == "Yes"
        assert vote["vote_type"] == VoteType.TOPIC_CONCLUSION.value

    def test_vote_on_topic_conclusion_no(self):
        """Test voting no on topic conclusion."""
        mock_response = Mock()
        mock_response.content = "No, we need more discussion on this important aspect."
        self.mock_llm.invoke.return_value = mock_response

        vote_result, reasoning = self.agent.vote_on_topic_conclusion("Test Topic")

        assert vote_result is False
        assert "need more discussion" in reasoning

        vote = self.agent.vote_history[0]
        assert vote["choice"] == "No"

    def test_parse_conclusion_vote_clear_yes(self):
        """Test parsing clear yes vote."""
        response = "Yes, the discussion has been thorough and complete."

        vote_result, reasoning = self.agent._parse_conclusion_vote(response)

        assert vote_result is True
        assert "the discussion has been thorough and complete." in reasoning

    def test_parse_conclusion_vote_clear_no(self):
        """Test parsing clear no vote."""
        response = "No, there are still important aspects to consider."

        vote_result, reasoning = self.agent._parse_conclusion_vote(response)

        assert vote_result is False
        assert "there are still important aspects to consider." in reasoning

    def test_parse_conclusion_vote_ambiguous(self):
        """Test parsing ambiguous vote defaults to no."""
        response = "I think maybe we could discuss more but it's also been good."

        vote_result, reasoning = self.agent._parse_conclusion_vote(response)

        assert vote_result is False  # Default to No for ambiguous
        assert len(reasoning) > 0

    def test_vote_on_conclusion_with_context(self):
        """Test conclusion voting with context messages."""
        mock_response = Mock()
        mock_response.content = "Yes, based on the previous discussion."
        self.mock_llm.invoke.return_value = mock_response

        context = [
            {
                "id": "1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Previous discussion content",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Test Topic",
            }
        ]

        vote_result, reasoning = self.agent.vote_on_topic_conclusion(
            "Test Topic", context
        )

        assert vote_result is True
        self.mock_llm.invoke.assert_called_once()


class TestDiscussionAgentDiscussion:
    """Test discussion response generation."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "test-agent",
            self.mock_llm,
            max_context_messages=3,
            enable_error_handling=False,
        )

    def test_generate_discussion_response(self):
        """Test generating discussion response."""
        mock_response = Mock()
        mock_response.content = "This is a thoughtful response to the topic."
        self.mock_llm.invoke.return_value = mock_response

        response = self.agent.generate_discussion_response("Test Topic")

        assert response == "This is a thoughtful response to the topic."
        assert self.agent.current_topic == "Test Topic"

    def test_generate_discussion_response_with_context(self):
        """Test discussion response with context."""
        mock_response = Mock()
        mock_response.content = "Building on previous points, I think..."
        self.mock_llm.invoke.return_value = mock_response

        context = [
            {
                "id": str(i),
                "speaker_id": f"agent{i}",
                "speaker_role": "participant",
                "content": f"Message {i} content",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Test Topic",
            }
            for i in range(5)  # 5 messages, but agent limits to 3
        ]

        response = self.agent.generate_discussion_response("Test Topic", context)

        assert response == "Building on previous points, I think..."
        # Should have called with limited context (max 3 messages)
        self.mock_llm.invoke.assert_called_once()

    def test_generate_discussion_response_muted(self):
        """Test discussion response when muted."""
        # Mute the agent
        self.agent.add_warning("Warning 1")
        self.agent.add_warning("Warning 2")

        response = self.agent.generate_discussion_response("Test Topic")

        assert response == ""
        self.mock_llm.invoke.assert_not_called()

    def test_generate_discussion_response_topic_change(self):
        """Test response when topic changes."""
        # Set initial topic
        self.agent.current_topic = "Old Topic"
        self.agent.warning_count = 1

        mock_response = Mock()
        mock_response.content = "Response to new topic."
        self.mock_llm.invoke.return_value = mock_response

        response = self.agent.generate_discussion_response("New Topic")

        assert response == "Response to new topic."
        assert self.agent.current_topic == "New Topic"
        assert self.agent.warning_count == 0  # Should reset for new topic


class TestDiscussionAgentMinorityConsideration:
    """Test minority consideration functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "test-agent", self.mock_llm, enable_error_handling=False
        )

    def test_provide_minority_consideration(self):
        """Test providing minority consideration."""
        mock_response = Mock()
        mock_response.content = (
            "I believe there's an important aspect we haven't fully explored."
        )
        self.mock_llm.invoke.return_value = mock_response

        consideration = self.agent.provide_minority_consideration("Concluded Topic")

        assert (
            consideration
            == "I believe there's an important aspect we haven't fully explored."
        )
        self.mock_llm.invoke.assert_called_once()

    def test_provide_minority_consideration_with_context(self):
        """Test minority consideration with discussion context."""
        mock_response = Mock()
        mock_response.content = (
            "Based on our discussion, I still have concerns about..."
        )
        self.mock_llm.invoke.return_value = mock_response

        context = [
            {
                "id": "1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Discussion point about the topic",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Concluded Topic",
            }
        ]

        consideration = self.agent.provide_minority_consideration(
            "Concluded Topic", context
        )

        assert (
            consideration == "Based on our discussion, I still have concerns about..."
        )

    def test_provide_minority_consideration_muted(self):
        """Test minority consideration when agent is muted."""
        # Mute the agent
        self.agent.add_warning("Warning 1")
        self.agent.add_warning("Warning 2")

        consideration = self.agent.provide_minority_consideration("Topic")

        assert consideration == ""
        self.mock_llm.invoke.assert_not_called()


class TestDiscussionAgentPromptBuilding:
    """Test prompt building methods."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "test-agent", self.mock_llm, enable_error_handling=False
        )

    def test_build_topic_proposal_prompt(self):
        """Test building topic proposal prompt."""
        prompt = self.agent._build_topic_proposal_prompt("AI Ethics")

        assert "AI Ethics" in prompt
        assert "3-5 specific sub-topics" in prompt
        assert "numbered list" in prompt
        assert "1. [First sub-topic]" in prompt

    def test_build_topic_proposal_prompt_with_context(self):
        """Test topic proposal prompt with context."""
        context = [
            {
                "id": "1",
                "speaker_id": "user",
                "speaker_role": "user",
                "content": "Previous discussion about machine learning fairness and bias",
                "timestamp": datetime.now(),
                "phase": 1,
                "topic": "AI",
            }
        ]

        prompt = self.agent._build_topic_proposal_prompt("AI Ethics", context)

        assert "AI Ethics" in prompt
        assert "context from our discussion" in prompt
        assert "user:" in prompt
        assert "machine learning fairness" in prompt

    def test_build_agenda_voting_prompt(self):
        """Test building agenda voting prompt."""
        topics = ["Topic A", "Topic B", "Topic C"]

        prompt = self.agent._build_agenda_voting_prompt(topics)

        assert "proposed discussion topics" in prompt
        assert "preferred order" in prompt
        assert "1. Topic A" in prompt
        assert "2. Topic B" in prompt
        assert "3. Topic C" in prompt
        assert "reasoning for your preferences" in prompt

    def test_build_conclusion_voting_prompt(self):
        """Test building conclusion voting prompt."""
        prompt = self.agent._build_conclusion_voting_prompt("Test Topic")

        assert "conclude the discussion on 'Test Topic'" in prompt
        assert "'Yes' or 'No'" in prompt
        assert "key aspects" in prompt
        assert "adequately discussed" in prompt

    def test_build_conclusion_voting_prompt_with_context(self):
        """Test conclusion voting prompt with context."""
        context = [
            {
                "id": "1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Recent discussion point about the topic",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Test Topic",
            }
        ]

        prompt = self.agent._build_conclusion_voting_prompt("Test Topic", context)

        assert "Test Topic" in prompt
        assert "recent discussion" in prompt
        assert "agent1:" in prompt

    def test_build_discussion_prompt(self):
        """Test building discussion prompt."""
        prompt = self.agent._build_discussion_prompt("Current Topic")

        assert "currently discussing: Current Topic" in prompt
        assert "thoughtful, well-reasoned comment" in prompt
        assert "Stay strictly on the topic" in prompt
        assert "2-4 sentences" in prompt

    def test_build_discussion_prompt_with_context(self):
        """Test discussion prompt with context."""
        context = [
            {
                "id": "1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "content": "Previous discussion point that is quite long and should be truncated in the prompt to avoid overwhelming context",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Current Topic",
            }
        ]

        prompt = self.agent._build_discussion_prompt("Current Topic", context)

        assert "Current Topic" in prompt
        assert "recent discussion context" in prompt
        assert "agent1:" in prompt
        # Should truncate long content
        assert (
            len(prompt.split("agent1:")[1].split("\n")[0]) <= 210
        )  # 200 chars + "..."

    def test_build_minority_consideration_prompt(self):
        """Test building minority consideration prompt."""
        prompt = self.agent._build_minority_consideration_prompt("Concluded Topic")

        assert "Concluded Topic" in prompt
        assert "concluded by majority vote" in prompt
        assert "voted to continue" in prompt
        assert "final considerations" in prompt
        assert "overlooked aspects" in prompt


class TestDiscussionAgentErrorHandling:
    """Test error handling in DiscussionAgent."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "test-agent", self.mock_llm, enable_error_handling=False
        )

    def test_propose_topics_error_handling(self):
        """Test error handling in topic proposal."""
        self.mock_llm.invoke.side_effect = Exception("LLM Error")

        topics = self.agent.propose_topics("Test Topic")

        assert topics == []

    def test_vote_on_agenda_error_handling(self):
        """Test error handling in agenda voting."""
        self.mock_llm.invoke.side_effect = Exception("LLM Error")

        vote_response = self.agent.vote_on_agenda(["Topic A"])

        assert "difficulty processing" in vote_response

    def test_vote_on_conclusion_error_handling(self):
        """Test error handling in conclusion voting."""
        self.mock_llm.invoke.side_effect = Exception("LLM Error")

        vote_result, reasoning = self.agent.vote_on_topic_conclusion("Test Topic")

        assert vote_result is False
        assert "difficulty processing" in reasoning

    def test_generate_discussion_response_error_handling(self):
        """Test error handling in discussion response."""
        self.mock_llm.invoke.side_effect = Exception("LLM Error")

        response = self.agent.generate_discussion_response("Test Topic")

        assert "difficulty contributing" in response

    def test_provide_minority_consideration_error_handling(self):
        """Test error handling in minority consideration."""
        self.mock_llm.invoke.side_effect = Exception("LLM Error")

        consideration = self.agent.provide_minority_consideration("Test Topic")

        assert "difficulty providing" in consideration


@pytest.fixture
def sample_context_messages():
    """Sample context messages for testing."""
    return [
        {
            "id": "1",
            "speaker_id": "user",
            "speaker_role": "user",
            "content": "Let's discuss AI safety",
            "timestamp": datetime.now(),
            "phase": 1,
            "topic": "AI Safety",
        },
        {
            "id": "2",
            "speaker_id": "agent1",
            "speaker_role": "participant",
            "content": "AI safety is crucial for future development",
            "timestamp": datetime.now(),
            "phase": 2,
            "topic": "AI Safety",
        },
    ]


class TestDiscussionAgentIntegration:
    """Integration tests for DiscussionAgent."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock(spec=BaseChatModel)
        self.mock_llm.__class__.__name__ = "ChatOpenAI"
        self.mock_llm.model_name = "gpt-4"
        self.mock_llm.with_fallbacks.return_value = self.mock_llm

        self.agent = DiscussionAgent(
            "integration-agent", self.mock_llm, enable_error_handling=False
        )

    def test_full_discussion_workflow(self, sample_context_messages):
        """Test complete discussion workflow."""
        # Mock responses for different phases
        responses = [
            Mock(
                content="1. Ethics in AI\n2. Safety measures\n3. Regulatory frameworks"
            ),  # Topic proposal
            Mock(
                content="I prefer topics 1, 3, 2 because ethics is fundamental"
            ),  # Agenda vote
            Mock(
                content="Ethics in AI requires careful consideration of bias and fairness"
            ),  # Discussion
            Mock(
                content="Yes, we have covered the main ethical concerns adequately"
            ),  # Conclusion vote
            Mock(
                content="I wanted to add that transparency is also crucial for AI ethics"
            ),  # Minority consideration
        ]
        self.mock_llm.invoke.side_effect = responses

        # 1. Propose topics
        topics = self.agent.propose_topics("AI Development", sample_context_messages)
        assert len(topics) == 3
        assert "Ethics in AI" in topics

        # 2. Vote on agenda
        vote_response = self.agent.vote_on_agenda(topics, sample_context_messages)
        assert "prefer topics 1, 3, 2" in vote_response
        assert len(self.agent.vote_history) == 1

        # 3. Generate discussion response
        discussion_response = self.agent.generate_discussion_response(
            "Ethics in AI", sample_context_messages
        )
        assert "careful consideration" in discussion_response

        # 4. Vote on conclusion
        vote_result, reasoning = self.agent.vote_on_topic_conclusion(
            "Ethics in AI", sample_context_messages
        )
        assert vote_result is True
        assert "adequately" in reasoning
        assert len(self.agent.vote_history) == 2

        # 5. Provide minority consideration (if needed)
        consideration = self.agent.provide_minority_consideration(
            "Ethics in AI", sample_context_messages
        )
        assert "transparency" in consideration

        # Check final state
        assert self.agent.current_state == AgentState.ACTIVE
        assert len(self.agent.vote_history) == 2
        assert self.agent.current_topic == "Ethics in AI"

    def test_agent_state_management_workflow(self):
        """Test agent state management through warnings and muting."""
        # Start active
        assert self.agent.current_state == AgentState.ACTIVE
        assert not self.agent.is_muted()

        # Add first warning
        should_mute = self.agent.add_warning("Off-topic response")
        assert not should_mute
        assert self.agent.current_state == AgentState.WARNED

        # Add second warning (should mute)
        should_mute = self.agent.add_warning("Inappropriate language")
        assert should_mute
        assert self.agent.current_state == AgentState.MUTED
        assert self.agent.is_muted()

        # Reset for new topic should clear warnings
        self.agent.reset_for_new_topic("New Topic")
        assert self.agent.current_state == AgentState.ACTIVE
        assert not self.agent.is_muted()
        assert self.agent.warning_count == 0

    def test_context_message_handling(self):
        """Test handling of context messages with length limits."""
        # Create many context messages
        many_context = [
            {
                "id": str(i),
                "speaker_id": f"agent{i}",
                "speaker_role": "participant",
                "content": f"Context message {i} content",
                "timestamp": datetime.now(),
                "phase": 2,
                "topic": "Test Topic",
            }
            for i in range(10)
        ]

        mock_response = Mock()
        mock_response.content = "Response considering context"
        self.mock_llm.invoke.return_value = mock_response

        # Should limit context to max_context_messages (default 10, but our agent uses 5)
        response = self.agent.generate_discussion_response("Test Topic", many_context)

        assert response == "Response considering context"
        # Verify that invoke was called (context was properly limited)
        self.mock_llm.invoke.assert_called_once()
