"""Tests for MessageCoordinator class.

This module provides comprehensive tests for the centralized message coordination
functionality, ensuring 100% coverage and backward compatibility.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from src.virtual_agora.flow.message_coordinator import MessageCoordinator
from src.virtual_agora.flow.round_manager import RoundManager
from src.virtual_agora.state.schema import VirtualAgoraState
from src.virtual_agora.context.types import ContextData
from src.virtual_agora.context.message_processor import ProcessedMessage


class TestMessageCoordinator:
    """Test suite for MessageCoordinator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator(self.round_manager)

    def create_test_state(self, **kwargs) -> Dict[str, Any]:
        """Create a test state with default values.

        Args:
            **kwargs: Override default state values

        Returns:
            Dictionary representing VirtualAgoraState for testing
        """
        default_state = {
            "current_round": 2,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "messages": [],
            "round_summaries": [],
            "topic_summaries": {},
            "speaking_order": ["agent1", "agent2", "agent3"],
        }
        default_state.update(kwargs)
        return default_state

    def create_test_context_data(self, **kwargs):
        """Create mock ContextData for testing."""
        context_data = Mock(spec=ContextData)
        context_data.formatted_context_messages = kwargs.get("messages", [])
        context_data.has_user_input = kwargs.get("has_user_input", True)
        context_data.user_input = kwargs.get("user_input", "Test Theme")
        context_data.has_context_documents = kwargs.get("has_context_documents", False)
        context_data.context_documents = kwargs.get("context_documents", "")
        context_data.has_round_summaries = kwargs.get("has_round_summaries", True)
        context_data.round_summaries = kwargs.get(
            "round_summaries",
            [{"summary_text": "Round 1 summary"}, {"summary_text": "Round 2 summary"}],
        )
        return context_data

    # ===== Tests for assemble_agent_context =====

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_assemble_agent_context_basic(self, mock_get_builder):
        """Test basic agent context assembly."""
        state = self.create_test_state()

        # Setup mock context builder
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data()
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        result_text, result_messages = self.message_coordinator.assemble_agent_context(
            agent_id="agent1",
            round_num=2,
            state=state,
            agent_position=0,
            speaking_order=["agent1", "agent2", "agent3"],
            system_prompt="Test prompt",
        )

        # Verify context builder was called correctly
        mock_builder.build_context.assert_called_once()
        call_kwargs = mock_builder.build_context.call_args.kwargs
        assert call_kwargs["agent_id"] == "agent1"
        assert call_kwargs["current_round"] == 2
        assert call_kwargs["agent_position"] == 0
        assert call_kwargs["topic"] == "Test Topic"
        assert call_kwargs["system_prompt"] == "Test prompt"

        # Verify result format
        assert isinstance(result_text, str)
        assert isinstance(result_messages, list)
        assert "Theme: Test Theme" in result_text
        assert "Current Topic: Test Topic" in result_text
        assert "Round: 2" in result_text
        assert "Your speaking position: 1/3" in result_text

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_assemble_agent_context_with_defaults(self, mock_get_builder):
        """Test agent context assembly with default parameters."""
        state = self.create_test_state()

        # Setup mock context builder
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data()
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        result_text, result_messages = self.message_coordinator.assemble_agent_context(
            agent_id="agent1", round_num=2, state=state
        )

        # Verify defaults were used
        call_kwargs = mock_builder.build_context.call_args.kwargs
        assert call_kwargs["agent_position"] == 0
        assert call_kwargs["speaking_order"] == [
            "agent1",
            "agent2",
            "agent3",
        ]  # From state
        assert call_kwargs["system_prompt"] == ""

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_assemble_agent_context_with_context_documents(self, mock_get_builder):
        """Test agent context assembly with context documents."""
        state = self.create_test_state()

        # Setup mock context builder with documents
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data(
            has_context_documents=True,
            context_documents="Important background information",
        )
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        result_text, result_messages = self.message_coordinator.assemble_agent_context(
            agent_id="agent1", round_num=2, state=state
        )

        # Verify context documents are included
        assert "Background Information:" in result_text
        assert "Important background information" in result_text

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_assemble_agent_context_with_topic_conclusions(self, mock_get_builder):
        """Test agent context assembly with previous topic conclusions."""
        state = self.create_test_state(
            topic_summaries={
                "Previous Topic_conclusion": "Previous topic was concluded successfully",
                "Test Topic_conclusion": "Current topic conclusion",  # Should be excluded
                "Another Topic_conclusion": "Another topic summary",
            }
        )

        # Setup mock context builder
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data()
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        result_text, result_messages = self.message_coordinator.assemble_agent_context(
            agent_id="agent1", round_num=2, state=state
        )

        # Verify previous topic conclusions are included
        assert "Previously Concluded Topics:" in result_text
        assert (
            "Previous Topic: Previous topic was concluded successfully" in result_text
        )
        assert "Another Topic: Another topic summary" in result_text
        # Current topic conclusion should be excluded
        assert "Test Topic_conclusion" not in result_text

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_assemble_agent_context_caching(self, mock_get_builder):
        """Test that context builders are cached properly."""
        state = self.create_test_state()

        # Setup mock context builder
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data()
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        # First call
        self.message_coordinator.assemble_agent_context("agent1", 2, state)

        # Second call
        self.message_coordinator.assemble_agent_context("agent2", 3, state)

        # Verify get_context_builder was only called once (cached)
        assert mock_get_builder.call_count == 1

    # ===== Tests for store_user_message =====

    def test_store_user_message_basic(self):
        """Test basic user message storage."""
        state = self.create_test_state()

        result = self.message_coordinator.store_user_message(
            content="User message content", round_num=2, state=state
        )

        # Verify result structure
        assert "messages" in result
        assert "user_participation_message" in result
        assert result["user_participation_message"] == "User message content"

        # Verify message format
        user_msg = result["messages"][0]
        assert isinstance(user_msg, HumanMessage)
        assert user_msg.content == "User message content"

        # Verify metadata
        additional_kwargs = user_msg.additional_kwargs
        assert additional_kwargs["speaker_id"] == "user"
        assert additional_kwargs["speaker_role"] == "user"
        assert additional_kwargs["round"] == 3  # Next round by default
        assert additional_kwargs["topic"] == "Test Topic"
        assert additional_kwargs["participation_type"] == "user_turn_participation"
        assert "timestamp" in additional_kwargs

    def test_store_user_message_current_round(self):
        """Test user message storage for current round."""
        state = self.create_test_state()

        result = self.message_coordinator.store_user_message(
            content="User message content",
            round_num=2,
            state=state,
            use_next_round=False,
        )

        # Verify current round is used
        user_msg = result["messages"][0]
        assert user_msg.additional_kwargs["round"] == 2

    def test_store_user_message_custom_topic(self):
        """Test user message storage with custom topic."""
        state = self.create_test_state()

        result = self.message_coordinator.store_user_message(
            content="User message content",
            round_num=2,
            state=state,
            topic="Custom Topic",
        )

        # Verify custom topic is used
        user_msg = result["messages"][0]
        assert user_msg.additional_kwargs["topic"] == "Custom Topic"

    def test_store_user_message_custom_participation_type(self):
        """Test user message storage with custom participation type."""
        state = self.create_test_state()

        result = self.message_coordinator.store_user_message(
            content="User message content",
            round_num=2,
            state=state,
            participation_type="custom_participation",
        )

        # Verify custom participation type is used
        user_msg = result["messages"][0]
        assert (
            user_msg.additional_kwargs["participation_type"] == "custom_participation"
        )

    def test_store_user_message_no_active_topic(self):
        """Test user message storage when no active topic is set."""
        state = self.create_test_state(active_topic=None)

        result = self.message_coordinator.store_user_message(
            content="User message content", round_num=2, state=state
        )

        # Verify default topic is used
        user_msg = result["messages"][0]
        assert user_msg.additional_kwargs["topic"] == "Unknown Topic"

    # ===== Tests for get_messages_for_round =====

    def test_get_messages_for_round_basic(self):
        """Test basic message retrieval for specific round."""
        # Create test messages
        test_messages = [
            {
                "content": "Message 1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 1,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Message 2",
                "speaker_id": "agent2",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Message 3",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        state = self.create_test_state(messages=test_messages)

        result = self.message_coordinator.get_messages_for_round(2, state)

        # Verify correct filtering
        assert len(result) == 2
        assert all(isinstance(msg, ProcessedMessage) for msg in result)
        assert all(msg.round_number == 2 for msg in result)

    def test_get_messages_for_round_with_topic_filter(self):
        """Test message retrieval with topic filtering."""
        # Create test messages with different topics
        test_messages = [
            {
                "content": "Message 1",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Topic A",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Message 2",
                "speaker_id": "agent2",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Topic B",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        state = self.create_test_state(messages=test_messages)

        result = self.message_coordinator.get_messages_for_round(
            2, state, topic="Topic A"
        )

        # Verify topic filtering
        assert len(result) == 1
        assert result[0].topic == "Topic A"

    def test_get_messages_for_round_empty_state(self):
        """Test message retrieval from empty state."""
        state = self.create_test_state(messages=[])

        result = self.message_coordinator.get_messages_for_round(2, state)

        assert len(result) == 0
        assert isinstance(result, list)

    # ===== Tests for get_user_messages_for_round =====

    def test_get_user_messages_for_round(self):
        """Test user message retrieval for specific round."""
        # Create test messages with mix of user and agent messages
        test_messages = [
            {
                "content": "User message",
                "speaker_id": "user",
                "speaker_role": "user",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Agent message",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        state = self.create_test_state(messages=test_messages)

        result = self.message_coordinator.get_user_messages_for_round(2, state)

        # Verify only user messages are returned
        assert len(result) == 1
        assert result[0].speaker_role == "user"
        assert result[0].content == "User message"

    # ===== Tests for get_agent_messages_for_round =====

    def test_get_agent_messages_for_round(self):
        """Test agent message retrieval for specific round."""
        # Create test messages with mix of user and agent messages
        test_messages = [
            {
                "content": "User message",
                "speaker_id": "user",
                "speaker_role": "user",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Agent1 message",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Agent2 message",
                "speaker_id": "agent2",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        state = self.create_test_state(messages=test_messages)

        result = self.message_coordinator.get_agent_messages_for_round(2, state)

        # Verify only agent messages are returned
        assert len(result) == 2
        assert all(msg.speaker_role != "user" for msg in result)
        assert all(msg.speaker_id != "user" for msg in result)

    def test_get_agent_messages_for_round_specific_agent(self):
        """Test agent message retrieval for specific agent."""
        # Create test messages from multiple agents
        test_messages = [
            {
                "content": "Agent1 message",
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "content": "Agent2 message",
                "speaker_id": "agent2",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        state = self.create_test_state(messages=test_messages)

        result = self.message_coordinator.get_agent_messages_for_round(
            2, state, agent_id="agent1"
        )

        # Verify only specified agent messages are returned
        assert len(result) == 1
        assert result[0].speaker_id == "agent1"

    # ===== Tests for clear_context_cache =====

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_clear_context_cache(self, mock_get_builder):
        """Test context cache clearing."""
        state = self.create_test_state()

        # Setup mock context builder
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data()
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        # Build context to populate cache
        self.message_coordinator.assemble_agent_context("agent1", 2, state)
        assert len(self.message_coordinator._context_builders) == 1

        # Clear cache
        self.message_coordinator.clear_context_cache()
        assert len(self.message_coordinator._context_builders) == 0

    # ===== Integration Tests =====

    def test_initialization_with_round_manager(self):
        """Test MessageCoordinator initialization with RoundManager."""
        round_manager = RoundManager()
        coordinator = MessageCoordinator(round_manager)

        assert coordinator.round_manager is round_manager
        assert isinstance(coordinator._context_builders, dict)
        assert len(coordinator._context_builders) == 0

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_integration_with_round_manager(self, mock_get_builder):
        """Test integration between MessageCoordinator and RoundManager."""
        state = self.create_test_state(current_round=5)

        # Setup mock context builder
        mock_builder = Mock()
        mock_context_data = self.create_test_context_data()
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        # Test that MessageCoordinator uses RoundManager for round operations
        current_round = self.message_coordinator.round_manager.get_current_round(state)
        assert current_round == 5

        # Test context assembly uses round from RoundManager
        result_text, result_messages = self.message_coordinator.assemble_agent_context(
            agent_id="agent1", round_num=current_round, state=state
        )

        assert "Round: 5" in result_text

    def test_langchain_message_compatibility(self):
        """Test compatibility with LangChain message formats."""
        # Create test state with LangChain messages
        langchain_msg = HumanMessage(
            content="LangChain message",
            additional_kwargs={
                "speaker_id": "user",
                "speaker_role": "user",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
        )

        state = self.create_test_state(messages=[langchain_msg])

        result = self.message_coordinator.get_messages_for_round(2, state)

        # Verify LangChain messages are properly processed
        assert len(result) == 1
        assert result[0].content == "LangChain message"
        assert result[0].speaker_id == "user"

    # ===== Error Handling Tests =====

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_context_builder_error_handling(self, mock_get_builder):
        """Test error handling when context builder fails."""
        state = self.create_test_state()

        # Setup mock to raise exception
        mock_get_builder.side_effect = Exception("Context builder error")

        # Verify exception is propagated
        with pytest.raises(Exception, match="Context builder error"):
            self.message_coordinator.assemble_agent_context("agent1", 2, state)


class TestMessageCoordinatorIntegration:
    """Integration tests for MessageCoordinator with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()
        self.message_coordinator = MessageCoordinator(self.round_manager)

    def test_complete_message_flow(self):
        """Test complete message coordination flow."""
        # Initial state
        state = {
            "current_round": 1,
            "active_topic": "AI Ethics",
            "main_topic": "Future of AI",
            "messages": [],
            "round_summaries": [],
            "topic_summaries": {},
            "speaking_order": ["alice", "bob", "charlie"],
        }

        # Store user message
        user_updates = self.message_coordinator.store_user_message(
            content="I think we should focus on transparency", round_num=1, state=state
        )

        # Update state
        state["messages"].extend(user_updates["messages"])

        # Verify user message was stored correctly
        user_messages = self.message_coordinator.get_user_messages_for_round(2, state)
        assert len(user_messages) == 1
        assert "transparency" in user_messages[0].content

    @patch("src.virtual_agora.flow.message_coordinator.get_context_builder")
    def test_backward_compatibility_patterns(self, mock_get_builder):
        """Test that MessageCoordinator maintains backward compatibility."""
        # Setup mock context builder that mimics existing behavior
        mock_builder = Mock()
        mock_context_data = Mock()
        mock_context_data.formatted_context_messages = []
        mock_context_data.has_user_input = True
        mock_context_data.user_input = "Test Theme"
        mock_context_data.has_context_documents = False
        mock_context_data.context_documents = ""
        mock_context_data.has_round_summaries = False
        mock_context_data.round_summaries = []
        mock_builder.build_context.return_value = mock_context_data
        mock_get_builder.return_value = mock_builder

        state = {
            "current_round": 0,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "messages": [],
            "round_summaries": [],
            "topic_summaries": {},
            "speaking_order": ["agent1"],
        }

        # Test that context assembly produces expected format
        context_text, context_messages = (
            self.message_coordinator.assemble_agent_context(
                agent_id="agent1",
                round_num=0,
                state=state,
                agent_position=0,
                system_prompt="Test prompt",
            )
        )

        # Verify essential elements are present
        assert "Theme: Test Theme" in context_text
        assert "Current Topic: Test Topic" in context_text
        assert "Round: 0" in context_text
        assert "Your speaking position: 1/1" in context_text

    def test_message_standardization_across_formats(self):
        """Test message standardization works across different input formats."""
        # Create state with mixed message formats
        dict_message = {
            "content": "Dict format message",
            "speaker_id": "agent1",
            "speaker_role": "participant",
            "round": 2,
            "topic": "Test Topic",
            "timestamp": datetime.now().isoformat(),
        }

        langchain_message = HumanMessage(
            content="LangChain format message",
            additional_kwargs={
                "speaker_id": "user",
                "speaker_role": "user",
                "round": 2,
                "topic": "Test Topic",
                "timestamp": datetime.now().isoformat(),
            },
        )

        state = {
            "current_round": 2,
            "active_topic": "Test Topic",
            "messages": [dict_message, langchain_message],
            "round_summaries": [],
            "topic_summaries": {},
        }

        # Retrieve messages for round
        messages = self.message_coordinator.get_messages_for_round(2, state)

        # Verify both formats are standardized
        assert len(messages) == 2
        assert all(isinstance(msg, ProcessedMessage) for msg in messages)
        assert messages[0].content == "Dict format message"
        assert messages[1].content == "LangChain format message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
