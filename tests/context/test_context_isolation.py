"""Integration tests for context isolation system."""

import pytest
from unittest.mock import Mock, patch

from virtual_agora.agents.report_writer_agent import ReportWriterAgent
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.context.builders import (
    ReportWriterContextBuilder,
    DiscussionAgentContextBuilder,
)
from virtual_agora.context.types import ContextData
from virtual_agora.state.schema import VirtualAgoraState


class TestContextIsolation:
    """Test that context isolation is working correctly."""

    def test_report_writer_does_not_receive_context_files(self):
        """Test that ReportWriterAgent does not receive context directory files."""

        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Test response")

        # Create a ReportWriterAgent
        report_agent = ReportWriterAgent(agent_id="test_report_writer", llm=mock_llm)

        # Create a mock state with context data
        mock_state = {
            "main_topic": "Test Theme",
            "active_topic": "Test Topic",
            "round_summaries": [
                {"topic": "Test Topic", "summary_text": "Round 1 summary"},
                {"topic": "Test Topic", "summary_text": "Round 2 summary"},
            ],
            "topic_reports": {
                "Topic A": "Report for Topic A",
                "Topic B": "Report for Topic B",
            },
        }

        # Mock the context documents function to return some content
        with patch(
            "virtual_agora.context.repository.load_context_documents"
        ) as mock_load_context:
            mock_load_context.return_value = "=== domain_knowledge.txt ===\nThis is domain knowledge from context directory"

            # Use the new context-aware message formatting
            messages = report_agent.format_messages_with_context(
                state=mock_state,
                prompt="Generate a topic report",
                report_type="topic",
                topic="Test Topic",
            )

            # Verify that context files were NOT loaded for report writer
            mock_load_context.assert_not_called()

            # Check that the messages don't contain context directory content
            message_content = " ".join(
                [msg.content for msg in messages if hasattr(msg, "content")]
            )
            assert "domain_knowledge.txt" not in message_content
            assert (
                "This is domain knowledge from context directory" not in message_content
            )

            # But should contain filtered discussion data
            assert (
                "Round 1 summary" in message_content
                or "Round 2 summary" in message_content
            )

    def test_discussion_agent_receives_full_context(self):
        """Test that discussion agents still receive full context including context files."""

        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Test response")

        # Create a discussion agent (regular LLMAgent for comparison)
        discussion_agent = LLMAgent(
            agent_id="test_discussion_agent",
            llm=mock_llm,
            system_prompt="You are a discussion participant",
        )

        # Create a mock state
        mock_state = {
            "main_topic": "Test Theme",
            "active_topic": "Test Topic",
            "messages": [
                {"speaker_id": "agent1", "content": "First message"},
                {"speaker_id": "agent2", "content": "Second message"},
            ],
        }

        # Mock the context documents function to return some content
        with patch(
            "virtual_agora.agents.llm_agent.load_context_documents"
        ) as mock_load_context:
            mock_load_context.return_value = "=== domain_knowledge.txt ===\nThis is domain knowledge from context directory"

            # Use the standard format_messages method (which loads context)
            messages = discussion_agent.format_messages(
                prompt="Participate in discussion",
                context_messages=None,
                include_system=True,
            )

            # Verify that context files WERE loaded for discussion agent
            mock_load_context.assert_called_once()

            # Check that the messages contain context directory content
            message_content = " ".join(
                [msg.content for msg in messages if hasattr(msg, "content")]
            )
            assert "domain_knowledge.txt" in message_content
            assert "This is domain knowledge from context directory" in message_content

    def test_context_builder_strategies(self):
        """Test that different context builders provide appropriate context."""

        mock_state = {
            "main_topic": "Test Theme",
            "active_topic": "Test Topic",
            "messages": [{"speaker_id": "agent1", "content": "Message"}],
            "round_summaries": [{"topic": "Test Topic", "summary_text": "Summary"}],
            "topic_summaries": {"Topic A": "Report A"},
        }

        # Test ReportWriterContextBuilder for topic reports
        report_builder = ReportWriterContextBuilder()

        with patch(
            "virtual_agora.context.repository.load_context_documents"
        ) as mock_load_context:
            mock_load_context.return_value = "Context files content"

            context_data = report_builder.build_context(
                state=mock_state,
                system_prompt="System prompt",
                report_type="topic",
                topic="Test Topic",
            )

            # Report writer should NOT get context documents
            assert not context_data.has_context_documents
            assert context_data.context_documents is None

            # But should get filtered discussion data
            assert context_data.has_round_summaries
            assert len(context_data.round_summaries) > 0

            # Context documents should not have been loaded
            mock_load_context.assert_not_called()

        # Test ReportWriterContextBuilder for session reports
        session_context_data = report_builder.build_context(
            state=mock_state, system_prompt="System prompt", report_type="session"
        )

        # Session reports should only get topic reports
        assert not session_context_data.has_context_documents
        assert session_context_data.has_topic_reports
        assert len(session_context_data.topic_reports) > 0

        # Test DiscussionAgentContextBuilder
        discussion_builder = DiscussionAgentContextBuilder()

        with patch(
            "virtual_agora.context.repository.load_context_documents"
        ) as mock_load_context:
            mock_load_context.return_value = "Context files content"

            discussion_context_data = discussion_builder.build_context(
                state=mock_state, system_prompt="System prompt", topic="Test Topic"
            )

            # Discussion agent should get full context including context documents
            assert discussion_context_data.has_context_documents
            assert discussion_context_data.context_documents == "Context files content"
            assert discussion_context_data.has_user_input
            assert discussion_context_data.user_input == "Test Theme"

            # Context documents should have been loaded
            mock_load_context.assert_called_once()

    def test_context_data_structure(self):
        """Test the ContextData structure and its properties."""

        # Test empty context data
        empty_context = ContextData(system_prompt="Test prompt")

        assert not empty_context.has_context_documents
        assert not empty_context.has_user_input
        assert not empty_context.has_topic_messages
        assert not empty_context.has_round_summaries
        assert not empty_context.has_topic_reports

        # Test populated context data
        populated_context = ContextData(
            system_prompt="Test prompt",
            context_documents="Context content",
            user_input="User theme",
            topic_messages=[{"speaker_id": "agent1", "content": "Message"}],
            round_summaries=[{"summary_text": "Summary"}],
            topic_reports={"Topic A": "Report A"},
        )

        assert populated_context.has_context_documents
        assert populated_context.has_user_input
        assert populated_context.has_topic_messages
        assert populated_context.has_round_summaries
        assert populated_context.has_topic_reports
