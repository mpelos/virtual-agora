"""Tests for the new context assembly architecture.

This module tests the redesigned context assembly system including:
- MessageProcessor for consistent message format handling
- ContextRules for business rule enforcement
- Enhanced context builders with round-aware logic
- Context validation and debugging tools
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from virtual_agora.context.message_processor import MessageProcessor, ProcessedMessage
from virtual_agora.context.rules import ContextRules, ContextType, ContextRuleSet
from virtual_agora.context.builders import (
    DiscussionAgentContextBuilder,
    DiscussionRoundContextBuilder,
    get_context_builder,
)
from virtual_agora.context.types import ContextData
from virtual_agora.context.validation import ContextValidator, ContextDebugger
from virtual_agora.state.schema import VirtualAgoraState


class TestMessageProcessor:
    """Test the MessageProcessor class for consistent message handling."""

    def test_extract_message_info_from_dict(self):
        """Test extracting info from Virtual Agora dict format."""
        msg = {
            "speaker_id": "agent1",
            "speaker_role": "participant",
            "content": "Test message",
            "round": 1,
            "topic": "Test Topic",
            "timestamp": "2025-08-02T10:00:00",
        }

        speaker_id, speaker_role, content, metadata = (
            MessageProcessor.extract_message_info(msg)
        )

        assert speaker_id == "agent1"
        assert speaker_role == "participant"
        assert content == "Test message"
        assert metadata["round"] == 1
        assert metadata["topic"] == "Test Topic"

    def test_extract_message_info_from_langchain(self):
        """Test extracting info from LangChain BaseMessage format."""
        msg = AIMessage(
            content="Test AI message",
            name="agent1",
            additional_kwargs={
                "speaker_id": "agent1",
                "speaker_role": "participant",
                "round": 2,
                "topic": "Test Topic",
            },
        )

        speaker_id, speaker_role, content, metadata = (
            MessageProcessor.extract_message_info(msg)
        )

        assert speaker_id == "agent1"
        assert speaker_role == "participant"
        assert content == "Test AI message"
        assert metadata["round"] == 2

    def test_standardize_message(self):
        """Test message standardization."""
        msg = {
            "speaker_id": "user",
            "speaker_role": "user",
            "content": "User participation",
            "round": 1,
            "topic": "Test Topic",
            "timestamp": "2025-08-02T10:00:00",
            "participation_type": "user_turn_participation",
        }

        processed = MessageProcessor.standardize_message(msg)

        assert isinstance(processed, ProcessedMessage)
        assert processed.speaker_id == "user"
        assert processed.speaker_role == "user"
        assert processed.content == "User participation"
        assert processed.round_number == 1
        assert processed.topic == "Test Topic"
        assert processed.original_format == "dict"

    def test_to_langchain_message(self):
        """Test conversion to LangChain message format."""
        processed_msg = ProcessedMessage(
            speaker_id="agent1",
            speaker_role="participant",
            content="Test content",
            round_number=1,
            topic="Test Topic",
            timestamp=datetime.now(),
            metadata={"test": "data"},
            original_format="dict",
        )

        # Test AI message conversion
        ai_msg = MessageProcessor.to_langchain_message(processed_msg)
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "Test content"
        assert ai_msg.name == "agent1"

        # Test forced Human message conversion
        human_msg = MessageProcessor.to_langchain_message(
            processed_msg, force_role="human"
        )
        assert isinstance(human_msg, HumanMessage)
        assert human_msg.content == "Test content"

    def test_filter_messages_by_round(self):
        """Test filtering messages by round number."""
        messages = [
            {
                "speaker_id": "agent1",
                "content": "Round 1",
                "round": 1,
                "topic": "Topic A",
            },
            {
                "speaker_id": "agent2",
                "content": "Round 1",
                "round": 1,
                "topic": "Topic A",
            },
            {
                "speaker_id": "agent1",
                "content": "Round 2",
                "round": 2,
                "topic": "Topic A",
            },
            {
                "speaker_id": "user",
                "content": "Round 1",
                "round": 1,
                "topic": "Topic B",
            },
        ]

        round_1_messages = MessageProcessor.filter_messages_by_round(
            messages, 1, "Topic A"
        )
        assert len(round_1_messages) == 2
        assert all(msg.round_number == 1 for msg in round_1_messages)
        assert all(msg.topic == "Topic A" for msg in round_1_messages)

    def test_filter_user_participation_messages(self):
        """Test filtering user participation messages."""
        messages = [
            {
                "speaker_id": "user",
                "content": "User input round 1",
                "topic": "Topic A",
                "round": 1,
                "participation_type": "user_turn_participation",
            },
            {
                "speaker_id": "user",
                "content": "User input round 2",
                "topic": "Topic A",
                "round": 2,
                "participation_type": "user_turn_participation",
            },
            {
                "speaker_id": "agent1",
                "content": "Agent message",
                "topic": "Topic A",
                "round": 1,
            },
            {
                "speaker_id": "user",
                "content": "Different topic",
                "topic": "Topic B",
                "round": 1,
                "participation_type": "user_turn_participation",
            },
        ]

        user_messages = MessageProcessor.filter_user_participation_messages(
            messages, "Topic A"
        )
        assert len(user_messages) == 2
        assert all(msg.speaker_id == "user" for msg in user_messages)
        assert all(msg.topic == "Topic A" for msg in user_messages)
        assert user_messages[0].round_number == 1  # Sorted by round

    def test_create_context_messages_for_agent(self):
        """Test creating formatted context messages for an agent."""
        # Create test messages
        user_messages = [
            ProcessedMessage(
                speaker_id="user",
                speaker_role="user",
                content="User input from round 1",
                round_number=1,
                topic="Test Topic",
                timestamp=datetime.now(),
                metadata={"participation_type": "user_turn_participation"},
                original_format="dict",
            )
        ]

        round_messages = [
            ProcessedMessage(
                speaker_id="agent1",
                speaker_role="participant",
                content="Agent 1 response",
                round_number=2,
                topic="Test Topic",
                timestamp=datetime.now(),
                metadata={},
                original_format="dict",
            )
        ]

        context_messages = MessageProcessor.create_context_messages_for_agent(
            round_messages, user_messages, "agent2"
        )

        assert len(context_messages) == 2  # 1 user + 1 colleague

        # Check user message formatting
        user_msg = context_messages[0]
        assert isinstance(user_msg, HumanMessage)
        assert "[Round Moderator - Round 1]:" in user_msg.content
        assert "User input from round 1" in user_msg.content

        # Check colleague message formatting
        colleague_msg = context_messages[1]
        assert isinstance(colleague_msg, HumanMessage)
        assert "[agent1]:" in colleague_msg.content
        assert "Agent 1 response" in colleague_msg.content


class TestContextRules:
    """Test the ContextRules class for business rule enforcement."""

    def test_get_context_requirements_round_0(self):
        """Test context requirements for round 0."""
        rule_set, params = ContextRules.get_context_requirements(
            ContextType.DISCUSSION_ROUND, 0, {}
        )

        assert not rule_set.include_round_summaries  # No previous rounds
        assert not rule_set.include_user_messages  # No previous user input
        assert rule_set.include_theme  # Theme always included
        assert rule_set.include_topic  # Topic always included
        assert params["round_zero"] == True

    def test_get_context_requirements_round_1(self):
        """Test context requirements for round 1."""
        rule_set, params = ContextRules.get_context_requirements(
            ContextType.DISCUSSION_ROUND, 1, {}
        )

        assert rule_set.include_round_summaries  # Summaries from round 0
        assert rule_set.include_user_messages  # Potential user input
        assert rule_set.max_round_summaries == 5  # Limited for early round
        assert rule_set.max_user_messages == 10  # Limited for early round
        assert params["early_round"] == True

    def test_get_context_requirements_later_rounds(self):
        """Test context requirements for round 2+."""
        rule_set, params = ContextRules.get_context_requirements(
            ContextType.DISCUSSION_ROUND, 3, {}
        )

        assert rule_set.include_round_summaries  # Full summaries
        assert rule_set.include_user_messages  # Full user messages
        assert rule_set.max_round_summaries == 10  # Full limits
        assert rule_set.max_user_messages == 20  # Full limits
        assert params["established_round"] == True

    def test_validate_context_assembly_valid(self):
        """Test validation of valid context assembly."""
        user_messages = []
        round_summaries = []
        current_round_messages = []

        is_valid, issues, compliance = ContextRules.validate_context_assembly(
            ContextType.DISCUSSION_ROUND,
            0,  # Round 0
            "Test Theme",
            "Test Topic",
            round_summaries,
            user_messages,
            current_round_messages,
        )

        assert is_valid
        assert len(issues) == 0
        assert compliance > 0.8

    def test_validate_context_assembly_missing_theme(self):
        """Test validation with missing theme."""
        is_valid, issues, compliance = ContextRules.validate_context_assembly(
            ContextType.DISCUSSION_ROUND,
            1,  # Round 1 requires theme
            None,  # Missing theme
            "Test Topic",
            [],
            [],
            [],
        )

        assert not is_valid
        assert "Missing required theme" in issues
        assert compliance < 1.0

    def test_enforce_message_limits(self):
        """Test message limit enforcement."""
        messages = [
            ProcessedMessage(
                speaker_id=f"agent{i}",
                speaker_role="participant",
                content=f"Message {i}",
                round_number=1,
                topic="Test",
                timestamp=datetime.now(),
                metadata={},
                original_format="dict",
            )
            for i in range(10)
        ]

        # Test recent strategy
        limited = ContextRules.enforce_message_limits(messages, 5, "recent")
        assert len(limited) == 5
        assert limited[0].speaker_id == "agent5"  # Last 5 messages
        assert limited[-1].speaker_id == "agent9"

    def test_get_round_context_requirements(self):
        """Test round-specific context requirements."""
        requirements = ContextRules.get_round_context_requirements(2, 1, 3)

        assert requirements["include_previous_speakers"] == True  # Not first
        assert requirements["max_previous_speakers"] == 1
        assert requirements["position_context"] == "2/3"
        assert requirements["context_emphasis"] == "continued_deliberation"


class TestDiscussionRoundContextBuilder:
    """Test the enhanced DiscussionRoundContextBuilder."""

    def create_mock_state(self) -> Dict[str, Any]:
        """Create a mock state for testing."""
        return {
            "main_topic": "Test Theme",
            "active_topic": "Current Topic",
            "current_round": 2,
            "messages": [
                {
                    "speaker_id": "user",
                    "speaker_role": "user",
                    "content": "User participated in round 1",
                    "round": 1,
                    "topic": "Current Topic",
                    "participation_type": "user_turn_participation",
                    "timestamp": "2025-08-02T10:00:00",
                },
                {
                    "speaker_id": "agent1",
                    "speaker_role": "participant",
                    "content": "Agent 1 speaks first in round 2",
                    "round": 2,
                    "topic": "Current Topic",
                    "timestamp": "2025-08-02T10:30:00",
                },
            ],
            "round_summaries": [
                {
                    "round_number": 1,
                    "topic": "Current Topic",
                    "summary_text": "Round 1 summary",
                }
            ],
        }

    def test_build_context_round_2_agent_2(self):
        """Test building context for agent 2 in round 2."""
        state = self.create_mock_state()
        builder = DiscussionRoundContextBuilder()

        with patch(
            "virtual_agora.context.repository.ContextRepository.get_context_documents"
        ) as mock_docs:
            mock_docs.return_value = "Test context documents"

            context_data = builder.build_context(
                state=state,
                system_prompt="Test system prompt",
                agent_id="agent2",
                current_round=2,
                agent_position=1,  # Second to speak
                speaking_order=["agent1", "agent2", "agent3"],
                topic="Current Topic",
            )

        # Verify context data structure
        assert context_data.system_prompt == "Test system prompt"
        assert context_data.has_user_input  # Theme included
        assert context_data.has_round_summaries  # Previous round summaries
        assert hasattr(context_data, "formatted_context_messages")

        # Check formatted context messages
        formatted_messages = context_data.formatted_context_messages
        assert len(formatted_messages) >= 1  # At least user message from round 1

        # Verify metadata
        assert context_data.metadata["agent_id"] == "agent2"
        assert context_data.metadata["current_round"] == 2
        assert context_data.metadata["agent_position"] == 1
        assert "compliance_score" in context_data.metadata

    def test_build_context_round_0_first_agent(self):
        """Test building context for first agent in round 0."""
        state = {
            "main_topic": "Test Theme",
            "active_topic": "Current Topic",
            "current_round": 0,
            "messages": [],
            "round_summaries": [],
        }

        builder = DiscussionRoundContextBuilder()

        context_data = builder.build_context(
            state=state,
            system_prompt="Test system prompt",
            agent_id="agent1",
            current_round=0,
            agent_position=0,  # First to speak
            speaking_order=["agent1", "agent2"],
            topic="Current Topic",
        )

        # Round 0, first agent should have minimal context
        assert context_data.has_user_input  # Theme
        assert not context_data.has_round_summaries  # No previous rounds
        assert not context_data.has_user_participation_messages  # No user input yet

        # Should have empty formatted context messages
        formatted_messages = context_data.formatted_context_messages or []
        assert len(formatted_messages) == 0

    def test_build_context_validation_error_handling(self):
        """Test error handling when required parameters are missing."""
        builder = DiscussionRoundContextBuilder()
        state = self.create_mock_state()

        # Test missing agent_id
        with pytest.raises(ValueError, match="agent_id is required"):
            builder.build_context(
                state=state,
                system_prompt="Test",
                # Missing agent_id
                current_round=1,
                agent_position=0,
            )

        # Test missing current_round
        with pytest.raises(ValueError, match="current_round is required"):
            builder.build_context(
                state=state,
                system_prompt="Test",
                agent_id="agent1",
                # Missing current_round
                agent_position=0,
            )


class TestContextValidator:
    """Test the ContextValidator for validation and debugging."""

    def create_test_context_data(self) -> ContextData:
        """Create test context data."""
        context_data = ContextData(system_prompt="Test system prompt")
        context_data.user_input = "Test theme"
        context_data.round_summaries = [{"round_number": 1, "summary_text": "Summary"}]
        context_data.user_participation_messages = []
        context_data.current_round_messages = []
        context_data.metadata = {
            "agent_id": "test_agent",
            "current_round": 2,
            "compliance_score": 0.95,
        }
        return context_data

    def test_validate_context_data_valid(self):
        """Test validation of valid context data."""
        context_data = self.create_test_context_data()

        result = ContextValidator.validate_context_data(
            context_data, "test_agent", 2, ContextType.DISCUSSION_ROUND
        )

        assert result.is_valid
        assert result.compliance_score > 0.8
        assert len(result.validation_issues) == 0
        assert result.agent_id == "test_agent"
        assert result.round_number == 2

    def test_validate_context_data_missing_system_prompt(self):
        """Test validation with missing system prompt."""
        context_data = self.create_test_context_data()
        context_data.system_prompt = ""  # Missing

        result = ContextValidator.validate_context_data(
            context_data, "test_agent", 2, ContextType.DISCUSSION_ROUND
        )

        assert not result.is_valid
        assert "Missing system prompt" in result.validation_issues
        assert result.compliance_score < 1.0

    def test_debug_message_processing(self):
        """Test message processing debugging."""
        state = {
            "messages": [
                {
                    "speaker_id": "agent1",
                    "speaker_role": "participant",
                    "content": "Test",
                    "round": 1,
                    "topic": "Topic A",
                },
                {
                    "speaker_id": "user",
                    "speaker_role": "user",
                    "content": "User input",
                    "round": 1,
                    "topic": "Topic A",
                    "participation_type": "user_turn_participation",
                },
                {
                    "speaker_id": "agent2",
                    "speaker_role": "participant",
                    "content": "Test 2",
                    "round": 2,
                    "topic": "Topic A",
                },
            ],
            "round_summaries": [{"round_number": 1, "summary_text": "Summary"}],
            "main_topic": "Main Theme",
        }

        debug_info = ContextValidator.debug_message_processing(
            state, "agent1", 2, "Topic A"
        )

        assert debug_info.agent_id == "agent1"
        assert debug_info.round_number == 2
        assert debug_info.topic == "Topic A"
        assert debug_info.message_counts["total"] == 3
        assert debug_info.message_counts["user_participation"] == 1
        assert debug_info.message_counts["agent_messages"] == 2
        assert debug_info.rule_compliance["theme_available"] == True

    def test_validate_message_consistency(self):
        """Test message consistency validation."""
        messages = [
            {"speaker_id": "agent1", "content": "Test", "round": 1},
            AIMessage(
                content="AI message",
                additional_kwargs={"speaker_id": "agent2", "round": 1},
            ),
            {"speaker_id": "user", "content": "User", "round": 1},
        ]

        is_valid, issues, analysis = ContextValidator.validate_message_consistency(
            messages
        )

        assert analysis["total_messages"] == 3
        assert analysis["format_distribution"]["dict"] == 2
        assert analysis["format_distribution"]["langchain"] == 1
        assert "agent1" in analysis["speaker_roles"]
        assert "user" in analysis["speaker_roles"]


class TestContextDebugger:
    """Test the ContextDebugger for tracing context assembly."""

    def test_trace_context_assembly(self):
        """Test tracing the complete context assembly process."""
        state = {
            "main_topic": "Test Theme",
            "active_topic": "Test Topic",
            "current_round": 1,
            "messages": [],
        }

        trace = ContextDebugger.trace_context_assembly(
            state=state,
            agent_id="test_agent",
            current_round=1,
            agent_position=0,
            speaking_order=["test_agent"],
            topic="Test Topic",
        )

        assert trace["agent_id"] == "test_agent"
        assert "timestamp" in trace
        assert len(trace["steps"]) >= 3  # At least 3 steps in process
        assert trace["total_time_ms"] > 0

        # Check step structure
        for step in trace["steps"]:
            assert "step" in step
            assert "name" in step
            assert "time_ms" in step
            assert "details" in step


class TestContextBuilderIntegration:
    """Test integration between different context builders."""

    def test_get_context_builder_factory(self):
        """Test the context builder factory function."""
        discussion_builder = get_context_builder("discussion")
        assert isinstance(discussion_builder, DiscussionAgentContextBuilder)

        round_builder = get_context_builder("discussion_round")
        assert isinstance(round_builder, DiscussionRoundContextBuilder)

        # Test invalid builder type
        with pytest.raises(KeyError):
            get_context_builder("invalid_type")

    def test_context_builder_consistency(self):
        """Test that different builders provide consistent interfaces."""
        state = {
            "main_topic": "Test Theme",
            "active_topic": "Test Topic",
            "messages": [],
            "round_summaries": [],
        }

        # Test both builders with same parameters
        discussion_builder = get_context_builder("discussion")
        round_builder = get_context_builder("discussion_round")

        # Both should accept the same core parameters
        common_params = {
            "state": state,
            "system_prompt": "Test prompt",
            "agent_id": "test_agent",
            "current_round": 1,
            "topic": "Test Topic",
        }

        discussion_context = discussion_builder.build_context(**common_params)

        # Round builder requires additional parameters
        round_params = {
            **common_params,
            "agent_position": 0,
            "speaking_order": ["test_agent"],
        }
        round_context = round_builder.build_context(**round_params)

        # Both should produce ContextData objects
        assert isinstance(discussion_context, ContextData)
        assert isinstance(round_context, ContextData)

        # Round builder should provide formatted messages
        assert hasattr(round_context, "formatted_context_messages")
        assert round_context.metadata["context_type"] == "discussion_round"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
