"""Tests for Moderator agent implementation."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError

from virtual_agora.agents.moderator import (
    ModeratorAgent,
    AgendaResponse,
    create_moderator_agent,
    create_gemini_moderator,
    AGENDA_SCHEMA,
)
from virtual_agora.state.schema import Message


class TestModeratorAgent:
    """Test ModeratorAgent core functionality."""

    def setup_method(self):
        """Set up test method."""
        # Create mock LLM instance
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        # Create moderator agent with error handling disabled to avoid LLM wrapping
        self.moderator = ModeratorAgent(
            agent_id="test-moderator",
            llm=self.mock_llm,
            enable_error_handling=False,
        )

    def test_moderator_initialization(self):
        """Test moderator initialization."""
        assert self.moderator.agent_id == "test-moderator"
        assert self.moderator.llm == self.mock_llm
        assert self.moderator.role == "moderator"
        assert self.moderator.model == "gemini-2.5-pro"
        assert self.moderator.provider == "google"
        assert isinstance(self.moderator.created_at, datetime)
        assert self.moderator.PROMPT_VERSION == "1.0"

        # Test prompt contains key moderator elements
        assert "impartial Moderator" in self.moderator.system_prompt
        assert "NOT to have an opinion" in self.moderator.system_prompt
        assert "FACILITATION mode" in self.moderator.system_prompt

    def test_moderator_initialization_with_custom_prompt(self):
        """Test moderator initialization with custom system prompt."""
        custom_prompt = "Custom moderator prompt"
        moderator = ModeratorAgent(
            agent_id="custom-mod", llm=self.mock_llm, system_prompt=custom_prompt
        )

        assert moderator.system_prompt == custom_prompt
        assert moderator._custom_system_prompt == custom_prompt

        # Mode switching tests removed in v1.3
        """Test switching between operational modes."""
        # Test switching to synthesis mode
        self.moderator.set_mode("synthesis")
        assert self.moderator.current_mode == "synthesis"
        assert "SYNTHESIS mode" in self.moderator.system_prompt
        assert "agent-agnostic summaries" in self.moderator.system_prompt

        # Test switching to writer mode
        self.moderator.set_mode("writer")
        assert self.moderator.current_mode == "writer"
        assert "WRITER mode" in self.moderator.system_prompt
        assert "The Writer" in self.moderator.system_prompt

        # Test switching back to facilitation
        self.moderator.set_mode("facilitation")
        assert self.moderator.current_mode == "facilitation"
        assert "FACILITATION mode" in self.moderator.system_prompt

        """Test that custom prompts are preserved during mode switching."""
        custom_prompt = "Custom moderator prompt"
        moderator = ModeratorAgent(
            agent_id="custom-mod", llm=self.mock_llm, system_prompt=custom_prompt
        )

        # Mode switching should not change custom prompt
        moderator.set_mode("synthesis")
        assert moderator.system_prompt == custom_prompt
        assert moderator.current_mode == "synthesis"

        """Test switching to the same mode (no-op)."""
        original_prompt = self.moderator.system_prompt
        self.moderator.set_mode("facilitation")  # Same mode
        assert self.moderator.current_mode == "facilitation"
        assert self.moderator.system_prompt == original_prompt

        """Test getting current mode."""
        assert self.moderator.get_current_mode() == "facilitation"

        self.moderator.set_mode("synthesis")
        assert self.moderator.get_current_mode() == "synthesis"

        """Test that each mode has appropriate prompt content."""
        # Facilitation mode
        self.moderator.set_mode("facilitation")
        prompt = self.moderator.system_prompt
        assert "facilitate the creation of a discussion agenda" in prompt.lower()
        assert "conduct polls" in prompt.lower()
        assert "final considerations" in prompt.lower()

        # Synthesis mode
        self.moderator.set_mode("synthesis")
        prompt = self.moderator.system_prompt
        assert "analyze natural language votes" in prompt.lower()
        assert "agent-agnostic summaries" in prompt.lower()
        assert "never include agent attributions" in prompt.lower()

        # Writer mode
        self.moderator.set_mode("writer")
        prompt = self.moderator.system_prompt
        assert "the writer" in prompt.lower()
        assert "structured final reports" in prompt.lower()
        assert "professional, analytical tone" in prompt.lower()

    def test_conversation_context_management(self):
        """Test conversation context management."""
        # Test setting and getting context
        self.moderator.update_conversation_context("current_topic", "AI Ethics")
        self.moderator.update_conversation_context("round_number", 3)

        assert self.moderator.get_conversation_context("current_topic") == "AI Ethics"
        assert self.moderator.get_conversation_context("round_number") == 3
        assert (
            self.moderator.get_conversation_context("nonexistent", "default")
            == "default"
        )

        # Test clearing context
        self.moderator.clear_conversation_context()
        assert self.moderator.get_conversation_context("current_topic") is None

    def test_neutrality_validation(self):
        """Test neutrality validation of responses."""
        # Test neutral responses
        neutral_responses = [
            "The discussion will proceed to the next topic.",
            "Based on the votes, the agenda is: Topic A, Topic B, Topic C.",
            "This concludes the summary of the current round.",
            "The poll results indicate a majority vote to continue.",
        ]

        for response in neutral_responses:
            assert self.moderator.validate_neutrality(response) is True

        # Test non-neutral responses (containing opinions)
        opinion_responses = [
            "I think this is a great idea.",
            "I believe we should focus on this topic.",
            "In my opinion, the discussion is going well.",
            "Personally, I would recommend another approach.",
            "I agree with the agent's point.",
            "That's clearly wrong.",
            "Obviously, this is the best solution.",
        ]

        for response in opinion_responses:
            assert self.moderator.validate_neutrality(response) is False

    def test_generate_response_with_neutrality_validation(self):
        """Test that generate_response validates neutrality."""
        # Mock neutral response
        mock_response = Mock()
        mock_response.content = "The discussion will proceed as planned."
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        response = self.moderator.generate_response("Test prompt")
        assert response == "The discussion will proceed as planned."

        # Mock opinion response (should still return but log warning)
        mock_response.content = "I think this is a good idea."
        response = self.moderator.generate_response("Test prompt")
        assert response == "I think this is a good idea."
        # Note: In practice, this would log a warning

    def test_generate_response_skip_neutrality_validation(self):
        """Test generate_response with neutrality validation disabled."""
        mock_response = Mock()
        mock_response.content = "I believe this is correct."
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        response = self.moderator.generate_response(
            "Test prompt", validate_neutrality=False
        )
        assert response == "I believe this is correct."


class TestModeratorJSONGeneration:
    """Test JSON generation and validation functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="json-moderator",
            llm=self.mock_llm,
            mode="synthesis",
            enable_error_handling=False,
        )

    def test_generate_json_response_valid(self):
        """Test generating valid JSON response."""
        # Mock valid JSON response
        json_response = '{"proposed_agenda": ["Topic A", "Topic B", "Topic C"]}'
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.generate_json_response(
            "Generate agenda", expected_schema=AGENDA_SCHEMA
        )

        expected = {"proposed_agenda": ["Topic A", "Topic B", "Topic C"]}
        assert result == expected

    def test_generate_json_response_with_extra_text(self):
        """Test JSON extraction from response with extra text."""
        # Mock response with extra text around JSON
        response_with_text = (
            "Here is the agenda based on the votes:\n"
            '{"proposed_agenda": ["AI Ethics", "Climate Change"]}\n'
            "This concludes the agenda synthesis."
        )
        mock_response = Mock()
        mock_response.content = response_with_text
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.generate_json_response(
            "Generate agenda", model_class=AgendaResponse
        )

        expected = {"proposed_agenda": ["AI Ethics", "Climate Change"]}
        assert result == expected

    def test_generate_json_response_invalid_json(self):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(ValueError, match="No JSON structure found"):
            self.moderator.generate_json_response("Generate agenda")

    def test_generate_json_response_malformed_json(self):
        """Test handling of malformed JSON response."""
        mock_response = Mock()
        mock_response.content = (
            # Missing closing bracket
            '{"proposed_agenda": ["Topic A", "Topic B"'
        )
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(
            ValueError, match="Invalid JSON response|No JSON structure found"
        ):
            self.moderator.generate_json_response("Generate agenda")

    def test_generate_json_response_schema_validation_failure(self):
        """Test JSON schema validation failure."""
        # Valid JSON but missing required field
        json_response = '{"wrong_field": ["Topic A", "Topic B"]}'
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(
            ValueError, match="Required field 'proposed_agenda' missing"
        ):
            self.moderator.generate_json_response(
                "Generate agenda", expected_schema=AGENDA_SCHEMA
            )

    def test_generate_json_response_pydantic_validation(self):
        """Test Pydantic model validation."""
        # Valid JSON
        json_response = '{"proposed_agenda": ["Topic A", "Topic B"]}'
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.generate_json_response(
            "Generate agenda", model_class=AgendaResponse
        )

        assert result == {"proposed_agenda": ["Topic A", "Topic B"]}

    def test_generate_json_response_pydantic_validation_failure(self):
        """Test Pydantic validation failure."""
        # Invalid JSON for the model
        json_response = '{"proposed_agenda": "should be array not string"}'
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(ValidationError):
            self.moderator.generate_json_response(
                "Generate agenda", model_class=AgendaResponse
            )

    def test_parse_agenda_json(self):
        """Test agenda parsing functionality."""
        # Mock valid agenda response
        json_response = (
            '{"proposed_agenda": ["AI Safety", "Climate Tech", "Future of Work"]}'
        )
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        agenda = self.moderator.parse_agenda_json(
            "Based on the votes, synthesize the agenda"
        )

        expected = ["AI Safety", "Climate Tech", "Future of Work"]
        assert agenda == expected

    def test_parse_agenda_json_failure(self):
        """Test agenda parsing failure handling."""
        mock_response = Mock()
        mock_response.content = "Invalid response"
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(ValueError, match="Failed to parse agenda"):
            self.moderator.parse_agenda_json("Generate agenda")


class TestModeratorReportGeneration:
    """Test report generation functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="report-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False,
        )

    def test_generate_report_structure(self):
        """Test report structure generation."""
        # Mock JSON response for report structure
        json_response = """
        {
            "report_sections": [
                "Executive Summary",
                "Key Insights",
                "Recommendations",
                "Conclusion"
            ]
        }
        """
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        # Test topic summaries
        topic_summaries = [
            "Discussion on AI safety revealed concerns about alignment.",
            "Climate technology discussion focused on renewable energy solutions.",
        ]

        # Should switch to writer mode temporarily
        original_mode = self.moderator.current_mode
        sections = self.moderator.generate_report_structure(topic_summaries)

        expected_sections = [
            "Executive Summary",
            "Key Insights",
            "Recommendations",
            "Conclusion",
        ]
        assert sections == expected_sections

        # Should restore original mode
        assert self.moderator.current_mode == original_mode

    def test_generate_report_structure_failure(self):
        """Test report structure generation failure."""
        mock_response = Mock()
        mock_response.content = "Invalid response"
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        topic_summaries = ["Test summary"]

        with pytest.raises(ValueError, match="Failed to generate report structure"):
            self.moderator.generate_report_structure(topic_summaries)

    def test_generate_section_content(self):
        """Test section content generation."""
        mock_response = Mock()
        mock_response.content = (
            "This executive summary synthesizes the key insights from our "
            "Virtual Agora session on AI safety and climate technology..."
        )
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        topic_summaries = [
            "AI safety discussion covered alignment challenges.",
            "Climate tech discussion emphasized renewable solutions.",
        ]

        original_mode = self.moderator.current_mode
        content = self.moderator.generate_section_content(
            "Executive Summary",
            topic_summaries,
            "High-level overview of session insights",
        )

        assert "executive summary synthesizes" in content.lower()
        assert len(content) > 50  # Should be substantial content

        # Should restore original mode
        assert self.moderator.current_mode == original_mode

    def test_generate_section_content_mode_handling(self):
        """Test that section content generation properly handles mode switching."""
        # Start in synthesis mode
        self.moderator.set_mode("synthesis")
        original_mode = self.moderator.current_mode

        mock_response = Mock()
        mock_response.content = "Section content"
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        # Generate content (should temporarily switch to writer mode)
        self.moderator.generate_section_content("Test Section", ["Test summary"])

        # Should restore original mode
        assert self.moderator.current_mode == original_mode


class TestModeratorLangGraphIntegration:
    """Test ModeratorAgent's LangGraph integration."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="langgraph-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False,
        )

    def test_moderator_as_langgraph_node(self):
        """Test moderator can be used as a LangGraph node."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "I will facilitate the agenda creation."
        self.mock_llm.invoke.return_value = mock_response

        # Create state with messages
        state = {
            "messages": [HumanMessage(content="Please facilitate the discussion.")]
        }

        # Call moderator as node
        result = self.moderator(state)

        # Check result
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "I will facilitate the agenda creation."
        assert result["messages"][0].name == "langgraph-moderator"

    def test_moderator_with_virtual_agora_state(self):
        """Test moderator with VirtualAgoraState."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "Please propose 3-5 discussion topics for our session."
        self.mock_llm.invoke.return_value = mock_response

        # Create VirtualAgoraState for agenda phase
        state = {
            "session_id": "test-session",
            "current_phase": 1,  # Agenda setting
            "agents": {"langgraph-moderator": {"id": "langgraph-moderator"}},
            "messages": [],
            "active_topic": None,
            "messages_by_agent": {},
            "messages_by_topic": {},
        }

        # Call moderator
        result = self.moderator(state)

        # Check result has appropriate prompt for phase 1
        assert "messages" in result
        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        assert "propose 3-5 discussion topics" in message.content

    def test_moderator_phase_specific_behavior(self):
        """Test moderator generates phase-appropriate responses."""
        mock_response = Mock()
        self.mock_llm.invoke.return_value = mock_response

        # Test different phases
        test_cases = [
            (1, None, "Please propose 3-5 discussion topics"),
            (2, "AI Ethics", "Please share your thoughts on: AI Ethics"),
            (
                3,
                "Climate Change",
                "Should we conclude our discussion on 'Climate Change'?",
            ),
        ]

        for phase, topic, expected_prompt in test_cases:
            mock_response.content = f"Phase {phase} response"

            state = {
                "current_phase": phase,
                "active_topic": topic,
                "agents": {},
                "messages": [],
                "messages_by_agent": {},
                "messages_by_topic": {},
            }

            result = self.moderator(state)

            # Verify the correct prompt was used
            call_args = self.mock_llm.invoke.call_args[0][0]
            human_messages = [msg for msg in call_args if isinstance(msg, HumanMessage)]
            assert len(human_messages) > 0
            assert expected_prompt in human_messages[-1].content

    @pytest.mark.asyncio
    async def test_moderator_async_call(self):
        """Test moderator async call functionality."""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "Async facilitation response"
        self.mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Create state
        state = {"messages": [HumanMessage(content="Facilitate async discussion")]}

        # Call async
        result = await self.moderator.__acall__(state)

        # Verify
        assert "messages" in result
        assert result["messages"][0].content == "Async facilitation response"
        assert result["messages"][0].name == "langgraph-moderator"


class TestModeratorFactoryFunctions:
    """Test factory functions for creating moderators."""

    def test_create_moderator_agent(self):
        """Test create_moderator_agent factory function."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm.model_name = "gemini-2.5-pro"

        moderator = create_moderator_agent(
            agent_id="factory-moderator",
            llm=mock_llm,
            mode="synthesis",
            enable_error_handling=False,
        )

        assert isinstance(moderator, ModeratorAgent)
        assert moderator.agent_id == "factory-moderator"
        assert moderator.llm == mock_llm
        assert moderator.current_mode == "synthesis"
        assert moderator.role == "moderator"

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_create_gemini_moderator(self, mock_chat_google):
        """Test create_gemini_moderator factory function."""
        # Mock the ChatGoogleGenerativeAI class
        mock_llm_instance = Mock()
        mock_llm_instance.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm_instance.model_name = "gemini-2.5-pro"
        mock_chat_google.return_value = mock_llm_instance

        moderator = create_gemini_moderator(
            agent_id="gemini-moderator",
            mode="writer",
            temperature=0.7,
            enable_error_handling=False,
        )

        # Verify ChatGoogleGenerativeAI was called with correct parameters
        mock_chat_google.assert_called_once_with(
            model="gemini-2.5-pro", temperature=0.7
        )

        # Verify moderator was created correctly
        assert isinstance(moderator, ModeratorAgent)
        assert moderator.agent_id == "gemini-moderator"
        assert moderator.current_mode == "writer"
        assert moderator.llm == mock_llm_instance

    def test_create_gemini_moderator_import_error(self):
        """Test create_gemini_moderator handles import error."""
        with patch.dict("sys.modules", {"langchain_google_genai": None}):
            with pytest.raises(
                ImportError, match="google-generativeai package required"
            ):
                create_gemini_moderator()


class TestModeratorJSONSchemaValidation:
    """Test JSON schema validation functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.moderator = ModeratorAgent(
            "test", self.mock_llm, enable_error_handling=False
        )

    def test_validate_json_schema_success(self):
        """Test successful JSON schema validation."""
        data = {"proposed_agenda": ["Topic A", "Topic B"]}
        schema = AGENDA_SCHEMA

        # Should not raise exception
        self.moderator._validate_json_schema(data, schema)

    def test_validate_json_schema_missing_required_field(self):
        """Test validation failure for missing required field."""
        data = {"wrong_field": ["Topic A", "Topic B"]}
        schema = AGENDA_SCHEMA

        with pytest.raises(
            ValueError, match="Required field 'proposed_agenda' missing"
        ):
            self.moderator._validate_json_schema(data, schema)

    def test_validate_json_schema_wrong_type(self):
        """Test validation failure for wrong field type."""
        # String instead of array
        data = {"proposed_agenda": "Topic A"}
        schema = AGENDA_SCHEMA

        with pytest.raises(
            ValueError, match="Field 'proposed_agenda' should be an array"
        ):
            self.moderator._validate_json_schema(data, schema)

    def test_validate_json_schema_complex(self):
        """Test validation with report structure schema."""
        data = {"report_sections": ["Section 1", "Section 2"]}
        schema = REPORT_STRUCTURE_SCHEMA

        # Should not raise exception
        self.moderator._validate_json_schema(data, schema)


class TestModeratorErrorHandling:
    """Test error handling in moderator functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            "error-test", self.mock_llm, enable_error_handling=False
        )

    def test_llm_error_in_json_generation(self):
        """Test handling of LLM errors during JSON generation."""
        self.mock_llm.invoke.side_effect = Exception("LLM connection error")

        with pytest.raises(Exception, match="LLM connection error"):
            self.moderator.generate_json_response("Generate agenda")

    def test_json_parsing_error_logging(self):
        """Test that JSON parsing errors are properly logged."""
        mock_response = Mock()
        mock_response.content = "Not JSON at all!"
        self.mock_llm.invoke.return_value = mock_response

        with pytest.raises(ValueError):
            self.moderator.generate_json_response("Generate agenda")

        # Error should be logged (would need to capture logs to verify)

    def test_agenda_parsing_error_handling(self):
        """Test agenda parsing error handling."""
        mock_response = Mock()
        mock_response.content = "Invalid response format"
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(ValueError, match="Failed to parse agenda"):
            self.moderator.parse_agenda_json("Parse this agenda")

    def test_report_structure_error_handling(self):
        """Test report structure generation error handling."""
        self.mock_llm.invoke.side_effect = Exception("Generation failed")

        with pytest.raises(ValueError, match="Failed to generate report structure"):
            self.moderator.generate_report_structure(["Test summary"])


class TestModeratorPydanticModels:
    """Test Pydantic model definitions."""

    def test_agenda_response_model(self):
        """Test AgendaResponse Pydantic model."""
        # Valid data
        valid_data = {"proposed_agenda": ["Topic A", "Topic B", "Topic C"]}
        agenda = AgendaResponse(**valid_data)
        assert agenda.proposed_agenda == ["Topic A", "Topic B", "Topic C"]
        assert agenda.model_dump() == valid_data

        # Invalid data - wrong type
        with pytest.raises(ValidationError):
            AgendaResponse(proposed_agenda="should be list")

        # Invalid data - missing field
        with pytest.raises(ValidationError):
            AgendaResponse(wrong_field=["Topic A"])

    def test_report_structure_model(self):
        """Test ReportStructure Pydantic model."""
        # Valid data
        valid_data = {"report_sections": ["Introduction", "Analysis", "Conclusion"]}
        structure = ReportStructure(**valid_data)
        assert structure.report_sections == ["Introduction", "Analysis", "Conclusion"]
        assert structure.model_dump() == valid_data

        # Invalid data
        with pytest.raises(ValidationError):
            ReportStructure(report_sections="should be list")


class TestModeratorIntegration:
    """Integration tests for ModeratorAgent."""

    def test_full_moderator_workflow(self):
        """Test a complete moderator workflow across modes."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm.model_name = "gemini-2.5-pro"

        moderator = ModeratorAgent(
            "workflow-test", mock_llm, enable_error_handling=False
        )

        # Mock responses for different operations
        responses = [
            # Agenda synthesis response
            Mock(content='{"proposed_agenda": ["AI Safety", "Climate Tech"]}'),
            # Topic summary response
            Mock(content="Comprehensive summary of AI Safety discussion..."),
            # Report structure response
            Mock(content='{"report_sections": ["Summary", "Recommendations"]}'),
            # Section content response
            Mock(content="Executive summary content..."),
        ]
        mock_llm.invoke.side_effect = responses
        mock_llm.bind.return_value = mock_llm

        # 1. Parse agenda (synthesis mode)
        moderator.set_mode("synthesis")
        agenda = moderator.parse_agenda_json("Synthesize the voting results")
        assert agenda == ["AI Safety", "Climate Tech"]
        assert moderator.current_mode == "synthesis"

        # 2. Generate topic summary (synthesis mode)
        summary = moderator.generate_response("Summarize the AI Safety discussion")
        assert "Comprehensive summary" in summary

        # 3. Generate report structure (switches to writer mode)
        structure = moderator.generate_report_structure([summary])
        assert structure == ["Summary", "Recommendations"]
        assert moderator.current_mode == "synthesis"  # Should restore

        # 4. Generate section content (switches to writer mode)
        content = moderator.generate_section_content("Summary", [summary])
        assert "Executive summary content" in content
        assert moderator.current_mode == "synthesis"  # Should restore

        # Verify message count increased
        assert moderator.message_count == 4


class TestModeratorAgendaSynthesis:
    """Test ModeratorAgent agenda synthesis and voting management functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="synthesis-moderator",
            llm=self.mock_llm,
            mode="synthesis",
            enable_error_handling=False,
        )

    def test_request_topic_proposals(self):
        """Test requesting topic proposals from agents."""
        prompt = self.moderator.request_topic_proposals(
            main_topic="AI Ethics", agent_count=4
        )

        # Verify prompt content
        assert "AI Ethics" in prompt
        assert "3-5 sub-topics" in prompt
        assert "4 participants" in prompt
        assert "numbered list" in prompt

        # Should restore original mode
        assert self.moderator.current_mode == "synthesis"

    def test_request_topic_proposals_with_context(self):
        """Test requesting topic proposals with context messages."""
        from virtual_agora.state.schema import Message
        from datetime import datetime

        context_messages = [
            Message(
                id="msg1",
                speaker_id="user",
                speaker_role="moderator",
                content="Previous context",
                timestamp=datetime.now(),
                phase=1,
                topic=None,
            )
        ]

        prompt = self.moderator.request_topic_proposals(
            main_topic="Climate Change",
            agent_count=3,
            context_messages=context_messages,
        )

        assert "Climate Change" in prompt
        assert "3 participants" in prompt

    def test_collect_proposals_basic(self):
        """Test basic proposal collection and deduplication."""
        agent_responses = [
            "1. AI Safety\n2. Machine Learning Ethics\n3. Privacy Rights",
            "1. AI Safety\n2. Algorithmic Bias\n3. Data Protection",
            "1. Privacy Rights\n2. Future of Work\n3. AI Governance",
        ]

        result = self.moderator.collect_proposals(agent_responses)

        # Should have unique topics
        expected_topics = [
            "Ai Safety",
            "Machine Learning Ethics",
            "Privacy Rights",
            "Algorithmic Bias",
            "Data Protection",
            "Future Of Work",
            "Ai Governance",
        ]

        assert len(result["unique_topics"]) == 7
        assert result["total_responses"] == 3
        assert result["total_unique_topics"] == 7

        # Check specific topics are present (accounting for title case normalization)
        unique_lower = [t.lower() for t in result["unique_topics"]]
        assert "ai safety" in unique_lower
        assert "privacy rights" in unique_lower

    def test_collect_proposals_with_agent_ids(self):
        """Test proposal collection with agent ID tracking."""
        agent_responses = ["1. Topic A\n2. Topic B", "1. Topic A\n2. Topic C"]
        agent_ids = ["agent1", "agent2"]

        result = self.moderator.collect_proposals(agent_responses, agent_ids)

        # Check topic sources tracking
        assert "Topic A" in result["topic_sources"]
        assert len(result["topic_sources"]["Topic A"]) == 2
        assert "agent1" in result["topic_sources"]["Topic A"]
        assert "agent2" in result["topic_sources"]["Topic A"]

    def test_collect_proposals_mismatched_ids(self):
        """Test error handling for mismatched agent IDs."""
        agent_responses = ["1. Topic A"]
        agent_ids = ["agent1", "agent2"]  # More IDs than responses

        with pytest.raises(
            ValueError, match="agent_ids and agent_responses must have same length"
        ):
            self.moderator.collect_proposals(agent_responses, agent_ids)

    def test_extract_topics_from_response(self):
        """Test topic extraction from various response formats."""
        # Test numbered lists
        response1 = "1. First Topic\n2. Second Topic\n3. Third Topic"
        topics1 = self.moderator._extract_topics_from_response(response1)
        assert topics1 == ["First Topic", "Second Topic", "Third Topic"]

        # Test bullet points
        response2 = "- First Topic\n• Second Topic\n* Third Topic"
        topics2 = self.moderator._extract_topics_from_response(response2)
        assert topics2 == ["First Topic", "Second Topic", "Third Topic"]

        # Test mixed format
        response3 = (
            "Some intro text\n1. Topic One\nRandom text\n2. Topic Two\n- Bullet topic"
        )
        topics3 = self.moderator._extract_topics_from_response(response3)
        assert topics3 == ["Topic One", "Topic Two", "Bullet topic"]

        # Test empty response
        response4 = "No topics here, just plain text"
        topics4 = self.moderator._extract_topics_from_response(response4)
        assert topics4 == []

    def test_request_votes_basic(self):
        """Test basic voting request generation."""
        topics = ["AI Safety", "Privacy Rights", "Future of Work"]

        prompt = self.moderator.request_votes(topics)

        # Verify prompt contains all topics
        for topic in topics:
            assert topic in prompt

        # Verify voting instructions
        assert "vote on the order" in prompt
        assert "preferred order" in prompt
        assert "reasoning" in prompt

    def test_request_votes_with_instructions(self):
        """Test voting request with custom instructions."""
        topics = ["Topic A", "Topic B"]
        custom_instructions = "Consider urgency when voting"

        prompt = self.moderator.request_votes(topics, custom_instructions)

        assert custom_instructions in prompt
        assert "Topic A" in prompt
        assert "Topic B" in prompt

    def test_request_votes_empty_topics(self):
        """Test error handling for empty topic list."""
        with pytest.raises(
            ValueError, match="Cannot request votes on empty topic list"
        ):
            self.moderator.request_votes([])

    def test_synthesize_agenda_basic(self):
        """Test basic agenda synthesis from votes."""
        topics = ["AI Safety", "Privacy Rights", "Future of Work"]
        votes = [
            "My preferred order: 1, 3, 2. AI Safety should come first as it's fundamental.",
            "I prefer: 2, 1, 3. Privacy is most urgent given current regulations.",
            "Order: 1, 2, 3. Follow the logical progression.",
        ]

        # Mock successful JSON response
        mock_response = Mock()
        mock_response.content = (
            '{"proposed_agenda": ["AI Safety", "Privacy Rights", "Future of Work"]}'
        )
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        agenda = self.moderator.synthesize_agenda(topics, votes)

        assert agenda == ["AI Safety", "Privacy Rights", "Future of Work"]
        assert self.moderator.current_mode == "synthesis"  # Should restore mode

    def test_synthesize_agenda_with_voter_ids(self):
        """Test agenda synthesis with voter ID tracking."""
        topics = ["Topic A", "Topic B"]
        votes = ["Order: 1, 2", "Order: 2, 1"]
        voter_ids = ["agent1", "agent2"]

        # Mock successful JSON response
        mock_response = Mock()
        mock_response.content = '{"proposed_agenda": ["Topic A", "Topic B"]}'
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        agenda = self.moderator.synthesize_agenda(topics, votes, voter_ids)

        assert len(agenda) == 2
        # Verify the LLM was called with voter IDs in the prompt
        call_args = self.mock_llm.invoke.call_args[0][0]
        human_messages = [msg for msg in call_args if hasattr(msg, "content")]
        prompt_content = " ".join([msg.content for msg in human_messages])
        assert "agent1:" in prompt_content
        assert "agent2:" in prompt_content

    def test_synthesize_agenda_empty_inputs(self):
        """Test error handling for empty inputs."""
        # Empty topics
        with pytest.raises(
            ValueError, match="Cannot synthesize agenda from empty topics or votes"
        ):
            self.moderator.synthesize_agenda([], ["vote1"])

        # Empty votes
        with pytest.raises(
            ValueError, match="Cannot synthesize agenda from empty topics or votes"
        ):
            self.moderator.synthesize_agenda(["topic1"], [])

    def test_synthesize_agenda_failure(self):
        """Test agenda synthesis failure handling."""
        topics = ["Topic A"]
        votes = ["Invalid vote"]

        # Mock LLM failure
        self.mock_llm.invoke.side_effect = Exception("LLM error")

        with pytest.raises(ValueError, match="Failed to synthesize agenda"):
            self.moderator.synthesize_agenda(topics, votes)

    def test_parse_agenda_json_enhanced_validation(self):
        """Test enhanced validation in parse_agenda_json."""
        # Test empty agenda
        mock_response = Mock()
        mock_response.content = '{"proposed_agenda": []}'
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        with pytest.raises(ValueError, match="Agenda cannot be empty"):
            self.moderator.parse_agenda_json("Generate empty agenda")

    def test_parse_agenda_json_duplicate_handling(self):
        """Test duplicate topic handling in parse_agenda_json."""
        # Mock response with duplicates
        mock_response = Mock()
        mock_response.content = '{"proposed_agenda": ["AI Safety", "Privacy Rights", "AI SAFETY", "Future of Work"]}'
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        agenda = self.moderator.parse_agenda_json("Generate agenda with duplicates")

        # Should remove duplicates while preserving order
        assert len(agenda) == 3
        assert "AI Safety" in agenda
        assert "Privacy Rights" in agenda
        assert "Future of Work" in agenda
        # Should not contain the duplicate "AI SAFETY"
        assert agenda.count("AI Safety") == 1

    def test_parse_agenda_json_invalid_types(self):
        """Test validation of agenda item types."""
        # Mock response with non-string item
        mock_response = Mock()
        mock_response.content = (
            '{"proposed_agenda": ["Valid Topic", 123, "Another Topic"]}'
        )
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        # Pydantic validation will catch this first and wrap it in our ValueError
        with pytest.raises(ValueError, match="Failed to parse agenda"):
            self.moderator.parse_agenda_json("Generate invalid agenda")

    def test_full_agenda_workflow(self):
        """Test the complete agenda synthesis workflow."""
        # 1. Request proposals
        proposal_prompt = self.moderator.request_topic_proposals("AI Ethics", 3)
        assert "AI Ethics" in proposal_prompt

        # 2. Collect proposals
        agent_responses = [
            "1. AI Safety\n2. Privacy Rights\n3. Algorithmic Bias",
            "1. AI Safety\n2. Future of Work\n3. Data Protection",
            "1. Privacy Rights\n2. AI Governance\n3. Transparency",
        ]
        collected = self.moderator.collect_proposals(agent_responses)
        topics = collected["unique_topics"]
        assert len(topics) >= 3  # Should have multiple unique topics

        # 3. Request votes
        voting_prompt = self.moderator.request_votes(topics[:3])  # Use first 3 topics
        assert "vote on the order" in voting_prompt

        # 4. Synthesize agenda
        votes = [
            "My order: 1, 2, 3. This makes logical sense.",
            "I prefer: 2, 1, 3. Privacy should come first.",
            "Order: 1, 3, 2. AI Safety is foundational.",
        ]

        # Mock successful synthesis
        mock_response = Mock()
        mock_response.content = f'{{"proposed_agenda": {json.dumps(topics[:3])}}}'
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        final_agenda = self.moderator.synthesize_agenda(topics[:3], votes)

        assert len(final_agenda) == 3
        assert all(isinstance(topic, str) for topic in final_agenda)

    def test_mode_switching_during_operations(self):
        """Test that mode switching works correctly during operations."""
        original_mode = "facilitation"
        self.moderator.set_mode(original_mode)

        # Each method should temporarily switch to synthesis mode
        self.moderator.request_topic_proposals("Test Topic", 2)
        assert self.moderator.current_mode == original_mode

        topics = ["Topic A", "Topic B"]
        self.moderator.request_votes(topics)
        assert self.moderator.current_mode == original_mode

        # synthesize_agenda will also test mode switching
        mock_response = Mock()
        mock_response.content = '{"proposed_agenda": ["Topic A", "Topic B"]}'
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        self.moderator.synthesize_agenda(topics, ["vote1", "vote2"])
        assert self.moderator.current_mode == original_mode

    def test_moderator_context_preservation(self):
        """Test that context is preserved across mode switches."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm.model_name = "gemini-2.5-pro"

        moderator = ModeratorAgent(
            "context-test", mock_llm, enable_error_handling=False
        )

        # Set some context
        moderator.update_conversation_context("session_id", "test-123")
        moderator.update_conversation_context("topics_discussed", ["AI", "Climate"])

        # Switch modes
        moderator.set_mode("synthesis")
        moderator.set_mode("writer")
        moderator.set_mode("facilitation")

        # Context should be preserved
        assert moderator.get_conversation_context("session_id") == "test-123"
        assert moderator.get_conversation_context("topics_discussed") == [
            "AI",
            "Climate",
        ]

    def test_moderator_neutrality_across_modes(self):
        """Test that neutrality validation works across all modes."""
        mock_llm = Mock()
        moderator = ModeratorAgent(
            "neutrality-test", mock_llm, enable_error_handling=False
        )

        # Test neutrality validation in each mode
        modes = ["facilitation", "synthesis", "writer"]

        for mode in modes:
            moderator.set_mode(mode)

            # Neutral content should pass
            assert moderator.validate_neutrality("The discussion will proceed.") is True

            # Opinion content should fail
            assert moderator.validate_neutrality("I think this is great.") is False


class TestModeratorRoundManagement:
    """Test Story 3.3: Discussion Round Management functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="round-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False,
            turn_timeout_seconds=120,
        )

    def test_round_management_initialization(self):
        """Test that round management attributes are properly initialized."""
        assert self.moderator.current_round == 0
        assert self.moderator.turn_timeout_seconds == 120
        assert self.moderator.participation_metrics == {}
        assert self.moderator.speaking_order == []
        assert self.moderator.current_speaker_index == 0

    def test_announce_topic(self):
        """Test topic announcement functionality."""
        self.moderator.speaking_order = ["agent1", "agent2", "agent3"]

        announcement = self.moderator.announce_topic("AI Ethics", 1)

        assert "Round 1: AI Ethics" in announcement
        assert "agent1 → agent2 → agent3" in announcement

    def test_manage_turn_order_basic(self):
        """Test basic turn order management."""
        agent_ids = ["agent1", "agent2", "agent3"]

        self.moderator.manage_turn_order(agent_ids)

        assert self.moderator.speaking_order == agent_ids
        assert self.moderator.current_speaker_index == 0

        # Check participation metrics initialization
        for agent_id in agent_ids:
            assert agent_id in self.moderator.participation_metrics
            metrics = self.moderator.participation_metrics[agent_id]
            assert metrics["turns_taken"] == 0
            assert metrics["total_response_time"] == 0.0
            assert metrics["timeouts"] == 0
            assert metrics["words_contributed"] == 0

    def test_manage_turn_order_randomized(self):
        """Test randomized turn order management."""
        agent_ids = ["agent1", "agent2", "agent3", "agent4", "agent5"]

        # Run multiple times to check randomization
        orders = []
        for _ in range(10):
            moderator = ModeratorAgent(
                "test", self.mock_llm, enable_error_handling=False
            )
            moderator.manage_turn_order(agent_ids, randomize=True)
            orders.append(moderator.speaking_order.copy())

        # Should have some variation (not all identical)
        unique_orders = set(tuple(order) for order in orders)
        assert len(unique_orders) > 1  # At least some randomization occurred

    def test_manage_turn_order_empty_list(self):
        """Test error handling for empty agent list."""
        self.moderator.manage_turn_order([])

        # Should handle gracefully
        assert self.moderator.speaking_order == []
        assert self.moderator.current_speaker_index == 0

    def test_get_next_speaker(self):
        """Test speaker rotation functionality."""
        self.moderator.speaking_order = ["agent1", "agent2", "agent3"]
        self.moderator.current_speaker_index = 0

        # Test rotation
        assert self.moderator.get_next_speaker() == "agent1"
        assert self.moderator.current_speaker_index == 1

        assert self.moderator.get_next_speaker() == "agent2"
        assert self.moderator.current_speaker_index == 2

        assert self.moderator.get_next_speaker() == "agent3"
        assert self.moderator.current_speaker_index == 0  # Should wrap around

        assert self.moderator.get_next_speaker() == "agent1"  # Back to first

    def test_get_next_speaker_empty_order(self):
        """Test get_next_speaker with no speaking order."""
        result = self.moderator.get_next_speaker()
        assert result is None

    def test_track_participation(self):
        """Test participation tracking functionality."""
        agent_id = "agent1"
        response_content = "This is a test response with ten words in it."
        response_time = 45.5

        # Track first participation
        self.moderator.track_participation(agent_id, response_content, response_time)

        metrics = self.moderator.participation_metrics[agent_id]
        assert metrics["turns_taken"] == 1
        assert metrics["total_response_time"] == 45.5
        assert metrics["average_response_time"] == 45.5
        assert metrics["words_contributed"] == 10
        assert metrics["timeouts"] == 0

        # Track second participation
        self.moderator.track_participation(agent_id, "Shorter response.", 30.0)

        metrics = self.moderator.participation_metrics[agent_id]
        assert metrics["turns_taken"] == 2
        assert metrics["total_response_time"] == 75.5
        assert metrics["average_response_time"] == 37.75
        assert metrics["words_contributed"] == 12  # 10 + 2

    def test_handle_agent_timeout(self):
        """Test agent timeout handling."""
        agent_id = "agent1"

        # Initialize participation metrics
        self.moderator.participation_metrics[agent_id] = {
            "turns_taken": 2,
            "total_response_time": 60.0,
            "average_response_time": 30.0,
            "timeouts": 0,
            "words_contributed": 20,
            "last_activity": datetime.now(),
        }

        timeout_message = self.moderator.handle_agent_timeout(agent_id)

        # Check timeout was recorded
        assert self.moderator.participation_metrics[agent_id]["timeouts"] == 1

        # Check message content
        assert "Agent agent1 did not respond" in timeout_message
        assert "120 seconds" in timeout_message
        assert "continue to the next participant" in timeout_message

    def test_signal_round_completion(self):
        """Test round completion signaling."""
        # Set up speaking order and metrics
        self.moderator.speaking_order = ["agent1", "agent2", "agent3"]
        self.moderator.participation_metrics = {
            "agent1": {"turns_taken": 2, "words_contributed": 50},
            "agent2": {"turns_taken": 1, "words_contributed": 25},
            "agent3": {"turns_taken": 0, "words_contributed": 0},  # Inactive
        }

        completion_message = self.moderator.signal_round_completion(1, "AI Ethics")

        # Check round increment
        assert self.moderator.current_round == 1

        # Check message content
        assert "Round 1 Complete: AI Ethics" in completion_message
        assert "Total participants: 3" in completion_message
        assert "Active participants: 2" in completion_message
        assert "Participation rate: 66.7%" in completion_message

    def test_get_participation_summary(self):
        """Test participation summary generation."""
        # Set up test data
        self.moderator.participation_metrics = {
            "agent1": {
                "turns_taken": 3,
                "total_response_time": 120.0,
                "average_response_time": 40.0,
                "timeouts": 1,
                "words_contributed": 150,
            },
            "agent2": {
                "turns_taken": 2,
                "total_response_time": 80.0,
                "average_response_time": 40.0,
                "timeouts": 0,
                "words_contributed": 100,
            },
            "agent3": {
                "turns_taken": 0,
                "total_response_time": 0.0,
                "average_response_time": 0.0,
                "timeouts": 0,
                "words_contributed": 0,
            },
        }
        self.moderator.current_round = 2

        summary = self.moderator.get_participation_summary()

        # Check summary statistics
        assert summary["total_agents"] == 3
        assert summary["active_agents"] == 2
        assert summary["participation_rate"] == pytest.approx(66.67, rel=1e-2)
        assert summary["average_response_time"] == 40.0
        assert summary["total_words"] == 250
        assert summary["timeout_rate"] == 20.0  # 1 timeout out of 5 turns
        assert summary["current_round"] == 2

        # Check detailed metrics are included
        assert "detailed_metrics" in summary
        assert len(summary["detailed_metrics"]) == 3

    def test_get_participation_summary_empty(self):
        """Test participation summary with no participation."""
        summary = self.moderator.get_participation_summary()

        assert summary["total_agents"] == 0
        assert summary["active_agents"] == 0
        assert summary["average_response_time"] == 0.0
        assert summary["total_words"] == 0
        assert summary["timeout_rate"] == 0.0


class TestModeratorRelevanceEnforcement:
    """Test Story 3.4: Relevance Enforcement System functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="relevance-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False,
            relevance_threshold=0.7,
            warning_threshold=2,
            mute_duration_minutes=5,
        )

    def test_relevance_enforcement_initialization(self):
        """Test that relevance enforcement attributes are properly initialized."""
        assert self.moderator.relevance_threshold == 0.7
        assert self.moderator.warning_threshold == 2
        assert self.moderator.mute_duration_minutes == 5
        assert self.moderator.relevance_violations == {}
        assert self.moderator.muted_agents == {}
        assert self.moderator.current_topic_context is None

    def test_set_topic_context(self):
        """Test setting topic context for relevance evaluation."""
        self.moderator.set_topic_context("AI Ethics")
        assert self.moderator.current_topic_context == "AI Ethics"

        # Test with description
        self.moderator.set_topic_context(
            "AI Ethics", "Discussion of ethical considerations in AI development"
        )
        assert (
            self.moderator.current_topic_context
            == "AI Ethics: Discussion of ethical considerations in AI development"
        )

    def test_evaluate_message_relevance_no_context(self):
        """Test relevance evaluation without topic context."""
        result = self.moderator.evaluate_message_relevance("Test message", "agent1")

        # Should assume relevant when no context
        assert result["relevance_score"] == 1.0
        assert result["is_relevant"] is True
        assert result["reason"] == "No topic context available for evaluation"
        assert result["topic"] is None

    def test_evaluate_message_relevance_relevant(self):
        """Test evaluation of relevant message."""
        self.moderator.set_topic_context("AI Ethics")

        # Mock relevant assessment
        mock_response = Mock()
        mock_response.content = """
        {
            "relevance_assessment": {
                "relevance_score": 0.9,
                "is_relevant": true,
                "key_points": ["ethical considerations", "AI development"],
                "reason": "Message directly addresses AI ethics concerns"
            }
        }
        """
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.evaluate_message_relevance(
            "I believe we need stronger ethical guidelines for AI development", "agent1"
        )

        assert result["relevance_score"] == 0.9
        assert result["is_relevant"] is True
        assert result["topic"] == "AI Ethics"
        assert "ethical considerations" in result["key_points"]

    def test_evaluate_message_relevance_irrelevant(self):
        """Test evaluation of irrelevant message."""
        self.moderator.set_topic_context("AI Ethics")

        # Mock irrelevant assessment
        mock_response = Mock()
        mock_response.content = """
        {
            "relevance_assessment": {
                "relevance_score": 0.3,
                "is_relevant": false,
                "key_points": ["weather", "sports"],
                "reason": "Message about weather is unrelated to AI ethics"
            }
        }
        """
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.evaluate_message_relevance(
            "The weather is really nice today for a game", "agent1"
        )

        assert result["relevance_score"] == 0.3
        assert result["is_relevant"] is False  # Below threshold
        assert "weather" in result["key_points"]

    def test_track_relevance_violation(self):
        """Test tracking of relevance violations."""
        agent_id = "agent1"
        message_content = "This is an off-topic message"
        relevance_assessment = {
            "relevance_score": 0.4,
            "topic": "AI Ethics",
            "reason": "Message is unrelated to the topic",
        }

        self.moderator.track_relevance_violation(
            agent_id, message_content, relevance_assessment
        )

        # Check violation was recorded
        assert agent_id in self.moderator.relevance_violations
        violation_data = self.moderator.relevance_violations[agent_id]

        assert violation_data["total_violations"] == 1
        assert violation_data["warnings_issued"] == 0
        assert len(violation_data["violations_history"]) == 1
        assert violation_data["is_muted"] is False

        # Check violation record details
        violation_record = violation_data["violations_history"][0]
        assert violation_record["message_content"] == message_content
        assert violation_record["relevance_score"] == 0.4
        assert violation_record["topic"] == "AI Ethics"

    def test_issue_relevance_warning_first_warning(self):
        """Test issuing first relevance warning."""
        agent_id = "agent1"
        self.moderator.relevance_violations[agent_id] = {
            "total_violations": 1,
            "warnings_issued": 0,
            "violations_history": [],
            "last_violation": datetime.now(),
            "is_muted": False,
        }

        relevance_assessment = {
            "reason": "Message is off-topic",
            "suggestions": "Please focus on AI ethics",
        }

        warning_message = self.moderator.issue_relevance_warning(
            agent_id, relevance_assessment
        )

        # Check warning was recorded
        assert self.moderator.relevance_violations[agent_id]["warnings_issued"] == 1

        # Check warning message content
        assert "Relevance Warning for agent1" in warning_message
        assert "Message is off-topic" in warning_message
        assert "Please focus on AI ethics" in warning_message
        assert "Warnings remaining:** 1" in warning_message

    def test_issue_relevance_warning_final(self):
        """Test issuing final warning."""
        agent_id = "agent1"
        self.moderator.relevance_violations[agent_id] = {
            "total_violations": 2,
            "warnings_issued": 1,
            "violations_history": [],
            "last_violation": datetime.now(),
            "is_muted": False,
        }

        relevance_assessment = {
            "reason": "Another off-topic message",
            "suggestions": "Stay focused on the current topic",
        }

        warning_message = self.moderator.issue_relevance_warning(
            agent_id, relevance_assessment
        )

        # Check this was the final warning
        assert "Final Warning for agent1" in warning_message
        assert "next irrelevant message will result in" in warning_message
        assert "5-minute mute" in warning_message

    def test_mute_agent(self):
        """Test muting an agent."""
        agent_id = "agent1"
        reason = "Excessive relevance violations"

        # Initialize violation record
        self.moderator.relevance_violations[agent_id] = {
            "total_violations": 3,
            "warnings_issued": 2,
            "violations_history": [],
            "last_violation": datetime.now(),
            "is_muted": False,
        }

        mute_message = self.moderator.mute_agent(agent_id, reason)

        # Check agent was muted
        assert agent_id in self.moderator.muted_agents
        mute_end_time = self.moderator.muted_agents[agent_id]
        time_diff = (mute_end_time - datetime.now()).total_seconds()
        # Should be about 5 minutes (300 seconds)
        assert 290 <= time_diff <= 305

        # Check violation record updated
        assert self.moderator.relevance_violations[agent_id]["is_muted"] is True

        # Check mute message
        assert "Agent agent1 has been temporarily muted" in mute_message
        assert reason in mute_message
        assert "Duration:** 5 minutes" in mute_message

    def test_check_agent_mute_status_not_muted(self):
        """Test checking mute status for non-muted agent."""
        status = self.moderator.check_agent_mute_status("agent1")

        assert status["is_muted"] is False
        assert status["mute_end_time"] is None
        assert status["time_remaining_minutes"] == 0

    def test_check_agent_mute_status_muted(self):
        """Test checking mute status for muted agent."""
        from datetime import timedelta

        agent_id = "agent1"
        mute_end_time = datetime.now() + timedelta(minutes=3)
        self.moderator.muted_agents[agent_id] = mute_end_time

        status = self.moderator.check_agent_mute_status(agent_id)

        assert status["is_muted"] is True
        assert status["mute_end_time"] == mute_end_time
        assert 2.8 <= status["time_remaining_minutes"] <= 3.2

    def test_check_agent_mute_status_expired(self):
        """Test checking mute status for expired mute."""
        from datetime import timedelta

        agent_id = "agent1"
        # Set mute to expire 1 minute ago
        mute_end_time = datetime.now() - timedelta(minutes=1)
        self.moderator.muted_agents[agent_id] = mute_end_time

        # Initialize violation record
        self.moderator.relevance_violations[agent_id] = {"is_muted": True}

        status = self.moderator.check_agent_mute_status(agent_id)

        # Should have been automatically unmuted
        assert status["is_muted"] is False
        assert agent_id not in self.moderator.muted_agents
        assert self.moderator.relevance_violations[agent_id]["is_muted"] is False

    def test_process_message_for_relevance_allowed(self):
        """Test processing relevant message."""
        self.moderator.set_topic_context("AI Ethics")

        # Mock relevant assessment
        mock_response = Mock()
        mock_response.content = """
        {
            "relevance_assessment": {
                "relevance_score": 0.8,
                "is_relevant": true,
                "key_points": ["AI ethics"],
                "reason": "Message is relevant"
            }
        }
        """
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.process_message_for_relevance(
            "AI ethics is important", "agent1"
        )

        assert result["action"] == "allowed"
        assert result["reason"] == "relevant_content"
        assert result["message"] is None
        assert result["relevance_assessment"]["relevance_score"] == 0.8

    def test_process_message_for_relevance_warned(self):
        """Test processing irrelevant message (first warning)."""
        self.moderator.set_topic_context("AI Ethics")

        # Mock irrelevant assessment
        mock_response = Mock()
        mock_response.content = """
        {
            "relevance_assessment": {
                "relevance_score": 0.3,
                "is_relevant": false,
                "key_points": ["weather"],
                "reason": "Message is off-topic"
            }
        }
        """
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.process_message_for_relevance(
            "The weather is nice today", "agent1"
        )

        assert result["action"] == "warned"
        assert result["reason"] == "irrelevant_content"
        assert "Relevance Warning" in result["message"]
        assert result["warnings_issued"] == 1
        assert result["warnings_remaining"] == 1

    def test_process_message_for_relevance_muted(self):
        """Test processing message that results in muting."""
        self.moderator.set_topic_context("AI Ethics")

        # Set up agent with warnings at threshold
        agent_id = "agent1"
        self.moderator.relevance_violations[agent_id] = {
            "total_violations": 2,
            "warnings_issued": 2,  # At threshold
            "violations_history": [],
            "last_violation": datetime.now(),
            "is_muted": False,
        }

        # Mock irrelevant assessment
        mock_response = Mock()
        mock_response.content = """
        {
            "relevance_assessment": {
                "relevance_score": 0.2,
                "is_relevant": false,
                "key_points": ["sports"],
                "reason": "Message is completely off-topic"
            }
        }
        """
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        result = self.moderator.process_message_for_relevance(
            "Who won the game last night?", agent_id
        )

        assert result["action"] == "muted"
        assert result["reason"] == "excessive_violations"
        assert "Agent agent1 has been temporarily muted" in result["message"]
        assert result["total_violations"] == 3

    def test_process_message_for_relevance_blocked_muted(self):
        """Test processing message from already muted agent."""
        from datetime import timedelta

        agent_id = "agent1"
        self.moderator.muted_agents[agent_id] = datetime.now() + timedelta(minutes=3)

        result = self.moderator.process_message_for_relevance("Any message", agent_id)

        assert result["action"] == "blocked"
        assert result["reason"] == "agent_muted"
        assert "agent1 is muted for" in result["message"]
        assert result["relevance_assessment"] is None

    def test_get_relevance_enforcement_summary(self):
        """Test getting relevance enforcement summary."""
        # Set up test data
        self.moderator.current_topic_context = "AI Ethics"
        self.moderator.relevance_violations = {
            "agent1": {
                "total_violations": 2,
                "warnings_issued": 1,
                "violations_history": [],
                "is_muted": False,
            },
            "agent2": {
                "total_violations": 3,
                "warnings_issued": 2,
                "violations_history": [],
                "is_muted": True,
            },
        }
        self.moderator.muted_agents = {"agent2": datetime.now()}

        summary = self.moderator.get_relevance_enforcement_summary()

        assert summary["current_topic"] == "AI Ethics"
        assert summary["relevance_threshold"] == 0.7
        assert summary["warning_threshold"] == 2
        assert summary["mute_duration_minutes"] == 5
        assert summary["total_violations"] == 5
        assert summary["total_warnings_issued"] == 3
        assert summary["agents_with_violations"] == 2
        assert summary["currently_muted_agents"] == 1
        assert summary["muted_agents_list"] == ["agent2"]


class TestModeratorSummarization:
    """Test Story 3.5: Round and Topic Summarization functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="summary-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False,
        )

    def test_generate_round_summary(self):
        """Test round summary generation."""
        # Mock response
        mock_response = Mock()
        mock_response.content = "This round focused on AI safety concerns with key insights about alignment."
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        # Test messages
        messages = [
            {
                "id": "msg1",
                "speaker_id": "agent1",
                "content": "AI safety is crucial for future development",
                "topic": "AI Safety",
                "round": 1,
            },
            {
                "id": "msg2",
                "speaker_id": "agent2",
                "content": "We need better alignment techniques",
                "topic": "AI Safety",
                "round": 1,
            },
        ]

        participation_summary = {"active_agents": 2, "average_response_time": 30.0}

        original_mode = self.moderator.current_mode
        summary = self.moderator.generate_round_summary(
            1, "AI Safety", messages, participation_summary
        )

        assert "AI safety concerns" in summary
        assert self.moderator.current_mode == original_mode  # Should restore mode

    def test_generate_round_summary_no_messages(self):
        """Test round summary with no messages."""
        summary = self.moderator.generate_round_summary(1, "AI Safety", [])

        assert "Round 1 on 'AI Safety' had no recorded contributions" in summary

    def test_generate_topic_summary_with_round_summaries(self):
        """Test topic summary using round summaries."""
        mock_response = Mock()
        mock_response.content = "Comprehensive topic summary based on round summaries."
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        # Create some messages for the topic
        messages = [
            Message(
                id="msg1",
                speaker_id="agent1",
                speaker_role="participant",
                content="Initial thoughts on AI safety",
                timestamp=datetime.now(),
                phase=2,
                topic="AI Safety",
            )
        ]

        round_summaries = [
            "Round 1 focused on basic AI safety concepts",
            "Round 2 discussed technical implementation challenges",
        ]

        summary = self.moderator.generate_topic_summary(
            "AI Safety", messages, round_summaries
        )

        assert "Comprehensive topic summary" in summary

    def test_generate_topic_summary_map_reduce(self):
        """Test topic summary using map-reduce for large message sets."""
        # Mock responses for map phase
        map_responses = [
            Mock(content="Summary of chunk 1"),
            Mock(content="Summary of chunk 2"),
            Mock(content="Summary of chunk 3"),
        ]
        # Mock response for reduce phase
        reduce_response = Mock(content="Final comprehensive summary from map-reduce")

        # Set up mock to return different responses in sequence
        self.mock_llm.invoke.side_effect = map_responses + [reduce_response]
        self.mock_llm.bind.return_value = self.mock_llm

        # Create 15 messages (more than threshold of 10)
        messages = []
        for i in range(15):
            messages.append(
                {
                    "id": f"msg{i}",
                    "speaker_id": f"agent{i%3}",
                    "content": f"Message {i} about AI safety",
                    "topic": "AI Safety",
                }
            )

        summary = self.moderator.generate_topic_summary("AI Safety", messages)

        assert "Final comprehensive summary from map-reduce" in summary

    def test_extract_key_insights(self):
        """Test key insights extraction from topic summaries."""
        # Mock JSON response
        mock_response = Mock()
        mock_response.content = """
        {
            "key_insights": {
                "main_themes": ["AI safety", "Technical challenges"],
                "areas_of_consensus": ["Need for better alignment"],
                "points_of_disagreement": ["Timeline for implementation"],
                "action_items": ["Research alignment techniques"],
                "future_considerations": ["Long-term safety measures"]
            }
        }
        """
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        topic_summaries = [
            "AI safety discussion revealed consensus on alignment importance",
            "Technical challenges discussion showed disagreement on timelines",
        ]

        session_context = {"duration": "2 hours", "participants": 5}

        insights = self.moderator.extract_key_insights(topic_summaries, session_context)

        assert insights["main_themes"] == ["AI safety", "Technical challenges"]
        assert insights["areas_of_consensus"] == ["Need for better alignment"]
        assert len(insights["action_items"]) == 1

    def test_extract_key_insights_empty_summaries(self):
        """Test key insights extraction with empty summaries list."""
        insights = self.moderator.extract_key_insights([])

        # Should return empty structure
        assert insights["main_themes"] == []
        assert insights["areas_of_consensus"] == []
        assert insights["points_of_disagreement"] == []
        assert insights["action_items"] == []
        assert insights["future_considerations"] == []

    def test_extract_key_insights_error_handling(self):
        """Test key insights extraction error handling."""
        # Mock LLM error
        self.mock_llm.invoke.side_effect = Exception("LLM error")

        insights = self.moderator.extract_key_insights(["Test summary"])

        # Should return empty structure on error
        assert insights["main_themes"] == []
        assert insights["areas_of_consensus"] == []

    def test_generate_progressive_summary_initial(self):
        """Test progressive summary generation (initial)."""
        mock_response = Mock()
        mock_response.content = "Initial summary of the discussion."
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        messages = [
            {"speaker_id": "agent1", "content": "First message about AI safety"}
        ]

        summary = self.moderator.generate_progressive_summary(
            messages, topic="AI Safety"
        )

        assert "Initial summary" in summary

    def test_generate_progressive_summary_refine(self):
        """Test progressive summary refinement."""
        mock_response = Mock()
        mock_response.content = "Updated summary incorporating new insights."
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm

        existing_summary = "Previous summary of the discussion"
        new_messages = [
            {
                "speaker_id": "agent2",
                "content": "New insights about technical challenges",
            }
        ]

        summary = self.moderator.generate_progressive_summary(
            new_messages, existing_summary=existing_summary, topic="AI Safety"
        )

        assert "Updated summary" in summary

    def test_generate_progressive_summary_no_messages(self):
        """Test progressive summary with no new messages."""
        existing_summary = "Existing summary"

        summary = self.moderator.generate_progressive_summary([], existing_summary)

        assert summary == existing_summary


class TestModeratorPolling:
    """Test Story 3.6: Topic Conclusion Polling functionality."""

    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-2.5-pro"

        self.moderator = ModeratorAgent(
            agent_id="polling-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False,
        )

    def test_initiate_conclusion_poll(self):
        """Test initiating a conclusion poll."""
        eligible_voters = ["agent1", "agent2", "agent3"]

        result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", eligible_voters, 10
        )

        assert result["status"] == "initiated"
        assert "poll_id" in result
        assert "AI Ethics" in result["announcement"]
        assert "Duration:** 10 minutes" in result["announcement"]
        assert "agent1, agent2, agent3" in result["announcement"]

        # Check poll data stored in context
        poll_data = result["poll_data"]
        assert poll_data["topic"] == "AI Ethics"
        assert poll_data["eligible_voters"] == eligible_voters
        assert poll_data["status"] == "active"
        assert poll_data["options"] == ["continue", "conclude"]

    def test_cast_vote_valid(self):
        """Test casting a valid vote."""
        # Set up poll
        poll_result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", ["agent1", "agent2"]
        )
        poll_id = poll_result["poll_id"]

        # Cast vote
        result = self.moderator.cast_vote(
            poll_id, "agent1", "conclude", "We've covered the main points"
        )

        assert result["success"] is True
        assert "Vote Recorded" in result["message"]
        assert "agent1" in result["message"]
        assert "conclude" in result["message"]
        assert "We've covered the main points" in result["message"]
        assert result["votes_received"] == 1
        assert result["votes_required"] == 2
        assert result["poll_complete"] is False

    def test_cast_vote_completes_poll(self):
        """Test that poll completes when all votes are cast."""
        # Set up poll with 2 voters
        poll_result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", ["agent1", "agent2"]
        )
        poll_id = poll_result["poll_id"]

        # Cast first vote
        self.moderator.cast_vote(poll_id, "agent1", "conclude", "Ready to conclude")

        # Cast second vote (should complete poll)
        result = self.moderator.cast_vote(
            poll_id, "agent2", "continue", "Need more discussion"
        )

        assert result["success"] is True
        assert result["poll_complete"] is True
        assert "All votes have been received" in result["message"]

    def test_cast_vote_poll_not_found(self):
        """Test casting vote in non-existent poll."""
        result = self.moderator.cast_vote("nonexistent", "agent1", "conclude")

        assert result["success"] is False
        assert result["error"] == "Poll nonexistent not found"

    def test_cast_vote_voter_not_eligible(self):
        """Test casting vote by ineligible voter."""
        poll_result = self.moderator.initiate_conclusion_poll("AI Ethics", ["agent1"])
        poll_id = poll_result["poll_id"]

        result = self.moderator.cast_vote(poll_id, "agent2", "conclude")

        assert result["success"] is False
        assert result["error"] == "voter_not_eligible"
        assert "agent2 is not eligible" in result["message"]

    def test_cast_vote_already_voted(self):
        """Test casting vote when already voted."""
        poll_result = self.moderator.initiate_conclusion_poll("AI Ethics", ["agent1"])
        poll_id = poll_result["poll_id"]

        # Cast first vote
        self.moderator.cast_vote(poll_id, "agent1", "conclude")

        # Try to vote again
        result = self.moderator.cast_vote(poll_id, "agent1", "continue")

        assert result["success"] is False
        assert result["error"] == "already_voted"
        assert "agent1 has already voted" in result["message"]

    def test_cast_vote_invalid_choice(self):
        """Test casting vote with invalid choice."""
        poll_result = self.moderator.initiate_conclusion_poll("AI Ethics", ["agent1"])
        poll_id = poll_result["poll_id"]

        result = self.moderator.cast_vote(poll_id, "agent1", "invalid_choice")

        assert result["success"] is False
        assert result["error"] == "invalid_choice"
        assert "Invalid vote choice 'invalid_choice'" in result["message"]

    def test_tally_poll_results_conclude_wins(self):
        """Test tallying poll results where conclude wins."""
        # Set up poll
        poll_result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", ["agent1", "agent2", "agent3"]
        )
        poll_id = poll_result["poll_id"]

        # Cast votes
        self.moderator.cast_vote(poll_id, "agent1", "conclude", "We've covered enough")
        self.moderator.cast_vote(poll_id, "agent2", "conclude", "Time to move on")
        self.moderator.cast_vote(poll_id, "agent3", "continue", "Need more time")

        # Tally results
        results = self.moderator.tally_poll_results(poll_id)

        assert results["success"] is True
        assert results["decision"] == "conclude"
        assert results["vote_counts"]["conclude"] == 2
        assert results["vote_counts"]["continue"] == 1
        assert results["total_votes"] == 3
        assert results["margin"] == 1
        assert pytest.approx(results["winning_percentage"], rel=1e-2) == 66.67

        # Check message content
        assert "Decision:** CONCLUDE" in results["message"]
        assert "**Conclude:** 2 votes (66.7%)" in results["message"]
        assert "**Continue:** 1 votes (33.3%)" in results["message"]

    def test_tally_poll_results_continue_wins(self):
        """Test tallying poll results where continue wins."""
        poll_result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", ["agent1", "agent2"]
        )
        poll_id = poll_result["poll_id"]

        self.moderator.cast_vote(poll_id, "agent1", "continue", "More to discuss")
        self.moderator.cast_vote(
            poll_id, "agent2", "continue", "Important points missed"
        )

        results = self.moderator.tally_poll_results(poll_id)

        assert results["decision"] == "continue"
        assert results["vote_counts"]["continue"] == 2
        assert results["vote_counts"]["conclude"] == 0
        assert results["margin"] == 2

    def test_tally_poll_results_tie(self):
        """Test tallying poll results with tie."""
        poll_result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", ["agent1", "agent2"]
        )
        poll_id = poll_result["poll_id"]

        self.moderator.cast_vote(poll_id, "agent1", "conclude", "Ready to conclude")
        self.moderator.cast_vote(poll_id, "agent2", "continue", "Need more time")

        results = self.moderator.tally_poll_results(poll_id)

        assert results["decision"] == "continue"  # Tie defaults to continue
        assert results["margin"] == 0
        assert "tied vote" in results["message"]

    def test_tally_poll_results_no_votes(self):
        """Test tallying poll with no votes."""
        poll_result = self.moderator.initiate_conclusion_poll("AI Ethics", ["agent1"])
        poll_id = poll_result["poll_id"]

        results = self.moderator.tally_poll_results(poll_id)

        assert results["success"] is False
        assert results["error"] == "no_votes"

    def test_handle_minority_considerations(self):
        """Test handling minority considerations after poll."""
        # Create poll results where conclude wins
        poll_results = {
            "success": True,
            "decision": "conclude",
            "vote_counts": {"conclude": 2, "continue": 1},
            "margin": 1,
            "poll_data": {
                "votes_cast": {
                    "agent1": {"choice": "conclude", "reasoning": "Ready to move on"},
                    "agent2": {"choice": "conclude", "reasoning": "Covered enough"},
                    "agent3": {"choice": "continue", "reasoning": "Need more time"},
                }
            },
        }

        minority_message = self.moderator.handle_minority_considerations(
            poll_results, "AI Ethics"
        )

        assert "Minority Final Considerations" in minority_message
        assert "agent3" in minority_message  # The minority voter
        assert "Poll Decision:** CONCLUDE" in minority_message
        assert "3 minutes to provide any final considerations" in minority_message

    def test_handle_minority_considerations_no_minority(self):
        """Test minority considerations when there's no minority."""
        poll_results = {
            "success": True,
            "decision": "conclude",
            "vote_counts": {"conclude": 3, "continue": 0},
            "margin": 3,
        }

        minority_message = self.moderator.handle_minority_considerations(
            poll_results, "AI Ethics"
        )

        assert minority_message == ""  # No minority message

    def test_handle_minority_considerations_tie(self):
        """Test minority considerations with tied vote."""
        poll_results = {
            "success": True,
            "decision": "continue",
            "vote_counts": {"conclude": 1, "continue": 1},
            "margin": 0,
        }

        minority_message = self.moderator.handle_minority_considerations(
            poll_results, "AI Ethics"
        )

        assert minority_message == ""  # No minority when tied

    def test_check_poll_status_active(self):
        """Test checking status of active poll."""
        poll_result = self.moderator.initiate_conclusion_poll(
            "AI Ethics", ["agent1", "agent2"]
        )
        poll_id = poll_result["poll_id"]

        status = self.moderator.check_poll_status(poll_id)

        assert status["exists"] is True
        assert status["poll_id"] == poll_id
        assert status["topic"] == "AI Ethics"
        assert status["status"] == "active"
        assert status["votes_cast"] == 0
        assert status["votes_required"] == 2
        assert status["remaining_voters"] == ["agent1", "agent2"]
        # Should be close to 5 minutes
        assert status["time_remaining_minutes"] > 4.5

    def test_check_poll_status_ready_for_tally(self):
        """Test poll status when ready for tally."""
        poll_result = self.moderator.initiate_conclusion_poll("AI Ethics", ["agent1"])
        poll_id = poll_result["poll_id"]

        # Cast the only required vote
        self.moderator.cast_vote(poll_id, "agent1", "conclude")

        status = self.moderator.check_poll_status(poll_id)

        assert status["status"] == "ready_for_tally"
        assert status["votes_cast"] == 1
        assert status["remaining_voters"] == []

    def test_check_poll_status_nonexistent(self):
        """Test checking status of non-existent poll."""
        status = self.moderator.check_poll_status("nonexistent")

        assert status["exists"] is False
        assert "not found" in status["error"]

    def test_get_active_polls(self):
        """Test getting list of active polls."""
        # Create multiple polls
        poll1 = self.moderator.initiate_conclusion_poll("AI Ethics", ["agent1"])
        poll2 = self.moderator.initiate_conclusion_poll(
            "Climate Change", ["agent1", "agent2"]
        )

        active_polls = self.moderator.get_active_polls()

        assert len(active_polls) == 2

        # Check poll details
        poll_topics = [poll["topic"] for poll in active_polls]
        assert "AI Ethics" in poll_topics
        assert "Climate Change" in poll_topics

        # Check that all are active
        for poll in active_polls:
            assert poll["status"] in ["active", "ready_for_tally"]
            assert "votes_progress" in poll
            assert "time_remaining" in poll

    def test_get_active_polls_empty(self):
        """Test getting active polls when none exist."""
        active_polls = self.moderator.get_active_polls()

        assert active_polls == []
