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
    ReportStructure,
    create_moderator_agent,
    create_gemini_moderator,
    AGENDA_SCHEMA,
    REPORT_STRUCTURE_SCHEMA
)
from virtual_agora.state.schema import Message


class TestModeratorAgent:
    """Test ModeratorAgent core functionality."""
    
    def setup_method(self):
        """Set up test method."""
        # Create mock LLM instance
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-1.5-pro"
        
        # Create moderator agent with error handling disabled to avoid LLM wrapping
        self.moderator = ModeratorAgent(
            agent_id="test-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False
        )
    
    def test_moderator_initialization(self):
        """Test moderator initialization."""
        assert self.moderator.agent_id == "test-moderator"
        assert self.moderator.llm == self.mock_llm
        assert self.moderator.role == "moderator"
        assert self.moderator.current_mode == "facilitation"
        assert self.moderator.model == "gemini-1.5-pro"
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
            agent_id="custom-mod",
            llm=self.mock_llm,
            system_prompt=custom_prompt
        )
        
        assert moderator.system_prompt == custom_prompt
        assert moderator._custom_system_prompt == custom_prompt
    
    def test_mode_switching(self):
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
    
    def test_mode_switching_with_custom_prompt(self):
        """Test that custom prompts are preserved during mode switching."""
        custom_prompt = "Custom moderator prompt"
        moderator = ModeratorAgent(
            agent_id="custom-mod",
            llm=self.mock_llm,
            system_prompt=custom_prompt
        )
        
        # Mode switching should not change custom prompt
        moderator.set_mode("synthesis")
        assert moderator.system_prompt == custom_prompt
        assert moderator.current_mode == "synthesis"
    
    def test_mode_switching_same_mode(self):
        """Test switching to the same mode (no-op)."""
        original_prompt = self.moderator.system_prompt
        self.moderator.set_mode("facilitation")  # Same mode
        assert self.moderator.current_mode == "facilitation"
        assert self.moderator.system_prompt == original_prompt
    
    def test_get_current_mode(self):
        """Test getting current mode."""
        assert self.moderator.get_current_mode() == "facilitation"
        
        self.moderator.set_mode("synthesis")
        assert self.moderator.get_current_mode() == "synthesis"
    
    def test_mode_specific_prompts(self):
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
        assert self.moderator.get_conversation_context("nonexistent", "default") == "default"
        
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
            "The poll results indicate a majority vote to continue."
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
            "Obviously, this is the best solution."
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
            "Test prompt",
            validate_neutrality=False
        )
        assert response == "I believe this is correct."


class TestModeratorJSONGeneration:
    """Test JSON generation and validation functionality."""
    
    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-1.5-pro"
        
        self.moderator = ModeratorAgent(
            agent_id="json-moderator",
            llm=self.mock_llm,
            mode="synthesis",
            enable_error_handling=False
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
            "Generate agenda",
            expected_schema=AGENDA_SCHEMA
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
            "Generate agenda",
            model_class=AgendaResponse
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
        mock_response.content = '{"proposed_agenda": ["Topic A", "Topic B"'  # Missing closing bracket
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm
        
        with pytest.raises(ValueError, match="Invalid JSON response|No JSON structure found"):
            self.moderator.generate_json_response("Generate agenda")
    
    def test_generate_json_response_schema_validation_failure(self):
        """Test JSON schema validation failure."""
        # Valid JSON but missing required field
        json_response = '{"wrong_field": ["Topic A", "Topic B"]}'
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm
        
        with pytest.raises(ValueError, match="Required field 'proposed_agenda' missing"):
            self.moderator.generate_json_response(
                "Generate agenda",
                expected_schema=AGENDA_SCHEMA
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
            "Generate agenda",
            model_class=AgendaResponse
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
                "Generate agenda",
                model_class=AgendaResponse
            )
    
    def test_parse_agenda_json(self):
        """Test agenda parsing functionality."""
        # Mock valid agenda response
        json_response = '{"proposed_agenda": ["AI Safety", "Climate Tech", "Future of Work"]}'
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
        self.mock_llm.model_name = "gemini-1.5-pro"
        
        self.moderator = ModeratorAgent(
            agent_id="report-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False
        )
    
    def test_generate_report_structure(self):
        """Test report structure generation."""
        # Mock JSON response for report structure
        json_response = '''
        {
            "report_sections": [
                "Executive Summary",
                "Key Insights",
                "Recommendations",
                "Conclusion"
            ]
        }
        '''
        mock_response = Mock()
        mock_response.content = json_response
        self.mock_llm.invoke.return_value = mock_response
        self.mock_llm.bind.return_value = self.mock_llm
        
        # Test topic summaries
        topic_summaries = [
            "Discussion on AI safety revealed concerns about alignment.",
            "Climate technology discussion focused on renewable energy solutions."
        ]
        
        # Should switch to writer mode temporarily
        original_mode = self.moderator.current_mode
        sections = self.moderator.generate_report_structure(topic_summaries)
        
        expected_sections = [
            "Executive Summary",
            "Key Insights", 
            "Recommendations",
            "Conclusion"
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
            "Climate tech discussion emphasized renewable solutions."
        ]
        
        original_mode = self.moderator.current_mode
        content = self.moderator.generate_section_content(
            "Executive Summary",
            topic_summaries,
            "High-level overview of session insights"
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
        self.moderator.generate_section_content(
            "Test Section",
            ["Test summary"]
        )
        
        # Should restore original mode
        assert self.moderator.current_mode == original_mode


class TestModeratorLangGraphIntegration:
    """Test ModeratorAgent's LangGraph integration."""
    
    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        self.mock_llm.model_name = "gemini-1.5-pro"
        
        self.moderator = ModeratorAgent(
            agent_id="langgraph-moderator",
            llm=self.mock_llm,
            mode="facilitation",
            enable_error_handling=False
        )
    
    def test_moderator_as_langgraph_node(self):
        """Test moderator can be used as a LangGraph node."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "I will facilitate the agenda creation."
        self.mock_llm.invoke.return_value = mock_response
        
        # Create state with messages
        state = {
            "messages": [
                HumanMessage(content="Please facilitate the discussion.")
            ]
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
            "messages_by_topic": {}
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
            (3, "Climate Change", "Should we conclude our discussion on 'Climate Change'?")
        ]
        
        for phase, topic, expected_prompt in test_cases:
            mock_response.content = f"Phase {phase} response"
            
            state = {
                "current_phase": phase,
                "active_topic": topic,
                "agents": {},
                "messages": [],
                "messages_by_agent": {},
                "messages_by_topic": {}
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
        state = {
            "messages": [
                HumanMessage(content="Facilitate async discussion")
            ]
        }
        
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
        mock_llm.model_name = "gemini-1.5-pro"
        
        moderator = create_moderator_agent(
            agent_id="factory-moderator",
            llm=mock_llm,
            mode="synthesis",
            enable_error_handling=False
        )
        
        assert isinstance(moderator, ModeratorAgent)
        assert moderator.agent_id == "factory-moderator"
        assert moderator.llm == mock_llm
        assert moderator.current_mode == "synthesis"
        assert moderator.role == "moderator"
    
    @patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_create_gemini_moderator(self, mock_chat_google):
        """Test create_gemini_moderator factory function."""
        # Mock the ChatGoogleGenerativeAI class
        mock_llm_instance = Mock()
        mock_llm_instance.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm_instance.model_name = "gemini-1.5-pro"
        mock_chat_google.return_value = mock_llm_instance
        
        moderator = create_gemini_moderator(
            agent_id="gemini-moderator",
            mode="writer",
            temperature=0.7,
            enable_error_handling=False
        )
        
        # Verify ChatGoogleGenerativeAI was called with correct parameters
        mock_chat_google.assert_called_once_with(
            model="gemini-1.5-pro",
            temperature=0.7
        )
        
        # Verify moderator was created correctly
        assert isinstance(moderator, ModeratorAgent)
        assert moderator.agent_id == "gemini-moderator"
        assert moderator.current_mode == "writer"
        assert moderator.llm == mock_llm_instance
    
    def test_create_gemini_moderator_import_error(self):
        """Test create_gemini_moderator handles import error."""
        with patch.dict('sys.modules', {'langchain_google_genai': None}):
            with pytest.raises(ImportError, match="google-generativeai package required"):
                create_gemini_moderator()


class TestModeratorJSONSchemaValidation:
    """Test JSON schema validation functionality."""
    
    def setup_method(self):
        """Set up test method."""
        self.mock_llm = Mock()
        self.moderator = ModeratorAgent("test", self.mock_llm, enable_error_handling=False)
    
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
        
        with pytest.raises(ValueError, match="Required field 'proposed_agenda' missing"):
            self.moderator._validate_json_schema(data, schema)
    
    def test_validate_json_schema_wrong_type(self):
        """Test validation failure for wrong field type."""
        # String instead of array
        data = {"proposed_agenda": "Topic A"}
        schema = AGENDA_SCHEMA
        
        with pytest.raises(ValueError, match="Field 'proposed_agenda' should be an array"):
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
        self.mock_llm.model_name = "gemini-1.5-pro"
        
        self.moderator = ModeratorAgent("error-test", self.mock_llm, enable_error_handling=False)
    
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
        mock_llm.model_name = "gemini-1.5-pro"
        
        moderator = ModeratorAgent("workflow-test", mock_llm, enable_error_handling=False)
        
        # Mock responses for different operations
        responses = [
            # Agenda synthesis response
            Mock(content='{"proposed_agenda": ["AI Safety", "Climate Tech"]}'),
            # Topic summary response
            Mock(content="Comprehensive summary of AI Safety discussion..."),
            # Report structure response
            Mock(content='{"report_sections": ["Summary", "Recommendations"]}'),
            # Section content response
            Mock(content="Executive summary content...")
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
    
    def test_moderator_context_preservation(self):
        """Test that context is preserved across mode switches."""
        mock_llm = Mock()
        mock_llm.__class__.__name__ = "ChatGoogleGenerativeAI"
        mock_llm.model_name = "gemini-1.5-pro"
        
        moderator = ModeratorAgent("context-test", mock_llm, enable_error_handling=False)
        
        # Set some context
        moderator.update_conversation_context("session_id", "test-123")
        moderator.update_conversation_context("topics_discussed", ["AI", "Climate"])
        
        # Switch modes
        moderator.set_mode("synthesis")
        moderator.set_mode("writer") 
        moderator.set_mode("facilitation")
        
        # Context should be preserved
        assert moderator.get_conversation_context("session_id") == "test-123"
        assert moderator.get_conversation_context("topics_discussed") == ["AI", "Climate"]
    
    def test_moderator_neutrality_across_modes(self):
        """Test that neutrality validation works across all modes."""
        mock_llm = Mock()
        moderator = ModeratorAgent("neutrality-test", mock_llm, enable_error_handling=False)
        
        # Test neutrality validation in each mode
        modes = ["facilitation", "synthesis", "writer"]
        
        for mode in modes:
            moderator.set_mode(mode)
            
            # Neutral content should pass
            assert moderator.validate_neutrality("The discussion will proceed.") is True
            
            # Opinion content should fail
            assert moderator.validate_neutrality("I think this is great.") is False