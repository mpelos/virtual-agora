"""Moderator Agent implementation for Virtual Agora.

This module provides a specialized Moderator agent that inherits from LLMAgent
and adds specific functionality for discussion facilitation, agenda synthesis,
and report generation.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.state.schema import Message, VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)

# Type alias for moderator operational modes
ModeratorMode = Literal["facilitation", "synthesis", "writer"]

# JSON schema definitions for structured outputs
AGENDA_SCHEMA = {
    "type": "object",
    "properties": {
        "proposed_agenda": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of topics for discussion"
        }
    },
    "required": ["proposed_agenda"]
}

REPORT_STRUCTURE_SCHEMA = {
    "type": "object", 
    "properties": {
        "report_sections": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of report section titles"
        }
    },
    "required": ["report_sections"]
}


class AgendaResponse(BaseModel):
    """Pydantic model for agenda response validation."""
    proposed_agenda: List[str]


class ReportStructure(BaseModel):
    """Pydantic model for report structure validation."""
    report_sections: List[str]


class ModeratorAgent(LLMAgent):
    """Specialized Moderator agent for Virtual Agora discussions.
    
    This agent inherits from LLMAgent and adds moderator-specific functionality:
    - Multiple operational modes (facilitation, synthesis, writer)
    - Structured output generation (JSON)
    - Process-oriented prompting that avoids opinion expression
    - Conversation context management across modes
    """
    
    PROMPT_VERSION = "1.0"
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        mode: ModeratorMode = "facilitation",
        system_prompt: Optional[str] = None,
        enable_error_handling: bool = True,
        max_retries: int = 3,
        fallback_llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        """Initialize the Moderator agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: LangChain chat model instance
            mode: Initial operational mode
            system_prompt: Optional custom system prompt (overrides mode-based prompt)
            enable_error_handling: Whether to enable enhanced error handling
            max_retries: Maximum number of retries for failed operations
            fallback_llm: Optional fallback LLM for error recovery
            tools: Optional list of tools the agent can use
        """
        self.current_mode = mode
        self.conversation_context: Dict[str, Any] = {}
        self._custom_system_prompt = system_prompt
        
        # Initialize with moderator-specific system prompt
        mode_prompt = self._get_mode_specific_prompt(mode) if not system_prompt else system_prompt
        
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            role="moderator",
            system_prompt=mode_prompt,
            enable_error_handling=enable_error_handling,
            max_retries=max_retries,
            fallback_llm=fallback_llm,
            tools=tools
        )
        
        logger.info(
            f"Initialized ModeratorAgent {agent_id} with mode={mode}, "
            f"prompt_version={self.PROMPT_VERSION}"
        )
    
    def set_mode(self, mode: ModeratorMode) -> None:
        """Switch the moderator to a different operational mode.
        
        Args:
            mode: The new operational mode to switch to
        """
        if mode == self.current_mode:
            logger.debug(f"ModeratorAgent {self.agent_id} already in mode {mode}")
            return
        
        old_mode = self.current_mode
        self.current_mode = mode
        
        # Update system prompt unless custom prompt was provided
        if not self._custom_system_prompt:
            self.system_prompt = self._get_mode_specific_prompt(mode)
        
        logger.info(
            f"ModeratorAgent {self.agent_id} switched from {old_mode} to {mode} mode"
        )
    
    def get_current_mode(self) -> ModeratorMode:
        """Get the current operational mode.
        
        Returns:
            Current moderator mode
        """
        return self.current_mode
    
    def _get_mode_specific_prompt(self, mode: ModeratorMode) -> str:
        """Get the system prompt for a specific mode.
        
        Args:
            mode: The mode to get the prompt for
            
        Returns:
            System prompt string for the specified mode
        """
        base_moderator_identity = (
            f"You are the impartial Moderator of 'Virtual Agora' (v{self.PROMPT_VERSION}). "
            "Your role is NOT to have an opinion on the topic. You must remain neutral and "
            "process-oriented at all times. "
        )
        
        if mode == "facilitation":
            return base_moderator_identity + (
                "Your responsibilities in FACILITATION mode are: "
                "1. Facilitate the creation of a discussion agenda by requesting proposals and tallying votes from agents. "
                "2. Announce the current topic and turn order. "
                "3. Ensure all agents' comments are relevant to the current sub-topic. "
                "4. Summarize discussion rounds. "
                "5. Conduct polls to decide when a topic is finished. "
                "If a vote to conclude passes, you MUST offer the dissenting voters a chance for 'final considerations.' "
                "You must communicate clearly. When a structured output like JSON is required, "
                "you must adhere to it strictly."
            )
        
        elif mode == "synthesis":
            return base_moderator_identity + (
                "Your responsibilities in SYNTHESIS mode are: "
                "1. Analyze natural language votes from agents and synthesize them into ranked agendas. "
                "2. Break ties fairly and transparently. "
                "3. Create comprehensive, agent-agnostic summaries of concluded topics. "
                "4. Maintain focus on key insights while managing context length. "
                "5. Ensure summaries are concise but complete. "
                "Always output structured data in the exact JSON format requested. "
                "Never include agent attributions in summaries - focus on ideas, not sources."
            )
        
        elif mode == "writer":
            return base_moderator_identity + (
                "Your responsibilities in WRITER mode are: "
                "1. Act as 'The Writer' to generate structured final reports. "
                "2. Review all topic summaries and define logical report structures. "
                "3. Generate professional, comprehensive content for each report section. "
                "4. Maintain consistent tone and style throughout the report. "
                "5. Reference specific discussions appropriately without attribution. "
                "6. Ensure reports synthesize insights from the entire session. "
                "Focus on the substance of discussions and their implications. "
                "Write in a professional, analytical tone suitable for stakeholders."
            )
        
        else:
            # Fallback to base prompt
            return base_moderator_identity + (
                "Your core responsibilities are process-oriented facilitation and content synthesis. "
                "Maintain neutrality and focus on procedural tasks rather than topic opinions."
            )
    
    def _get_default_system_prompt(self) -> str:
        """Override parent method to return mode-specific prompt.
        
        Returns:
            Mode-specific system prompt
        """
        return self._get_mode_specific_prompt(self.current_mode)
    
    def generate_json_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        model_class: Optional[BaseModel] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a JSON response and validate it against a schema.
        
        Args:
            prompt: The prompt to respond to
            context_messages: Previous messages for context
            expected_schema: JSON schema to validate against
            model_class: Pydantic model class for validation
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Validated JSON response as dictionary
            
        Raises:
            ValueError: If JSON is invalid or doesn't match schema
            ValidationError: If Pydantic validation fails
        """
        # Add JSON formatting instruction to prompt
        json_prompt = prompt + (
            "\n\nYou must respond with valid JSON only. "
            "Do not include any additional text or explanations outside the JSON structure."
        )
        
        response_text = self.generate_response(json_prompt, context_messages, **kwargs)
        
        try:
            # Extract JSON from response (handle potential extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON structure found in response")
            
            json_text = response_text[json_start:json_end]
            json_data = json.loads(json_text)
            
            # Validate with Pydantic model if provided
            if model_class:
                validated_data = model_class(**json_data)
                json_data = validated_data.model_dump()
            
            # Validate with JSON schema if provided
            elif expected_schema:
                # Basic schema validation (could be enhanced with jsonschema library)
                self._validate_json_schema(json_data, expected_schema)
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated valid JSON response"
            )
            
            return json_data
            
        except json.JSONDecodeError as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} generated invalid JSON: {e}",
                extra={"response_text": response_text}
            )
            raise ValueError(f"Invalid JSON response: {e}")
        
        except ValidationError as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} JSON validation failed: {e}",
                extra={"json_data": json_data}
            )
            raise
    
    def _validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Basic JSON schema validation.
        
        Args:
            data: JSON data to validate
            schema: Schema to validate against
            
        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from JSON response")
        
        # Check field types
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get("type")
                value = data[field]
                
                if expected_type == "array" and not isinstance(value, list):
                    raise ValueError(f"Field '{field}' should be an array")
                elif expected_type == "string" and not isinstance(value, str):
                    raise ValueError(f"Field '{field}' should be a string")
                elif expected_type == "object" and not isinstance(value, dict):
                    raise ValueError(f"Field '{field}' should be an object")
    
    def parse_agenda_json(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        **kwargs
    ) -> List[str]:
        """Generate and parse an agenda from agent votes.
        
        Args:
            prompt: Prompt containing voting information
            context_messages: Previous messages for context
            **kwargs: Additional LLM parameters
            
        Returns:
            List of topics in ranked order
            
        Raises:
            ValueError: If agenda format is invalid
        """
        try:
            json_response = self.generate_json_response(
                prompt,
                context_messages,
                model_class=AgendaResponse,
                **kwargs
            )
            
            agenda = json_response["proposed_agenda"]
            
            logger.info(
                f"ModeratorAgent {self.agent_id} parsed agenda with {len(agenda)} topics"
            )
            
            return agenda
            
        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to parse agenda: {e}"
            )
            raise ValueError(f"Failed to parse agenda: {e}")
    
    def generate_report_structure(
        self,
        topic_summaries: List[str],
        session_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate a structured report outline from topic summaries.
        
        Args:
            topic_summaries: List of topic summary texts
            session_context: Optional session metadata
            
        Returns:
            List of report section titles
            
        Raises:
            ValueError: If report structure generation fails
        """
        # Switch to writer mode for this operation
        original_mode = self.current_mode
        self.set_mode("writer")
        
        try:
            # Construct prompt for report structure
            prompt = (
                "Based on the following topic summaries from our Virtual Agora session, "
                "define a logical structure for a comprehensive final report. "
                "Consider the flow of ideas, key themes, and stakeholder interests.\n\n"
                "Topic Summaries:\n"
            )
            
            for i, summary in enumerate(topic_summaries, 1):
                prompt += f"{i}. {summary}\n\n"
            
            prompt += (
                "Generate a JSON response with an ordered list of report section titles "
                "that would best organize these insights into a coherent final report."
            )
            
            json_response = self.generate_json_response(
                prompt,
                model_class=ReportStructure
            )
            
            sections = json_response["report_sections"]
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated report structure with {len(sections)} sections"
            )
            
            return sections
            
        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to generate report structure: {e}"
            )
            raise ValueError(f"Failed to generate report structure: {e}")
        
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def generate_section_content(
        self,
        section_title: str,
        topic_summaries: List[str],
        section_context: Optional[str] = None
    ) -> str:
        """Generate content for a specific report section.
        
        Args:
            section_title: Title of the section to generate
            topic_summaries: All topic summaries for reference
            section_context: Optional context about section purpose
            
        Returns:
            Generated section content
        """
        # Ensure we're in writer mode
        original_mode = self.current_mode
        self.set_mode("writer")
        
        try:
            prompt = (
                f"Generate comprehensive content for the report section titled: '{section_title}'\n\n"
                "Base your content on these topic summaries from the Virtual Agora session:\n\n"
            )
            
            for i, summary in enumerate(topic_summaries, 1):
                prompt += f"Topic {i}:\n{summary}\n\n"
            
            if section_context:
                prompt += f"Section Context: {section_context}\n\n"
            
            prompt += (
                "Write professional, analytical content that synthesizes relevant insights "
                "for this section. Focus on substance and implications rather than process. "
                "Do not attribute ideas to specific agents - present ideas as collective insights."
            )
            
            content = self.generate_response(prompt)
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated content for section '{section_title}'"
            )
            
            return content
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def update_conversation_context(self, key: str, value: Any) -> None:
        """Update the conversation context for cross-mode continuity.
        
        Args:
            key: Context key
            value: Context value
        """
        self.conversation_context[key] = value
        logger.debug(
            f"ModeratorAgent {self.agent_id} updated context key '{key}'"
        )
    
    def get_conversation_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the conversation context.
        
        Args:
            key: Context key to retrieve
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        return self.conversation_context.get(key, default)
    
    def clear_conversation_context(self) -> None:
        """Clear all conversation context."""
        self.conversation_context.clear()
        logger.info(f"ModeratorAgent {self.agent_id} cleared conversation context")
    
    def validate_neutrality(self, response: str) -> bool:
        """Validate that a response maintains moderator neutrality.
        
        This is a basic implementation that checks for opinion indicators.
        Could be enhanced with more sophisticated analysis.
        
        Args:
            response: Response text to validate
            
        Returns:
            True if response appears neutral, False otherwise
        """
        # Opinion indicators to avoid (using word boundaries for more precise matching)
        opinion_phrases = [
            "i think", "i believe", "in my opinion", "i feel",
            "personally", "i would say", "i agree", "i disagree",
            "that's wrong", "that's right", "obviously", "clearly this shows",
            "clearly wrong", "clearly right"
        ]
        
        response_lower = response.lower()
        
        for phrase in opinion_phrases:
            # Check if phrase appears as whole words (at beginning, middle, or end)
            padded_response = f" {response_lower} "
            padded_phrase = f" {phrase} "
            
            if (padded_phrase in padded_response or 
                response_lower.startswith(f"{phrase} ") or
                response_lower.startswith(f"{phrase},") or
                response_lower.endswith(f" {phrase}") or
                response_lower.endswith(f" {phrase}.")):
                logger.warning(
                    f"ModeratorAgent {self.agent_id} response contains opinion indicator: '{phrase}'"
                )
                return False
        
        return True
    
    def generate_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        validate_neutrality: bool = True,
        **kwargs
    ) -> str:
        """Override parent method to add neutrality validation.
        
        Args:
            prompt: The prompt to respond to
            context_messages: Previous messages for context
            validate_neutrality: Whether to validate response neutrality
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If neutrality validation fails
        """
        response = super().generate_response(prompt, context_messages, **kwargs)
        
        if validate_neutrality and not self.validate_neutrality(response):
            logger.error(
                f"ModeratorAgent {self.agent_id} generated non-neutral response"
            )
            # For now, just log the warning. In production, might want to regenerate
            # or implement more sophisticated handling
        
        return response


# Factory functions for common moderator configurations

def create_moderator_agent(
    agent_id: str,
    llm: BaseChatModel,
    mode: ModeratorMode = "facilitation",
    **kwargs
) -> ModeratorAgent:
    """Factory function to create a ModeratorAgent with standard configuration.
    
    Args:
        agent_id: Unique identifier for the agent
        llm: LangChain chat model (should be gemini-1.5-pro for optimal performance)
        mode: Initial operational mode
        **kwargs: Additional arguments for ModeratorAgent
        
    Returns:
        Configured ModeratorAgent instance
    """
    return ModeratorAgent(
        agent_id=agent_id,
        llm=llm,
        mode=mode,
        **kwargs
    )


def create_gemini_moderator(
    agent_id: str = "moderator",
    mode: ModeratorMode = "facilitation",
    enable_error_handling: bool = True,
    **llm_kwargs
) -> ModeratorAgent:
    """Factory function to create a ModeratorAgent with Gemini 1.5 Pro.
    
    This function requires google-generativeai to be installed and configured.
    
    Args:
        agent_id: Unique identifier for the agent
        mode: Initial operational mode
        **llm_kwargs: Arguments for the Gemini LLM (api_key, temperature, etc.)
        
    Returns:
        ModeratorAgent configured with Gemini 1.5 Pro
        
    Raises:
        ImportError: If google-generativeai is not available
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            **llm_kwargs
        )
        
        return ModeratorAgent(
            agent_id=agent_id,
            llm=llm,
            mode=mode,
            enable_error_handling=enable_error_handling
        )
        
    except ImportError as e:
        logger.error("Failed to create Gemini moderator: google-generativeai not available")
        raise ImportError(
            "google-generativeai package required for Gemini moderator. "
            "Install with: pip install langchain-google-genai"
        ) from e