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
            
            # Enhanced validation
            if not agenda:
                raise ValueError("Agenda cannot be empty")
            
            if not isinstance(agenda, list):
                raise ValueError("Agenda must be a list of topics")
            
            # Check for duplicate topics
            seen_topics = set()
            duplicates = []
            for topic in agenda:
                if not isinstance(topic, str):
                    raise ValueError(f"All agenda items must be strings, got {type(topic)}")
                
                normalized_topic = topic.strip().lower()
                if normalized_topic in seen_topics:
                    duplicates.append(topic)
                else:
                    seen_topics.add(normalized_topic)
            
            if duplicates:
                logger.warning(
                    f"ModeratorAgent {self.agent_id} found duplicate topics in agenda: {duplicates}"
                )
                # Remove duplicates while preserving order
                unique_agenda = []
                seen = set()
                for topic in agenda:
                    normalized = topic.strip().lower()
                    if normalized not in seen:
                        unique_agenda.append(topic)
                        seen.add(normalized)
                agenda = unique_agenda
            
            # Log voting data for transparency
            logger.info(
                f"ModeratorAgent {self.agent_id} parsed agenda with {len(agenda)} topics"
            )
            logger.debug(f"Final agenda order: {agenda}")
            
            return agenda
            
        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to parse agenda: {e}"
            )
            raise ValueError(f"Failed to parse agenda: {e}")
    
    def request_topic_proposals(
        self,
        main_topic: str,
        agent_count: int,
        context_messages: Optional[List[Message]] = None
    ) -> str:
        """Generate a prompt requesting topic proposals from agents.
        
        Args:
            main_topic: The main discussion topic
            agent_count: Number of agents to request proposals from
            context_messages: Previous messages for context
            
        Returns:
            Formatted prompt for topic proposal request
        """
        # Switch to synthesis mode for this operation
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            prompt = (
                f"Please propose 3-5 sub-topics for discussion related to '{main_topic}'. "
                f"Each agent should provide thoughtful sub-topics that would facilitate "
                f"meaningful discussion among {agent_count} participants. "
                f"Consider different perspectives and aspects of the main topic. "
                f"Present your proposals clearly as a numbered list."
            )
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated topic proposal request for '{main_topic}'"
            )
            
            return prompt
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def collect_proposals(
        self,
        agent_responses: List[str],
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Collect and deduplicate topic proposals from agent responses.
        
        Args:
            agent_responses: List of agent response texts containing proposals
            agent_ids: Optional list of agent IDs for tracking proposals
            
        Returns:
            Dictionary containing unique topics and metadata
        """
        if agent_ids and len(agent_ids) != len(agent_responses):
            raise ValueError("agent_ids and agent_responses must have same length")
        
        all_topics = []
        topic_sources = {}  # topic -> list of agents who proposed it
        
        # Extract topics from each response
        for i, response in enumerate(agent_responses):
            agent_id = agent_ids[i] if agent_ids else f"agent_{i}"
            
            # Simple extraction - look for numbered lists
            topics_in_response = self._extract_topics_from_response(response)
            
            for topic in topics_in_response:
                # Normalize topic text (strip, title case)
                normalized_topic = topic.strip().title()
                
                if normalized_topic not in topic_sources:
                    topic_sources[normalized_topic] = []
                    all_topics.append(normalized_topic)
                
                topic_sources[normalized_topic].append(agent_id)
        
        # Remove duplicates while preserving order
        unique_topics = []
        seen = set()
        for topic in all_topics:
            if topic not in seen:
                unique_topics.append(topic)
                seen.add(topic)
        
        logger.info(
            f"ModeratorAgent {self.agent_id} collected {len(unique_topics)} unique topics "
            f"from {len(agent_responses)} responses"
        )
        
        return {
            "unique_topics": unique_topics,
            "topic_sources": topic_sources,
            "total_responses": len(agent_responses),
            "total_unique_topics": len(unique_topics)
        }
    
    def _extract_topics_from_response(self, response: str) -> List[str]:
        """Extract topic proposals from an agent response.
        
        Args:
            response: Agent response text
            
        Returns:
            List of extracted topics
        """
        topics = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered lists (1., 2., etc.) or bullet points
            if any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Remove the number and period
                topic = line.split('.', 1)[1].strip()
                if topic:
                    topics.append(topic)
            elif line.startswith(('- ', '• ', '* ')):
                # Remove bullet point
                topic = line[2:].strip()
                if topic:
                    topics.append(topic)
        
        return topics
    
    def request_votes(
        self,
        topics: List[str],
        voting_instructions: Optional[str] = None
    ) -> str:
        """Generate a prompt requesting votes on topic ordering.
        
        Args:
            topics: List of topics to vote on
            voting_instructions: Optional custom voting instructions
            
        Returns:
            Formatted prompt for voting request
        """
        if not topics:
            raise ValueError("Cannot request votes on empty topic list")
        
        # Switch to synthesis mode for this operation
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            prompt = (
                "Please vote on the order of discussion for the following topics. "
                "Rank them in your preferred order from first to last, and provide "
                "brief reasoning for your preferences.\n\n"
                "Topics to vote on:\n"
            )
            
            for i, topic in enumerate(topics, 1):
                prompt += f"{i}. {topic}\n"
            
            prompt += (
                "\nPlease respond with your preferred order (e.g., '1, 3, 2, 4, 5') "
                "and explain your reasoning. Consider which topics would flow "
                "best together and which should be addressed first."
            )
            
            if voting_instructions:
                prompt += f"\n\nAdditional instructions: {voting_instructions}"
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated voting request for {len(topics)} topics"
            )
            
            return prompt
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def synthesize_agenda(
        self,
        topics: List[str],
        votes: List[str],
        voter_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Synthesize agent votes into a ranked agenda.
        
        Args:
            topics: List of topics being voted on
            votes: List of vote responses from agents
            voter_ids: Optional list of voter IDs for logging
            
        Returns:
            List of topics in ranked order
            
        Raises:
            ValueError: If agenda synthesis fails
        """
        if not topics or not votes:
            raise ValueError("Cannot synthesize agenda from empty topics or votes")
        
        # Switch to synthesis mode for this operation
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            # Construct prompt for vote analysis
            prompt = (
                "Analyze the following votes and synthesize them into a ranked agenda. "
                "Consider the preferences expressed by each voter and use fair tie-breaking "
                "when votes are close. Output the result as a JSON object.\n\n"
                "Topics being voted on:\n"
            )
            
            for i, topic in enumerate(topics, 1):
                prompt += f"{i}. {topic}\n"
            
            prompt += "\nVotes received:\n"
            
            for i, vote in enumerate(votes):
                voter_id = voter_ids[i] if voter_ids else f"Voter {i+1}"
                prompt += f"{voter_id}: {vote}\n\n"
            
            prompt += (
                "Analyze these votes to determine the preferred order of discussion. "
                "Break ties fairly by considering the reasoning provided. "
                "Return a JSON object with the topics in ranked order."
            )
            
            # Use existing JSON response generation with validation
            agenda = self.parse_agenda_json(prompt)
            
            logger.info(
                f"ModeratorAgent {self.agent_id} synthesized agenda from {len(votes)} votes"
            )
            
            return agenda
            
        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to synthesize agenda: {e}"
            )
            raise ValueError(f"Failed to synthesize agenda: {e}")
        
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
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