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
        tools: Optional[List[BaseTool]] = None,
        turn_timeout_seconds: int = 300,
        relevance_threshold: float = 0.7,  # Story 3.4: Minimum relevance score
        warning_threshold: int = 2,        # Story 3.4: Warnings before muting
        mute_duration_minutes: int = 5     # Story 3.4: Duration of muting
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
            turn_timeout_seconds: Maximum time to wait for agent response (default: 5 minutes)
            relevance_threshold: Minimum relevance score (0.0-1.0) for messages (Story 3.4)
            warning_threshold: Number of warnings before muting an agent (Story 3.4)
            mute_duration_minutes: How long to mute agents in minutes (Story 3.4)
        """
        self.current_mode = mode
        self.conversation_context: Dict[str, Any] = {}
        self._custom_system_prompt = system_prompt
        
        # Story 3.3: Discussion Round Management attributes
        self.current_round = 0
        self.turn_timeout_seconds = turn_timeout_seconds
        self.participation_metrics: Dict[str, Dict[str, Any]] = {}
        self.speaking_order: List[str] = []
        self.current_speaker_index = 0
        
        # Story 3.4: Relevance Enforcement attributes
        self.relevance_threshold = relevance_threshold
        self.warning_threshold = warning_threshold
        self.mute_duration_minutes = mute_duration_minutes
        self.relevance_violations: Dict[str, Dict[str, Any]] = {}  # agent_id -> violation data
        self.muted_agents: Dict[str, datetime] = {}  # agent_id -> mute_end_time
        self.current_topic_context: Optional[str] = None
        
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
            f"prompt_version={self.PROMPT_VERSION}, relevance_threshold={relevance_threshold}"
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

    # Story 3.3: Discussion Round Management methods
    
    def announce_topic(self, topic: str, round_number: int) -> str:
        """Announce the current topic and round to participants.
        
        Args:
            topic: Current discussion topic
            round_number: Current round number
            
        Returns:
            Announcement message
        """
        announcement = (
            f"📍 **Round {round_number}: {topic}**\n\n"
            f"We are now beginning Round {round_number} of our discussion on '{topic}'. "
            f"Please stay focused on this specific topic and await your turn to speak. "
            f"The current speaking order is: {' → '.join(self.speaking_order)}"
        )
        
        logger.info(f"ModeratorAgent {self.agent_id} announced topic '{topic}' for round {round_number}")
        return announcement
    
    def manage_turn_order(self, agent_ids: List[str], randomize: bool = False) -> None:
        """Set up and manage the turn order for discussion rounds.
        
        Args:
            agent_ids: List of participating agent IDs
            randomize: Whether to randomize the initial order (default: False)
        """
        import random
        
        if not agent_ids:
            logger.warning(f"ModeratorAgent {self.agent_id} received empty agent list for turn order")
            return
        
        self.speaking_order = agent_ids.copy()
        
        if randomize:
            random.shuffle(self.speaking_order)
        
        self.current_speaker_index = 0
        
        # Initialize participation metrics for all agents
        for agent_id in agent_ids:
            if agent_id not in self.participation_metrics:
                self.participation_metrics[agent_id] = {
                    "turns_taken": 0,
                    "total_response_time": 0.0,
                    "average_response_time": 0.0,
                    "timeouts": 0,
                    "words_contributed": 0,
                    "last_activity": datetime.now()
                }
        
        logger.info(
            f"ModeratorAgent {self.agent_id} set turn order: {self.speaking_order}, "
            f"randomized={randomize}"
        )
    
    def get_next_speaker(self) -> Optional[str]:
        """Get the next speaker in the rotation.
        
        Returns:
            Agent ID of next speaker, or None if no speakers available
        """
        if not self.speaking_order:
            logger.warning(f"ModeratorAgent {self.agent_id} has no speaking order defined")
            return None
        
        next_speaker = self.speaking_order[self.current_speaker_index]
        self.current_speaker_index = (self.current_speaker_index + 1) % len(self.speaking_order)
        
        logger.debug(f"ModeratorAgent {self.agent_id} selected next speaker: {next_speaker}")
        return next_speaker
    
    def track_participation(
        self, 
        agent_id: str, 
        response_content: str, 
        response_time_seconds: float
    ) -> None:
        """Track participation metrics for an agent.
        
        Args:
            agent_id: ID of the participating agent
            response_content: Content of the agent's response
            response_time_seconds: Time taken to respond in seconds
        """
        if agent_id not in self.participation_metrics:
            self.participation_metrics[agent_id] = {
                "turns_taken": 0,
                "total_response_time": 0.0,
                "average_response_time": 0.0,
                "timeouts": 0,
                "words_contributed": 0,
                "last_activity": datetime.now()
            }
        
        metrics = self.participation_metrics[agent_id]
        
        # Update metrics
        metrics["turns_taken"] += 1
        metrics["total_response_time"] += response_time_seconds
        metrics["average_response_time"] = metrics["total_response_time"] / metrics["turns_taken"]
        metrics["words_contributed"] += len(response_content.split())
        metrics["last_activity"] = datetime.now()
        
        logger.debug(
            f"ModeratorAgent {self.agent_id} tracked participation for {agent_id}: "
            f"turns={metrics['turns_taken']}, avg_time={metrics['average_response_time']:.1f}s, "
            f"words={metrics['words_contributed']}"
        )
    
    def handle_agent_timeout(self, agent_id: str) -> str:
        """Handle when an agent times out during their turn.
        
        Args:
            agent_id: ID of the agent that timed out
            
        Returns:
            Timeout handling message
        """
        if agent_id in self.participation_metrics:
            self.participation_metrics[agent_id]["timeouts"] += 1
        
        timeout_message = (
            f"⏰ Agent {agent_id} did not respond within the allocated time "
            f"({self.turn_timeout_seconds} seconds). We will continue to the next participant. "
            f"The discussion remains open for {agent_id} to contribute when ready."
        )
        
        logger.warning(
            f"ModeratorAgent {self.agent_id} handled timeout for {agent_id} "
            f"(timeout #{self.participation_metrics.get(agent_id, {}).get('timeouts', 1)})"
        )
        
        return timeout_message
    
    def signal_round_completion(self, round_number: int, topic: str) -> str:
        """Signal that a discussion round has been completed.
        
        Args:
            round_number: The completed round number
            topic: The topic that was discussed
            
        Returns:
            Round completion message
        """
        # Generate participation summary
        total_participants = len(self.speaking_order)
        active_participants = sum(
            1 for agent_id in self.speaking_order 
            if self.participation_metrics.get(agent_id, {}).get("turns_taken", 0) > 0
        )
        
        completion_message = (
            f"✅ **Round {round_number} Complete: {topic}**\n\n"
            f"Participation Summary:\n"
            f"• Total participants: {total_participants}\n"
            f"• Active participants: {active_participants}\n"
            f"• Participation rate: {(active_participants/total_participants*100):.1f}%\n\n"
            f"Moving to next phase of discussion..."
        )
        
        self.current_round += 1
        
        logger.info(
            f"ModeratorAgent {self.agent_id} completed round {round_number} for topic '{topic}' "
            f"with {active_participants}/{total_participants} participation"
        )
        
        return completion_message
    
    def get_participation_summary(self) -> Dict[str, Any]:
        """Get a summary of current participation metrics.
        
        Returns:
            Dictionary containing participation statistics
        """
        if not self.participation_metrics:
            return {
                "total_agents": 0,
                "active_agents": 0,
                "average_response_time": 0.0,
                "total_words": 0,
                "timeout_rate": 0.0
            }
        
        total_agents = len(self.participation_metrics)
        active_agents = sum(
            1 for metrics in self.participation_metrics.values()
            if metrics["turns_taken"] > 0
        )
        
        if active_agents > 0:
            avg_response_time = sum(
                metrics["average_response_time"] 
                for metrics in self.participation_metrics.values()
                if metrics["turns_taken"] > 0
            ) / active_agents
            
            total_words = sum(
                metrics["words_contributed"] 
                for metrics in self.participation_metrics.values()
            )
            
            total_timeouts = sum(
                metrics["timeouts"] 
                for metrics in self.participation_metrics.values()
            )
            total_turns = sum(
                metrics["turns_taken"] 
                for metrics in self.participation_metrics.values()
            )
            timeout_rate = (total_timeouts / max(total_turns, 1)) * 100
        else:
            avg_response_time = 0.0
            total_words = 0
            timeout_rate = 0.0
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "participation_rate": (active_agents / max(total_agents, 1)) * 100,
            "average_response_time": avg_response_time,
            "total_words": total_words,
            "timeout_rate": timeout_rate,
            "current_round": self.current_round,
            "detailed_metrics": self.participation_metrics.copy()
        }

    # Story 3.5: Round and Topic Summarization methods
    
    def generate_round_summary(
        self, 
        round_number: int, 
        topic: str, 
        messages: List[Message],
        participation_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a summary of a completed discussion round.
        
        Args:
            round_number: The round number that was completed
            topic: The topic that was discussed in this round
            messages: List of messages from this round
            participation_summary: Optional participation metrics for the round
            
        Returns:
            Generated round summary
        """
        # Switch to synthesis mode for summarization
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            # Filter messages for this specific round and topic
            round_messages = [
                msg for msg in messages 
                if msg.get("topic") == topic and msg.get("round", 0) == round_number
            ]
            
            if not round_messages:
                logger.warning(
                    f"ModeratorAgent {self.agent_id} found no messages for round {round_number} on topic '{topic}'"
                )
                return f"Round {round_number} on '{topic}' had no recorded contributions."
            
            # Construct context from messages
            messages_context = "\n\n".join([
                f"Agent {msg['speaker_id']}: {msg['content']}"
                for msg in round_messages
            ])
            
            # Add participation context if available
            participation_context = ""
            if participation_summary:
                participation_context = (
                    f"\n\nParticipation Summary:\n"
                    f"- Active participants: {participation_summary.get('active_agents', 'N/A')}\n"
                    f"- Total contributions: {len(round_messages)}\n"
                    f"- Average response time: {participation_summary.get('average_response_time', 0):.1f}s"
                )
            
            prompt = (
                f"Create a concise summary of Round {round_number} discussion on '{topic}'. "
                f"Focus on key insights, main points raised, areas of agreement or disagreement, "
                f"and significant contributions. Do not attribute specific ideas to individual agents - "
                f"present ideas as collective insights from the discussion.\n\n"
                f"Discussion Content:\n{messages_context}"
                f"{participation_context}\n\n"
                f"Please provide a structured summary that captures the essence of this round's discussion."
            )
            
            summary = self.generate_response(prompt)
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated summary for round {round_number} "
                f"on topic '{topic}' with {len(round_messages)} messages"
            )
            
            return summary
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def generate_topic_summary(
        self, 
        topic: str, 
        all_messages: List[Message],
        round_summaries: Optional[List[str]] = None
    ) -> str:
        """Generate a comprehensive summary of all discussion on a specific topic.
        
        Args:
            topic: The topic to summarize
            all_messages: All messages from the entire discussion
            round_summaries: Optional list of round summaries for this topic
            
        Returns:
            Generated topic summary
        """
        # Switch to synthesis mode for summarization
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            # Filter messages for this specific topic
            topic_messages = [
                msg for msg in all_messages 
                if msg.get("topic") == topic
            ]
            
            if not topic_messages:
                logger.warning(
                    f"ModeratorAgent {self.agent_id} found no messages for topic '{topic}'"
                )
                return f"No discussion recorded for topic '{topic}'."
            
            # Use round summaries if available, otherwise use raw messages
            if round_summaries:
                content_context = "\n\n".join([
                    f"Round Summary {i+1}: {summary}" 
                    for i, summary in enumerate(round_summaries)
                ])
                context_type = "round summaries"
            else:
                # Use map-reduce approach for large message sets
                if len(topic_messages) > 10:  # Threshold for map-reduce
                    return self._generate_topic_summary_map_reduce(topic, topic_messages)
                else:
                    content_context = "\n\n".join([
                        f"Agent {msg['speaker_id']}: {msg['content']}"
                        for msg in topic_messages
                    ])
                    context_type = "individual messages"
            
            # Generate comprehensive topic statistics
            unique_contributors = len(set(msg['speaker_id'] for msg in topic_messages))
            total_contributions = len(topic_messages)
            
            prompt = (
                f"Create a comprehensive summary of the entire discussion on '{topic}'. "
                f"This summary should synthesize all insights, identify key themes, "
                f"highlight areas of consensus and disagreement, and extract the most "
                f"significant conclusions reached. Present ideas as collective insights "
                f"without attributing them to specific agents.\n\n"
                f"Discussion Overview:\n"
                f"- Topic: {topic}\n"
                f"- Total contributions: {total_contributions}\n"
                f"- Unique contributors: {unique_contributors}\n"
                f"- Content source: {context_type}\n\n"
                f"Discussion Content:\n{content_context}\n\n"
                f"Please provide a well-structured summary that captures the full scope "
                f"and depth of the discussion on this topic."
            )
            
            summary = self.generate_response(prompt)
            
            logger.info(
                f"ModeratorAgent {self.agent_id} generated topic summary for '{topic}' "
                f"with {total_contributions} messages from {unique_contributors} contributors"
            )
            
            return summary
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def _generate_topic_summary_map_reduce(
        self, 
        topic: str, 
        messages: List[Message]
    ) -> str:
        """Generate topic summary using map-reduce approach for large message sets.
        
        Args:
            topic: The topic being summarized
            messages: List of messages to summarize
            
        Returns:
            Generated summary using map-reduce approach
        """
        # Chunk messages into groups for map phase
        chunk_size = 5  # Messages per chunk
        message_chunks = [
            messages[i:i + chunk_size] 
            for i in range(0, len(messages), chunk_size)
        ]
        
        # Map phase: Generate summary for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(message_chunks):
            chunk_content = "\n\n".join([
                f"Agent {msg['speaker_id']}: {msg['content']}"
                for msg in chunk
            ])
            
            map_prompt = (
                f"Write a concise summary of the following discussion segment on '{topic}'. "
                f"Focus on key points and insights:\n\n{chunk_content}"
            )
            
            chunk_summary = self.generate_response(map_prompt)
            chunk_summaries.append(chunk_summary)
            
            logger.debug(
                f"ModeratorAgent {self.agent_id} generated map summary {i+1}/{len(message_chunks)} "
                f"for topic '{topic}'"
            )
        
        # Reduce phase: Combine chunk summaries into final summary
        combined_summaries = "\n\n".join([
            f"Segment {i+1}: {summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        reduce_prompt = (
            f"The following are summaries of different segments of discussion on '{topic}'. "
            f"Distill these into a single, comprehensive summary that captures the main themes "
            f"and key insights:\n\n{combined_summaries}"
        )
        
        final_summary = self.generate_response(reduce_prompt)
        
        logger.info(
            f"ModeratorAgent {self.agent_id} completed map-reduce summary for topic '{topic}' "
            f"with {len(message_chunks)} chunks"
        )
        
        return final_summary
    
    def extract_key_insights(
        self, 
        topic_summaries: List[str], 
        session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Extract key insights from topic summaries for the final report.
        
        Args:
            topic_summaries: List of topic summary texts
            session_context: Optional session metadata and statistics
            
        Returns:
            Dictionary containing categorized key insights
        """
        # Switch to synthesis mode for analysis
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            if not topic_summaries:
                logger.warning(f"ModeratorAgent {self.agent_id} received empty topic summaries list")
                return {
                    "main_themes": [],
                    "areas_of_consensus": [],
                    "points_of_disagreement": [],
                    "action_items": [],
                    "future_considerations": []
                }
            
            # Combine all topic summaries
            combined_summaries = "\n\n".join([
                f"Topic Summary {i+1}: {summary}" 
                for i, summary in enumerate(topic_summaries)
            ])
            
            # Add session context if available
            context_info = ""
            if session_context:
                context_info = (
                    f"\nSession Context:\n"
                    f"- Total topics discussed: {len(topic_summaries)}\n"
                    f"- Session duration: {session_context.get('duration', 'N/A')}\n"
                    f"- Total participants: {session_context.get('participants', 'N/A')}\n"
                )
            
            # Use structured JSON output for key insights
            insights_schema = {
                "type": "object",
                "properties": {
                    "key_insights": {
                        "type": "object",
                        "properties": {
                            "main_themes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Primary themes that emerged across topics"
                            },
                            "areas_of_consensus": {
                                "type": "array", 
                                "items": {"type": "string"},
                                "description": "Points where participants generally agreed"
                            },
                            "points_of_disagreement": {
                                "type": "array",
                                "items": {"type": "string"}, 
                                "description": "Issues where significant disagreement emerged"
                            },
                            "action_items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Concrete next steps or actions identified"
                            },
                            "future_considerations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Topics or issues to address in future discussions"
                            }
                        },
                        "required": ["main_themes", "areas_of_consensus", "points_of_disagreement", 
                                   "action_items", "future_considerations"]
                    }
                },
                "required": ["key_insights"]
            }
            
            prompt = (
                f"Analyze the following topic summaries from our Virtual Agora session and extract "
                f"key insights organized into the specified categories. Focus on patterns, themes, "
                f"and significant points that emerged across the discussion.\n\n"
                f"Topic Summaries:\n{combined_summaries}"
                f"{context_info}\n\n"
                f"Please extract and categorize the key insights as requested in the JSON format."
            )
            
            insights_json = self.generate_json_response(
                prompt, 
                expected_schema=insights_schema
            )
            
            key_insights = insights_json["key_insights"]
            
            logger.info(
                f"ModeratorAgent {self.agent_id} extracted key insights from {len(topic_summaries)} "
                f"topic summaries: {sum(len(v) for v in key_insights.values())} total insights"
            )
            
            return key_insights
            
        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to extract key insights: {e}"
            )
            # Return empty structure on failure
            return {
                "main_themes": [],
                "areas_of_consensus": [],
                "points_of_disagreement": [],
                "action_items": [],
                "future_considerations": []
            }
        
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def generate_progressive_summary(
        self, 
        new_messages: List[Message],
        existing_summary: Optional[str] = None,
        topic: Optional[str] = None
    ) -> str:
        """Generate or update a summary progressively as new content arrives.
        
        This implements a "refine" approach where an existing summary is updated
        with new information, similar to LangChain's refine summarization.
        
        Args:
            new_messages: New messages to incorporate into the summary
            existing_summary: Existing summary to refine (None for initial summary)
            topic: Optional topic context for focused summarization
            
        Returns:
            Updated summary incorporating new messages
        """
        # Switch to synthesis mode for summarization
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            if not new_messages:
                logger.debug(f"ModeratorAgent {self.agent_id} received no new messages for progressive summary")
                return existing_summary or "No discussion content available."
            
            # Format new messages context
            new_content = "\n\n".join([
                f"Agent {msg['speaker_id']}: {msg['content']}"
                for msg in new_messages
            ])
            
            topic_context = f" on '{topic}'" if topic else ""
            
            if existing_summary:
                # Refine existing summary with new content
                refine_prompt = (
                    f"Update and refine the existing summary with the new discussion content{topic_context}. "
                    f"Integrate new insights while maintaining the coherence of the original summary. "
                    f"Do not attribute ideas to specific agents.\n\n"
                    f"Existing Summary:\n{existing_summary}\n\n"
                    f"New Discussion Content:\n{new_content}\n\n"
                    f"Please provide an updated summary that incorporates the new information."
                )
            else:
                # Generate initial summary
                refine_prompt = (
                    f"Create a concise summary of the following discussion{topic_context}. "
                    f"Focus on key points, insights, and significant contributions. "
                    f"Do not attribute ideas to specific agents.\n\n"
                    f"Discussion Content:\n{new_content}"
                )
            
            updated_summary = self.generate_response(refine_prompt)
            
            logger.debug(
                f"ModeratorAgent {self.agent_id} generated progressive summary "
                f"({len(new_messages)} new messages, existing_summary={'yes' if existing_summary else 'no'})"
            )
            
            return updated_summary
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)

    # Story 3.4: Relevance Enforcement System methods
    
    def set_topic_context(self, topic: str, description: Optional[str] = None) -> None:
        """Set the current topic context for relevance enforcement.
        
        Args:
            topic: The current discussion topic
            description: Optional detailed description of the topic scope
        """
        if description:
            self.current_topic_context = f"{topic}: {description}"
        else:
            self.current_topic_context = topic
        
        logger.info(f"ModeratorAgent {self.agent_id} set topic context: '{self.current_topic_context}'")
    
    def evaluate_message_relevance(
        self, 
        message_content: str, 
        agent_id: str
    ) -> Dict[str, Any]:
        """Evaluate the relevance of a message to the current topic.
        
        Args:
            message_content: The content of the message to evaluate
            agent_id: ID of the agent who sent the message
            
        Returns:
            Dictionary containing relevance score and assessment details
        """
        if not self.current_topic_context:
            logger.warning(f"ModeratorAgent {self.agent_id} has no topic context for relevance evaluation")
            return {
                "relevance_score": 1.0,  # No topic context = assume relevant
                "is_relevant": True,
                "reason": "No topic context available for evaluation",
                "topic": None
            }
        
        # Switch to synthesis mode for relevance analysis
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            # Create a structured prompt for relevance evaluation
            evaluation_prompt = (
                f"Evaluate the relevance of the following message to the current discussion topic. "
                f"Provide a detailed assessment including a relevance score from 0.0 (completely irrelevant) "
                f"to 1.0 (highly relevant).\n\n"
                f"Current Topic: {self.current_topic_context}\n\n"
                f"Message to Evaluate: \"{message_content}\"\n\n"
                f"Consider:\n"
                f"- Does the message directly address the topic?\n"
                f"- Does it provide relevant insights, examples, or perspectives?\n"
                f"- Is it a constructive contribution to the discussion?\n"
                f"- Are any tangential points still meaningfully connected?\n\n"
                f"Respond with a JSON object containing your assessment."
            )
            
            # Define schema for relevance evaluation
            relevance_schema = {
                "type": "object",
                "properties": {
                    "relevance_assessment": {
                        "type": "object",
                        "properties": {
                            "relevance_score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Score from 0.0 (irrelevant) to 1.0 (highly relevant)"
                            },
                            "is_relevant": {
                                "type": "boolean",
                                "description": "Whether the message meets the relevance threshold"
                            },
                            "key_points": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Main points that connect to the topic"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Explanation of the relevance assessment"
                            },
                            "suggestions": {
                                "type": "string",
                                "description": "Optional suggestions for staying on topic"
                            }
                        },
                        "required": ["relevance_score", "is_relevant", "key_points", "reason"]
                    }
                },
                "required": ["relevance_assessment"]
            }
            
            assessment_json = self.generate_json_response(
                evaluation_prompt,
                expected_schema=relevance_schema
            )
            
            assessment = assessment_json["relevance_assessment"]
            assessment["topic"] = self.current_topic_context
            
            # Override is_relevant based on threshold
            assessment["is_relevant"] = assessment["relevance_score"] >= self.relevance_threshold
            
            logger.debug(
                f"ModeratorAgent {self.agent_id} evaluated message relevance for {agent_id}: "
                f"score={assessment['relevance_score']:.2f}, relevant={assessment['is_relevant']}"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to evaluate message relevance: {e}"
            )
            # Return neutral assessment on failure
            return {
                "relevance_score": 0.5,
                "is_relevant": True,  # Err on the side of allowing messages
                "key_points": [],
                "reason": f"Assessment failed: {e}",
                "topic": self.current_topic_context
            }
        
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def track_relevance_violation(
        self, 
        agent_id: str, 
        message_content: str,
        relevance_assessment: Dict[str, Any]
    ) -> None:
        """Track a relevance violation for an agent.
        
        Args:
            agent_id: ID of the agent who violated relevance
            message_content: The irrelevant message content
            relevance_assessment: The relevance assessment results
        """
        if agent_id not in self.relevance_violations:
            self.relevance_violations[agent_id] = {
                "total_violations": 0,
                "warnings_issued": 0,
                "violations_history": [],
                "last_violation": None,
                "is_muted": False
            }
        
        violation_data = self.relevance_violations[agent_id]
        
        # Record the violation
        violation_record = {
            "timestamp": datetime.now(),
            "message_content": message_content[:200] + "..." if len(message_content) > 200 else message_content,
            "relevance_score": relevance_assessment["relevance_score"],
            "topic": relevance_assessment["topic"],
            "reason": relevance_assessment["reason"]
        }
        
        violation_data["total_violations"] += 1
        violation_data["violations_history"].append(violation_record)
        violation_data["last_violation"] = datetime.now()
        
        logger.info(
            f"ModeratorAgent {self.agent_id} tracked relevance violation for {agent_id} "
            f"(total: {violation_data['total_violations']}, score: {relevance_assessment['relevance_score']:.2f})"
        )
    
    def issue_relevance_warning(
        self, 
        agent_id: str, 
        relevance_assessment: Dict[str, Any]
    ) -> str:
        """Issue a warning to an agent for irrelevant content.
        
        Args:
            agent_id: ID of the agent to warn
            relevance_assessment: The relevance assessment that triggered the warning
            
        Returns:
            Warning message to be sent
        """
        if agent_id not in self.relevance_violations:
            logger.error(f"ModeratorAgent {self.agent_id} attempted to warn {agent_id} with no violation record")
            return ""
        
        violation_data = self.relevance_violations[agent_id]
        violation_data["warnings_issued"] += 1
        
        warnings_remaining = self.warning_threshold - violation_data["warnings_issued"]
        
        if warnings_remaining > 0:
            warning_message = (
                f"⚠️ **Relevance Warning for {agent_id}**\n\n"
                f"Your recent message appears to be off-topic for our current discussion: "
                f"'{self.current_topic_context}'\n\n"
                f"**Assessment:** {relevance_assessment['reason']}\n\n"
                f"**Suggestions:** {relevance_assessment.get('suggestions', 'Please focus on the current topic.')}\n\n"
                f"**Warnings remaining:** {warnings_remaining}\n"
                f"**Note:** After {self.warning_threshold} warnings, you will be temporarily muted "
                f"for {self.mute_duration_minutes} minutes to allow the discussion to continue on-topic."
            )
        else:
            # Final warning before muting
            warning_message = (
                f"🚨 **Final Warning for {agent_id}**\n\n"
                f"This is your final warning for off-topic contributions. "
                f"Your next irrelevant message will result in a {self.mute_duration_minutes}-minute "
                f"mute from the discussion.\n\n"
                f"**Current Topic:** {self.current_topic_context}\n"
                f"**Please focus:** {relevance_assessment.get('suggestions', 'Stay focused on the current topic.')}"
            )
        
        logger.info(
            f"ModeratorAgent {self.agent_id} issued warning #{violation_data['warnings_issued']} "
            f"to {agent_id} (threshold: {self.warning_threshold})"
        )
        
        return warning_message
    
    def mute_agent(
        self, 
        agent_id: str, 
        reason: str = "Multiple relevance violations"
    ) -> str:
        """Mute an agent for a specified duration.
        
        Args:
            agent_id: ID of the agent to mute
            reason: Reason for muting
            
        Returns:
            Muting announcement message
        """
        from datetime import timedelta
        
        mute_end_time = datetime.now() + timedelta(minutes=self.mute_duration_minutes)
        self.muted_agents[agent_id] = mute_end_time
        
        # Update violation record
        if agent_id in self.relevance_violations:
            self.relevance_violations[agent_id]["is_muted"] = True
        
        mute_message = (
            f"🔇 **Agent {agent_id} has been temporarily muted**\n\n"
            f"**Reason:** {reason}\n"
            f"**Duration:** {self.mute_duration_minutes} minutes\n"
            f"**Mute ends at:** {mute_end_time.strftime('%H:%M:%S')}\n\n"
            f"During this time, {agent_id} will not be able to participate in the discussion. "
            f"This allows the conversation to continue focused on: '{self.current_topic_context}'\n\n"
            f"**{agent_id}:** Please use this time to review the current topic and prepare "
            f"relevant contributions for when your mute expires."
        )
        
        logger.warning(
            f"ModeratorAgent {self.agent_id} muted {agent_id} for {self.mute_duration_minutes} minutes. "
            f"Reason: {reason}"
        )
        
        return mute_message
    
    def check_agent_mute_status(self, agent_id: str) -> Dict[str, Any]:
        """Check if an agent is currently muted.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            Dictionary containing mute status information
        """
        current_time = datetime.now()
        
        if agent_id not in self.muted_agents:
            return {
                "is_muted": False,
                "mute_end_time": None,
                "time_remaining_minutes": 0
            }
        
        mute_end_time = self.muted_agents[agent_id]
        
        if current_time >= mute_end_time:
            # Mute has expired, remove from muted agents
            del self.muted_agents[agent_id]
            
            # Update violation record
            if agent_id in self.relevance_violations:
                self.relevance_violations[agent_id]["is_muted"] = False
            
            logger.info(f"ModeratorAgent {self.agent_id} unmuted {agent_id} (mute expired)")
            
            return {
                "is_muted": False,
                "mute_end_time": None,
                "time_remaining_minutes": 0
            }
        
        time_remaining = mute_end_time - current_time
        time_remaining_minutes = time_remaining.total_seconds() / 60
        
        return {
            "is_muted": True,
            "mute_end_time": mute_end_time,
            "time_remaining_minutes": time_remaining_minutes
        }
    
    def process_message_for_relevance(
        self, 
        message_content: str, 
        agent_id: str
    ) -> Dict[str, Any]:
        """Process a message through the complete relevance enforcement pipeline.
        
        Args:
            message_content: The message content to process
            agent_id: ID of the agent who sent the message
            
        Returns:
            Dictionary containing processing results and any actions taken
        """
        # Check if agent is currently muted
        mute_status = self.check_agent_mute_status(agent_id)
        if mute_status["is_muted"]:
            return {
                "action": "blocked",
                "reason": "agent_muted",
                "message": f"Message blocked: {agent_id} is muted for {mute_status['time_remaining_minutes']:.1f} more minutes",
                "mute_status": mute_status,
                "relevance_assessment": None
            }
        
        # Evaluate message relevance
        relevance_assessment = self.evaluate_message_relevance(message_content, agent_id)
        
        if relevance_assessment["is_relevant"]:
            # Message is relevant, allow it
            return {
                "action": "allowed",
                "reason": "relevant_content",
                "message": None,
                "mute_status": mute_status,
                "relevance_assessment": relevance_assessment
            }
        
        # Message is not relevant, track violation
        self.track_relevance_violation(agent_id, message_content, relevance_assessment)
        
        violation_data = self.relevance_violations[agent_id]
        warnings_issued = violation_data["warnings_issued"]
        
        if warnings_issued < self.warning_threshold:
            # Issue warning
            warning_message = self.issue_relevance_warning(agent_id, relevance_assessment)
            return {
                "action": "warned",
                "reason": "irrelevant_content", 
                "message": warning_message,
                "mute_status": mute_status,
                "relevance_assessment": relevance_assessment,
                "warnings_issued": warnings_issued + 1,
                "warnings_remaining": self.warning_threshold - (warnings_issued + 1)
            }
        else:
            # Mute the agent
            mute_message = self.mute_agent(
                agent_id, 
                f"Exceeded {self.warning_threshold} relevance warnings"
            )
            return {
                "action": "muted",
                "reason": "excessive_violations",
                "message": mute_message,
                "mute_status": self.check_agent_mute_status(agent_id),
                "relevance_assessment": relevance_assessment,
                "total_violations": violation_data["total_violations"]
            }
    
    def get_relevance_enforcement_summary(self) -> Dict[str, Any]:
        """Get a summary of relevance enforcement activity.
        
        Returns:
            Dictionary containing enforcement statistics and status
        """
        total_violations = sum(
            data["total_violations"] 
            for data in self.relevance_violations.values()
        )
        
        total_warnings = sum(
            data["warnings_issued"] 
            for data in self.relevance_violations.values()
        )
        
        currently_muted = len(self.muted_agents)
        
        agents_with_violations = len(self.relevance_violations)
        
        return {
            "current_topic": self.current_topic_context,
            "relevance_threshold": self.relevance_threshold,
            "warning_threshold": self.warning_threshold,
            "mute_duration_minutes": self.mute_duration_minutes,
            "total_violations": total_violations,
            "total_warnings_issued": total_warnings,
            "agents_with_violations": agents_with_violations,
            "currently_muted_agents": currently_muted,
            "muted_agents_list": list(self.muted_agents.keys()),
            "violation_details": self.relevance_violations.copy()
        }

    # Story 3.6: Topic Conclusion Polling methods
    
    def initiate_conclusion_poll(
        self, 
        topic: str, 
        eligible_voters: List[str],
        poll_duration_minutes: int = 5
    ) -> Dict[str, Any]:
        """Initiate a democratic poll to determine if a topic discussion should conclude.
        
        Args:
            topic: The topic being voted on for conclusion
            eligible_voters: List of agent IDs eligible to vote
            poll_duration_minutes: How long the poll should remain open
            
        Returns:
            Dictionary containing poll details and instructions
        """
        from datetime import timedelta
        import uuid
        
        poll_id = f"poll_{uuid.uuid4().hex[:8]}"
        poll_end_time = datetime.now() + timedelta(minutes=poll_duration_minutes)
        
        poll_data = {
            "poll_id": poll_id,
            "topic": topic,
            "poll_type": "topic_conclusion",
            "start_time": datetime.now(),
            "end_time": poll_end_time,
            "eligible_voters": eligible_voters.copy(),
            "votes_cast": {},  # voter_id -> vote_data
            "status": "active",
            "required_votes": len(eligible_voters),
            "options": ["continue", "conclude"]
        }
        
        # Store poll in conversation context
        self.update_conversation_context(f"active_poll_{poll_id}", poll_data)
        
        # Switch to facilitation mode for polling
        original_mode = self.current_mode
        self.set_mode("facilitation")
        
        try:
            poll_announcement = (
                f"📊 **Topic Conclusion Poll Initiated**\n\n"
                f"**Topic:** {topic}\n"
                f"**Poll ID:** {poll_id}\n"
                f"**Duration:** {poll_duration_minutes} minutes (ends at {poll_end_time.strftime('%H:%M:%S')})\n"
                f"**Eligible Voters:** {', '.join(eligible_voters)}\n\n"
                f"**Question:** Should we conclude the discussion on '{topic}' and move to the next topic?\n\n"
                f"**Voting Options:**\n"
                f"• **continue** - Continue discussing this topic\n"
                f"• **conclude** - Conclude this topic and move forward\n\n"
                f"**Instructions:** Please respond with your vote and brief reasoning. "
                f"Each participant gets one vote. The poll will close automatically when the time expires "
                f"or when all eligible voters have participated.\n\n"
                f"**Vote Format:** 'I vote [continue/conclude] because [your reasoning]'"
            )
            
            logger.info(
                f"ModeratorAgent {self.agent_id} initiated conclusion poll for topic '{topic}' "
                f"with {len(eligible_voters)} eligible voters"
            )
            
            return {
                "poll_id": poll_id,
                "announcement": poll_announcement,
                "poll_data": poll_data,
                "status": "initiated"
            }
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def cast_vote(
        self, 
        poll_id: str, 
        voter_id: str, 
        vote_choice: str, 
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cast a vote in an active conclusion poll.
        
        Args:
            poll_id: ID of the poll to vote in
            voter_id: ID of the agent casting the vote
            vote_choice: The vote choice ('continue' or 'conclude')
            reasoning: Optional reasoning for the vote
            
        Returns:
            Dictionary containing vote result and updated poll status
        """
        poll_key = f"active_poll_{poll_id}"
        poll_data = self.get_conversation_context(poll_key)
        
        if not poll_data:
            return {
                "success": False,
                "error": f"Poll {poll_id} not found",
                "message": f"❌ Poll {poll_id} does not exist or has expired."
            }
        
        # Check if poll is still active
        current_time = datetime.now()
        if current_time > poll_data["end_time"]:
            poll_data["status"] = "expired"
            self.update_conversation_context(poll_key, poll_data)
            return {
                "success": False,
                "error": "poll_expired",
                "message": f"❌ Poll {poll_id} has expired. Voting is no longer allowed."
            }
        
        # Check if voter is eligible
        if voter_id not in poll_data["eligible_voters"]:
            return {
                "success": False,
                "error": "voter_not_eligible",
                "message": f"❌ {voter_id} is not eligible to vote in this poll."
            }
        
        # Check if voter has already voted
        if voter_id in poll_data["votes_cast"]:
            return {
                "success": False,
                "error": "already_voted",
                "message": f"❌ {voter_id} has already voted in this poll. Only one vote per participant is allowed."
            }
        
        # Validate vote choice
        valid_choices = poll_data["options"]
        if vote_choice.lower() not in valid_choices:
            return {
                "success": False,
                "error": "invalid_choice",
                "message": f"❌ Invalid vote choice '{vote_choice}'. Valid options are: {', '.join(valid_choices)}."
            }
        
        # Cast the vote
        vote_data = {
            "choice": vote_choice.lower(),
            "reasoning": reasoning or "No reasoning provided",
            "timestamp": current_time,
            "voter_id": voter_id
        }
        
        poll_data["votes_cast"][voter_id] = vote_data
        votes_received = len(poll_data["votes_cast"])
        
        # Update poll data
        self.update_conversation_context(poll_key, poll_data)
        
        # Check if poll should close (all votes received)
        poll_complete = votes_received >= poll_data["required_votes"]
        
        vote_confirmation = (
            f"✅ **Vote Recorded**\n\n"
            f"**Voter:** {voter_id}\n"
            f"**Choice:** {vote_choice.lower()}\n"
            f"**Reasoning:** {reasoning or 'No reasoning provided'}\n"
            f"**Votes Received:** {votes_received}/{poll_data['required_votes']}\n\n"
        )
        
        if poll_complete:
            vote_confirmation += "📊 All votes have been received. The poll will now be tallied."
        else:
            remaining_voters = [v for v in poll_data["eligible_voters"] if v not in poll_data["votes_cast"]]
            vote_confirmation += f"⏳ Still waiting for votes from: {', '.join(remaining_voters)}"
        
        logger.info(
            f"ModeratorAgent {self.agent_id} recorded vote from {voter_id} in poll {poll_id}: "
            f"{vote_choice} ({votes_received}/{poll_data['required_votes']} votes)"
        )
        
        return {
            "success": True,
            "message": vote_confirmation,
            "poll_complete": poll_complete,
            "votes_received": votes_received,
            "votes_required": poll_data["required_votes"]
        }
    
    def tally_poll_results(self, poll_id: str) -> Dict[str, Any]:
        """Tally the results of a conclusion poll and determine the outcome.
        
        Args:
            poll_id: ID of the poll to tally
            
        Returns:
            Dictionary containing poll results and decision
        """
        poll_key = f"active_poll_{poll_id}"
        poll_data = self.get_conversation_context(poll_key)
        
        if not poll_data:
            return {
                "success": False,
                "error": f"Poll {poll_id} not found"
            }
        
        # Switch to synthesis mode for result analysis
        original_mode = self.current_mode
        self.set_mode("synthesis")
        
        try:
            votes_cast = poll_data["votes_cast"]
            total_votes = len(votes_cast)
            
            if total_votes == 0:
                return {
                    "success": False,
                    "error": "no_votes",
                    "message": "❌ No votes were cast in this poll."
                }
            
            # Count votes
            vote_counts = {}
            for option in poll_data["options"]:
                vote_counts[option] = 0
            
            for vote_data in votes_cast.values():
                choice = vote_data["choice"]
                if choice in vote_counts:
                    vote_counts[choice] += 1
            
            # Determine winner (simple majority)
            conclude_votes = vote_counts.get("conclude", 0)
            continue_votes = vote_counts.get("continue", 0)
            
            # Calculate percentages
            conclude_percentage = (conclude_votes / total_votes) * 100
            continue_percentage = (continue_votes / total_votes) * 100
            
            # Determine decision
            if conclude_votes > continue_votes:
                decision = "conclude"
                winning_percentage = conclude_percentage
                margin = conclude_votes - continue_votes
            elif continue_votes > conclude_votes:
                decision = "continue"
                winning_percentage = continue_percentage
                margin = continue_votes - conclude_votes
            else:
                # Tie - default to continue discussion
                decision = "continue"
                winning_percentage = continue_percentage
                margin = 0
            
            # Mark poll as completed
            poll_data["status"] = "completed"
            poll_data["results"] = {
                "decision": decision,
                "vote_counts": vote_counts,
                "total_votes": total_votes,
                "winning_percentage": winning_percentage,
                "margin": margin,
                "tally_time": datetime.now()
            }
            
            self.update_conversation_context(poll_key, poll_data)
            
            # Generate detailed results message
            results_message = (
                f"📊 **Poll Results: {poll_data['topic']}**\n\n"
                f"**Poll ID:** {poll_id}\n"
                f"**Total Votes:** {total_votes}/{poll_data['required_votes']}\n\n"
                f"**Vote Breakdown:**\n"
                f"• **Conclude:** {conclude_votes} votes ({conclude_percentage:.1f}%)\n"
                f"• **Continue:** {continue_votes} votes ({continue_percentage:.1f}%)\n\n"
                f"**Decision:** {decision.upper()}\n"
                f"**Margin:** {margin} vote{'s' if margin != 1 else ''}\n\n"
            )
            
            # Add reasoning summary
            if votes_cast:
                results_message += "**Voting Reasoning Summary:**\n"
                for voter_id, vote_data in votes_cast.items():
                    results_message += f"• **{voter_id}** ({vote_data['choice']}): {vote_data['reasoning']}\n"
                results_message += "\n"
            
            # Add decision explanation
            if decision == "conclude":
                if margin == 0:
                    results_message += (
                        "**Next Steps:** Despite the tie, the default decision is to continue discussion. "
                        "However, given the split opinion, we will conclude this topic to maintain "
                        "progress while ensuring all perspectives have been heard.\n\n"
                    )
                else:
                    results_message += (
                        f"**Next Steps:** The majority has voted to conclude discussion on '{poll_data['topic']}'. "
                        f"We will now move to topic summarization and transition to the next agenda item.\n\n"
                    )
            else:
                if margin == 0:
                    results_message += (
                        "**Next Steps:** With a tied vote, we will continue the discussion to allow "
                        "for further exploration of the topic and potential consensus building.\n\n"
                    )
                else:
                    results_message += (
                        f"**Next Steps:** The majority has voted to continue discussion on '{poll_data['topic']}'. "
                        f"The discussion will proceed with consideration of the points raised in the voting.\n\n"
                    )
            
            logger.info(
                f"ModeratorAgent {self.agent_id} tallied poll {poll_id}: "
                f"{decision} ({conclude_votes} conclude, {continue_votes} continue)"
            )
            
            return {
                "success": True,
                "decision": decision,
                "vote_counts": vote_counts,
                "total_votes": total_votes,
                "winning_percentage": winning_percentage,
                "margin": margin,
                "message": results_message,
                "poll_data": poll_data
            }
            
        finally:
            # Restore original mode
            self.set_mode(original_mode)
    
    def handle_minority_considerations(
        self, 
        poll_results: Dict[str, Any], 
        topic: str
    ) -> str:
        """Handle minority considerations after a conclusion poll.
        
        Args:
            poll_results: Results from the concluded poll
            topic: The topic that was voted on
            
        Returns:
            Message offering minority final considerations
        """
        if not poll_results.get("success") or poll_results.get("margin", 0) == 0:
            # No minority if poll failed or was tied
            return ""
        
        decision = poll_results["decision"]
        vote_counts = poll_results["vote_counts"]
        
        if decision == "conclude":
            # Those who voted to continue are the minority
            minority_choice = "continue"
            minority_count = vote_counts.get("continue", 0)
        else:
            # Those who voted to conclude are the minority
            minority_choice = "conclude"
            minority_count = vote_counts.get("conclude", 0)
        
        if minority_count == 0:
            # No minority voters
            return ""
        
        # Find minority voters
        poll_data = poll_results.get("poll_data", {})
        votes_cast = poll_data.get("votes_cast", {})
        minority_voters = [
            voter_id for voter_id, vote_data in votes_cast.items()
            if vote_data["choice"] == minority_choice
        ]
        
        if not minority_voters:
            return ""
        
        minority_message = (
            f"🗣️ **Minority Final Considerations**\n\n"
            f"**Topic:** {topic}\n"
            f"**Poll Decision:** {decision.upper()}\n\n"
            f"The following participants voted for '{minority_choice}' and are invited to share "
            f"any final thoughts or concerns before we proceed:\n\n"
            f"**Minority Voters:** {', '.join(minority_voters)}\n\n"
            f"**Instructions:** You have 3 minutes to provide any final considerations, "
            f"alternative perspectives, or important points that should be noted before we "
            f"{'conclude this topic' if decision == 'conclude' else 'continue the discussion'}. "
            f"This ensures all viewpoints are properly represented in our final decision.\n\n"
            f"**Note:** This is not a re-vote, but an opportunity to voice any critical "
            f"points that might influence how we proceed or should be remembered in our summary."
        )
        
        logger.info(
            f"ModeratorAgent {self.agent_id} offered minority final considerations to "
            f"{len(minority_voters)} voters for topic '{topic}'"
        )
        
        return minority_message
    
    def check_poll_status(self, poll_id: str) -> Dict[str, Any]:
        """Check the status of an active or completed poll.
        
        Args:
            poll_id: ID of the poll to check
            
        Returns:
            Dictionary containing current poll status and details
        """
        poll_key = f"active_poll_{poll_id}"
        poll_data = self.get_conversation_context(poll_key)
        
        if not poll_data:
            return {
                "exists": False,
                "error": f"Poll {poll_id} not found"
            }
        
        current_time = datetime.now()
        votes_cast = len(poll_data["votes_cast"])
        votes_required = poll_data["required_votes"]
        
        # Check if poll has expired
        if current_time > poll_data["end_time"] and poll_data["status"] == "active":
            poll_data["status"] = "expired"
            self.update_conversation_context(poll_key, poll_data)
        
        # Check if poll should be auto-completed
        if votes_cast >= votes_required and poll_data["status"] == "active":
            poll_data["status"] = "ready_for_tally"
            self.update_conversation_context(poll_key, poll_data)
        
        remaining_voters = [
            voter for voter in poll_data["eligible_voters"] 
            if voter not in poll_data["votes_cast"]
        ]
        
        time_remaining = max(0, (poll_data["end_time"] - current_time).total_seconds() / 60)
        
        return {
            "exists": True,
            "poll_id": poll_id,
            "topic": poll_data["topic"],
            "status": poll_data["status"],
            "votes_cast": votes_cast,
            "votes_required": votes_required,
            "remaining_voters": remaining_voters,
            "time_remaining_minutes": time_remaining,
            "start_time": poll_data["start_time"],
            "end_time": poll_data["end_time"],
            "options": poll_data["options"]
        }
    
    def get_active_polls(self) -> List[Dict[str, Any]]:
        """Get all currently active polls.
        
        Returns:
            List of active poll summaries
        """
        active_polls = []
        
        # Search through conversation context for active polls
        for key, value in self.conversation_context.items():
            if key.startswith("active_poll_") and isinstance(value, dict):
                poll_id = key.replace("active_poll_", "")
                poll_status = self.check_poll_status(poll_id)
                
                if poll_status["exists"] and poll_status["status"] in ["active", "ready_for_tally"]:
                    active_polls.append({
                        "poll_id": poll_id,
                        "topic": poll_status["topic"],
                        "status": poll_status["status"],
                        "votes_progress": f"{poll_status['votes_cast']}/{poll_status['votes_required']}",
                        "time_remaining": f"{poll_status['time_remaining_minutes']:.1f} min",
                        "remaining_voters": poll_status["remaining_voters"]
                    })
        
        return active_polls


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