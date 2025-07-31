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
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, ValidationError

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.state.schema import Message, VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)

# ModeratorMode removed in v1.3 - moderator now focuses only on facilitation

# JSON schema definitions for structured outputs
AGENDA_SCHEMA = {
    "type": "object",
    "properties": {
        "proposed_agenda": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered list of topics for discussion",
        }
    },
    "required": ["proposed_agenda"],
}


class AgendaResponse(BaseModel):
    """Pydantic model for agenda response validation."""

    proposed_agenda: List[str]


class ModeratorAgent(LLMAgent):
    """Specialized Moderator agent for Virtual Agora discussions (v1.3).

    This agent inherits from LLMAgent and focuses on process facilitation:
    - Discussion facilitation and turn management
    - Structured output generation (JSON)
    - Process-oriented prompting that avoids opinion expression
    - Conversation context management
    - Relevance enforcement and agent muting

    Note: In v1.3, synthesis and report writing have been moved to specialized agents
    (SummarizerAgent, TopicReportAgent, EcclesiaReportAgent)
    """

    PROMPT_VERSION = "1.0"

    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        system_prompt: Optional[str] = None,
        enable_error_handling: bool = True,
        max_retries: int = 3,
        fallback_llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        turn_timeout_seconds: int = 300,
        relevance_threshold: float = 0.7,  # Story 3.4: Minimum relevance score
        warning_threshold: int = 2,  # Story 3.4: Warnings before muting
        mute_duration_minutes: int = 5,  # Story 3.4: Duration of muting
    ):
        """Initialize the Moderator agent.

        Args:
            agent_id: Unique identifier for the agent
            llm: LangChain chat model instance
            system_prompt: Optional custom system prompt
            enable_error_handling: Whether to enable enhanced error handling
            max_retries: Maximum number of retries for failed operations
            fallback_llm: Optional fallback LLM for error recovery
            tools: Optional list of tools the agent can use
            turn_timeout_seconds: Maximum time to wait for agent response (default: 5 minutes)
            relevance_threshold: Minimum relevance score (0.0-1.0) for messages (Story 3.4)
            warning_threshold: Number of warnings before muting an agent (Story 3.4)
            mute_duration_minutes: How long to mute agents in minutes (Story 3.4)
        """
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
        self.relevance_violations: Dict[str, Dict[str, Any]] = (
            {}
        )  # agent_id -> violation data
        # agent_id -> mute_end_time
        self.muted_agents: Dict[str, datetime] = {}
        self.current_topic_context: Optional[str] = None

        # Initialize with moderator-specific system prompt
        facilitation_prompt = (
            self._get_default_system_prompt() if not system_prompt else system_prompt
        )

        super().__init__(
            agent_id=agent_id,
            llm=llm,
            role="moderator",
            system_prompt=facilitation_prompt,
            enable_error_handling=enable_error_handling,
            max_retries=max_retries,
            fallback_llm=fallback_llm,
            tools=tools,
        )

        logger.info(
            f"Initialized ModeratorAgent {agent_id} with "
            f"prompt_version={self.PROMPT_VERSION}, relevance_threshold={relevance_threshold}"
        )

    # Mode-related methods removed in v1.3 - moderator now focuses only on facilitation

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for moderator facilitation.

        Returns:
            Default system prompt for the moderator
        """
        return (
            f"**Identity:** You are the Moderator, a specialized, neutral reasoning tool for the Virtual Agora (v{self.PROMPT_VERSION}).\n"
            "**Core Directive:** Your sole function is to execute specific process-facilitation tasks with precision and objectivity. You are a tool, not a participant. You have no opinions.\n\n"
            "**Primary Capabilities & Tasks:**\n"
            "1.  **Proposal Compilation:** Given a set of topic proposals from participants, your task is to create a single, deduplicated list of unique agenda items. You will identify and merge similar concepts to produce a clean list.\n"
            '2.  **Vote Synthesis:** Given a list of agenda items and a set of natural language votes from participants, your task is to analyze the preferences and produce a final, rank-ordered agenda. You must break any ties based on objective criteria (e.g., logical flow, relevance to the main theme). Your output for this task **MUST** be a valid JSON object in the format: `{{"proposed_agenda": ["Item A", "Item B", "Item C"]}}`.\n'
            "3.  **Relevance Enforcement:** Given a message from a participant, your task is to evaluate its relevance to the current topic and provide a structured analysis in JSON format.\n\n"
            "**Strict Constraints:**\n"
            "- **Absolute Neutrality:** You must never express opinions, preferences, or agreement/disagreement with the content of the discussion.\n"
            "- **Process-Oriented:** Your responses must be strictly focused on the process task you are given. Do not add conversational filler.\n"
            "- **Format Adherence:** When a specific output format (e.g., JSON) is required, you must adhere to it precisely. Do not include any explanatory text outside the specified format."
        )

    def generate_json_response(
        self,
        prompt: str,
        context_messages: Optional[List[Message]] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        model_class: Optional[BaseModel] = None,
        **kwargs,
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
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

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

            logger.info(f"ModeratorAgent {self.agent_id} generated valid JSON response")

            return json_data

        except json.JSONDecodeError as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} generated invalid JSON: {e}",
                extra={"response_text": response_text},
            )
            raise ValueError(f"Invalid JSON response: {e}")

        except ValidationError as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} JSON validation failed: {e}",
                extra={"json_data": json_data},
            )
            raise

    def _validate_json_schema(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> None:
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
        self, prompt: str, context_messages: Optional[List[Message]] = None, **kwargs
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
                prompt, context_messages, model_class=AgendaResponse, **kwargs
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
                    raise ValueError(
                        f"All agenda items must be strings, got {type(topic)}"
                    )

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
            logger.error(f"ModeratorAgent {self.agent_id} failed to parse agenda: {e}")
            raise ValueError(f"Failed to parse agenda: {e}")

    def collect_proposals(self, agent_proposals: List[Dict[str, str]]) -> List[str]:
        """Collect and deduplicate topic proposals from agent responses.

        In v1.3, this method focuses on pure deduplication without any synthesis.

        Args:
            agent_proposals: List of dicts with 'agent_id' and 'proposals' keys

        Returns:
            List of unique topics extracted from all proposals
        """
        all_topics = []
        topic_sources = {}  # topic -> list of agents who proposed it

        # Extract topics from each agent's proposals
        for proposal_dict in agent_proposals:
            agent_id = proposal_dict.get("agent_id", "unknown")
            proposals_text = proposal_dict.get("proposals", "")

            # Simple extraction - look for numbered lists
            topics_in_response = self._extract_topics_from_response(proposals_text)

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
            f"from {len(agent_proposals)} proposals"
        )

        return unique_topics

    def _extract_topics_from_response(self, response: str) -> List[str]:
        """Extract topic proposals from an agent response.

        Args:
            response: Agent response text

        Returns:
            List of extracted topics
        """
        topics = []
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            # Look for numbered lists (1., 2., etc.) or bullet points
            if any(line.startswith(f"{i}.") for i in range(1, 10)):
                # Remove the number and period
                topic = line.split(".", 1)[1].strip()
                if topic:
                    topics.append(topic)
            elif (
                line.startswith(f"- ")
                or line.startswith(f"• ")
                or line.startswith(f"* ")
            ):
                # Remove bullet point
                topic = line[2:].strip()
                if topic:
                    topics.append(topic)

        return topics

    def synthesize_agenda(
        self, agent_votes: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """Synthesize agent votes into a ranked agenda.

        In v1.3, this method uses the LLM to produce a JSON response with the ranked agenda.

        Args:
            agent_votes: List of dicts with 'agent_id' and 'vote' keys

        Returns:
            Dictionary with 'proposed_agenda' key containing list of topics in ranked order

        Raises:
            ValueError: If agenda synthesis fails
        """
        if not agent_votes:
            raise ValueError("Cannot synthesize agenda from empty votes")
        try:
            # Construct prompt for vote analysis
            prompt_parts = [
                "**Task:** Synthesize Agent Votes into a Final Agenda.",
                "**Objective:** Analyze the provided natural language votes to determine the collective preference for the discussion order. Produce a single, rank-ordered agenda.",
                "",
                "**Input Data:**",
            ]

            for vote_dict in agent_votes:
                agent_id = vote_dict.get("agent_id", "Unknown")
                vote_content = vote_dict.get("vote", "No preference stated.")
                prompt_parts.append(f"- **Agent {agent_id} Vote:** {vote_content}")

            prompt_parts.extend(
                [
                    "",
                    "**Analysis & Tie-Breaking Criteria:**",
                    "1.  **Explicit Ranking:** Prioritize clear numerical or ordered preferences.",
                    "2.  **Implicit Priority:** Infer priority from phrases like 'most importantly' or 'we should start with'.",
                    "3.  **Logical Dependencies:** If one topic is a prerequisite for another, order it first.",
                    "4.  **Relevance & Scope:** If still tied, prioritize topics that are most central to the main theme and have a clear, debatable scope.",
                    "",
                    "**Output Requirement:**",
                    "Your response **MUST** be a single, valid JSON object containing the final ranked agenda. Do not include any other text, explanations, or conversational filler.",
                    '**Format:** `{{"proposed_agenda": ["Final Topic 1", "Final Topic 2", "Final Topic 3", ...]}}`',
                ]
            )

            prompt = "\n".join(prompt_parts)

            # For v1.3, use generate_response which handles LLM invocation properly
            response_text = self.generate_response(prompt)

            # Parse JSON response using LangChain's JsonOutputParser which handles markdown code blocks
            try:
                parser = JsonOutputParser()
                # Create an AIMessage from the response text for the parser
                ai_message = AIMessage(content=response_text)
                result = parser.invoke(ai_message)

                if not isinstance(result, dict) or "proposed_agenda" not in result:
                    raise ValueError("Response missing 'proposed_agenda' key")
                if not isinstance(result["proposed_agenda"], list):
                    raise ValueError("'proposed_agenda' must be a list")

                logger.info(
                    f"ModeratorAgent {self.agent_id} synthesized agenda from {len(agent_votes)} votes"
                )

                return result

            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response_text}")
                # Fallback: return a reasonable default
                return {"proposed_agenda": ["Topic 1", "Topic 2", "Topic 3"]}

        except Exception as e:
            logger.error(
                f"ModeratorAgent {self.agent_id} failed to synthesize agenda: {e}"
            )
            raise ValueError(f"Failed to synthesize agenda: {e}")

    # Report Writing methods removed in v1.3
    # These methods have been moved to specialized agents:
    # - generate_report_structure() → EcclesiaReportAgent
    # - generate_section_content() → EcclesiaReportAgent

    def update_conversation_context(self, key: str, value: Any) -> None:
        """Update the conversation context for cross-mode continuity.

        Args:
            key: Context key
            value: Context value
        """
        self.conversation_context[key] = value
        logger.debug(f"ModeratorAgent {self.agent_id} updated context key '{key}'")

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
            "i think",
            "i believe",
            "in my opinion",
            "i feel",
            "personally",
            "i would say",
            "i agree",
            "i disagree",
            "that's wrong",
            "that's right",
            "obviously",
            "clearly this shows",
            "clearly wrong",
            "clearly right",
        ]

        response_lower = response.lower()

        for phrase in opinion_phrases:
            # Check if phrase appears as whole words (at beginning, middle, or end)
            padded_response = f" {response_lower} "
            padded_phrase = f" {phrase} "

            if (
                padded_phrase in padded_response
                or response_lower.startswith(f"{phrase} ")
                or response_lower.startswith(f"{phrase},")
                or response_lower.endswith(f" {phrase}")
                or response_lower.endswith(f" {phrase}.")
            ):
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
        **kwargs,
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

        logger.info(
            f"ModeratorAgent {self.agent_id} announced topic '{topic}' for round {round_number}"
        )
        return announcement

    def manage_turn_order(self, agent_ids: List[str], randomize: bool = False) -> None:
        """Set up and manage the turn order for discussion rounds.

        Args:
            agent_ids: List of participating agent IDs
            randomize: Whether to randomize the initial order (default: False)
        """
        import random

        if not agent_ids:
            logger.warning(
                f"ModeratorAgent {self.agent_id} received empty agent list for turn order"
            )
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
                    "last_activity": datetime.now(),
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
            logger.warning(
                f"ModeratorAgent {self.agent_id} has no speaking order defined"
            )
            return None

        next_speaker = self.speaking_order[self.current_speaker_index]
        self.current_speaker_index = (self.current_speaker_index + 1) % len(
            self.speaking_order
        )

        logger.debug(
            f"ModeratorAgent {self.agent_id} selected next speaker: {next_speaker}"
        )
        return next_speaker

    def track_participation(
        self, agent_id: str, response_content: str, response_time_seconds: float
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
                "last_activity": datetime.now(),
            }

        metrics = self.participation_metrics[agent_id]

        # Update metrics
        metrics["turns_taken"] += 1
        metrics["total_response_time"] += response_time_seconds
        metrics["average_response_time"] = (
            metrics["total_response_time"] / metrics["turns_taken"]
        )
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
            1
            for agent_id in self.speaking_order
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
                "timeout_rate": 0.0,
            }

        total_agents = len(self.participation_metrics)
        active_agents = sum(
            1
            for metrics in self.participation_metrics.values()
            if metrics["turns_taken"] > 0
        )

        if active_agents > 0:
            avg_response_time = (
                sum(
                    metrics["average_response_time"]
                    for metrics in self.participation_metrics.values()
                    if metrics["turns_taken"] > 0
                )
                / active_agents
            )

            total_words = sum(
                metrics["words_contributed"]
                for metrics in self.participation_metrics.values()
            )

            total_timeouts = sum(
                metrics["timeouts"] for metrics in self.participation_metrics.values()
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
            "detailed_metrics": self.participation_metrics.copy(),
        }

    # Story 3.5: Round and Topic Summarization methods removed in v1.3
    # These methods have been moved to specialized agents:
    # - generate_round_summary() → SummarizerAgent
    # - generate_topic_summary() → SummarizerAgent
    # - _generate_topic_summary_map_reduce() → SummarizerAgent
    # - extract_key_insights() → SummarizerAgent
    # - generate_progressive_summary() → SummarizerAgent

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

        logger.info(
            f"ModeratorAgent {self.agent_id} set topic context: '{self.current_topic_context}'"
        )

    def evaluate_message_relevance(
        self, message_content: str, agent_id: str
    ) -> Dict[str, Any]:
        """Evaluate the relevance of a message to the current topic.

        Args:
            message_content: The content of the message to evaluate
            agent_id: ID of the agent who sent the message

        Returns:
            Dictionary containing relevance score and assessment details
        """
        if not self.current_topic_context:
            logger.warning(
                f"ModeratorAgent {self.agent_id} has no topic context for relevance evaluation"
            )
            return {
                "relevance_score": 1.0,  # No topic context = assume relevant
                "is_relevant": True,
                "reason": "No topic context available for evaluation",
                "topic": None,
            }

        # In v1.3, moderator evaluates relevance without mode switching
        try:
            # Create a structured prompt for relevance evaluation
            evaluation_prompt = (
                f"**Task:** Evaluate Message Relevance.\n"
                f"**Objective:** Analyze the provided message and determine its relevance to the current discussion topic. Your analysis must be objective and strictly based on the provided text.\n\n"
                f"**Current Discussion Topic:** {self.current_topic_context}\n\n"
                f"**Message to Evaluate:**\n---\n{message_content}\n---\n\n"
                f"**Analysis Criteria:**\n"
                f"1.  **Directness:** Does the message directly address the topic? Or is it a tangent?\n"
                f"2.  **Contribution:** Does it add value (e.g., new evidence, a counterargument, a clarifying question) to the specific topic?\n"
                f"3.  **Focus:** Is the core of the message centered on the topic, or is the connection superficial?\n\n"
                f"**Output Requirement:** Respond with a single, valid JSON object containing your structured analysis. Do not include any other text.\n"
                f'**JSON Schema:** `{{"relevance_assessment": {{"relevance_score": float (0.0-1.0), "is_relevant": boolean, "key_points": ["string"], "reason": "string (brief explanation)", "suggestions": "string (brief, actionable feedback for the agent)"}}}}`'
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
                                "description": "Score from 0.0 (irrelevant) to 1.0 (highly relevant)",
                            },
                            "is_relevant": {
                                "type": "boolean",
                                "description": "Whether the message meets the relevance threshold",
                            },
                            "key_points": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Main points that connect to the topic",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Explanation of the relevance assessment",
                            },
                            "suggestions": {
                                "type": "string",
                                "description": "Optional suggestions for staying on topic",
                            },
                        },
                        "required": [
                            "relevance_score",
                            "is_relevant",
                            "key_points",
                            "reason",
                        ],
                    }
                },
                "required": ["relevance_assessment"],
            }

            assessment_json = self.generate_json_response(
                evaluation_prompt, expected_schema=relevance_schema
            )

            assessment = assessment_json["relevance_assessment"]
            assessment["topic"] = self.current_topic_context

            # Override is_relevant based on threshold
            assessment["is_relevant"] = (
                assessment["relevance_score"] >= self.relevance_threshold
            )

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
                "topic": self.current_topic_context,
            }

    def track_relevance_violation(
        self, agent_id: str, message_content: str, relevance_assessment: Dict[str, Any]
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
                "is_muted": False,
            }

        violation_data = self.relevance_violations[agent_id]

        # Record the violation
        violation_record = {
            "timestamp": datetime.now(),
            "message_content": (
                message_content[:200] + "..."
                if len(message_content) > 200
                else message_content
            ),
            "relevance_score": relevance_assessment["relevance_score"],
            "topic": relevance_assessment["topic"],
            "reason": relevance_assessment["reason"],
        }

        violation_data["total_violations"] += 1
        violation_data["violations_history"].append(violation_record)
        violation_data["last_violation"] = datetime.now()

        logger.info(
            f"ModeratorAgent {self.agent_id} tracked relevance violation for {agent_id} "
            f"(total: {violation_data['total_violations']}, score: {relevance_assessment['relevance_score']:.2f})"
        )

    def issue_relevance_warning(
        self, agent_id: str, relevance_assessment: Dict[str, Any]
    ) -> str:
        """Issue a warning to an agent for irrelevant content.

        Args:
            agent_id: ID of the agent to warn
            relevance_assessment: The relevance assessment that triggered the warning

        Returns:
            Warning message to be sent
        """
        if agent_id not in self.relevance_violations:
            logger.error(
                f"ModeratorAgent {self.agent_id} attempted to warn {agent_id} with no violation record"
            )
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
        self, agent_id: str, reason: str = "Multiple relevance violations"
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
                "time_remaining_minutes": 0,
            }

        mute_end_time = self.muted_agents[agent_id]

        if current_time >= mute_end_time:
            # Mute has expired, remove from muted agents
            del self.muted_agents[agent_id]

            # Update violation record
            if agent_id in self.relevance_violations:
                self.relevance_violations[agent_id]["is_muted"] = False

            logger.info(
                f"ModeratorAgent {self.agent_id} unmuted {agent_id} (mute expired)"
            )

            return {
                "is_muted": False,
                "mute_end_time": None,
                "time_remaining_minutes": 0,
            }

        time_remaining = mute_end_time - current_time
        time_remaining_minutes = time_remaining.total_seconds() / 60

        return {
            "is_muted": True,
            "mute_end_time": mute_end_time,
            "time_remaining_minutes": time_remaining_minutes,
        }

    def process_message_for_relevance(
        self, message_content: str, agent_id: str
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
                "relevance_assessment": None,
            }

        # Evaluate message relevance
        relevance_assessment = self.evaluate_message_relevance(
            message_content, agent_id
        )

        if relevance_assessment["is_relevant"]:
            # Message is relevant, allow it
            return {
                "action": "allowed",
                "reason": "relevant_content",
                "message": None,
                "mute_status": mute_status,
                "relevance_assessment": relevance_assessment,
            }

        # Message is not relevant, track violation
        self.track_relevance_violation(agent_id, message_content, relevance_assessment)

        violation_data = self.relevance_violations[agent_id]
        warnings_issued = violation_data["warnings_issued"]

        if warnings_issued < self.warning_threshold:
            # Issue warning
            warning_message = self.issue_relevance_warning(
                agent_id, relevance_assessment
            )
            return {
                "action": "warned",
                "reason": "irrelevant_content",
                "message": warning_message,
                "mute_status": mute_status,
                "relevance_assessment": relevance_assessment,
                "warnings_issued": warnings_issued + 1,
                "warnings_remaining": self.warning_threshold - (warnings_issued + 1),
            }
        else:
            # Mute the agent
            mute_message = self.mute_agent(
                agent_id, f"Exceeded {self.warning_threshold} relevance warnings"
            )
            return {
                "action": "muted",
                "reason": "excessive_violations",
                "message": mute_message,
                "mute_status": self.check_agent_mute_status(agent_id),
                "relevance_assessment": relevance_assessment,
                "total_violations": violation_data["total_violations"],
            }

    def get_relevance_enforcement_summary(self) -> Dict[str, Any]:
        """Get a summary of relevance enforcement activity.

        Returns:
            Dictionary containing enforcement statistics and status
        """
        total_violations = sum(
            data["total_violations"] for data in self.relevance_violations.values()
        )

        total_warnings = sum(
            data["warnings_issued"] for data in self.relevance_violations.values()
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
            "violation_details": self.relevance_violations.copy(),
        }

    # Story 3.6: Topic Conclusion Polling methods

    def initiate_conclusion_poll(
        self, topic: str, eligible_voters: List[str], poll_duration_minutes: int = 5
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
            "options": ["continue", "conclude"],
        }

        # Store poll in conversation context
        self.update_conversation_context(f"active_poll_{poll_id}", poll_data)

        # In v1.3, moderator is always in facilitation mode
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
            "status": "initiated",
        }

    def cast_vote(
        self,
        poll_id: str,
        voter_id: str,
        vote_choice: str,
        reasoning: Optional[str] = None,
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
                "message": f"❌ Poll {poll_id} does not exist or has expired.",
            }

        # Check if poll is still active
        current_time = datetime.now()
        if current_time > poll_data["end_time"]:
            poll_data["status"] = "expired"
            self.update_conversation_context(poll_key, poll_data)
            return {
                "success": False,
                "error": "poll_expired",
                "message": f"❌ Poll {poll_id} has expired. Voting is no longer allowed.",
            }

        # Check if voter is eligible
        if voter_id not in poll_data["eligible_voters"]:
            return {
                "success": False,
                "error": "voter_not_eligible",
                "message": f"❌ {voter_id} is not eligible to vote in this poll.",
            }

        # Check if voter has already voted
        if voter_id in poll_data["votes_cast"]:
            return {
                "success": False,
                "error": "already_voted",
                "message": f"❌ {voter_id} has already voted in this poll. Only one vote per participant is allowed.",
            }

        # Validate vote choice
        valid_choices = poll_data["options"]
        if vote_choice.lower() not in valid_choices:
            return {
                "success": False,
                "error": "invalid_choice",
                "message": f"❌ Invalid vote choice '{vote_choice}'. Valid options are: {', '.join(valid_choices)}.",
            }

        # Cast the vote
        vote_data = {
            "choice": vote_choice.lower(),
            "reasoning": reasoning or "No reasoning provided",
            "timestamp": current_time,
            "voter_id": voter_id,
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
            vote_confirmation += (
                "📊 All votes have been received. The poll will now be tallied."
            )
        else:
            remaining_voters = [
                v
                for v in poll_data["eligible_voters"]
                if v not in poll_data["votes_cast"]
            ]
            vote_confirmation += (
                f"⏳ Still waiting for votes from: {', '.join(remaining_voters)}"
            )

        logger.info(
            f"ModeratorAgent {self.agent_id} recorded vote from {voter_id} in poll {poll_id}: "
            f"{vote_choice} ({votes_received}/{poll_data['required_votes']} votes)"
        )

        return {
            "success": True,
            "message": vote_confirmation,
            "poll_complete": poll_complete,
            "votes_received": votes_received,
            "votes_required": poll_data["required_votes"],
        }

    def handle_minority_considerations(
        self, poll_results: Dict[str, Any], topic: str
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
            voter_id
            for voter_id, vote_data in votes_cast.items()
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
            return {"exists": False, "error": f"Poll {poll_id} not found"}

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
            voter
            for voter in poll_data["eligible_voters"]
            if voter not in poll_data["votes_cast"]
        ]

        time_remaining = max(
            0, (poll_data["end_time"] - current_time).total_seconds() / 60
        )

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
            "options": poll_data["options"],
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

                if poll_status["exists"] and poll_status["status"] in [
                    "active",
                    "ready_for_tally",
                ]:
                    active_polls.append(
                        {
                            "poll_id": poll_id,
                            "topic": poll_status["topic"],
                            "status": poll_status["status"],
                            "votes_progress": f"{poll_status['votes_cast']}/{poll_status['votes_required']}",
                            "time_remaining": f"{poll_status['time_remaining_minutes']:.1f} min",
                            "remaining_voters": poll_status["remaining_voters"],
                        }
                    )

        return active_polls

    # Story 3.7: Minority Considerations Management methods

    async def collect_minority_consideration(
        self, voter_id: str, topic: str, state: VirtualAgoraState
    ) -> Optional[str]:
        """Collect final consideration from a minority voter.

        Args:
            voter_id: ID of the agent who voted in minority
            topic: The topic being concluded
            state: Current VirtualAgoraState

        Returns:
            The minority voter's final consideration, or None if failed
        """
        try:
            # Create a prompt for the minority voter
            prompt = (
                f"As a participant who voted against the majority to conclude the topic '{topic}', "
                f"you have the opportunity to share any final considerations, concerns, or important "
                f"points that should be noted before we proceed. This ensures all perspectives are "
                f"properly represented in our final summary.\n\n"
                f"Please share your final thoughts on this topic. Take the time you need to express your complete perspective."
            )

            # This would normally call the specific agent - for now return a placeholder
            # In real implementation, this would use the agent communication system
            consideration = (
                f"[Minority consideration from {voter_id} on topic '{topic}']"
            )

            logger.info(f"Collected minority consideration from {voter_id}")
            return consideration

        except Exception as e:
            logger.error(
                f"Failed to collect minority consideration from {voter_id}: {e}"
            )
            return None

    async def request_agenda_modification(
        self,
        agent_id: str,
        current_queue: List[str],
        completed_topics: List[str],
        state: VirtualAgoraState,
    ) -> Optional[str]:
        """Request agenda modification suggestions from an agent.

        Args:
            agent_id: ID of the agent to request modifications from
            current_queue: Current list of remaining topics
            completed_topics: List of topics already completed
            state: Current VirtualAgoraState

        Returns:
            Agent's modification suggestion, or None if failed
        """
        try:
            prompt = (
                f"Based on our discussions so far, we have completed these topics: "
                f"{', '.join(completed_topics) if completed_topics else 'None yet'}\n\n"
                f"The remaining topics in our agenda are:\n"
            )

            for i, topic in enumerate(current_queue, 1):
                prompt += f"{i}. {topic}\n"

            prompt += (
                f"\n\nBased on the insights gained from our previous discussions, "
                f"would you like to suggest any modifications to the remaining agenda? "
                f"You can suggest:\n"
                f"- Adding new topics that emerged from our discussions\n"
                f"- Removing topics that may no longer be relevant\n"
                f"- Combining or splitting existing topics\n\n"
                f"Please provide your suggestions or respond with 'No changes' if you're "
                f"satisfied with the current agenda."
            )

            # This would normally call the specific agent - for now return a placeholder
            # In real implementation, this would use the agent communication system
            suggestion = f"[Agenda modification suggestion from {agent_id}]"

            logger.info(f"Collected agenda modification suggestion from {agent_id}")
            return suggestion

        except Exception as e:
            logger.error(f"Failed to collect agenda modification from {agent_id}: {e}")
            return None

    async def collect_agenda_vote(
        self,
        agent_id: str,
        proposed_topics: List[str],
        state: VirtualAgoraState,
    ) -> Optional[str]:
        """Collect vote on proposed agenda from an agent.

        Args:
            agent_id: ID of the agent to collect vote from
            proposed_topics: List of proposed topics to vote on
            state: Current VirtualAgoraState

        Returns:
            Agent's vote response, or None if failed
        """
        try:
            prompt = (
                f"Please vote on the following proposed agenda by ranking the topics "
                f"in your preferred order of discussion:\n\n"
            )

            for i, topic in enumerate(proposed_topics, 1):
                prompt += f"{i}. {topic}\n"

            prompt += (
                f"\n\nPlease provide your ranking with brief reasoning for your preferences. "
                f"You can reference topics by their numbers or names."
            )

            # This would normally call the specific agent - for now return a placeholder
            # In real implementation, this would use the agent communication system
            vote = f"[Agenda vote from {agent_id}]"

            logger.info(f"Collected agenda vote from {agent_id}")
            return vote

        except Exception as e:
            logger.error(f"Failed to collect agenda vote from {agent_id}: {e}")
            return None

    async def synthesize_agenda_votes(
        self, votes: Dict[str, str], proposed_topics: List[str]
    ) -> Optional[List[str]]:
        """Synthesize agenda votes into final ordered topic queue.

        Args:
            votes: Dictionary mapping agent IDs to their vote responses
            proposed_topics: List of topics that were voted on

        Returns:
            Final ordered topic queue, or None if failed
        """
        try:
            # Use existing synthesize_agenda method
            vote_list = list(votes.values())
            voter_ids = list(votes.keys())

            final_agenda = self.synthesize_agenda(proposed_topics, vote_list, voter_ids)

            logger.info(f"Synthesized final agenda from {len(votes)} votes")
            return final_agenda

        except Exception as e:
            logger.error(f"Failed to synthesize agenda votes: {e}")
            return None


# Factory functions for common moderator configurations


def create_moderator_agent(
    agent_id: str, llm: BaseChatModel, **kwargs
) -> ModeratorAgent:
    """Factory function to create a ModeratorAgent with standard configuration.

    Args:
        agent_id: Unique identifier for the agent
        llm: LangChain chat model (should be gemini-2.5-pro for optimal performance)
        **kwargs: Additional arguments for ModeratorAgent

    Returns:
        Configured ModeratorAgent instance
    """
    return ModeratorAgent(agent_id=agent_id, llm=llm, **kwargs)


def create_gemini_moderator(
    agent_id: str = "moderator",
    enable_error_handling: bool = True,
    **llm_kwargs,
) -> ModeratorAgent:
    """Factory function to create a ModeratorAgent with Gemini 1.5 Pro.

    This function requires google-generativeai to be installed and configured.

    Args:
        agent_id: Unique identifier for the agent
        **llm_kwargs: Arguments for the Gemini LLM (api_key, temperature, etc.)

    Returns:
        ModeratorAgent configured with Gemini 1.5 Pro

    Raises:
        ImportError: If google-generativeai is not available
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", **llm_kwargs)

        return ModeratorAgent(
            agent_id=agent_id,
            llm=llm,
            enable_error_handling=enable_error_handling,
        )

    except ImportError as e:
        logger.error(
            "Failed to create Gemini moderator: google-generativeai not available"
        )
        raise ImportError(
            "google-generativeai package required for Gemini moderator. "
            "Install with: pip install langchain-google-genai"
        ) from e
