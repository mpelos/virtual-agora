"""Node implementations for Virtual Agora v1.3 discussion flow.

This module contains all node functions for the node-centric architecture
where nodes orchestrate specialized agents as tools.
"""

import uuid
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from virtual_agora.state.schema import (
    VirtualAgoraState,
    RoundInfo,
    PhaseTransition,
    HITLState,
    Vote,
    Agenda,
    RoundSummary,
)
from virtual_agora.state.manager import StateManager
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.report_writer_agent import ReportWriterAgent
from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.flow.cycle_detection import CyclePreventionManager
from virtual_agora.ui.console import get_console
from virtual_agora.ui.progress_display import (
    show_mini_progress,
    get_progress_visualizer,
)
from virtual_agora.utils.logging import get_logger
from virtual_agora.ui.human_in_the_loop import (
    get_initial_topic,
    get_agenda_approval,
    get_continuation_approval,
    get_agenda_modifications,
    display_session_status,
)
from virtual_agora.ui.preferences import get_user_preferences
from virtual_agora.ui.components import LoadingSpinner
from virtual_agora.ui.discussion_display import display_agent_response
from virtual_agora.providers.config import ProviderType
import time

logger = get_logger(__name__)


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use as filename or directory name.

    Args:
        text: Text to sanitize
        max_length: Maximum length for the sanitized name

    Returns:
        Sanitized filename safe for all operating systems
    """
    if not text:
        return "unnamed"

    # Remove/replace problematic characters
    # Windows reserved characters: < > : " | ? * \ /
    # Also handle other problematic characters
    sanitized = re.sub(r'[<>:"|?*\\/]', "_", text)

    # Replace parentheses, brackets, and other special characters
    sanitized = re.sub(r"[()[\]{}]", "_", sanitized)

    # Replace multiple spaces with single space, then replace spaces with underscores
    sanitized = re.sub(r"\s+", " ", sanitized.strip())
    sanitized = sanitized.replace(" ", "_")

    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "unnamed"

    # Intelligent truncation at word boundaries
    if len(sanitized) > max_length:
        # Try to cut at underscore (word boundary)
        truncated = sanitized[:max_length]
        last_underscore = truncated.rfind("_")

        if last_underscore > max_length * 0.6:  # If we can preserve at least 60%
            sanitized = truncated[:last_underscore]
        else:
            sanitized = truncated

        # Remove trailing underscore if any
        sanitized = sanitized.rstrip("_")

    # Ensure it's not a Windows reserved name
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    if sanitized.upper() in windows_reserved:
        sanitized = f"{sanitized}_file"

    return sanitized


def create_report_directory_structure(
    session_id: str, topic_name: str = None
) -> tuple[str, str]:
    """Create organized directory structure for reports.

    Args:
        session_id: Session identifier
        topic_name: Optional topic name for topic-specific directories

    Returns:
        Tuple of (session_dir_path, topic_dir_path or session_dir_path)
    """
    # Base reports directory
    reports_base = "reports"
    os.makedirs(reports_base, exist_ok=True)

    # Session directory
    safe_session_id = sanitize_filename(session_id, 30)
    session_dir = os.path.join(reports_base, f"session_{safe_session_id}")
    os.makedirs(session_dir, exist_ok=True)

    if topic_name:
        # Topic-specific directory within session
        safe_topic = sanitize_filename(topic_name, 40)
        topic_dir = os.path.join(session_dir, safe_topic)
        os.makedirs(topic_dir, exist_ok=True)
        return session_dir, topic_dir

    return session_dir, session_dir


def get_provider_type_from_agent_id(agent_id: str) -> ProviderType:
    """Extract provider type from agent ID.

    Agent IDs typically follow patterns like:
    - gpt-4o-1, gpt-4o-2 -> OpenAI
    - claude-3-opus-1 -> Anthropic
    - gemini-2.5-pro-1 -> Google
    - grok-beta-1 -> Grok
    """
    agent_id_lower = agent_id.lower()

    if "gpt" in agent_id_lower or "openai" in agent_id_lower:
        return ProviderType.OPENAI
    elif "claude" in agent_id_lower or "anthropic" in agent_id_lower:
        return ProviderType.ANTHROPIC
    elif "gemini" in agent_id_lower or "google" in agent_id_lower:
        return ProviderType.GOOGLE
    elif "grok" in agent_id_lower:
        return ProviderType.GROK
    else:
        # Default fallback
        return ProviderType.OPENAI


def retry_agent_call(
    agent, state, prompt, context_messages=None, max_attempts=3, base_delay=1.0
):
    """Retry agent calls with exponential backoff and atmospheric effects.

    Args:
        agent: The agent to call
        state: Current state
        prompt: Prompt to send to agent
        context_messages: List of BaseMessage objects for conversation context
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds between attempts

    Returns:
        Response dict or None if all attempts fail
    """
    # Show thinking indicator if in assembly mode
    try:
        from virtual_agora.ui.display_modes import is_assembly_mode
        from virtual_agora.ui.atmospheric import show_thinking

        use_atmospheric = is_assembly_mode()
    except ImportError:
        use_atmospheric = False

    if use_atmospheric:
        thinking_context = show_thinking(agent.agent_id, "formulating response...")
    else:
        thinking_context = None

    try:
        for attempt in range(max_attempts):
            try:
                # Use thinking context if available
                if thinking_context:
                    with thinking_context:
                        response_dict = agent(
                            state, prompt=prompt, context_messages=context_messages
                        )
                else:
                    response_dict = agent(
                        state, prompt=prompt, context_messages=context_messages
                    )
                return response_dict
            except Exception as e:
                logger.warning(
                    f"Agent {agent.agent_id} attempt {attempt + 1}/{max_attempts} failed: {e}"
                )
                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying agent {agent.agent_id} in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_attempts} attempts failed for agent {agent.agent_id}"
                    )
                    return None
    finally:
        # Ensure thinking context is properly closed
        pass


def extract_message_info(msg):
    """Extract speaker and content information from message objects safely.

    Handles both LangChain BaseMessage and Virtual Agora dict formats.

    Args:
        msg: Message object (BaseMessage or dict)

    Returns:
        tuple: (speaker_id, content, metadata_dict)
    """
    if hasattr(msg, "content"):  # LangChain BaseMessage
        speaker_id = getattr(msg, "additional_kwargs", {}).get("speaker_id", "unknown")
        content = msg.content
        metadata = getattr(msg, "additional_kwargs", {})
    else:  # Virtual Agora dict format
        speaker_id = msg.get("speaker_id", "unknown")
        content = msg.get("content", "")
        metadata = {k: v for k, v in msg.items() if k not in ["speaker_id", "content"]}

    return speaker_id, content, metadata


def get_message_attribute(msg, attr_name, default=None):
    """Get attribute from message in a standardized way.

    Args:
        msg: Message object (BaseMessage or dict)
        attr_name: Name of attribute to get
        default: Default value if attribute not found

    Returns:
        Attribute value or default
    """
    if hasattr(msg, "content"):  # LangChain BaseMessage
        return getattr(msg, "additional_kwargs", {}).get(attr_name, default)
    else:  # Virtual Agora dict format
        return msg.get(attr_name, default)


def validate_agent_context(theme, current_topic, topic_summaries, round_messages):
    """Validate that agent context contains all required elements per Phase 2 spec.

    According to spec, agents must receive:
    1. The initial user-provided theme
    2. The specific agenda item being discussed
    3. Summaries from previously concluded topics (if any) to understand what has been resolved
    4. A collection of all compacted summaries from previous rounds
    5. The live, verbatim comments from current round participants

    Args:
        theme: Discussion theme
        current_topic: Current agenda item
        topic_summaries: Previous round summaries
        round_messages: Current round messages

    Returns:
        tuple: (is_valid, missing_elements, completeness_score)
    """
    missing_elements = []

    # Check required elements
    if not theme or theme.strip() == "":
        missing_elements.append("theme")

    if not current_topic or current_topic.strip() == "":
        missing_elements.append("current_topic")

    # Note: topic_summaries and round_messages can be empty lists (valid for early rounds)
    # but they should be provided as lists, not None
    if topic_summaries is None:
        missing_elements.append("topic_summaries")

    if round_messages is None:
        missing_elements.append("round_messages")

    is_valid = len(missing_elements) == 0
    completeness_score = 1.0 - (len(missing_elements) / 4.0)

    return is_valid, missing_elements, completeness_score


def create_langchain_message(
    speaker_role: str, content: str, **metadata
) -> BaseMessage:
    """Create a LangChain message object with Virtual Agora metadata.

    Args:
        speaker_role: 'moderator', 'participant', etc.
        content: Message content
        **metadata: Additional metadata to store

    Returns:
        BaseMessage compatible with LangGraph add_messages reducer
    """
    if speaker_role == "moderator":
        return HumanMessage(content=content, additional_kwargs=metadata)
    else:
        return AIMessage(content=content, additional_kwargs=metadata)


class V13FlowNodes:
    """Container for all v1.3 flow node implementations.

    In v1.3, nodes orchestrate specialized agents as tools rather than
    having a single moderator agent with multiple modes.
    """

    def __init__(
        self,
        specialized_agents: Dict[str, LLMAgent],
        discussing_agents: List[LLMAgent],
        state_manager: StateManager,
    ):
        """Initialize with specialized agents and state manager.

        Args:
            specialized_agents: Dictionary of specialized agents (moderator, summarizer, etc.)
            discussing_agents: List of discussing agent instances
            state_manager: State manager instance
        """
        self.specialized_agents = specialized_agents
        self.discussing_agents = discussing_agents
        self.state_manager = state_manager
        self.context_manager = ContextWindowManager()
        self.cycle_manager = CyclePreventionManager()

    # ===== Phase 0: Initialization Nodes =====

    def config_and_keys_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Load configuration and API keys.

        This node:
        1. Loads .env file for API keys
        2. Parses config.yml
        3. Validates configuration
        4. Initializes logging

        Note: This is typically handled during app initialization,
        so this node mainly validates the setup.
        """
        logger.info("Node: config_and_keys - Validating configuration")
        logger.debug(f"Received state with session_id: {state.get('session_id')}")
        logger.debug(f"State keys: {list(state.keys()) if state else 'No state'}")

        # Configuration is already loaded at this point
        # This node validates and confirms readiness
        # Update the current phase to indicate configuration is complete
        logger.debug("Configuration validated successfully")

        return {"current_phase": 0}

    def agent_instantiation_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Create all agent instances.

        This node:
        1. Creates 5 specialized agents from config
        2. Creates N discussing agents from config
        3. Stores agent references in state

        Note: Agents are already instantiated in __init__,
        this node records their presence in state.
        """
        logger.info("Node: agent_instantiation - Recording agent instances")

        # Store specialized agent IDs
        specialized_agent_ids = {
            "moderator": self.specialized_agents["moderator"].agent_id,
            "summarizer": self.specialized_agents["summarizer"].agent_id,
            "report_writer": self.specialized_agents["report_writer"].agent_id,
        }

        # Store discussing agent information
        discussing_agent_info = {
            agent.agent_id: {
                "id": agent.agent_id,
                "model": getattr(agent, "model_name", "unknown"),
                "provider": getattr(agent, "provider", "unknown"),
                "role": "participant",
            }
            for agent in self.discussing_agents
        }

        # Initialize speaking order
        speaking_order = [agent.agent_id for agent in self.discussing_agents]

        updates = {
            "specialized_agents": specialized_agent_ids,
            "agents": discussing_agent_info,
            "speaking_order": speaking_order,
            "current_phase": 0,
        }

        logger.debug(f"Recorded {len(specialized_agent_ids)} specialized agents")
        logger.debug(f"Recorded {len(discussing_agent_info)} discussing agents")

        return updates

    def get_theme_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL node to get discussion theme.

        This node checks if a theme is already provided, otherwise
        it relies on the application's initial topic collection.
        """
        logger.info("Node: get_theme - Checking for discussion theme")

        # Check if theme already provided
        if state.get("main_topic"):
            logger.info(f"Theme already set: {state['main_topic']}")
            return {
                "phase_history": {
                    "from_phase": -1,
                    "to_phase": 0,
                    "timestamp": datetime.now(),
                    "reason": "Theme provided at session creation",
                    "triggered_by": "user",
                }
            }

        # If no theme is set, this indicates an error in the flow
        # The application should provide the theme during session creation
        logger.error("No theme provided at session creation")
        return {
            "error": "No discussion theme provided",
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "theme_input",
                "prompt_message": "Please provide a discussion theme",
                "options": None,
                "approval_history": state.get("hitl_state", {}).get(
                    "approval_history", []
                ),
            },
        }

    # ===== Phase 1: Agenda Setting Nodes =====

    def agenda_proposal_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Request topic proposals from discussing agents.

        This node:
        1. Prompts each discussing agent for 3-5 topics
        2. Collects all proposals
        3. Updates state with raw proposals
        """
        logger.info("Node: agenda_proposal - Collecting topic proposals")

        theme = state["main_topic"]
        proposals = []

        # Request proposals from each discussing agent
        with LoadingSpinner(
            f"Collecting topic proposals from {len(self.discussing_agents)} agents..."
        ) as spinner:
            for i, agent in enumerate(self.discussing_agents):
                spinner.update(
                    f"Getting proposals from {agent.agent_id} ({i+1}/{len(self.discussing_agents)})"
                )

                prompt = f"""Based on the theme '{theme}', propose 3-5 specific
                sub-topics for discussion. Be concise and specific."""

                try:
                    # Call agent with proper state and prompt
                    response_dict = agent(state, prompt=prompt)

                    # Extract response content
                    messages = response_dict.get("messages", [])
                    if messages:
                        response_content = (
                            messages[-1].content
                            if hasattr(messages[-1], "content")
                            else str(messages[-1])
                        )
                        proposals.append(
                            {"agent_id": agent.agent_id, "proposals": response_content}
                        )
                        logger.info(f"Collected proposals from {agent.agent_id}")
                except Exception as e:
                    logger.error(f"Failed to get proposals from {agent.agent_id}: {e}")
                    proposals.append(
                        {
                            "agent_id": agent.agent_id,
                            "proposals": "Failed to provide proposals",
                            "error": str(e),
                        }
                    )

        updates = {
            "proposed_topics": proposals,
            "current_phase": 1,
            "phase_start_time": datetime.now(),
        }

        logger.info(f"Collected proposals from {len(proposals)} agents")

        return updates

    def collate_proposals_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Moderator deduplicates and compiles proposals.

        This node invokes the ModeratorAgent as a tool to:
        1. Read all proposals
        2. Remove duplicates
        3. Create unified list
        """
        logger.info("Node: collate_proposals - Moderator processing proposals")

        proposals = state["proposed_topics"]
        moderator = self.specialized_agents["moderator"]

        try:
            # Invoke moderator as a tool to collect proposals
            # Pass the proposals list directly as it already has the right format
            unified_list = moderator.collect_proposals(proposals)

            logger.info(f"Moderator compiled {len(unified_list)} unique topics")

        except Exception as e:
            logger.error(f"Failed to collate proposals: {e}")
            # Fallback: extract topics manually
            unified_list = []
            for p in proposals:
                # Simple extraction - look for numbered items
                text = p["proposals"]
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
                        # Remove numbering/bullets
                        topic = line.lstrip("0123456789.-) ").strip()
                        if topic and topic not in unified_list:
                            unified_list.append(topic)

            unified_list = unified_list[:10]  # Limit to 10 topics

        updates = {
            "topic_queue": unified_list,
            "current_phase": 1,  # Move to next phase after collating
        }

        return updates

    def agenda_voting_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Collect votes on agenda ordering.

        Similar to proposal collection but for votes.
        """
        logger.info("Node: agenda_voting - Collecting votes on topics")

        topics = state["topic_queue"]
        votes = []

        # Format topics for presentation
        topics_formatted = "\n".join(
            f"{i+1}. {topic}" for i, topic in enumerate(topics)
        )

        for agent in self.discussing_agents:
            prompt = f"""Vote on your preferred discussion order for these topics:
            {topics_formatted}

            Express your preferences in natural language. You may rank all topics
            or just indicate your top priorities."""

            try:
                # Call agent with proper state and prompt
                response_dict = agent(state, prompt=prompt)

                # Extract vote content
                messages = response_dict.get("messages", [])
                if messages:
                    vote_content = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )
                    vote_obj = {
                        "id": f"vote_{uuid.uuid4().hex[:8]}",
                        "voter_id": agent.agent_id,
                        "phase": 1,  # Agenda voting phase
                        "vote_type": "topic_selection",
                        "choice": vote_content,
                        "timestamp": datetime.now(),
                    }
                    votes.append(vote_obj)
                    logger.info(f"Collected vote from {agent.agent_id}")
            except Exception as e:
                logger.error(f"Failed to get vote from {agent.agent_id}: {e}")
                vote_obj = {
                    "id": f"vote_{uuid.uuid4().hex[:8]}",
                    "voter_id": agent.agent_id,
                    "phase": 1,
                    "vote_type": "topic_selection",
                    "choice": "No preference",
                    "timestamp": datetime.now(),
                    "metadata": {"error": str(e)},
                }
                votes.append(vote_obj)

        updates = {
            "votes": votes,  # Store votes in the proper schema field
            "current_phase": 2,  # Progress to next phase after voting
        }

        return updates

    def synthesize_agenda_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Moderator synthesizes votes into final agenda.

        This node invokes ModeratorAgent to:
        1. Analyze all votes
        2. Break ties
        3. Produce JSON agenda
        """
        logger.info("Node: synthesize_agenda - Moderator creating final agenda")

        votes = state["votes"]  # Use the correct schema field
        topics = state["topic_queue"]
        moderator = self.specialized_agents["moderator"]

        try:
            # Extract vote content from Vote objects
            # Handle case where votes might be nested lists due to reducer behavior
            flat_votes = []
            for v in votes:
                if isinstance(v, list):
                    # If vote is wrapped in a list, flatten it
                    flat_votes.extend(v)
                elif isinstance(v, dict):
                    # Normal vote object
                    flat_votes.append(v)
                else:
                    logger.warning(f"Unexpected vote type: {type(v)}, value: {v}")

            topic_votes = [
                v for v in flat_votes if v.get("vote_type") == "topic_selection"
            ]
            vote_responses = [v["choice"] for v in topic_votes]
            voter_ids = [v["voter_id"] for v in topic_votes]

            # Invoke moderator for synthesis
            # Convert votes to the expected format: List[Dict[str, str]]
            agent_votes = [
                {"agent_id": voter_id, "vote": vote_response}
                for voter_id, vote_response in zip(voter_ids, vote_responses)
            ]
            agenda_json = moderator.synthesize_agenda(agent_votes)

            # Extract the proposed agenda
            if isinstance(agenda_json, dict) and "proposed_agenda" in agenda_json:
                final_agenda = agenda_json["proposed_agenda"]
            else:
                # Fallback if JSON parsing fails
                final_agenda = topics[:5]  # Take first 5 topics

        except Exception as e:
            logger.error(f"Failed to synthesize agenda: {e}")
            # Fallback: use topics in original order
            final_agenda = topics[:5]

        agenda_obj = Agenda(
            topics=final_agenda,
            current_topic_index=0,
            completed_topics=[],
        )

        updates = {
            "agenda": agenda_obj,
            "proposed_agenda": final_agenda,  # For HITL approval
            "current_phase": 3,  # Move to next phase after synthesis
        }

        logger.info(f"Synthesized agenda with {len(final_agenda)} topics")

        return updates

    def agenda_approval_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL node for user agenda approval.

        Uses LangGraph interrupt mechanism to pause execution and wait for user input.
        """
        logger.info("Node: agenda_approval - Requesting user approval")

        proposed_agenda = state.get("proposed_agenda", [])

        # If no proposed agenda, create a fallback to prevent infinite loops
        if not proposed_agenda:
            logger.warning("No proposed agenda, creating fallback with main topic")
            fallback_agenda = [state.get("main_topic", "General Discussion")]
            return {
                "agenda_approved": True,
                "topic_queue": fallback_agenda,
                "final_agenda": fallback_agenda,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                },
            }

        # Get user preferences for auto-approval
        prefs = get_user_preferences()

        # Check for auto-approval conditions (only unanimous consensus)
        if prefs.auto_approve_agenda_on_consensus and state.get(
            "unanimous_agenda_vote", False
        ):
            logger.info("Auto-approving agenda due to unanimous vote")
            return {
                "agenda_approved": True,
                "topic_queue": proposed_agenda,
                "final_agenda": proposed_agenda,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                    "approval_history": state.get("hitl_state", {}).get(
                        "approval_history", []
                    )
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "auto_approved",
                            "timestamp": datetime.now(),
                        }
                    ],
                },
            }

        # Use LangGraph interrupt to pause execution and wait for user input
        logger.info("Interrupting for user agenda approval")
        user_input = interrupt(
            {
                "type": "agenda_approval",
                "proposed_agenda": proposed_agenda,
                "message": "Please review and approve the proposed discussion agenda.",
                "options": ["approve", "edit", "reorder", "reject"],
            }
        )

        # Process user input (this will be called when execution resumes)
        if user_input is None:
            # This should not happen in normal flow, but provides a fallback
            logger.error("No user input received for agenda approval")
            return {
                "agenda_approved": False,
                "agenda_rejected": True,
            }

        action = user_input.get("action", "approve")
        final_agenda = user_input.get("agenda", proposed_agenda)

        if action == "approve":
            logger.info("User approved the agenda")
            return {
                "agenda_approved": True,
                "topic_queue": final_agenda,
                "final_agenda": final_agenda,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                    "approval_history": state.get("hitl_state", {}).get(
                        "approval_history", []
                    )
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "approved",
                            "timestamp": datetime.now(),
                        }
                    ],
                },
            }
        elif action == "edit":
            logger.info("User edited the agenda")
            return {
                "agenda_approved": True,
                "topic_queue": final_agenda,
                "final_agenda": final_agenda,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                    "approval_history": state.get("hitl_state", {}).get(
                        "approval_history", []
                    )
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "edited",
                            "timestamp": datetime.now(),
                        }
                    ],
                },
            }
        else:  # reject or other
            logger.info("User rejected the agenda")
            return {
                "agenda_approved": False,
                "agenda_rejected": True,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                    "approval_history": state.get("hitl_state", {}).get(
                        "approval_history", []
                    )
                    + [
                        {
                            "type": "agenda_approval",
                            "result": "rejected",
                            "timestamp": datetime.now(),
                        }
                    ],
                },
            }

    # ===== Phase 2: Discussion Loop Nodes =====

    def announce_item_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Announce current agenda item.

        Simple announcement, no complex logic.
        """
        logger.debug("=== FLOW DEBUG: Entering announce_item_node ===")
        logger.info(f"topic_queue length: {len(state.get('topic_queue', []))}")
        logger.info(f"active_topic: {state.get('active_topic')}")
        logger.info("Node: announce_item - Announcing current topic")

        # Get current topic
        topic_queue = state.get("topic_queue", [])
        if not topic_queue:
            logger.error("No topics in queue")
            return {
                "last_error": "No topics in queue",
                "error_count": state.get("error_count", 0) + 1,
            }

        # Set active topic if not already set
        if not state.get("active_topic"):
            current_topic = topic_queue[0]
            updates = {
                "active_topic": current_topic,
                "current_round": 0,  # Reset round counter
            }

            logger.info(f"Starting discussion on topic: {current_topic}")

            # Set topic context for moderator to enable proper relevance evaluation
            moderator = self.specialized_agents.get("moderator")
            if moderator:
                moderator.set_topic_context(current_topic)
                logger.info(f"Set moderator topic context: {current_topic}")

            # Moderator announcement can be logged but not stored in state
            announcement = f"We will now begin discussing: {current_topic}"
            logger.info(f"Moderator announcement: {announcement}")

            logger.info(
                f"=== FLOW DEBUG: Exiting announce_item_node with updates: {updates} ==="
            )
            return updates

        # Topic is already active, just confirm it's still current
        result = {"active_topic": state.get("active_topic")}
        logger.info(
            f"=== FLOW DEBUG: Exiting announce_item_node (topic already active) with: {result} ==="
        )
        return result

    def discussion_round_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute one round of discussion.

        This node:
        1. Manages turn rotation
        2. Collects agent responses
        3. Enforces relevance
        """
        logger.debug("=== FLOW DEBUG: Entering discussion_round_node ===")
        logger.debug(f"State keys available: {list(state.keys())}")
        logger.info(f"current_round: {state.get('current_round', 0)}")
        logger.info(f"active_topic: {state.get('active_topic')}")
        logger.info(f"speaking_order: {state.get('speaking_order')}")
        logger.info("Node: discussion_round - Executing discussion round")

        # Ensure we have an active topic
        current_topic = state.get("active_topic")
        if not current_topic:
            logger.error("No active topic set for discussion round")
            return {
                "last_error": "No active topic for discussion",
                "error_count": state.get("error_count", 0) + 1,
            }
        current_round = state.get("current_round", 0) + 1
        theme = state.get("main_topic", "Unknown Topic")
        moderator = self.specialized_agents.get("moderator")

        if not moderator:
            logger.error("No moderator agent available")
            return {
                "last_error": "No moderator agent available",
                "error_count": state.get("error_count", 0) + 1,
            }

        # Get round summaries for context
        round_summaries = state.get("round_summaries", [])
        topic_summaries = [
            s
            for s in round_summaries
            if isinstance(s, dict) and s.get("topic") == current_topic
        ]

        # Rotate speaking order
        speaking_order = state.get("speaking_order", [])
        if not speaking_order:
            # Initialize speaking order if not set
            speaking_order = [agent.agent_id for agent in self.discussing_agents]
            logger.debug(f"Initialized speaking order: {speaking_order}")
        else:
            speaking_order = speaking_order.copy()
        if current_round > 1:
            # Rotate: [A,B,C] -> [B,C,A]
            speaking_order = speaking_order[1:] + [speaking_order[0]]

        # Show round announcement with atmospheric effects if in assembly mode
        try:
            from virtual_agora.ui.display_modes import is_assembly_mode
            from virtual_agora.ui.atmospheric import show_round_announcement

            if is_assembly_mode():
                show_round_announcement(current_round, current_topic, speaking_order)
        except ImportError:
            pass

        # Execute the round
        round_messages = []
        round_id = str(uuid.uuid4())
        round_start = datetime.now()

        for i, agent_id in enumerate(speaking_order):
            # Find the agent
            agent = next(a for a in self.discussing_agents if a.agent_id == agent_id)

            # Validate context completeness before building context
            is_valid, missing_elements, completeness_score = validate_agent_context(
                theme, current_topic, topic_summaries, round_messages
            )

            if not is_valid:
                logger.warning(
                    f"Context validation failed for agent {agent_id}. Missing: {missing_elements}. Completeness: {completeness_score:.2f}"
                )
                # Log for debugging but continue with available context
            else:
                logger.debug(
                    f"Context validation passed for agent {agent_id}. Completeness: 100%"
                )

            # Build context for agent
            context = f"""
            Theme: {theme}
            Current Topic: {current_topic}
            Round: {current_round}
            Your turn: {i + 1}/{len(speaking_order)}
            """

            # Add previous topic conclusions first (from previously discussed topics)
            previous_topic_summaries = state.get("topic_summaries", {})
            topic_conclusions = []
            for key, summary in previous_topic_summaries.items():
                if key.endswith("_conclusion") and not key.startswith(current_topic):
                    # Extract topic name by removing "_conclusion" suffix
                    topic_name = key.replace("_conclusion", "")
                    topic_conclusions.append(f"{topic_name}: {summary}")

            if topic_conclusions:
                context += "\n\nPreviously Concluded Topics:\n"
                for conclusion in topic_conclusions:
                    context += f"- {conclusion}\n"

            # Add previous round summaries
            if topic_summaries:
                context += "\n\nPrevious Round Summaries:\n"
                # Last 3 summaries
                for j, summary in enumerate(topic_summaries[-3:]):
                    # Fix: Use 'summary_text' field as defined in RoundSummary schema
                    summary_text = (
                        summary.get("summary_text", "")
                        if isinstance(summary, dict)
                        else getattr(summary, "summary_text", "")
                    )
                    round_num = (
                        summary.get("round_number", j + 1)
                        if isinstance(summary, dict)
                        else getattr(summary, "round_number", j + 1)
                    )
                    context += f"Round {round_num}: {summary_text}\n"

            # Build context messages from current round for natural conversation flow
            context_messages = []
            if round_messages:
                for msg in round_messages:
                    # Use standardized message extraction
                    speaker_id, content, _ = extract_message_info(msg)
                    # Convert colleague messages to HumanMessage format for assembly-style conversation
                    colleague_message = HumanMessage(
                        content=f"[{speaker_id}]: {content}", name=speaker_id
                    )
                    context_messages.append(colleague_message)

            # Enhanced assembly-style prompt emphasizing democratic deliberation
            assembly_context = f"""
You are participating in a democratic assembly discussing the topic: '{current_topic}'

Assembly Context:
- Theme: {theme}
- Round: {current_round}
- Your speaking position: {i + 1}/{len(speaking_order)}

Other assembly members have shared their perspectives before you. Listen to their viewpoints, then provide your own contribution to this democratic deliberation.

Instructions:
- Be strong in your convictions and opinions, even if others disagree
- Your goal is collaborative discussion toward shared understanding
- Build upon, challenge, or expand on previous speakers' points
- Maintain respectful but firm discourse as befits a democratic assembly

Please provide your thoughts on '{current_topic}'. Provide a substantive contribution (1-2 paragraphs, approximately 4-8 sentences) that engages with previous speakers and advances our democratic deliberation."""

            context += assembly_context

            try:
                # Get agent response with retry logic, passing colleague messages as conversation context
                response_dict = retry_agent_call(
                    agent, state, context, context_messages
                )

                if response_dict is None:
                    # All retry attempts failed
                    response_content = (
                        f"[Failed to get response from {agent_id} after retries]"
                    )
                    # Still display the failure for transparency
                    provider_type = get_provider_type_from_agent_id(agent_id)
                    display_agent_response(
                        agent_id=agent_id,
                        provider=provider_type,
                        content=f"⚠️ Connection failed after multiple attempts",
                        round_number=current_round,
                        topic=current_topic,
                        timestamp=datetime.now(),
                    )
                else:
                    # Extract response
                    messages = response_dict.get("messages", [])
                    if messages:
                        response_content = (
                            messages[-1].content
                            if hasattr(messages[-1], "content")
                            else str(messages[-1])
                        )
                    else:
                        response_content = f"[No response from {agent_id}]"
                        # Display no response case
                        provider_type = get_provider_type_from_agent_id(agent_id)
                        display_agent_response(
                            agent_id=agent_id,
                            provider=provider_type,
                            content=f"⚠️ No response received",
                            round_number=current_round,
                            topic=current_topic,
                            timestamp=datetime.now(),
                        )

                # Check relevance with moderator
                relevance_check = moderator.evaluate_message_relevance(
                    response_content, current_topic
                )

                if relevance_check["is_relevant"]:
                    # Display the agent response in assembly-style panel
                    provider_type = get_provider_type_from_agent_id(agent_id)
                    display_agent_response(
                        agent_id=agent_id,
                        provider=provider_type,
                        content=response_content,
                        round_number=current_round,
                        topic=current_topic,
                        timestamp=datetime.now(),
                    )

                    # Create LangChain-compatible message
                    message = create_langchain_message(
                        speaker_role="participant",
                        content=response_content,
                        message_id=str(uuid.uuid4()),
                        speaker_id=agent_id,
                        timestamp=datetime.now().isoformat(),
                        phase=2,
                        round=current_round,
                        topic=current_topic,
                        turn_order=i + 1,
                        relevance_score=relevance_check.get("relevance_score", 1.0),
                    )
                    round_messages.append(message)
                    logger.info(f"Agent {agent_id} provided relevant response")
                else:
                    # Display the irrelevant response with a warning indicator
                    provider_type = get_provider_type_from_agent_id(agent_id)
                    display_agent_response(
                        agent_id=agent_id,
                        provider=provider_type,
                        content=f"⚠️ Off-topic response: {response_content[:200]}{'...' if len(response_content) > 200 else ''}",
                        round_number=current_round,
                        topic=current_topic,
                        timestamp=datetime.now(),
                    )

                    # Handle irrelevant response
                    warning = moderator.issue_relevance_warning(agent_id)
                    logger.warning(f"Agent {agent_id} provided irrelevant response")

                    # Add warning message
                    warning_message = create_langchain_message(
                        speaker_role="moderator",
                        content=warning,
                        message_id=str(uuid.uuid4()),
                        speaker_id="moderator",
                        timestamp=datetime.now().isoformat(),
                        phase=2,
                        round=current_round,
                        topic=current_topic,
                        warning_for=agent_id,
                        warning_type="relevance",
                    )
                    round_messages.append(warning_message)

            except Exception as e:
                logger.error(f"Error getting response from {agent_id}: {e}")

                # Display the error for transparency
                provider_type = get_provider_type_from_agent_id(agent_id)
                display_agent_response(
                    agent_id=agent_id,
                    provider=provider_type,
                    content=f"❌ Error occurred: {str(e)[:150]}{'...' if len(str(e)) > 150 else ''}",
                    round_number=current_round,
                    topic=current_topic,
                    timestamp=datetime.now(),
                )

                error_message = create_langchain_message(
                    speaker_role="participant",
                    content=f"[Failed to respond: {str(e)}]",
                    message_id=str(uuid.uuid4()),
                    speaker_id=agent_id,
                    timestamp=datetime.now().isoformat(),
                    phase=2,
                    round=current_round,
                    topic=current_topic,
                    error=True,
                )
                round_messages.append(error_message)

        # Create round info
        round_info = {
            "round_id": round_id,
            "round_number": current_round,
            "topic": current_topic,
            "start_time": round_start,
            "end_time": datetime.now(),
            "participants": [
                (
                    getattr(msg, "additional_kwargs", {}).get("speaker_id", "unknown")
                    if hasattr(msg, "content")
                    else msg.get("speaker_id", "unknown")
                )
                for msg in round_messages
                if (
                    getattr(msg, "additional_kwargs", {}).get(
                        "speaker_role", "participant"
                    )
                    if hasattr(msg, "content")
                    else msg.get("speaker_role", "participant")
                )
                == "participant"
            ],
            "message_count": len(round_messages),
            "summary": None,  # Will be filled by summarization node
        }

        updates = {
            "current_round": current_round,
            "speaking_order": speaking_order,
            # Uses add_messages reducer (expects list)
            "messages": round_messages,
            # Uses list.append reducer (expects single item)
            "round_history": round_info,
            # Uses list.append reducer (expects single item)
            "turn_order_history": speaking_order,
            "rounds_per_topic": {
                **state.get("rounds_per_topic", {}),
                current_topic: state.get("rounds_per_topic", {}).get(current_topic, 0)
                + 1,
            },
        }

        logger.info(
            f"Completed round {current_round} with {len(round_messages)} messages"
        )

        return updates

    def round_summarization_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Invoke Summarizer to compress round.

        This node specifically invokes the SummarizerAgent.
        """
        logger.info("Node: round_summarization - Creating round summary")

        summarizer = self.specialized_agents["summarizer"]
        current_round = state["current_round"]
        current_topic = state["active_topic"]

        # Get messages from current round using standardized attribute access
        all_messages = state.get("messages", [])
        round_messages = [
            msg
            for msg in all_messages
            if (
                get_message_attribute(msg, "round") == current_round
                and get_message_attribute(msg, "topic") == current_topic
                and get_message_attribute(msg, "speaker_role") == "participant"
            )  # Only participant messages
        ]

        if not round_messages:
            logger.warning(f"No messages found for round {current_round}")
            # Return minimal state update
            return {"current_round": current_round}

        try:
            # Invoke summarizer as a tool
            summary = summarizer.summarize_round(
                messages=round_messages, topic=current_topic, round_number=current_round
            )

            # Update round history with summary
            round_history = state.get("round_history", [])
            if round_history:
                # Find and update the current round's info
                for i in range(len(round_history) - 1, -1, -1):
                    if round_history[i]["round_number"] == current_round:
                        round_history[i]["summary"] = summary
                        break

            # Create proper RoundSummary object
            round_summary_obj = RoundSummary(
                round_number=current_round,
                topic=current_topic,
                summary_text=summary,
                created_by="summarizer",  # Assuming summarizer agent created it
                timestamp=datetime.now(),
                token_count=len(summary.split()),  # Rough token estimate
                compression_ratio=0.1,  # Placeholder ratio
            )

            updates = {
                "round_summaries": [round_summary_obj],  # Appends via reducer
                "current_round": current_round,  # Update current round
            }

            logger.info(f"Created summary for round {current_round}")

        except Exception as e:
            logger.error(f"Failed to create round summary: {e}")
            # On error, just update error tracking without invalid fields
            updates = {
                "error_count": state.get("error_count", 0) + 1,
                "last_error": f"Round summarization failed: {str(e)}",
            }

        return updates

    def end_topic_poll_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Poll agents on topic conclusion.

        Only triggered after round 2+.
        """
        logger.info("Node: end_topic_poll - Polling for topic conclusion")

        current_topic = state["active_topic"]
        current_round = state["current_round"]

        # Show dramatic voting announcement if in assembly mode
        try:
            from virtual_agora.ui.display_modes import is_assembly_mode
            from virtual_agora.ui.voting_display import announce_vote, VoteType

            if is_assembly_mode():
                participants = [agent.agent_id for agent in self.discussing_agents]
                context = f"After {current_round} rounds of deliberation"
                announce_vote(
                    VoteType.TOPIC_CONCLUSION, current_topic, participants, context
                )
        except ImportError:
            pass

        # Collect votes
        votes = []

        for agent in self.discussing_agents:
            prompt = f"""We have discussed "{current_topic}" for {current_round} rounds.
            Should we conclude the discussion on '{current_topic}'?

            Please respond with 'Yes' or 'No' and provide a short justification."""

            try:
                response_dict = retry_agent_call(agent, state, prompt)

                if response_dict is None:
                    # All retry attempts failed - exclude from vote
                    logger.warning(
                        f"Agent {agent.agent_id} excluded from topic conclusion vote due to repeated failures"
                    )
                    continue

                # Extract vote
                messages = response_dict.get("messages", [])
                if messages:
                    vote_content = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )

                    # Parse vote
                    vote_lower = vote_content.lower()
                    if "yes" in vote_lower[:20]:  # Check beginning of response
                        vote = "yes"
                    else:
                        vote = "no"

                    votes.append(
                        {
                            "agent_id": agent.agent_id,
                            "vote": vote,
                            "justification": vote_content,
                            "timestamp": datetime.now(),
                        }
                    )
                    logger.info(f"Agent {agent.agent_id} voted: {vote}")

            except Exception as e:
                logger.error(f"Failed to get vote from {agent.agent_id}: {e}")
                # Fix: Don't add biased vote - let the agent be excluded from count
                # This prevents systematic bias toward continuing discussions
                logger.warning(
                    f"Agent {agent.agent_id} excluded from vote due to error: {e}"
                )
                # Note: Not adding to votes list - this agent simply doesn't participate in this vote

        # Tally votes
        yes_votes = sum(1 for v in votes if v["vote"] == "yes")
        total_votes = len(votes)
        majority_threshold = total_votes // 2 + 1

        conclusion_passed = yes_votes >= majority_threshold

        # Identify minority voters
        if conclusion_passed:
            minority_voters = [v["agent_id"] for v in votes if v["vote"] == "no"]
        else:
            minority_voters = [v["agent_id"] for v in votes if v["vote"] == "yes"]

        updates = {
            "topic_conclusion_votes": votes,
            "conclusion_vote": {
                "topic": current_topic,
                "round": current_round,
                "votes": votes,
                "yes_votes": yes_votes,
                "total_votes": total_votes,
                "passed": conclusion_passed,
                "minority_voters": minority_voters,
                "timestamp": datetime.now(),
            },
            "vote_history": {  # Uses list.append reducer (expects single item)
                "vote_type": "topic_conclusion",
                "topic": current_topic,
                "round": current_round,
                "result": "passed" if conclusion_passed else "failed",
                "yes_votes": yes_votes,
                "total_votes": total_votes,
            },
        }

        logger.info(
            f"Conclusion poll: {yes_votes}/{total_votes} votes. "
            f"Result: {'Passed' if conclusion_passed else 'Failed'}"
        )

        # Show dramatic vote results if in assembly mode
        try:
            from virtual_agora.ui.display_modes import is_assembly_mode
            from virtual_agora.ui.voting_display import reveal_vote_results, VoteType

            if is_assembly_mode():
                result_text = (
                    "Motion carries - discussion will conclude"
                    if conclusion_passed
                    else "Motion fails - discussion continues"
                )
                details = {"show_details": True}
                reveal_vote_results(
                    VoteType.TOPIC_CONCLUSION,
                    current_topic,
                    votes,
                    result_text,
                    details,
                )
        except ImportError:
            pass

        return updates

    def vote_evaluation_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Intermediate node for vote evaluation routing.

        This node exists to satisfy LangGraph's requirement that all nodes
        must update at least one state field. It tracks vote evaluations
        without affecting the conditional routing logic.
        """
        logger.info("Node: vote_evaluation - Processing conclusion vote results")

        # Get the conclusion vote from state
        conclusion_vote = state.get("conclusion_vote", {})
        current_topic = state.get("active_topic", "unknown")
        current_round = state.get("current_round", 0)

        # Minimal state update to satisfy LangGraph validation
        # This doesn't change the flow logic, just records the evaluation
        updates = {
            "vote_history": {  # Uses list.append reducer (expects single item)
                "vote_type": "vote_evaluation",
                "topic": current_topic,
                "round": current_round,
                "result": "evaluated",
                "passed": conclusion_vote.get("passed", False),
                "timestamp": datetime.now(),
            }
        }

        logger.info(f"Vote evaluation completed for topic: {current_topic}")
        return updates

    def periodic_user_stop_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL node for 5-round periodic stops.

        New in v1.3 - gives user periodic control every 5 rounds.
        Uses LangGraph interrupt mechanism to pause execution.
        """
        logger.info("Node: periodic_user_stop - User checkpoint")

        current_round = state["current_round"]
        current_topic = state["active_topic"]

        # Use LangGraph interrupt to pause execution and wait for user input
        logger.info(
            f"Interrupting for periodic user checkpoint at round {current_round}"
        )
        user_input = interrupt(
            {
                "type": "periodic_stop",
                "current_round": current_round,
                "current_topic": current_topic,
                "message": (
                    f"You've reached a 5-round checkpoint (Round {current_round}).\n"
                    f"Currently discussing: {current_topic}\n\n"
                    "What would you like to do?"
                ),
                "options": ["continue", "end_topic", "modify", "skip"],
            }
        )

        # Process user input (this will be called when execution resumes)
        if user_input is None:
            # Fallback to continue discussion
            logger.warning(
                "No user input received for periodic stop, continuing discussion"
            )
            return {
                "user_periodic_decision": "continue",
                "periodic_stop_counter": state.get("periodic_stop_counter", 0) + 1,
            }

        decision = user_input.get("action", "continue")

        logger.info(f"User periodic decision: {decision}")

        # Store decision for conditional routing
        updates = {
            "user_periodic_decision": decision,
            "periodic_stop_counter": state.get("periodic_stop_counter", 0) + 1,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "periodic_stop",
                "approval_history": state.get("hitl_state", {}).get(
                    "approval_history", []
                )
                + [
                    {
                        "type": "periodic_stop",
                        "result": decision,
                        "round": current_round,
                        "timestamp": datetime.now(),
                    }
                ],
            },
        }

        # Handle different user decisions
        if decision == "end_topic":
            updates["user_forced_conclusion"] = True
        elif decision == "modify":
            updates["user_requested_modification"] = True
        elif decision == "skip":
            updates["user_skip_to_final"] = True

        return updates

    # ===== Phase 3: Topic Conclusion Nodes =====

    def final_considerations_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Collect final thoughts from agents.

        Logic varies based on how conclusion was triggered:
        - Vote-based: Only dissenting agents
        - User-forced: All agents
        """
        logger.info("Node: final_considerations - Collecting final thoughts")

        current_topic = state["active_topic"]
        user_forced = state.get("user_forced_conclusion", False)

        # Determine which agents to prompt
        if user_forced:
            # User forced conclusion - all agents provide final thoughts
            agents_to_prompt = self.discussing_agents
            logger.info("User forced conclusion - collecting from all agents")
        else:
            # Vote-based conclusion - only minority voters
            conclusion_vote = state.get("conclusion_vote", {})
            minority_voters = conclusion_vote.get("minority_voters", [])
            agents_to_prompt = [
                a for a in self.discussing_agents if a.agent_id in minority_voters
            ]
            logger.info(
                f"Vote-based conclusion - collecting from {len(agents_to_prompt)} minority voters"
            )

        # Collect final thoughts
        final_thoughts = []

        for agent in agents_to_prompt:
            prompt = f"""The discussion on '{current_topic}' is concluding.
            {'You voted against conclusion, but the majority has decided to move on.' if not user_forced else ''}
            Please provide your final considerations on this topic."""

            try:
                response_dict = agent(state, prompt=prompt)

                # Extract response
                messages = response_dict.get("messages", [])
                if messages:
                    response_content = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )

                    final_thoughts.append(
                        {
                            "agent_id": agent.agent_id,
                            "consideration": response_content,
                            "timestamp": datetime.now(),
                        }
                    )
                    logger.info(f"Collected final thoughts from {agent.agent_id}")

            except Exception as e:
                logger.error(f"Failed to get final thoughts from {agent.agent_id}: {e}")

        # Store final considerations in consensus_proposals for the current topic
        # This is a temporary storage that will be used by topic_report_generation_node
        current_topic_key = current_topic or "unknown_topic"
        updates = {
            "consensus_proposals": {
                **state.get("consensus_proposals", {}),
                f"{current_topic_key}_final_considerations": [
                    fc["consideration"] for fc in final_thoughts
                ],
            }
        }

        logger.info(f"Collected {len(final_thoughts)} final considerations")

        return updates

    def topic_report_generation_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Invoke Report Writer Agent for iterative topic synthesis.

        This node uses the ReportWriterAgent to create comprehensive topic reports
        through an iterative process: structure creation followed by section writing.
        """
        logger.info("Node: topic_report_generation - Creating topic report iteratively")

        report_writer_agent = self.specialized_agents["report_writer"]
        current_topic = state["active_topic"]
        theme = state["main_topic"]

        # Gather round summaries for this topic
        all_summaries = state.get("round_summaries", [])

        # Handle potentially nested summary structures (similar to vote processing fix)
        flattened_summaries = []
        for item in all_summaries:
            if isinstance(item, list):
                # If item is a list, flatten it
                flattened_summaries.extend(item)
            else:
                # If item is a dict, add it directly
                flattened_summaries.append(item)

        topic_summaries = [
            s.get("summary", "")
            for s in flattened_summaries
            if isinstance(s, dict) and s.get("topic") == current_topic
        ]

        # Get final considerations from consensus_proposals storage
        current_topic_key = current_topic or "unknown_topic"
        consideration_texts = state.get("consensus_proposals", {}).get(
            f"{current_topic_key}_final_considerations", []
        )

        try:
            # Get console for user feedback
            console = get_console()

            # Prepare source material for the report writer
            source_material = {
                "topic": current_topic,
                "theme": theme,
                "round_summaries": topic_summaries,
                "final_considerations": consideration_texts,
            }

            # Phase 1: Create report structure with progress feedback
            console.print("\n[cyan]📝 Generating topic report...[/cyan]")

            sections, structure_response = report_writer_agent.create_report_structure(
                source_material, "topic"
            )

            # Calculate total steps for consistent progress tracking
            total_steps = (
                len(sections) + 2
            )  # structure creation + sections + completion

            show_mini_progress(1, total_steps, "Report structure created")
            show_mini_progress(
                2, total_steps, f"Structure defined with {len(sections)} sections"
            )

            # Phase 2: Write all sections iteratively with progress feedback
            report_parts = []
            previous_sections = []

            for i, section in enumerate(sections):
                current_step = i + 3  # After structure creation (steps 1-2)
                step_name = f"Writing section: {section['title']}"

                # Show progress for this section
                show_mini_progress(current_step, total_steps, step_name)

                logger.info(
                    f"Writing topic report section {i+1}/{len(sections)}: {section['title']}"
                )

                section_content = report_writer_agent.write_section(
                    section, source_material, "topic", previous_sections
                )
                report_parts.append(section_content)
                previous_sections.append(section_content)

            # Show completion
            show_mini_progress(total_steps, total_steps, "Report generation completed")
            console.print("[green]✅ Topic report generated successfully![/green]\n")

            # Combine all sections into final report
            report = "\n\n".join(report_parts)

            updates = {
                "topic_summaries": {
                    **state.get("topic_summaries", {}),
                    current_topic: report,
                },
                "report_structures": {
                    **state.get("report_structures", {}),
                    f"{current_topic}_structure": sections,
                },
                "current_phase": 3,
            }

            logger.info(
                f"Generated topic report for: {current_topic} with {len(sections)} sections"
            )

        except Exception as e:
            logger.error(f"Failed to generate topic report: {e}")
            updates = {
                "topic_summaries": {
                    **state.get("topic_summaries", {}),
                    current_topic: f"Failed to generate report: {str(e)}",
                },
                "report_error": str(e),
            }

        return updates

    def topic_summary_generation_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Create a concise topic conclusion summary for future reference.

        This node invokes the SummarizerAgent to create a one-paragraph summary
        that captures the key resolution, consensus points, outstanding questions,
        and practical implications from the concluded topic discussion.
        """
        logger.info(
            "Node: topic_summary_generation - Creating topic conclusion summary"
        )

        summarizer_agent = self.specialized_agents["summarizer"]
        current_topic = state["active_topic"]

        # Gather round summaries for this topic
        all_summaries = state.get("round_summaries", [])

        # Handle potentially nested summary structures
        flattened_summaries = []
        for item in all_summaries:
            if isinstance(item, list):
                flattened_summaries.extend(item)
            else:
                flattened_summaries.append(item)

        topic_round_summaries = [
            s.get("summary", "")
            for s in flattened_summaries
            if isinstance(s, dict) and s.get("topic") == current_topic
        ]

        # Get final considerations from consensus_proposals storage
        current_topic_key = current_topic or "unknown_topic"
        consideration_texts = state.get("consensus_proposals", {}).get(
            f"{current_topic_key}_final_considerations", []
        )

        try:
            # Get console for user feedback
            console = get_console()
            console.print("\n[cyan]📋 Creating topic conclusion summary...[/cyan]")

            # Create topic conclusion summary using SummarizerAgent
            topic_conclusion_summary = summarizer_agent.summarize_topic_conclusion(
                round_summaries=topic_round_summaries,
                final_considerations=consideration_texts,
                topic=current_topic,
            )

            console.print("[green]✅ Topic conclusion summary created![/green]\n")

            # Store the summary in topic_summaries state field using the reducer
            updates = {
                "topic_summaries": {
                    **state.get("topic_summaries", {}),
                    f"{current_topic}_conclusion": topic_conclusion_summary,
                }
            }

            logger.info(f"Generated topic conclusion summary for: {current_topic}")

        except Exception as e:
            logger.error(f"Failed to generate topic conclusion summary: {e}")
            updates = {
                "topic_summaries": {
                    **state.get("topic_summaries", {}),
                    f"{current_topic}_conclusion": f"Failed to generate summary: {str(e)}",
                },
                "summary_error": str(e),
            }

        return updates

    def file_output_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Save topic report to file.

        Pure I/O operation, no agent invocation.
        """
        logger.info("Node: file_output - Saving topic report")

        current_topic = state["active_topic"]
        topic_summaries = state.get("topic_summaries", {})
        report = topic_summaries.get(current_topic, "")
        session_id = state["session_id"]

        # Create organized directory structure
        session_dir, topic_dir = create_report_directory_structure(
            session_id, current_topic
        )

        # Generate clean filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"topic_report_{timestamp}.md"

        filepath = os.path.join(topic_dir, filename)

        try:
            # Save file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Topic Report: {current_topic}\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Discussion Theme: {state['main_topic']}\n")
                f.write(
                    f"Rounds Discussed: {state.get('rounds_per_topic', {}).get(current_topic, 0)}\n\n"
                )
                f.write("---\n\n")
                f.write(report)

            updates = {
                "topic_report_saved": filename,
                "topic_summary_files": {
                    **state.get("topic_summary_files", {}),
                    current_topic: filepath,
                },
                # Uses list.append reducer (expects single item)
                "completed_topics": current_topic,
                # Remove from queue
                "topic_queue": [
                    t for t in state.get("topic_queue", []) if t != current_topic
                ],
                "active_topic": None,  # Clear active topic
            }

            # Clear moderator topic context when topic concludes
            moderator = self.specialized_agents.get("moderator")
            if moderator:
                moderator.current_topic_context = None
                logger.info("Cleared moderator topic context after topic conclusion")

            # Get console for user feedback
            console = get_console()
            console.print(
                f"\n[green]💾 Topic report saved to:[/green] [cyan]{filepath}[/cyan]"
            )
            logger.info(f"Saved topic report to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save topic report: {e}")
            updates = {
                "file_save_error": str(e),
                # Uses list.append reducer (expects single item)
                "completed_topics": current_topic,
                "topic_queue": [
                    t for t in state.get("topic_queue", []) if t != current_topic
                ],
                "active_topic": None,
            }

            # Clear moderator topic context even in error case
            moderator = self.specialized_agents.get("moderator")
            if moderator:
                moderator.current_topic_context = None
                logger.info(
                    "Cleared moderator topic context after topic conclusion (error case)"
                )

        return updates

    # ===== Phase 4: Continuation & Agenda Re-evaluation Nodes =====

    def agent_poll_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Poll agents on whether to end the entire session.

        New in v1.3 - agents can vote to end the ecclesia.
        """
        logger.info("Node: agent_poll - Checking if agents want to end session")

        completed_topics = state.get("completed_topics", [])
        remaining_topics = state.get("topic_queue", [])

        # Handle potentially nested topic structures (similar to other fixes)
        flattened_completed = []
        for item in completed_topics:
            if isinstance(item, list):
                flattened_completed.extend(item)
            elif isinstance(item, str):
                flattened_completed.append(item)

        flattened_remaining = []
        for item in remaining_topics:
            if isinstance(item, list):
                flattened_remaining.extend(item)
            elif isinstance(item, str):
                flattened_remaining.append(item)

        # Poll agents
        votes_to_end = 0
        votes_to_continue = 0

        for agent in self.discussing_agents:
            prompt = f"""We have completed discussion on: {', '.join(flattened_completed)}

            Remaining topics: {', '.join(flattened_remaining) if flattened_remaining else 'None'}

            Should we end the entire discussion session (the "ecclesia")?
            Please respond with 'End' to conclude or 'Continue' to discuss remaining topics."""

            try:
                response_dict = retry_agent_call(agent, state, prompt)

                if response_dict is None:
                    # All retry attempts failed - exclude from session vote
                    logger.warning(
                        f"Agent {agent.agent_id} excluded from session vote due to repeated failures"
                    )
                    continue

                # Extract vote
                messages = response_dict.get("messages", [])
                if messages:
                    vote_content = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )

                    # Parse vote
                    vote_lower = vote_content.lower()
                    if "end" in vote_lower[:20]:
                        votes_to_end += 1
                    else:
                        votes_to_continue += 1

                    logger.info(
                        f"Agent {agent.agent_id} voted: {'end' if 'end' in vote_lower[:20] else 'continue'}"
                    )

            except Exception as e:
                logger.error(f"Failed to get session vote from {agent.agent_id}: {e}")
                # Fix: Don't add biased vote - exclude failed agents from session vote count
                # This prevents systematic bias toward continuing sessions
                logger.warning(
                    f"Agent {agent.agent_id} excluded from session vote due to error: {e}"
                )
                # Note: Not incrementing any counter - this agent simply doesn't participate

        # Determine outcome (majority wins)
        agents_vote_end = votes_to_end > votes_to_continue

        # Store session vote results in existing schema fields
        updates = {
            "agents_vote_end_session": agents_vote_end,
            # Store vote details in the warnings field for now (similar to other fixes)
            "warnings": [
                f"Session vote: {votes_to_end} to end, {votes_to_continue} to continue"
            ],
        }

        logger.info(
            f"Session vote: {votes_to_end} to end, {votes_to_continue} to continue. "
            f"Result: {'End session' if agents_vote_end else 'Continue'}"
        )

        return updates

    def user_approval_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL: Get user approval to continue with next topic.

        Uses LangGraph interrupt mechanism to get user continuation decision.
        User has final say regardless of agent vote.
        """
        logger.info("Node: user_approval - Getting user continuation approval")

        # Handle potentially nested completed_topics structure
        completed_topics = state.get("completed_topics", [])
        flattened_completed = []
        for item in completed_topics:
            if isinstance(item, list):
                flattened_completed.extend(item)
            elif isinstance(item, str):
                flattened_completed.append(item)

        completed_topic = flattened_completed[-1] if flattened_completed else "Unknown"
        remaining_topics = state.get("topic_queue", [])
        agents_vote_end = state.get("agents_vote_end_session", False)

        # Use LangGraph interrupt to pause execution and wait for user input
        logger.info("Interrupting for user continuation approval")
        user_input = interrupt(
            {
                "type": "topic_continuation",
                "completed_topic": completed_topic,
                "remaining_topics": remaining_topics,
                "agent_recommendation": (
                    "end_session" if agents_vote_end else "continue"
                ),
                "message": (
                    f"Topic '{completed_topic}' has been concluded.\n"
                    f"Remaining topics: {len(remaining_topics)}\n"
                    f"Agent recommendation: {'End session' if agents_vote_end else 'Continue'}\n\n"
                    "What would you like to do?"
                ),
                "options": ["continue", "end_session", "modify_agenda"],
            }
        )

        # Process user input (this will be called when execution resumes)
        if user_input is None:
            # Fallback: continue if there are remaining topics, otherwise end
            logger.warning("No user input received for continuation approval")
            decision = "continue" if remaining_topics else "end_session"
        else:
            decision = user_input.get("action", "continue")

        logger.info(f"User continuation decision: {decision}")

        # Process user decision
        updates = {
            "user_approves_continuation": decision == "continue",
            "user_requests_end": decision == "end_session",
            "user_requested_modification": decision == "modify_agenda",
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "topic_continuation",
                "approval_history": state.get("hitl_state", {}).get(
                    "approval_history", []
                )
                + [
                    {
                        "type": "topic_continuation",
                        "result": decision,
                        "completed_topic": completed_topic,
                        "timestamp": datetime.now(),
                    }
                ],
            },
        }

        return updates

    def agenda_modification_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Allow agents to modify remaining agenda.

        Agents can propose additions/removals based on discussion so far.
        """
        logger.info("Node: agenda_modification - Processing agenda changes")

        moderator = self.specialized_agents["moderator"]
        remaining_topics = state.get("topic_queue", [])
        completed_topics = state.get("completed_topics", [])

        # Handle potentially nested topic structures (similar to other fixes)
        flattened_completed = []
        for item in completed_topics:
            if isinstance(item, list):
                flattened_completed.extend(item)
            elif isinstance(item, str):
                flattened_completed.append(item)

        flattened_remaining = []
        for item in remaining_topics:
            if isinstance(item, list):
                flattened_remaining.extend(item)
            elif isinstance(item, str):
                flattened_remaining.append(item)

        # Check if user requested modification
        if state.get("user_requested_modification"):
            # Get user's modifications
            new_agenda = get_agenda_modifications(flattened_remaining)

            updates = {
                "topic_queue": new_agenda,
                "agenda_modifications": {
                    "type": "user_modification",
                    "original": flattened_remaining,
                    "revised": new_agenda,
                    "timestamp": datetime.now(),
                },
                "current_phase": 4,
            }

            return updates

        # Otherwise, ask agents for modifications
        modifications = []

        modification_prompt = f"""Based on our discussions on: {', '.join(flattened_completed)}

        We have these remaining topics: {', '.join(flattened_remaining)}

        Should we add any new topics to our agenda, or remove any of the remaining ones?
        Please provide your suggestions."""

        for agent in self.discussing_agents:
            try:
                response_dict = agent(state, prompt=modification_prompt)

                # Extract suggestions
                messages = response_dict.get("messages", [])
                if messages:
                    suggestion = (
                        messages[-1].content
                        if hasattr(messages[-1], "content")
                        else str(messages[-1])
                    )
                    modifications.append(
                        {
                            "agent_id": agent.agent_id,
                            "suggestion": suggestion,
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to get modification from {agent.agent_id}: {e}")

        # Moderator synthesizes modifications
        synthesis_prompt = f"""Here are the agents' suggestions for agenda modifications:
        {chr(10).join(f"{m['agent_id']}: {m['suggestion']}" for m in modifications)}

        Current remaining topics: {flattened_remaining}

        Based on these suggestions, create a revised agenda. Consider:
        - Adding genuinely valuable new topics
        - Removing topics that may no longer be relevant
        - Maintaining reasonable scope

        Respond with a JSON object: {{"revised_agenda": ["Topic 1", "Topic 2"]}}"""

        try:
            response = moderator.generate_json_response(synthesis_prompt)
            revised_agenda = response.get("revised_agenda", flattened_remaining)
        except Exception as e:
            logger.error(f"Failed to synthesize modifications: {e}")
            revised_agenda = flattened_remaining

        updates = {
            "topic_queue": revised_agenda,
            "agenda_modifications": {
                "type": "agent_modification",
                "suggestions": modifications,
                "original": flattened_remaining,
                "revised": revised_agenda,
                "timestamp": datetime.now(),
            },
            "current_phase": 4,
        }

        logger.info(f"Agenda modified. New queue: {revised_agenda}")

        return updates

    # ===== Phase 5: Final Report Generation Nodes =====

    def final_report_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Invoke Report Writer Agent to generate final session report iteratively.

        The agent:
        1. Reads all topic reports
        2. Defines report structure
        3. Writes each section iteratively
        """
        logger.info("Node: final_report - Generating final session report iteratively")

        report_writer_agent = self.specialized_agents["report_writer"]
        topic_summaries = state.get("topic_summaries", {})
        theme = state["main_topic"]

        if not topic_summaries:
            logger.warning("No topic summaries available for final report")
            return {
                "report_generation_status": "failed",
                "report_error": "No topic summaries available",
            }

        try:
            # Get console for user feedback
            console = get_console()

            # Prepare source material for the report writer
            source_material = {
                "theme": theme,
                "topic_reports": topic_summaries,
            }

            # Phase 1: Create report structure with progress feedback
            console.print("\n[cyan]📊 Generating final session report...[/cyan]")

            sections, structure_response = report_writer_agent.create_report_structure(
                source_material, "session"
            )

            # Calculate total steps for consistent progress tracking
            total_steps = (
                len(sections) + 2
            )  # structure creation + sections + completion

            show_mini_progress(1, total_steps, "Session report structure created")
            show_mini_progress(
                2, total_steps, f"Structure defined with {len(sections)} sections"
            )
            logger.info(f"Report structure defined with {len(sections)} sections")

            # Phase 2: Generate content for each section iteratively with progress feedback
            report_content = {}
            previous_sections_list = []

            for i, section in enumerate(sections):
                current_step = i + 3  # After structure creation (steps 1-2)
                step_name = f"Writing section: {section['title']}"

                # Show progress for this section
                show_mini_progress(current_step, total_steps, step_name)

                logger.info(
                    f"Writing session report section {i+1}/{len(sections)}: {section['title']}"
                )

                section_content = report_writer_agent.write_section(
                    section=section,
                    source_material=source_material,
                    report_type="session",
                    previous_sections=previous_sections_list,
                )

                report_content[section["title"]] = section_content
                previous_sections_list.append(section_content)

            # Show completion
            show_mini_progress(
                total_steps, total_steps, "Session report generation completed"
            )
            console.print(
                "[green]✅ Final session report generated successfully![/green]\n"
            )

            # Extract section titles for backward compatibility
            report_sections = [section["title"] for section in sections]

            updates = {
                "current_phase": 5,
                "report_structure": report_sections,
                "report_sections": report_content,
                "report_generation_status": "completed",
                "final_report": report_content,
                "session_report_structures": sections,
            }

            logger.info(
                f"Final report generation completed with {len(sections)} sections"
            )

        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            updates = {
                "current_phase": 5,
                "report_generation_status": "failed",
                "report_error": str(e),
            }

        return updates

    def multi_file_output_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Save final report sections to multiple files.

        Each section gets its own numbered markdown file.
        """
        logger.info("Node: multi_file_output - Saving final report files")

        report_sections = state.get("report_sections", {})
        theme = state["main_topic"]
        session_id = state.get("session_id", "unknown")

        # Create organized directory structure
        session_dir, _ = create_report_directory_structure(session_id)
        final_report_dir = os.path.join(session_dir, "final_report")
        os.makedirs(final_report_dir, exist_ok=True)

        saved_files = []

        try:
            # Save each section
            for i, (section_title, content) in enumerate(report_sections.items()):
                # Generate clean filename using sanitization
                safe_title = sanitize_filename(section_title, 40)
                filename = f"{i+1:02d}_{safe_title}.md"
                filepath = os.path.join(final_report_dir, filename)

                # Write file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {section_title}\n\n")
                    f.write(f"**Virtual Agora Session**: {session_id}\n")
                    f.write(f"**Theme**: {theme}\n")
                    f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
                    f.write("---\n\n")
                    f.write(content)

                saved_files.append(
                    {
                        "section": section_title,
                        "filename": filename,
                        "filepath": filepath,
                    }
                )

                logger.info(f"Saved section '{section_title}' to {filename}")

            # Create index file
            index_path = os.path.join(final_report_dir, "00_index.md")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(f"# Virtual Agora Final Report Index\n\n")
                f.write(f"**Session ID**: {session_id}\n")
                f.write(f"**Theme**: {theme}\n")
                f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
                f.write("## Report Sections\n\n")
                for file_info in saved_files:
                    f.write(f"- [{file_info['section']}](./{file_info['filename']})\n")

            updates = {
                "final_report_files": saved_files,
                "final_report_directory": final_report_dir,
                "session_completed": True,
                "completion_timestamp": datetime.now(),
                "phase_history": {
                    "from_phase": 4,
                    "to_phase": 5,
                    "timestamp": datetime.now(),
                    "reason": "Final report generation completed",
                    "triggered_by": "system",
                },
            }

            # Get console for user feedback
            console = get_console()
            console.print(
                f"\n[green]📁 Final session report saved to:[/green] [cyan]{final_report_dir}[/cyan]"
            )
            console.print(
                f"[green]📄 Generated {len(saved_files)} report sections + index[/green]"
            )
            logger.info(f"Saved {len(saved_files)} report files to {final_report_dir}")

        except Exception as e:
            logger.error(f"Failed to save report files: {e}")
            updates = {
                "file_save_error": str(e),
                "session_completed": True,
                "completion_timestamp": datetime.now(),
            }

        return updates
