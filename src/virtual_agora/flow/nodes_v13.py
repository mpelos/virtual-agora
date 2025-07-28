"""Node implementations for Virtual Agora v1.3 discussion flow.

This module contains all node functions for the node-centric architecture
where nodes orchestrate specialized agents as tools.
"""

import uuid
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
from virtual_agora.agents.topic_report_agent import TopicReportAgent
from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent
from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.flow.cycle_detection import CyclePreventionManager
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

logger = get_logger(__name__)


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
        logger.info("Configuration validated successfully")

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
            "topic_report": self.specialized_agents["topic_report"].agent_id,
            "ecclesia_report": self.specialized_agents["ecclesia_report"].agent_id,
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

        logger.info(f"Recorded {len(specialized_agent_ids)} specialized agents")
        logger.info(f"Recorded {len(discussing_agent_info)} discussing agents")

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
        logger.info("=== FLOW DEBUG: Entering announce_item_node ===")
        logger.info(f"State keys available: {list(state.keys())}")
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
        logger.info("=== FLOW DEBUG: Entering discussion_round_node ===")
        logger.info(f"State keys available: {list(state.keys())}")
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
            logger.info(f"Initialized speaking order: {speaking_order}")
        else:
            speaking_order = speaking_order.copy()
        if current_round > 1:
            # Rotate: [A,B,C] -> [B,C,A]
            speaking_order = speaking_order[1:] + [speaking_order[0]]

        # Execute the round
        round_messages = []
        round_id = str(uuid.uuid4())
        round_start = datetime.now()

        for i, agent_id in enumerate(speaking_order):
            # Find the agent
            agent = next(a for a in self.discussing_agents if a.agent_id == agent_id)

            # Build context for agent
            context = f"""
            Theme: {theme}
            Current Topic: {current_topic}
            Round: {current_round}
            Your turn: {i + 1}/{len(speaking_order)}
            """

            # Add previous round summaries
            if topic_summaries:
                context += "\n\nPrevious Round Summaries:\n"
                for j, summary in enumerate(topic_summaries[-3:]):  # Last 3 summaries
                    context += f"Round {summary.get('round_number', j+1)}: {summary.get('summary', '')}\n"

            # Add current round comments
            if round_messages:
                context += "\n\nCurrent Round Comments:\n"
                for msg in round_messages:
                    # Handle both LangChain BaseMessage and Virtual Agora dict formats
                    if hasattr(msg, "content"):  # LangChain BaseMessage
                        speaker_id = getattr(msg, "additional_kwargs", {}).get(
                            "speaker_id", "unknown"
                        )
                        content = msg.content
                    else:  # Virtual Agora dict format
                        speaker_id = msg.get("speaker_id", "unknown")
                        content = msg.get("content", "")
                    context += f"{speaker_id}: {content[:200]}...\n"

            context += f"\n\nPlease provide your thoughts on '{current_topic}'. Keep your response focused and substantive (2-4 sentences)."

            try:
                # Get agent response
                response_dict = agent(state, prompt=context)

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

                # Check relevance with moderator
                relevance_check = moderator.evaluate_message_relevance(
                    response_content, current_topic
                )

                if relevance_check["is_relevant"]:
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
            "messages": round_messages,  # Uses add_messages reducer (expects list)
            "round_history": round_info,  # Uses list.append reducer (expects single item)
            "turn_order_history": speaking_order,  # Uses list.append reducer (expects single item)
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

        # Get messages from current round
        all_messages = state.get("messages", [])
        round_messages = [
            msg
            for msg in all_messages
            if (
                getattr(msg, "additional_kwargs", {}).get("round")
                if hasattr(msg, "content")
                else msg.get("round")
            )
            == current_round
            and (
                getattr(msg, "additional_kwargs", {}).get("topic")
                if hasattr(msg, "content")
                else msg.get("topic")
            )
            == current_topic
            and (
                getattr(msg, "additional_kwargs", {}).get("speaker_role")
                if hasattr(msg, "content")
                else msg.get("speaker_role")
            )
            == "participant"  # Only participant messages
        ]

        if not round_messages:
            logger.warning(f"No messages found for round {current_round}")
            return {"current_round": current_round}  # Return minimal state update

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

        Only triggered after round 3+.
        """
        logger.info("Node: end_topic_poll - Polling for topic conclusion")

        current_topic = state["active_topic"]
        current_round = state["current_round"]

        # Collect votes
        votes = []

        for agent in self.discussing_agents:
            prompt = f"""We have discussed "{current_topic}" for {current_round} rounds.
            Should we conclude the discussion on '{current_topic}'?
            
            Please respond with 'Yes' or 'No' and provide a short justification."""

            try:
                response_dict = agent(state, prompt=prompt)

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
                votes.append(
                    {
                        "agent_id": agent.agent_id,
                        "vote": "no",  # Default to continue
                        "justification": "Failed to vote",
                        "error": str(e),
                    }
                )

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
        """Invoke Topic Report Agent for synthesis.

        This node specifically invokes the TopicReportAgent.
        """
        logger.info("Node: topic_report_generation - Creating topic report")

        topic_report_agent = self.specialized_agents["topic_report"]
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
            # Invoke topic report agent
            report = topic_report_agent.synthesize_topic(
                round_summaries=topic_summaries,
                final_considerations=consideration_texts,
                topic=current_topic,
                discussion_theme=theme,
            )

            updates = {
                "topic_summaries": {
                    **state.get("topic_summaries", {}),
                    current_topic: report,
                },
                "last_topic_report": report,
                "current_phase": 3,
            }

            logger.info(f"Generated topic report for: {current_topic}")

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

    def file_output_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Save topic report to file.

        Pure I/O operation, no agent invocation.
        """
        logger.info("Node: file_output - Saving topic report")

        current_topic = state["active_topic"]
        report = state.get("last_topic_report", "")

        # Generate filename
        safe_topic = current_topic.replace(" ", "_").replace("/", "_")[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agenda_summary_{safe_topic}_{timestamp}.md"

        # Ensure reports directory exists
        import os

        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)

        filepath = os.path.join(reports_dir, filename)

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
                "completed_topics": current_topic,  # Uses list.append reducer (expects single item)
                # Remove from queue
                "topic_queue": [
                    t for t in state.get("topic_queue", []) if t != current_topic
                ],
                "active_topic": None,  # Clear active topic
            }

            logger.info(f"Saved topic report to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save topic report: {e}")
            updates = {
                "file_save_error": str(e),
                "completed_topics": current_topic,  # Uses list.append reducer (expects single item)
                "topic_queue": [
                    t for t in state.get("topic_queue", []) if t != current_topic
                ],
                "active_topic": None,
            }

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
                response_dict = agent(state, prompt=prompt)

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
                votes_to_continue += 1  # Default to continue

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
        """Invoke Ecclesia Report Agent to generate final report.

        The agent:
        1. Reads all topic reports
        2. Defines report structure
        3. Writes each section
        """
        logger.info("Node: final_report - Generating final session report")

        ecclesia_agent = self.specialized_agents["ecclesia_report"]
        topic_summaries = state.get("topic_summaries", {})
        theme = state["main_topic"]

        if not topic_summaries:
            logger.warning("No topic summaries available for final report")
            return {
                "report_generation_status": "failed",
                "report_error": "No topic summaries available",
            }

        try:
            # Step 1: Define report structure
            report_sections = ecclesia_agent.generate_report_structure(
                topic_reports=topic_summaries, discussion_theme=theme
            )

            logger.info(
                f"Report structure defined with {len(report_sections)} sections"
            )

            # Step 2: Generate content for each section
            report_content = {}
            previous_sections = {}

            for i, section_title in enumerate(report_sections):
                logger.info(
                    f"Writing section {i+1}/{len(report_sections)}: {section_title}"
                )

                section_content = ecclesia_agent.write_section(
                    section_title=section_title,
                    topic_reports=topic_summaries,
                    discussion_theme=theme,
                    previous_sections=previous_sections,
                )

                report_content[section_title] = section_content
                previous_sections[section_title] = section_content

            updates = {
                "current_phase": 5,
                "report_structure": report_sections,
                "report_sections": report_content,
                "report_generation_status": "completed",
                "final_report": report_content,
            }

            logger.info("Final report generation completed")

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

        # Ensure reports directory exists
        import os

        reports_dir = "reports"
        final_report_dir = os.path.join(reports_dir, f"final_report_{session_id}")
        os.makedirs(final_report_dir, exist_ok=True)

        saved_files = []

        try:
            # Save each section
            for i, (section_title, content) in enumerate(report_sections.items()):
                # Generate filename
                safe_title = section_title.replace(" ", "_").replace("/", "_")[:50]
                filename = f"final_report_{i+1:02d}_{safe_title}.md"
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

            logger.info(f"Saved {len(saved_files)} report files to {final_report_dir}")

        except Exception as e:
            logger.error(f"Failed to save report files: {e}")
            updates = {
                "file_save_error": str(e),
                "session_completed": True,
                "completion_timestamp": datetime.now(),
            }

        return updates
