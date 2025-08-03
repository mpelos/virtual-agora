"""Discussion Round Node for Virtual Agora with configurable user participation.

This module implements the unified DiscussionRoundNode that handles complete
discussion rounds with configurable user participation timing. It replaces
the separate user_participation_check, user_turn_participation, and
discussion_round nodes with a single integrated node.
"""

import uuid
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt

from virtual_agora.flow.nodes.base import HITLNode, NodeDependencies
from virtual_agora.flow.participation_strategies import UserParticipationStrategy
from virtual_agora.flow.state_manager import FlowStateManager
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


# Import functions from nodes_v13 that are used in discussion logic
def validate_agent_context(theme, current_topic, topic_summaries, round_messages):
    """Validate that agent context contains all required elements per Phase 2 spec.

    According to spec, agents must receive:
    1. The initial user-provided theme
    2. The specific agenda item being discussed
    3. Summaries of previous rounds for this agenda item
    4. Messages from other agents in the current round

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


def retry_agent_call(
    agent, state, prompt, context_messages=None, max_attempts=3, base_delay=1.0
):
    """Retry agent calls with exponential backoff and proper response handling.

    Args:
        agent: The LLM agent to call
        state: Current Virtual Agora state
        prompt: The prompt to send to the agent
        context_messages: Optional context messages
        max_attempts: Maximum retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        Dict with 'messages' key containing the response, or None if all attempts fail
    """
    import time

    for attempt in range(max_attempts):
        try:
            response = agent(state, prompt=prompt, context_messages=context_messages)

            # Handle different response types
            if response is None:
                logger.warning(
                    f"Agent {getattr(agent, 'agent_id', 'unknown')} returned None response"
                )
                if attempt < max_attempts - 1:
                    time.sleep(base_delay * (2**attempt))
                    continue
                return None

            # If response is already a dictionary with messages, return it
            if isinstance(response, dict) and "messages" in response:
                return response

            # If response is a dictionary but no messages key, check for content
            if isinstance(response, dict):
                # Look for content in various possible keys
                content = None
                if "content" in response:
                    content = response["content"]
                elif "response" in response:
                    content = response["response"]
                else:
                    # Convert the entire dict to string as fallback
                    content = str(response)

                # Create proper message structure
                from langchain_core.messages import AIMessage

                message = AIMessage(content=content)
                return {"messages": [message]}

            # If response is a string, wrap it in proper structure
            if isinstance(response, str):
                from langchain_core.messages import AIMessage

                message = AIMessage(content=response)
                return {"messages": [message]}

            # If response is a LangChain message object, wrap it
            if hasattr(response, "content"):
                return {"messages": [response]}

            # Fallback: convert to string and wrap
            logger.warning(
                f"Unexpected response type {type(response)} from agent {getattr(agent, 'agent_id', 'unknown')}, converting to string"
            )
            from langchain_core.messages import AIMessage

            message = AIMessage(content=str(response))
            return {"messages": [message]}

        except Exception as e:
            logger.error(f"Agent call attempt {attempt + 1}/{max_attempts} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                logger.error(
                    f"All {max_attempts} attempts failed for agent {getattr(agent, 'agent_id', 'unknown')}"
                )
                return None

    return None


def get_provider_type_from_agent_id(agent_id):
    """Get provider type from agent ID.

    Extract provider type from agent ID patterns like:
    - gpt-4o-1, gpt-4o-2 -> OpenAI
    - claude-3-opus-1 -> Anthropic
    - gemini-2.5-pro-1 -> Google
    - grok-beta-1 -> Grok
    """
    from virtual_agora.providers.config import ProviderType

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


def create_langchain_message(speaker_role: str, content: str, **metadata):
    """Create a LangChain message object with Virtual Agora metadata.

    Args:
        speaker_role: 'moderator', 'participant', etc.
        content: Message content
        **metadata: Additional metadata to store

    Returns:
        BaseMessage compatible with LangGraph add_messages reducer
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Ensure speaker_role is stored in metadata for later retrieval
    metadata["speaker_role"] = speaker_role

    if speaker_role == "moderator":
        return HumanMessage(content=content, additional_kwargs=metadata)
    else:
        return AIMessage(content=content, additional_kwargs=metadata)


class DiscussionRoundNode(HITLNode):
    """Unified discussion round node with configurable user participation timing.

    This node handles complete discussion rounds with three configurable phases:
    1. Pre-agent user participation (if strategy allows)
    2. Agent discussion execution
    3. Post-agent user participation (if strategy allows)

    The participation timing is controlled by the UserParticipationStrategy,
    enabling easy switching between start-of-round, end-of-round, or disabled
    participation without requiring graph structure changes.

    Key features:
    - Integrates with existing FlowStateManager for round state management
    - Uses MessageCoordinator for agent context assembly
    - Preserves all existing agent discussion logic and error handling
    - Supports atmospheric effects and UI display
    - Maintains relevance checking and moderator warnings
    """

    def __init__(
        self,
        flow_state_manager: FlowStateManager,
        discussing_agents: List[LLMAgent],
        specialized_agents: Dict[str, LLMAgent],
        participation_strategy: UserParticipationStrategy,
        node_dependencies: Optional[NodeDependencies] = None,
    ):
        """Initialize the discussion round node.

        Args:
            flow_state_manager: FlowStateManager for round state operations
            discussing_agents: List of discussion agents that participate in rounds
            specialized_agents: Dictionary of specialized agents (moderator, etc.)
            participation_strategy: Strategy for user participation timing
            node_dependencies: Optional node dependencies for dependency injection
        """
        super().__init__(node_dependencies)
        self.flow_state_manager = flow_state_manager
        self.discussing_agents = discussing_agents
        self.specialized_agents = specialized_agents
        self.participation_strategy = participation_strategy

        logger.info(
            f"Initialized DiscussionRoundNode with {participation_strategy.get_timing_name()}"
        )

    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate preconditions for discussion round execution.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met, False otherwise
        """
        # Check for required state elements
        required_keys = ["active_topic"]
        if not self._validate_required_keys(state, required_keys):
            return False

        # Validate agents are available
        if not self.discussing_agents:
            self._validation_errors.append("No discussing agents available")
            return False

        if "moderator" not in self.specialized_agents:
            self._validation_errors.append("Moderator agent not available")
            return False

        # Validate flow state manager
        if not self.flow_state_manager:
            self._validation_errors.append("FlowStateManager not available")
            return False

        # Validate participation strategy
        if not self.participation_strategy:
            self._validation_errors.append("UserParticipationStrategy not available")
            return False

        return True

    def create_interrupt_payload(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Create interrupt payload for user participation.

        Args:
            state: Current Virtual Agora state

        Returns:
            Dictionary containing interrupt data for the UI
        """
        # Get participation context from strategy
        context = self.participation_strategy.get_participation_context(state)

        # Build interrupt payload
        current_round = state.get("current_round", 0)
        current_topic = state.get("active_topic", "Unknown Topic")

        # Get previous round summary if available
        previous_round_summary = None
        round_summaries = state.get("round_summaries", [])
        if round_summaries and current_round > 1:
            for summary in reversed(round_summaries):
                if isinstance(summary, dict) and summary.get("topic") == current_topic:
                    previous_round_summary = summary.get("summary", "")
                    break

        return {
            "type": "user_turn_participation",
            "current_round": current_round,
            "current_topic": current_topic,
            "previous_summary": previous_round_summary,
            "message": context["message"],
            "options": ["continue", "participate", "finalize"],
            "timing": context["timing"],
            "participation_type": context["participation_type"],
            "round_phase": context["round_phase"],
        }

    def process_user_input(
        self, user_input: Dict[str, Any], state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Process user input and return state updates.

        Args:
            user_input: User's response from the interrupt
            state: Current Virtual Agora state

        Returns:
            State updates based on user input
        """
        updates = {}

        if user_input is None:
            logger.warning("User input is None, treating as 'continue'")
            return updates

        action = user_input.get("action", "continue")
        current_round = state.get("current_round", 0)
        current_topic = state.get("active_topic", "Unknown Topic")

        logger.info(f"Processing user turn participation: action={action}")

        # Handle user participation message
        if action == "participate":
            user_message = user_input.get("message", "").strip()
            if user_message:
                # Determine if this is before or after agent discussion
                timing_phase = user_input.get("timing_phase", "unknown")
                use_next_round = (
                    timing_phase == "after_agents"
                )  # End-of-round uses next round

                # Apply user participation using FlowStateManager
                user_message_updates = self.flow_state_manager.apply_user_participation(
                    user_input=user_message,
                    state=state,
                    current_round=current_round,
                    topic=current_topic,
                    participation_type="user_turn_participation",
                    use_next_round=use_next_round,
                )

                updates.update(user_message_updates)
                logger.info(
                    f"Added user participation message for round {current_round}"
                )

        # Set routing flags
        if action == "finalize":
            updates["user_forced_conclusion"] = True
            updates["user_finalize_reason"] = (
                "User chose to finalize during turn participation"
            )

        logger.info(f"User turn participation updates: {updates}")
        return updates

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute complete discussion round with configurable user participation.

        This method implements the three-phase discussion round:
        1. Pre-agent user participation (if strategy allows)
        2. Agent discussion execution
        3. Post-agent user participation (if strategy allows)

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates from the complete discussion round
        """
        logger.debug("=== FLOW DEBUG: Entering DiscussionRoundNode.execute ===")
        logger.debug(f"State keys available: {list(state.keys())}")

        updates = {}
        working_state = state.copy()  # Working copy for progressive updates

        # Phase 1: Pre-agent user participation (if strategy allows)
        if self.participation_strategy.should_request_participation_before_agents(
            state
        ):
            logger.info("Requesting user participation before agents")
            try:
                participation_updates = self._handle_user_participation(
                    working_state, "before_agents"
                )
                updates.update(participation_updates)
                # Update working state with user input for agent context
                working_state.update(participation_updates)
            except GraphInterrupt:
                # Allow GraphInterrupt to propagate for HITL functionality
                # This is critical for user input flow to work correctly
                raise
            except Exception as e:
                logger.error(f"Error in pre-agent user participation: {e}")
                return self.handle_error(e, state)

        # Phase 2: Execute agent discussions
        try:
            discussion_updates = self._execute_agent_discussions(working_state)
            updates.update(discussion_updates)
            working_state.update(discussion_updates)
        except Exception as e:
            logger.error(f"Error in agent discussions: {e}")
            return self.handle_error(e, state)

        # Phase 3: Post-agent user participation (if strategy allows)
        if self.participation_strategy.should_request_participation_after_agents(
            working_state
        ):
            logger.info("Requesting user participation after agents")
            try:
                participation_updates = self._handle_user_participation(
                    working_state, "after_agents"
                )
                updates.update(participation_updates)
            except GraphInterrupt:
                # Allow GraphInterrupt to propagate for HITL functionality
                # This is critical for user input flow to work correctly
                raise
            except Exception as e:
                logger.error(f"Error in post-agent user participation: {e}")
                return self.handle_error(e, state)

        logger.info(f"Completed discussion round with updates: {list(updates.keys())}")
        return updates

    def _handle_user_participation(
        self, state: VirtualAgoraState, timing: str
    ) -> Dict[str, Any]:
        """Handle user participation with proper context and timing.

        Args:
            state: Current Virtual Agora state
            timing: Either "before_agents" or "after_agents"

        Returns:
            State updates from user participation
        """
        # Create interrupt payload
        interrupt_payload = self.create_interrupt_payload(state)
        interrupt_payload["timing_phase"] = timing

        logger.info(
            f"Interrupting for user turn participation ({timing}): {interrupt_payload}"
        )

        # Use LangGraph interrupt mechanism
        user_input = interrupt(interrupt_payload)
        logger.info(f"User turn participation input received: {user_input}")

        # Process user input
        return self.process_user_input(user_input, state)

    def _execute_agent_discussions(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute agent discussions with proper context and error handling.

        This method extracts and preserves all the existing agent discussion
        logic from nodes_v13.py, including context assembly, relevance checking,
        moderator warnings, and atmospheric effects.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates from agent discussions
        """
        logger.info("Executing agent discussions")

        # Use FlowStateManager to prepare round state
        try:
            round_state = self.flow_state_manager.prepare_round_state(state)
            current_round = round_state.round_number
            current_topic = round_state.current_topic
            speaking_order = round_state.speaking_order
            theme = round_state.theme
        except ValueError as e:
            logger.error(f"Failed to prepare round state: {e}")
            raise

        # Initialize speaking order if empty
        if not speaking_order:
            speaking_order = [agent.agent_id for agent in self.discussing_agents]
            random.shuffle(speaking_order)
            logger.debug(f"Initialized speaking order: {speaking_order}")
            # Update the round state with the initialized speaking order
            round_state = round_state._replace(speaking_order=speaking_order)

        moderator = self.specialized_agents.get("moderator")
        if not moderator:
            raise ValueError("No moderator agent available")

        # Get round summaries for context
        round_summaries = state.get("round_summaries", [])
        topic_summaries = [
            s
            for s in round_summaries
            if isinstance(s, dict) and s.get("topic") == current_topic
        ]

        # Show round announcement with atmospheric effects if enabled
        try:
            from virtual_agora.ui.display_modes import should_show_atmospheric_elements
            from virtual_agora.ui.atmospheric import show_round_announcement

            if should_show_atmospheric_elements():
                show_round_announcement(current_round, current_topic, speaking_order)
        except ImportError:
            pass

        # Execute the round
        round_messages = []

        for i, agent_id in enumerate(speaking_order):
            try:
                # Find the agent
                agent = next(
                    a for a in self.discussing_agents if a.agent_id == agent_id
                )

                # Validate context completeness before building context
                is_valid, missing_elements, completeness_score = validate_agent_context(
                    theme, current_topic, topic_summaries, round_messages
                )

                if not is_valid:
                    logger.warning(
                        f"Context validation failed for agent {agent_id}. "
                        f"Missing: {missing_elements}. Completeness: {completeness_score:.2f}"
                    )
                else:
                    logger.debug(
                        f"Context validation passed for agent {agent_id}. Completeness: 100%"
                    )

                # Build context using MessageCoordinator for centralized message assembly
                try:
                    # Use MessageCoordinator to assemble agent context
                    context_text, context_messages = (
                        self.flow_state_manager.message_coordinator.assemble_agent_context(
                            agent_id=agent_id,
                            round_num=current_round,
                            state=state,
                            agent_position=i,
                            speaking_order=speaking_order,
                            topic=current_topic,
                            system_prompt=(
                                agent.system_prompt
                                if hasattr(agent, "system_prompt")
                                else ""
                            ),
                            current_round_messages=round_messages,  # Pass current round messages for colleague context
                        )
                    )

                    # Build assembly-style prompt with coordinated context
                    assembly_context = f"""
You are participating in a democratic assembly discussing the topic: '{current_topic}'

Assembly Context:
{context_text}

Other assembly members and the Round Moderator have shared their perspectives before you. Listen to their viewpoints, then provide your own contribution to this democratic deliberation.

Instructions:
- Be strong in your convictions and opinions, even if others disagree
- Your goal is collaborative discussion toward shared understanding and actionable conclusions
- Build upon, challenge, or expand on previous speakers' points, always driving toward resolution
- Pay special attention to guidance from the Round Moderator, who may provide direction or clarification
- Actively work to identify consensus, highlight key decisions, and propose concrete next steps
- Maintain respectful but firm discourse as befits a democratic assembly focused on reaching decisions

Please provide your thoughts on '{current_topic}'. Provide a thorough and collaborative contribution that engages deeply with previous speakers, develops key points comprehensively, and works together with other agents to advance our democratic deliberation toward meaningful conclusions. Take whatever time and space you need to fully express your analysis and contribute to our collective understanding."""

                    context = assembly_context

                    # Log context assembly success
                    logger.info(
                        f"Built coordinated context for {agent_id}: "
                        f"{len(context_messages)} context messages, "
                        f"{len(context_text)} chars"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to build coordinated context for {agent_id}: {e}"
                    )
                    logger.error("Falling back to basic context assembly")

                    # Fallback to minimal context assembly
                    context_messages = []
                    context = f"""
You are participating in a democratic assembly discussing the topic: '{current_topic}'

Assembly Context:
- Theme: {theme}
- Round: {current_round}
- Your speaking position: {i + 1}/{len(speaking_order)}

Please provide your thoughts on '{current_topic}'. Provide a thorough and collaborative contribution.
"""

                # Get agent response with retry logic
                logger.debug(
                    f"Calling agent {agent_id} with {len(context_messages)} context messages"
                )
                if context_messages:
                    logger.debug(
                        f"Sample context message for agent call: {context_messages[0].content[:100] if hasattr(context_messages[0], 'content') else str(context_messages[0])[:100]}..."
                    )

                response_dict = retry_agent_call(
                    agent, state, context, context_messages
                )

                if response_dict is None:
                    # All retry attempts failed
                    response_content = (
                        f"[Failed to get response from {agent_id} after retries]"
                    )
                    # Display the failure for transparency
                    provider_type = get_provider_type_from_agent_id(agent_id)

                    from virtual_agora.ui.discussion_display import (
                        display_agent_response,
                    )

                    display_agent_response(
                        agent_id=agent_id,
                        provider=provider_type,
                        content=f"⚠️ Connection failed after multiple attempts",
                        round_number=current_round,
                        topic=current_topic,
                        timestamp=datetime.now(),
                    )
                else:
                    # Extract response with enhanced type checking
                    response_content = None

                    # Ensure response_dict is a dictionary
                    if not isinstance(response_dict, dict):
                        logger.warning(
                            f"Expected dict response from {agent_id}, got {type(response_dict)}: {response_dict}"
                        )
                        response_content = (
                            str(response_dict)
                            if response_dict
                            else f"[Invalid response from {agent_id}]"
                        )
                    else:
                        # Extract messages from dictionary
                        messages = response_dict.get("messages", [])
                        if messages:
                            # Get the last message content
                            last_message = messages[-1]
                            if hasattr(last_message, "content"):
                                response_content = last_message.content
                            elif (
                                isinstance(last_message, dict)
                                and "content" in last_message
                            ):
                                response_content = last_message["content"]
                            elif isinstance(last_message, str):
                                response_content = last_message
                            else:
                                response_content = str(last_message)
                        else:
                            # No messages in response
                            response_content = f"[No response from {agent_id}]"

                    # Validate response_content is not empty
                    if not response_content or not response_content.strip():
                        response_content = f"[Empty response from {agent_id}]"
                        # Display no response case
                        provider_type = get_provider_type_from_agent_id(agent_id)

                        from virtual_agora.ui.discussion_display import (
                            display_agent_response,
                        )

                        display_agent_response(
                            agent_id=agent_id,
                            provider=provider_type,
                            content=f"⚠️ No response received",
                            round_number=current_round,
                            topic=current_topic,
                            timestamp=datetime.now(),
                        )

                # Check relevance with moderator - add safety check
                relevance_check = {
                    "is_relevant": True,
                    "relevance_score": 1.0,
                }  # Default safe values
                try:
                    if response_content and response_content.strip():
                        relevance_check = moderator.evaluate_message_relevance(
                            response_content, current_topic
                        )
                    else:
                        logger.warning(
                            f"Empty response content for {agent_id}, skipping relevance check"
                        )
                        relevance_check = {"is_relevant": False, "relevance_score": 0.0}
                except Exception as relevance_error:
                    logger.error(
                        f"Relevance check failed for {agent_id}: {relevance_error}"
                    )
                    # Continue with default relevance check to avoid breaking the flow

                if relevance_check["is_relevant"]:
                    # Display the agent response in assembly-style panel
                    provider_type = get_provider_type_from_agent_id(agent_id)

                    from virtual_agora.ui.discussion_display import (
                        display_agent_response,
                    )

                    display_agent_response(
                        agent_id=agent_id,
                        provider=provider_type,
                        content=response_content,
                        round_number=current_round,
                        topic=current_topic,
                        timestamp=datetime.now(),
                    )

                    # Create LangChain-compatible message with safety checks
                    try:
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
                    except Exception as message_error:
                        logger.error(
                            f"Failed to create message for {agent_id}: {message_error}"
                        )
                        # Create a fallback simple message
                        from langchain_core.messages import HumanMessage

                        fallback_message = HumanMessage(
                            content=f"[{agent_id}]: {response_content}"
                        )
                        round_messages.append(fallback_message)
                    logger.info(f"Agent {agent_id} provided relevant response")
                else:
                    # Display the irrelevant response with a warning indicator
                    provider_type = get_provider_type_from_agent_id(agent_id)

                    from virtual_agora.ui.discussion_display import (
                        display_agent_response,
                    )

                    display_agent_response(
                        agent_id=agent_id,
                        provider=provider_type,
                        content=f"⚠️ Off-topic response: {response_content[:500]}{'...' if len(response_content) > 500 else ''}",
                        round_number=current_round,
                        topic=current_topic,
                        timestamp=datetime.now(),
                    )

                    # Handle irrelevant response
                    moderator.track_relevance_violation(
                        agent_id, response_content, relevance_check
                    )
                    warning = moderator.issue_relevance_warning(
                        agent_id, relevance_check
                    )
                    logger.warning(f"Agent {agent_id} provided irrelevant response")

                    # Add off-topic message to round messages for summary purposes
                    from virtual_agora.state.schema import Message

                    off_topic_message = Message(
                        id=str(uuid.uuid4()),
                        speaker_id=agent_id,
                        speaker_role="participant",
                        content=f"[OFF-TOPIC] {response_content}",
                        timestamp=datetime.now().isoformat(),
                        phase=2,
                        round=current_round,
                        topic=current_topic,
                        turn_order=i + 1,
                        relevance_score=relevance_check.get("relevance_score", 0.0),
                    )
                    round_messages.append(off_topic_message)

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

                from virtual_agora.ui.discussion_display import display_agent_response

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

        # Use FlowStateManager to finalize round state
        updates = self.flow_state_manager.finalize_round(
            state=state, round_state=round_state, round_messages=round_messages
        )

        logger.info(
            f"Completed round {current_round} with {len(round_messages)} messages"
        )
        return updates

    def get_node_name(self) -> str:
        """Get human-readable node name.

        Returns:
            Node name with participation strategy information
        """
        strategy_name = (
            self.participation_strategy.get_timing_name()
            if self.participation_strategy
            else "Unknown"
        )
        return f"DiscussionRoundNode({strategy_name})"
