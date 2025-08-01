"""Virtual Agora v1.3 LangGraph implementation.

This module defines the node-centric graph structure where nodes
orchestrate specialized agents as tools.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig

from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.flow.persistence import (
    EnhancedMemorySaver,
    create_enhanced_checkpointer,
)
from virtual_agora.flow.monitoring import (
    FlowMonitor,
    FlowDebugger,
    create_flow_monitor,
    create_flow_debugger,
)
from virtual_agora.state.schema import VirtualAgoraState, HITLState, FlowControl
from virtual_agora.state.manager import StateManager
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import StateError
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.report_writer_agent import ReportWriterAgent
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.providers import create_provider

from .nodes_v13 import V13FlowNodes
from .edges_v13 import V13FlowConditions
from .interrupt_manager import get_interrupt_manager

logger = get_logger(__name__)


class VirtualAgoraV13Flow:
    """Manages the complete v1.3 LangGraph discussion flow.

    Key changes from v1.1:
    - Node-centric architecture (nodes orchestrate agents)
    - 5 specialized agents instead of single moderator with modes
    - Enhanced HITL with periodic stops
    - Dual polling system (agents + user)
    - More complex phase transitions
    """

    def __init__(
        self,
        config: VirtualAgoraConfig,
        enable_monitoring: bool = True,
        checkpoint_interval: int = 3,
    ):
        """Initialize the v1.3 flow with configuration.

        Args:
            config: Virtual Agora configuration
            enable_monitoring: Whether to enable flow monitoring and debugging
            checkpoint_interval: Number of rounds between periodic user checkpoints
        """
        self.config = config
        self.checkpoint_interval = checkpoint_interval
        self.state_manager = StateManager(config)
        self.graph: Optional[StateGraph] = None
        self.compiled_graph: Optional[Any] = None
        self.checkpointer = create_enhanced_checkpointer()

        # Initialize monitoring and debugging
        self.monitoring_enabled = enable_monitoring
        if enable_monitoring:
            self.monitor = create_flow_monitor()
            self.debugger = create_flow_debugger(self.monitor)
        else:
            self.monitor = None
            self.debugger = None

        # Initialize specialized agents
        self.specialized_agents: Dict[str, LLMAgent] = {}
        self.discussing_agents: List[LLMAgent] = []
        self._initialize_agents()

        # Initialize flow components with specialized agents
        self.nodes = V13FlowNodes(
            self.specialized_agents,
            self.discussing_agents,
            self.state_manager,
            self.checkpoint_interval,
        )
        self.conditions = V13FlowConditions(self.checkpoint_interval)

    def _initialize_agents(self) -> None:
        """Initialize all specialized agents and discussing agents."""
        # Create moderator (facilitation only in v1.3)
        moderator_llm = create_provider(
            provider=self.config.moderator.provider.value,
            model=self.config.moderator.model,
            temperature=getattr(self.config.moderator, "temperature", 0.7),
            max_tokens=getattr(self.config.moderator, "max_tokens", None),
        )
        self.specialized_agents["moderator"] = ModeratorAgent(
            agent_id="moderator", llm=moderator_llm
        )
        logger.debug("Initialized moderator agent")

        # Create summarizer
        if hasattr(self.config, "summarizer"):
            summarizer_llm = create_provider(
                provider=self.config.summarizer.provider.value,
                model=self.config.summarizer.model,
                temperature=getattr(self.config.summarizer, "temperature", 0.7),
                max_tokens=getattr(self.config.summarizer, "max_tokens", None),
            )
            self.specialized_agents["summarizer"] = SummarizerAgent(
                agent_id="summarizer", llm=summarizer_llm
            )
            logger.debug("Initialized summarizer agent")
        else:
            # Fallback: use moderator config for summarizer
            logger.warning("No summarizer config found, using moderator config")
            self.specialized_agents["summarizer"] = SummarizerAgent(
                agent_id="summarizer", llm=moderator_llm
            )

        # Create report writer agent
        if hasattr(self.config, "report_writer"):
            report_writer_llm = create_provider(
                provider=self.config.report_writer.provider.value,
                model=self.config.report_writer.model,
                temperature=getattr(self.config.report_writer, "temperature", 0.6),
                max_tokens=getattr(self.config.report_writer, "max_tokens", None),
            )
            self.specialized_agents["report_writer"] = ReportWriterAgent(
                agent_id="report_writer", llm=report_writer_llm
            )
            logger.debug("Initialized report writer agent")
        else:
            # Fallback
            logger.warning("No report_writer config found, using moderator config")
            self.specialized_agents["report_writer"] = ReportWriterAgent(
                agent_id="report_writer", llm=moderator_llm
            )

        # Create discussing agents
        agent_counter = 0
        for agent_config in self.config.agents:
            provider_llm = create_provider(
                provider=agent_config.provider.value,
                model=agent_config.model,
                temperature=getattr(agent_config, "temperature", 0.7),
                max_tokens=getattr(agent_config, "max_tokens", None),
            )

            for i in range(agent_config.count):
                agent_counter += 1
                agent_id = f"{agent_config.model}_{agent_counter}"

                # Use DiscussionAgent if available, otherwise LLMAgent
                try:
                    agent = DiscussionAgent(agent_id=agent_id, llm=provider_llm)
                except:
                    agent = LLMAgent(
                        agent_id=agent_id, llm=provider_llm, role="participant"
                    )

                self.discussing_agents.append(agent)
                logger.debug(f"Initialized discussing agent: {agent_id}")

    def build_graph(self) -> StateGraph:
        """Build the v1.3 Virtual Agora discussion flow graph.

        Returns:
            Configured state graph matching v1.3 specification
        """
        # Create graph with our state schema
        graph = StateGraph(VirtualAgoraState)

        # ===== Phase 0: Initialization Nodes =====
        graph.add_node("config_and_keys", self.nodes.config_and_keys_node)
        graph.add_node("agent_instantiation", self.nodes.agent_instantiation_node)
        graph.add_node("get_theme", self.nodes.get_theme_node)

        # ===== Phase 1: Agenda Setting Nodes =====
        graph.add_node("agenda_proposal", self.nodes.agenda_proposal_node)
        graph.add_node("topic_refinement", self.nodes.topic_refinement_node)
        graph.add_node("collate_proposals", self.nodes.collate_proposals_node)
        graph.add_node("agenda_voting", self.nodes.agenda_voting_node)
        graph.add_node("synthesize_agenda", self.nodes.synthesize_agenda_node)
        graph.add_node("agenda_approval", self.nodes.agenda_approval_node)

        # ===== Phase 2: Discussion Loop Nodes =====
        graph.add_node("announce_item", self.nodes.announce_item_node)
        graph.add_node("discussion_round", self.nodes.discussion_round_node)
        graph.add_node("round_summarization", self.nodes.round_summarization_node)
        graph.add_node(
            "user_turn_participation", self.nodes.user_turn_participation_node
        )
        graph.add_node("end_topic_poll", self.nodes.end_topic_poll_node)
        graph.add_node("periodic_user_stop", self.nodes.periodic_user_stop_node)

        # ===== Phase 3: Topic Conclusion Nodes =====
        graph.add_node("final_considerations", self.nodes.final_considerations_node)
        graph.add_node(
            "topic_report_generation", self.nodes.topic_report_generation_node
        )
        graph.add_node(
            "topic_summary_generation", self.nodes.topic_summary_generation_node
        )
        graph.add_node("file_output", self.nodes.file_output_node)

        # ===== Phase 4: Continuation Nodes =====
        graph.add_node("agent_poll", self.nodes.agent_poll_node)
        graph.add_node("user_approval", self.nodes.user_approval_node)
        graph.add_node("agenda_modification", self.nodes.agenda_modification_node)

        # ===== Phase 5: Final Report Nodes =====
        graph.add_node("final_report_generation", self.nodes.final_report_node)
        graph.add_node("multi_file_output", self.nodes.multi_file_output_node)

        # ===== Define the flow with edges =====

        # Start -> Phase 0
        graph.add_edge(START, "config_and_keys")
        graph.add_edge("config_and_keys", "agent_instantiation")
        graph.add_edge("agent_instantiation", "get_theme")

        # Conditional: Route based on user topic preference
        graph.add_conditional_edges(
            "get_theme",
            self.conditions.theme_routing_decision,
            {
                "agent_agenda": "agenda_proposal",  # Traditional agent-driven flow
                "user_agenda": "agenda_proposal",  # Fallback (shouldn't be used)
                "discussion": "announce_item",  # Skip directly to discussion
            },
        )

        # Phase 1: Agenda Setting Flow
        graph.add_edge("agenda_proposal", "topic_refinement")
        graph.add_edge("topic_refinement", "collate_proposals")
        graph.add_edge("collate_proposals", "agenda_voting")
        graph.add_edge("agenda_voting", "synthesize_agenda")
        graph.add_edge("synthesize_agenda", "agenda_approval")

        # Conditional: Agenda approval
        graph.add_conditional_edges(
            "agenda_approval",
            self.conditions.should_start_discussion,
            {
                "discussion": "announce_item",
                "agenda_setting": "agenda_proposal",  # Loop back if rejected
            },
        )

        # Phase 2: Discussion Flow
        graph.add_edge("announce_item", "discussion_round")
        graph.add_edge("discussion_round", "round_summarization")

        # Conditional: Check if user participation should be shown (round >= 2)
        graph.add_conditional_edges(
            "round_summarization",
            self.conditions.should_show_user_participation,
            {
                "user_participation": "user_turn_participation",
                "continue_flow": "end_topic_poll",  # Skip user participation for round 1
            },
        )

        # Handle user turn decision
        graph.add_conditional_edges(
            "user_turn_participation",
            self.conditions.evaluate_user_turn_decision,
            {
                "continue_discussion": "end_topic_poll",
                "conclude_topic": "final_considerations",
            },
        )

        # Conditional: Check for periodic stop (round % 5 == 0) first
        graph.add_conditional_edges(
            "end_topic_poll",
            self.conditions.check_periodic_stop,
            {
                "periodic_stop": "periodic_user_stop",
                "check_votes": "vote_evaluation",
            },
        )

        # Add a intermediate node for vote evaluation to simplify routing
        graph.add_node("vote_evaluation", self.nodes.vote_evaluation_node)
        graph.add_conditional_edges(
            "vote_evaluation",
            self.conditions.evaluate_conclusion_vote,
            {
                "continue_discussion": "discussion_round",
                "conclude_topic": "final_considerations",
            },
        )

        # Handle periodic stop result
        graph.add_conditional_edges(
            "periodic_user_stop",
            self.conditions.evaluate_conclusion_vote,
            {
                "continue_discussion": "discussion_round",
                "conclude_topic": "final_considerations",
            },
        )

        # Phase 3: Topic Conclusion Flow
        graph.add_edge("final_considerations", "topic_report_generation")
        graph.add_edge("topic_report_generation", "topic_summary_generation")
        graph.add_edge("topic_summary_generation", "file_output")
        graph.add_edge("file_output", "agent_poll")

        # Phase 4: Continuation Logic
        # Agent poll for session end
        graph.add_conditional_edges(
            "agent_poll",
            self.conditions.check_agent_session_vote,
            {
                "end_session": "final_report_generation",
                "check_user": "user_approval",
            },
        )

        # User approval with agenda check - enhanced to handle explicit final report requests
        graph.add_conditional_edges(
            "user_approval",
            self._evaluate_user_approval_routing,
            {
                "end_session": "final_report_generation",
                "no_items": "final_report_generation",
                "has_items": "announce_item",  # Go directly to next topic
                "modify_agenda": "agenda_modification",  # Only when user explicitly requests it
            },
        )

        # After agenda modification, check what to do
        graph.add_conditional_edges(
            "agenda_modification",
            self.conditions.should_modify_agenda,
            {
                "modify_agenda": "topic_refinement",  # Re-evaluate agenda with refinement
                "next_topic": "announce_item",  # Continue with next topic
            },
        )

        # Phase 5: Final Report
        graph.add_edge("final_report_generation", "multi_file_output")
        graph.add_edge("multi_file_output", END)

        self.graph = graph
        return graph

    def _evaluate_user_approval_routing(self, state: VirtualAgoraState) -> str:
        """Evaluate user approval routing with proper state validation.

        This replaces the complex lambda function with proper error handling
        and state validation to prevent the third iteration bug.

        Args:
            state: Current state

        Returns:
            Routing decision: "end_session", "no_items", "has_items", or "modify_agenda"
        """
        logger.info("=== USER APPROVAL ROUTING EVALUATION START ===")

        # LOG: Complete state context for routing decision
        routing_context = {
            "user_approves_continuation": state.get("user_approves_continuation"),
            "agents_vote_end_session": state.get("agents_vote_end_session"),
            "user_forced_conclusion": state.get("user_forced_conclusion"),
            "user_requested_modification": state.get("user_requested_modification"),
            "topic_queue": state.get("topic_queue", []),
            "active_topic": state.get("active_topic"),
            "completed_topics": state.get("completed_topics", []),
            "current_phase": state.get("current_phase"),
            "current_round": state.get("current_round", 0),
        }
        logger.info(f"Routing context: {routing_context}")

        # First check explicit session continuation evaluation
        logger.debug("Checking session continuation evaluation...")
        session_decision = self.conditions.evaluate_session_continuation(state)
        logger.info(f"Session continuation decision: {session_decision}")

        if session_decision == "end_session":
            logger.info(
                "ROUTING DECISION: end_session (from session continuation evaluation)"
            )
            return "end_session"

        # Check if user explicitly requested agenda modification
        if state.get("user_requested_modification", False):
            logger.info(
                "ROUTING DECISION: modify_agenda (user explicitly requested modification)"
            )
            return "modify_agenda"

        # Validate and clean topic_queue
        topic_queue = state.get("topic_queue", [])
        logger.debug(
            f"Raw topic_queue from state: {topic_queue} (Type: {type(topic_queue)})"
        )

        # Handle nested list corruption
        cleaned_queue = []
        corruption_detected = False

        for i, item in enumerate(topic_queue):
            logger.debug(
                f"Processing topic_queue item {i}: {item} (Type: {type(item)})"
            )
            if isinstance(item, list):
                logger.warning(f"Nested list detected at index {i}: {item}")
                corruption_detected = True
                for nested_item in item:
                    if isinstance(nested_item, str) and nested_item.strip():
                        cleaned_queue.append(nested_item.strip())
                        logger.debug(
                            f"Extracted from nested list: '{nested_item.strip()}'"
                        )
                cleaned_queue.extend(item)
            elif isinstance(item, str) and item.strip():
                cleaned_queue.append(item.strip())
                logger.debug(f"Added valid topic: '{item.strip()}'")
            else:
                logger.warning(
                    f"Invalid topic_queue item ignored: {item} (Type: {type(item)})"
                )

        # Helper function to truncate long topic titles for cleaner logs
        def truncate_topic(topic, max_length=60):
            if isinstance(topic, str) and len(topic) > max_length:
                return topic[:max_length] + "..."
            return topic

        def truncate_topic_list(topics):
            return [truncate_topic(topic) for topic in topics]

        # LOG: State validation results
        logger.info(f"=== TOPIC QUEUE VALIDATION ===")
        logger.debug(f"Original topic_queue: {topic_queue}")
        logger.debug(f"Cleaned topic_queue: {cleaned_queue}")
        logger.info(f"Original topic_queue: {truncate_topic_list(topic_queue)}")
        logger.info(f"Cleaned topic_queue: {truncate_topic_list(cleaned_queue)}")
        logger.info(f"Queue length: {len(topic_queue)} -> {len(cleaned_queue)}")
        logger.info(f"Corruption detected: {corruption_detected}")
        logger.info(f"Active topic: {truncate_topic(state.get('active_topic'))}")
        logger.info(
            f"Completed topics: {truncate_topic_list(state.get('completed_topics', []))}"
        )
        logger.info(f"Completed topics count: {len(state.get('completed_topics', []))}")

        # Update state with cleaned queue if corruption was detected
        if cleaned_queue != topic_queue:
            logger.error(f"CRITICAL: Topic queue corruption detected!")
            logger.error(f"  Original: {topic_queue}")
            logger.error(f"  Cleaned:  {cleaned_queue}")
            logger.error(
                "  Note: Cannot update state from routing function - this may cause issues!"
            )

        # Handle interrupt context cleanup for completed topic
        interrupt_manager = get_interrupt_manager()
        completed_topics = state.get("completed_topics", [])
        if completed_topics:
            # Get the most recently completed topic for cleanup
            last_completed_topic = (
                completed_topics[-1]
                if isinstance(completed_topics[-1], str)
                else str(completed_topics[-1])
            )
            logger.info(f"=== TOPIC COMPLETION CLEANUP ===")
            logger.info(
                f"Clearing interrupt context for completed topic: {last_completed_topic}"
            )
            interrupt_manager.clear_topic_context(last_completed_topic)

        # Make routing decision based on cleaned queue
        if not cleaned_queue:
            logger.info(
                "ROUTING DECISION: no_items (no topics remaining in cleaned queue)"
            )
            logger.info(
                f"Decision factors: cleaned_queue={cleaned_queue}, len={len(cleaned_queue)}"
            )
            return "no_items"
        else:
            # Set up interrupt context for next topic if we're continuing
            if cleaned_queue:
                next_topic = cleaned_queue[0]
                logger.info(f"=== PREPARING FOR NEXT TOPIC ===")
                logger.info(f"Resetting interrupt context for next topic: {next_topic}")
                interrupt_manager.reset_for_new_topic(next_topic)

            logger.info(
                f"ROUTING DECISION: has_items ({len(cleaned_queue)} topics remaining)"
            )
            logger.info(f"Remaining topics: {truncate_topic_list(cleaned_queue)}")
            return "has_items"

    def compile(self) -> Any:
        """Compile the graph with checkpointing.

        Returns:
            Compiled graph ready for execution
        """
        if self.graph is None:
            self.build_graph()

        # Debug: Log graph structure before compilation
        logger.debug("=== FLOW DEBUG: Graph structure before compilation ===")
        logger.debug(f"Graph nodes: {list(self.graph.nodes.keys())}")
        logger.debug(f"Graph edges: {len(self.graph.edges)} total edges")

        # Check if announce_item node exists
        if "announce_item" in self.graph.nodes:
            logger.debug("✅ announce_item node found in graph")
        else:
            logger.error("❌ announce_item node NOT found in graph")

        # Check if agenda_approval node exists
        if "agenda_approval" in self.graph.nodes:
            logger.debug("✅ agenda_approval node found in graph")
        else:
            logger.error("❌ agenda_approval node NOT found in graph")

        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        logger.info("V1.3 Graph compiled with checkpointing enabled")

        # Debug: Log compiled graph info
        logger.debug("=== FLOW DEBUG: Graph compiled successfully ===")
        logger.debug(f"Compiled graph type: {type(self.compiled_graph)}")

        return self.compiled_graph

    def create_session(
        self,
        session_id: Optional[str] = None,
        main_topic: Optional[str] = None,
        user_defines_topics: bool = False,
    ) -> str:
        """Create a new discussion session.

        Args:
            session_id: Optional session ID
            main_topic: Main discussion topic provided by user
            user_defines_topics: Whether user will define topics manually

        Returns:
            Session ID
        """
        # Initialize state through state manager
        initial_state = self.state_manager.initialize_state(session_id)
        session_id = initial_state["session_id"]

        # Add v1.3 specific state initialization
        flow_control = FlowControl(
            max_rounds_per_topic=10,
            auto_conclude_threshold=3,
            context_window_limit=8000,
            cycle_detection_enabled=True,
            max_iterations_per_phase=5,
        )

        hitl_state = HITLState(
            awaiting_approval=False,
            approval_type=None,
            prompt_message=None,
            options=None,
            approval_history=[],
        )

        # Update state with v1.3 additions
        v13_updates = {
            "current_round": 0,
            # "round_history": [], # Remove - uses safe_list_append reducer
            # "turn_order_history": [], # Remove - uses safe_list_append reducer
            "rounds_per_topic": {},
            "hitl_state": hitl_state,
            "flow_control": flow_control,
            "specialized_agents": {},  # Will be populated by agent_instantiation_node
            "agents": {},  # Will be populated by agent_instantiation_node
            # "vote_history": [], # Remove - uses safe_list_append reducer
            "periodic_stop_counter": 0,
            "checkpoint_interval": self.checkpoint_interval,
            "user_forced_conclusion": False,
            "agents_vote_end_session": False,
            "user_approves_continuation": True,
            "user_requested_modification": False,
        }

        # If main topic provided, store it for agenda setting
        if main_topic:
            v13_updates["main_topic"] = main_topic

        # Store user's topic definition preference
        v13_updates["user_defines_topics"] = user_defines_topics

        self.state_manager.update_state(v13_updates)

        # Ensure graph is compiled
        if self.compiled_graph is None:
            self.compile()

        # Initialize interrupt manager for this session
        interrupt_manager = get_interrupt_manager()
        interrupt_manager.set_session_context(session_id)
        logger.debug(f"Initialized interrupt manager for session: {session_id}")

        # Start monitoring for this session
        if self.monitoring_enabled:
            self.start_monitoring(session_id)

        logger.info(f"Created v1.3 session: {session_id}")
        return session_id

    # Inherit remaining methods from base class
    # (get_state_manager, invoke, stream, update_state, etc.)
    # These remain the same as in v1.1

    def get_state_manager(self) -> StateManager:
        """Get the state manager instance."""
        return self.state_manager

    def invoke(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the graph with current state."""
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")

        if config is None:
            config = {
                "configurable": {"thread_id": self.state_manager.state["session_id"]}
            }

        # For a fresh start, we need to pass the current state from state manager
        # The checkpointer will handle state persistence across the execution
        input_data = self.state_manager.state

        # Invoke graph with current state
        result = self.compiled_graph.invoke(input_data, config)

        return result

    def stream(
        self,
        config: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: bool = False,
    ):
        """Stream graph execution with enhanced interrupt handling.

        Args:
            config: Optional configuration dictionary
            resume_from_checkpoint: If True, resume from checkpoint (pass None as input_data).
                                  If False, start fresh (pass current state as input_data).
        """
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")

        if config is None:
            config = {
                "configurable": {"thread_id": self.state_manager.state["session_id"]}
            }

        logger.debug(f"VirtualAgoraV13Flow.stream called with config: {config}")
        logger.debug(f"Resume from checkpoint: {resume_from_checkpoint}")
        logger.debug(
            f"State manager session_id: {self.state_manager.state.get('session_id')}"
        )
        logger.debug(f"Compiled graph type: {type(self.compiled_graph)}")

        # For checkpoint resumption (after interrupts), pass None to continue from checkpoint
        # For fresh start, pass the current state from state manager
        if resume_from_checkpoint:
            input_data = None
            logger.debug(
                "=== FLOW DEBUG: Resuming from checkpoint (input_data=None) ==="
            )
        else:
            input_data = self.state_manager.state
            logger.debug("=== FLOW DEBUG: Starting fresh (input_data=state) ===")
            logger.debug(
                f"Input data for stream: session_id={input_data.get('session_id')}, keys_count={len(input_data.keys())}"
            )

        try:
            # Stream graph execution with current state
            logger.debug("Starting compiled_graph.stream execution")
            if input_data is not None:
                logger.debug(
                    f"=== FLOW DEBUG: Stream input data session_id: {input_data.get('session_id')}"
                )
            else:
                logger.debug(
                    "=== FLOW DEBUG: Stream input data: None (resuming from checkpoint)"
                )
            logger.debug(f"=== FLOW DEBUG: Stream config: {config}")

            # LOG: Initial state snapshot for debugging
            initial_state_snapshot = {
                "topic_queue": self.state_manager.state.get("topic_queue", []),
                "active_topic": self.state_manager.state.get("active_topic"),
                "completed_topics": self.state_manager.state.get(
                    "completed_topics", []
                ),
                "current_round": self.state_manager.state.get("current_round", 0),
                "current_phase": self.state_manager.state.get("current_phase", 0),
                "total_messages": len(self.state_manager.state.get("messages", [])),
                "session_id": self.state_manager.state.get("session_id"),
            }
            logger.info(f"=== STREAM START STATE SNAPSHOT ===")
            logger.debug(f"Initial state: {initial_state_snapshot}")

            update_count = 0
            for update in self.compiled_graph.stream(input_data, config):
                update_count += 1

                # LOG: Pre-processing state for each update
                pre_state_snapshot = {
                    "topic_queue": self.state_manager.state.get("topic_queue", []),
                    "active_topic": self.state_manager.state.get("active_topic"),
                    "completed_topics": self.state_manager.state.get(
                        "completed_topics", []
                    ),
                    "current_round": self.state_manager.state.get("current_round", 0),
                    "total_messages": len(self.state_manager.state.get("messages", [])),
                }
                logger.debug(f"=== PRE-UPDATE {update_count} STATE ===")
                logger.debug(f"Pre-state: {pre_state_snapshot}")
                logger.debug(f"Update received: {update}")

                # CRITICAL FIX: Sync state manager with graph state after each update
                # This prevents state corruption during interrupt processing
                if isinstance(update, dict) and update:
                    # Extract actual state updates from node results
                    # LangGraph returns {node_name: node_result} format
                    all_state_updates = {}
                    for node_name, node_result in update.items():
                        # Skip LangGraph internal keys
                        if node_name in ["__interrupt__", "__end__", "__nodes__"]:
                            continue
                        # Only process if node_result is a dict with state updates
                        if isinstance(node_result, dict):
                            logger.debug(f"=== NODE {node_name} RESULT ===")
                            logger.debug(f"Node result: {node_result}")
                            all_state_updates.update(node_result)

                    logger.debug(f"=== AGGREGATED STATE UPDATES ===")
                    logger.debug(f"All state updates to apply: {all_state_updates}")

                    if all_state_updates:
                        try:
                            self.state_manager.update_state(all_state_updates)
                            logger.debug(
                                f"✅ Synced state manager with {len(all_state_updates)} updates from nodes: {list(update.keys())}"
                            )
                        except Exception as sync_error:
                            logger.error(f"❌ Failed to sync state: {sync_error}")
                            logger.error(f"Failed updates: {all_state_updates}")

                # LOG: Post-processing state for each update
                post_state_snapshot = {
                    "topic_queue": self.state_manager.state.get("topic_queue", []),
                    "active_topic": self.state_manager.state.get("active_topic"),
                    "completed_topics": self.state_manager.state.get(
                        "completed_topics", []
                    ),
                    "current_round": self.state_manager.state.get("current_round", 0),
                    "total_messages": len(self.state_manager.state.get("messages", [])),
                }

                # LOG: State diff analysis
                state_changes = {}
                for key in pre_state_snapshot:
                    if pre_state_snapshot[key] != post_state_snapshot[key]:
                        state_changes[key] = {
                            "before": pre_state_snapshot[key],
                            "after": post_state_snapshot[key],
                        }

                logger.debug(f"=== POST-UPDATE {update_count} STATE ===")
                logger.debug(f"Post-state: {post_state_snapshot}")
                if state_changes:
                    logger.info(f"=== STATE CHANGES DETECTED ===")
                    for field, change in state_changes.items():
                        logger.info(
                            f"  {field}: {change['before']} → {change['after']}"
                        )
                else:
                    logger.debug("No state changes detected in critical fields")

                # Log specific node executions
                if isinstance(update, dict):
                    for key in update.keys():
                        if key not in ["__interrupt__", "__end__"]:
                            logger.info(f"=== EXECUTING NODE: {key} ===")

                yield update
            logger.debug(
                f"Stream execution completed successfully after {update_count} updates"
            )
        except Exception as e:
            logger.error(f"Error in stream execution: {e}", exc_info=True)
            logger.error(f"Error occurred with input_data: {input_data}")
            logger.error(f"Error occurred with config: {config}")
            raise

    def start_monitoring(self, session_id: str):
        """Start monitoring for a session."""
        if self.monitor:
            self.monitor.start_session(session_id)
            logger.debug(f"Started monitoring for v1.3 session: {session_id}")
