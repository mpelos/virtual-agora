"""Main LangGraph implementation for Virtual Agora discussion flow.

This module defines the core graph structure with all phases, nodes, and
conditional edges according to the project specification.
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
from virtual_agora.providers import create_provider

from .nodes import FlowNodes
from .edges import FlowConditions

logger = get_logger(__name__)


class VirtualAgoraFlow:
    """Manages the complete LangGraph discussion flow for Virtual Agora."""

    def __init__(self, config: VirtualAgoraConfig, enable_monitoring: bool = True):
        """Initialize the flow with configuration.

        Args:
            config: Virtual Agora configuration
            enable_monitoring: Whether to enable flow monitoring and debugging
        """
        self.config = config
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

        # Initialize agents
        self.agents: Dict[str, LLMAgent] = {}
        self._initialize_agents()

        # Initialize flow components
        self.nodes = FlowNodes(self.agents, self.state_manager)
        self.conditions = FlowConditions()

    def _initialize_agents(self) -> None:
        """Initialize LLM agents from configuration."""
        # Create moderator
        moderator_llm = create_provider(
            provider=self.config.moderator.provider.value,
            model=self.config.moderator.model,
            temperature=getattr(self.config.moderator, "temperature", 0.7),
            max_tokens=getattr(self.config.moderator, "max_tokens", None),
        )
        self.agents["moderator"] = LLMAgent(
            agent_id="moderator", llm=moderator_llm, role="moderator"
        )
        logger.info("Initialized moderator agent")

        # Create participant agents
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
                self.agents[agent_id] = LLMAgent(
                    agent_id=agent_id, llm=provider_llm, role="participant"
                )
                logger.info(f"Initialized participant agent: {agent_id}")

    def build_graph(self) -> StateGraph:
        """Build the Virtual Agora discussion flow graph.

        Returns:
            Configured state graph
        """
        # Create graph with our state schema
        graph = StateGraph(VirtualAgoraState)

        # Phase 0: Initialization
        graph.add_node("initialization", self.nodes.initialization_node)

        # Phase 1: Agenda Setting
        graph.add_node("agenda_setting", self.nodes.agenda_setting_node)
        graph.add_node("agenda_approval", self.nodes.agenda_approval_node)

        # Phase 2: Discussion
        graph.add_node("discussion_round", self.nodes.discussion_round_node)
        graph.add_node("round_summary", self.nodes.round_summary_node)

        # Phase 3: Topic Conclusion
        graph.add_node("conclusion_poll", self.nodes.conclusion_poll_node)
        graph.add_node(
            "minority_considerations", self.nodes.minority_considerations_node
        )
        graph.add_node("topic_summary", self.nodes.topic_summary_node)

        # Phase 4: Agenda Re-evaluation
        graph.add_node("continuation_approval", self.nodes.continuation_approval_node)
        graph.add_node("agenda_modification", self.nodes.agenda_modification_node)

        # Phase 5: Final Report Generation
        graph.add_node("report_generation", self.nodes.report_generation_node)

        # Define the flow with edges
        graph.add_edge(START, "initialization")
        graph.add_edge("initialization", "agenda_setting")

        # Agenda setting flow
        graph.add_edge("agenda_setting", "agenda_approval")
        graph.add_conditional_edges(
            "agenda_approval",
            self.conditions.should_start_discussion,
            {
                "discussion": "discussion_round",
                "agenda_setting": "agenda_setting",  # If rejected, go back
            },
        )

        # Discussion flow with rounds
        graph.add_edge("discussion_round", "round_summary")
        graph.add_conditional_edges(
            "round_summary",
            self.conditions.should_start_conclusion_poll,
            {
                "continue_discussion": "discussion_round",
                "conclusion_poll": "conclusion_poll",
            },
        )

        # Topic conclusion flow
        graph.add_conditional_edges(
            "conclusion_poll",
            self.conditions.evaluate_conclusion_vote,
            {
                "continue_discussion": "discussion_round",
                "minority_considerations": "minority_considerations",
            },
        )
        graph.add_edge("minority_considerations", "topic_summary")
        graph.add_edge("topic_summary", "continuation_approval")

        # Agenda re-evaluation flow
        graph.add_conditional_edges(
            "continuation_approval",
            self.conditions.should_continue_session,
            {
                "continue": "agenda_modification",
                "end": "report_generation",
            },
        )

        graph.add_conditional_edges(
            "agenda_modification",
            self.conditions.has_topics_remaining,
            {
                "next_topic": "discussion_round",
                "generate_report": "report_generation",
                "re_evaluate_agenda": "agenda_setting",
            },
        )

        # Final report generation
        graph.add_edge("report_generation", END)

        self.graph = graph
        return graph

    def compile(self) -> Any:
        """Compile the graph with checkpointing.

        Returns:
            Compiled graph ready for execution
        """
        if self.graph is None:
            self.build_graph()

        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        logger.info("Graph compiled with checkpointing enabled")

        return self.compiled_graph

    def create_session(
        self, session_id: Optional[str] = None, main_topic: Optional[str] = None
    ) -> str:
        """Create a new discussion session.

        Args:
            session_id: Optional session ID
            main_topic: Main discussion topic provided by user

        Returns:
            Session ID
        """
        # Initialize state through state manager
        initial_state = self.state_manager.initialize_state(session_id)
        session_id = initial_state["session_id"]

        # Add Epic 6 specific state initialization
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

        # Update state with Epic 6 additions
        epic6_updates = {
            "current_round": 0,
            "round_history": [],
            "turn_order_history": [],
            "rounds_per_topic": {},
            "hitl_state": hitl_state,
            "flow_control": flow_control,
        }

        # If main topic provided, store it for agenda setting
        if main_topic:
            epic6_updates["main_topic"] = main_topic

        self.state_manager.update_state(epic6_updates)

        # Ensure graph is compiled
        if self.compiled_graph is None:
            self.compile()

        # Start monitoring for this session
        if self.monitoring_enabled:
            self.start_monitoring(session_id)

        logger.info(f"Created session: {session_id}")
        return session_id

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

        # Get current state
        current_state = self.state_manager.get_snapshot()

        # Invoke graph
        result = self.compiled_graph.invoke(current_state, config)

        return result

    def stream(self, config: Optional[Dict[str, Any]] = None):
        """Stream graph execution."""
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")

        if config is None:
            config = {
                "configurable": {"thread_id": self.state_manager.state["session_id"]}
            }

        # Get current state
        current_state = self.state_manager.get_snapshot()

        # Stream graph execution
        for update in self.compiled_graph.stream(current_state, config):
            yield update

    def update_state(
        self, updates: Dict[str, Any], as_node: Optional[str] = None
    ) -> None:
        """Update the graph state."""
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")

        config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}

        # Update through graph
        self.compiled_graph.update_state(config, updates, as_node=as_node)

    def get_graph_state(self) -> Any:
        """Get the current graph state."""
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")

        config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        return self.compiled_graph.get_state(config)

    def get_state_history(self) -> list:
        """Get the state history from the checkpointer."""
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")

        config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        return list(self.compiled_graph.get_state_history(config))

    def visualize_graph(self) -> bytes:
        """Generate graph visualization as PNG bytes."""
        if self.graph is None:
            self.build_graph()

        return self.graph.get_graph().draw_mermaid_png()

    def create_recovery_point(
        self, description: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a recovery point for the current state.

        Args:
            description: Description of the recovery point
            metadata: Optional metadata

        Returns:
            Recovery point ID
        """
        if not isinstance(self.checkpointer, EnhancedMemorySaver):
            raise StateError("Enhanced checkpointer required for recovery points")

        current_state = self.state_manager.state
        recovery_point = self.checkpointer.create_recovery_point(
            session_id=current_state["session_id"],
            state=current_state,
            description=description,
            metadata=metadata,
        )

        logger.info(f"Created recovery point: {recovery_point.recovery_id}")
        return recovery_point.recovery_id

    def rollback_to_recovery_point(self, recovery_id: str) -> None:
        """Rollback to a specific recovery point.

        Args:
            recovery_id: Recovery point ID
        """
        if not isinstance(self.checkpointer, EnhancedMemorySaver):
            raise StateError("Enhanced checkpointer required for rollback")

        session_id = self.state_manager.state["session_id"]
        restored_state, recovery_point = self.checkpointer.rollback_to_recovery_point(
            session_id=session_id, recovery_id=recovery_id
        )

        # Update state manager with restored state
        self.state_manager._state = restored_state

        logger.info(f"Rolled back to recovery point: {recovery_point.description}")

    def get_recovery_points(self) -> List[Dict[str, Any]]:
        """Get all recovery points for the current session.

        Returns:
            List of recovery point information
        """
        if not isinstance(self.checkpointer, EnhancedMemorySaver):
            return []

        session_id = self.state_manager.state["session_id"]
        recovery_points = self.checkpointer.get_recovery_points(session_id)

        return [
            {
                "recovery_id": rp.recovery_id,
                "timestamp": rp.timestamp.isoformat(),
                "phase": rp.phase,
                "description": rp.description,
                "validation_passed": rp.validation_passed,
                "metadata": rp.recovery_metadata,
            }
            for rp in recovery_points
        ]

    def create_state_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a state snapshot.

        Args:
            metadata: Optional metadata

        Returns:
            Snapshot ID
        """
        if not isinstance(self.checkpointer, EnhancedMemorySaver):
            raise StateError("Enhanced checkpointer required for snapshots")

        current_state = self.state_manager.state
        snapshot = self.checkpointer.create_snapshot(
            session_id=current_state["session_id"],
            state=current_state,
            metadata=metadata,
        )

        logger.info(f"Created state snapshot: {snapshot.checkpoint_id}")
        return snapshot.checkpoint_id

    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence statistics.

        Returns:
            Persistence statistics
        """
        if not isinstance(self.checkpointer, EnhancedMemorySaver):
            return {"enhanced_persistence": False}

        stats = self.checkpointer.get_persistence_stats()
        stats["enhanced_persistence"] = True
        return stats

    def cleanup_old_data(self, retention_days: int = 7):
        """Clean up old snapshots and recovery points.

        Args:
            retention_days: Number of days to retain data
        """
        if isinstance(self.checkpointer, EnhancedMemorySaver):
            self.checkpointer.cleanup_old_snapshots(retention_days)
            logger.info(f"Cleaned up data older than {retention_days} days")

    # Monitoring and Debugging Methods

    def start_monitoring(self, session_id: str):
        """Start monitoring for a session.

        Args:
            session_id: Session identifier
        """
        if self.monitor:
            self.monitor.start_session(session_id)
            logger.info(f"Started monitoring for session: {session_id}")

    def create_debug_snapshot(
        self, current_node: Optional[str] = None
    ) -> Optional[str]:
        """Create a debug snapshot of current state.

        Args:
            current_node: Currently executing node

        Returns:
            Snapshot timestamp or None if monitoring disabled
        """
        if not self.monitor:
            return None

        current_state = self.state_manager.state
        snapshot = self.monitor.create_debug_snapshot(current_state, current_node)

        logger.info(f"Created debug snapshot at {snapshot.timestamp}")
        return snapshot.timestamp.isoformat()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and statistics.

        Returns:
            Performance metrics dictionary
        """
        if not self.monitor:
            return {"monitoring_enabled": False}

        return self.monitor.get_performance_report()

    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis and summary.

        Returns:
            Error analysis dictionary
        """
        if not self.monitor:
            return {"monitoring_enabled": False}

        return self.monitor.get_error_summary()

    def export_monitoring_data(self, include_snapshots: bool = True) -> Dict[str, Any]:
        """Export all monitoring and debugging data.

        Args:
            include_snapshots: Whether to include debug snapshots

        Returns:
            Complete monitoring data export
        """
        if not self.monitor:
            return {"monitoring_enabled": False}

        return self.monitor.export_debug_data(include_snapshots)

    def set_debug_breakpoint(self, name: str, condition_fn):
        """Set a debug breakpoint with condition.

        Args:
            name: Breakpoint name
            condition_fn: Function that takes state and returns bool
        """
        if self.debugger:
            self.debugger.set_breakpoint(name, condition_fn)
            logger.info(f"Set debug breakpoint: {name}")

    def add_debug_watch(self, name: str, expression_fn):
        """Add a debug watch expression.

        Args:
            name: Watch name
            expression_fn: Function that extracts value from state
        """
        if self.debugger:
            self.debugger.add_watch(name, expression_fn)
            logger.info(f"Added debug watch: {name}")

    def enable_debug_mode(self):
        """Enable debug mode for detailed monitoring."""
        if self.debugger:
            self.debugger.enable_debug_mode()
            logger.info("Debug mode enabled")

    def disable_debug_mode(self):
        """Disable debug mode."""
        if self.debugger:
            self.debugger.disable_debug_mode()
            logger.info("Debug mode disabled")

    def get_debug_status(self) -> Dict[str, Any]:
        """Get current debug status and configuration.

        Returns:
            Debug status dictionary
        """
        if not self.debugger:
            return {"debug_available": False}

        status = self.debugger.get_debug_status()
        status["debug_available"] = True
        return status

    def check_debug_conditions(self, current_node: Optional[str] = None) -> List[str]:
        """Check debug breakpoints and conditions.

        Args:
            current_node: Currently executing node

        Returns:
            List of triggered breakpoint names
        """
        if not self.debugger:
            return []

        current_state = self.state_manager.state
        triggered = self.debugger.check_breakpoints(current_state, current_node)

        if triggered:
            logger.warning(f"Debug breakpoints triggered: {triggered}")

        return triggered

    def evaluate_debug_watches(self) -> Dict[str, Any]:
        """Evaluate all debug watch expressions.

        Returns:
            Dictionary of watch names to values
        """
        if not self.debugger:
            return {}

        current_state = self.state_manager.state
        return self.debugger.evaluate_watches(current_state)
