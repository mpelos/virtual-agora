"""LangGraph integration for Virtual Agora state management.

This module provides the integration between Virtual Agora's state management
and LangGraph's graph execution framework.
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import StateError


logger = get_logger(__name__)


class VirtualAgoraGraph:
    """Manages the LangGraph integration for Virtual Agora."""
    
    def __init__(self, config: VirtualAgoraConfig):
        """Initialize the graph with configuration.
        
        Args:
            config: Virtual Agora configuration
        """
        self.config = config
        self.state_manager = StateManager(config)
        self.graph: Optional[StateGraph] = None
        self.compiled_graph: Optional[Any] = None
        self.checkpointer = MemorySaver()
        
    def build_graph(self) -> StateGraph:
        """Build the Virtual Agora state graph.
        
        Returns:
            Configured state graph
        """
        # Create graph with our state schema
        graph = StateGraph(VirtualAgoraState)
        
        # Add nodes for each phase
        graph.add_node("initialization", self._initialization_node)
        graph.add_node("agenda_setting", self._agenda_setting_node)
        graph.add_node("discussion", self._discussion_node)
        graph.add_node("consensus", self._consensus_node)
        graph.add_node("summary", self._summary_node)
        
        # Add edges for phase transitions
        graph.add_edge(START, "initialization")
        graph.add_edge("initialization", "agenda_setting")
        graph.add_edge("agenda_setting", "discussion")
        
        # Discussion can loop or go to consensus
        graph.add_conditional_edges(
            "discussion",
            self._should_continue_discussion,
            {
                "continue": "discussion",
                "consensus": "consensus",
            }
        )
        
        # Consensus can go back to discussion or to summary
        graph.add_conditional_edges(
            "consensus",
            self._should_continue_topics,
            {
                "next_topic": "discussion",
                "summary": "summary",
            }
        )
        
        graph.add_edge("summary", END)
        
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
        logger.info("Graph compiled with in-memory checkpointer")
        
        return self.compiled_graph
    
    def _initialization_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Node for initialization phase.
        
        Args:
            state: Current state
            
        Returns:
            State updates
        """
        logger.debug("Executing initialization node")
        
        # The state manager handles initialization separately
        # This node primarily serves as a transition point
        updates = {
            "phase_history": [{
                "from_phase": -1,
                "to_phase": 0,
                "timestamp": datetime.now(),
                "reason": "Session started",
                "triggered_by": "system"
            }]
        }
        
        return updates
    
    def _agenda_setting_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Node for agenda setting phase.
        
        Args:
            state: Current state
            
        Returns:
            State updates
        """
        logger.debug("Executing agenda setting node")
        
        # Transition to phase 1
        updates = {
            "current_phase": 1,
            "phase_start_time": datetime.now(),
        }
        
        return updates
    
    def _discussion_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Node for discussion phase.
        
        Args:
            state: Current state
            
        Returns:
            State updates
        """
        logger.debug("Executing discussion node")
        
        # Handle phase transition if needed
        if state["current_phase"] != 2:
            return {
                "current_phase": 2,
                "phase_start_time": datetime.now(),
            }
        
        # Discussion continues with current state
        return {}
    
    def _consensus_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Node for consensus building phase.
        
        Args:
            state: Current state
            
        Returns:
            State updates
        """
        logger.debug("Executing consensus node")
        
        return {
            "current_phase": 3,
            "phase_start_time": datetime.now(),
        }
    
    def _summary_node(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Node for summary generation phase.
        
        Args:
            state: Current state
            
        Returns:
            State updates
        """
        logger.debug("Executing summary node")
        
        return {
            "current_phase": 4,
            "phase_start_time": datetime.now(),
        }
    
    def _should_continue_discussion(self, state: VirtualAgoraState) -> str:
        """Determine if discussion should continue or move to consensus.
        
        Args:
            state: Current state
            
        Returns:
            Next node: "continue" or "consensus"
        """
        # This is a simplified decision - in practice this would be
        # based on voting results or other criteria
        
        # For now, check if there's a flag indicating consensus is needed
        if state.get("consensus_reached", {}).get(state["active_topic"], False):
            return "consensus"
        
        # Check if discussion time limit reached (placeholder logic)
        # In real implementation, this would check actual discussion progress
        messages_on_topic = sum(
            1 for msg in state["messages"]
            if msg.get("topic") == state["active_topic"]
        )
        
        if messages_on_topic >= 10:  # Arbitrary threshold
            return "consensus"
        
        return "continue"
    
    def _should_continue_topics(self, state: VirtualAgoraState) -> str:
        """Determine if there are more topics or move to summary.
        
        Args:
            state: Current state
            
        Returns:
            Next node: "next_topic" or "summary"
        """
        # Check if there are more topics in the queue
        if state["topic_queue"]:
            return "next_topic"
        
        # Check if all proposed topics have been discussed
        remaining_topics = set(state["proposed_topics"]) - set(state["completed_topics"])
        if remaining_topics:
            return "next_topic"
        
        return "summary"
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session with initialized state.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Session ID
        """
        # Initialize state through state manager
        initial_state = self.state_manager.initialize_state(session_id)
        session_id = initial_state["session_id"]
        
        # Ensure graph is compiled
        if self.compiled_graph is None:
            self.compile()
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_state_manager(self) -> StateManager:
        """Get the state manager instance.
        
        Returns:
            State manager
        """
        return self.state_manager
    
    def invoke(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the graph with current state.
        
        Args:
            config: Optional LangGraph configuration
            
        Returns:
            Updated state
        """
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")
        
        if config is None:
            config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        
        # Get current state
        current_state = self.state_manager.get_snapshot()
        
        # Invoke graph
        result = self.compiled_graph.invoke(current_state, config)
        
        return result
    
    def stream(self, config: Optional[Dict[str, Any]] = None):
        """Stream graph execution.
        
        Args:
            config: Optional LangGraph configuration
            
        Yields:
            State updates
        """
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")
        
        if config is None:
            config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        
        # Get current state
        current_state = self.state_manager.get_snapshot()
        
        # Stream graph execution
        for update in self.compiled_graph.stream(current_state, config):
            yield update
    
    def update_state(self, updates: Dict[str, Any], as_node: Optional[str] = None) -> None:
        """Update the graph state.
        
        Args:
            updates: State updates to apply
            as_node: Optional node name to update as
        """
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")
        
        config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        
        # Update through graph
        self.compiled_graph.update_state(config, updates, as_node=as_node)
    
    def get_graph_state(self) -> Any:
        """Get the current graph state.
        
        Returns:
            Current graph state snapshot
        """
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")
        
        config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        return self.compiled_graph.get_state(config)
    
    def get_state_history(self) -> list:
        """Get the state history from the checkpointer.
        
        Returns:
            List of historical states
        """
        if self.compiled_graph is None:
            raise StateError("Graph not compiled")
        
        config = {"configurable": {"thread_id": self.state_manager.state["session_id"]}}
        return list(self.compiled_graph.get_state_history(config))