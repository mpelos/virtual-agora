"""LangGraph integration for Virtual Agora state management.

This module provides the integration between Virtual Agora's state management
and LangGraph's graph execution framework.
"""

from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from virtual_agora.state.schema import VirtualAgoraState, MessagesState
from virtual_agora.state.manager import StateManager
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import StateError
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.providers import create_provider


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
        
        # Initialize agents
        self.agents: Dict[str, LLMAgent] = {}
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize LLM agents from configuration."""
        # Create moderator
        moderator_llm = create_provider(
            provider=self.config.moderator.provider.value,
            model=self.config.moderator.model,
            temperature=getattr(self.config.moderator, 'temperature', 0.7),
            max_tokens=getattr(self.config.moderator, 'max_tokens', None)
        )
        self.agents["moderator"] = LLMAgent(
            agent_id="moderator",
            llm=moderator_llm,
            role="moderator"
        )
        logger.info("Initialized moderator agent")
        
        # Create participant agents
        agent_counter = 0
        for agent_config in self.config.agents:
            provider_llm = create_provider(
                provider=agent_config.provider.value,
                model=agent_config.model,
                temperature=getattr(agent_config, 'temperature', 0.7),
                max_tokens=getattr(agent_config, 'max_tokens', None)
            )
            
            for i in range(agent_config.count):
                agent_counter += 1
                agent_id = f"{agent_config.model}_{agent_counter}"
                self.agents[agent_id] = LLMAgent(
                    agent_id=agent_id,
                    llm=provider_llm,
                    role="participant"
                )
                logger.info(f"Initialized participant agent: {agent_id}")
        
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
        
        # Get current speaker
        speaker_id = state.get("current_speaker_id")
        if not speaker_id:
            # Start with first agent in speaking order
            if state["speaking_order"]:
                speaker_id = state["speaking_order"][0]
            else:
                return {}
        
        # Get the agent and let them speak
        if speaker_id in self.agents:
            agent = self.agents[speaker_id]
            # Use the agent's __call__ method for LangGraph integration
            updates = agent(state)
            
            # Update speaker rotation
            speaking_order = state["speaking_order"]
            current_index = speaking_order.index(speaker_id) if speaker_id in speaking_order else -1
            next_index = (current_index + 1) % len(speaking_order)
            updates["current_speaker_id"] = speaking_order[next_index]
            
            return updates
        
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
    
    def create_agent_node(self, agent_id: str) -> Callable:
        """Create a node function for a specific agent.
        
        This allows adding agents as individual nodes in the graph:
        ```python
        graph.add_node("moderator", self.create_agent_node("moderator"))
        ```
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Node function for the agent
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        def agent_node(state: VirtualAgoraState, config: RunnableConfig) -> Dict[str, Any]:
            """Agent-specific node function."""
            return agent(state, config)
        
        # Set function name for debugging
        agent_node.__name__ = f"{agent_id}_node"
        
        return agent_node
    
    def create_streaming_agent_node(self, agent_id: str) -> Callable:
        """Create a streaming node function for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Async streaming node function
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        async def streaming_agent_node(
            state: VirtualAgoraState,
            config: RunnableConfig,
            writer: Any
        ) -> Dict[str, Any]:
            """Agent-specific streaming node."""
            return await agent.__acall__(state, config, writer=writer)
        
        streaming_agent_node.__name__ = f"{agent_id}_streaming_node"
        
        return streaming_agent_node