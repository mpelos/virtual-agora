"""Base classes for Virtual Agora flow nodes.

This module provides the foundation for the pluggable node architecture
introduced in Step 2.1 of the refactoring plan. It establishes clear
interfaces for all node types and provides common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import logging
import time
from copy import deepcopy

from langgraph.types import interrupt
from langchain_core.messages import BaseMessage

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class NodeValidationError(Exception):
    """Raised when node precondition validation fails."""

    pass


class NodeExecutionError(Exception):
    """Raised when node execution fails."""

    pass


class FlowNode(ABC):
    """Abstract base class for all Virtual Agora flow nodes.

    This class defines the core interface that all nodes must implement,
    providing standardized execution, validation, and error handling patterns.

    Key responsibilities:
    - Execute node logic with state updates
    - Validate preconditions before execution
    - Handle errors consistently
    - Provide node identification and metadata
    """

    def __init__(self, node_dependencies: Optional["NodeDependencies"] = None):
        """Initialize the flow node with optional dependencies.

        Args:
            node_dependencies: Shared dependencies for all nodes
        """
        self.dependencies = node_dependencies
        self._validation_errors: List[str] = []
        self._execution_metadata: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute the node logic and return state updates.

        This is the main entry point for node execution. Implementations
        should perform their specific logic and return a dictionary of
        state updates to be applied to the current state.

        Args:
            state: Current Virtual Agora state

        Returns:
            Dictionary of state updates to apply

        Raises:
            NodeExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def validate_preconditions(self, state: VirtualAgoraState) -> bool:
        """Validate that the node can execute with the given state.

        This method should check all required preconditions for node
        execution, such as required state keys, agent availability,
        and data validation.

        Args:
            state: Current Virtual Agora state

        Returns:
            True if preconditions are met, False otherwise

        Note:
            Use self._validation_errors to store specific error messages
            for debugging and logging purposes.
        """
        pass

    def get_node_name(self) -> str:
        """Get human-readable node name.

        Returns:
            Node name for logging and debugging
        """
        return self.__class__.__name__

    def get_validation_errors(self) -> List[str]:
        """Get validation errors from last precondition check.

        Returns:
            List of validation error messages
        """
        return self._validation_errors.copy()

    def handle_error(
        self, error: Exception, state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Handle errors that occur during node execution.

        This method provides a standardized way to handle errors and
        return appropriate state updates for error conditions.

        Args:
            error: Exception that occurred
            state: Current state when error occurred

        Returns:
            State updates for error handling
        """
        error_message = str(error)
        logger.error(f"Node {self.get_node_name()} execution failed: {error_message}")

        # Record error in execution metadata
        self._execution_metadata["last_error"] = {
            "error_type": type(error).__name__,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "node_name": self.get_node_name(),
        }

        return {
            "last_error": error_message,
            "error_count": state.get("error_count", 0) + 1,
            "error_source": self.get_node_name(),
            "error_timestamp": datetime.now().isoformat(),
        }

    def safe_execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute node with validation and error handling.

        This method wraps the execute() method with standardized
        validation and error handling logic.

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates from successful execution or error handling
        """
        # Clear previous validation errors
        self._validation_errors.clear()

        # Validate preconditions
        if not self.validate_preconditions(state):
            error_msg = f"Precondition validation failed for {self.get_node_name()}"
            if self._validation_errors:
                error_msg += f": {', '.join(self._validation_errors)}"

            logger.warning(error_msg)
            return self.handle_error(NodeValidationError(error_msg), state)

        # Execute with error handling
        try:
            logger.info(f"Executing node: {self.get_node_name()}")
            start_time = datetime.now()

            result = self.execute(state)

            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Node {self.get_node_name()} completed in {execution_time:.2f}s"
            )

            # Record execution metadata
            self._execution_metadata["last_execution"] = {
                "timestamp": start_time.isoformat(),
                "execution_time": execution_time,
                "success": True,
            }

            return result

        except Exception as e:
            logger.exception(f"Node {self.get_node_name()} execution failed")
            return self.handle_error(e, state)

    def _validate_required_keys(
        self, state: VirtualAgoraState, required_keys: List[str]
    ) -> bool:
        """Helper method to validate required state keys.

        Args:
            state: State to validate
            required_keys: List of required key names

        Returns:
            True if all keys present, False otherwise
        """
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            self._validation_errors.extend(
                [f"Missing required key: {key}" for key in missing_keys]
            )
            return False
        return True

    def _validate_state_type(
        self, state: VirtualAgoraState, key: str, expected_type: Type
    ) -> bool:
        """Helper method to validate state value types.

        Args:
            state: State to validate
            key: State key to check
            expected_type: Expected type for the value

        Returns:
            True if type matches, False otherwise
        """
        if key in state and not isinstance(state[key], expected_type):
            self._validation_errors.append(
                f"Key '{key}' expected type {expected_type.__name__}, got {type(state[key]).__name__}"
            )
            return False
        return True


class HITLNode(FlowNode):
    """Base class for Human-in-the-Loop (HITL) nodes.

    HITL nodes pause execution to request user input through the LangGraph
    interrupt mechanism. This class provides a standardized pattern for
    handling user interactions.

    Key responsibilities:
    - Create interrupt payloads for user interaction
    - Process user input and return state updates
    - Handle user interaction errors and timeouts
    """

    @abstractmethod
    def create_interrupt_payload(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Create interrupt payload for user interaction.

        This method should build the data structure that will be sent
        to the user interface for interaction.

        Args:
            state: Current Virtual Agora state

        Returns:
            Dictionary containing interrupt data for the UI
        """
        pass

    @abstractmethod
    def process_user_input(
        self, user_input: Dict[str, Any], state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Process user input and return state updates.

        This method handles the user's response to the interrupt and
        converts it into appropriate state updates.

        Args:
            user_input: User's response from the interrupt
            state: Current Virtual Agora state

        Returns:
            State updates based on user input
        """
        pass

    def execute(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Execute HITL node with standardized interrupt handling.

        This implementation provides the standard HITL pattern:
        1. Create interrupt payload
        2. Send interrupt and wait for user input
        3. Process user input and return state updates

        Args:
            state: Current Virtual Agora state

        Returns:
            State updates from user interaction
        """
        # Create interrupt payload
        interrupt_payload = self.create_interrupt_payload(state)

        # Add standard HITL metadata
        interrupt_payload.update(
            {
                "node_name": self.get_node_name(),
                "timestamp": datetime.now().isoformat(),
                "state_summary": self._create_state_summary(state),
            }
        )

        logger.info(f"HITL node {self.get_node_name()} requesting user input")
        logger.debug(f"Interrupt payload: {interrupt_payload}")

        # Use LangGraph interrupt mechanism
        try:
            user_input = interrupt(interrupt_payload)
            logger.info(f"HITL node {self.get_node_name()} received user input")

            # Process user input
            return self.process_user_input(user_input, state)

        except Exception as e:
            logger.error(f"HITL interrupt failed for {self.get_node_name()}: {e}")
            return self.handle_error(e, state)

    def _create_state_summary(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Create a summary of current state for user context.

        Args:
            state: Current state

        Returns:
            Summary information for user interface
        """
        return {
            "current_round": state.get("current_round", 0),
            "active_topic": state.get("active_topic", "Unknown"),
            "current_phase": state.get("current_phase", 0),
            "session_id": state.get("session_id", "unknown"),
            "completed_topics": len(state.get("completed_topics", [])),
            "total_messages": len(state.get("messages", [])),
        }

    def validate_interrupt_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate that interrupt payload contains required fields.

        Args:
            payload: Interrupt payload to validate

        Returns:
            True if payload is valid, False otherwise
        """
        required_fields = ["type", "message"]
        missing_fields = [field for field in required_fields if field not in payload]

        if missing_fields:
            self._validation_errors.extend(
                [f"Missing interrupt field: {field}" for field in missing_fields]
            )
            return False

        return True


class AgentOrchestratorNode(FlowNode):
    """Base class for nodes that orchestrate agents.

    This class provides common functionality for nodes that need to
    coordinate with LLM agents, including retry logic, agent validation,
    and response processing.

    Key responsibilities:
    - Validate agent availability
    - Execute agents with retry logic
    - Handle agent errors and timeouts
    - Process agent responses
    """

    def __init__(
        self,
        specialized_agents: Dict[str, LLMAgent],
        discussing_agents: List[LLMAgent],
        node_dependencies: Optional["NodeDependencies"] = None,
    ):
        """Initialize agent orchestrator node.

        Args:
            specialized_agents: Dictionary of specialized agents by ID
            discussing_agents: List of discussion agents
            node_dependencies: Shared node dependencies
        """
        super().__init__(node_dependencies)
        self.specialized_agents = specialized_agents
        self.discussing_agents = discussing_agents

    def validate_agent_availability(self, agent_id: str) -> bool:
        """Validate that required agent is available.

        Args:
            agent_id: ID of the agent to validate

        Returns:
            True if agent is available, False otherwise
        """
        if agent_id not in self.specialized_agents:
            self._validation_errors.append(f"Required agent not available: {agent_id}")
            return False

        agent = self.specialized_agents[agent_id]
        if not hasattr(agent, "__call__"):
            self._validation_errors.append(f"Agent {agent_id} is not callable")
            return False

        return True

    def call_agent_with_retry(
        self,
        agent: LLMAgent,
        state: VirtualAgoraState,
        prompt: str,
        context_messages: Optional[List[BaseMessage]] = None,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Call agent with retry logic and error handling.

        Args:
            agent: Agent to call
            state: Current state
            prompt: Prompt to send to agent
            context_messages: Context messages for agent
            max_attempts: Maximum retry attempts
            base_delay: Base delay between attempts (exponential backoff)

        Returns:
            Agent response or None if all attempts fail
        """
        agent_id = getattr(agent, "agent_id", agent.__class__.__name__)

        for attempt in range(max_attempts):
            try:
                logger.debug(
                    f"Calling agent {agent_id}, attempt {attempt + 1}/{max_attempts}"
                )

                # Call agent with context
                response = agent(
                    state, prompt=prompt, context_messages=context_messages
                )

                if response is not None:
                    logger.debug(f"Agent {agent_id} responded successfully")
                    return response
                else:
                    logger.warning(f"Agent {agent_id} returned None response")

            except Exception as e:
                logger.warning(
                    f"Agent {agent_id} attempt {attempt + 1}/{max_attempts} failed: {e}"
                )

                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying agent {agent_id} in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_attempts} attempts failed for agent {agent_id}"
                    )
                    return None

        return None

    def validate_agent_response(self, response: Dict[str, Any], agent_id: str) -> bool:
        """Validate agent response format and content.

        Args:
            response: Agent response to validate
            agent_id: ID of the agent that provided the response

        Returns:
            True if response is valid, False otherwise
        """
        if not isinstance(response, dict):
            self._validation_errors.append(
                f"Agent {agent_id} response must be a dictionary"
            )
            return False

        # Basic response validation - can be extended by subclasses
        if "content" not in response and "message" not in response:
            self._validation_errors.append(
                f"Agent {agent_id} response missing content or message"
            )
            return False

        return True

    def get_available_agents(self) -> Dict[str, str]:
        """Get list of available agents and their types.

        Returns:
            Dictionary mapping agent IDs to their types
        """
        agents = {}

        # Add specialized agents
        for agent_id, agent in self.specialized_agents.items():
            agents[agent_id] = agent.__class__.__name__

        # Add discussing agents
        for i, agent in enumerate(self.discussing_agents):
            agent_id = getattr(agent, "agent_id", f"discussing_agent_{i}")
            agents[agent_id] = agent.__class__.__name__

        return agents


class NodeDependencies:
    """Dependency injection container for flow nodes.

    This class provides a centralized way to manage shared dependencies
    that nodes need, such as managers, coordinators, and configuration.
    """

    def __init__(
        self,
        state_manager: Optional[Any] = None,
        round_manager: Optional[Any] = None,
        message_coordinator: Optional[Any] = None,
        flow_state_manager: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        cycle_manager: Optional[Any] = None,
        checkpoint_interval: int = 3,
    ):
        """Initialize node dependencies.

        Args:
            state_manager: State manager instance
            round_manager: Round manager instance
            message_coordinator: Message coordinator instance
            flow_state_manager: Flow state manager instance
            context_manager: Context window manager instance
            cycle_manager: Cycle prevention manager instance
            checkpoint_interval: Rounds between checkpoints
        """
        self.state_manager = state_manager
        self.round_manager = round_manager
        self.message_coordinator = message_coordinator
        self.flow_state_manager = flow_state_manager
        self.context_manager = context_manager
        self.cycle_manager = cycle_manager
        self.checkpoint_interval = checkpoint_interval

        # Validation flags
        self._validated = False
        self._validation_errors: List[str] = []

    def validate(self) -> bool:
        """Validate that required dependencies are available.

        Returns:
            True if all required dependencies are present, False otherwise
        """
        self._validation_errors.clear()

        # Check required dependencies
        required_deps = [
            ("round_manager", self.round_manager),
            ("message_coordinator", self.message_coordinator),
            ("flow_state_manager", self.flow_state_manager),
        ]

        for dep_name, dep_instance in required_deps:
            if dep_instance is None:
                self._validation_errors.append(
                    f"Missing required dependency: {dep_name}"
                )

        self._validated = len(self._validation_errors) == 0
        return self._validated

    def get_validation_errors(self) -> List[str]:
        """Get dependency validation errors.

        Returns:
            List of validation error messages
        """
        return self._validation_errors.copy()

    def is_validated(self) -> bool:
        """Check if dependencies have been validated successfully.

        Returns:
            True if dependencies are validated, False otherwise
        """
        return self._validated


class NodeExecutionContext:
    """Context object for node execution tracking and debugging.

    This class provides execution context and metadata for nodes,
    useful for debugging, monitoring, and performance analysis.
    """

    def __init__(self, node: FlowNode, state: VirtualAgoraState):
        """Initialize execution context.

        Args:
            node: Node being executed
            state: Current state
        """
        self.node = node
        self.state = state
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.success = False
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[Exception] = None
        self.validation_errors: List[str] = []

    def mark_completed(self, result: Dict[str, Any]) -> None:
        """Mark execution as completed successfully.

        Args:
            result: Execution result
        """
        self.end_time = datetime.now()
        self.success = True
        self.result = result

    def mark_failed(self, error: Exception) -> None:
        """Mark execution as failed.

        Args:
            error: Exception that caused failure
        """
        self.end_time = datetime.now()
        self.success = False
        self.error = error

    def get_execution_time(self) -> float:
        """Get execution time in seconds.

        Returns:
            Execution time or 0 if not completed
        """
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/debugging.

        Returns:
            Dictionary representation of execution context
        """
        return {
            "node_name": self.node.get_node_name(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": self.get_execution_time(),
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "validation_errors": self.validation_errors,
            "state_keys": list(self.state.keys()) if self.state else [],
        }
