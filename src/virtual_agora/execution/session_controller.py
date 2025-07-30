"""Session execution controller for Virtual Agora.

This module provides the SessionController class which serves as the single
source of truth for session state and execution control, replacing the
complex nested stream logic in main.py.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator, Union
from datetime import datetime, timedelta
import uuid

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class SessionState(Enum):
    """Session execution states."""

    INITIALIZING = "initializing"
    AGENDA_SETTING = "agenda_setting"
    DISCUSSING = "discussing"
    WAITING_FOR_USER = "waiting_for_user"
    TRANSITIONING = "transitioning"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ERROR = "error"


class EventType(Enum):
    """Types of execution events."""

    NODE_UPDATE = "node_update"
    INTERRUPT_REQUIRED = "interrupt_required"
    SESSION_COMPLETE = "session_complete"
    TOPIC_COMPLETE = "topic_complete"
    ERROR = "error"
    STATE_CHANGE = "state_change"


@dataclass
class ExecutionEvent:
    """Represents an execution event in the session."""

    type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SessionStatistics:
    """Accurate session statistics."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    topics_completed: int = 0
    total_messages: int = 0
    interrupts_handled: int = 0
    checkpoints_created: int = 0
    error_count: int = 0

    @property
    def duration(self) -> timedelta:
        """Get session duration."""
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string."""
        duration = self.duration
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


@dataclass
class ExecutionContext:
    """Tracks the current execution context."""

    current_node: Optional[str] = None
    current_phase: int = 0
    active_topic: Optional[str] = None
    topics_remaining: List[str] = field(default_factory=list)
    topics_completed: List[str] = field(default_factory=list)
    last_update: Optional[Dict[str, Any]] = None
    interrupt_pending: bool = False
    should_terminate: bool = False
    termination_reason: Optional[str] = None


class SessionController:
    """Single source of truth for session state and execution control.

    This class replaces the complex nested stream logic in main.py with
    a clean, event-driven architecture that properly tracks session state
    and provides a single decision point for session continuation.
    """

    def __init__(self, flow_instance, session_id: str):
        """Initialize the session controller.

        Args:
            flow_instance: The VirtualAgoraV13Flow instance
            session_id: Unique session identifier
        """
        self.flow = flow_instance
        self.session_id = session_id
        self.state = SessionState.INITIALIZING
        self.context = ExecutionContext()
        self.statistics = SessionStatistics(
            session_id=session_id, start_time=datetime.now()
        )
        self.event_history: List[ExecutionEvent] = []
        self._last_flow_update: Optional[Dict[str, Any]] = None

        logger.info(f"SessionController initialized for session: {session_id}")

    def should_continue_session(self) -> bool:
        """Single decision point for session continuation.

        This method replaces the complex continuation logic scattered
        throughout main.py with a single, clear decision point.

        Returns:
            True if session should continue, False if it should terminate
        """
        # Check for explicit termination conditions
        if self.context.should_terminate:
            logger.info(
                f"Session termination requested: {self.context.termination_reason}"
            )
            return False

        # Check if we're in a terminal state
        if self.state in [SessionState.COMPLETED, SessionState.ERROR]:
            logger.info(f"Session in terminal state: {self.state}")
            return False

        # Check if there are topics remaining and we're in a discussion phase
        if self.state == SessionState.DISCUSSING:
            if not self.context.topics_remaining:
                logger.info("No topics remaining, session should complete")
                self._request_termination("No topics remaining")
                return False
            else:
                logger.info(
                    f"Topics remaining: {len(self.context.topics_remaining)}, continuing session"
                )
                return True

        # Check if we're waiting for user input
        if self.state == SessionState.WAITING_FOR_USER:
            logger.debug("Session waiting for user input, continuing")
            return True

        # Default: continue unless explicitly told to stop
        logger.debug(f"Session continuing in state: {self.state}")
        return True

    def update_execution_context(self, flow_update: Dict[str, Any]) -> None:
        """Track where we are in the execution.

        This method processes LangGraph flow updates and maintains accurate
        execution context, replacing the scattered state tracking in main.py.

        Args:
            flow_update: Update from LangGraph flow execution
        """
        self._last_flow_update = flow_update

        # Log the update for debugging
        if isinstance(flow_update, dict):
            node_names = [
                k for k in flow_update.keys() if k not in ["__interrupt__", "__end__"]
            ]
            if node_names:
                logger.debug(f"Processing flow update from nodes: {node_names}")

        # Handle special LangGraph updates
        if "__interrupt__" in flow_update:
            self._handle_interrupt_update(flow_update["__interrupt__"])
            return

        if "__end__" in flow_update:
            self._handle_completion_update(flow_update["__end__"])
            return

        # Process regular node updates
        for node_name, node_data in flow_update.items():
            if node_name.startswith("__"):
                continue

            self.context.current_node = node_name
            logger.debug(f"Executing node: {node_name}")

            # Extract state information from node data
            if isinstance(node_data, dict):
                self._extract_state_updates(node_data)

        # Update statistics
        if flow_update:
            self.statistics.total_messages += 1

        # Emit state change event
        self._emit_event(
            EventType.STATE_CHANGE,
            {
                "old_state": self.state.value,
                "new_context": {
                    "current_node": self.context.current_node,
                    "current_phase": self.context.current_phase,
                    "active_topic": self.context.active_topic,
                    "topics_remaining": len(self.context.topics_remaining),
                    "topics_completed": len(self.context.topics_completed),
                },
            },
        )

    def get_session_statistics(self) -> SessionStatistics:
        """Get accurate session statistics from single source.

        Returns:
            Current session statistics
        """
        # Update statistics with current context
        self.statistics.topics_completed = len(self.context.topics_completed)

        # Get message count from flow state if available
        if hasattr(self.flow, "get_state_manager"):
            flow_state = self.flow.get_state_manager().state
            messages = flow_state.get("messages", [])
            self.statistics.total_messages = len(messages)

        return self.statistics

    def execute_session(self) -> Iterator[ExecutionEvent]:
        """Execute the session with clean event-driven architecture.

        This method replaces the complex nested stream logic in main.py
        with a clean, understandable execution flow.

        Yields:
            ExecutionEvent: Events during session execution
        """
        logger.info(f"Starting session execution: {self.session_id}")
        self.state = SessionState.INITIALIZING

        try:
            # Start the flow stream
            config_dict = {"configurable": {"thread_id": self.session_id}}

            for flow_update in self.flow.stream(config_dict):
                # Update our execution context
                self.update_execution_context(flow_update)

                # Handle different types of updates
                if "__interrupt__" in flow_update:
                    # Emit interrupt event and wait for handling
                    yield self._emit_event(
                        EventType.INTERRUPT_REQUIRED,
                        {
                            "interrupt_data": flow_update["__interrupt__"],
                            "context": self.context,
                        },
                    )

                    # After interrupt handling, check if we should continue
                    if not self.should_continue_session():
                        break

                elif "__end__" in flow_update:
                    # Session completed naturally
                    logger.info("Session completed naturally via __end__")
                    yield self._emit_event(
                        EventType.SESSION_COMPLETE,
                        {"reason": "natural_completion", "final_state": self.context},
                    )
                    break

                else:
                    # Regular node update
                    yield self._emit_event(
                        EventType.NODE_UPDATE,
                        {"update": flow_update, "context": self.context},
                    )

                # Check if we should continue after each update
                if not self.should_continue_session():
                    logger.info("Session controller determined session should end")
                    yield self._emit_event(
                        EventType.SESSION_COMPLETE,
                        {
                            "reason": self.context.termination_reason
                            or "controller_decision",
                            "final_state": self.context,
                        },
                    )
                    break

        except Exception as e:
            logger.error(f"Error during session execution: {e}", exc_info=True)
            self.state = SessionState.ERROR
            self.statistics.error_count += 1
            yield self._emit_event(
                EventType.ERROR,
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "context": self.context,
                },
            )

        finally:
            self.state = SessionState.COMPLETED
            self.statistics.end_time = datetime.now()
            logger.info(f"Session execution completed: {self.session_id}")

    def handle_interrupt_response(self, user_response: Dict[str, Any]) -> None:
        """Handle user response to an interrupt.

        This method processes user responses and updates the flow state,
        replacing the complex interrupt handling in main.py.

        Args:
            user_response: User's response to the interrupt
        """
        logger.info(f"Handling interrupt response: {list(user_response.keys())}")

        # Update flow state with user response
        try:
            if self._last_flow_update and "__interrupt__" in self._last_flow_update:
                interrupt_data = self._last_flow_update["__interrupt__"]

                # Extract node information for state update
                if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
                    interrupt_obj = interrupt_data[0]
                    if hasattr(interrupt_obj, "ns") and interrupt_obj.ns:
                        node_name = (
                            interrupt_obj.ns[0].split(":")[0]
                            if ":" in interrupt_obj.ns[0]
                            else interrupt_obj.ns[0]
                        )

                        # Update the flow state
                        config_dict = {"configurable": {"thread_id": self.session_id}}
                        self.flow.compiled_graph.update_state(
                            config_dict, user_response, as_node=node_name
                        )
                        logger.info(f"Updated flow state for node: {node_name}")
                    else:
                        # Fallback: update without specific node
                        config_dict = {"configurable": {"thread_id": self.session_id}}
                        self.flow.compiled_graph.update_state(
                            config_dict, user_response
                        )
                        logger.info("Updated flow state without specific node")

            self.statistics.interrupts_handled += 1
            self.context.interrupt_pending = False

        except Exception as e:
            logger.error(f"Error handling interrupt response: {e}", exc_info=True)
            self.statistics.error_count += 1
            raise

    def _handle_interrupt_update(self, interrupt_data: Any) -> None:
        """Handle interrupt update from flow."""
        self.state = SessionState.WAITING_FOR_USER
        self.context.interrupt_pending = True
        logger.info("Session waiting for user input due to interrupt")

    def _handle_completion_update(self, end_data: Any) -> None:
        """Handle completion update from flow."""
        logger.info("Flow signaled completion")
        self._request_termination("Flow completed naturally")

    def _extract_state_updates(self, node_data: Dict[str, Any]) -> None:
        """Extract state information from node data."""
        # Update current phase
        if "current_phase" in node_data:
            old_phase = self.context.current_phase
            self.context.current_phase = node_data["current_phase"]
            if old_phase != self.context.current_phase:
                logger.debug(
                    f"Phase transition: {old_phase} -> {self.context.current_phase}"
                )

        # Update active topic
        if "active_topic" in node_data:
            old_topic = self.context.active_topic
            self.context.active_topic = node_data["active_topic"]
            if old_topic != self.context.active_topic:
                logger.info(f"Topic change: {old_topic} -> {self.context.active_topic}")

        # Update topic queue
        if "topic_queue" in node_data:
            self.context.topics_remaining = node_data["topic_queue"] or []
            logger.debug(f"Topics remaining: {len(self.context.topics_remaining)}")

        # Update completed topics
        if "completed_topics" in node_data:
            completed = node_data["completed_topics"] or []
            # Handle potential corruption - ensure it's a list
            if isinstance(completed, str):
                completed = [completed] if completed.strip() else []
            elif isinstance(completed, list):
                # Flatten any nested lists
                flattened = []
                for item in completed:
                    if isinstance(item, str) and item.strip():
                        flattened.append(item.strip())
                    elif isinstance(item, list):
                        for nested in item:
                            if isinstance(nested, str) and nested.strip():
                                flattened.append(nested.strip())
                completed = flattened

            old_count = len(self.context.topics_completed)
            self.context.topics_completed = completed
            new_count = len(self.context.topics_completed)

            if new_count > old_count:
                logger.info(f"Topic completed! Total completed: {new_count}")

        # Determine session state based on context
        self._update_session_state()

    def _update_session_state(self) -> None:
        """Update session state based on current context."""
        if self.context.interrupt_pending:
            self.state = SessionState.WAITING_FOR_USER
        elif self.context.current_phase <= 1:
            self.state = SessionState.AGENDA_SETTING
        elif self.context.current_phase in [2, 3]:
            self.state = SessionState.DISCUSSING
        elif self.context.current_phase >= 5:
            self.state = SessionState.COMPLETING
        else:
            self.state = SessionState.TRANSITIONING

    def _request_termination(self, reason: str) -> None:
        """Request session termination with reason."""
        self.context.should_terminate = True
        self.context.termination_reason = reason
        logger.info(f"Session termination requested: {reason}")

    def _emit_event(
        self, event_type: EventType, data: Dict[str, Any]
    ) -> ExecutionEvent:
        """Emit an execution event."""
        event = ExecutionEvent(
            type=event_type,
            timestamp=datetime.now(),
            data=data,
            context={
                "session_id": self.session_id,
                "session_state": self.state.value,
                "current_node": self.context.current_node,
                "current_phase": self.context.current_phase,
            },
        )

        self.event_history.append(event)
        return event

    def get_execution_timeline(self) -> List[ExecutionEvent]:
        """Get complete timeline of session execution events.

        Returns:
            List of execution events in chronological order
        """
        return self.event_history.copy()

    def get_current_context(self) -> ExecutionContext:
        """Get current execution context.

        Returns:
            Current execution context
        """
        return self.context
