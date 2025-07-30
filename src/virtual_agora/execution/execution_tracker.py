"""Execution tracking and visibility for Virtual Agora.

This module provides the ExecutionTracker class which adds clear execution
tracking and debugging capabilities to replace the scattered logging and
unclear execution flow in the original architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionEventType(Enum):
    """Types of execution events we track."""

    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    INTERRUPT_START = "interrupt_start"
    INTERRUPT_COMPLETE = "interrupt_complete"
    ROUTING_DECISION = "routing_decision"
    STATE_UPDATE = "state_update"
    PHASE_TRANSITION = "phase_transition"
    TOPIC_TRANSITION = "topic_transition"
    ERROR = "error"
    USER_INTERACTION = "user_interaction"
    STATISTICS_UPDATE = "statistics_update"


@dataclass
class ExecutionEvent:
    """Represents a single execution event."""

    event_id: str
    event_type: ExecutionEventType
    timestamp: datetime
    node_name: Optional[str] = None

    # State information
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None

    # Event-specific data
    data: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    duration_ms: Optional[float] = None

    # Context information
    session_id: Optional[str] = None
    phase: Optional[int] = None
    topic: Optional[str] = None

    # Error information (if applicable)
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "node_name": self.node_name,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "session_id": self.session_id,
            "phase": self.phase,
            "topic": self.topic,
            "error": self.error,
            "error_type": self.error_type,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for execution tracking."""

    total_events: int = 0
    total_duration_ms: float = 0.0
    node_execution_times: Dict[str, List[float]] = field(default_factory=dict)
    interrupt_handling_times: List[float] = field(default_factory=list)
    routing_decision_times: List[float] = field(default_factory=list)

    # Error tracking
    error_count: int = 0
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Topic metrics
    topics_completed: int = 0
    topic_completion_times: List[float] = field(default_factory=list)

    @property
    def average_node_execution_time(self) -> float:
        """Calculate average node execution time."""
        all_times = []
        for times in self.node_execution_times.values():
            all_times.extend(times)
        return sum(all_times) / len(all_times) if all_times else 0.0

    @property
    def average_interrupt_handling_time(self) -> float:
        """Calculate average interrupt handling time."""
        return (
            sum(self.interrupt_handling_times) / len(self.interrupt_handling_times)
            if self.interrupt_handling_times
            else 0.0
        )

    @property
    def average_topic_completion_time(self) -> float:
        """Calculate average topic completion time."""
        return (
            sum(self.topic_completion_times) / len(self.topic_completion_times)
            if self.topic_completion_times
            else 0.0
        )


class ExecutionTracker:
    """Tracks execution flow and provides visibility into system behavior.

    This class replaces the scattered logging and unclear execution tracking
    in the original architecture with a comprehensive, structured approach
    to understanding what's happening during session execution.
    """

    def __init__(self, session_id: str, max_events: int = 1000):
        """Initialize the execution tracker.

        Args:
            session_id: Session identifier
            max_events: Maximum number of events to keep in memory
        """
        self.session_id = session_id
        self.max_events = max_events

        # Event storage
        self.events: deque = deque(maxlen=max_events)
        self.event_index: Dict[str, ExecutionEvent] = {}

        # Performance tracking
        self.metrics = PerformanceMetrics()

        # Active tracking
        self.active_nodes: Dict[str, datetime] = {}
        self.active_interrupts: Dict[str, datetime] = {}

        # Timeline tracking
        self.session_start = datetime.now()
        self.last_event_time = self.session_start

        logger.info(f"ExecutionTracker initialized for session: {session_id}")

    def track_node_execution(
        self, node_name: str, state_before: Dict[str, Any], state_after: Dict[str, Any]
    ) -> str:
        """Track the execution of a node with before/after state.

        Args:
            node_name: Name of the executed node
            state_before: State before node execution
            state_after: State after node execution

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Calculate duration if we were tracking this node
        duration_ms = None
        if node_name in self.active_nodes:
            duration = timestamp - self.active_nodes[node_name]
            duration_ms = duration.total_seconds() * 1000
            del self.active_nodes[node_name]

            # Update metrics
            if node_name not in self.metrics.node_execution_times:
                self.metrics.node_execution_times[node_name] = []
            self.metrics.node_execution_times[node_name].append(duration_ms)

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.NODE_COMPLETE,
            timestamp=timestamp,
            node_name=node_name,
            state_before=self._sanitize_state(state_before),
            state_after=self._sanitize_state(state_after),
            duration_ms=duration_ms,
            session_id=self.session_id,
            phase=state_after.get("current_phase"),
            topic=state_after.get("active_topic"),
        )

        self._record_event(event)

        duration_str = (
            f"{duration_ms:.1f}ms" if duration_ms is not None else "unmeasured"
        )
        logger.debug(f"Node execution tracked: {node_name} ({duration_str})")
        return event_id

    def track_node_start(self, node_name: str, state: Dict[str, Any]) -> str:
        """Track the start of node execution.

        Args:
            node_name: Name of the node starting
            state: Current state

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Mark node as active
        self.active_nodes[node_name] = timestamp

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.NODE_START,
            timestamp=timestamp,
            node_name=node_name,
            state_before=self._sanitize_state(state),
            session_id=self.session_id,
            phase=state.get("current_phase"),
            topic=state.get("active_topic"),
        )

        self._record_event(event)

        logger.debug(f"Node start tracked: {node_name}")
        return event_id

    def track_routing_decision(
        self, decision_point: str, decision: str, reasoning: Dict[str, Any]
    ) -> str:
        """Track routing decisions with full context.

        Args:
            decision_point: Where the routing decision was made
            decision: The decision that was made
            reasoning: Context and reasoning for the decision

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Calculate decision time (rough estimate based on last event)
        duration_ms = None
        if self.last_event_time:
            duration = timestamp - self.last_event_time
            duration_ms = duration.total_seconds() * 1000
            self.metrics.routing_decision_times.append(duration_ms)

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.ROUTING_DECISION,
            timestamp=timestamp,
            data={
                "decision_point": decision_point,
                "decision": decision,
                "reasoning": reasoning,
            },
            duration_ms=duration_ms,
            session_id=self.session_id,
            phase=reasoning.get("current_phase"),
            topic=reasoning.get("active_topic"),
        )

        self._record_event(event)

        logger.info(f"Routing decision tracked: {decision_point} -> {decision}")
        return event_id

    def track_interrupt_handling(
        self,
        interrupt_type: str,
        start_time: datetime,
        end_time: datetime,
        user_response: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Track interrupt handling with timing.

        Args:
            interrupt_type: Type of interrupt handled
            start_time: When interrupt handling started
            end_time: When interrupt handling completed
            user_response: User's response to the interrupt

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        duration = end_time - start_time
        duration_ms = duration.total_seconds() * 1000

        # Update metrics
        self.metrics.interrupt_handling_times.append(duration_ms)

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.INTERRUPT_COMPLETE,
            timestamp=end_time,
            data={
                "interrupt_type": interrupt_type,
                "start_time": start_time.isoformat(),
                "user_response": user_response,
            },
            duration_ms=duration_ms,
            session_id=self.session_id,
        )

        self._record_event(event)

        logger.info(
            f"Interrupt handling tracked: {interrupt_type} ({duration_ms:.1f}ms)"
        )
        return event_id

    def track_phase_transition(
        self, old_phase: int, new_phase: int, context: Dict[str, Any]
    ) -> str:
        """Track phase transitions.

        Args:
            old_phase: Previous phase number
            new_phase: New phase number
            context: Context information about the transition

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.PHASE_TRANSITION,
            timestamp=timestamp,
            data={"old_phase": old_phase, "new_phase": new_phase, "context": context},
            session_id=self.session_id,
            phase=new_phase,
        )

        self._record_event(event)

        logger.info(f"Phase transition tracked: {old_phase} -> {new_phase}")
        return event_id

    def track_topic_transition(
        self,
        old_topic: Optional[str],
        new_topic: Optional[str],
        topics_remaining: int,
        topics_completed: int,
    ) -> str:
        """Track topic transitions.

        Args:
            old_topic: Previous topic
            new_topic: New topic
            topics_remaining: Number of topics remaining
            topics_completed: Number of topics completed

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # If a topic was completed, update metrics
        if old_topic and new_topic != old_topic:
            self.metrics.topics_completed += 1

            # Calculate topic completion time (rough estimate)
            topic_events = [e for e in self.events if e.topic == old_topic]
            if topic_events:
                topic_start = min(e.timestamp for e in topic_events)
                completion_time = (timestamp - topic_start).total_seconds() * 1000
                self.metrics.topic_completion_times.append(completion_time)

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.TOPIC_TRANSITION,
            timestamp=timestamp,
            data={
                "old_topic": old_topic,
                "new_topic": new_topic,
                "topics_remaining": topics_remaining,
                "topics_completed": topics_completed,
            },
            session_id=self.session_id,
            topic=new_topic,
        )

        self._record_event(event)

        logger.info(
            f"Topic transition tracked: {old_topic} -> {new_topic} (remaining: {topics_remaining})"
        )
        return event_id

    def track_error(self, error: Exception, context: Dict[str, Any]) -> str:
        """Track errors with context.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            Event ID for the recorded event
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Update metrics
        self.metrics.error_count += 1
        error_type = type(error).__name__
        self.metrics.error_types[error_type] += 1

        # Create event
        event = ExecutionEvent(
            event_id=event_id,
            event_type=ExecutionEventType.ERROR,
            timestamp=timestamp,
            data=context,
            error=str(error),
            error_type=error_type,
            session_id=self.session_id,
            phase=context.get("current_phase"),
            topic=context.get("active_topic"),
        )

        self._record_event(event)

        logger.error(f"Error tracked: {error_type} - {error}")
        return event_id

    def get_execution_timeline(self) -> List[ExecutionEvent]:
        """Get complete timeline of execution events.

        Returns:
            List of execution events in chronological order
        """
        return list(self.events)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Dictionary containing performance metrics and analysis
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "total_events": len(self.events),
            # Node performance
            "node_performance": {
                "average_execution_time_ms": self.metrics.average_node_execution_time,
                "slowest_nodes": self._get_slowest_nodes(),
                "node_execution_counts": {
                    node: len(times)
                    for node, times in self.metrics.node_execution_times.items()
                },
            },
            # Interrupt performance
            "interrupt_performance": {
                "total_interrupts": len(self.metrics.interrupt_handling_times),
                "average_handling_time_ms": self.metrics.average_interrupt_handling_time,
                "total_handling_time_ms": sum(self.metrics.interrupt_handling_times),
            },
            # Topic performance
            "topic_performance": {
                "topics_completed": self.metrics.topics_completed,
                "average_completion_time_ms": self.metrics.average_topic_completion_time,
                "topics_per_minute": (
                    self.metrics.topics_completed / max(1, session_duration)
                )
                * 60,
            },
            # Error analysis
            "error_analysis": {
                "total_errors": self.metrics.error_count,
                "error_rate": self.metrics.error_count / max(1, len(self.events)),
                "error_types": dict(self.metrics.error_types),
            },
            # Routing performance
            "routing_performance": {
                "total_decisions": len(self.metrics.routing_decision_times),
                "average_decision_time_ms": sum(self.metrics.routing_decision_times)
                / max(1, len(self.metrics.routing_decision_times)),
            },
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a high-level session summary for debugging.

        Returns:
            Dictionary with session summary information
        """
        recent_events = list(self.events)[-10:] if self.events else []

        return {
            "session_id": self.session_id,
            "session_duration": str(datetime.now() - self.session_start),
            "total_events": len(self.events),
            "topics_completed": self.metrics.topics_completed,
            "errors_encountered": self.metrics.error_count,
            "active_nodes": list(self.active_nodes.keys()),
            "active_interrupts": list(self.active_interrupts.keys()),
            "recent_events": [
                {
                    "type": e.event_type.value,
                    "node": e.node_name,
                    "timestamp": e.timestamp.strftime("%H:%M:%S.%f")[:-3],
                }
                for e in recent_events
            ],
        }

    def _record_event(self, event: ExecutionEvent) -> None:
        """Record an event in the tracker."""
        self.events.append(event)
        self.event_index[event.event_id] = event
        self.last_event_time = event.timestamp
        self.metrics.total_events += 1

        if event.duration_ms:
            self.metrics.total_duration_ms += event.duration_ms

    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state for storage (remove large objects, etc.)."""
        if not isinstance(state, dict):
            return {}

        # Keep only important fields for tracking
        important_fields = [
            "session_id",
            "current_phase",
            "current_round",
            "active_topic",
            "topic_queue",
            "completed_topics",
            "user_approves_continuation",
            "agents_vote_end_session",
            "agenda_approved",
        ]

        sanitized = {}
        for field in important_fields:
            if field in state:
                value = state[field]
                # Truncate large lists for storage efficiency
                if isinstance(value, list) and len(value) > 10:
                    sanitized[field] = f"<list with {len(value)} items>"
                else:
                    sanitized[field] = value

        return sanitized

    def _get_slowest_nodes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the slowest nodes by average execution time."""
        node_averages = []

        for node_name, times in self.metrics.node_execution_times.items():
            if times:
                avg_time = sum(times) / len(times)
                node_averages.append(
                    {
                        "node_name": node_name,
                        "average_time_ms": avg_time,
                        "execution_count": len(times),
                        "total_time_ms": sum(times),
                    }
                )

        # Sort by average time, descending
        node_averages.sort(key=lambda x: x["average_time_ms"], reverse=True)

        return node_averages[:top_n]
