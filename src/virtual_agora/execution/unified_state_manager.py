"""Unified state management for Virtual Agora.

This module provides the UnifiedStateManager class which centralizes all state
management, replacing the scattered state tracking across multiple managers
in the original architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from copy import deepcopy

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class StateLayer(Enum):
    """Different layers of state management."""

    SESSION = "session"  # Session-level state (high-level session info)
    FLOW = "flow"  # LangGraph flow state (execution state)
    UI = "ui"  # UI/display state (presentation state)
    STATISTICS = "statistics"  # Statistics and metrics (derived state)


@dataclass
class SessionState:
    """Session-level state information."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    main_topic: Optional[str] = None
    current_phase: int = 0
    total_phases: int = 6
    is_active: bool = True
    termination_reason: Optional[str] = None


@dataclass
class FlowState:
    """LangGraph flow execution state."""

    current_node: Optional[str] = None
    current_phase: int = 0
    current_round: int = 0
    active_topic: Optional[str] = None
    topic_queue: List[str] = field(default_factory=list)
    completed_topics: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # User interaction state
    user_approves_continuation: bool = True
    user_requests_end: bool = False
    user_requested_modification: bool = False
    user_forced_conclusion: bool = False

    # Agent state
    agents_vote_end_session: bool = False
    agent_recommendation: Optional[str] = None

    # Agenda state
    agenda_approved: bool = False
    proposed_agenda: List[str] = field(default_factory=list)
    final_agenda: List[str] = field(default_factory=list)

    # HITL state
    awaiting_user_input: bool = False
    last_interrupt_type: Optional[str] = None


@dataclass
class UIState:
    """UI and display state."""

    display_mode: str = "standard"
    show_debug_info: bool = False
    theme: str = "default"
    last_update: Optional[datetime] = None
    notifications: List[str] = field(default_factory=list)


@dataclass
class Statistics:
    """Statistics and metrics."""

    topics_discussed: int = 0
    total_messages: int = 0
    interrupts_handled: int = 0
    checkpoints_created: int = 0
    errors_encountered: int = 0
    user_interactions: int = 0

    # Performance metrics
    average_response_time: float = 0.0
    session_duration: float = 0.0

    # Calculated properties
    @property
    def completion_rate(self) -> float:
        """Calculate topic completion rate."""
        if not hasattr(self, "_total_planned_topics"):
            return 0.0
        return self.topics_discussed / max(1, self._total_planned_topics)

    def set_total_planned_topics(self, count: int):
        """Set total planned topics for completion rate calculation."""
        self._total_planned_topics = count


class UnifiedStateManager:
    """Centralized state management for all Virtual Agora state.

    This class replaces the scattered state management across StateManager,
    InterruptManager, UI state, and Recovery state with a single, unified
    approach that maintains clear separation of concerns while providing
    a single source of truth.
    """

    def __init__(self, session_id: str):
        """Initialize the unified state manager.

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id

        # Initialize state layers
        self.session_state = SessionState(
            session_id=session_id, start_time=datetime.now()
        )
        self.flow_state = FlowState()
        self.ui_state = UIState()
        self.statistics = Statistics()

        # State synchronization tracking
        self._state_version = 0
        self._dirty_layers: Set[StateLayer] = set()
        self._last_sync = datetime.now()
        self._sync_history: List[Dict[str, Any]] = []

        # State validation
        self._validation_errors: List[str] = []

        logger.info(f"UnifiedStateManager initialized for session: {session_id}")

    def update_session_state(self, updates: Dict[str, Any]) -> None:
        """Update session-level state.

        Args:
            updates: Dictionary of session state updates
        """
        logger.debug(f"Updating session state: {list(updates.keys())}")

        old_values = {}
        for key, value in updates.items():
            if hasattr(self.session_state, key):
                old_values[key] = getattr(self.session_state, key)
                setattr(self.session_state, key, value)
                logger.debug(f"Session state updated: {key} = {value}")

        self._mark_dirty(StateLayer.SESSION)
        self._record_state_change(StateLayer.SESSION, updates, old_values)

    def update_flow_state(self, updates: Dict[str, Any]) -> None:
        """Update flow execution state.

        Args:
            updates: Dictionary of flow state updates
        """
        logger.debug(f"Updating flow state: {list(updates.keys())}")

        old_values = {}
        updated_fields = []

        for key, value in updates.items():
            if hasattr(self.flow_state, key):
                old_value = getattr(self.flow_state, key)
                old_values[key] = old_value

                # Handle special field types
                if key in ["topic_queue", "messages"]:
                    # These fields expect lists directly
                    if isinstance(value, list):
                        setattr(self.flow_state, key, value.copy())
                    elif value is None:
                        setattr(self.flow_state, key, [])
                    else:
                        logger.warning(
                            f"Expected list for {key}, got {type(value)}: {value}"
                        )
                        continue
                elif key == "completed_topics":
                    # completed_topics uses safe_list_append reducer
                    # String values are individual items to append, lists are final state
                    if isinstance(value, list):
                        setattr(self.flow_state, key, value.copy())
                    elif value is None:
                        setattr(self.flow_state, key, [])
                    elif isinstance(value, str):
                        # This is a single topic to append - handle via reducer pattern
                        current_list = getattr(self.flow_state, key, [])
                        if value not in current_list:
                            new_list = current_list + [value]
                            setattr(self.flow_state, key, new_list)
                            logger.debug(
                                f"Appended '{value}' to completed_topics: {new_list}"
                            )
                        else:
                            logger.debug(
                                f"Topic '{value}' already in completed_topics, skipping"
                            )
                    else:
                        logger.warning(
                            f"Expected list or string for completed_topics, got {type(value)}: {value}"
                        )
                        continue
                else:
                    setattr(self.flow_state, key, value)

                updated_fields.append(key)

                # Log significant changes
                if key == "completed_topics":
                    current_list = getattr(self.flow_state, key, [])
                    old_count = len(old_value) if isinstance(old_value, list) else 0
                    new_count = len(current_list)
                    if new_count > old_count:
                        logger.info(f"Topic completed! Total: {new_count}")
                        self.statistics.topics_discussed = new_count

                elif key == "active_topic" and old_value != value:
                    logger.info(f"Active topic changed: {old_value} -> {value}")

                elif key == "current_phase" and old_value != value:
                    logger.info(f"Phase transition: {old_value} -> {value}")
                    self.session_state.current_phase = value

        if updated_fields:
            self._mark_dirty(StateLayer.FLOW)
            self._record_state_change(
                StateLayer.FLOW,
                {k: v for k, v in updates.items() if k in updated_fields},
                {k: v for k, v in old_values.items() if k in updated_fields},
            )

            # Update statistics
            self.statistics.total_messages = len(self.flow_state.messages)

    def update_ui_state(self, updates: Dict[str, Any]) -> None:
        """Update UI state.

        Args:
            updates: Dictionary of UI state updates
        """
        logger.debug(f"Updating UI state: {list(updates.keys())}")

        old_values = {}
        for key, value in updates.items():
            if hasattr(self.ui_state, key):
                old_values[key] = getattr(self.ui_state, key)
                setattr(self.ui_state, key, value)

        self.ui_state.last_update = datetime.now()
        self._mark_dirty(StateLayer.UI)
        self._record_state_change(StateLayer.UI, updates, old_values)

    def update_statistics(self, updates: Dict[str, Any]) -> None:
        """Update statistics.

        Args:
            updates: Dictionary of statistics updates
        """
        logger.debug(f"Updating statistics: {list(updates.keys())}")

        old_values = {}
        for key, value in updates.items():
            if hasattr(self.statistics, key):
                old_values[key] = getattr(self.statistics, key)
                setattr(self.statistics, key, value)

        self._mark_dirty(StateLayer.STATISTICS)
        self._record_state_change(StateLayer.STATISTICS, updates, old_values)

    def sync_all_states(self) -> None:
        """Synchronize all state layers and ensure consistency.

        This method ensures that all state layers are consistent with each other
        and updates derived state based on primary state.
        """
        logger.debug("Synchronizing all state layers")

        # Sync session and flow state
        if self.flow_state.current_phase != self.session_state.current_phase:
            self.session_state.current_phase = self.flow_state.current_phase

        # Update statistics from flow state
        self.statistics.topics_discussed = len(self.flow_state.completed_topics)
        self.statistics.total_messages = len(self.flow_state.messages)

        # Calculate session duration
        if self.session_state.end_time:
            duration = self.session_state.end_time - self.session_state.start_time
        else:
            duration = datetime.now() - self.session_state.start_time
        self.statistics.session_duration = duration.total_seconds()

        # Clear dirty flags
        self._dirty_layers.clear()
        self._last_sync = datetime.now()
        self._state_version += 1

        logger.debug(f"State synchronization completed (version {self._state_version})")

    def get_unified_state(self) -> Dict[str, Any]:
        """Get complete unified state for external use.

        Returns:
            Dictionary containing all state information
        """
        self.sync_all_states()

        return {
            "session_id": self.session_id,
            "state_version": self._state_version,
            "last_sync": self._last_sync,
            # Session state
            "session": {
                "session_id": self.session_state.session_id,
                "start_time": self.session_state.start_time,
                "end_time": self.session_state.end_time,
                "main_topic": self.session_state.main_topic,
                "current_phase": self.session_state.current_phase,
                "total_phases": self.session_state.total_phases,
                "is_active": self.session_state.is_active,
                "termination_reason": self.session_state.termination_reason,
            },
            # Flow state (compatible with existing VirtualAgoraState)
            "flow": {
                "current_node": self.flow_state.current_node,
                "current_round": self.flow_state.current_round,
                "active_topic": self.flow_state.active_topic,
                "topic_queue": self.flow_state.topic_queue.copy(),
                "completed_topics": self.flow_state.completed_topics.copy(),
                "messages": self.flow_state.messages.copy(),
                "user_approves_continuation": self.flow_state.user_approves_continuation,
                "user_requests_end": self.flow_state.user_requests_end,
                "user_requested_modification": self.flow_state.user_requested_modification,
                "user_forced_conclusion": self.flow_state.user_forced_conclusion,
                "agents_vote_end_session": self.flow_state.agents_vote_end_session,
                "agent_recommendation": self.flow_state.agent_recommendation,
                "agenda_approved": self.flow_state.agenda_approved,
                "proposed_agenda": self.flow_state.proposed_agenda.copy(),
                "final_agenda": self.flow_state.final_agenda.copy(),
                "awaiting_user_input": self.flow_state.awaiting_user_input,
                "last_interrupt_type": self.flow_state.last_interrupt_type,
            },
            # UI state
            "ui": {
                "display_mode": self.ui_state.display_mode,
                "show_debug_info": self.ui_state.show_debug_info,
                "theme": self.ui_state.theme,
                "last_update": self.ui_state.last_update,
                "notifications": self.ui_state.notifications.copy(),
            },
            # Statistics
            "statistics": {
                "topics_discussed": self.statistics.topics_discussed,
                "total_messages": self.statistics.total_messages,
                "interrupts_handled": self.statistics.interrupts_handled,
                "checkpoints_created": self.statistics.checkpoints_created,
                "errors_encountered": self.statistics.errors_encountered,
                "user_interactions": self.statistics.user_interactions,
                "average_response_time": self.statistics.average_response_time,
                "session_duration": self.statistics.session_duration,
                "completion_rate": self.statistics.completion_rate,
            },
        }

    def get_legacy_state(self) -> Dict[str, Any]:
        """Get state in legacy VirtualAgoraState format for compatibility.

        This method provides backwards compatibility with existing code
        that expects the old state format.

        Returns:
            State dictionary in legacy format
        """
        self.sync_all_states()

        # Build legacy state format
        legacy_state = {
            # Core session fields
            "session_id": self.session_state.session_id,
            "current_phase": self.flow_state.current_phase,
            "current_round": self.flow_state.current_round,
            # Topic management
            "active_topic": self.flow_state.active_topic,
            "topic_queue": self.flow_state.topic_queue.copy(),
            "completed_topics": self.flow_state.completed_topics.copy(),
            # Messages
            "messages": self.flow_state.messages.copy(),
            # User interaction state
            "user_approves_continuation": self.flow_state.user_approves_continuation,
            "user_requests_end": self.flow_state.user_requests_end,
            "user_requested_modification": self.flow_state.user_requested_modification,
            "user_forced_conclusion": self.flow_state.user_forced_conclusion,
            # Agent state
            "agents_vote_end_session": self.flow_state.agents_vote_end_session,
            "agent_recommendation": self.flow_state.agent_recommendation,
            # Agenda state
            "agenda_approved": self.flow_state.agenda_approved,
            "proposed_agenda": self.flow_state.proposed_agenda.copy(),
            "final_agenda": self.flow_state.final_agenda.copy(),
            # Timestamps
            "start_time": self.session_state.start_time,
            "end_time": self.session_state.end_time,
        }

        return legacy_state

    def validate_state_consistency(self) -> List[str]:
        """Validate state consistency across all layers.

        Returns:
            List of validation errors found
        """
        errors = []

        # Check session-flow consistency
        if self.session_state.current_phase != self.flow_state.current_phase:
            errors.append(
                f"Phase mismatch: session={self.session_state.current_phase}, flow={self.flow_state.current_phase}"
            )

        # Check completed topics consistency
        if self.statistics.topics_discussed != len(self.flow_state.completed_topics):
            errors.append(
                f"Topics count mismatch: stats={self.statistics.topics_discussed}, flow={len(self.flow_state.completed_topics)}"
            )

        # Check message count consistency
        if self.statistics.total_messages != len(self.flow_state.messages):
            errors.append(
                f"Message count mismatch: stats={self.statistics.total_messages}, flow={len(self.flow_state.messages)}"
            )

        # Check topic queue integrity
        for i, topic in enumerate(self.flow_state.topic_queue):
            if not isinstance(topic, str):
                errors.append(f"Invalid topic at index {i}: {type(topic)} - {topic}")
            elif not topic.strip():
                errors.append(f"Empty topic at index {i}")

        # Check completed topics integrity
        for i, topic in enumerate(self.flow_state.completed_topics):
            if not isinstance(topic, str):
                errors.append(
                    f"Invalid completed topic at index {i}: {type(topic)} - {topic}"
                )
            elif not topic.strip():
                errors.append(f"Empty completed topic at index {i}")

        self._validation_errors = errors

        if errors:
            logger.warning(f"State validation found {len(errors)} errors")
            for error in errors:
                logger.warning(f"Validation error: {error}")

        return errors

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for debugging.

        Returns:
            Dictionary with state summary information
        """
        return {
            "session_id": self.session_id,
            "state_version": self._state_version,
            "is_active": self.session_state.is_active,
            "current_phase": self.session_state.current_phase,
            "active_topic": self.flow_state.active_topic,
            "topics_remaining": len(self.flow_state.topic_queue),
            "topics_completed": len(self.flow_state.completed_topics),
            "total_messages": len(self.flow_state.messages),
            "awaiting_user_input": self.flow_state.awaiting_user_input,
            "user_approves_continuation": self.flow_state.user_approves_continuation,
            "agents_vote_end_session": self.flow_state.agents_vote_end_session,
            "dirty_layers": [layer.value for layer in self._dirty_layers],
            "last_sync": self._last_sync,
            "validation_errors": len(self._validation_errors),
        }

    def _mark_dirty(self, layer: StateLayer) -> None:
        """Mark a state layer as dirty (needing synchronization)."""
        self._dirty_layers.add(layer)

    def _record_state_change(
        self, layer: StateLayer, updates: Dict[str, Any], old_values: Dict[str, Any]
    ) -> None:
        """Record a state change for history tracking."""
        change_record = {
            "timestamp": datetime.now(),
            "layer": layer.value,
            "updates": deepcopy(updates),
            "old_values": deepcopy(old_values),
            "state_version": self._state_version,
        }

        self._sync_history.append(change_record)

        # Keep only last 100 changes to prevent memory bloat
        if len(self._sync_history) > 100:
            self._sync_history = self._sync_history[-100:]

    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get state change history for debugging.

        Returns:
            List of state change records
        """
        return self._sync_history.copy()
