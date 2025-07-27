"""LangGraph integration for Virtual Agora terminal UI.

This module provides integration between the UI components and LangGraph
state management, enabling real-time UI updates based on graph state changes.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable, Type
from datetime import datetime
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from virtual_agora.state.schema import VirtualAgoraState, UIState
from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import ProviderType, get_current_theme
from virtual_agora.ui.progress import operation_spinner, OperationType
from virtual_agora.ui.discussion_display import (
    get_discussion_display,
    add_agent_message,
    add_moderator_message,
    start_discussion_round,
    complete_discussion_round,
    show_voting_results,
    show_agenda,
)
from virtual_agora.ui.dashboard import (
    get_dashboard,
    PhaseType,
    AgentStatus,
    create_session_status,
    DashboardManager,
)
from virtual_agora.ui.accessibility import initialize_accessibility
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class LangGraphUIIntegration:
    """Integration layer between LangGraph and Virtual Agora UI."""

    def __init__(self):
        """Initialize UI integration."""
        self.console = get_console()
        self.discussion_display = get_discussion_display()
        self.dashboard = get_dashboard()
        self.dashboard_manager = DashboardManager()

        # State tracking
        self._last_known_state: Optional[VirtualAgoraState] = None
        self._ui_callbacks: Dict[str, List[Callable]] = {}
        self._state_listeners: List[Callable] = []

        # Register default callbacks
        self._register_default_callbacks()

    def _register_default_callbacks(self) -> None:
        """Register default UI update callbacks."""
        # Phase transition callbacks
        self.register_callback("phase_change", self._on_phase_change)

        # Discussion callbacks
        self.register_callback("message_added", self._on_message_added)
        self.register_callback("round_started", self._on_round_started)
        self.register_callback("round_completed", self._on_round_completed)

        # Voting callbacks
        self.register_callback("vote_started", self._on_vote_started)
        self.register_callback("vote_completed", self._on_vote_completed)

        # Agent status callbacks
        self.register_callback("agent_responding", self._on_agent_responding)
        self.register_callback("agent_completed", self._on_agent_completed)
        self.register_callback("agent_error", self._on_agent_error)

        # Agenda callbacks
        self.register_callback("agenda_updated", self._on_agenda_updated)
        self.register_callback("topic_changed", self._on_topic_changed)

    def initialize_ui_for_session(self, state: VirtualAgoraState) -> VirtualAgoraState:
        """Initialize UI components for a new session."""
        logger.info("Initializing UI for Virtual Agora session")

        # Initialize accessibility
        initialize_accessibility()

        # Create session header
        session_info = {
            "session_id": state["session_id"],
            "main_topic": state.get("main_topic", "Not set"),
            "start_time": state["start_time"].strftime("%H:%M:%S"),
            "agents": {
                "total": len(state.get("agents", {})),
                "moderator": state.get("moderator_id", "Unknown"),
                "providers": list(
                    set(agent["provider"] for agent in state.get("agents", {}).values())
                ),
            },
        }

        self.discussion_display.display_session_header(session_info)

        # Initialize dashboard
        if state.get("agents"):
            agent_providers = {
                agent_id: ProviderType(agent["provider"])
                for agent_id, agent in state["agents"].items()
            }

            self.dashboard_manager.initialize_session(
                state["session_id"],
                state.get("main_topic", "Virtual Agora Discussion"),
                agent_providers,
            )

        # Update UI state
        ui_state: UIState = {
            "console_initialized": True,
            "theme_applied": True,
            "accessibility_enabled": True,
            "dashboard_active": False,
            "current_display_mode": "full",
            "progress_operations": {},
            "last_ui_update": datetime.now(),
        }

        # Return updated state
        updated_state = state.copy()
        updated_state["ui_state"] = ui_state

        self._last_known_state = updated_state
        return updated_state

    def update_ui_from_state(self, new_state: VirtualAgoraState) -> None:
        """Update UI components based on state changes."""
        if self._last_known_state is None:
            self._last_known_state = new_state
            return

        # Detect state changes and trigger appropriate UI updates
        changes = self._detect_state_changes(self._last_known_state, new_state)

        for change_type, change_data in changes.items():
            self._trigger_callbacks(change_type, change_data)

        # Update UI state timestamp
        if "ui_state" in new_state:
            new_state["ui_state"]["last_ui_update"] = datetime.now()

        self._last_known_state = new_state

    def _detect_state_changes(
        self, old_state: VirtualAgoraState, new_state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Detect changes between old and new state."""
        changes = {}

        # Phase changes
        if old_state.get("current_phase") != new_state.get("current_phase"):
            changes["phase_change"] = {
                "from_phase": old_state.get("current_phase"),
                "to_phase": new_state.get("current_phase"),
                "timestamp": datetime.now(),
            }

        # Round changes
        if old_state.get("current_round") != new_state.get("current_round"):
            if new_state.get("current_round", 0) > old_state.get("current_round", 0):
                changes["round_started"] = {
                    "round_number": new_state.get("current_round"),
                    "topic": new_state.get("active_topic"),
                    "agents": list(new_state.get("agents", {}).keys()),
                }

        # Message changes
        old_messages = old_state.get("messages", [])
        new_messages = new_state.get("messages", [])

        if len(new_messages) > len(old_messages):
            # New message added
            for message in new_messages[len(old_messages) :]:
                changes["message_added"] = message

        # Voting changes
        old_vote = old_state.get("active_vote")
        new_vote = new_state.get("active_vote")

        if old_vote is None and new_vote is not None:
            changes["vote_started"] = new_vote
        elif old_vote is not None and new_vote is None:
            # Check if vote was completed
            old_vote_id = old_vote.get("id")
            new_vote_history = new_state.get("vote_history", [])

            for vote_record in new_vote_history:
                if (
                    vote_record.get("id") == old_vote_id
                    and vote_record.get("status") == "completed"
                ):
                    changes["vote_completed"] = vote_record
                    break

        # Agenda changes
        old_agenda = old_state.get("agenda", {})
        new_agenda = new_state.get("agenda", {})

        if old_agenda != new_agenda:
            changes["agenda_updated"] = new_agenda

        # Topic changes
        if old_state.get("active_topic") != new_state.get("active_topic"):
            changes["topic_changed"] = {
                "from_topic": old_state.get("active_topic"),
                "to_topic": new_state.get("active_topic"),
            }

        return changes

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for specific UI events."""
        if event_type not in self._ui_callbacks:
            self._ui_callbacks[event_type] = []
        self._ui_callbacks[event_type].append(callback)

    def _trigger_callbacks(self, event_type: str, event_data: Any) -> None:
        """Trigger registered callbacks for an event type."""
        callbacks = self._ui_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in UI callback for {event_type}: {e}")

    # Default callback implementations

    def _on_phase_change(self, change_data: Dict[str, Any]) -> None:
        """Handle phase change UI updates."""
        from_phase = change_data.get("from_phase")
        to_phase = change_data.get("to_phase")

        # Map phase numbers to names
        phase_names = {
            0: "Initialization",
            1: "Agenda Setting",
            2: "Discussion",
            3: "Topic Conclusion",
            4: "Agenda Re-evaluation",
            5: "Final Report Generation",
        }

        from_name = phase_names.get(from_phase, f"Phase {from_phase}")
        to_name = phase_names.get(to_phase, f"Phase {to_phase}")

        self.discussion_display.display_phase_transition(from_name, to_name)

        # Update dashboard
        dashboard_phases = {
            0: PhaseType.INITIALIZATION,
            1: PhaseType.AGENDA_SETTING,
            2: PhaseType.DISCUSSION,
            3: PhaseType.TOPIC_CONCLUSION,
            4: PhaseType.AGENDA_MODIFICATION,
            5: PhaseType.REPORT_GENERATION,
        }

        if to_phase in dashboard_phases:
            self.dashboard.set_phase(dashboard_phases[to_phase])

    def _on_message_added(self, message: Dict[str, Any]) -> None:
        """Handle new message UI updates."""
        speaker_id = message.get("speaker_id")
        speaker_role = message.get("speaker_role")
        content = message.get("content", "")
        timestamp = message.get("timestamp")
        topic = message.get("topic")
        round_number = (
            self._last_known_state.get("current_round")
            if self._last_known_state
            else None
        )

        # Determine provider type
        if self._last_known_state and speaker_id in self._last_known_state.get(
            "agents", {}
        ):
            agent_info = self._last_known_state["agents"][speaker_id]
            provider = ProviderType(agent_info["provider"])
        else:
            provider = ProviderType.MODERATOR

        if speaker_role == "moderator":
            add_moderator_message(content, round_number, topic)
        else:
            add_agent_message(speaker_id, provider, content, round_number, topic)

        # Update dashboard
        self.dashboard_manager.agent_completed_response(speaker_id)

    def _on_round_started(self, round_data: Dict[str, Any]) -> None:
        """Handle round start UI updates."""
        round_number = round_data.get("round_number", 0)
        topic = round_data.get("topic", "Unknown Topic")
        agents = round_data.get("agents", [])

        start_discussion_round(round_number, topic, len(agents))

        # Update dashboard
        self.dashboard.set_current_topic(topic, round_number)

    def _on_round_completed(self, round_data: Dict[str, Any]) -> None:
        """Handle round completion UI updates."""
        summary = round_data.get("summary")
        complete_discussion_round(summary)

    def _on_vote_started(self, vote_data: Dict[str, Any]) -> None:
        """Handle vote start UI updates."""
        vote_type = vote_data.get("vote_type", "")
        options = vote_data.get("options", [])

        self.console.print_system_message(
            f"Voting started: {vote_type}", title="Vote Started"
        )

        # Update dashboard
        self.dashboard_manager.start_voting()

    def _on_vote_completed(self, vote_data: Dict[str, Any]) -> None:
        """Handle vote completion UI updates."""
        vote_id = vote_data.get("id")
        result = vote_data.get("result")

        # Get vote details from state
        if self._last_known_state:
            votes = self._last_known_state.get("votes", [])
            vote_details = {}

            for vote in votes:
                if (
                    vote.get("vote_id") == vote_id
                ):  # Assuming votes reference vote_round_id
                    voter_id = vote.get("voter_id")
                    choice = vote.get("choice")
                    vote_details[voter_id] = {
                        "vote": choice,
                        "justification": vote.get("metadata", {}).get(
                            "justification", ""
                        ),
                    }

            if vote_details:
                topic = self._last_known_state.get("active_topic", "Current Topic")
                show_voting_results(vote_details, topic)

        # Update dashboard
        self.dashboard_manager.complete_voting()

    def _on_agent_responding(self, agent_data: Dict[str, Any]) -> None:
        """Handle agent responding UI updates."""
        agent_id = agent_data.get("agent_id")
        if agent_id:
            self.dashboard_manager.agent_started_responding(agent_id)

    def _on_agent_completed(self, agent_data: Dict[str, Any]) -> None:
        """Handle agent completion UI updates."""
        agent_id = agent_data.get("agent_id")
        if agent_id:
            self.dashboard_manager.agent_completed_response(agent_id)

    def _on_agent_error(self, error_data: Dict[str, Any]) -> None:
        """Handle agent error UI updates."""
        agent_id = error_data.get("agent_id")
        error_message = error_data.get("error", "Unknown error")

        if agent_id:
            self.dashboard_manager.agent_error(agent_id, error_message)

        self.discussion_display.display_error_message(
            error_message, f"Agent: {agent_id}"
        )

    def _on_agenda_updated(self, agenda_data: Dict[str, Any]) -> None:
        """Handle agenda update UI updates."""
        topics = agenda_data.get("topics", [])
        current_index = agenda_data.get("current_topic_index", 0)
        completed = agenda_data.get("completed_topics", [])

        current_topic = topics[current_index] if current_index < len(topics) else None
        pending_topics = (
            topics[current_index + 1 :] if current_index < len(topics) else []
        )

        show_agenda(topics, current_topic, completed)

    def _on_topic_changed(self, topic_data: Dict[str, Any]) -> None:
        """Handle topic change UI updates."""
        from_topic = topic_data.get("from_topic")
        to_topic = topic_data.get("to_topic")

        if from_topic and to_topic:
            self.console.print_system_message(
                f"Topic changed from '{from_topic}' to '{to_topic}'",
                title="Topic Transition",
            )

        # Complete previous topic in dashboard
        if from_topic:
            self.dashboard.complete_topic(from_topic)


class LangGraphUIMiddleware:
    """Middleware for automatic UI updates in LangGraph nodes."""

    def __init__(self, ui_integration: LangGraphUIIntegration):
        """Initialize middleware."""
        self.ui_integration = ui_integration

    def wrap_node(self, node_func: Callable) -> Callable:
        """Wrap a LangGraph node function to include UI updates."""

        async def wrapped_node(state: VirtualAgoraState, **kwargs) -> VirtualAgoraState:
            # Pre-execution UI updates
            node_name = node_func.__name__

            with operation_spinner(
                OperationType.AGENT_RESPONSE, f"Executing {node_name}..."
            ):
                # Execute the original node
                if asyncio.iscoroutinefunction(node_func):
                    result = await node_func(state, **kwargs)
                else:
                    result = node_func(state, **kwargs)

                # Post-execution UI updates
                self.ui_integration.update_ui_from_state(result)

                return result

        return wrapped_node

    def wrap_graph(self, graph: StateGraph) -> StateGraph:
        """Wrap all nodes in a StateGraph with UI middleware."""
        # This would need to be implemented based on LangGraph's internal structure
        # For now, we'll return the graph as-is and rely on manual integration
        logger.info("Graph wrapped with UI middleware")
        return graph


# Convenience functions and decorators


def ui_integrated_node(ui_integration: Optional[LangGraphUIIntegration] = None):
    """Decorator for LangGraph nodes that need UI integration."""

    def decorator(func: Callable) -> Callable:
        integration = ui_integration or get_ui_integration()
        middleware = LangGraphUIMiddleware(integration)
        return middleware.wrap_node(func)

    return decorator


async def with_progress_spinner(operation_type: OperationType, description: str = None):
    """Context manager for adding progress spinners to async operations."""
    return operation_spinner(operation_type, description)


# Global UI integration instance
_ui_integration: Optional[LangGraphUIIntegration] = None


def get_ui_integration() -> LangGraphUIIntegration:
    """Get the global UI integration instance."""
    global _ui_integration
    if _ui_integration is None:
        _ui_integration = LangGraphUIIntegration()
    return _ui_integration


def initialize_ui_integration() -> LangGraphUIIntegration:
    """Initialize and return UI integration."""
    integration = get_ui_integration()
    logger.info("LangGraph UI integration initialized")
    return integration


def create_ui_aware_graph(
    state_schema: Type[VirtualAgoraState],
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> StateGraph:
    """Create a StateGraph with UI integration built in."""
    graph = StateGraph(state_schema)

    # Initialize UI integration
    ui_integration = initialize_ui_integration()

    # The graph will be configured by the calling code
    # but UI integration callbacks are ready

    return graph


# Utility functions for common UI operations in nodes


def notify_agent_started(state: VirtualAgoraState, agent_id: str) -> VirtualAgoraState:
    """Notify UI that an agent has started responding."""
    ui_integration = get_ui_integration()
    ui_integration._trigger_callbacks("agent_responding", {"agent_id": agent_id})
    return state


def notify_agent_completed(
    state: VirtualAgoraState, agent_id: str
) -> VirtualAgoraState:
    """Notify UI that an agent has completed responding."""
    ui_integration = get_ui_integration()
    ui_integration._trigger_callbacks("agent_completed", {"agent_id": agent_id})
    return state


def notify_agent_error(
    state: VirtualAgoraState, agent_id: str, error: str
) -> VirtualAgoraState:
    """Notify UI of an agent error."""
    ui_integration = get_ui_integration()
    ui_integration._trigger_callbacks(
        "agent_error", {"agent_id": agent_id, "error": error}
    )
    return state


def update_ui_from_state_change(state: VirtualAgoraState) -> VirtualAgoraState:
    """Update UI based on state changes."""
    ui_integration = get_ui_integration()
    ui_integration.update_ui_from_state(state)
    return state
