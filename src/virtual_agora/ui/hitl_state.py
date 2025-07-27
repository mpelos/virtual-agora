"""Enhanced HITL state management for Virtual Agora v1.3.

This module provides comprehensive state management for all Human-in-the-Loop
interactions in the v1.3 node-centric architecture.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class HITLApprovalType(Enum):
    """All HITL interaction types in v1.3."""

    # Existing types from v1.1
    THEME_INPUT = "theme_input"
    AGENDA_APPROVAL = "agenda_approval"

    # New v1.3 types
    PERIODIC_STOP = "periodic_stop"  # Every 5 rounds
    TOPIC_OVERRIDE = "topic_override"  # Force topic end
    TOPIC_CONTINUATION = "topic_continuation"
    AGENT_POLL_OVERRIDE = "agent_poll_override"
    SESSION_CONTINUATION = "session_continuation"
    FINAL_REPORT_APPROVAL = "final_report_approval"


class HITLInteraction:
    """Represents a single HITL interaction."""

    def __init__(
        self,
        approval_type: HITLApprovalType,
        prompt_message: str,
        options: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ):
        self.approval_type = approval_type
        self.prompt_message = prompt_message
        self.options = options or []
        self.context = context or {}
        self.timeout_seconds = timeout_seconds
        self.timestamp = datetime.now()
        self.response = None
        self.response_time = None
        self.duration_seconds = None

    def record_response(self, response: Any) -> None:
        """Record user response with timing."""
        self.response = response
        self.response_time = datetime.now()
        self.duration_seconds = (self.response_time - self.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "approval_type": self.approval_type.value,
            "prompt_message": self.prompt_message,
            "options": self.options,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "response": self.response,
            "response_time": (
                self.response_time.isoformat() if self.response_time else None
            ),
            "duration_seconds": self.duration_seconds,
        }


class HITLContext:
    """Context provided to HITL interactions."""

    def __init__(
        self,
        current_round: Optional[int] = None,
        active_topic: Optional[str] = None,
        completed_topics: Optional[List[str]] = None,
        remaining_topics: Optional[List[str]] = None,
        proposed_agenda: Optional[List[str]] = None,
        session_stats: Optional[Dict[str, Any]] = None,
        agent_vote_result: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ):
        self.current_round = current_round
        self.active_topic = active_topic
        self.completed_topics = completed_topics or []
        self.remaining_topics = remaining_topics or []
        self.proposed_agenda = proposed_agenda or []
        self.session_stats = session_stats or {}
        self.agent_vote_result = agent_vote_result
        self.custom_data = custom_data or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for interaction context."""
        return {
            "current_round": self.current_round,
            "active_topic": self.active_topic,
            "completed_topics": self.completed_topics,
            "remaining_topics": self.remaining_topics,
            "proposed_agenda": self.proposed_agenda,
            "session_stats": self.session_stats,
            "agent_vote_result": self.agent_vote_result,
            **self.custom_data,
        }


class HITLResponse:
    """Structured response from HITL interaction."""

    def __init__(
        self,
        approved: bool,
        action: str,
        modified_data: Optional[Any] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.approved = approved
        self.action = action  # e.g., 'continue', 'end', 'modify', 'skip'
        self.modified_data = modified_data
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state updates."""
        return {
            "approved": self.approved,
            "action": self.action,
            "modified_data": self.modified_data,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class HITLStateTracker:
    """Tracks HITL interaction history and state."""

    def __init__(self):
        self.interactions: List[HITLInteraction] = []
        self.current_interaction: Optional[HITLInteraction] = None
        self.periodic_stop_counter = 0
        self.last_periodic_stop_round = 0
        self.approval_history: List[Dict[str, Any]] = []

    def start_interaction(self, interaction: HITLInteraction) -> None:
        """Start a new HITL interaction."""
        self.current_interaction = interaction
        self.interactions.append(interaction)

    def complete_interaction(self, response: HITLResponse) -> None:
        """Complete the current interaction."""
        if self.current_interaction:
            self.current_interaction.record_response(response)
            self.approval_history.append(
                {
                    "type": self.current_interaction.approval_type.value,
                    "result": response.action,
                    "timestamp": response.timestamp,
                    "duration": self.current_interaction.duration_seconds,
                    **response.metadata,
                }
            )
            self.current_interaction = None

    def should_trigger_periodic_stop(self, current_round: int) -> bool:
        """Check if periodic stop should be triggered."""
        if current_round > 0 and current_round % 5 == 0:
            if current_round > self.last_periodic_stop_round:
                return True
        return False

    def record_periodic_stop(self, round_number: int) -> None:
        """Record that a periodic stop occurred."""
        self.last_periodic_stop_round = round_number
        self.periodic_stop_counter += 1

    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get statistics about HITL interactions."""
        total_interactions = len(self.interactions)
        if total_interactions == 0:
            return {
                "total_interactions": 0,
                "average_response_time": 0,
                "interactions_by_type": {},
            }

        total_duration = sum(
            i.duration_seconds
            for i in self.interactions
            if i.duration_seconds is not None
        )

        interactions_by_type = {}
        for interaction in self.interactions:
            type_name = interaction.approval_type.value
            interactions_by_type[type_name] = interactions_by_type.get(type_name, 0) + 1

        return {
            "total_interactions": total_interactions,
            "average_response_time": (
                total_duration / total_interactions if total_interactions > 0 else 0
            ),
            "interactions_by_type": interactions_by_type,
            "periodic_stops": self.periodic_stop_counter,
            "last_periodic_stop_round": self.last_periodic_stop_round,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire state to dictionary."""
        return {
            "interactions": [i.to_dict() for i in self.interactions],
            "periodic_stop_counter": self.periodic_stop_counter,
            "last_periodic_stop_round": self.last_periodic_stop_round,
            "approval_history": self.approval_history,
            "stats": self.get_interaction_stats(),
        }
