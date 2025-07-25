"""Application state management for Virtual Agora.

This package provides the core state management functionality for Virtual Agora
sessions, including state schema, validation, and management utilities.
"""

from virtual_agora.state.schema import (
    VirtualAgoraState,
    AgentInfo,
    Message,
    Vote,
    PhaseTransition,
)
from virtual_agora.state.manager import StateManager
from virtual_agora.state.validators import StateValidator

__all__ = [
    "VirtualAgoraState",
    "AgentInfo",
    "Message",
    "Vote",
    "PhaseTransition",
    "StateManager",
    "StateValidator",
]