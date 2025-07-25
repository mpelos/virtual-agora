"""Core application logic for Virtual Agora.

This module contains the fundamental components for managing
application state, workflow orchestration, and session handling.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .state import ApplicationState
    from .workflow import WorkflowEngine
    from .session import SessionManager

__all__ = [
    "ApplicationState",
    "WorkflowEngine", 
    "SessionManager",
]