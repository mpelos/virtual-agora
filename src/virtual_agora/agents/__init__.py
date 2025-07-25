"""Agent implementations for Virtual Agora.

This module contains the base agent classes and specific implementations
for discussion agents and the moderator agent.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .base import BaseAgent, AgentRole
    from .discussion import DiscussionAgent
    from .moderator import ModeratorAgent

__all__ = [
    "BaseAgent",
    "AgentRole",
    "DiscussionAgent",
    "ModeratorAgent",
]