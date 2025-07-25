"""Agent implementations for Virtual Agora.

This module contains the base agent classes and specific implementations
for discussion agents and the moderator agent.
"""

from typing import TYPE_CHECKING

# Import actual implementations
from .llm_agent import LLMAgent
from .moderator import ModeratorAgent

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .base import BaseAgent, AgentRole
    from .discussion import DiscussionAgent
    from .tool_enabled_factory import (
        create_tool_enabled_moderator,
        create_tool_enabled_participant,
        create_discussion_agents_with_tools,
        add_tools_to_existing_agent,
        create_specialized_agent,
    )

__all__ = [
    "BaseAgent",
    "AgentRole",
    "DiscussionAgent",
    "ModeratorAgent",
    "LLMAgent",
    "create_tool_enabled_moderator",
    "create_tool_enabled_participant",
    "create_discussion_agents_with_tools",
    "add_tools_to_existing_agent",
    "create_specialized_agent",
]