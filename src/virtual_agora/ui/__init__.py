"""Terminal UI components for Virtual Agora.

This module provides rich terminal interface components for
displaying agent discussions, prompts, and system messages.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .console import Console, ConsoleTheme
    from .prompts import UserPrompt, PromptType

__all__ = [
    "Console",
    "ConsoleTheme",
    "UserPrompt",
    "PromptType",
]