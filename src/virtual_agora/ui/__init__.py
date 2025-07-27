"""Terminal UI components for Virtual Agora.

This module provides rich terminal interface components for
displaying agent discussions, prompts, and system messages.
"""

from typing import TYPE_CHECKING

# Import core components
from .console import (
    VirtualAgoraConsole,
    get_console,
    print_agent_message,
    print_system_message,
    print_user_prompt,
)
from .theme import (
    VirtualAgoraTheme,
    ProviderType,
    MessageType,
    get_current_theme,
    set_theme,
)
from .progress import (
    VirtualAgoraProgress,
    OperationType,
    ProgressType,
    operation_spinner,
)
from .discussion_display import (
    DiscussionDisplay,
    get_discussion_display,
    add_agent_message,
    add_moderator_message,
)
from .interactive import (
    VirtualAgoraPrompts,
    PromptOption,
    ask_text,
    ask_choice,
    ask_confirmation,
)
from .dashboard import VirtualAgoraDashboard, get_dashboard, PhaseType, AgentStatus
from .accessibility import (
    AccessibilityManager,
    get_accessibility_manager,
    initialize_accessibility,
)
from .formatters import VirtualAgoraFormatter, get_formatter, FormatType, format_content

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .console import VirtualAgoraConsole, ConsoleCapabilities
    from .theme import VirtualAgoraTheme, AccessibilityOptions
    from .progress import ProgressSpinner, ProgressBar
    from .discussion_display import DiscussionMessage, DiscussionRound
    from .interactive import ValidationRule, PromptType
    from .dashboard import SessionStatus, AgentInfo
    from .accessibility import AccessibilityProfile, AccessibilityLevel
    from .formatters import FormattingOptions, ExportFormat

__all__ = [
    # Console components
    "VirtualAgoraConsole",
    "get_console",
    "print_agent_message",
    "print_system_message",
    "print_user_prompt",
    # Theme system
    "VirtualAgoraTheme",
    "ProviderType",
    "MessageType",
    "get_current_theme",
    "set_theme",
    # Progress indicators
    "VirtualAgoraProgress",
    "OperationType",
    "ProgressType",
    "operation_spinner",
    # Discussion display
    "DiscussionDisplay",
    "get_discussion_display",
    "add_agent_message",
    "add_moderator_message",
    # Interactive components
    "VirtualAgoraPrompts",
    "PromptOption",
    "ask_text",
    "ask_choice",
    "ask_confirmation",
    # Dashboard
    "VirtualAgoraDashboard",
    "get_dashboard",
    "PhaseType",
    "AgentStatus",
    # Accessibility
    "AccessibilityManager",
    "get_accessibility_manager",
    "initialize_accessibility",
    # Formatters
    "VirtualAgoraFormatter",
    "get_formatter",
    "FormatType",
    "format_content",
]
