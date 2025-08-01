"""Context management system for Virtual Agora.

This module provides a clean Context Strategy Pattern implementation for
managing different types of context data that agents need. It separates
context loading, context building, and context injection to provide
agent-specific context while maintaining clean architecture.
"""

from .types import ContextData
from .repository import ContextRepository
from .builders import (
    ContextBuilder,
    DiscussionAgentContextBuilder,
    ReportWriterContextBuilder,
    ModeratorContextBuilder,
    SummarizerContextBuilder,
)

__all__ = [
    "ContextData",
    "ContextRepository",
    "ContextBuilder",
    "DiscussionAgentContextBuilder",
    "ReportWriterContextBuilder",
    "ModeratorContextBuilder",
    "SummarizerContextBuilder",
]
