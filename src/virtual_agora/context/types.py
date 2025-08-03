"""Context data types for Virtual Agora context management system."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from virtual_agora.state.schema import Message, RoundSummary


@dataclass
class ContextData:
    """Clean container for context data provided to agents.

    This class encapsulates all the different types of context that
    agents might need, allowing for clean separation and agent-specific
    context building.
    """

    # Core prompt and system context
    system_prompt: str

    # Context directory files (for discussion agents)
    context_documents: Optional[str] = None

    # User's initial input/theme
    user_input: Optional[str] = None

    # Topic-specific discussion messages (legacy - use processed messages instead)
    topic_messages: List[Message] = None

    # Round summaries (all or topic-specific)
    round_summaries: List[RoundSummary] = None

    # Generated topic reports (for session reports)
    topic_reports: Dict[str, str] = None

    # Process-specific context (for moderators)
    process_context: Optional[Dict[str, Any]] = None

    # Content to summarize (for summarizers)
    content_to_summarize: Optional[List[str]] = None

    # NEW: Processed message types for enhanced context assembly

    # User participation messages from previous rounds
    user_participation_messages: Optional[List[Any]] = None  # List[ProcessedMessage]

    # Current round messages from colleagues
    current_round_messages: Optional[List[Any]] = None  # List[ProcessedMessage]

    # Additional metadata for context validation and debugging
    metadata: Optional[Dict[str, Any]] = None

    # Formatted context messages ready for LangChain
    formatted_context_messages: Optional[List[Any]] = None  # List[BaseMessage]

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.topic_messages is None:
            self.topic_messages = []
        if self.round_summaries is None:
            self.round_summaries = []
        if self.topic_reports is None:
            self.topic_reports = {}
        if self.user_participation_messages is None:
            self.user_participation_messages = []
        if self.current_round_messages is None:
            self.current_round_messages = []
        if self.metadata is None:
            self.metadata = {}
        if self.formatted_context_messages is None:
            self.formatted_context_messages = []

    @property
    def has_context_documents(self) -> bool:
        """Check if context documents are available."""
        return self.context_documents is not None and self.context_documents.strip()

    @property
    def has_user_input(self) -> bool:
        """Check if user input is available."""
        return self.user_input is not None and self.user_input.strip()

    @property
    def has_topic_messages(self) -> bool:
        """Check if topic messages are available."""
        return len(self.topic_messages) > 0

    @property
    def has_round_summaries(self) -> bool:
        """Check if round summaries are available."""
        return len(self.round_summaries) > 0

    @property
    def has_topic_reports(self) -> bool:
        """Check if topic reports are available."""
        return len(self.topic_reports) > 0

    @property
    def has_user_participation_messages(self) -> bool:
        """Check if user participation messages are available."""
        return len(self.user_participation_messages) > 0

    @property
    def has_current_round_messages(self) -> bool:
        """Check if current round messages are available."""
        return len(self.current_round_messages) > 0
