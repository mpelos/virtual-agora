"""Tools package for Virtual Agora.

This package provides tools that agents can use during discussions,
including proposal creation, voting, and summarization tools.
"""

from virtual_agora.tools.tool_integration import (
    ProposalTool,
    VotingTool,
    SummaryTool,
    create_discussion_tools,
    count_votes,
    format_agenda,
)
from virtual_agora.tools.tool_node_wrapper import (
    VirtualAgoraToolNode,
    create_tool_node_with_fallback,
)
from virtual_agora.tools.tool_serialization import (
    serialize_tool_call,
    deserialize_tool_call,
    serialize_tool_calls_batch,
    deserialize_tool_calls_batch,
    serialize_ai_message_with_tools,
    deserialize_ai_message_with_tools,
    serialize_tool_message,
    deserialize_tool_message,
    extract_tool_calls_from_state,
    extract_tool_results_from_state,
    match_tool_calls_with_results,
    create_tool_call_summary,
)
from virtual_agora.tools.tool_error_handling import (
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError,
    ToolRetryableError,
    create_error_tool_message,
    handle_tool_error,
    validate_tool_input,
    with_tool_error_handling,
    ToolErrorRecoveryStrategy,
)

__all__ = [
    # Tool classes
    "ProposalTool",
    "VotingTool",
    "SummaryTool",
    # Tool factory functions
    "create_discussion_tools",
    "count_votes",
    "format_agenda",
    # ToolNode wrapper
    "VirtualAgoraToolNode",
    "create_tool_node_with_fallback",
    # Serialization utilities
    "serialize_tool_call",
    "deserialize_tool_call",
    "serialize_tool_calls_batch",
    "deserialize_tool_calls_batch",
    "serialize_ai_message_with_tools",
    "deserialize_ai_message_with_tools",
    "serialize_tool_message",
    "deserialize_tool_message",
    "extract_tool_calls_from_state",
    "extract_tool_results_from_state",
    "match_tool_calls_with_results",
    "create_tool_call_summary",
    # Error handling
    "ToolExecutionError",
    "ToolValidationError",
    "ToolTimeoutError",
    "ToolRetryableError",
    "create_error_tool_message",
    "handle_tool_error",
    "validate_tool_input",
    "with_tool_error_handling",
    "ToolErrorRecoveryStrategy",
]
