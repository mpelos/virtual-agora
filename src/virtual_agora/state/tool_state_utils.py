"""Utility functions for managing tool-related state in Virtual Agora.

This module provides functions for updating and querying tool execution
state within LangGraph workflows.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from virtual_agora.state.schema import (
    ToolCallInfo,
    ToolExecutionMetrics,
    VirtualAgoraState,
    ToolEnabledState,
)
from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


def create_tool_call_info(
    agent_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    phase: int = -1,
    topic: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> ToolCallInfo:
    """Create a ToolCallInfo object.

    Args:
        agent_id: ID of the agent making the call
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        phase: Current discussion phase
        topic: Current discussion topic
        tool_call_id: Optional ID for the tool call

    Returns:
        ToolCallInfo object
    """
    return ToolCallInfo(
        id=tool_call_id or str(uuid.uuid4()),
        agent_id=agent_id,
        tool_name=tool_name,
        arguments=arguments,
        timestamp=datetime.now(),
        phase=phase,
        topic=topic,
        status="pending",
    )


def update_tool_call_status(
    state: Union[VirtualAgoraState, ToolEnabledState],
    tool_call_id: str,
    status: str,
    result: Optional[str] = None,
    error: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Update the status of a tool call in state.

    Args:
        state: Current state
        tool_call_id: ID of the tool call to update
        status: New status ('executing', 'completed', 'failed')
        result: Tool execution result (if completed)
        error: Error message (if failed)
        execution_time_ms: Execution time in milliseconds

    Returns:
        State update dictionary
    """
    updates = {}

    # Update active tool calls
    if tool_call_id in state.get("active_tool_calls", {}):
        tool_call = state["active_tool_calls"][tool_call_id].copy()
        tool_call["status"] = status

        if result is not None:
            tool_call["result"] = result
        if error is not None:
            tool_call["error"] = error
        if execution_time_ms is not None:
            tool_call["execution_time_ms"] = execution_time_ms

        # Update or remove from active calls
        if status in ["completed", "failed"]:
            # Remove from active calls
            active_calls = state.get("active_tool_calls", {}).copy()
            active_calls.pop(tool_call_id, None)
            updates["active_tool_calls"] = active_calls

            # Add to tool calls history
            updates["tool_calls"] = (
                tool_call  # Use reducer properly - pass individual tool call, not list
            )
        else:
            # Update in active calls
            active_calls = state.get("active_tool_calls", {}).copy()
            active_calls[tool_call_id] = tool_call
            updates["active_tool_calls"] = active_calls

    return updates


def track_tool_execution_from_messages(
    state: Union[VirtualAgoraState, ToolEnabledState],
    message: AIMessage,
    tool_results: List[ToolMessage],
) -> Dict[str, Any]:
    """Track tool execution from AI message and results.

    Args:
        state: Current state
        message: AIMessage containing tool calls
        tool_results: List of ToolMessage results

    Returns:
        State update dictionary
    """
    updates = {}

    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return updates

    # Extract agent ID from message
    agent_id = getattr(message, "name", "unknown")
    phase = state.get("current_phase", -1)
    topic = state.get("active_topic")

    # Create tool call info for each tool call
    tool_calls_to_add = []
    active_calls = state.get("active_tool_calls", {}).copy()

    # Map results by tool_call_id
    results_map = {}
    for result in tool_results:
        if hasattr(result, "tool_call_id"):
            results_map[result.tool_call_id] = result

    for tool_call in message.tool_calls:
        call_id = tool_call.get("id", str(uuid.uuid4()))
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("args", {})

        # Create tool call info
        tool_info = create_tool_call_info(
            agent_id=agent_id,
            tool_name=tool_name,
            arguments=arguments,
            phase=phase,
            topic=topic,
            tool_call_id=call_id,
        )

        # Check if we have a result
        if call_id in results_map:
            result = results_map[call_id]
            tool_info["status"] = "completed"
            tool_info["result"] = result.content

            # Check for error status
            if hasattr(result, "status") and result.status == "error":
                tool_info["status"] = "failed"
                tool_info["error"] = result.content

            # Add execution time if available
            if hasattr(result, "additional_kwargs"):
                exec_time = result.additional_kwargs.get("execution_time", 0) * 1000
                tool_info["execution_time_ms"] = exec_time

            tool_calls_to_add.append(tool_info)
        else:
            # No result yet, add to active calls
            tool_info["status"] = "executing"
            active_calls[call_id] = tool_info

    # Update state
    if tool_calls_to_add:
        updates["tool_calls"] = tool_calls_to_add

    if active_calls != state.get("active_tool_calls", {}):
        updates["active_tool_calls"] = active_calls

    # Update metrics
    if "tool_metrics" in state or tool_calls_to_add:
        updates["tool_metrics"] = update_tool_metrics(
            state.get("tool_metrics"), tool_calls_to_add
        )

    return updates


def update_tool_metrics(
    current_metrics: Optional[ToolExecutionMetrics], new_tool_calls: List[ToolCallInfo]
) -> ToolExecutionMetrics:
    """Update tool execution metrics with new calls.

    Args:
        current_metrics: Current metrics (if any)
        new_tool_calls: New tool calls to add to metrics

    Returns:
        Updated metrics
    """
    if not current_metrics:
        metrics = ToolExecutionMetrics(
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            average_execution_time_ms=0.0,
            calls_by_tool={},
            calls_by_agent={},
            errors_by_type={},
        )
    else:
        # Create a copy
        metrics = dict(current_metrics)

    # Update with new calls
    total_exec_time = metrics["average_execution_time_ms"] * metrics["total_calls"]

    for call in new_tool_calls:
        metrics["total_calls"] += 1

        # Update tool counts
        tool_name = call["tool_name"]
        metrics["calls_by_tool"][tool_name] = (
            metrics["calls_by_tool"].get(tool_name, 0) + 1
        )

        # Update agent counts
        agent_id = call["agent_id"]
        metrics["calls_by_agent"][agent_id] = (
            metrics["calls_by_agent"].get(agent_id, 0) + 1
        )

        # Update success/failure counts
        if call["status"] == "completed":
            metrics["successful_calls"] += 1
        elif call["status"] == "failed":
            metrics["failed_calls"] += 1

            # Track error types
            error_type = "Unknown"
            if "error" in call:
                # Try to extract error type from message
                error_msg = call["error"]
                if "ValidationError" in error_msg:
                    error_type = "ValidationError"
                elif "TimeoutError" in error_msg:
                    error_type = "TimeoutError"
                elif "NetworkError" in error_msg:
                    error_type = "NetworkError"

            metrics["errors_by_type"][error_type] = (
                metrics["errors_by_type"].get(error_type, 0) + 1
            )

        # Update execution time
        if "execution_time_ms" in call:
            total_exec_time += call["execution_time_ms"]

    # Calculate new average
    if metrics["total_calls"] > 0:
        metrics["average_execution_time_ms"] = total_exec_time / metrics["total_calls"]

    return ToolExecutionMetrics(**metrics)


def get_agent_tool_history(
    state: Union[VirtualAgoraState, ToolEnabledState], agent_id: str
) -> List[ToolCallInfo]:
    """Get tool call history for a specific agent.

    Args:
        state: Current state
        agent_id: Agent ID to get history for

    Returns:
        List of tool calls made by the agent
    """
    tool_calls = state.get("tool_calls", [])
    return [call for call in tool_calls if call["agent_id"] == agent_id]


def get_tool_usage_summary(state: Union[VirtualAgoraState, ToolEnabledState]) -> str:
    """Generate a human-readable summary of tool usage.

    Args:
        state: Current state

    Returns:
        Summary string
    """
    metrics = state.get("tool_metrics")
    if not metrics or metrics["total_calls"] == 0:
        return "No tools have been used in this session."

    summary_parts = [
        f"Tool Usage Summary:",
        f"- Total calls: {metrics['total_calls']}",
        f"- Successful: {metrics['successful_calls']} ({metrics['successful_calls']/metrics['total_calls']*100:.1f}%)",
        f"- Failed: {metrics['failed_calls']} ({metrics['failed_calls']/metrics['total_calls']*100:.1f}%)",
        f"- Average execution time: {metrics['average_execution_time_ms']:.1f}ms",
        "",
        "By Tool:",
    ]

    for tool, count in sorted(
        metrics["calls_by_tool"].items(), key=lambda x: x[1], reverse=True
    ):
        summary_parts.append(f"  - {tool}: {count} calls")

    if metrics["calls_by_agent"]:
        summary_parts.extend(["", "By Agent:"])
        for agent, count in sorted(
            metrics["calls_by_agent"].items(), key=lambda x: x[1], reverse=True
        ):
            summary_parts.append(f"  - {agent}: {count} calls")

    if metrics["errors_by_type"]:
        summary_parts.extend(["", "Errors:"])
        for error_type, count in sorted(
            metrics["errors_by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            summary_parts.append(f"  - {error_type}: {count} occurrences")

    return "\n".join(summary_parts)


def should_enable_tools_for_agent(
    state: VirtualAgoraState, agent_id: str, phase: int
) -> bool:
    """Determine if tools should be enabled for an agent in current phase.

    Args:
        state: Current state
        agent_id: Agent ID
        phase: Current phase

    Returns:
        True if tools should be enabled
    """
    # Tools are most useful in certain phases
    if phase == 1:  # Agenda setting - proposal tools useful
        return True
    elif phase == 2:  # Discussion - summary tools useful
        return True
    elif phase == 3:  # Consensus - voting tools useful
        return True

    # Check if agent is already tool-enabled
    if agent_id in state.get("tools_enabled_agents", []):
        return True

    # Moderator might benefit from tools
    if agent_id == state.get("moderator_id"):
        return True

    return False
