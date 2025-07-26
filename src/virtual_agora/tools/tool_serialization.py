"""Tool Call Serialization Utilities for Virtual Agora.

This module provides utilities for serializing and deserializing tool calls
for state management and persistence in LangGraph workflows.
"""

from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


def serialize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a tool call for state storage.

    Args:
        tool_call: Tool call dictionary from AIMessage

    Returns:
        Serialized tool call safe for JSON storage
    """
    serialized = {
        "id": tool_call.get("id", str(uuid.uuid4())),
        "name": tool_call.get("name"),
        "args": tool_call.get("args", {}),
        "timestamp": datetime.now().isoformat(),
    }

    # Handle special argument types
    if isinstance(serialized["args"], dict):
        serialized["args"] = _serialize_args(serialized["args"])

    return serialized


def deserialize_tool_call(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize a tool call from state storage.

    Args:
        serialized: Serialized tool call

    Returns:
        Tool call dictionary for use with AIMessage
    """
    tool_call = {
        "id": serialized["id"],
        "name": serialized["name"],
        "args": serialized.get("args", {}),
    }

    # Restore special argument types
    if isinstance(tool_call["args"], dict):
        tool_call["args"] = _deserialize_args(tool_call["args"])

    return tool_call


def serialize_tool_calls_batch(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Serialize a batch of tool calls.

    Args:
        tool_calls: List of tool calls

    Returns:
        List of serialized tool calls
    """
    return [serialize_tool_call(tc) for tc in tool_calls]


def deserialize_tool_calls_batch(
    serialized: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deserialize a batch of tool calls.

    Args:
        serialized: List of serialized tool calls

    Returns:
        List of tool calls
    """
    return [deserialize_tool_call(tc) for tc in serialized]


def serialize_ai_message_with_tools(message: AIMessage) -> Dict[str, Any]:
    """Serialize an AIMessage that may contain tool calls.

    Args:
        message: AIMessage to serialize

    Returns:
        Serialized message data
    """
    serialized = {
        "content": message.content,
        "name": getattr(message, "name", None),
        "id": getattr(message, "id", str(uuid.uuid4())),
        "timestamp": datetime.now().isoformat(),
        "additional_kwargs": message.additional_kwargs.copy(),
    }

    # Handle tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        serialized["tool_calls"] = serialize_tool_calls_batch(message.tool_calls)

    return serialized


def deserialize_ai_message_with_tools(data: Dict[str, Any]) -> AIMessage:
    """Deserialize data into an AIMessage with tool calls.

    Args:
        data: Serialized message data

    Returns:
        AIMessage instance
    """
    # Extract tool calls if present
    tool_calls = None
    if "tool_calls" in data and data["tool_calls"]:
        tool_calls = deserialize_tool_calls_batch(data["tool_calls"])

    # Create message
    message = AIMessage(
        content=data.get("content", ""),
        name=data.get("name"),
        tool_calls=tool_calls,
        additional_kwargs=data.get("additional_kwargs", {}),
    )

    # Add ID if available
    if "id" in data:
        message.id = data["id"]

    return message


def serialize_tool_message(message: ToolMessage) -> Dict[str, Any]:
    """Serialize a ToolMessage.

    Args:
        message: ToolMessage to serialize

    Returns:
        Serialized message data
    """
    serialized = {
        "content": message.content,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "status": getattr(message, "status", "success"),
        "timestamp": datetime.now().isoformat(),
        "additional_kwargs": message.additional_kwargs.copy(),
    }

    return serialized


def deserialize_tool_message(data: Dict[str, Any]) -> ToolMessage:
    """Deserialize data into a ToolMessage.

    Args:
        data: Serialized message data

    Returns:
        ToolMessage instance
    """
    message = ToolMessage(
        content=data["content"],
        name=data.get("name", "unknown"),
        tool_call_id=data["tool_call_id"],
        additional_kwargs=data.get("additional_kwargs", {}),
    )

    # Add status if available
    if "status" in data:
        message.status = data["status"]

    return message


def extract_tool_calls_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all tool calls from a state's messages.

    Args:
        state: State dictionary with messages

    Returns:
        List of all tool calls found
    """
    tool_calls = []
    messages = state.get("messages", [])

    for message in messages:
        if isinstance(message, AIMessage) and hasattr(message, "tool_calls"):
            if message.tool_calls:
                tool_calls.extend(message.tool_calls)

    return tool_calls


def extract_tool_results_from_state(state: Dict[str, Any]) -> List[ToolMessage]:
    """Extract all tool results from a state's messages.

    Args:
        state: State dictionary with messages

    Returns:
        List of all ToolMessage instances
    """
    tool_results = []
    messages = state.get("messages", [])

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results.append(message)

    return tool_results


def match_tool_calls_with_results(
    tool_calls: List[Dict[str, Any]], tool_results: List[ToolMessage]
) -> Dict[str, ToolMessage]:
    """Match tool calls with their results.

    Args:
        tool_calls: List of tool calls
        tool_results: List of tool results

    Returns:
        Dictionary mapping tool_call_id to ToolMessage
    """
    results_map = {}

    for result in tool_results:
        if hasattr(result, "tool_call_id"):
            results_map[result.tool_call_id] = result

    return results_map


def create_tool_call_summary(
    tool_calls: List[Dict[str, Any]], tool_results: Optional[List[ToolMessage]] = None
) -> str:
    """Create a human-readable summary of tool calls and results.

    Args:
        tool_calls: List of tool calls
        tool_results: Optional list of tool results

    Returns:
        Summary string
    """
    if not tool_calls:
        return "No tool calls made."

    summary_parts = [f"Made {len(tool_calls)} tool call(s):"]

    # Create results map if results provided
    results_map = {}
    if tool_results:
        results_map = match_tool_calls_with_results(tool_calls, tool_results)

    for i, call in enumerate(tool_calls, 1):
        call_id = call.get("id", "unknown")
        tool_name = call.get("name", "unknown")
        args = call.get("args", {})

        summary_parts.append(f"\n{i}. {tool_name}")

        # Add arguments
        if args:
            args_str = json.dumps(args, indent=2)
            summary_parts.append(f"   Args: {args_str}")

        # Add result if available
        if call_id in results_map:
            result = results_map[call_id]
            status = getattr(result, "status", "success")
            summary_parts.append(f"   Result ({status}): {result.content[:100]}...")

    return "\n".join(summary_parts)


def _serialize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize tool arguments, handling special types.

    Args:
        args: Tool arguments

    Returns:
        Serialized arguments
    """
    serialized = {}

    for key, value in args.items():
        if isinstance(value, datetime):
            serialized[key] = {"_type": "datetime", "value": value.isoformat()}
        elif isinstance(value, (list, tuple)):
            serialized[key] = {"_type": "list", "value": list(value)}
        elif isinstance(value, dict):
            serialized[key] = {"_type": "dict", "value": _serialize_args(value)}
        else:
            # Store as-is for JSON-compatible types
            serialized[key] = value

    return serialized


def _deserialize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize tool arguments, restoring special types.

    Args:
        args: Serialized arguments

    Returns:
        Deserialized arguments
    """
    deserialized = {}

    for key, value in args.items():
        if isinstance(value, dict) and "_type" in value:
            type_name = value["_type"]
            if type_name == "datetime":
                deserialized[key] = datetime.fromisoformat(value["value"])
            elif type_name == "list":
                deserialized[key] = value["value"]
            elif type_name == "dict":
                deserialized[key] = _deserialize_args(value["value"])
            else:
                deserialized[key] = value
        else:
            deserialized[key] = value

    return deserialized
