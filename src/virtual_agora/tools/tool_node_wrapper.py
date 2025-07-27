"""ToolNode wrapper for Virtual Agora.

This module provides a wrapper around LangGraph's ToolNode to add
Virtual Agora-specific functionality and error handling.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ValidationError,
    ProviderError,
)
from virtual_agora.state.schema import MessagesState, Message


logger = get_logger(__name__)


class VirtualAgoraToolNode:
    """Enhanced ToolNode wrapper for Virtual Agora.

    This class wraps LangGraph's ToolNode to provide:
    - Virtual Agora-specific error handling
    - Tool call validation
    - State integration
    - Batch operation support
    - Metrics tracking
    """

    def __init__(
        self,
        tools: Sequence[BaseTool],
        handle_tool_errors: Union[bool, str] = True,
        validate_inputs: bool = True,
        track_metrics: bool = True,
    ):
        """Initialize the Virtual Agora ToolNode.

        Args:
            tools: List of tools available to the node
            handle_tool_errors: Whether to handle tool errors (or custom message)
            validate_inputs: Whether to validate tool inputs
            track_metrics: Whether to track tool usage metrics
        """
        self.tools = list(tools)
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.validate_inputs = validate_inputs
        self.track_metrics = track_metrics

        # Create underlying ToolNode
        self._tool_node = ToolNode(
            tools=self.tools, handle_tool_errors=handle_tool_errors
        )

        # Metrics tracking
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "calls_by_tool": {tool.name: 0 for tool in self.tools},
            "errors_by_type": {},
            "total_execution_time": 0.0,
        }

        logger.info(f"Initialized VirtualAgoraToolNode with {len(self.tools)} tools")

    def __call__(
        self,
        state: Union[MessagesState, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute tool calls from the last message in state.

        Args:
            state: Current state with messages
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            State updates with tool results
        """
        start_time = datetime.now()

        try:
            # Extract messages from state
            messages = state.get("messages", [])
            if not messages:
                logger.warning("No messages in state for tool execution")
                return {"messages": []}

            # Get last message
            last_message = messages[-1]
            if not isinstance(last_message, AIMessage) or not hasattr(
                last_message, "tool_calls"
            ):
                logger.debug("Last message has no tool calls")
                return {"messages": []}

            # Validate tool calls if enabled
            if self.validate_inputs:
                self._validate_tool_calls(last_message.tool_calls)

            # Track metrics
            if self.track_metrics:
                self.metrics["total_calls"] += 1
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    if tool_name in self.metrics["calls_by_tool"]:
                        self.metrics["calls_by_tool"][tool_name] += 1

            # Execute tools through underlying ToolNode
            result = self._tool_node.invoke(state, config)

            # Process results
            tool_messages = result.get("messages", [])

            # Track successful execution
            if self.track_metrics:
                self.metrics["successful_calls"] += 1
                execution_time = (datetime.now() - start_time).total_seconds()
                self.metrics["total_execution_time"] += execution_time

            # Add metadata to tool messages
            enhanced_messages = []
            for msg in tool_messages:
                if isinstance(msg, ToolMessage):
                    # Add Virtual Agora metadata
                    msg.additional_kwargs.update(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "execution_time": (
                                datetime.now() - start_time
                            ).total_seconds(),
                        }
                    )
                enhanced_messages.append(msg)

            logger.info(
                f"Executed {len(last_message.tool_calls)} tool calls successfully"
            )

            return {"messages": enhanced_messages}

        except Exception as e:
            # Track failure
            if self.track_metrics:
                self.metrics["failed_calls"] += 1
                error_type = type(e).__name__
                self.metrics["errors_by_type"][error_type] = (
                    self.metrics["errors_by_type"].get(error_type, 0) + 1
                )

            logger.error(f"Error executing tools: {e}")

            # Re-raise for proper error handling
            raise

    async def __acall__(
        self,
        state: Union[MessagesState, Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async version of tool execution.

        Args:
            state: Current state with messages
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            State updates with tool results
        """
        # For now, delegate to sync version
        # In future, could implement true async tool execution
        return self.__call__(state, config, **kwargs)

    def _validate_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Validate tool calls before execution.

        Args:
            tool_calls: List of tool call dictionaries

        Raises:
            ValidationError: If validation fails
        """
        for tool_call in tool_calls:
            # Check required fields
            if "name" not in tool_call:
                raise ValidationError("Tool call missing 'name' field")

            if "args" not in tool_call:
                raise ValidationError(
                    f"Tool call '{tool_call['name']}' missing 'args' field"
                )

            # Check if tool exists
            tool_name = tool_call["name"]
            if tool_name not in self.tool_map:
                available_tools = ", ".join(self.tool_map.keys())
                raise ValidationError(
                    f"Unknown tool '{tool_name}'. Available tools: {available_tools}"
                )

            # Validate args against tool schema if available
            tool = self.tool_map[tool_name]
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    # This will validate the args against the Pydantic schema
                    tool.args_schema(**tool_call["args"])
                except Exception as e:
                    raise ValidationError(
                        f"Invalid arguments for tool '{tool_name}': {str(e)}"
                    )

    def batch_execute(
        self,
        tool_calls_batch: List[List[Dict[str, Any]]],
        config: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
    ) -> List[List[ToolMessage]]:
        """Execute multiple batches of tool calls.

        Args:
            tool_calls_batch: List of tool call lists
            config: Optional configuration
            parallel: Whether to execute batches in parallel

        Returns:
            List of tool message lists
        """
        results = []

        for i, tool_calls in enumerate(tool_calls_batch):
            try:
                # Create temporary state with tool calls
                temp_message = AIMessage(content="", tool_calls=tool_calls)
                temp_state = {"messages": [temp_message]}

                # Execute through tool node
                result = self(temp_state, config)

                # Extract tool messages
                tool_messages = result.get("messages", [])
                results.append(tool_messages)

            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                # Create error messages for this batch
                error_messages = []
                for tool_call in tool_calls:
                    error_msg = ToolMessage(
                        content=f"Error: {str(e)}",
                        tool_call_id=tool_call.get("id", str(uuid.uuid4())),
                        name=tool_call.get("name", "unknown"),
                        status="error",
                    )
                    error_messages.append(error_msg)
                results.append(error_messages)

        return results

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get information about available tools.

        Returns:
            List of tool information dictionaries
        """
        tool_info = []

        for tool in self.tools:
            info = {
                "name": tool.name,
                "description": tool.description,
                "args_schema": None,
                "return_direct": getattr(tool, "return_direct", False),
            }

            # Add schema info if available
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    schema = tool.args_schema.schema()
                    info["args_schema"] = schema
                except Exception:
                    pass

            tool_info.append(info)

        return tool_info

    def get_metrics(self) -> Dict[str, Any]:
        """Get tool usage metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()

        # Calculate success rate
        if metrics["total_calls"] > 0:
            metrics["success_rate"] = (
                metrics["successful_calls"] / metrics["total_calls"]
            )
            metrics["average_execution_time"] = (
                metrics["total_execution_time"] / metrics["total_calls"]
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["average_execution_time"] = 0.0

        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "calls_by_tool": {tool.name: 0 for tool in self.tools},
            "errors_by_type": {},
            "total_execution_time": 0.0,
        }
        logger.info("Reset tool node metrics")

    def add_tool(self, tool: BaseTool) -> None:
        """Add a new tool to the node.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
        self.metrics["calls_by_tool"][tool.name] = 0

        # Recreate underlying ToolNode
        self._tool_node = ToolNode(
            tools=self.tools, handle_tool_errors=self._tool_node.handle_tool_errors
        )

        logger.info(f"Added tool '{tool.name}' to ToolNode")

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the node.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if tool_name not in self.tool_map:
            return False

        # Remove from collections
        tool = self.tool_map.pop(tool_name)
        self.tools.remove(tool)
        self.metrics["calls_by_tool"].pop(tool_name, None)

        # Recreate underlying ToolNode
        self._tool_node = ToolNode(
            tools=self.tools, handle_tool_errors=self._tool_node.handle_tool_errors
        )

        logger.info(f"Removed tool '{tool_name}' from ToolNode")
        return True


def create_tool_node_with_fallback(
    primary_tools: Sequence[BaseTool],
    fallback_tools: Optional[Sequence[BaseTool]] = None,
    error_message: str = "Tool execution failed. Please try again.",
) -> VirtualAgoraToolNode:
    """Create a tool node with fallback behavior.

    Args:
        primary_tools: Primary tools to use
        fallback_tools: Optional fallback tools
        error_message: Custom error message

    Returns:
        Configured VirtualAgoraToolNode
    """
    all_tools = list(primary_tools)
    if fallback_tools:
        all_tools.extend(fallback_tools)

    return VirtualAgoraToolNode(
        tools=all_tools,
        handle_tool_errors=error_message,
        validate_inputs=True,
        track_metrics=True,
    )
