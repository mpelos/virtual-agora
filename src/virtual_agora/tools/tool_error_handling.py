"""Error handling for tool operations in Virtual Agora.

This module provides error handling utilities and strategies for
tool execution within LangGraph workflows.
"""

from typing import Dict, Any, Optional, Callable, Type, List, Union
from functools import wraps
import traceback
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from pydantic import ValidationError as PydanticValidationError

from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ValidationError,
    ProviderError,
    TimeoutError,
    NetworkTransientError
)
from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


class ToolExecutionError(VirtualAgoraError):
    """Base error for tool execution failures."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
        retry_after: Optional[int] = None
    ):
        """Initialize tool execution error.
        
        Args:
            message: Error message
            tool_name: Name of the tool that failed
            tool_call_id: ID of the tool call
            original_error: Original exception if any
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.original_error = original_error
        self.retry_after = retry_after


class ToolValidationError(ToolExecutionError):
    """Error for tool input validation failures."""
    pass


class ToolTimeoutError(ToolExecutionError):
    """Error for tool execution timeouts."""
    pass


class ToolRetryableError(ToolExecutionError):
    """Error that indicates the tool call can be retried."""
    pass


def create_error_tool_message(
    tool_call_id: str,
    tool_name: str,
    error: Exception,
    include_traceback: bool = False
) -> ToolMessage:
    """Create a ToolMessage for an error.
    
    Args:
        tool_call_id: ID of the failed tool call
        tool_name: Name of the tool
        error: The exception that occurred
        include_traceback: Whether to include full traceback
        
    Returns:
        ToolMessage with error information
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Build error content
    content_parts = [
        f"Tool execution failed: {error_type}",
        f"Message: {error_message}"
    ]
    
    if include_traceback and hasattr(error, "__traceback__"):
        tb_str = "".join(traceback.format_tb(error.__traceback__))
        content_parts.append(f"Traceback:\n{tb_str}")
    
    # Add retry information if available
    if isinstance(error, ToolRetryableError) and error.retry_after:
        content_parts.append(f"Retry after: {error.retry_after} seconds")
    
    content = "\n".join(content_parts)
    
    return ToolMessage(
        content=content,
        name=tool_name,
        tool_call_id=tool_call_id,
        status="error",
        additional_kwargs={
            "error_type": error_type,
            "timestamp": datetime.now().isoformat(),
            "retryable": isinstance(error, (ToolRetryableError, NetworkTransientError))
        }
    )


def handle_tool_error(
    error: Exception,
    tool: BaseTool,
    tool_call_id: str,
    fallback_response: Optional[str] = None,
    max_retries: int = 0,
    current_retry: int = 0
) -> Union[str, ToolMessage]:
    """Handle errors during tool execution.
    
    Args:
        error: The exception that occurred
        tool: The tool that failed
        tool_call_id: ID of the tool call
        fallback_response: Optional fallback response
        max_retries: Maximum retry attempts
        current_retry: Current retry attempt number
        
    Returns:
        Either a fallback string or an error ToolMessage
    """
    logger.error(
        f"Tool '{tool.name}' failed (attempt {current_retry + 1}/{max_retries + 1}): {error}",
        extra={
            "tool_name": tool.name,
            "tool_call_id": tool_call_id,
            "error_type": type(error).__name__,
            "retry_attempt": current_retry
        }
    )
    
    # Check if error is retryable
    retryable_errors = (
        NetworkTransientError,
        TimeoutError,
        ToolRetryableError
    )
    
    if isinstance(error, retryable_errors) and current_retry < max_retries:
        # This is a retryable error and we have retries left
        raise ToolRetryableError(
            f"Retryable error in tool '{tool.name}': {error}",
            tool_name=tool.name,
            tool_call_id=tool_call_id,
            original_error=error,
            retry_after=getattr(error, "retry_after", 1)
        )
    
    # Check if we have a fallback response
    if fallback_response:
        logger.info(f"Using fallback response for tool '{tool.name}'")
        return fallback_response
    
    # Create error message
    return create_error_tool_message(
        tool_call_id=tool_call_id,
        tool_name=tool.name,
        error=error,
        include_traceback=logger.isEnabledFor(10)  # DEBUG level
    )


def validate_tool_input(
    tool: BaseTool,
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate tool input arguments.
    
    Args:
        tool: The tool to validate for
        args: Arguments to validate
        
    Returns:
        Validated arguments
        
    Raises:
        ToolValidationError: If validation fails
    """
    try:
        # Use tool's args_schema if available
        if hasattr(tool, "args_schema") and tool.args_schema:
            # This will raise ValidationError if invalid
            validated = tool.args_schema(**args)
            # Convert back to dict
            return validated.dict() if hasattr(validated, "dict") else dict(validated)
        
        # No schema, return as-is
        return args
        
    except PydanticValidationError as e:
        raise ToolValidationError(
            f"Invalid arguments for tool '{tool.name}': {e}",
            tool_name=tool.name,
            original_error=e
        )
    except Exception as e:
        raise ToolValidationError(
            f"Failed to validate arguments for tool '{tool.name}': {e}",
            tool_name=tool.name,
            original_error=e
        )


def with_tool_error_handling(
    fallback_response: Optional[str] = None,
    max_retries: int = 3,
    include_traceback: bool = False,
    error_types_to_catch: Optional[List[Type[Exception]]] = None
):
    """Decorator for tool functions to add error handling.
    
    Args:
        fallback_response: Response to use if tool fails
        max_retries: Maximum retry attempts
        include_traceback: Whether to include traceback in errors
        error_types_to_catch: Specific error types to catch
        
    Returns:
        Decorated function
    """
    if error_types_to_catch is None:
        error_types_to_catch = [Exception]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except tuple(error_types_to_catch) as e:
                    last_error = e
                    
                    # Check if this is a retryable error
                    if isinstance(e, (NetworkTransientError, TimeoutError, ToolRetryableError)):
                        if attempt < max_retries:
                            wait_time = getattr(e, "retry_after", 2 ** attempt)
                            logger.warning(
                                f"Retryable error in {func.__name__}, "
                                f"waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                            )
                            # In a real implementation, we'd wait here
                            continue
                    
                    # Non-retryable error or out of retries
                    break
            
            # All attempts failed
            if fallback_response:
                logger.warning(f"Using fallback response for {func.__name__}")
                return fallback_response
            
            # Re-raise the last error
            if last_error:
                raise last_error
        
        return wrapper
    return decorator


class ToolErrorRecoveryStrategy:
    """Strategy for recovering from tool errors."""
    
    def __init__(
        self,
        max_retries: int = 3,
        exponential_backoff: bool = True,
        fallback_tools: Optional[Dict[str, BaseTool]] = None,
        error_message_template: Optional[str] = None
    ):
        """Initialize recovery strategy.
        
        Args:
            max_retries: Maximum retry attempts
            exponential_backoff: Whether to use exponential backoff
            fallback_tools: Alternative tools to try on failure
            error_message_template: Template for error messages
        """
        self.max_retries = max_retries
        self.exponential_backoff = exponential_backoff
        self.fallback_tools = fallback_tools or {}
        self.error_message_template = error_message_template or (
            "I encountered an error while using the {tool_name} tool: {error_message}. "
            "Let me try a different approach."
        )
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry after an error.
        
        Args:
            error: The error that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False
        
        # List of retryable error types
        retryable = (
            NetworkTransientError,
            TimeoutError,
            ToolRetryableError,
            ProviderError  # Some provider errors are transient
        )
        
        return isinstance(error, retryable)
    
    def get_wait_time(self, attempt: int, error: Exception) -> float:
        """Get wait time before retry.
        
        Args:
            attempt: Current attempt number
            error: The error that occurred
            
        Returns:
            Seconds to wait
        """
        # Check if error specifies retry_after
        if hasattr(error, "retry_after") and error.retry_after:
            return float(error.retry_after)
        
        # Use exponential backoff if enabled
        if self.exponential_backoff:
            return min(2 ** attempt, 60)  # Cap at 60 seconds
        
        return 1.0
    
    def get_fallback_tool(self, original_tool: str) -> Optional[BaseTool]:
        """Get fallback tool for a failed tool.
        
        Args:
            original_tool: Name of the tool that failed
            
        Returns:
            Fallback tool if available
        """
        return self.fallback_tools.get(original_tool)
    
    def format_error_message(self, tool_name: str, error: Exception) -> str:
        """Format user-friendly error message.
        
        Args:
            tool_name: Name of the tool that failed
            error: The error that occurred
            
        Returns:
            Formatted error message
        """
        error_message = str(error)
        
        # Simplify technical errors for users
        if isinstance(error, ValidationError):
            error_message = "The provided arguments were invalid"
        elif isinstance(error, TimeoutError):
            error_message = "The operation took too long to complete"
        elif isinstance(error, NetworkTransientError):
            error_message = "A temporary network issue occurred"
        
        return self.error_message_template.format(
            tool_name=tool_name,
            error_message=error_message
        )