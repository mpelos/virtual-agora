"""Custom exceptions for Virtual Agora.

This module defines application-specific exceptions for better
error handling and debugging.
"""

from typing import Optional, Any


class VirtualAgoraError(Exception):
    """Base exception for all Virtual Agora errors.
    
    All custom exceptions in the application should inherit from this class
    to allow for easy catching of application-specific errors.
    """
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(VirtualAgoraError):
    """Raised when there's an error in configuration.
    
    This includes missing configuration files, invalid YAML syntax,
    missing required fields, or invalid field values.
    """
    pass


class ProviderError(VirtualAgoraError):
    """Raised when there's an error with an LLM provider.
    
    This includes API errors, authentication failures, rate limiting,
    or unsupported operations.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize provider error.
        
        Args:
            message: Error message.
            provider: Name of the provider that caused the error.
            error_code: Provider-specific error code.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.provider = provider
        self.error_code = error_code


class AgentError(VirtualAgoraError):
    """Raised when there's an error with an agent.
    
    This includes agent initialization failures, response generation errors,
    or invalid agent states.
    """
    
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize agent error.
        
        Args:
            message: Error message.
            agent_name: Name of the agent that caused the error.
            agent_type: Type of agent (moderator, discussion).
            details: Additional error details.
        """
        super().__init__(message, details)
        self.agent_name = agent_name
        self.agent_type = agent_type


class WorkflowError(VirtualAgoraError):
    """Raised when there's an error in the discussion workflow.
    
    This includes state transition errors, phase execution failures,
    or workflow validation errors.
    """
    
    def __init__(
        self,
        message: str,
        phase: Optional[str] = None,
        state: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize workflow error.
        
        Args:
            message: Error message.
            phase: Current workflow phase when error occurred.
            state: Current application state.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.phase = phase
        self.state = state


class ValidationError(VirtualAgoraError):
    """Raised when validation fails.
    
    This includes input validation, response validation,
    or state validation errors.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize validation error.
        
        Args:
            message: Error message.
            field: Field that failed validation.
            value: Value that failed validation.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.field = field
        self.value = value


class TimeoutError(VirtualAgoraError):
    """Raised when an operation times out.
    
    This includes agent response timeouts, API call timeouts,
    or user input timeouts.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message.
            operation: Operation that timed out.
            timeout_seconds: Timeout duration in seconds.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds