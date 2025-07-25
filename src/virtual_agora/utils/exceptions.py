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


class CoordinationError(VirtualAgoraError):
    """Raised when there's an error in multi-agent coordination.

    This includes turn management errors, agent communication failures,
    or coordination protocol violations.
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        coordination_phase: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize coordination error.

        Args:
            message: Error message.
            agent_id: Agent ID involved in the error.
            coordination_phase: Current coordination phase.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.agent_id = agent_id
        self.coordination_phase = coordination_phase


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


class StateError(VirtualAgoraError):
    """Raised when there's an error with application state.

    This includes state initialization errors, invalid state transitions,
    or state corruption issues.
    """

    def __init__(
        self,
        message: str,
        state_field: Optional[str] = None,
        current_value: Optional[Any] = None,
        expected_value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize state error.

        Args:
            message: Error message.
            state_field: State field that caused the error.
            current_value: Current value of the field.
            expected_value: Expected value of the field.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.state_field = state_field
        self.current_value = current_value
        self.expected_value = expected_value


class RecoverableError(VirtualAgoraError):
    """Base class for errors that can be recovered through retry or other means.

    This indicates an error that is temporary and the operation
    can be retried with a reasonable expectation of success.
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        max_retries: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize recoverable error.

        Args:
            message: Error message.
            retry_after: Suggested delay before retry (seconds).
            max_retries: Maximum number of retries recommended.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.retry_after = retry_after
        self.max_retries = max_retries


class CriticalError(VirtualAgoraError):
    """Raised when a critical error occurs that requires immediate shutdown.

    This indicates an unrecoverable error that compromises the
    integrity or security of the application.
    """

    def __init__(
        self,
        message: str,
        system_component: Optional[str] = None,
        requires_restart: bool = True,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize critical error.

        Args:
            message: Error message.
            system_component: Component that failed critically.
            requires_restart: Whether application restart is required.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.system_component = system_component
        self.requires_restart = requires_restart


class ProviderRateLimitError(ProviderError, RecoverableError):
    """Raised when a provider rate limit is exceeded.

    This is a specific type of provider error that indicates
    the request was rejected due to rate limiting.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: Optional[float] = None,
        limit_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message.
            provider: Provider that returned the rate limit.
            retry_after: Time to wait before retry (from provider).
            limit_type: Type of limit hit (requests, tokens, etc).
            details: Additional error details.
        """
        ProviderError.__init__(self, message, provider, "rate_limit", details)
        RecoverableError.__init__(self, message, retry_after, 3, details)
        self.limit_type = limit_type


class NetworkTransientError(RecoverableError):
    """Raised when a temporary network error occurs.

    This includes connection timeouts, DNS failures, and other
    network-related issues that are likely temporary.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        endpoint: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize network error.

        Args:
            message: Error message.
            operation: Operation that failed.
            endpoint: Network endpoint involved.
            details: Additional error details.
        """
        super().__init__(message, retry_after=1.0, max_retries=5, details=details)
        self.operation = operation
        self.endpoint = endpoint


class StateCorruptionError(StateError, CriticalError):
    """Raised when application state is corrupted beyond repair.

    This indicates that the state has become inconsistent in a way
    that cannot be automatically recovered.
    """

    def __init__(
        self,
        message: str,
        corrupted_fields: Optional[list[str]] = None,
        last_known_good_state: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize state corruption error.

        Args:
            message: Error message.
            corrupted_fields: List of corrupted state fields.
            last_known_good_state: Snapshot of last valid state.
            details: Additional error details.
        """
        StateError.__init__(self, message, details=details)
        CriticalError.__init__(
            self,
            message,
            system_component="state_manager",
            requires_restart=True,
            details=details,
        )
        self.corrupted_fields = corrupted_fields or []
        self.last_known_good_state = last_known_good_state


class UserInterventionRequired(VirtualAgoraError):
    """Raised when user intervention is required to proceed.

    This indicates a situation where the application cannot
    proceed without explicit user input or decision.
    """

    def __init__(
        self,
        message: str,
        prompt: str,
        options: Optional[list[str]] = None,
        default_option: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize user intervention error.

        Args:
            message: Error message.
            prompt: Prompt to show the user.
            options: Available options for the user.
            default_option: Default option if user doesn't respond.
            details: Additional error details.
        """
        super().__init__(message, details)
        self.prompt = prompt
        self.options = options or []
        self.default_option = default_option
