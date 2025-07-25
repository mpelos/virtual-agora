"""Error context management for Virtual Agora.

This module provides comprehensive error handling with context capture,
recovery strategies, and user-friendly error reporting. Enhanced with
LangGraph integration for improved error resilience in workflows.
"""

import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ProviderError,
    StateError,
    ValidationError,
    ConfigurationError,
    TimeoutError,
    NetworkTransientError,
    RecoverableError,
)

# Import LangGraph components if available
try:
    from langgraph.pregel import RetryPolicy
    from langgraph.checkpoint import Checkpoint
    from virtual_agora.utils.langgraph_error_handler import (
        LangGraphErrorHandler,
        with_langgraph_error_handling,
    )
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    RetryPolicy = None
    Checkpoint = None
    LangGraphErrorHandler = None
    with_langgraph_error_handling = None


logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    LOW = "low"  # Informational, no action needed
    MEDIUM = "medium"  # Warning, may need attention
    HIGH = "high"  # Error that can be recovered
    CRITICAL = "critical"  # Fatal error, cannot continue


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    
    NONE = "none"  # No recovery possible
    RETRY = "retry"  # Retry the operation
    SKIP = "skip"  # Skip the operation
    ROLLBACK = "rollback"  # Rollback to previous state
    FALLBACK = "fallback"  # Use fallback behavior
    USER_INTERVENTION = "user_intervention"  # Ask user what to do
    GRACEFUL_SHUTDOWN = "graceful_shutdown"  # Shutdown cleanly


@dataclass
class ErrorContext:
    """Captures comprehensive error context with LangGraph integration."""
    
    error: Exception
    timestamp: datetime = field(default_factory=datetime.now)
    operation: Optional[str] = None
    phase: Optional[str] = None
    state_snapshot: Optional[Dict[str, Any]] = None
    breadcrumbs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.HIGH
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NONE
    retry_count: int = 0
    max_retries: int = 3
    
    # LangGraph specific fields
    graph_node: Optional[str] = None  # Current graph node where error occurred
    graph_state: Optional[Dict[str, Any]] = None  # LangGraph state at error time
    checkpoint_id: Optional[str] = None  # Checkpoint ID for recovery
    parent_errors: List['ErrorContext'] = field(default_factory=list)  # Chain of errors
    
    def add_breadcrumb(self, message: str) -> None:
        """Add a breadcrumb to track execution path.
        
        Args:
            message: Breadcrumb message
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.breadcrumbs.append(f"[{timestamp}] {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging.
        
        Returns:
            Dictionary representation
        """
        result = {
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "phase": self.phase,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value,
            "retry_count": self.retry_count,
            "breadcrumbs": self.breadcrumbs[-10:],  # Last 10 breadcrumbs
            "metadata": self.metadata,
        }
        
        # Add LangGraph fields if available
        if self.graph_node:
            result["graph_node"] = self.graph_node
        if self.graph_state:
            result["graph_state_keys"] = list(self.graph_state.keys())
        if self.checkpoint_id:
            result["checkpoint_id"] = self.checkpoint_id
        if self.parent_errors:
            result["parent_errors"] = [
                {"type": type(e.error).__name__, "message": str(e.error)}
                for e in self.parent_errors
            ]
        
        return result


class ErrorHandler:
    """Global error handler with context management."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_callbacks: Dict[Type[Exception], Callable] = {}
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self) -> None:
        """Set up default recovery strategies for known error types."""
        self.recovery_strategies = {
            # Provider errors can often be retried
            ProviderError: RecoveryStrategy.RETRY,
            TimeoutError: RecoveryStrategy.RETRY,
            
            # State errors need rollback
            StateError: RecoveryStrategy.ROLLBACK,
            
            # Validation errors need user intervention
            ValidationError: RecoveryStrategy.USER_INTERVENTION,
            
            # Configuration errors are critical
            ConfigurationError: RecoveryStrategy.GRACEFUL_SHUTDOWN,
        }
    
    def capture_context(
        self,
        error: Exception,
        operation: Optional[str] = None,
        phase: Optional[str] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
        **metadata
    ) -> ErrorContext:
        """Capture comprehensive error context.
        
        Args:
            error: The exception that occurred
            operation: Current operation being performed
            phase: Current application phase
            state_snapshot: Snapshot of application state
            **metadata: Additional metadata
            
        Returns:
            Captured error context
        """
        # Determine severity
        severity = self._determine_severity(error)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error)
        
        # Create context
        context = ErrorContext(
            error=error,
            operation=operation,
            phase=phase,
            state_snapshot=state_snapshot,
            metadata=metadata,
            severity=severity,
            recovery_strategy=recovery_strategy,
        )
        
        # Add to history
        self.error_history.append(context)
        
        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error
        logger.error(
            f"Error captured: {error_type} during {operation or 'unknown operation'}",
            extra={"error_context": context.to_dict()}
        )
        
        return context
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type.
        
        Args:
            error: The exception
            
        Returns:
            Error severity level
        """
        if isinstance(error, ConfigurationError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, StateError):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ProviderError, TimeoutError)):
            # Check if it's a rate limit error
            if isinstance(error, ProviderError) and "rate" in str(error).lower():
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.HIGH
        elif isinstance(error, ValidationError):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.HIGH
    
    def _determine_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Determine recovery strategy based on error type.
        
        Args:
            error: The exception
            
        Returns:
            Recovery strategy
        """
        error_type = type(error)
        
        # Check exact type match
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # Check inheritance
        for exc_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exc_type):
                return strategy
        
        # Default strategy
        return RecoveryStrategy.USER_INTERVENTION
    
    def format_user_message(self, context: ErrorContext) -> str:
        """Format a user-friendly error message.
        
        Args:
            context: Error context
            
        Returns:
            Formatted error message
        """
        error_type = type(context.error).__name__.replace("Error", "")
        
        # Base message
        if isinstance(context.error, ProviderError):
            provider = context.error.provider or "Unknown"
            message = f"There was an issue with the {provider} service"
        elif isinstance(context.error, TimeoutError):
            message = "The operation took too long to complete"
        elif isinstance(context.error, StateError):
            message = "There was an issue with the application state"
        elif isinstance(context.error, ValidationError):
            message = "The provided input was invalid"
        elif isinstance(context.error, ConfigurationError):
            message = "There is a configuration problem"
        else:
            message = f"An unexpected {error_type} occurred"
        
        # Add context
        if context.operation:
            message += f" while {context.operation}"
        
        # Add details
        error_details = str(context.error)
        if error_details:
            message += f": {error_details}"
        
        # Add recovery suggestion
        suggestions = {
            RecoveryStrategy.RETRY: "The operation will be retried automatically.",
            RecoveryStrategy.SKIP: "This step will be skipped.",
            RecoveryStrategy.ROLLBACK: "Rolling back to the previous state.",
            RecoveryStrategy.USER_INTERVENTION: "Please check your input and try again.",
            RecoveryStrategy.GRACEFUL_SHUTDOWN: "The application needs to restart.",
        }
        
        if context.recovery_strategy in suggestions:
            message += f"\n\n{suggestions[context.recovery_strategy]}"
        
        return message
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors in the session.
        
        Returns:
            Error summary statistics
        """
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "most_common": None,
                "severity_breakdown": {},
            }
        
        # Severity breakdown
        severity_counts = {}
        for context in self.error_history:
            severity = context.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Most common error
        most_common = max(self.error_counts.items(), key=lambda x: x[1])
        
        # Calculate error rate (errors per minute)
        if self.error_history:
            time_span = (
                self.error_history[-1].timestamp - self.error_history[0].timestamp
            ).total_seconds() / 60.0
            error_rate = total_errors / max(time_span, 1.0)
        else:
            error_rate = 0.0
        
        return {
            "total_errors": total_errors,
            "error_rate": round(error_rate, 2),
            "most_common": {
                "type": most_common[0],
                "count": most_common[1],
                "percentage": round(most_common[1] / total_errors * 100, 1),
            },
            "severity_breakdown": severity_counts,
            "error_types": dict(self.error_counts),
        }
    
    def should_circuit_break(self, error_type: Type[Exception], threshold: int = 5) -> bool:
        """Check if we should circuit break for a specific error type.
        
        Args:
            error_type: Type of exception to check
            threshold: Number of errors before circuit breaking
            
        Returns:
            True if circuit should break
        """
        error_count = self.error_counts.get(error_type.__name__, 0)
        return error_count >= threshold
    
    @contextmanager
    def error_boundary(
        self,
        operation: str,
        phase: Optional[str] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
    ):
        """Context manager for error boundaries.
        
        Args:
            operation: Operation being performed
            phase: Current phase
            state_snapshot: State snapshot
            reraise: Whether to reraise the exception
            
        Yields:
            Error context if an error occurs
        """
        context = None
        try:
            yield context
        except Exception as e:
            context = self.capture_context(
                error=e,
                operation=operation,
                phase=phase,
                state_snapshot=state_snapshot,
            )
            
            if reraise:
                raise
    
    def register_recovery_callback(
        self,
        error_type: Type[Exception],
        callback: Callable[[ErrorContext], None]
    ) -> None:
        """Register a recovery callback for a specific error type.
        
        Args:
            error_type: Type of exception
            callback: Recovery callback function
        """
        self.recovery_callbacks[error_type] = callback
    
    def attempt_recovery(self, context: ErrorContext) -> bool:
        """Attempt to recover from an error.
        
        Args:
            context: Error context
            
        Returns:
            True if recovery was successful
        """
        error_type = type(context.error)
        
        # Check for registered callback
        if error_type in self.recovery_callbacks:
            try:
                self.recovery_callbacks[error_type](context)
                logger.info(f"Successfully recovered from {error_type.__name__}")
                return True
            except Exception as e:
                logger.error(f"Recovery failed: {e}")
                return False
        
        # Default recovery based on strategy
        if context.recovery_strategy == RecoveryStrategy.SKIP:
            logger.info(f"Skipping operation due to {error_type.__name__}")
            return True
        
        return False


    def create_langgraph_retry_policy(
        self,
        error_types: Optional[List[Type[Exception]]] = None,
        max_attempts: int = 3,
    ) -> Optional['RetryPolicy']:
        """Create a LangGraph RetryPolicy based on error handler configuration.
        
        Args:
            error_types: Error types to retry on (uses defaults if None)
            max_attempts: Maximum retry attempts
            
        Returns:
            LangGraph RetryPolicy or None if not available
        """
        if not LANGGRAPH_AVAILABLE or RetryPolicy is None:
            logger.debug("LangGraph RetryPolicy not available")
            return None
        
        # Default retryable errors
        if error_types is None:
            error_types = [
                ProviderError,
                TimeoutError,
                NetworkTransientError,
                RecoverableError,
            ]
        
        def should_retry(error: Exception) -> bool:
            """Determine if error should be retried."""
            # Check if error type is retryable
            for error_type in error_types:
                if isinstance(error, error_type):
                    # Check if we should circuit break
                    if self.should_circuit_break(type(error)):
                        logger.warning(f"Circuit breaker open for {type(error).__name__}")
                        return False
                    return True
            return False
        
        return RetryPolicy(
            retry_on=should_retry,
            max_attempts=max_attempts,
        )
    
    def capture_langgraph_context(
        self,
        error: Exception,
        graph_node: str,
        graph_state: Dict[str, Any],
        checkpoint_id: Optional[str] = None,
        **kwargs
    ) -> ErrorContext:
        """Capture error context with LangGraph-specific information.
        
        Args:
            error: The exception that occurred
            graph_node: Current graph node name
            graph_state: Current graph state
            checkpoint_id: Optional checkpoint ID
            **kwargs: Additional context parameters
            
        Returns:
            Captured error context
        """
        # Extract operation and phase from kwargs or graph info
        operation = kwargs.get("operation", f"graph_node_{graph_node}")
        phase = kwargs.get("phase", "graph_execution")
        
        # Create base context
        context = self.capture_context(
            error=error,
            operation=operation,
            phase=phase,
            state_snapshot=graph_state,
            **kwargs
        )
        
        # Add LangGraph-specific fields
        context.graph_node = graph_node
        context.graph_state = graph_state
        context.checkpoint_id = checkpoint_id
        
        # Add breadcrumb
        context.add_breadcrumb(f"Error in graph node: {graph_node}")
        
        return context
    
    def create_error_recovery_node(
        self,
        recovery_strategies: Optional[Dict[str, RecoveryStrategy]] = None
    ) -> Callable:
        """Create a LangGraph node for error recovery.
        
        This node can be added to a StateGraph to handle errors from other nodes.
        
        Args:
            recovery_strategies: Custom recovery strategies by node name
            
        Returns:
            Callable node function for error recovery
        """
        def error_recovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """LangGraph node for error recovery."""
            # Check if there's an error in state
            error_info = state.get("error")
            if not error_info:
                return state
            
            # Extract error details
            error = error_info.get("error")
            node_name = error_info.get("node", "unknown")
            
            # Capture context
            context = self.capture_langgraph_context(
                error=error,
                graph_node=node_name,
                graph_state=state,
            )
            
            # Determine recovery strategy
            if recovery_strategies and node_name in recovery_strategies:
                context.recovery_strategy = recovery_strategies[node_name]
            
            # Attempt recovery
            recovery_successful = self.attempt_recovery(context)
            
            # Update state based on recovery
            if recovery_successful:
                logger.info(f"Successfully recovered from error in node {node_name}")
                state["error_recovered"] = True
                state.pop("error", None)  # Remove error from state
            else:
                logger.error(f"Failed to recover from error in node {node_name}")
                state["error_recovered"] = False
                state["error_context"] = context.to_dict()
            
            return state
        
        return error_recovery_node
    
    @contextmanager
    def langgraph_error_boundary(
        self,
        node_name: str,
        state: Dict[str, Any],
        checkpoint_id: Optional[str] = None,
        reraise: bool = True,
    ):
        """Context manager for LangGraph node error boundaries.
        
        Args:
            node_name: Name of the graph node
            state: Current graph state
            checkpoint_id: Optional checkpoint ID
            reraise: Whether to reraise the exception
            
        Yields:
            Error context if an error occurs
        """
        context = None
        try:
            yield context
        except Exception as e:
            context = self.capture_langgraph_context(
                error=e,
                graph_node=node_name,
                graph_state=state,
                checkpoint_id=checkpoint_id,
            )
            
            # Add error to state for downstream handling
            state["error"] = {
                "error": e,
                "node": node_name,
                "context": context.to_dict(),
            }
            
            if reraise:
                raise
    
    def get_node_error_summary(self, node_name: str) -> Dict[str, Any]:
        """Get error summary for a specific graph node.
        
        Args:
            node_name: Name of the graph node
            
        Returns:
            Error summary for the node
        """
        node_errors = [
            ctx for ctx in self.error_history
            if ctx.graph_node == node_name
        ]
        
        if not node_errors:
            return {
                "node": node_name,
                "total_errors": 0,
                "error_rate": 0.0,
                "error_types": {},
            }
        
        # Count error types
        error_types = {}
        for ctx in node_errors:
            error_type = type(ctx.error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Calculate error rate
        time_span = (
            node_errors[-1].timestamp - node_errors[0].timestamp
        ).total_seconds() / 60.0
        error_rate = len(node_errors) / max(time_span, 1.0)
        
        return {
            "node": node_name,
            "total_errors": len(node_errors),
            "error_rate": round(error_rate, 2),
            "error_types": error_types,
            "last_error": node_errors[-1].timestamp.isoformat(),
        }


# Global error handler instance
error_handler = ErrorHandler()