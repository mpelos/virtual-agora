"""Retry mechanism with exponential backoff for Virtual Agora.

This module provides retry functionality with exponential backoff,
jitter, and circuit breaker patterns for handling transient failures.
Enhanced with LangGraph RetryPolicy integration.
"""

import asyncio
import functools
import random
import time
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from enum import Enum

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import (
    ProviderError,
    TimeoutError,
    VirtualAgoraError,
    StateError,
    NetworkTransientError,
    ProviderRateLimitError,
)
from virtual_agora.utils.error_handler import error_handler, ErrorContext

# Import LangGraph RetryPolicy if available
try:
    from langgraph.pregel import RetryPolicy

    LANGGRAPH_AVAILABLE = True
except ImportError:
    RetryPolicy = None
    LANGGRAPH_AVAILABLE = False


logger = get_logger(__name__)


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class RetryState(Enum):
    """State of retry mechanism."""

    IDLE = "idle"
    RETRYING = "retrying"
    SUCCESS = "success"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStatistics:
    """Track retry statistics."""

    def __init__(self):
        """Initialize statistics."""
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_retries = 0
        self.retry_durations: List[float] = []
        self.last_attempt_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.last_failure_time: Optional[datetime] = None

    def record_attempt(self, success: bool, duration: float, retries: int) -> None:
        """Record an attempt.

        Args:
            success: Whether the attempt succeeded
            duration: Total duration including retries
            retries: Number of retries performed
        """
        self.total_attempts += 1
        self.total_retries += retries
        self.retry_durations.append(duration)
        self.last_attempt_time = datetime.now()

        if success:
            self.successful_attempts += 1
            self.last_success_time = datetime.now()
        else:
            self.failed_attempts += 1
            self.last_failure_time = datetime.now()

    def get_success_rate(self) -> float:
        """Get success rate as percentage.

        Returns:
            Success rate (0-100)
        """
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100

    def get_average_retry_duration(self) -> float:
        """Get average retry duration in seconds.

        Returns:
            Average duration
        """
        if not self.retry_durations:
            return 0.0
        return sum(self.retry_durations) / len(self.retry_durations)


class CircuitBreaker:
    """Circuit breaker pattern implementation with LangGraph integration."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: Optional[str] = None,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to count as failure
            name: Optional name for the circuit breaker (useful for tracking)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or f"circuit_{id(self)}"

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

        # Statistics tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opens = 0
        self.last_state_change: Optional[datetime] = None

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        self.total_calls += 1

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN state")
            else:
                raise VirtualAgoraError(
                    f"Circuit breaker {self.name} is open",
                    details={
                        "failure_count": self.failure_count,
                        "recovery_timeout": self.recovery_timeout,
                        "time_until_reset": (
                            (
                                self.recovery_timeout
                                - (
                                    datetime.now() - self.last_failure_time
                                ).total_seconds()
                            )
                            if self.last_failure_time
                            else 0
                        ),
                    },
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    async def call_async(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Call async function through circuit breaker.

        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        self.total_calls += 1

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN state")
            else:
                raise VirtualAgoraError(
                    f"Circuit breaker {self.name} is open",
                    details={
                        "failure_count": self.failure_count,
                        "recovery_timeout": self.recovery_timeout,
                        "time_until_reset": (
                            (
                                self.recovery_timeout
                                - (
                                    datetime.now() - self.last_failure_time
                                ).total_seconds()
                            )
                            if self.last_failure_time
                            else 0
                        ),
                    },
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit.

        Returns:
            True if enough time has passed
        """
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.successful_calls += 1

        # Close circuit if it was half-open
        if self.state == CircuitState.HALF_OPEN:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.last_state_change = datetime.now()
            logger.info(f"Circuit breaker {self.name} closed after successful call")

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.failed_calls += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                old_state = self.state
                self.state = CircuitState.OPEN
                self.circuit_opens += 1
                self.last_state_change = datetime.now()
                logger.warning(
                    f"Circuit breaker {self.name} opened after {self.failure_count} failures"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Statistics dictionary
        """
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = (self.successful_calls / self.total_calls) * 100

        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(success_rate, 2),
            "failure_count": self.failure_count,
            "circuit_opens": self.circuit_opens,
            "last_failure": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": (
                self.last_state_change.isoformat() if self.last_state_change else None
            ),
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        logger.info(f"Circuit breaker {self.name} reset")

    def as_langgraph_policy(self) -> Optional["RetryPolicy"]:
        """Convert circuit breaker to LangGraph RetryPolicy.

        Returns:
            LangGraph RetryPolicy that respects circuit breaker state
        """
        if not LANGGRAPH_AVAILABLE or RetryPolicy is None:
            return None

        def circuit_aware_retry(error: Exception) -> bool:
            """Check if retry is allowed based on circuit state."""
            # Don't retry if circuit is open
            if self.state == CircuitState.OPEN:
                # Check if we should attempt reset
                if not self._should_attempt_reset():
                    logger.debug(f"Circuit breaker {self.name} is open, blocking retry")
                    return False

            # Check if error is expected type
            return isinstance(error, self.expected_exception)

        # Create retry policy with circuit breaker awareness
        return RetryPolicy(
            retry_on=circuit_aware_retry,
            max_attempts=self.failure_threshold,  # Use threshold as max attempts
        )


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # Calculate exponential delay
    delay = min(base_delay * (exponential_base**attempt), max_delay)

    # Add jitter to prevent thundering herd
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


def is_retryable_error(
    error: Exception, retryable_errors: Tuple[Type[Exception], ...]
) -> bool:
    """Check if an error is retryable.

    Args:
        error: The exception
        retryable_errors: Tuple of retryable exception types

    Returns:
        True if error is retryable
    """
    # Check exact type match
    if type(error) in retryable_errors:
        return True

    # Check inheritance
    for error_type in retryable_errors:
        if isinstance(error, error_type):
            return True

    # Check for specific error conditions
    if isinstance(error, ProviderError):
        # Rate limit errors are retryable
        if "rate" in str(error).lower():
            return True
        # Temporary failures are retryable
        if any(code in str(error) for code in ["429", "503", "504"]):
            return True

    return False


def retry_with_backoff(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[F], F]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exceptions to retry on
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter
        on_retry: Callback called on each retry

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            start_time = time.time()

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # Record success
                    if attempt > 0:
                        duration = time.time() - start_time
                        logger.info(
                            f"Succeeded after {attempt + 1} attempts "
                            f"({duration:.2f}s total)"
                        )

                    return result

                except exceptions as e:
                    last_exception = e

                    # Check if retryable
                    if not is_retryable_error(e, exceptions):
                        raise

                    # Check if we have more attempts
                    if attempt >= max_attempts - 1:
                        break

                    # Calculate delay
                    delay = calculate_backoff_delay(
                        attempt,
                        base_delay,
                        max_delay,
                        exponential_base,
                        jitter,
                    )

                    # Log retry
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Call retry callback
                    if on_retry:
                        on_retry(e, attempt + 1)

                    # Wait before retry
                    time.sleep(delay)

            # All attempts failed
            duration = time.time() - start_time
            logger.error(f"All {max_attempts} attempts failed after {duration:.2f}s")

            if last_exception:
                raise last_exception
            else:
                raise VirtualAgoraError("Retry failed with unknown error")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            start_time = time.time()

            for attempt in range(max_attempts):
                try:
                    result = await func(*args, **kwargs)

                    # Record success
                    if attempt > 0:
                        duration = time.time() - start_time
                        logger.info(
                            f"Succeeded after {attempt + 1} attempts "
                            f"({duration:.2f}s total)"
                        )

                    return result

                except exceptions as e:
                    last_exception = e

                    # Check if retryable
                    if not is_retryable_error(e, exceptions):
                        raise

                    # Check if we have more attempts
                    if attempt >= max_attempts - 1:
                        break

                    # Calculate delay
                    delay = calculate_backoff_delay(
                        attempt,
                        base_delay,
                        max_delay,
                        exponential_base,
                        jitter,
                    )

                    # Log retry
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Call retry callback
                    if on_retry:
                        on_retry(e, attempt + 1)

                    # Wait before retry
                    await asyncio.sleep(delay)

            # All attempts failed
            duration = time.time() - start_time
            logger.error(f"All {max_attempts} attempts failed after {duration:.2f}s")

            if last_exception:
                raise last_exception
            else:
                raise VirtualAgoraError("Retry failed with unknown error")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


class RetryManager:
    """Manages retry policies and statistics with LangGraph integration."""

    def __init__(self):
        """Initialize retry manager."""
        self.statistics: Dict[str, RetryStatistics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.policies: Dict[str, Dict[str, Any]] = self._default_policies()
        self.langgraph_policies: Dict[str, "RetryPolicy"] = (
            {}
        )  # Cache for LangGraph policies

    def _default_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get default retry policies.

        Returns:
            Default policies by operation type
        """
        return {
            "provider_api": {
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 30.0,
                "exponential_base": 2.0,
                "exceptions": (ProviderError, TimeoutError, NetworkTransientError),
            },
            "network": {
                "max_attempts": 5,
                "base_delay": 0.5,
                "max_delay": 20.0,
                "exponential_base": 1.5,
                "exceptions": (TimeoutError, NetworkTransientError),
            },
            "state_operation": {
                "max_attempts": 2,
                "base_delay": 0.1,
                "max_delay": 1.0,
                "exponential_base": 2.0,
                "exceptions": (StateError,),
            },
        }

    def get_policy(self, operation_type: str) -> Dict[str, Any]:
        """Get retry policy for operation type.

        Args:
            operation_type: Type of operation

        Returns:
            Retry policy configuration
        """
        return self.policies.get(operation_type, self.policies["network"])

    def get_statistics(self, operation: str) -> RetryStatistics:
        """Get statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            Retry statistics
        """
        if operation not in self.statistics:
            self.statistics[operation] = RetryStatistics()
        return self.statistics[operation]

    def get_circuit_breaker(
        self,
        operation: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for operation.

        Args:
            operation: Operation name
            failure_threshold: Failure threshold
            recovery_timeout: Recovery timeout
            expected_exception: Exception type to track

        Returns:
            Circuit breaker instance
        """
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                name=operation,
            )
        return self.circuit_breakers[operation]

    def with_retry(
        self,
        operation: str,
        operation_type: str = "network",
        use_circuit_breaker: bool = True,
    ) -> Callable[[F], F]:
        """Decorator with retry and circuit breaker.

        Args:
            operation: Operation name for tracking
            operation_type: Type of operation for policy
            use_circuit_breaker: Whether to use circuit breaker

        Returns:
            Decorator function
        """
        policy = self.get_policy(operation_type)
        stats = self.get_statistics(operation)

        def decorator(func: F) -> F:
            # Apply retry decorator
            retry_func = retry_with_backoff(**policy)(func)

            # Apply circuit breaker if requested
            breaker = (
                self.get_circuit_breaker(operation) if use_circuit_breaker else None
            )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                success = False
                retries = 0

                try:
                    if breaker:
                        result = breaker.call(retry_func, *args, **kwargs)
                    else:
                        result = retry_func(*args, **kwargs)
                    success = True
                    return result
                except Exception:
                    raise
                finally:
                    duration = time.time() - start_time
                    stats.record_attempt(success, duration, retries)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                success = False
                retries = 0

                try:
                    if breaker:
                        result = await breaker.call_async(retry_func, *args, **kwargs)
                    else:
                        result = await retry_func(*args, **kwargs)
                    success = True
                    return result
                except Exception:
                    raise
                finally:
                    duration = time.time() - start_time
                    stats.record_attempt(success, duration, retries)

            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            else:
                return cast(F, sync_wrapper)

        return decorator

    def create_langgraph_policy(
        self,
        operation_type: str = "network",
        custom_retry_on: Optional[Callable[[Exception], bool]] = None,
    ) -> Optional["RetryPolicy"]:
        """Create a LangGraph RetryPolicy from our retry configuration.

        Args:
            operation_type: Type of operation for policy selection
            custom_retry_on: Custom function to determine if error should be retried

        Returns:
            LangGraph RetryPolicy or None if not available
        """
        if not LANGGRAPH_AVAILABLE or RetryPolicy is None:
            logger.debug("LangGraph RetryPolicy not available")
            return None

        # Check cache first
        if operation_type in self.langgraph_policies:
            return self.langgraph_policies[operation_type]

        # Get our policy configuration
        policy_config = self.get_policy(operation_type)

        # Create retry_on function if not provided
        if custom_retry_on is None:
            retryable_exceptions = policy_config.get("exceptions", (Exception,))

            def default_retry_on(error: Exception) -> bool:
                return is_retryable_error(error, retryable_exceptions)

            custom_retry_on = default_retry_on

        # Create LangGraph RetryPolicy
        langgraph_policy = RetryPolicy(
            retry_on=custom_retry_on,
            max_attempts=policy_config.get("max_attempts", 3),
        )

        # Cache the policy
        self.langgraph_policies[operation_type] = langgraph_policy

        logger.debug(f"Created LangGraph RetryPolicy for {operation_type}")
        return langgraph_policy

    def with_langgraph_retry(
        self,
        operation: str,
        operation_type: str = "network",
    ) -> Dict[str, Any]:
        """Get configuration for LangGraph node with retry policy.

        This method returns a configuration dict that can be used when
        adding nodes to a LangGraph StateGraph.

        Args:
            operation: Operation name for tracking
            operation_type: Type of operation for policy

        Returns:
            Configuration dict with retry policy

        Example:
            ```python
            retry_config = retry_manager.with_langgraph_retry(
                "api_call", "provider_api"
            )
            graph.add_node("my_node", my_func, **retry_config)
            ```
        """
        config = {
            "metadata": {
                "operation": operation,
                "operation_type": operation_type,
            }
        }

        # Add LangGraph retry policy if available
        retry_policy = self.create_langgraph_policy(operation_type)
        if retry_policy:
            config["retry_policy"] = retry_policy

        # Track operation in statistics
        stats = self.get_statistics(operation)
        config["metadata"]["stats"] = stats

        return config

    def create_resilient_node(
        self,
        func: Callable,
        operation: str,
        operation_type: str = "network",
        use_circuit_breaker: bool = True,
    ) -> Callable:
        """Create a resilient function for use as a LangGraph node.

        This wraps a function with both our retry logic and circuit breaker,
        making it suitable for use in LangGraph workflows.

        Args:
            func: Function to wrap
            operation: Operation name for tracking
            operation_type: Type of operation
            use_circuit_breaker: Whether to use circuit breaker

        Returns:
            Wrapped function with error resilience
        """
        # Apply our retry decorator
        retry_func = self.with_retry(operation, operation_type, use_circuit_breaker)(
            func
        )

        # Add LangGraph metadata
        retry_func._langgraph_metadata = {
            "operation": operation,
            "operation_type": operation_type,
            "retry_policy": self.create_langgraph_policy(operation_type),
        }

        return retry_func

    def get_retry_context(self, operation: str) -> Dict[str, Any]:
        """Get retry context information for an operation.

        This is useful for logging and debugging in LangGraph workflows.

        Args:
            operation: Operation name

        Returns:
            Context dict with statistics and circuit breaker state
        """
        context = {
            "operation": operation,
            "statistics": {},
            "circuit_breaker": None,
        }

        # Add statistics if available
        if operation in self.statistics:
            stats = self.statistics[operation]
            context["statistics"] = {
                "total_attempts": stats.total_attempts,
                "success_rate": stats.get_success_rate(),
                "average_duration": stats.get_average_retry_duration(),
                "last_attempt": stats.last_attempt_time,
                "last_success": stats.last_success_time,
                "last_failure": stats.last_failure_time,
            }

        # Add circuit breaker state if available
        if operation in self.circuit_breakers:
            breaker = self.circuit_breakers[operation]
            context["circuit_breaker"] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time,
            }

        return context

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dictionary mapping operation names to circuit breaker stats
        """
        status = {}
        for operation, breaker in self.circuit_breakers.items():
            status[operation] = breaker.get_stats()
        return status

    def create_langgraph_circuit_aware_policy(
        self,
        operation: str,
        operation_type: str = "network",
    ) -> Optional["RetryPolicy"]:
        """Create a LangGraph RetryPolicy that's aware of circuit breakers.

        This creates a retry policy that checks both the circuit breaker
        state and the standard retry logic.

        Args:
            operation: Operation name
            operation_type: Type of operation

        Returns:
            Circuit-aware RetryPolicy or None
        """
        # Get or create circuit breaker
        policy_config = self.get_policy(operation_type)
        breaker = self.get_circuit_breaker(
            operation,
            expected_exception=policy_config.get("exceptions", (Exception,))[0],
        )

        # Use circuit breaker's LangGraph policy
        return breaker.as_langgraph_policy()


# Global retry manager instance
retry_manager = RetryManager()
