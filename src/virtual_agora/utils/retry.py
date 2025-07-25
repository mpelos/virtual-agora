"""Retry mechanism with exponential backoff for Virtual Agora.

This module provides retry functionality with exponential backoff,
jitter, and circuit breaker patterns for handling transient failures.
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
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
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
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise VirtualAgoraError(
                    "Circuit breaker is open",
                    details={"failure_count": self.failure_count}
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
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise VirtualAgoraError(
                    "Circuit breaker is open",
                    details={"failure_count": self.failure_count}
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
        self.state = CircuitState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
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
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    
    # Add jitter to prevent thundering herd
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


def is_retryable_error(error: Exception, retryable_errors: Tuple[Type[Exception], ...]) -> bool:
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
            logger.error(
                f"All {max_attempts} attempts failed after {duration:.2f}s"
            )
            
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
            logger.error(
                f"All {max_attempts} attempts failed after {duration:.2f}s"
            )
            
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
    """Manages retry policies and statistics."""
    
    def __init__(self):
        """Initialize retry manager."""
        self.statistics: Dict[str, RetryStatistics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.policies: Dict[str, Dict[str, Any]] = self._default_policies()
    
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
    ) -> CircuitBreaker:
        """Get or create circuit breaker for operation.
        
        Args:
            operation: Operation name
            failure_threshold: Failure threshold
            recovery_timeout: Recovery timeout
            
        Returns:
            Circuit breaker instance
        """
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
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
            breaker = self.get_circuit_breaker(operation) if use_circuit_breaker else None
            
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


# Global retry manager instance
retry_manager = RetryManager()