"""Unit tests for retry mechanism."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from virtual_agora.utils.retry import (
    retry_with_backoff,
    calculate_backoff_delay,
    is_retryable_error,
    RetryManager,
    CircuitBreaker,
    CircuitState,
    RetryState,
    RetryStatistics,
)
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ProviderError,
    NetworkTransientError,
    ProviderRateLimitError,
    ConfigurationError,
    TimeoutError,
)


class TestRetryStatistics:
    """Test retry statistics tracking."""

    def test_record_attempt(self):
        """Test recording retry attempts."""
        stats = RetryStatistics()

        # Record successful attempt
        stats.record_attempt(success=True, duration=1.5, retries=0)

        assert stats.total_attempts == 1
        assert stats.successful_attempts == 1
        assert stats.failed_attempts == 0
        assert stats.total_retries == 0
        assert stats.retry_durations == [1.5]
        assert stats.last_success_time is not None

        # Record failed attempt with retries
        stats.record_attempt(success=False, duration=3.0, retries=2)

        assert stats.total_attempts == 2
        assert stats.successful_attempts == 1
        assert stats.failed_attempts == 1
        assert stats.total_retries == 2
        assert stats.retry_durations == [1.5, 3.0]
        assert stats.last_failure_time is not None

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = RetryStatistics()

        # No attempts
        assert stats.get_success_rate() == 0.0

        # All successful
        stats.record_attempt(True, 1.0, 0)
        stats.record_attempt(True, 1.0, 0)
        assert stats.get_success_rate() == 100.0

        # Mixed results
        stats.record_attempt(False, 1.0, 1)
        stats.record_attempt(False, 1.0, 1)
        assert stats.get_success_rate() == 50.0

    def test_average_retry_duration(self):
        """Test average retry duration calculation."""
        stats = RetryStatistics()

        # No attempts
        assert stats.get_average_retry_duration() == 0.0

        # Multiple attempts
        stats.record_attempt(True, 1.0, 0)
        stats.record_attempt(True, 2.0, 1)
        stats.record_attempt(False, 3.0, 2)

        assert stats.get_average_retry_duration() == 2.0


class TestBackoffCalculation:
    """Test backoff delay calculations."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        # Base cases
        assert calculate_backoff_delay(0, base_delay=1.0, jitter=False) == 1.0
        assert calculate_backoff_delay(1, base_delay=1.0, jitter=False) == 2.0
        assert calculate_backoff_delay(2, base_delay=1.0, jitter=False) == 4.0
        assert calculate_backoff_delay(3, base_delay=1.0, jitter=False) == 8.0

        # Different base delay
        assert calculate_backoff_delay(2, base_delay=2.0, jitter=False) == 8.0

        # Different exponential base
        assert (
            calculate_backoff_delay(
                2, base_delay=1.0, exponential_base=3.0, jitter=False
            )
            == 9.0
        )

    def test_max_delay_cap(self):
        """Test maximum delay capping."""
        # Should cap at max_delay
        delay = calculate_backoff_delay(
            10, base_delay=1.0, max_delay=10.0, jitter=False
        )
        assert delay == 10.0

        # Very high attempt should still cap
        delay = calculate_backoff_delay(
            100, base_delay=1.0, max_delay=30.0, jitter=False
        )
        assert delay == 30.0

    def test_jitter(self):
        """Test jitter addition."""
        # With jitter, delay should be between 50% and 100% of calculated
        base_delay = 4.0  # For attempt 2 with base 1.0

        for _ in range(10):
            delay = calculate_backoff_delay(2, base_delay=1.0, jitter=True)
            assert 2.0 <= delay <= 4.0  # 50% to 100% of 4.0


class TestRetryableErrorDetection:
    """Test retryable error detection."""

    def test_exact_type_match(self):
        """Test exact error type matching."""
        error = NetworkTransientError("Network error")
        assert is_retryable_error(error, (NetworkTransientError,))
        assert not is_retryable_error(error, (ProviderError,))

    def test_inheritance_match(self):
        """Test error inheritance matching."""
        # ProviderRateLimitError inherits from ProviderError
        error = ProviderRateLimitError("Rate limited", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))
        assert is_retryable_error(error, (ProviderRateLimitError,))

    def test_provider_error_conditions(self):
        """Test special provider error conditions."""
        # Rate limit errors are retryable
        error = ProviderError("Rate limit exceeded", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        error = ProviderError("Too many requests", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        # HTTP error codes
        error = ProviderError("Error 429: Too Many Requests", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        error = ProviderError("Error 503: Service Unavailable", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        error = ProviderError("Error 504: Gateway Timeout", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        # Non-retryable provider errors
        error = ProviderError("Invalid API key", provider="OpenAI")
        assert not is_retryable_error(error, (NetworkTransientError,))


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_sync_retry_success(self):
        """Test sync function retry that eventually succeeds."""
        attempt_count = 0

        @retry_with_backoff(
            max_attempts=3, exceptions=(ValueError,), base_delay=0.01, jitter=False
        )
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

    def test_sync_retry_exhausted(self):
        """Test sync function that exhausts retries."""
        attempt_count = 0

        @retry_with_backoff(max_attempts=2, exceptions=(ValueError,), base_delay=0.01)
        def always_failing():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"Attempt {attempt_count} failed")

        with pytest.raises(ValueError, match="Attempt 2 failed"):
            always_failing()

        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async function retry that eventually succeeds."""
        attempt_count = 0

        @retry_with_backoff(
            max_attempts=3, exceptions=(ValueError,), base_delay=0.01, jitter=False
        )
        async def async_flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return "async success"

        result = await async_flaky_function()
        assert result == "async success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_exhausted(self):
        """Test async function that exhausts retries."""
        attempt_count = 0

        @retry_with_backoff(max_attempts=2, exceptions=(ValueError,), base_delay=0.01)
        async def async_always_failing():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"Async attempt {attempt_count} failed")

        with pytest.raises(ValueError, match="Async attempt 2 failed"):
            await async_always_failing()

        assert attempt_count == 2

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        attempt_count = 0

        @retry_with_backoff(max_attempts=3, exceptions=(ValueError,), base_delay=0.01)
        def wrong_exception():
            nonlocal attempt_count
            attempt_count += 1
            raise TypeError("Wrong exception type")

        with pytest.raises(TypeError, match="Wrong exception type"):
            wrong_exception()

        # Should not retry TypeError
        assert attempt_count == 1

    def test_retry_callback(self):
        """Test retry callback functionality."""
        retry_calls = []

        def on_retry(error, attempt):
            retry_calls.append((str(error), attempt))

        @retry_with_backoff(
            max_attempts=3, exceptions=(ValueError,), base_delay=0.01, on_retry=on_retry
        )
        def function_with_callback():
            if len(retry_calls) < 2:
                raise ValueError("Still failing")
            return "success"

        result = function_with_callback()
        assert result == "success"
        assert len(retry_calls) == 2
        assert retry_calls[0] == ("Still failing", 1)
        assert retry_calls[1] == ("Still failing", 2)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=0.1, expected_exception=ValueError
        )

        assert breaker.state == CircuitState.CLOSED

        def failing_function():
            raise ValueError("Test failure")

        # First failures
        for i in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_function)
            assert breaker.failure_count == i + 1

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

        # Calls should be rejected
        with pytest.raises(VirtualAgoraError, match=r"Circuit breaker .* is open"):
            breaker.call(lambda: "success")

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        def failing_function():
            raise Exception("Fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should enter half-open state
        def success_function():
            return "success"

        result = breaker.call(success_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test circuit breaker with async functions."""
        breaker = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.1, expected_exception=ValueError
        )

        async def async_failing():
            raise ValueError("Async fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call_async(async_failing)

        assert breaker.state == CircuitState.OPEN

        # Should reject calls
        with pytest.raises(VirtualAgoraError, match=r"Circuit breaker .* is open"):
            await breaker.call_async(async_failing)

        # Wait and recover
        await asyncio.sleep(0.15)

        async def async_success():
            return "async success"

        result = await breaker.call_async(async_success)
        assert result == "async success"
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failing in half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        def failing_function():
            raise Exception("Always fails")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(failing_function)

        # Wait for recovery
        time.sleep(0.15)

        # Try in half-open state - should fail and re-open
        with pytest.raises(Exception):
            breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3


class TestRetryManager:
    """Test retry manager functionality."""

    def test_default_policies(self):
        """Test default retry policies."""
        manager = RetryManager()

        # Provider API policy
        policy = manager.get_policy("provider_api")
        assert policy["max_attempts"] == 3
        assert policy["base_delay"] == 1.0
        assert policy["max_delay"] == 30.0
        assert ProviderError in policy["exceptions"]

        # Network policy
        policy = manager.get_policy("network")
        assert policy["max_attempts"] == 5
        assert policy["base_delay"] == 0.5
        assert policy["max_delay"] == 20.0

        # State operation policy
        policy = manager.get_policy("state_operation")
        assert policy["max_attempts"] == 2
        assert policy["base_delay"] == 0.1

    def test_statistics_tracking(self):
        """Test statistics tracking."""
        manager = RetryManager()

        # Get statistics for new operation
        stats = manager.get_statistics("test_op")
        assert stats.total_attempts == 0

        # Update and verify
        stats.record_attempt(True, 1.0, 0)
        stats = manager.get_statistics("test_op")
        assert stats.total_attempts == 1
        assert stats.successful_attempts == 1

    def test_circuit_breaker_management(self):
        """Test circuit breaker management."""
        manager = RetryManager()

        # Get circuit breaker
        breaker1 = manager.get_circuit_breaker("test_op")
        assert breaker1.failure_threshold == 5  # Default

        # Get same breaker again
        breaker2 = manager.get_circuit_breaker("test_op")
        assert breaker1 is breaker2

        # Get breaker with custom settings
        breaker3 = manager.get_circuit_breaker(
            "custom_op", failure_threshold=10, recovery_timeout=120.0
        )
        assert breaker3.failure_threshold == 10

    def test_with_retry_decorator(self):
        """Test with_retry decorator."""
        manager = RetryManager()
        attempt_count = 0

        @manager.with_retry(
            operation="test_operation",
            operation_type="network",
            use_circuit_breaker=False,
        )
        def test_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise TimeoutError("Timeout")
            return "success"

        result = test_function()
        assert result == "success"
        assert attempt_count == 3

        # Check statistics
        stats = manager.get_statistics("test_operation")
        # The wrapper tracks overall attempts, not individual retries
        assert stats.total_attempts > 0  # At least one attempt was recorded

    @pytest.mark.asyncio
    async def test_async_with_retry_decorator(self):
        """Test async with_retry decorator."""
        manager = RetryManager()
        attempt_count = 0

        @manager.with_retry(
            operation="async_test",
            operation_type="provider_api",
            use_circuit_breaker=True,
        )
        async def async_test_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ProviderError("API error")
            return "async success"

        result = await async_test_function()
        assert result == "async success"
        assert attempt_count == 2

        # Check statistics
        stats = manager.get_statistics("async_test")
        # The wrapper tracks overall attempts, not individual retries
        assert stats.total_attempts > 0  # At least one attempt was recorded


class TestRetryIntegration:
    """Test retry mechanism integration scenarios."""

    def test_provider_api_retry_scenario(self):
        """Test realistic provider API retry scenario."""
        manager = RetryManager()
        api_calls = []

        @manager.with_retry(operation="openai_api_call", operation_type="provider_api")
        def call_openai_api():
            api_calls.append(time.time())

            if len(api_calls) == 1:
                raise ProviderRateLimitError(
                    "Rate limit exceeded", provider="OpenAI", retry_after=1.0
                )
            elif len(api_calls) == 2:
                raise NetworkTransientError("Connection timeout")
            else:
                return {"response": "Success"}

        start_time = time.time()
        result = call_openai_api()
        elapsed = time.time() - start_time

        assert result == {"response": "Success"}
        assert len(api_calls) == 3
        # Should have delays between attempts
        assert elapsed >= 0.02  # At least some delay

    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker preventing retries."""
        manager = RetryManager()

        @manager.with_retry(
            operation="flaky_service",
            operation_type="network",
            use_circuit_breaker=True,
        )
        def call_flaky_service():
            raise NetworkTransientError("Service unavailable")

        # Open the circuit
        for _ in range(5):
            with pytest.raises(NetworkTransientError):
                call_flaky_service()

        # Circuit should be open, preventing further calls
        with pytest.raises(VirtualAgoraError, match=r"Circuit breaker .* is open"):
            call_flaky_service()

    def test_different_operation_isolation(self):
        """Test that different operations have isolated retry state."""
        manager = RetryManager()

        # Fail operation A multiple times
        @manager.with_retry("operation_a", "network")
        def operation_a():
            raise TimeoutError("Timeout A")

        for _ in range(5):
            with pytest.raises(TimeoutError):
                operation_a()

        # Operation B should still work
        @manager.with_retry("operation_b", "network")
        def operation_b():
            return "B works"

        result = operation_b()
        assert result == "B works"

        # Stats should be separate
        stats_a = manager.get_statistics("operation_a")
        stats_b = manager.get_statistics("operation_b")

        assert stats_a.failed_attempts == 5
        assert stats_b.successful_attempts == 1
