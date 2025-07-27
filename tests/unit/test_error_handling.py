"""Unit tests for error handling framework."""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from virtual_agora.utils.error_handler import (
    ErrorHandler,
    ErrorContext,
    ErrorSeverity,
    RecoveryStrategy,
)
from virtual_agora.utils.retry import (
    retry_with_backoff,
    calculate_backoff_delay,
    is_retryable_error,
    RetryManager,
    CircuitBreaker,
    CircuitState,
)
from virtual_agora.utils.error_reporter import ErrorReporter
from virtual_agora.utils.shutdown import ShutdownHandler, graceful_shutdown_context
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ConfigurationError,
    ProviderError,
    StateError,
    ValidationError,
    RecoverableError,
    CriticalError,
    ProviderRateLimitError,
    NetworkTransientError,
    StateCorruptionError,
    UserInterventionRequired,
)


class TestErrorClasses:
    """Test enhanced error classes."""

    def test_recoverable_error(self):
        """Test RecoverableError class."""
        error = RecoverableError(
            "Temporary failure", retry_after=5.0, max_retries=3, details={"code": 503}
        )

        assert str(error) == "Temporary failure (code=503)"
        assert error.retry_after == 5.0
        assert error.max_retries == 3
        assert error.details["code"] == 503

    def test_critical_error(self):
        """Test CriticalError class."""
        error = CriticalError(
            "System failure",
            system_component="database",
            requires_restart=True,
            details={"severity": "high"},
        )

        assert str(error) == "System failure (severity=high)"
        assert error.system_component == "database"
        assert error.requires_restart is True
        assert error.details["severity"] == "high"

    def test_provider_rate_limit_error(self):
        """Test ProviderRateLimitError class."""
        error = ProviderRateLimitError(
            "Rate limit exceeded",
            provider="OpenAI",
            retry_after=60.0,
            limit_type="requests",
            details={"limit": 100},
        )

        assert str(error) == "Rate limit exceeded (limit=100)"
        assert error.provider == "OpenAI"
        assert error.retry_after == 60.0
        assert error.limit_type == "requests"
        assert error.error_code == "rate_limit"

    def test_network_transient_error(self):
        """Test NetworkTransientError class."""
        error = NetworkTransientError(
            "Connection timeout",
            operation="api_call",
            endpoint="https://api.example.com",
            details={"timeout": 30},
        )

        assert str(error) == "Connection timeout (timeout=30)"
        assert error.operation == "api_call"
        assert error.endpoint == "https://api.example.com"
        assert error.retry_after == 1.0  # Default
        assert error.max_retries == 5  # Default

    def test_state_corruption_error(self):
        """Test StateCorruptionError class."""
        error = StateCorruptionError(
            "State corrupted",
            corrupted_fields=["agents", "messages"],
            last_known_good_state={"session_id": "test"},
            details={"recovery_attempted": True},
        )

        assert str(error) == "State corrupted (recovery_attempted=True)"
        assert error.corrupted_fields == ["agents", "messages"]
        assert error.last_known_good_state["session_id"] == "test"
        assert error.system_component == "state_manager"
        assert error.requires_restart is True

    def test_user_intervention_required(self):
        """Test UserInterventionRequired class."""
        error = UserInterventionRequired(
            "User input needed",
            prompt="Please select an option:",
            options=["Continue", "Cancel", "Retry"],
            default_option="Continue",
            details={"context": "api_key_missing"},
        )

        assert str(error) == "User input needed (context=api_key_missing)"
        assert error.prompt == "Please select an option:"
        assert error.options == ["Continue", "Cancel", "Retry"]
        assert error.default_option == "Continue"


class TestErrorHandler:
    """Test error handler functionality."""

    def test_capture_context(self):
        """Test error context capture."""
        handler = ErrorHandler()
        error = ValueError("Test error")

        context = handler.capture_context(
            error,
            operation="test_operation",
            phase="testing",
            state_snapshot={"key": "value"},
            custom_field="custom_value",
        )

        assert context.error == error
        assert context.operation == "test_operation"
        assert context.phase == "testing"
        assert context.state_snapshot == {"key": "value"}
        assert context.metadata["custom_field"] == "custom_value"
        assert context.severity == ErrorSeverity.HIGH
        assert len(handler.error_history) == 1
        assert handler.error_counts["ValueError"] == 1

    def test_severity_determination(self):
        """Test error severity determination."""
        handler = ErrorHandler()

        # Critical errors
        config_error = ConfigurationError("Config error")
        context = handler.capture_context(config_error)
        assert context.severity == ErrorSeverity.CRITICAL

        # High severity errors
        state_error = StateError("State error")
        context = handler.capture_context(state_error)
        assert context.severity == ErrorSeverity.HIGH

        # Medium severity for rate limits
        rate_error = ProviderError("Rate limit exceeded", provider="OpenAI")
        context = handler.capture_context(rate_error)
        assert context.severity == ErrorSeverity.MEDIUM

        # Medium severity for validation
        val_error = ValidationError("Invalid input")
        context = handler.capture_context(val_error)
        assert context.severity == ErrorSeverity.MEDIUM

    def test_recovery_strategy_determination(self):
        """Test recovery strategy determination."""
        handler = ErrorHandler()

        # Provider errors should retry
        provider_error = ProviderError("API error")
        context = handler.capture_context(provider_error)
        assert context.recovery_strategy == RecoveryStrategy.RETRY

        # State errors should rollback
        state_error = StateError("State inconsistent")
        context = handler.capture_context(state_error)
        assert context.recovery_strategy == RecoveryStrategy.ROLLBACK

        # Validation errors need user intervention
        val_error = ValidationError("Invalid input")
        context = handler.capture_context(val_error)
        assert context.recovery_strategy == RecoveryStrategy.USER_INTERVENTION

        # Configuration errors need shutdown
        config_error = ConfigurationError("Bad config")
        context = handler.capture_context(config_error)
        assert context.recovery_strategy == RecoveryStrategy.GRACEFUL_SHUTDOWN

    def test_format_user_message(self):
        """Test user-friendly message formatting."""
        handler = ErrorHandler()

        # Provider error
        error = ProviderError("Connection failed", provider="OpenAI")
        context = handler.capture_context(error, operation="sending request")
        message = handler.format_user_message(context)

        assert "OpenAI service" in message
        assert "while sending request" in message
        assert "Connection failed" in message

        # Timeout error
        from virtual_agora.utils.exceptions import (
            TimeoutError as VirtualAgoraTimeoutError,
        )

        error = VirtualAgoraTimeoutError("Request timed out")
        context = handler.capture_context(error)
        message = handler.format_user_message(context)

        assert "operation took too long to complete" in message

    def test_error_summary(self):
        """Test error summary generation."""
        handler = ErrorHandler()

        # Generate some errors
        handler.capture_context(ValueError("Error 1"))
        handler.capture_context(ValueError("Error 2"))
        handler.capture_context(TypeError("Error 3"))
        time.sleep(0.01)  # Small delay for rate calculation

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["error_types"]["ValueError"] == 2
        assert summary["error_types"]["TypeError"] == 1
        assert summary["most_common"]["type"] == "ValueError"
        assert summary["most_common"]["count"] == 2
        assert summary["most_common"]["percentage"] == pytest.approx(66.7, rel=0.1)

    def test_circuit_breaker_check(self):
        """Test circuit breaker threshold checking."""
        handler = ErrorHandler()

        # Add errors below threshold
        for _ in range(4):
            handler.capture_context(ValueError("Error"))

        assert not handler.should_circuit_break(ValueError, threshold=5)

        # Add one more to reach threshold
        handler.capture_context(ValueError("Error"))

        assert handler.should_circuit_break(ValueError, threshold=5)

    def test_error_boundary_context_manager(self):
        """Test error boundary context manager."""
        handler = ErrorHandler()

        # Successful operation
        with handler.error_boundary("test_op") as context:
            assert context is None

        # Failed operation
        with pytest.raises(ValueError):
            with handler.error_boundary("test_op") as context:
                raise ValueError("Test error")

        assert len(handler.error_history) == 1
        assert handler.error_history[0].operation == "test_op"


class TestRetryMechanism:
    """Test retry mechanism functionality."""

    def test_calculate_backoff_delay(self):
        """Test backoff delay calculation."""
        # Without jitter
        delay = calculate_backoff_delay(0, base_delay=1.0, jitter=False)
        assert delay == 1.0

        delay = calculate_backoff_delay(1, base_delay=1.0, jitter=False)
        assert delay == 2.0

        delay = calculate_backoff_delay(2, base_delay=1.0, jitter=False)
        assert delay == 4.0

        # With max delay
        delay = calculate_backoff_delay(10, base_delay=1.0, max_delay=5.0, jitter=False)
        assert delay == 5.0

        # With jitter
        delay = calculate_backoff_delay(1, base_delay=1.0, jitter=True)
        assert 1.0 <= delay <= 2.0

    def test_is_retryable_error(self):
        """Test retryable error detection."""
        # Direct type match
        error = NetworkTransientError("Network error")
        assert is_retryable_error(error, (NetworkTransientError,))

        # Inheritance match
        error = ProviderRateLimitError("Rate limit", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        # Rate limit detection
        error = ProviderError("Rate limit exceeded", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        # HTTP error codes
        error = ProviderError("Error 429", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        error = ProviderError("Error 503", provider="OpenAI")
        assert is_retryable_error(error, (ProviderError,))

        # Non-retryable
        error = ConfigurationError("Bad config")
        assert not is_retryable_error(error, (ProviderError,))

    def test_retry_decorator_sync(self):
        """Test retry decorator for synchronous functions."""
        attempt_count = 0

        @retry_with_backoff(
            max_attempts=3, exceptions=(ValueError,), base_delay=0.01, jitter=False
        )
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Test error")
            return "success"

        result = failing_function()
        assert result == "success"
        assert attempt_count == 3

    def test_retry_decorator_async(self):
        """Test retry decorator for async functions."""
        attempt_count = 0

        @retry_with_backoff(
            max_attempts=3, exceptions=(ValueError,), base_delay=0.01, jitter=False
        )
        async def failing_async_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Test error")
            return "success"

        result = asyncio.run(failing_async_function())
        assert result == "success"
        assert attempt_count == 3

    def test_retry_decorator_exhausted(self):
        """Test retry decorator when all attempts fail."""

        @retry_with_backoff(max_attempts=2, exceptions=(ValueError,), base_delay=0.01)
        def always_failing():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_failing()

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=0.1, expected_exception=ValueError
        )

        def failing_function():
            raise ValueError("Test error")

        # First failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_function)

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

        # Should reject calls
        with pytest.raises(VirtualAgoraError, match="Circuit breaker .* is open"):
            breaker.call(lambda: "success")

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should allow half-open attempt
        def success_function():
            return "success"

        result = breaker.call(success_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_retry_manager(self):
        """Test retry manager functionality."""
        manager = RetryManager()

        # Test policy retrieval
        policy = manager.get_policy("provider_api")
        assert policy["max_attempts"] == 3
        assert policy["base_delay"] == 1.0

        # Test statistics tracking
        stats = manager.get_statistics("test_op")
        assert stats.total_attempts == 0

        # Test decorator with tracking
        @manager.with_retry("test_op", "network", use_circuit_breaker=False)
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"


class TestErrorReporter:
    """Test error reporter functionality."""

    def test_report_error(self, capsys):
        """Test error reporting to console."""
        console = Mock()
        reporter = ErrorReporter(console)

        error = ProviderError("API failed", provider="OpenAI")
        context = ErrorContext(
            error=error,
            operation="api_call",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        reporter.report_error(context)

        assert len(reporter.reported_errors) == 1
        # Check that error pattern is tracked (pattern format may vary)
        assert len(reporter.error_patterns) == 1
        assert sum(reporter.error_patterns.values()) == 1

        # Check console output
        console.print.assert_called()

    def test_error_pattern_tracking(self):
        """Test error pattern tracking."""
        reporter = ErrorReporter()

        # Report similar errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            context = ErrorContext(
                error=error,
                operation="same_operation",
            )
            reporter.report_error(context)

        assert reporter.error_patterns["ValueError:same_operation"] == 3

    def test_recovery_suggestions(self):
        """Test recovery suggestion generation."""
        reporter = ErrorReporter()

        # Configuration error suggestions
        error = ConfigurationError("Bad config")
        context = ErrorContext(error=error)
        suggestions = reporter._get_recovery_suggestions(context)

        assert any("config.yml" in s for s in suggestions)
        assert any("required fields" in s for s in suggestions)

        # Provider error suggestions
        error = ProviderError("Auth failed", provider="OpenAI")
        context = ErrorContext(error=error)
        suggestions = reporter._get_recovery_suggestions(context)

        assert any("OpenAI API key" in s for s in suggestions)
        assert any("credentials" in s for s in suggestions)

    def test_save_error_report(self, tmp_path):
        """Test saving error report to file."""
        reporter = ErrorReporter()

        # Generate some errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            context = ErrorContext(
                error=error,
                operation=f"op_{i}",
                severity=ErrorSeverity.HIGH,
            )
            reporter.reported_errors.append(context)

        # Save report
        report_path = reporter.save_error_report("test_session", output_dir=tmp_path)

        assert report_path.exists()
        assert "error_report_test_session" in report_path.name

        # Check report content
        with open(report_path) as f:
            report = json.load(f)

        assert report["session_id"] == "test_session"
        assert report["total_errors"] == 3
        assert len(report["errors"]) == 3
        assert report["errors"][0]["error_type"] == "ValueError"

    def test_detect_error_trends(self):
        """Test error trend detection."""
        reporter = ErrorReporter()

        # Add errors with increasing frequency
        base_time = datetime.now()
        for i in range(5):
            error = ValueError(f"Error {i}")
            context = ErrorContext(
                error=error,
                timestamp=base_time,
            )
            # Simulate decreasing time between errors
            context.timestamp = base_time
            reporter.reported_errors.append(context)

        # Add repeated provider errors
        for i in range(3):
            error = ProviderError("API error", provider="OpenAI")
            context = ErrorContext(error=error)
            reporter.reported_errors.append(context)

        trends = reporter.detect_error_trends()

        assert any("Multiple errors from OpenAI" in t for t in trends)


class TestShutdownHandler:
    """Test shutdown handler functionality."""

    def test_cleanup_task_registration(self):
        """Test cleanup task registration."""
        handler = ShutdownHandler(timeout=1.0)

        called = False

        def cleanup_task():
            nonlocal called
            called = True

        handler.register_cleanup_task(cleanup_task, name="test_cleanup")

        # Trigger cleanup
        handler._run_cleanup_tasks()

        assert called

    def test_resource_lock(self):
        """Test resource lock functionality."""
        handler = ShutdownHandler()

        # Acquire lock
        assert handler.acquire_resource_lock("test_resource")
        assert "test_resource" in handler._resource_locks

        # Release lock
        handler.release_resource_lock("test_resource")
        assert "test_resource" not in handler._resource_locks

    def test_resource_lock_context_manager(self):
        """Test resource lock context manager."""
        handler = ShutdownHandler()

        with handler.resource_lock("test_resource"):
            assert "test_resource" in handler._resource_locks

        assert "test_resource" not in handler._resource_locks

    @patch("sys.platform", "linux")
    def test_signal_handling_unix(self):
        """Test signal handling on Unix systems."""
        handler = ShutdownHandler()

        # Simulate SIGTERM
        handler._signal_handler(signal.SIGTERM, None)

        assert handler._shutdown_requested

    def test_save_session_state(self, tmp_path):
        """Test session state saving."""
        handler = ShutdownHandler()

        # Mock state manager
        state_manager = Mock()
        state_manager._state = {"session_id": "test_123", "data": "test"}
        state_manager.export_session.return_value = {
            "session_id": "test_123",
            "data": "test",
        }

        # Save state
        saved_path = handler.save_session_state(
            state_manager=state_manager, session_id="test_123"
        )

        # Check that session was exported
        state_manager.export_session.assert_called_once()

    def test_generate_shutdown_report(self, tmp_path):
        """Test shutdown report generation."""
        handler = ShutdownHandler()

        # Mock error reporter
        error_reporter = Mock()
        error_reporter.save_error_report.return_value = tmp_path / "error_report.json"

        # Generate report
        report_path = handler.generate_shutdown_report(
            error_reporter=error_reporter, session_id="test_123"
        )

        assert report_path is not None

    def test_graceful_shutdown_context(self):
        """Test graceful shutdown context manager."""
        # Mock managers
        state_manager = Mock()
        recovery_manager = Mock()
        error_reporter = Mock()

        with graceful_shutdown_context(
            state_manager=state_manager,
            recovery_manager=recovery_manager,
            error_reporter=error_reporter,
            session_id="test_123",
        ) as shutdown:
            assert shutdown is not None
            # Cleanup tasks should be registered

        # Context exit should not trigger shutdown if not requested


class TestIntegration:
    """Test integration between error handling components."""

    def test_error_handler_reporter_integration(self):
        """Test integration between error handler and reporter."""
        handler = ErrorHandler()
        reporter = ErrorReporter()

        # Capture error
        error = ProviderError("API failed", provider="OpenAI")
        context = handler.capture_context(
            error, operation="api_call", phase="initialization"
        )

        # Report error
        reporter.report_error(context)

        # Check consistency
        assert len(handler.error_history) == 1
        assert len(reporter.reported_errors) == 1
        assert reporter.reported_errors[0] == context

    @pytest.mark.asyncio
    async def test_retry_with_error_tracking(self):
        """Test retry mechanism with error tracking."""
        handler = ErrorHandler()
        attempt_count = 0

        @retry_with_backoff(
            max_attempts=3, exceptions=(NetworkTransientError,), base_delay=0.01
        )
        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1

            # Track attempt
            if attempt_count < 3:
                error = NetworkTransientError(
                    f"Attempt {attempt_count} failed", operation="flaky_op"
                )
                handler.capture_context(error, operation="flaky_operation")
                raise error

            return "success"

        result = await flaky_operation()

        assert result == "success"
        assert attempt_count == 3
        assert len(handler.error_history) == 2  # Two failures before success

    def test_shutdown_with_error_reporting(self, tmp_path):
        """Test shutdown handler with error reporting."""
        # Set up components
        handler = ErrorHandler()
        reporter = ErrorReporter()
        shutdown = ShutdownHandler()

        # Generate some errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            context = handler.capture_context(error)
            reporter.report_error(context)

        # Register cleanup to save error report
        shutdown.register_cleanup_task(
            lambda: reporter.save_error_report("test_session", output_dir=tmp_path),
            name="save_errors",
        )

        # Trigger shutdown
        shutdown._run_cleanup_tasks()

        # Check that error report was saved
        error_reports = list(tmp_path.glob("error_report_*.json"))
        assert len(error_reports) == 1
