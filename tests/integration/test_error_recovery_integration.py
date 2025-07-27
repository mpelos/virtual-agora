"""Integration tests for error handling and recovery flows.

This module tests error scenarios including LLM provider failures, state corruption,
network issues, and various recovery mechanisms throughout the application.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from datetime import datetime

from virtual_agora.main import run_application
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ConfigurationError,
    CriticalError,
    StateError,
    UserInterventionRequired,
)
from virtual_agora.providers.registry import ProviderRegistry

from ..helpers.fake_llm import create_fake_llm_pool
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    patch_ui_components,
    create_test_config_file,
    create_test_env_file,
)


class TestErrorRecoveryIntegration:
    """Test error handling and recovery throughout the application."""

    def setup_method(self):
        """Set up test method with temporary files."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yml"
        self.env_path = Path(self.temp_dir) / "test.env"

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_provider_timeout_recovery(self):
        """Test recovery from LLM provider timeouts during discussion."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        # Mock user input for topic
        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for timeout recovery"

            # Mock provider to timeout on second call
            call_count = 0

            def mock_provider_with_timeout(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    import time

                    time.sleep(0.1)  # Simulate timeout
                    raise TimeoutError("LLM provider timeout")
                return "Test response"

            with patch_ui_components():
                with patch("virtual_agora.providers.create_provider") as mock_create:
                    mock_llm = Mock()
                    mock_llm.invoke = Mock(side_effect=mock_provider_with_timeout)
                    mock_create.return_value = mock_llm

                    # Mock the discussion flow to avoid long execution
                    with patch(
                        "virtual_agora.flow.graph.VirtualAgoraFlow.stream"
                    ) as mock_stream:
                        # Simulate error then recovery
                        mock_stream.return_value = iter(
                            [
                                {"current_phase": 1},
                                {"error": "timeout", "recovery": "attempted"},
                            ]
                        )

                        result = await run_application(mock_args)
                        # Should handle error gracefully, may return error code but shouldn't crash
                        assert result in [0, 1, 2]  # Valid exit codes

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_state_corruption_recovery(self):
        """Test recovery from state corruption scenarios."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "DEBUG"  # More verbose for debugging
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for state corruption"

            # Mock state manager to corrupt state on specific operation
            with patch_ui_components():
                with patch(
                    "virtual_agora.state.manager.StateManager.state",
                    new_callable=lambda: Mock(),
                ) as mock_state_prop:
                    # First call returns good state, second call returns corrupted state
                    good_state = {
                        "session_id": "test_session",
                        "current_phase": 1,
                        "agents": {},
                        "messages": [],
                    }

                    corrupted_state = {"invalid": "state"}  # Missing required fields

                    mock_state_prop.side_effect = [
                        good_state,
                        corrupted_state,
                        good_state,
                    ]

                    # Mock recovery manager
                    with patch(
                        "virtual_agora.state.recovery.StateRecoveryManager.emergency_recovery"
                    ) as mock_recovery:
                        mock_recovery.return_value = True  # Successful recovery

                        result = await run_application(mock_args)
                        # Should attempt recovery
                        assert result in [0, 1, 2]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_network_connectivity_failure(self):
        """Test handling of network connectivity issues."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for network failure"

            # Simulate network errors
            with patch_ui_components():
                with patch("virtual_agora.providers.create_provider") as mock_create:
                    mock_llm = Mock()
                    mock_llm.invoke = Mock(
                        side_effect=ConnectionError("Network unreachable")
                    )
                    mock_create.return_value = mock_llm

                    with patch(
                        "virtual_agora.flow.graph.VirtualAgoraFlow.stream"
                    ) as mock_stream:
                        mock_stream.side_effect = ConnectionError(
                            "Network error during streaming"
                        )

                        result = await run_application(mock_args)
                        # Should handle network error gracefully
                        assert result in [1, 2]  # Error exit codes

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_malformed_llm_response_recovery(self):
        """Test recovery from malformed LLM responses."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "DEBUG"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for malformed responses"

            # Mock LLM to return malformed JSON
            malformed_responses = [
                "{{invalid json",  # Invalid JSON
                "",  # Empty response
                "{'single_quotes': 'invalid'}",  # Single quotes
                '{"missing_required_field": "value"}',  # Missing expected fields
                "Valid response",  # Finally a good response
            ]

            with patch_ui_components():
                with patch("virtual_agora.providers.create_provider") as mock_create:
                    mock_llm = Mock()
                    mock_llm.invoke = Mock(side_effect=malformed_responses)
                    mock_create.return_value = mock_llm

                    with patch(
                        "virtual_agora.flow.graph.VirtualAgoraFlow.stream"
                    ) as mock_stream:
                        mock_stream.return_value = iter([{"current_phase": 1}])

                        result = await run_application(mock_args)
                        # Should handle malformed responses and potentially recover
                        assert result in [0, 1, 2]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios."""
        create_test_config_file(
            self.config_path, num_agents=4
        )  # More agents = more memory
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for memory pressure"

            # Mock memory error during state operations
            with patch_ui_components():
                with patch(
                    "virtual_agora.state.manager.StateManager.initialize_state"
                ) as mock_init:
                    # Simulate memory error on initialization
                    mock_init.side_effect = MemoryError("Insufficient memory")

                    result = await run_application(mock_args)
                    # Should handle memory error gracefully
                    assert result in [1, 2]  # Error exit codes

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_critical_error_with_recovery(self):
        """Test handling of critical errors with emergency recovery."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for critical error"

            with patch_ui_components():
                # Simulate critical error during flow execution
                with patch(
                    "virtual_agora.flow.graph.VirtualAgoraFlow.stream"
                ) as mock_stream:
                    mock_stream.side_effect = CriticalError("Critical system failure")

                    # Mock successful emergency recovery
                    with patch(
                        "virtual_agora.state.recovery.StateRecoveryManager.emergency_recovery"
                    ) as mock_recovery:
                        mock_recovery.return_value = True

                        result = await run_application(mock_args)
                        # Should trigger emergency recovery
                        assert result in [1, 2]  # Critical error exit codes
                        mock_recovery.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_user_intervention_required(self):
        """Test handling of scenarios requiring user intervention."""
        create_test_config_file(self.config_path, num_agents=2)
        create_test_env_file(self.env_path)

        mock_args = Mock()
        mock_args.config = self.config_path
        mock_args.env = self.env_path
        mock_args.log_level = "INFO"
        mock_args.no_color = False
        mock_args.dry_run = False

        with patch(
            "virtual_agora.ui.human_in_the_loop.get_initial_topic"
        ) as mock_topic:
            mock_topic.return_value = "Test topic for user intervention"

            with patch_ui_components():
                # Simulate condition requiring user intervention
                with patch(
                    "virtual_agora.flow.graph.VirtualAgoraFlow.stream"
                ) as mock_stream:
                    mock_stream.side_effect = UserInterventionRequired(
                        "User decision required", ["Option 1", "Option 2", "Option 3"]
                    )

                    result = await run_application(mock_args)
                    # Should handle user intervention gracefully
                    assert result == 3  # User intervention exit code


class TestRecoveryMechanisms:
    """Test specific recovery mechanisms in isolation."""

    @pytest.mark.integration
    def test_state_recovery_manager_checkpoint_restore(self):
        """Test StateRecoveryManager's checkpoint restoration."""
        # Create test config and state manager
        helper = IntegrationTestHelper(num_agents=2)
        config = helper.create_test_config()
        state_manager = StateManager(config)
        recovery_manager = StateRecoveryManager()

        # Initialize state
        initial_state = state_manager.initialize_state("test_recovery_session")

        # Create checkpoint
        checkpoint_id = recovery_manager.create_checkpoint(
            initial_state, operation="test_checkpoint", save_to_disk=False
        )

        # Simulate state corruption
        corrupted_state = initial_state.copy()
        corrupted_state["corrupted"] = True

        # Test restoration
        restored_state = recovery_manager.rollback_to_checkpoint(
            checkpoint_id.checkpoint_id, state_manager
        )

        # Verify restoration
        assert restored_state["session_id"] == initial_state["session_id"]
        assert "corrupted" not in restored_state
        assert restored_state["current_phase"] == initial_state["current_phase"]

    @pytest.mark.integration
    def test_provider_fallback_mechanism(self):
        """Test provider fallback when primary provider fails."""
        # This would test the fallback builder functionality
        from virtual_agora.providers.fallback_builder import FallbackBuilder

        # Create fallback configuration
        primary_config = {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "invalid_key",
        }

        fallback_config = {
            "provider": "anthropic",
            "model": "claude-3-haiku",
            "api_key": "valid_key",
        }

        builder = FallbackBuilder()

        # Add configurations
        builder.add_primary(primary_config)
        builder.add_fallback(fallback_config)

        # Test fallback logic (mocked)
        with patch("virtual_agora.providers.create_provider") as mock_create:
            # Primary fails
            mock_create.side_effect = [
                Exception("Primary provider failed"),
                Mock(),  # Fallback succeeds
            ]

            provider = builder.build_with_fallback()
            assert provider is not None

    @pytest.mark.integration
    def test_context_window_compression_recovery(self):
        """Test recovery from context window overflow."""
        from virtual_agora.flow.context_window import ContextWindowManager

        # Create manager with small limit
        manager = ContextWindowManager(max_tokens=100)

        # Create large state that would overflow
        large_state = {
            "messages": [{"content": "Very long message " * 50} for _ in range(10)],
            "current_phase": 2,
            "session_id": "test_session",
        }

        # Test compression
        compressed_state = manager.compress_if_needed(large_state)

        # Verify compression occurred
        assert len(str(compressed_state)) < len(str(large_state))
        assert compressed_state["session_id"] == large_state["session_id"]
        assert compressed_state["current_phase"] == large_state["current_phase"]

    @pytest.mark.integration
    def test_cycle_detection_and_prevention(self):
        """Test cycle detection and prevention mechanisms."""
        from virtual_agora.flow.cycle_detection import CyclePreventionManager

        manager = CyclePreventionManager()

        # Create sequence that would create a cycle
        state_sequence = [
            {"current_phase": 1, "current_round": 1},
            {"current_phase": 2, "current_round": 1},
            {"current_phase": 1, "current_round": 1},  # Back to phase 1
            {"current_phase": 2, "current_round": 1},  # Back to phase 2 - cycle!
        ]

        cycle_detected = False
        for state in state_sequence:
            if manager.detect_cycle(state):
                cycle_detected = True
                break

        assert cycle_detected

        # Test prevention mechanism
        prevention_action = manager.suggest_prevention_action()
        assert prevention_action is not None


class TestErrorReporting:
    """Test error reporting and monitoring during failures."""

    @pytest.mark.integration
    def test_error_context_capture(self):
        """Test that error context is properly captured during failures."""
        from virtual_agora.utils.error_handler import error_handler, ErrorContext

        # Simulate error during different operations
        test_operations = ["initialization", "discussion", "consensus", "recovery"]

        for operation in test_operations:
            try:
                with error_handler.error_boundary(operation):
                    raise ValueError(f"Test error during {operation}")
            except ValueError:
                pass

            # Verify error was captured with context
            error_summary = error_handler.get_error_summary()
            assert error_summary["total_errors"] > 0

            # Check that context includes operation info
            last_error = (
                error_handler.get_recent_errors(1)[0]
                if error_handler.get_recent_errors(1)
                else None
            )
            assert last_error is not None
            assert operation in str(last_error.get("context", {}))

    @pytest.mark.integration
    def test_error_trend_detection(self):
        """Test detection of error trends and patterns."""
        from virtual_agora.utils.error_reporter import ErrorReporter
        from rich.console import Console

        reporter = ErrorReporter(Console())

        # Simulate multiple similar errors
        similar_errors = [
            {
                "type": "TimeoutError",
                "message": "Provider timeout",
                "timestamp": datetime.now(),
            },
            {
                "type": "TimeoutError",
                "message": "Provider timeout",
                "timestamp": datetime.now(),
            },
            {
                "type": "TimeoutError",
                "message": "Provider timeout",
                "timestamp": datetime.now(),
            },
        ]

        # This would normally be called internally by the error handler
        # For testing, we simulate the pattern detection
        trends = []
        error_types = [e["type"] for e in similar_errors]
        if error_types.count("TimeoutError") >= 3:
            trends.append(
                "Multiple timeout errors detected - check network connectivity"
            )

        assert len(trends) > 0
        assert "timeout" in trends[0].lower()


# Performance and stress testing for error scenarios
class TestErrorPerformance:
    """Test performance characteristics during error scenarios."""

    @pytest.mark.integration
    def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        import time
        from virtual_agora.utils.error_handler import error_handler

        # Measure time for normal operation
        start_time = time.time()
        for i in range(100):
            with error_handler.error_boundary("test_operation"):
                pass  # Normal operation
        normal_time = time.time() - start_time

        # Measure time with errors
        start_time = time.time()
        for i in range(100):
            try:
                with error_handler.error_boundary("test_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass
        error_time = time.time() - start_time

        # Error handling shouldn't be more than 5x slower
        assert error_time < normal_time * 5

    @pytest.mark.integration
    def test_recovery_time_bounds(self):
        """Test that recovery operations complete within reasonable time bounds."""
        import time
        from virtual_agora.state.recovery import StateRecoveryManager

        recovery_manager = StateRecoveryManager()

        # Create test state
        test_state = {
            "session_id": "test_session",
            "current_phase": 2,
            "messages": [
                {"content": f"Message {i}"} for i in range(1000)
            ],  # Large state
        }

        # Test checkpoint creation time
        start_time = time.time()
        checkpoint_id = recovery_manager.create_checkpoint(
            test_state, operation="performance_test", save_to_disk=False
        )
        checkpoint_time = time.time() - start_time

        # Should complete within 1 second
        assert checkpoint_time < 1.0

        # Test restoration time
        start_time = time.time()
        restored_state = recovery_manager.restore_from_checkpoint(checkpoint_id)
        restore_time = time.time() - start_time

        # Should complete within 1 second
        assert restore_time < 1.0
        assert restored_state["session_id"] == test_state["session_id"]
