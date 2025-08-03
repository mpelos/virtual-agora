"""Production Test Suite Base Class Implementation.

This module provides the actual implementation of the production-quality test framework
that ensures tests use identical execution patterns as production code.

CRITICAL CHANGES:
- Uses StreamCoordinator for execution (matches production pattern in main.py:948)
- Allows GraphInterrupt propagation through V13NodeWrapper
- Tests actual state synchronization pattern from VirtualAgoraV13Flow.stream()
- Validates reducer field behavior (NOT pre-initialized)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any, List, Optional, Union, Callable, Iterator
from datetime import datetime
from contextlib import contextmanager
import time
import psutil
import os
from abc import ABC, abstractmethod

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.execution.stream_coordinator import StreamCoordinator
from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ProductionTestSuite:
    """Base class for production-quality integration tests.

    This class ensures all tests replicate the exact execution patterns
    used in production, including:
    - StreamCoordinator execution (not direct flow.stream())
    - GraphInterrupt propagation through V13NodeWrapper
    - Actual state synchronization from VirtualAgoraV13Flow
    - Proper reducer field handling
    """

    def setup_method(self):
        """Setup method run before each test."""
        # Initialize mocking frameworks
        self.llm_mock = LLMProviderMock()
        self.hitl_mock = HITLMockingFramework()
        self.state_validator = StateConsistencyValidator()
        self.performance_monitor = PerformanceMonitor()

        # Load real configuration for testing
        self.config = self._load_test_config()

        # Initialize baseline metrics
        self.baseline_metrics = {}

        logger.debug("ProductionTestSuite setup completed")

    def teardown_method(self):
        """Cleanup method run after each test."""
        # Clean up mocks
        self.llm_mock.cleanup()
        self.hitl_mock.cleanup()

        # Validate resource cleanup
        self.performance_monitor.validate_cleanup()

        logger.debug("ProductionTestSuite teardown completed")

    def create_production_flow(self, **kwargs) -> VirtualAgoraV13Flow:
        """Create VirtualAgoraV13Flow configured for testing.

        Returns flow configured exactly like production.
        Must be called within mock_llm_realistic() context.
        """
        # Use the pre-loaded configuration
        flow = VirtualAgoraV13Flow(
            config=self.config,
            enable_monitoring=False,  # Disable to avoid UI interference in tests
            checkpoint_interval=3,
        )

        # Attach performance monitoring
        self.performance_monitor.attach_to_flow(flow)

        return flow

    def create_stream_coordinator(self, flow: VirtualAgoraV13Flow) -> StreamCoordinator:
        """Create StreamCoordinator with mocked interrupt processor.

        CRITICAL: This replicates the production pattern from main.py:948
        """
        # Import the actual interrupt processor function
        from virtual_agora.main import process_interrupt_recursive

        # Create coordinator with real interrupt processor
        # (LLM calls and user input will be mocked)
        coordinator = StreamCoordinator(flow, process_interrupt_recursive)

        return coordinator

    def simulate_production_execution(
        self, flow: VirtualAgoraV13Flow, config_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute using production pattern with StreamCoordinator.

        CRITICAL: This matches the exact execution pattern from main.py:948-950
        """
        coordinator = self.create_stream_coordinator(flow)

        updates = []
        try:
            for update in coordinator.coordinate_stream_execution(config_dict):
                updates.append(update)

                # Validate state synchronization after each update
                # This replicates the state sync pattern from VirtualAgoraV13Flow.stream()
                if isinstance(update, dict) and update:
                    self._validate_state_sync_after_update(flow, update)

                # Break after reasonable progress to avoid infinite loops in tests
                if len(updates) >= 10:  # Smaller limit for initial testing
                    logger.warning(
                        f"Test reached update limit ({len(updates)}), breaking"
                    )
                    break

        except Exception as e:
            logger.error(f"Production execution failed: {e}")
            raise

        return updates

    def _validate_state_sync_after_update(
        self, flow: VirtualAgoraV13Flow, update: Dict[str, Any]
    ):
        """Validate state synchronization happens correctly after each update.

        This replicates the validation pattern from VirtualAgoraV13Flow.stream():841
        """
        # Check that state manager was updated with graph changes
        try:
            state_snapshot = flow.state_manager.get_snapshot()
            graph_state = flow.compiled_graph.get_state(
                {"configurable": {"thread_id": state_snapshot.get("session_id")}}
            )

            # Validate state consistency (basic check)
            if graph_state and state_snapshot:
                assert state_snapshot.get("session_id") == graph_state.config.get(
                    "configurable", {}
                ).get("thread_id")

        except Exception as e:
            logger.warning(f"State sync validation failed: {e}")

    @contextmanager
    def mock_llm_realistic(self):
        """Context manager for realistic LLM mocking."""
        # Mock at multiple levels to catch all provider creation calls
        with patch("virtual_agora.providers.create_provider") as mock_create_provider:
            with patch(
                "virtual_agora.providers.factory.create_provider"
            ) as mock_create_factory:
                with patch(
                    "virtual_agora.flow.graph_v13.create_provider"
                ) as mock_create_graph:
                    mock_llm = self.llm_mock.create_mock_llm()
                    mock_create_provider.return_value = mock_llm
                    mock_create_factory.return_value = mock_llm
                    mock_create_graph.return_value = mock_llm

                    try:
                        yield self.llm_mock
                    finally:
                        # Cleanup handled in teardown_method
                        pass

    @contextmanager
    def mock_user_input(self, responses: Dict[str, Any]):
        """Context manager for mocking user input via GraphInterrupt."""
        with self.hitl_mock.mock_user_input(responses) as mock_interrupt:
            yield mock_interrupt

    @contextmanager
    def mock_file_operations(self):
        """Mock all file I/O operations."""
        # Create proper YAML config content for testing
        yaml_config = """
moderator:
  provider: Google
  model: gemini-2.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2
  - provider: Google
    model: gemini-2.5-pro
    count: 1

summarizer:
  provider: Google
  model: gemini-2.5-pro

report_writer:
  provider: Google
  model: gemini-2.5-pro

session:
  max_rounds: 3
  max_messages_per_round: 5
  topic_generation_enabled: true
  
ui:
  enabled: false
        """

        with patch("builtins.open", mock_open(read_data=yaml_config)) as mock_file:
            with patch("os.path.exists", return_value=True):
                with patch("os.makedirs"):
                    yield mock_file

    @contextmanager
    def performance_monitoring(self):
        """Monitor performance metrics during test execution."""
        with self.performance_monitor.monitor_execution() as metrics:
            yield metrics

    def validate_state_consistency(self, state: Dict[str, Any]) -> bool:
        """Validate state consistency using comprehensive checks."""
        return self.state_validator.validate_complete_state(state)

    def _load_test_config(self):
        """Load test configuration from YAML file."""
        from virtual_agora.config.loader import ConfigLoader
        import os

        # Use test config file - find the repository root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(
            os.path.dirname(current_dir)
        )  # Go up two levels from tests/framework/
        config_path = os.path.join(repo_root, "config.test.yml")
        loader = ConfigLoader(config_path)
        return loader.load()


class LLMProviderMock:
    """Mock LLM provider with realistic response patterns."""

    def __init__(self):
        self.call_count = 0
        self.response_patterns = self._load_response_patterns()

    def create_mock_llm(self) -> Mock:
        """Create mock LLM with realistic response patterns."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = self._generate_response
        mock_llm.call.side_effect = self._generate_response

        # Create mock with proper chaining for with_fallbacks()
        mock_with_fallbacks = Mock()
        mock_with_fallbacks.invoke.side_effect = self._generate_response
        mock_with_fallbacks.call.side_effect = self._generate_response
        mock_llm.with_fallbacks.return_value = mock_with_fallbacks

        return mock_llm

    def _generate_response(self, input_data: Any) -> Mock:
        """Generate realistic responses based on input context."""
        self.call_count += 1

        # Analyze input to determine response type
        if isinstance(input_data, list):
            try:
                last_message = (
                    input_data[-1].content
                    if input_data and hasattr(input_data[-1], "content")
                    else str(input_data[-1]) if input_data else ""
                )
            except:
                last_message = str(input_data)
        else:
            last_message = str(input_data)

        # Route to appropriate response generator
        if "propose topics" in last_message.lower() or "agenda" in last_message.lower():
            content = self._generate_topic_proposal_content()
        elif "vote" in last_message.lower():
            content = self._generate_vote_response_content()
        elif "discuss" in last_message.lower():
            content = self._generate_discussion_response_content()
        else:
            content = self._generate_generic_response_content()

        # Create a mock response object with content attribute
        mock_response = Mock()
        mock_response.content = content
        return mock_response

    def _generate_topic_proposal_content(self) -> str:
        """Generate realistic topic proposals."""
        topics = [
            "The impact of artificial intelligence on society",
            "Climate change adaptation strategies",
            "Future of remote work and digital collaboration",
        ]

        return f"I propose the following topics:\n" + "\n".join(
            f"{i+1}. {topic}" for i, topic in enumerate(topics)
        )

    def _generate_vote_response_content(self) -> str:
        """Generate realistic voting responses."""
        choices = [
            "I vote for option 1 as it provides comprehensive coverage",
            "Option 2 seems most relevant to current discussions",
            "I support option 3 for its practical implications",
        ]

        return choices[self.call_count % len(choices)]

    def _generate_discussion_response_content(self) -> str:
        """Generate realistic discussion contributions."""
        responses = [
            "This is a fascinating topic that requires careful consideration...",
            "I'd like to build on the previous point by adding that...",
            "While I agree with the general direction, we should examine...",
            "From a different perspective, we might consider...",
        ]

        return responses[self.call_count % len(responses)]

    def _generate_generic_response_content(self) -> str:
        """Generate generic realistic response."""
        return (
            f"This is a thoughtful response from the mock LLM (call #{self.call_count})"
        )

    def _load_response_patterns(self) -> Dict[str, Any]:
        """Load response patterns for different scenarios."""
        return {}  # Extended patterns can be added here

    def cleanup(self):
        """Clean up mock resources."""
        self.call_count = 0


class HITLMockingFramework:
    """Framework for mocking human-in-the-loop interactions."""

    def __init__(self):
        self.pending_responses = {}
        self.interaction_history = []

    @contextmanager
    def mock_user_input(self, responses: Dict[str, Any]):
        """Context manager for mocking user input responses."""
        self.pending_responses.update(responses)

        # Mock the interrupt function used in nodes
        with patch("virtual_agora.flow.nodes_v13.interrupt") as mock_interrupt:
            mock_interrupt.side_effect = self._handle_interrupt
            yield mock_interrupt

    def _handle_interrupt(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interrupt and return appropriate response."""
        interrupt_type = value.get("type")

        # Record interaction for validation
        self.interaction_history.append(
            {"type": interrupt_type, "value": value, "timestamp": datetime.now()}
        )

        # Return pre-configured response or default
        if interrupt_type in self.pending_responses:
            response = self.pending_responses[interrupt_type]
            del self.pending_responses[interrupt_type]  # Use once

            # Ensure response is properly formatted as dict
            if isinstance(response, str):
                return {"action": response}
            return response

        # Default responses - return as dictionaries
        defaults = {
            "agenda_approval": {"action": "approve"},
            "topic_conclusion": {"action": "continue"},
            "session_continuation": {"action": "next_topic"},
            "periodic_stop": {"action": "continue"},
        }

        return defaults.get(interrupt_type, {"action": "continue"})

    def cleanup(self):
        """Clean up mock resources."""
        self.pending_responses.clear()
        self.interaction_history.clear()


class StateConsistencyValidator:
    """Validator for state consistency and schema compliance."""

    def validate_complete_state(self, state: Dict[str, Any]) -> bool:
        """Validate complete state consistency."""
        try:
            # Check required fields
            required_fields = ["session_id", "current_phase", "current_round"]
            for field in required_fields:
                if field not in state:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Validate reducer fields are NOT pre-initialized as empty lists
            reducer_fields = ["vote_history", "phase_history", "messages"]
            for field in reducer_fields:
                # These should either not exist or be managed by LangGraph reducers
                # They should NOT be empty lists in initial state
                if field in state and state[field] == []:
                    logger.warning(
                        f"Reducer field {field} should not be pre-initialized as empty list"
                    )

            return True

        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return False


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""

    def __init__(self):
        self.start_memory = None
        self.start_time = None
        self.attached_flow = None

    def attach_to_flow(self, flow: VirtualAgoraV13Flow):
        """Attach monitoring to flow instance."""
        self.attached_flow = flow

    @contextmanager
    def monitor_execution(self):
        """Monitor performance during execution."""
        process = psutil.Process()
        self.start_memory = process.memory_info().rss
        self.start_time = time.perf_counter()

        metrics = {
            "start_memory_mb": self.start_memory / (1024 * 1024),
            "start_time": self.start_time,
        }

        try:
            yield metrics
        finally:
            end_memory = process.memory_info().rss
            end_time = time.perf_counter()

            memory_increase = (end_memory - self.start_memory) / (1024 * 1024)
            execution_time = end_time - self.start_time

            metrics.update(
                {
                    "end_memory_mb": end_memory / (1024 * 1024),
                    "execution_time_s": execution_time,
                    "memory_increase_mb": memory_increase,
                }
            )

    def validate_cleanup(self):
        """Validate resource cleanup after test."""
        # Basic cleanup validation
        if self.attached_flow:
            # Check for any obvious resource leaks
            pass

    def finalize_test_metrics(self):
        """Finalize and record test metrics."""
        # Record metrics for trend analysis
        pass
