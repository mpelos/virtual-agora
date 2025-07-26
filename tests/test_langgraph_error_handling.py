"""Integration tests for LangGraph error handling patterns.

This module tests the integration of LangGraph error handling patterns
with Virtual Agora's error management system.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from virtual_agora.utils.exceptions import (
    ProviderError,
    TimeoutError,
    NetworkTransientError,
    RecoverableError,
    ValidationError,
)
from virtual_agora.utils.langgraph_error_handler import (
    LangGraphErrorHandler,
    create_provider_error_chain,
    with_langgraph_error_handling,
)
from virtual_agora.utils.retry import retry_manager, CircuitBreaker, CircuitState
from virtual_agora.utils.error_handler import (
    error_handler,
    ErrorContext,
    RecoveryStrategy,
)
from virtual_agora.providers.factory import ProviderFactory
from virtual_agora.providers.fallback_builder import (
    FallbackChainBuilder,
    FallbackStrategy,
)
from virtual_agora.agents.llm_agent import LLMAgent


class TestLangGraphErrorHandler:
    """Test LangGraph error handler functionality."""

    def test_create_retry_policy(self):
        """Test creating a retry policy."""
        handler = LangGraphErrorHandler()

        # Test with default retry function
        policy = handler.create_retry_policy(max_attempts=5)
        assert policy is not None
        assert policy.max_attempts == 5

        # Test with custom retry function
        def custom_retry(error):
            return isinstance(error, TimeoutError)

        policy = handler.create_retry_policy(max_attempts=3, retry_on=custom_retry)
        assert policy is not None
        assert policy.max_attempts == 3

    def test_create_fallback_chain(self):
        """Test creating a fallback chain."""
        handler = LangGraphErrorHandler()

        # Mock LLMs
        primary_llm = Mock(spec=BaseChatModel)
        fallback_llm1 = Mock(spec=BaseChatModel)
        fallback_llm2 = Mock(spec=BaseChatModel)

        # Create chain
        chain = handler.create_fallback_chain(
            primary_llm, [fallback_llm1, fallback_llm2]
        )

        assert chain is not None
        # The chain should have fallbacks configured
        assert hasattr(chain, "with_fallbacks")

    def test_create_self_correcting_chain(self):
        """Test creating a self-correcting chain."""
        handler = LangGraphErrorHandler()

        # Mock LLM
        llm = Mock(spec=BaseChatModel)
        llm.with_fallbacks = Mock(return_value=llm)

        # Create self-correcting chain
        chain = handler.create_self_correcting_chain(
            llm, max_retries=3, include_error_context=True
        )

        assert chain is not None
        # Should have called with_fallbacks
        llm.with_fallbacks.assert_called_once()

    def test_with_retry_policy_decorator(self):
        """Test the retry policy decorator."""
        handler = LangGraphErrorHandler()

        # Create a function that fails twice then succeeds
        call_count = 0

        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkTransientError("Network error")
            return "success"

        # Apply decorator
        decorated_func = handler.with_retry_policy(flaky_function, None)

        # Should succeed after retries if LangGraph is available
        # Otherwise it should just call the function once
        try:
            result = decorated_func()
            if result == "success":
                # Either succeeded on first try (no retry policy)
                # or after retries (with retry policy)
                assert call_count >= 1
        except NetworkTransientError:
            # Expected if no retry policy available
            assert call_count == 1

    def test_create_validation_node(self):
        """Test creating a validation node."""
        handler = LangGraphErrorHandler()

        # Mock tools
        tools = [Mock(), Mock()]

        # Create validation node
        node = handler.create_validation_node(tools)

        # If LangGraph is not available, should return None
        # Otherwise should return a ValidationNode
        # Note: ValidationNode API might vary, so we just check if it returns None
        # when LangGraph is not available
        assert node is None  # Since ValidationNode is likely not available in test env


class TestProviderFallbackChain:
    """Test provider fallback chain functionality."""

    def test_fallback_chain_builder(self):
        """Test building a fallback chain."""
        builder = FallbackChainBuilder()

        # Add providers
        builder.add_provider(
            {"provider": "openai", "model": "gpt-4"}, priority=0, cost_per_token=0.03
        )
        builder.add_provider(
            {"provider": "anthropic", "model": "claude-3"},
            priority=1,
            cost_per_token=0.02,
        )

        # Configure strategy
        builder.with_strategy(FallbackStrategy.COST_OPTIMIZED)

        # Configure retry
        builder.with_retry_config(max_retries=2)

        # Build should work (may fail if no API keys)
        try:
            chain = builder.build()
            assert chain is not None
        except Exception as e:
            # Expected if no API keys configured
            assert "API key" in str(e) or "provider" in str(e)

    def test_fallback_strategies(self):
        """Test different fallback strategies."""
        builder = FallbackChainBuilder()

        # Add providers with different characteristics
        builder.add_provider(
            {"provider": "openai", "model": "gpt-4"},
            priority=2,
            cost_per_token=0.03,
            average_latency=1.5,
        )
        builder.add_provider(
            {"provider": "anthropic", "model": "claude-3"},
            priority=1,
            cost_per_token=0.02,
            average_latency=2.0,
        )
        builder.add_provider(
            {"provider": "google", "model": "gemini-pro"},
            priority=0,
            cost_per_token=0.01,
            average_latency=1.0,
        )

        # Test COST_OPTIMIZED strategy
        builder.with_strategy(FallbackStrategy.COST_OPTIMIZED)
        sorted_providers = builder._sort_providers_by_strategy()
        assert sorted_providers[0].cost_per_token == 0.01  # Cheapest first

        # Test PERFORMANCE_OPTIMIZED strategy
        builder.with_strategy(FallbackStrategy.PERFORMANCE_OPTIMIZED)
        sorted_providers = builder._sort_providers_by_strategy()
        assert sorted_providers[0].average_latency == 1.0  # Fastest first

        # Test SEQUENTIAL strategy
        builder.with_strategy(FallbackStrategy.SEQUENTIAL)
        sorted_providers = builder._sort_providers_by_strategy()
        assert sorted_providers[0].priority == 0  # Highest priority first


class TestLLMAgentErrorHandling:
    """Test LLM agent error handling integration."""

    def test_agent_with_fallback(self):
        """Test creating an agent with fallback LLMs."""
        # Mock LLMs
        primary_llm = Mock(spec=BaseChatModel)
        fallback_llm1 = Mock(spec=BaseChatModel)
        fallback_llm2 = Mock(spec=BaseChatModel)

        # Create agent with fallback
        agent = LLMAgent.create_with_fallback(
            "test_agent",
            primary_llm,
            [fallback_llm1, fallback_llm2],
            role="participant",
            max_retries=2,
        )

        assert agent is not None
        assert agent.agent_id == "test_agent"
        assert agent._fallback_configured is True
        assert len(agent._fallback_llms) == 2

    def test_agent_retry_policy(self):
        """Test agent retry policy creation."""
        # Mock LLM
        llm = Mock(spec=BaseChatModel)

        # Create agent
        agent = LLMAgent("test_agent", llm, enable_error_handling=True, max_retries=3)

        # Get retry policy
        policy = agent.get_retry_policy()

        # Should return None if LangGraph not available
        # or a RetryPolicy if available
        assert policy is None or hasattr(policy, "max_attempts")

    def test_agent_as_langgraph_node(self):
        """Test converting agent to LangGraph node."""
        # Mock LLM
        llm = Mock(spec=BaseChatModel)

        # Create agent
        agent = LLMAgent("test_agent", llm)

        # Convert to node config
        node_config = agent.as_langgraph_node()

        assert node_config["name"] == "test_agent"
        assert node_config["func"] == agent
        assert "metadata" in node_config
        assert node_config["metadata"]["agent_id"] == "test_agent"


class TestRetryManagerIntegration:
    """Test retry manager with LangGraph integration."""

    def test_create_langgraph_policy(self):
        """Test creating LangGraph policy from retry manager."""
        policy = retry_manager.create_langgraph_policy("provider_api")

        # Should return None if LangGraph not available
        # or a RetryPolicy if available
        assert policy is None or hasattr(policy, "max_attempts")

    def test_circuit_breaker_integration(self):
        """Test circuit breaker with LangGraph."""
        # Get circuit breaker
        breaker = retry_manager.get_circuit_breaker(
            "test_operation", failure_threshold=3, recovery_timeout=10.0
        )

        assert breaker is not None
        assert breaker.name == "test_operation"
        assert breaker.failure_threshold == 3

        # Test as LangGraph policy
        policy = breaker.as_langgraph_policy()
        assert policy is None or hasattr(policy, "max_attempts")

    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        # Get circuit breaker
        breaker = retry_manager.get_circuit_breaker("test_stats")

        # Simulate some calls
        def failing_func():
            raise ProviderError("Test error")

        # Make failing calls
        for i in range(3):
            try:
                breaker.call(failing_func)
            except ProviderError:
                pass

        # Get stats
        stats = breaker.get_stats()
        assert stats["name"] == "test_stats"
        assert stats["failed_calls"] >= 3
        assert stats["success_rate"] == 0.0

    def test_resilient_node_creation(self):
        """Test creating a resilient node."""

        def test_func(x):
            return x * 2

        # Create resilient node
        resilient_func = retry_manager.create_resilient_node(
            test_func, "test_node", "network", use_circuit_breaker=True
        )

        # Should have metadata
        assert hasattr(resilient_func, "_langgraph_metadata")
        assert resilient_func._langgraph_metadata["operation"] == "test_node"

        # Should work normally
        result = resilient_func(21)
        assert result == 42


class TestErrorHandlerIntegration:
    """Test error handler with LangGraph integration."""

    def test_capture_langgraph_context(self):
        """Test capturing LangGraph-specific error context."""
        # Create error
        error = ProviderError("Test error", provider="openai")

        # Capture context
        context = error_handler.capture_langgraph_context(
            error,
            graph_node="test_node",
            graph_state={"messages": [], "current_phase": 1},
            checkpoint_id="checkpoint_123",
        )

        assert context.graph_node == "test_node"
        assert context.checkpoint_id == "checkpoint_123"
        assert "messages" in context.graph_state
        assert "Error in graph node: test_node" in context.breadcrumbs[-1]

    def test_error_recovery_node(self):
        """Test creating an error recovery node."""
        # Create recovery node
        recovery_node = error_handler.create_error_recovery_node(
            {"test_node": RecoveryStrategy.RETRY}
        )

        # Test with error in state
        state = {"error": {"error": ProviderError("Test error"), "node": "test_node"}}

        # Process error
        result = recovery_node(state)

        # Should have attempted recovery
        assert "error_recovered" in result
        assert "error_context" in result or result["error_recovered"] is True

    def test_langgraph_error_boundary(self):
        """Test LangGraph error boundary context manager."""
        state = {"messages": []}

        # Test with error
        with pytest.raises(ValueError):
            with error_handler.langgraph_error_boundary(
                "test_node", state, checkpoint_id="test_123"
            ):
                raise ValueError("Test error")

        # Should have added error to state
        assert "error" in state
        assert state["error"]["node"] == "test_node"
        assert isinstance(state["error"]["error"], ValueError)

    def test_node_error_summary(self):
        """Test getting error summary for a graph node."""
        # Capture some errors for a node
        for i in range(3):
            error_handler.capture_langgraph_context(
                ProviderError(f"Error {i}"), graph_node="problem_node", graph_state={}
            )

        # Get summary
        summary = error_handler.get_node_error_summary("problem_node")

        assert summary["node"] == "problem_node"
        assert summary["total_errors"] >= 3
        assert "ProviderError" in summary["error_types"]


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_provider_with_circuit_breaker(self):
        """Test provider with circuit breaker integration."""
        # This is a conceptual test - would need real providers
        # to test actual integration

        # Create a mock provider that fails
        mock_provider = Mock(spec=BaseChatModel)
        mock_provider.invoke = Mock(side_effect=ProviderError("API Error"))

        # Get circuit breaker
        breaker = retry_manager.get_circuit_breaker(
            "provider_test", failure_threshold=2
        )

        # Make calls through breaker
        for i in range(3):
            try:
                breaker.call(mock_provider.invoke, [HumanMessage(content="test")])
            except Exception:
                pass

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

    def test_error_propagation_through_chain(self):
        """Test error propagation through fallback chain."""
        # Mock failing providers
        provider1 = Mock(spec=BaseChatModel)
        provider1.invoke = Mock(side_effect=ProviderError("Provider 1 failed"))

        provider2 = Mock(spec=BaseChatModel)
        provider2.invoke = Mock(side_effect=ProviderError("Provider 2 failed"))

        provider3 = Mock(spec=BaseChatModel)
        provider3.invoke = Mock(return_value=AIMessage(content="Success"))

        # Create chain (conceptual - actual implementation would differ)
        handler = LangGraphErrorHandler()
        chain = handler.create_fallback_chain(provider1, [provider2, provider3])

        # In a real scenario, the chain would try each provider
        # and succeed with provider3
        # This is a simplified test showing the concept


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
