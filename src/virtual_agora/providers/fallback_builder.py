"""Provider Fallback Chain Builder for Virtual Agora.

This module provides utilities to build complex fallback chains for LLM providers
using LangGraph patterns and error handling strategies.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from virtual_agora.providers.config import ProviderConfig, ProviderType
from virtual_agora.providers.factory import ProviderFactory
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import (
    ConfigurationError,
    ProviderError,
    TimeoutError,
    NetworkTransientError,
)

# Import LangGraph components if available
try:
    from langgraph.pregel import RetryPolicy
    from virtual_agora.utils.langgraph_error_handler import (
        LangGraphErrorHandler,
        create_provider_error_chain,
    )

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    RetryPolicy = None
    LangGraphErrorHandler = None
    create_provider_error_chain = None


logger = get_logger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies for provider chains."""

    SEQUENTIAL = "sequential"  # Try providers in order
    COST_OPTIMIZED = "cost_optimized"  # Start with cheapest
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Start with fastest
    LOAD_BALANCED = "load_balanced"  # Distribute load across providers
    CAPABILITY_BASED = "capability_based"  # Match provider to task


@dataclass
class ProviderPriority:
    """Provider with priority and constraints."""

    config: Union[ProviderConfig, Dict[str, Any]]
    priority: int = 0  # Lower is higher priority
    cost_per_token: Optional[float] = None
    average_latency: Optional[float] = None
    capabilities: List[str] = field(default_factory=list)
    max_requests_per_minute: Optional[int] = None
    current_load: float = 0.0  # Current load percentage


class FallbackChainBuilder:
    """Builder for creating sophisticated provider fallback chains."""

    def __init__(self):
        """Initialize the fallback chain builder."""
        self.providers: List[ProviderPriority] = []
        self.strategy = FallbackStrategy.SEQUENTIAL
        self.retry_config: Dict[str, Any] = {
            "max_retries": 3,
            "retry_on_errors": [ProviderError, TimeoutError, NetworkTransientError],
        }
        self.circuit_breaker_config: Dict[str, Any] = {
            "enabled": True,
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
        }
        self.error_handler: Optional[Callable[[Exception], Dict[str, Any]]] = None
        self._factory = ProviderFactory()

    def add_provider(
        self,
        config: Union[ProviderConfig, Dict[str, Any]],
        priority: int = 0,
        cost_per_token: Optional[float] = None,
        average_latency: Optional[float] = None,
        capabilities: Optional[List[str]] = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> "FallbackChainBuilder":
        """Add a provider to the fallback chain.

        Args:
            config: Provider configuration
            priority: Priority (lower is higher priority)
            cost_per_token: Cost per token for this provider
            average_latency: Average response latency in seconds
            capabilities: List of capabilities (e.g., "streaming", "function_calling")
            max_requests_per_minute: Rate limit for this provider

        Returns:
            Self for method chaining
        """
        provider_priority = ProviderPriority(
            config=config,
            priority=priority,
            cost_per_token=cost_per_token,
            average_latency=average_latency,
            capabilities=capabilities or [],
            max_requests_per_minute=max_requests_per_minute,
        )
        self.providers.append(provider_priority)

        logger.debug(f"Added provider to fallback chain: {config}")
        return self

    def with_strategy(self, strategy: FallbackStrategy) -> "FallbackChainBuilder":
        """Set the fallback strategy.

        Args:
            strategy: Fallback strategy to use

        Returns:
            Self for method chaining
        """
        self.strategy = strategy
        logger.debug(f"Set fallback strategy: {strategy.value}")
        return self

    def with_retry_config(
        self,
        max_retries: int = 3,
        retry_on_errors: Optional[List[Type[Exception]]] = None,
    ) -> "FallbackChainBuilder":
        """Configure retry behavior.

        Args:
            max_retries: Maximum retries per provider
            retry_on_errors: Exception types to retry on

        Returns:
            Self for method chaining
        """
        self.retry_config["max_retries"] = max_retries
        if retry_on_errors:
            self.retry_config["retry_on_errors"] = retry_on_errors
        return self

    def with_circuit_breaker(
        self,
        enabled: bool = True,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> "FallbackChainBuilder":
        """Configure circuit breaker.

        Args:
            enabled: Whether to use circuit breaker
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before retrying

        Returns:
            Self for method chaining
        """
        self.circuit_breaker_config = {
            "enabled": enabled,
            "failure_threshold": failure_threshold,
            "recovery_timeout": recovery_timeout,
        }
        return self

    def with_error_handler(
        self, handler: Callable[[Exception], Dict[str, Any]]
    ) -> "FallbackChainBuilder":
        """Set custom error handler.

        Args:
            handler: Function to handle errors and return context

        Returns:
            Self for method chaining
        """
        self.error_handler = handler
        return self

    def _sort_providers_by_strategy(self) -> List[ProviderPriority]:
        """Sort providers based on the selected strategy.

        Returns:
            Sorted list of providers
        """
        if self.strategy == FallbackStrategy.SEQUENTIAL:
            # Sort by priority
            return sorted(self.providers, key=lambda p: p.priority)

        elif self.strategy == FallbackStrategy.COST_OPTIMIZED:
            # Sort by cost (cheapest first), with None values last
            return sorted(
                self.providers,
                key=lambda p: (
                    p.cost_per_token is None,
                    p.cost_per_token or float("inf"),
                ),
            )

        elif self.strategy == FallbackStrategy.PERFORMANCE_OPTIMIZED:
            # Sort by latency (fastest first), with None values last
            return sorted(
                self.providers,
                key=lambda p: (
                    p.average_latency is None,
                    p.average_latency or float("inf"),
                ),
            )

        elif self.strategy == FallbackStrategy.LOAD_BALANCED:
            # Sort by current load (least loaded first)
            return sorted(self.providers, key=lambda p: p.current_load)

        elif self.strategy == FallbackStrategy.CAPABILITY_BASED:
            # This requires matching capabilities to requirements
            # For now, just use priority
            return sorted(self.providers, key=lambda p: p.priority)

        else:
            return self.providers

    def build(self) -> BaseChatModel:
        """Build the fallback chain.

        Returns:
            LLM with fallback chain configured

        Raises:
            ConfigurationError: If no providers configured
        """
        if not self.providers:
            raise ConfigurationError("No providers configured for fallback chain")

        # Sort providers by strategy
        sorted_providers = self._sort_providers_by_strategy()

        # Create provider instances
        provider_instances = []
        for provider_priority in sorted_providers:
            try:
                instance = self._factory.create_provider(
                    provider_priority.config, use_cache=True
                )
                provider_instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to create provider: {e}")
                continue

        if not provider_instances:
            raise ConfigurationError("Failed to create any providers")

        # If LangGraph is available, use enhanced error handling
        if LANGGRAPH_AVAILABLE and create_provider_error_chain:
            logger.info("Building fallback chain with LangGraph error handling")

            # Create error-resilient chain
            chain = create_provider_error_chain(
                provider_instances,
                max_retries_per_provider=self.retry_config["max_retries"],
            )

            # Add custom error handler if provided
            if self.error_handler and LangGraphErrorHandler:
                handler = LangGraphErrorHandler()
                chain = handler.create_fallback_chain(
                    chain,
                    [],  # No additional fallbacks
                    error_handler=self.error_handler,
                )

            return chain

        else:
            # Use basic LangChain fallback pattern
            logger.info("Building fallback chain with basic error handling")

            primary = provider_instances[0]
            fallbacks = provider_instances[1:]

            if fallbacks:
                return self._factory.create_provider_with_fallbacks(
                    primary_config=sorted_providers[0].config,
                    fallback_configs=[p.config for p in sorted_providers[1:]],
                    **self.retry_config,
                )
            else:
                return primary

    def build_with_monitoring(self) -> Tuple[BaseChatModel, "FallbackMonitor"]:
        """Build the fallback chain with monitoring capabilities.

        Returns:
            Tuple of (LLM with fallbacks, Monitor instance)
        """
        chain = self.build()
        monitor = FallbackMonitor(self.providers, self.strategy)

        # Wrap chain with monitoring
        monitored_chain = monitor.wrap_chain(chain)

        return monitored_chain, monitor


class FallbackMonitor:
    """Monitor fallback chain performance and health."""

    def __init__(self, providers: List[ProviderPriority], strategy: FallbackStrategy):
        """Initialize monitor.

        Args:
            providers: List of provider priorities
            strategy: Fallback strategy being used
        """
        self.providers = providers
        self.strategy = strategy
        self.metrics: Dict[str, Dict[str, Any]] = {}

        # Initialize metrics for each provider
        for i, provider in enumerate(providers):
            provider_key = f"provider_{i}"
            self.metrics[provider_key] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_latency": 0.0,
                "error_types": {},
            }

    def wrap_chain(self, chain: BaseChatModel) -> BaseChatModel:
        """Wrap chain with monitoring.

        Args:
            chain: Chain to monitor

        Returns:
            Wrapped chain with monitoring
        """
        # This is a simplified version - in production, you'd want
        # to properly intercept calls and track metrics
        return chain

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers.

        Returns:
            Health status dict
        """
        status = {
            "strategy": self.strategy.value,
            "providers": [],
        }

        for i, provider in enumerate(self.providers):
            provider_key = f"provider_{i}"
            metrics = self.metrics.get(provider_key, {})

            total_requests = metrics.get("requests", 0)
            success_rate = 0.0
            if total_requests > 0:
                success_rate = metrics.get("successes", 0) / total_requests * 100

            avg_latency = 0.0
            if metrics.get("successes", 0) > 0:
                avg_latency = metrics.get("total_latency", 0) / metrics["successes"]

            provider_status = {
                "index": i,
                "config": provider.config,
                "priority": provider.priority,
                "total_requests": total_requests,
                "success_rate": success_rate,
                "average_latency": avg_latency,
                "current_load": provider.current_load,
                "error_types": metrics.get("error_types", {}),
            }

            status["providers"].append(provider_status)

        return status


def create_simple_fallback_chain(
    primary_config: Union[ProviderConfig, Dict[str, Any]],
    fallback_configs: List[Union[ProviderConfig, Dict[str, Any]]],
    strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL,
) -> BaseChatModel:
    """Convenience function to create a simple fallback chain.

    Args:
        primary_config: Primary provider configuration
        fallback_configs: List of fallback configurations
        strategy: Fallback strategy to use

    Returns:
        LLM with fallback chain
    """
    builder = FallbackChainBuilder()

    # Add primary provider with highest priority
    builder.add_provider(primary_config, priority=0)

    # Add fallback providers
    for i, config in enumerate(fallback_configs):
        builder.add_provider(config, priority=i + 1)

    # Set strategy and build
    return builder.with_strategy(strategy).build()
