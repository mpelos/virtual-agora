"""Deterministic LLM mocks for Virtual Agora integration testing.

This package provides deterministic LLM implementations that maintain realistic
behavior while being completely predictable for testing purposes. The mocks
integrate seamlessly with LangGraph and the Virtual Agora execution flow.

Key Components:
- BaseDeterministicLLM: Base class for all deterministic LLM implementations
- Role-specific LLMs: ModeratorDeterministicLLM, DiscussionAgentDeterministicLLM, etc.
- DeterministicProviderFactory: Factory for creating appropriate LLM instances
- Response templates: Realistic response patterns for each agent role
- Setup/teardown utilities: Easy integration with existing test framework

Usage:
    from tests.mocks import setup_deterministic_testing_environment

    setup_info = setup_deterministic_testing_environment()
    # Run tests with deterministic LLMs
    teardown_deterministic_testing_environment(setup_info)
"""

from .deterministic_llms import BaseDeterministicLLM
from .role_specific_llms import (
    ModeratorDeterministicLLM,
    DiscussionAgentDeterministicLLM,
    SummarizerDeterministicLLM,
    ReportWriterDeterministicLLM,
)
from .llm_responses import (
    get_response_templates,
    get_context_patterns,
    get_interrupt_patterns,
    match_context_to_pattern,
    should_trigger_interrupt,
)
from .mock_provider_factory import (
    DeterministicProviderFactory,
    create_deterministic_provider,
    get_provider_factory,
    reset_provider_factory,
    VirtualAgoraConfigMock,
    setup_deterministic_testing_environment,
    teardown_deterministic_testing_environment,
)
from .interrupt_simulation import (
    setup_default_interrupt_simulations,
    reset_interrupt_simulator,
)

__all__ = [
    # Base classes
    "BaseDeterministicLLM",
    # Role-specific LLMs
    "ModeratorDeterministicLLM",
    "DiscussionAgentDeterministicLLM",
    "SummarizerDeterministicLLM",
    "ReportWriterDeterministicLLM",
    # Response system
    "get_response_templates",
    "get_context_patterns",
    "get_interrupt_patterns",
    "match_context_to_pattern",
    "should_trigger_interrupt",
    # Provider factory
    "DeterministicProviderFactory",
    "create_deterministic_provider",
    "get_provider_factory",
    "reset_provider_factory",
    "VirtualAgoraConfigMock",
    # Environment setup
    "setup_deterministic_testing_environment",
    "teardown_deterministic_testing_environment",
    "setup_default_interrupt_simulations",
    "reset_interrupt_simulator",
]
