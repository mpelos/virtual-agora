"""Utility functions and helpers for Virtual Agora.

This module contains shared utilities including logging setup,
custom exceptions, and other helper functions.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .logging import setup_logging, get_logger
    from .exceptions import (
        VirtualAgoraError,
        ConfigurationError,
        ProviderError,
        AgentError,
        WorkflowError,
        ValidationError,
        TimeoutError,
        StateError,
    )
    from .langgraph_error_handler import (
        LangGraphErrorHandler,
        create_provider_error_chain,
        with_langgraph_error_handling,
    )

# Base exports
__all__ = [
    "setup_logging",
    "get_logger",
    "VirtualAgoraError",
    "ConfigurationError", 
    "ProviderError",
    "AgentError",
    "WorkflowError",
    "ValidationError",
    "TimeoutError",
    "StateError",
]

# Add LangGraph exports if available
try:
    from .langgraph_error_handler import (
        LangGraphErrorHandler,
        create_provider_error_chain,
        with_langgraph_error_handling,
    )
    __all__.extend([
        "LangGraphErrorHandler",
        "create_provider_error_chain", 
        "with_langgraph_error_handling",
    ])
except ImportError:
    # LangGraph not available, continue with base exports only
    pass