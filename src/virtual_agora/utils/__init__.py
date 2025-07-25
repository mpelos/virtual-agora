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
    )

__all__ = [
    "setup_logging",
    "get_logger",
    "VirtualAgoraError",
    "ConfigurationError", 
    "ProviderError",
    "AgentError",
]