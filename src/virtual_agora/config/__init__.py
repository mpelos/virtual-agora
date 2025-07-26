"""Configuration management for Virtual Agora.

This module handles loading, parsing, and validating configuration
from YAML files and environment variables.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type imports for better IDE support
    from .loader import ConfigLoader
    from .schema import Config, ModeratorConfig, AgentConfig
    from .validators import ConfigValidator

__all__ = [
    "ConfigLoader",
    "Config",
    "ModeratorConfig",
    "AgentConfig",
    "ConfigValidator",
]
