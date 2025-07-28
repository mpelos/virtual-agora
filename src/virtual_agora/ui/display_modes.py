"""Display mode system for Virtual Agora.

This module provides different viewing experiences:
- Developer Mode: Full logging and technical details for development
- Assembly Mode: Clean, engaging experience for watching democratic deliberation
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class DisplayMode(Enum):
    """Available display modes for Virtual Agora."""

    DEVELOPER = "developer"
    ASSEMBLY = "assembly"


@dataclass
class DisplayConfig:
    """Configuration for display modes."""

    mode: DisplayMode
    show_debug_logs: bool
    show_info_logs: bool
    show_system_events: bool
    show_timing_info: bool
    show_progress_indicators: bool
    show_atmospheric_elements: bool
    use_enhanced_panels: bool
    enable_animations: bool
    log_level_override: Optional[str] = None


class DisplayModeManager:
    """Manages display modes and their configurations."""

    # Pre-configured display modes
    DISPLAY_CONFIGS = {
        DisplayMode.DEVELOPER: DisplayConfig(
            mode=DisplayMode.DEVELOPER,
            show_debug_logs=True,
            show_info_logs=True,
            show_system_events=True,
            show_timing_info=True,
            show_progress_indicators=True,
            show_atmospheric_elements=False,
            use_enhanced_panels=False,
            enable_animations=False,
            log_level_override="INFO",
        ),
        DisplayMode.ASSEMBLY: DisplayConfig(
            mode=DisplayMode.ASSEMBLY,
            show_debug_logs=False,
            show_info_logs=False,  # Hide technical INFO logs
            show_system_events=True,  # Show user-relevant system messages
            show_timing_info=False,  # Hide technical timing
            show_progress_indicators=True,
            show_atmospheric_elements=True,
            use_enhanced_panels=True,
            enable_animations=True,
            log_level_override="WARNING",  # Only show warnings and errors
        ),
    }

    def __init__(self):
        """Initialize display mode manager."""
        self._current_mode = self._detect_default_mode()
        self._config = self.DISPLAY_CONFIGS[self._current_mode]
        logger.debug(f"Initialized display mode: {self._current_mode.value}")

    def _detect_default_mode(self) -> DisplayMode:
        """Detect the default display mode based on environment."""
        # Check environment variable first
        mode_env = os.environ.get("VIRTUAL_AGORA_DISPLAY_MODE", "").lower()
        if mode_env == "developer":
            return DisplayMode.DEVELOPER
        elif mode_env == "assembly":
            return DisplayMode.ASSEMBLY

        # Check for development indicators
        is_development = any(
            [
                os.environ.get("DEBUG", "").lower() in ("1", "true"),
                os.environ.get("DEVELOPMENT", "").lower() in ("1", "true"),
                "--debug" in os.sys.argv,
                "--dev" in os.sys.argv,
            ]
        )

        return DisplayMode.DEVELOPER if is_development else DisplayMode.ASSEMBLY

    @property
    def current_mode(self) -> DisplayMode:
        """Get the current display mode."""
        return self._current_mode

    @property
    def config(self) -> DisplayConfig:
        """Get the current display configuration."""
        return self._config

    def set_mode(self, mode: DisplayMode) -> None:
        """Set the display mode.

        Args:
            mode: The new display mode to use
        """
        if mode not in self.DISPLAY_CONFIGS:
            raise ValueError(f"Unknown display mode: {mode}")

        old_mode = self._current_mode
        self._current_mode = mode
        self._config = self.DISPLAY_CONFIGS[mode]

        logger.info(f"Display mode changed: {old_mode.value} → {mode.value}")

    def should_show_log(self, level: str) -> bool:
        """Determine if a log message should be shown based on current mode.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            True if the log should be displayed
        """
        level_upper = level.upper()

        if level_upper == "DEBUG":
            return self._config.show_debug_logs
        elif level_upper == "INFO":
            return self._config.show_info_logs
        elif level_upper in ("WARNING", "ERROR", "CRITICAL"):
            return True  # Always show warnings and errors
        else:
            return True  # Default to showing unknown levels

    def should_show_system_event(self, event_type: str) -> bool:
        """Determine if a system event should be shown.

        Args:
            event_type: Type of system event

        Returns:
            True if the event should be displayed
        """
        return self._config.show_system_events

    def get_log_level_override(self) -> Optional[str]:
        """Get the log level override for current mode.

        Returns:
            Log level string or None if no override
        """
        return self._config.log_level_override

    def is_assembly_mode(self) -> bool:
        """Check if currently in assembly mode.

        Returns:
            True if in assembly mode
        """
        return self._current_mode == DisplayMode.ASSEMBLY

    def is_developer_mode(self) -> bool:
        """Check if currently in developer mode.

        Returns:
            True if in developer mode
        """
        return self._current_mode == DisplayMode.DEVELOPER


# Global display mode manager instance
_display_manager: Optional[DisplayModeManager] = None


def get_display_manager() -> DisplayModeManager:
    """Get the global display mode manager instance."""
    global _display_manager
    if _display_manager is None:
        _display_manager = DisplayModeManager()
    return _display_manager


def set_display_mode(mode: DisplayMode) -> None:
    """Set the global display mode.

    Args:
        mode: The display mode to set
    """
    get_display_manager().set_mode(mode)


def get_current_mode() -> DisplayMode:
    """Get the current display mode.

    Returns:
        Current display mode
    """
    return get_display_manager().current_mode


def get_display_config() -> DisplayConfig:
    """Get the current display configuration.

    Returns:
        Current display configuration
    """
    return get_display_manager().config


def should_show_log(level: str) -> bool:
    """Check if a log should be shown in current mode.

    Args:
        level: Log level to check

    Returns:
        True if log should be displayed
    """
    return get_display_manager().should_show_log(level)


def is_assembly_mode() -> bool:
    """Check if currently in assembly mode.

    Returns:
        True if in assembly mode
    """
    return get_display_manager().is_assembly_mode()


def is_developer_mode() -> bool:
    """Check if currently in developer mode.

    Returns:
        True if in developer mode
    """
    return get_display_manager().is_developer_mode()
