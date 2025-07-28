"""Logging configuration and utilities for Virtual Agora.

This module provides centralized logging setup and helper functions
for consistent logging across the application with display mode awareness.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


# Global logger cache
_loggers: dict[str, logging.Logger] = {}


class DisplayModeFilter(logging.Filter):
    """Logging filter that respects display mode settings."""

    def filter(self, record):
        """Filter log records based on current display mode.

        Args:
            record: LogRecord to filter

        Returns:
            True if record should be logged
        """
        try:
            # Import here to avoid circular dependency
            from virtual_agora.ui.display_modes import should_show_log

            return should_show_log(record.levelname)
        except ImportError:
            # If display modes not available, show all logs
            return True


class AssemblyModeHandler(RichHandler):
    """Rich handler optimized for Assembly mode display."""

    def __init__(self, *args, **kwargs):
        """Initialize assembly mode handler."""
        super().__init__(
            rich_tracebacks=True,
            show_time=False,  # Hide timestamps in assembly mode
            show_path=False,
            markup=True,
            *args,
            **kwargs,
        )
        # Add display mode filter
        self.addFilter(DisplayModeFilter())

    def format(self, record):
        """Format log record for assembly mode."""
        # For assembly mode, we want cleaner, user-friendly messages
        if hasattr(record, "assembly_message"):
            # Use custom assembly message if provided
            record.msg = record.assembly_message

        return super().format(record)


def setup_logging(
    level: str = "WARNING",
    log_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
) -> None:
    """Set up logging configuration for the application with display mode awareness.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: WARNING.
        log_dir: Directory for log files. Defaults to 'logs' in project root.
        session_id: Unique session identifier for log file naming.
    """
    try:
        # Import display mode here to avoid circular dependency
        from virtual_agora.ui.display_modes import get_display_manager

        display_manager = get_display_manager()

        # Use display mode override if available
        if display_manager.get_log_level_override():
            level = display_manager.get_log_level_override()

    except ImportError:
        # If display modes not available, use provided level
        pass

    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.WARNING)

    # Set up log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Generate session ID if not provided
    if session_id is None:
        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Create log file path
    log_file = log_dir / f"session_{session_id}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to let filters handle it

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose console handler based on display mode
    try:
        from virtual_agora.ui.display_modes import is_assembly_mode

        if is_assembly_mode():
            console_handler = AssemblyModeHandler()
        else:
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
            console_handler.addFilter(DisplayModeFilter())
    except ImportError:
        # Fallback to standard handler
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )

    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler with detailed formatting (always logs everything)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Log startup message (this will be filtered in assembly mode)
    logger = get_logger(__name__)
    logger.debug(f"Logging initialized - Session ID: {session_id}")
    logger.debug(f"Log file: {log_file.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the specified name.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        Configured logger instance.
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


class SessionLogger:
    """Context manager for session-specific logging.

    This class provides a convenient way to log all activities
    within a session context with proper formatting.
    """

    def __init__(self, session_id: str, log_dir: Optional[Path] = None):
        """Initialize session logger.

        Args:
            session_id: Unique session identifier.
            log_dir: Directory for log files.
        """
        self.session_id = session_id
        self.log_dir = log_dir or Path("logs")
        self.logger = get_logger(f"session.{session_id}")

    def log_user_input(self, prompt: str, response: str) -> None:
        """Log user input with proper formatting.

        Args:
            prompt: The prompt shown to the user.
            response: The user's response.
        """
        self.logger.info(f"USER_INPUT - Prompt: {prompt}")
        self.logger.info(f"USER_INPUT - Response: {response}")

    def log_agent_response(self, agent_name: str, response: str) -> None:
        """Log agent response with proper formatting.

        Args:
            agent_name: Name of the agent.
            response: The agent's response.
        """
        self.logger.info(f"AGENT_RESPONSE - {agent_name}")
        self.logger.info(f"AGENT_RESPONSE - Content: {response}")

    def log_system_event(self, event: str, details: Optional[str] = None) -> None:
        """Log system event with proper formatting.

        Args:
            event: Event description.
            details: Optional additional details.
        """
        self.logger.info(f"SYSTEM_EVENT - {event}")
        if details:
            self.logger.info(f"SYSTEM_EVENT - Details: {details}")

    def log_error(self, error: str, exception: Optional[Exception] = None) -> None:
        """Log error with proper formatting.

        Args:
            error: Error description.
            exception: Optional exception object.
        """
        self.logger.error(f"ERROR - {error}")
        if exception:
            self.logger.exception("Exception details:", exc_info=exception)


def log_assembly_event(message: str, level: str = "INFO") -> None:
    """Log a user-friendly message for assembly viewers.

    This function creates log messages that are appropriate for users
    watching the democratic assembly, avoiding technical jargon.

    Args:
        message: User-friendly message to display
        level: Log level (INFO, WARNING, ERROR)
    """
    logger = get_logger("virtual_agora.assembly")
    log_func = getattr(logger, level.lower(), logger.info)

    # Create a log record with assembly-friendly formatting
    record = logging.LogRecord(
        name="virtual_agora.assembly",
        level=getattr(logging, level.upper(), logging.INFO),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.assembly_message = message

    # Send to appropriate handler
    log_func(message)


def log_system_transition(from_phase: str, to_phase: str, details: str = "") -> None:
    """Log a system phase transition in a user-friendly way.

    Args:
        from_phase: Phase transitioning from
        to_phase: Phase transitioning to
        details: Optional additional details
    """
    try:
        from virtual_agora.ui.display_modes import is_assembly_mode

        if is_assembly_mode():
            if details:
                message = f"Transitioning from {from_phase} to {to_phase} - {details}"
            else:
                message = f"Transitioning from {from_phase} to {to_phase}"
            log_assembly_event(message)
        else:
            # Standard technical logging for developer mode
            logger = get_logger("virtual_agora.system")
            logger.info(f"Phase transition: {from_phase} → {to_phase} ({details})")
    except ImportError:
        # Fallback
        logger = get_logger("virtual_agora.system")
        logger.info(f"Phase transition: {from_phase} → {to_phase} ({details})")
