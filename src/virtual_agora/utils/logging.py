"""Logging configuration and utilities for Virtual Agora.

This module provides centralized logging setup and helper functions
for consistent logging across the application.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


# Global logger cache
_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
) -> None:
    """Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to 'logs' in project root.
        session_id: Unique session identifier for log file naming.
    """
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

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
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler with detailed formatting
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

    # Log startup message
    logger = get_logger(__name__)
    logger.info(f"Logging initialized - Session ID: {session_id}")
    logger.info(f"Log file: {log_file.absolute()}")


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
