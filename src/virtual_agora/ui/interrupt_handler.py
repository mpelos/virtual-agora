"""Simple interrupt handler system for Virtual Agora.

This module implements immediate termination on Ctrl+C interrupt signals.
"""

import signal
import sys
from typing import Optional

from rich.console import Console

from virtual_agora.utils.logging import get_logger
from virtual_agora.ui.console import get_console
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager

console = get_console().rich_console  # Use singleton console
logger = get_logger(__name__)


class InterruptHandler:
    """Simple interrupt handling system.

    Manages signal handlers for immediate termination on user interrupt.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        recovery_manager: Optional[StateRecoveryManager] = None,
    ):
        """Initialize interrupt handler.

        Args:
            state_manager: State manager (kept for compatibility)
            recovery_manager: Recovery manager (kept for compatibility)
        """
        self.state_manager = state_manager
        self.recovery_manager = recovery_manager
        self._original_handlers = {}

    def setup(self) -> None:
        """Set up signal handlers."""
        # Store original handlers
        self._original_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, self._handle_sigint
        )

        # Handle terminal resize
        if hasattr(signal, "SIGWINCH"):
            self._original_handlers[signal.SIGWINCH] = signal.signal(
                signal.SIGWINCH, self._handle_terminal_resize
            )

        logger.info("Interrupt handlers installed")

    def teardown(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

        logger.info("Interrupt handlers removed")

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT (Ctrl+C) signal - exit immediately."""
        logger.info("Interrupt signal received (Ctrl+C) - exiting immediately")
        console.print("\n[yellow]Session interrupted by user[/yellow]")
        sys.exit(1)

    def _handle_terminal_resize(self, signum, frame):
        """Handle terminal window resize."""
        # Refresh display if needed
        console.clear()

    # All menu-related methods removed - interrupt now exits immediately


# Global interrupt handler instance
_interrupt_handler: Optional[InterruptHandler] = None


def get_interrupt_handler() -> InterruptHandler:
    """Get the global interrupt handler instance.

    Returns:
        InterruptHandler instance
    """
    global _interrupt_handler
    if _interrupt_handler is None:
        _interrupt_handler = InterruptHandler()
    return _interrupt_handler


def setup_interrupt_handlers(
    state_manager: Optional[StateManager] = None,
    recovery_manager: Optional[StateRecoveryManager] = None,
) -> InterruptHandler:
    """Set up interrupt handlers with state management.

    Args:
        state_manager: State manager instance
        recovery_manager: Recovery manager instance

    Returns:
        Configured InterruptHandler
    """
    handler = get_interrupt_handler()
    handler.state_manager = state_manager
    handler.recovery_manager = recovery_manager
    handler.setup()
    return handler
