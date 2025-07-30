"""Interrupt handler system for Virtual Agora.

This module implements Story 7.5: Emergency Controls with comprehensive
interrupt handling, state preservation, and recovery mechanisms.
"""

import signal
import sys
import threading
import time
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

from virtual_agora.utils.logging import get_logger
from virtual_agora.ui.components import VirtualAgoraTheme, create_status_panel
from virtual_agora.ui.console import get_console
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager

console = get_console().rich_console  # Use singleton console
logger = get_logger(__name__)


class InterruptContext:
    """Context information for interrupt handling."""

    def __init__(
        self,
        interrupt_type: str,
        timestamp: datetime,
        current_phase: Optional[int] = None,
        current_topic: Optional[str] = None,
        current_speaker: Optional[str] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.interrupt_type = interrupt_type
        self.timestamp = timestamp
        self.current_phase = current_phase
        self.current_topic = current_topic
        self.current_speaker = current_speaker
        self.state_snapshot = state_snapshot

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "interrupt_type": self.interrupt_type,
            "timestamp": self.timestamp.isoformat(),
            "current_phase": self.current_phase,
            "current_topic": self.current_topic,
            "current_speaker": self.current_speaker,
            "has_state_snapshot": self.state_snapshot is not None,
        }


class InterruptAction:
    """Available interrupt actions."""

    PAUSE = "pause"
    SKIP_SPEAKER = "skip_speaker"
    END_TOPIC = "end_topic"
    END_SESSION = "end_session"
    RESUME = "resume"
    SAVE_STATE = "save_state"
    SHOW_STATUS = "show_status"


class InterruptHandler:
    """Comprehensive interrupt handling system.

    Manages signal handlers, emergency controls, and state preservation
    for graceful handling of user interrupts.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        recovery_manager: Optional[StateRecoveryManager] = None,
    ):
        """Initialize interrupt handler.

        Args:
            state_manager: State manager for accessing current state
            recovery_manager: Recovery manager for state preservation
        """
        self.state_manager = state_manager
        self.recovery_manager = recovery_manager
        self._original_handlers = {}
        self._interrupt_count = 0
        self._last_interrupt_time = None
        self._interrupt_history: List[InterruptContext] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        self._emergency_save_path = Path.home() / ".virtual_agora" / "emergency_saves"
        self._emergency_save_path.mkdir(parents=True, exist_ok=True)

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
        """Handle SIGINT (Ctrl+C) signal."""
        self._interrupt_count += 1
        current_time = time.time()

        # Check for rapid interrupts (panic mode)
        if self._last_interrupt_time and (current_time - self._last_interrupt_time) < 2:
            if self._interrupt_count >= 3:
                self._emergency_shutdown()
                return

        self._last_interrupt_time = current_time

        # Create interrupt context
        context = self._create_interrupt_context("user_interrupt")
        self._interrupt_history.append(context)

        # Show interrupt menu
        action = self._show_interrupt_menu(context)
        self._execute_interrupt_action(action, context)

    def _handle_terminal_resize(self, signum, frame):
        """Handle terminal window resize."""
        # Refresh display if needed
        console.clear()

    def _create_interrupt_context(self, interrupt_type: str) -> InterruptContext:
        """Create context snapshot for interrupt."""
        state_snapshot = None
        current_phase = None
        current_topic = None
        current_speaker = None

        if self.state_manager:
            try:
                state = self.state_manager.get_snapshot()
                state_snapshot = state.copy()
                current_phase = state.get("current_phase")
                current_topic = state.get("active_topic")
                current_speaker = state.get("current_speaker_id")
            except Exception as e:
                logger.error(f"Error capturing state during interrupt: {e}")

        return InterruptContext(
            interrupt_type=interrupt_type,
            timestamp=datetime.now(),
            current_phase=current_phase,
            current_topic=current_topic,
            current_speaker=current_speaker,
            state_snapshot=state_snapshot,
        )

    def _show_interrupt_menu(self, context: InterruptContext) -> str:
        """Show interactive interrupt menu."""
        console.clear()

        # Header
        console.print(
            Panel(
                "[bold red]Session Interrupted[/bold red]",
                subtitle=f"Interrupt #{self._interrupt_count}",
                border_style="red",
            )
        )

        # Context information
        if context.current_phase is not None:
            info_table = Table(show_header=False, box=None)
            info_table.add_column("Field", style="bold")
            info_table.add_column("Value")

            phase_names = {
                0: "Initialization",
                1: "Agenda Setting",
                2: "Discussion",
                3: "Topic Conclusion",
                4: "Agenda Re-evaluation",
                5: "Final Report",
            }

            info_table.add_row(
                "Current Phase:", phase_names.get(context.current_phase, "Unknown")
            )
            if context.current_topic:
                info_table.add_row("Active Topic:", context.current_topic)
            if context.current_speaker:
                info_table.add_row("Current Speaker:", context.current_speaker)

            console.print(info_table)
            console.print()

        # Options
        options = {
            "p": (
                "[yellow]Pause[/yellow] session and save state",
                InterruptAction.PAUSE,
            ),
            "s": ("[blue]Skip[/blue] current speaker", InterruptAction.SKIP_SPEAKER),
            "t": ("[orange]End topic[/orange] early", InterruptAction.END_TOPIC),
            "e": ("[red]End session[/red] completely", InterruptAction.END_SESSION),
            "v": ("[cyan]Save[/cyan] current state", InterruptAction.SAVE_STATE),
            "i": ("[magenta]Show[/magenta] session info", InterruptAction.SHOW_STATUS),
            "r": ("[green]Resume[/green] normal operation", InterruptAction.RESUME),
        }

        option_table = Table(show_header=False, box=None)
        for key, (desc, _) in options.items():
            option_table.add_row(f"[bold]{key}[/bold]", desc)

        console.print(Panel(option_table, title="[bold]Emergency Options[/bold]"))

        # Get user choice
        while True:
            choice = Prompt.ask(
                "\nSelect action",
                choices=list(options.keys()),
                default="r",
                console=console,
            ).lower()

            if choice in options:
                return options[choice][1]

    def _execute_interrupt_action(self, action: str, context: InterruptContext) -> None:
        """Execute the selected interrupt action."""
        logger.info(f"Executing interrupt action: {action}")

        if action == InterruptAction.PAUSE:
            self._pause_session(context)
        elif action == InterruptAction.SKIP_SPEAKER:
            self._skip_current_speaker(context)
        elif action == InterruptAction.END_TOPIC:
            self._end_current_topic(context)
        elif action == InterruptAction.END_SESSION:
            self._end_session(context)
        elif action == InterruptAction.SAVE_STATE:
            self._save_state(context)
        elif action == InterruptAction.SHOW_STATUS:
            self._show_session_status(context)
        elif action == InterruptAction.RESUME:
            self._resume_session(context)

        # Execute callbacks
        self._execute_callbacks(action, context)

    def _pause_session(self, context: InterruptContext) -> None:
        """Pause the session with state preservation."""
        console.print(create_status_panel("Pausing session...", style="warning"))

        # Save state
        if self.state_manager and self.recovery_manager:
            checkpoint_id = self._create_emergency_checkpoint(context)
            console.print(f"[green]State saved to checkpoint: {checkpoint_id}[/green]")

        # Save interrupt context
        self._save_interrupt_context(context)

        console.print(
            "\n[yellow]Session paused. Run with --resume flag to continue.[/yellow]"
        )
        time.sleep(1)
        sys.exit(0)

    def _skip_current_speaker(self, context: InterruptContext) -> None:
        """Skip the current speaker."""
        if context.current_speaker:
            console.print(f"[blue]Skipping speaker: {context.current_speaker}[/blue]")

            # Set flag in state to skip
            if self.state_manager:
                self.state_manager.update_state(
                    {"skip_current_speaker": True, "interrupt_action": "skip_speaker"}
                )
        else:
            console.print("[yellow]No active speaker to skip[/yellow]")

    def _end_current_topic(self, context: InterruptContext) -> None:
        """End the current topic early."""
        if context.current_topic:
            console.print(
                f"[orange]Ending topic early: {context.current_topic}[/orange]"
            )

            # Set flag in state to end topic
            if self.state_manager:
                self.state_manager.update_state(
                    {"force_end_topic": True, "interrupt_action": "end_topic"}
                )
        else:
            console.print("[yellow]No active topic to end[/yellow]")

    def _end_session(self, context: InterruptContext) -> None:
        """End the session completely."""
        console.print(create_status_panel("Ending session...", style="error"))

        # Save final state
        if self.state_manager and self.recovery_manager:
            checkpoint_id = self._create_emergency_checkpoint(context, final=True)
            console.print(f"[green]Final state saved to: {checkpoint_id}[/green]")

        console.print("\n[red]Session terminated by user.[/red]")
        time.sleep(1)
        sys.exit(1)

    def _save_state(self, context: InterruptContext) -> None:
        """Save current state without exiting."""
        if self.state_manager and self.recovery_manager:
            checkpoint_id = self._create_emergency_checkpoint(context)
            console.print(
                f"[green]✓ State saved to checkpoint: {checkpoint_id}[/green]"
            )
        else:
            console.print("[red]Unable to save state - managers not available[/red]")

    def _show_session_status(self, context: InterruptContext) -> None:
        """Show detailed session status."""
        if not self.state_manager:
            console.print("[red]State information not available[/red]")
            return

        state = self.state_manager.get_snapshot()

        # Create status table
        status_table = Table(title="Session Status", show_header=False)
        status_table.add_column("Field", style="bold")
        status_table.add_column("Value")

        status_table.add_row("Session ID", state.get("session_id", "Unknown"))
        status_table.add_row("Start Time", str(state.get("start_time", "Unknown")))
        status_table.add_row("Current Phase", str(state.get("current_phase", 0)))
        status_table.add_row("Active Topic", state.get("active_topic", "None"))
        status_table.add_row(
            "Topics Completed", str(len(state.get("completed_topics", [])))
        )
        status_table.add_row("Total Messages", str(state.get("total_messages", 0)))

        console.print(status_table)

        # Wait for user to continue
        Prompt.ask("\nPress Enter to continue", console=console)
        console.clear()

    def _resume_session(self, context: InterruptContext) -> None:
        """Resume normal session operation."""
        console.print("[green]Resuming session...[/green]")
        self._interrupt_count = 0  # Reset interrupt count

    def _emergency_shutdown(self) -> None:
        """Emergency shutdown for panic mode (3+ rapid interrupts)."""
        console.print(
            "\n[bold red]EMERGENCY SHUTDOWN - Multiple interrupts detected[/bold red]"
        )

        # Try to save state
        try:
            context = self._create_interrupt_context("emergency_shutdown")
            if self.state_manager and self.recovery_manager:
                checkpoint_id = self._create_emergency_checkpoint(
                    context, emergency=True
                )
                console.print(
                    f"[yellow]Emergency state saved to: {checkpoint_id}[/yellow]"
                )
        except Exception as e:
            logger.error(f"Failed to save emergency state: {e}")

        console.print("[red]Exiting immediately.[/red]")
        sys.exit(2)

    def _create_emergency_checkpoint(
        self, context: InterruptContext, final: bool = False, emergency: bool = False
    ) -> str:
        """Create an emergency checkpoint."""
        checkpoint_type = (
            "emergency" if emergency else "final" if final else "interrupt"
        )

        if self.recovery_manager and context.state_snapshot:
            checkpoint = self.recovery_manager.create_checkpoint(
                state=context.state_snapshot,
                operation=f"{checkpoint_type}_save",
                metadata={
                    "interrupt_type": context.interrupt_type,
                    "timestamp": context.timestamp.isoformat(),
                    "phase": context.current_phase,
                    "topic": context.current_topic,
                },
                save_to_disk=True,
            )
            return checkpoint.checkpoint_id

        # Fallback to file save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{checkpoint_type}_state_{timestamp}.json"
        filepath = self._emergency_save_path / filename

        with open(filepath, "w") as f:
            json.dump(context.state_snapshot, f, indent=2, default=str)

        return str(filepath)

    def _save_interrupt_context(self, context: InterruptContext) -> None:
        """Save interrupt context for recovery."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interrupt_context_{timestamp}.json"
        filepath = self._emergency_save_path / filename

        with open(filepath, "w") as f:
            json.dump(context.to_dict(), f, indent=2)

    def register_callback(
        self, action: str, callback: Callable[[InterruptContext], None]
    ) -> None:
        """Register a callback for specific interrupt actions.

        Args:
            action: The interrupt action to trigger on
            callback: Function to call with interrupt context
        """
        if action not in self._callbacks:
            self._callbacks[action] = []
        self._callbacks[action].append(callback)

    def _execute_callbacks(self, action: str, context: InterruptContext) -> None:
        """Execute registered callbacks for an action."""
        if action in self._callbacks:
            for callback in self._callbacks[action]:
                try:
                    callback(context)
                except Exception as e:
                    logger.error(f"Error in interrupt callback: {e}")

    def get_interrupt_history(self) -> List[InterruptContext]:
        """Get history of interrupts in this session.

        Returns:
            List of interrupt contexts
        """
        return self._interrupt_history.copy()

    def clear_interrupt_history(self) -> None:
        """Clear interrupt history."""
        self._interrupt_history.clear()
        self._interrupt_count = 0


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
