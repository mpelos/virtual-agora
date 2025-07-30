"""Enhanced session control for Virtual Agora v1.3.

This module provides session control features including interrupt handling,
periodic checkpoints, and user control options.
"""

import signal
import sys
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich import box

from virtual_agora.ui.components import (
    create_status_panel,
    create_options_menu,
    create_info_table,
    VirtualAgoraTheme,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class SessionController:
    """Enhanced session control for v1.3."""

    def __init__(self, console: Console):
        self.console = console
        self.interrupt_callback: Optional[Callable] = None
        self.session_paused = False
        self.checkpoint_history = []
        self._setup_interrupt_handler()

    def _setup_interrupt_handler(self) -> None:
        """Setup keyboard interrupt handling."""

        def handle_interrupt(signum, frame):
            """Handle Ctrl+C interrupt."""
            self._show_interrupt_menu()

        signal.signal(signal.SIGINT, handle_interrupt)

    def set_interrupt_callback(self, callback: Callable) -> None:
        """Set callback for interrupt handling."""
        self.interrupt_callback = callback

    def _show_interrupt_menu(self) -> None:
        """Show menu when user interrupts."""
        self.console.print("\n[yellow]Session interrupted![/yellow]")
        self.console.bell()  # Audio notification

        # Show current session status
        if hasattr(self, "current_state"):
            status_info = {
                "Phase": self.current_state.get("current_phase", "Unknown"),
                "Topic": self.current_state.get("active_topic", "None"),
                "Round": self.current_state.get("current_round", 0),
            }
            status_table = create_info_table(status_info)
            self.console.print(status_table)

        # Show interrupt options
        options = {
            "r": "[green]Resume[/green] session",
            "e": "[orange]End[/orange] current topic",
            "s": "[blue]Skip[/blue] to final report",
            "p": "[yellow]Pause[/yellow] and save state",
            "q": "[red]Quit[/red] without saving",
            "h": "[cyan]Help[/cyan] - Show keyboard shortcuts",
        }

        options_menu = create_options_menu(options, "Interrupt Options")
        self.console.print(options_menu)

        try:
            choice = Prompt.ask(
                "What would you like to do?", choices=list(options.keys()), default="r"
            ).lower()

            self._handle_interrupt_choice(choice)

        except KeyboardInterrupt:
            # Double interrupt - emergency exit
            self.console.print("\n[red]Emergency shutdown![/red]")
            sys.exit(1)

    def _handle_interrupt_choice(self, choice: str) -> None:
        """Handle user's interrupt choice."""
        if choice == "r":
            self.console.print("[green]Resuming session...[/green]")
            return

        elif choice == "e":
            self.console.print("[orange]Ending current topic...[/orange]")
            if self.interrupt_callback:
                self.interrupt_callback(
                    {"action": "end_topic", "reason": "User interrupt"}
                )

        elif choice == "s":
            self.console.print("[blue]Skipping to final report...[/blue]")
            if self.interrupt_callback:
                self.interrupt_callback(
                    {"action": "skip_to_report", "reason": "User interrupt"}
                )

        elif choice == "p":
            self._pause_session()

        elif choice == "q":
            if Confirm.ask(
                "Are you sure you want to quit without saving?", default=False
            ):
                self.console.print("[red]Exiting without saving...[/red]")
                sys.exit(0)
            else:
                self._show_interrupt_menu()  # Show menu again

        elif choice == "h":
            self._show_help()
            self._show_interrupt_menu()  # Show menu again after help

    def _pause_session(self) -> None:
        """Pause the session and save state."""
        self.session_paused = True
        self.console.print("[yellow]Pausing session...[/yellow]")

        # Save checkpoint
        checkpoint = {
            "timestamp": datetime.now(),
            "state": getattr(self, "current_state", {}),
            "reason": "User pause",
        }
        self.checkpoint_history.append(checkpoint)

        # Show pause confirmation
        pause_panel = Panel(
            "Session paused successfully.\n"
            "State has been saved and can be resumed later.\n\n"
            "To resume, restart the application with --resume flag.",
            title="[yellow]Session Paused[/yellow]",
            border_style="yellow",
        )
        self.console.print(pause_panel)

        if self.interrupt_callback:
            self.interrupt_callback({"action": "pause", "checkpoint": checkpoint})

        sys.exit(0)

    def _show_help(self) -> None:
        """Show keyboard shortcuts help."""
        help_panel = Panel(
            "[bold]Keyboard Shortcuts:[/bold]\n\n"
            "[cyan]Ctrl+C[/cyan] - Interrupt menu\n"
            "[cyan]Ctrl+D[/cyan] - End input (where applicable)\n"
            "[cyan]Ctrl+L[/cyan] - Clear screen\n\n"
            "[bold]During Interrupts:[/bold]\n"
            "[green]r[/green] - Resume normal operation\n"
            "[orange]e[/orange] - End current discussion topic\n"
            "[blue]s[/blue] - Skip remaining topics, go to report\n"
            "[yellow]p[/yellow] - Pause and save session state\n"
            "[red]q[/red] - Quit without saving\n",
            title="Help",
            border_style="cyan",
        )
        self.console.print(help_panel)
        self.console.input("\nPress Enter to continue...")

    def check_periodic_control(
        self, round_num: int, checkpoint_interval: int = 5
    ) -> bool:
        """Check if periodic control point reached."""
        if round_num > 0 and round_num % checkpoint_interval == 0:
            # Check if we've already shown checkpoint for this round
            for checkpoint in self.checkpoint_history:
                if checkpoint.get("round") == round_num:
                    return False
            return True
        return False

    def display_checkpoint_notification(
        self, round_num: int, topic: str, state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Display checkpoint notification with enhanced visuals."""
        self.console.bell()  # Audio notification
        self.console.clear()

        # Update current state for interrupt handler
        if state:
            self.current_state = state

        # Create checkpoint panel
        checkpoint_content = (
            f"[bold yellow]🛑 5-Round Checkpoint Reached![/bold yellow]\n\n"
            f"[bold]Round:[/bold] {round_num}\n"
            f"[bold]Topic:[/bold] {topic}\n\n"
            "You now have the opportunity to:\n"
            "• Review the discussion progress\n"
            "• End the current topic discussion\n"
            "• Continue for another 5 rounds\n"
            "• Modify discussion parameters\n"
            "• Take a break or pause the session"
        )

        checkpoint_panel = Panel(
            checkpoint_content,
            title="User Control Point",
            border_style="yellow",
            box=box.DOUBLE,
            padding=(1, 2),
            expand=False,
        )

        self.console.print(checkpoint_panel)

        # Show session statistics
        if state:
            stats = {
                "Total Messages": len(state.get("messages", [])),
                "Topics Completed": len(state.get("completed_topics", [])),
                "Session Duration": self._format_duration(state.get("start_time")),
            }
            stats_table = create_info_table(stats, "Session Statistics")
            self.console.print(stats_table)

        # Record checkpoint
        self.checkpoint_history.append(
            {
                "timestamp": datetime.now(),
                "round": round_num,
                "topic": topic,
                "type": "periodic_5_round",
            }
        )

    def _format_duration(self, start_time: Optional[datetime]) -> str:
        """Format session duration."""
        if not start_time:
            return "Unknown"

        duration = datetime.now() - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def show_topic_transition(
        self, from_topic: str, to_topic: Optional[str] = None, reason: str = "Completed"
    ) -> None:
        """Display topic transition notification."""
        if to_topic:
            transition_text = (
                f"[green]✓[/green] Completed: {from_topic}\n"
                f"[cyan]→[/cyan] Next topic: {to_topic}"
            )
            style = "green"
        else:
            transition_text = f"[green]✓[/green] Completed: {from_topic}"
            style = "green"

        transition_panel = Panel(
            transition_text,
            title=f"Topic Transition ({reason})",
            border_style=style,
            padding=(1, 2),
        )

        self.console.print(transition_panel)

    def show_phase_transition(
        self, from_phase: int, to_phase: int, phase_names: Dict[int, str]
    ) -> None:
        """Display phase transition notification."""
        from_name = phase_names.get(from_phase, f"Phase {from_phase}")
        to_name = phase_names.get(to_phase, f"Phase {to_phase}")

        transition_panel = Panel(
            f"[dim]{from_name}[/dim] → [bold]{to_name}[/bold]",
            title="Phase Transition",
            border_style="blue",
            padding=(0, 1),
        )

        self.console.print(transition_panel)

    def confirm_session_end(
        self, reason: str = "Session complete", stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Confirm session end with summary."""
        # Show session summary
        summary_lines = [f"[bold]Reason:[/bold] {reason}", ""]

        if stats:
            summary_lines.extend(
                [
                    f"[bold]Duration:[/bold] {stats.get('duration', 'Unknown')}",
                    f"[bold]Topics Discussed:[/bold] {stats.get('topics_completed', 0)}",
                    f"[bold]Total Messages:[/bold] {stats.get('total_messages', 0)}",
                    f"[bold]Checkpoints:[/bold] {len(self.checkpoint_history)}",
                ]
            )

        summary_panel = Panel(
            "\n".join(summary_lines), title="Session Summary", border_style="blue"
        )

        self.console.print(summary_panel)

        return Confirm.ask("End session?", default=True, console=self.console)

    def get_checkpoint_history(self) -> list:
        """Get checkpoint history."""
        return self.checkpoint_history

    def clear_screen(self) -> None:
        """Clear the console screen."""
        self.console.clear()

    def show_thinking_indicator(self, message: str = "Thinking...") -> None:
        """Show a thinking indicator."""
        self.console.print(f"[dim]{message}[/dim]", end="")


class CheckpointManager:
    """Manages session checkpoints and recovery."""

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self) -> None:
        """Ensure checkpoint directory exists."""
        import os

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        session_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a session checkpoint."""
        import json

        checkpoint_id = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "metadata": metadata or {},
        }

        filepath = f"{self.checkpoint_dir}/{checkpoint_id}.json"

        try:
            with open(filepath, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            logger.info(f"Saved checkpoint: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load a checkpoint by ID."""
        import json

        filepath = f"{self.checkpoint_dir}/{checkpoint_id}.json"

        try:
            with open(filepath, "r") as f:
                checkpoint_data = json.load(f)

            logger.info(f"Loaded checkpoint: {checkpoint_id}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def list_checkpoints(self, session_id: Optional[str] = None) -> list:
        """List available checkpoints."""
        import os
        import json

        checkpoints = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.checkpoint_dir, filename)

                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    if session_id is None or data.get("session_id") == session_id:
                        checkpoints.append(
                            {
                                "checkpoint_id": data["checkpoint_id"],
                                "session_id": data["session_id"],
                                "timestamp": data["timestamp"],
                                "metadata": data.get("metadata", {}),
                            }
                        )

                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {filename}: {e}")

        # Sort by timestamp
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        import os

        filepath = f"{self.checkpoint_dir}/{checkpoint_id}.json"

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
