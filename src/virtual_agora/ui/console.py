"""Enhanced console system for Virtual Agora terminal UI.

This module provides a singleton Rich console with advanced features including
terminal capability detection, theme integration, and structured output management.
"""

import os
import sys
from typing import Optional, Dict, Any, Union, List
from contextlib import contextmanager
from threading import Lock
import signal

from rich.console import Console as RichConsole
from rich.terminal_theme import TerminalTheme
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.table import Table
from rich import box
from rich.measure import Measurement

from virtual_agora.ui.theme import (
    VirtualAgoraTheme,
    get_current_theme,
    ProviderType,
    MessageType,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ConsoleCapabilities:
    """Terminal capability detection and management."""

    def __init__(self):
        """Initialize capability detection."""
        self.width = self._detect_width()
        self.height = self._detect_height()
        self.color_support = self._detect_color_support()
        self.unicode_support = self._detect_unicode_support()
        self.emoji_support = self._detect_emoji_support()
        self.interactive = self._detect_interactive()

    def _detect_width(self) -> int:
        """Detect terminal width."""
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80  # Default fallback

    def _detect_height(self) -> int:
        """Detect terminal height."""
        try:
            return os.get_terminal_size().lines
        except OSError:
            return 24  # Default fallback

    def _detect_color_support(self) -> str:
        """Detect color support level."""
        if os.environ.get("NO_COLOR"):
            return "none"

        if os.environ.get("FORCE_COLOR"):
            return "truecolor"

        term = os.environ.get("TERM", "").lower()
        colorterm = os.environ.get("COLORTERM", "").lower()

        # Check for explicit 256color support first
        if "256color" in term or "256" in term:
            return "256"
        # Check for truecolor
        elif "truecolor" in colorterm or "24bit" in colorterm:
            return "truecolor"
        elif "color" in term:
            return "16"
        else:
            # Default to truecolor on modern systems unless specifically limited
            return "truecolor"

    def _detect_unicode_support(self) -> bool:
        """Detect Unicode support."""
        encoding = sys.stdout.encoding or ""
        return "utf" in encoding.lower()

    def _detect_emoji_support(self) -> bool:
        """Detect emoji support (basic heuristic)."""
        if not self.unicode_support:
            return False

        # Check for common emoji-supporting terminals
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        emoji_terminals = [
            "iterm",
            "terminal",
            "gnome-terminal",
            "konsole",
            "alacritty",
        ]

        return any(term in term_program for term in emoji_terminals)

    def _detect_interactive(self) -> bool:
        """Detect if running in interactive mode."""
        return sys.stdout.isatty()

    def update_on_resize(self) -> None:
        """Update capabilities when terminal is resized."""
        self.width = self._detect_width()
        self.height = self._detect_height()


class VirtualAgoraConsole:
    """Enhanced singleton console for Virtual Agora."""

    _instance: Optional["VirtualAgoraConsole"] = None
    _lock = Lock()

    def __new__(cls) -> "VirtualAgoraConsole":
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize console if not already initialized."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.capabilities = ConsoleCapabilities()
        self.theme = get_current_theme()
        self._setup_console()
        self._setup_resize_handler()

    def _setup_console(self) -> None:
        """Setup the Rich console with detected capabilities."""
        console_kwargs = {
            "width": self.capabilities.width,
            "force_terminal": self.capabilities.interactive,
            "no_color": self.capabilities.color_support == "none",
            "legacy_windows": False,
        }

        # Set color system based on capabilities
        if self.capabilities.color_support == "truecolor":
            console_kwargs["color_system"] = "truecolor"
        elif self.capabilities.color_support == "256":
            console_kwargs["color_system"] = "256"
        elif self.capabilities.color_support == "16":
            console_kwargs["color_system"] = "standard"
        else:
            console_kwargs["color_system"] = None

        self.rich_console = RichConsole(**console_kwargs)

        # Log capability detection
        logger.debug(f"Terminal capabilities: {vars(self.capabilities)}")

    def _setup_resize_handler(self) -> None:
        """Setup terminal resize signal handler."""

        def handle_resize(signum, frame):
            self.capabilities.update_on_resize()
            # Recreate console with new dimensions
            self._setup_console()

        if hasattr(signal, "SIGWINCH"):
            signal.signal(signal.SIGWINCH, handle_resize)

    def print(self, *objects, **kwargs) -> None:
        """Enhanced print with theme integration."""
        try:
            self.rich_console.print(*objects, **kwargs)
        except Exception as e:
            # Fallback to basic print if Rich fails
            print(f"Console error: {e}")
            print(*objects)

    def print_agent_message(
        self,
        agent_id: str,
        provider: ProviderType,
        content: str,
        timestamp: Optional[str] = None,
        round_number: Optional[int] = None,
    ) -> None:
        """Print a formatted agent message."""
        colors = self.theme.assign_agent_color(agent_id, provider)

        # Create header
        header_parts = [
            f"[{colors['primary']}]{colors['symbol']} {agent_id}[/{colors['primary']}]"
        ]

        if timestamp:
            header_parts.append(f"[dim]{timestamp}[/dim]")

        if round_number:
            header_parts.append(
                f"[{colors['accent']}]Round {round_number}[/{colors['accent']}]"
            )

        header = " • ".join(header_parts)

        # Create panel
        panel = Panel(
            content,
            title=header,
            border_style=colors["border"],
            padding=(1, 2),
            width=min(self.capabilities.width - 4, 100),
        )

        self.print(panel)

    def print_system_message(
        self,
        message: str,
        message_type: MessageType = MessageType.SYSTEM_INFO,
        title: Optional[str] = None,
    ) -> None:
        """Print a formatted system message."""
        style = self.theme.get_message_style(message_type)

        icon = style["icon"] if self.capabilities.emoji_support else ""
        color = style["color"]
        border = style["border"]

        # Format content
        content = (
            f"[{color}]{icon} {message}[/{color}]"
            if icon
            else f"[{color}]{message}[/{color}]"
        )

        panel = Panel(
            content,
            title=title,
            border_style=border,
            padding=(0, 1),
            width=min(self.capabilities.width - 4, 80),
        )

        self.print(panel)

    def print_user_prompt(
        self, prompt: str, options: Optional[List[str]] = None
    ) -> None:
        """Print a user prompt with optional choices."""
        style = self.theme.get_message_style(MessageType.USER_PROMPT)

        content = f"[{style['color']}]{style['icon']} {prompt}[/{style['color']}]"

        if options:
            content += "\n\n"
            for i, option in enumerate(options, 1):
                content += f"[cyan]{i}.[/cyan] {option}\n"

        panel = Panel(
            content.rstrip(),
            title="[bold]User Input Required[/bold]",
            border_style=style["border"],
            padding=(1, 2),
        )

        self.print(panel)

    def print_round_separator(self, round_number: int, topic: str) -> None:
        """Print a separator between discussion rounds."""
        separator_text = f"Round {round_number} • {topic}"

        rule = Rule(
            f"[bold cyan]{separator_text}[/bold cyan]", style="cyan", characters="─"
        )

        self.print()
        self.print(rule)
        self.print()

    def print_phase_transition(self, from_phase: str, to_phase: str) -> None:
        """Print a phase transition indicator."""
        content = f"[yellow]Transitioning from {from_phase} to {to_phase}[/yellow]"

        panel = Panel(
            content,
            title="[bold]Phase Transition[/bold]",
            border_style="yellow",
            padding=(0, 1),
        )

        self.print(panel)

    def print_voting_results(self, results: Dict[str, Any]) -> None:
        """Print formatted voting results."""
        table = Table(
            title="[bold]Voting Results[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Agent", style="cyan")
        table.add_column("Vote", justify="center")
        table.add_column("Justification", style="dim")

        for agent_id, vote_data in results.items():
            vote = vote_data.get("vote", "No vote")
            justification = vote_data.get("justification", "No justification provided")

            # Style vote based on value
            if vote.lower() == "yes":
                vote_styled = "[green]✓ Yes[/green]"
            elif vote.lower() == "no":
                vote_styled = "[red]✗ No[/red]"
            else:
                vote_styled = f"[yellow]{vote}[/yellow]"

            table.add_row(
                agent_id,
                vote_styled,
                (
                    justification[:50] + "..."
                    if len(justification) > 50
                    else justification
                ),
            )

        self.print(table)

    def print_agenda(
        self, agenda: List[str], current_topic: Optional[str] = None
    ) -> None:
        """Print formatted agenda."""
        table = Table(
            title="[bold]Discussion Agenda[/bold]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("#", width=3)
        table.add_column("Topic", style="white")
        table.add_column("Status", justify="center", width=10)

        for i, topic in enumerate(agenda, 1):
            if topic == current_topic:
                status = "[yellow]Current[/yellow]"
                topic_style = "bold yellow"
            else:
                status = "[dim]Pending[/dim]"
                topic_style = "white"

            table.add_row(str(i), f"[{topic_style}]{topic}[/{topic_style}]", status)

        self.print(table)

    def print_session_summary(self, summary_data: Dict[str, Any]) -> None:
        """Print session summary information."""
        columns = []

        # Left column - Basic info
        left_info = Table(box=None, show_header=False, padding=(0, 1))
        left_info.add_column("Field", style="bold")
        left_info.add_column("Value")

        left_info.add_row("Session ID", summary_data.get("session_id", "Unknown"))
        left_info.add_row("Duration", summary_data.get("duration", "Unknown"))
        left_info.add_row("Topics Discussed", str(summary_data.get("topics_count", 0)))
        left_info.add_row("Total Rounds", str(summary_data.get("rounds_count", 0)))

        columns.append(
            Panel(left_info, title="[bold]Session Info[/bold]", border_style="cyan")
        )

        # Right column - Agent participation
        right_info = Table(box=None, show_header=False, padding=(0, 1))
        right_info.add_column("Agent", style="bold")
        right_info.add_column("Messages")

        agents = summary_data.get("agent_stats", {})
        for agent_id, stats in agents.items():
            right_info.add_row(agent_id, str(stats.get("message_count", 0)))

        columns.append(
            Panel(
                right_info,
                title="[bold]Agent Participation[/bold]",
                border_style="green",
            )
        )

        self.print(Columns(columns, equal=True, expand=True))

    @contextmanager
    def status_spinner(self, message: str = "Processing..."):
        """Context manager for status spinner."""
        with self.rich_console.status(message, spinner="dots"):
            yield

    def clear(self) -> None:
        """Clear the console."""
        self.rich_console.clear()

    def set_theme(self, theme: VirtualAgoraTheme) -> None:
        """Update the console theme."""
        self.theme = theme

    def get_width(self) -> int:
        """Get current console width."""
        return self.capabilities.width

    def get_height(self) -> int:
        """Get current console height."""
        return self.capabilities.height

    def supports_color(self) -> bool:
        """Check if console supports color."""
        return self.capabilities.color_support != "none"

    def supports_unicode(self) -> bool:
        """Check if console supports Unicode."""
        return self.capabilities.unicode_support

    def supports_emoji(self) -> bool:
        """Check if console supports emoji."""
        return self.capabilities.emoji_support


# Global console instance
_console: Optional[VirtualAgoraConsole] = None


def get_console() -> VirtualAgoraConsole:
    """Get the global console instance."""
    global _console
    if _console is None:
        _console = VirtualAgoraConsole()
    return _console


def print_agent_message(
    agent_id: str, provider: ProviderType, content: str, **kwargs
) -> None:
    """Convenience function for printing agent messages."""
    get_console().print_agent_message(agent_id, provider, content, **kwargs)


def print_system_message(
    message: str, message_type: MessageType = MessageType.SYSTEM_INFO, **kwargs
) -> None:
    """Convenience function for printing system messages."""
    get_console().print_system_message(message, message_type, **kwargs)


def print_user_prompt(prompt: str, options: Optional[List[str]] = None) -> None:
    """Convenience function for printing user prompts."""
    get_console().print_user_prompt(prompt, options)
