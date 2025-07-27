"""Reusable UI components for Virtual Agora.

This module provides common Rich-based UI components that can be used
throughout the application for consistent user interface elements.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import asyncio
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box
from rich.prompt import Prompt, Confirm

from virtual_agora.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


class VirtualAgoraTheme:
    """Consistent theme colors and styles for the application."""

    # Brand colors
    PRIMARY = "cyan"
    SECONDARY = "magenta"
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "blue"

    # Text styles
    HEADING = "bold cyan"
    SUBHEADING = "bold magenta"
    EMPHASIS = "bold"
    DIM = "dim"

    # Component styles
    PANEL_BORDER = "cyan"
    TABLE_HEADER = "bold magenta"
    PROGRESS_BAR = "cyan"


class LoadingSpinner:
    """Context manager for showing a loading spinner."""

    def __init__(
        self, message: str = "Processing...", style: str = VirtualAgoraTheme.PRIMARY
    ):
        self.message = message
        self.style = style
        self.progress = None
        self.task_id = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(style=self.style),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(self.message, total=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()

    def update(self, message: str):
        """Update the spinner message."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=message)


class ProgressBar:
    """Enhanced progress bar with ETA and percentage."""

    def __init__(self, total: int, description: str = "Progress"):
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeRemainingColumn(),
            console=console,
        )
        self.task_id = None
        self.total = total
        self.description = description

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        if self.task_id is not None:
            if description:
                self.progress.update(self.task_id, description=description)
            self.progress.advance(self.task_id, advance)


def create_header_panel(title: str, subtitle: Optional[str] = None) -> Panel:
    """Create a consistent header panel."""
    content = f"[{VirtualAgoraTheme.HEADING}]{title}[/{VirtualAgoraTheme.HEADING}]"

    panel = Panel(
        Align.center(content),
        subtitle=subtitle,
        border_style=VirtualAgoraTheme.PANEL_BORDER,
        padding=(1, 2),
        expand=False,
    )

    return panel


def create_info_table(data: Dict[str, Any], title: Optional[str] = None) -> Table:
    """Create a formatted information table."""
    table = Table(
        title=title, show_header=False, box=box.SIMPLE, padding=(0, 1), expand=False
    )

    table.add_column("Field", style="bold")
    table.add_column("Value")

    for key, value in data.items():
        table.add_row(key, str(value))

    return table


def create_options_menu(options: Dict[str, str], title: str = "Options") -> Panel:
    """Create a formatted options menu."""
    lines = []
    for key, description in options.items():
        lines.append(
            f"[{VirtualAgoraTheme.SUCCESS}]{key}[/{VirtualAgoraTheme.SUCCESS}] - {description}"
        )

    content = "\n".join(lines)

    return Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style=VirtualAgoraTheme.SECONDARY,
        padding=(1, 2),
    )


def create_status_panel(
    status: str, style: str = "info", title: Optional[str] = None
) -> Panel:
    """Create a status notification panel."""
    styles = {
        "info": (VirtualAgoraTheme.INFO, "ℹ"),
        "success": (VirtualAgoraTheme.SUCCESS, "✓"),
        "warning": (VirtualAgoraTheme.WARNING, "⚠"),
        "error": (VirtualAgoraTheme.ERROR, "✗"),
    }

    color, icon = styles.get(style, styles["info"])

    content = f"[{color}]{icon}[/{color}] {status}"

    return Panel(content, title=title, border_style=color, padding=(0, 1), expand=False)


class SessionTimer:
    """Live updating session timer display."""

    def __init__(self, start_time: Optional[datetime] = None):
        self.start_time = start_time or datetime.now()
        self.live = None
        self._running = False

    def generate_display(self) -> Panel:
        """Generate the timer display."""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return Panel(
            f"[bold cyan]Session Time:[/bold cyan] {time_str}",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )

    async def run_async(self):
        """Run the timer with async updates."""
        with Live(self.generate_display(), refresh_per_second=1) as live:
            self.live = live
            self._running = True
            while self._running:
                await asyncio.sleep(1)
                live.update(self.generate_display())

    def stop(self):
        """Stop the timer."""
        self._running = False


class AgentMessageDisplay:
    """Formatted display for agent messages."""

    @staticmethod
    def format_message(
        agent_id: str,
        content: str,
        role: str = "participant",
        timestamp: Optional[datetime] = None,
    ) -> Panel:
        """Format an agent message for display."""
        timestamp = timestamp or datetime.now()
        time_str = timestamp.strftime("%H:%M:%S")

        # Role-based styling
        if role == "moderator":
            border_style = VirtualAgoraTheme.SECONDARY
            title_style = VirtualAgoraTheme.SUBHEADING
            icon = "🎯"
        else:
            border_style = VirtualAgoraTheme.PRIMARY
            title_style = VirtualAgoraTheme.HEADING
            icon = "💭"

        title = f"{icon} [{title_style}]{agent_id}[/{title_style}] - {time_str}"

        return Panel(content, title=title, border_style=border_style, padding=(1, 2))


class InteractiveList:
    """Interactive list selector with arrow key navigation."""

    def __init__(
        self,
        items: List[str],
        title: str = "Select an item",
        allow_multiple: bool = False,
    ):
        self.items = items
        self.title = title
        self.allow_multiple = allow_multiple
        self.selected_indices = set()
        self.current_index = 0

    def display(self) -> Table:
        """Generate the current display."""
        table = Table(
            title=self.title, show_header=False, box=box.ROUNDED, padding=(0, 1)
        )

        table.add_column("", width=3)
        table.add_column("Item")

        for i, item in enumerate(self.items):
            prefix = ""
            style = ""

            if i == self.current_index:
                prefix = "▶"
                style = "bold cyan"
            elif i in self.selected_indices:
                prefix = "✓"
                style = "green"

            table.add_row(prefix, item, style=style)

        return table


class ConfirmationDialog:
    """Enhanced confirmation dialog with custom styling."""

    @staticmethod
    def ask(question: str, default: bool = True, danger: bool = False) -> bool:
        """Show a confirmation dialog."""
        style = VirtualAgoraTheme.ERROR if danger else VirtualAgoraTheme.WARNING

        panel = Panel(
            question,
            title="[bold]Confirmation Required[/bold]",
            border_style=style,
            padding=(1, 2),
        )

        console.print(panel)

        return Confirm.ask("Proceed?", default=default)


class StatusDashboard:
    """Live updating status dashboard."""

    def __init__(self):
        self.layout = Layout()
        self.components = {}

    def setup_layout(self):
        """Set up the dashboard layout."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        self.layout["main"].split_row(Layout(name="left"), Layout(name="right"))

    def update_component(self, name: str, content: Any):
        """Update a dashboard component."""
        self.components[name] = content

        if name == "header":
            self.layout["header"].update(content)
        elif name == "footer":
            self.layout["footer"].update(content)
        elif name == "left":
            self.layout["main"]["left"].update(content)
        elif name == "right":
            self.layout["main"]["right"].update(content)

    def render(self) -> Layout:
        """Render the current dashboard state."""
        return self.layout


def create_markdown_panel(content: str, title: Optional[str] = None) -> Panel:
    """Create a panel with markdown content."""
    from rich.markdown import Markdown

    md = Markdown(content)

    return Panel(md, title=title, border_style=VirtualAgoraTheme.INFO, padding=(1, 2))


@contextmanager
def console_section(title: str, style: str = VirtualAgoraTheme.PRIMARY):
    """Context manager for console sections."""
    console.rule(f"[{style}]{title}[/{style}]", style=style)
    yield
    console.print()  # Add spacing after section


def format_timestamp(dt: datetime, relative: bool = False) -> str:
    """Format timestamp for display."""
    if relative:
        delta = datetime.now() - dt
        if delta.seconds < 60:
            return "just now"
        elif delta.seconds < 3600:
            return f"{delta.seconds // 60} min ago"
        elif delta.seconds < 86400:
            return f"{delta.seconds // 3600} hours ago"
        else:
            return f"{delta.days} days ago"
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class InputValidator:
    """Reusable input validation utilities."""

    @staticmethod
    def validate_length(
        value: str, min_length: Optional[int] = None, max_length: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate input length."""
        if min_length and len(value) < min_length:
            return False, f"Too short (minimum {min_length} characters)"
        if max_length and len(value) > max_length:
            return False, f"Too long (maximum {max_length} characters)"
        return True, None

    @staticmethod
    def validate_choice(value: str, choices: List[str]) -> tuple[bool, Optional[str]]:
        """Validate input is in allowed choices."""
        if value not in choices:
            return False, f"Invalid choice. Options: {', '.join(choices)}"
        return True, None

    @staticmethod
    def validate_not_empty(value: str) -> tuple[bool, Optional[str]]:
        """Validate input is not empty."""
        if not value.strip():
            return False, "Input cannot be empty"
        return True, None


def create_error_panel(error: Exception, show_traceback: bool = False) -> Panel:
    """Create an error display panel."""
    content = (
        f"[{VirtualAgoraTheme.ERROR}]Error:[/{VirtualAgoraTheme.ERROR}] {str(error)}"
    )

    if show_traceback:
        import traceback

        tb = traceback.format_exc()
        content += f"\n\n[dim]{tb}[/dim]"

    return Panel(
        content,
        title="[bold red]Error Occurred[/bold red]",
        border_style=VirtualAgoraTheme.ERROR,
        padding=(1, 2),
    )
