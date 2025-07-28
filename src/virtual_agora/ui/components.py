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
        # Check if we're in assembly mode and this is a flow spinner
        try:
            from virtual_agora.ui.display_modes import is_assembly_mode

            if is_assembly_mode() and "discussion flow" in self.message.lower():
                # In assembly mode, disable flow spinners entirely to prevent residue
                self.progress = None
                self.task_id = None
                self._is_assembly_flow_spinner = True
                return self
        except ImportError:
            pass

        # Normal spinner behavior for all other cases
        self.progress = Progress(
            SpinnerColumn(style=self.style),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(self.message, total=None)
        self._is_assembly_flow_spinner = False
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


class SpecializedAgentDisplay:
    """Display component for specialized agent activities in v1.3."""

    def __init__(self, console: Console):
        self.console = console
        self.agent_colors = {
            "moderator": "cyan",
            "summarizer": "magenta",
            "topic_report": "green",
            "ecclesia_report": "blue",
        }
        self.agent_icons = {
            "moderator": "🎯",
            "summarizer": "📝",
            "topic_report": "📋",
            "ecclesia_report": "📖",
        }

    def show_agent_invocation(
        self,
        agent_type: str,
        task: str,
        context: Dict[str, Any],
        status: str = "active",
    ) -> None:
        """Display when a specialized agent is invoked."""
        color = self.agent_colors.get(agent_type, "white")
        icon = self.agent_icons.get(agent_type, "🤖")

        # Status indicator
        status_icon = (
            "⚡" if status == "active" else "✓" if status == "completed" else "❌"
        )

        self.console.print(
            f"\n{status_icon} [{color}]{icon} {agent_type.replace('_', ' ').title()} Agent[/{color}]"
        )
        self.console.print(f"   [bold]Task:[/bold] {task}")

        if context:
            self.console.print("   [bold]Context:[/bold]")
            for key, value in context.items():
                if isinstance(value, list):
                    self.console.print(f"     • {key}: {len(value)} items")
                elif isinstance(value, dict):
                    self.console.print(f"     • {key}: {len(value)} entries")
                else:
                    self.console.print(f"     • {key}: {value}")

    def show_agent_result(
        self, agent_type: str, result: str, execution_time: float
    ) -> None:
        """Display agent execution result."""
        color = self.agent_colors.get(agent_type, "white")

        result_panel = Panel(
            result[:200] + "..." if len(result) > 200 else result,
            title=f"[{color}]{agent_type.replace('_', ' ').title()} Result[/{color}]",
            border_style=color,
            padding=(0, 1),
        )
        self.console.print(result_panel)
        self.console.print(f"[dim]Execution time: {execution_time:.2f}s[/dim]\n")


class EnhancedProgressTracker:
    """Enhanced progress tracking for v1.3 workflow."""

    def __init__(self, console: Console):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        )
        self.tasks = {}
        self.phase_names = {
            0: "Initialization",
            1: "Agenda Setting",
            2: "Discussion",
            3: "Topic Conclusion",
            4: "Continuation",
            5: "Final Report",
        }

    def start(self):
        """Start the progress tracker."""
        self.progress.start()

    def stop(self):
        """Stop the progress tracker."""
        self.progress.stop()

    def start_phase(self, phase_num: int, total_steps: int) -> None:
        """Start tracking a new phase."""
        phase_name = self.phase_names.get(phase_num, f"Phase {phase_num}")
        task_id = self.progress.add_task(
            f"[bold]{phase_name}[/bold]", total=total_steps
        )
        self.tasks[f"phase_{phase_num}"] = task_id

    def update_phase(self, phase_num: int, completed: int) -> None:
        """Update phase progress."""
        task_key = f"phase_{phase_num}"
        if task_key in self.tasks:
            self.progress.update(self.tasks[task_key], completed=completed)

    def complete_phase(self, phase_num: int) -> None:
        """Mark a phase as complete."""
        task_key = f"phase_{phase_num}"
        if task_key in self.tasks:
            phase_name = self.phase_names.get(phase_num, f"Phase {phase_num}")
            self.progress.update(
                self.tasks[task_key],
                description=f"[green]✓[/green] {phase_name}",
                completed=self.progress.tasks[self.tasks[task_key]].total,
            )

    def show_round_progress(
        self, round_num: int, topic: str, is_checkpoint: bool = False
    ) -> None:
        """Display round progress with checkpoint indicator."""
        checkpoint_marker = "🛑" if is_checkpoint else "  "

        self.console.print(
            f"\n{checkpoint_marker} [bold]Round {round_num}[/bold] - "
            f"Topic: [cyan]{topic}[/cyan]"
        )

        if is_checkpoint:
            self.console.print(
                "[yellow]   ⚠️  5-Round Checkpoint - User control available[/yellow]"
            )


class EnhancedDashboard:
    """Main dashboard for v1.3 session monitoring."""

    def __init__(self, console: Console):
        self.console = console
        self.layout = self._create_layout()
        self.agent_display = SpecializedAgentDisplay(console)
        self.progress_tracker = EnhancedProgressTracker(console)
        self.last_update = datetime.now()

    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(Layout(name="main", ratio=2), Layout(name="sidebar"))

        layout["body"]["main"].split_column(
            Layout(name="activity", ratio=2), Layout(name="progress")
        )

        return layout

    def update_session_info(self, state: Dict[str, Any]) -> None:
        """Update dashboard with current session state."""
        # Header: Session info
        theme = state.get("main_topic", "Not set")
        phase = state.get("current_phase", 0)
        phase_name = self.progress_tracker.phase_names.get(phase, f"Phase {phase}")

        header_content = Text()
        header_content.append("Virtual Agora Session\n", style="bold cyan")
        header_content.append(f"Theme: {theme} | Phase: {phase_name}", style="dim")

        self.layout["header"].update(Align.center(header_content, vertical="middle"))

        # Sidebar: Agent status
        agent_status = self._build_agent_status(state)
        self.layout["sidebar"].update(agent_status)

        # Main: Current activity
        current_activity = self._build_activity_display(state)
        self.layout["activity"].update(current_activity)

        # Progress
        progress_display = self._build_progress_display(state)
        self.layout["progress"].update(progress_display)

        # Footer
        self._update_footer(state)

        self.last_update = datetime.now()

    def _build_agent_status(self, state: Dict[str, Any]) -> Panel:
        """Build agent status display."""
        specialized = state.get("specialized_agents", {})

        status_lines = ["[bold]Specialized Agents:[/bold]\n"]

        for agent_type in [
            "moderator",
            "summarizer",
            "topic_report",
            "ecclesia_report",
        ]:
            agent_id = specialized.get(agent_type)
            color = self.agent_display.agent_colors.get(agent_type, "white")
            icon = self.agent_display.agent_icons.get(agent_type, "🤖")
            status = "🟢 Active" if agent_id else "🔴 Not initialized"

            status_lines.append(
                f"{icon} [{color}]{agent_type.replace('_', ' ').title()}[/{color}]: {status}"
            )

        discussing_count = len(state.get("agents", {}))
        status_lines.append(f"\n[bold]Discussing Agents:[/bold] {discussing_count}")

        # Add round counter if in discussion phase
        if state.get("current_phase") == 2:
            round_num = state.get("current_round", 0)
            is_checkpoint = round_num > 0 and round_num % 5 == 0
            checkpoint_text = " [yellow](Checkpoint)[/yellow]" if is_checkpoint else ""
            status_lines.append(
                f"\n[bold]Current Round:[/bold] {round_num}{checkpoint_text}"
            )

        return Panel("\n".join(status_lines), title="Agent Status", border_style="blue")

    def _build_activity_display(self, state: Dict[str, Any]) -> Panel:
        """Build current activity display."""
        phase = state.get("current_phase", 0)

        activity_lines = []

        if phase == 0:
            activity_lines.append("[yellow]Initializing session...[/yellow]")
        elif phase == 1:
            activity_lines.append("[cyan]Setting discussion agenda[/cyan]")
            if state.get("proposed_agenda"):
                activity_lines.append("\nProposed topics:")
                for i, topic in enumerate(state["proposed_agenda"][:3], 1):
                    activity_lines.append(f"  {i}. {topic}")
                if len(state.get("proposed_agenda", [])) > 3:
                    activity_lines.append(
                        f"  ... and {len(state['proposed_agenda']) - 3} more"
                    )
        elif phase == 2:
            topic = state.get("active_topic", "Unknown")
            activity_lines.append(f"[green]Discussing:[/green] {topic}")

            # Show recent messages
            recent_messages = state.get("messages", [])[-3:]
            if recent_messages:
                activity_lines.append("\n[dim]Recent comments:[/dim]")
                for msg in recent_messages:
                    speaker = msg.get("speaker_id", "Unknown")[:15]
                    content = msg.get("content", "")[:50] + "..."
                    activity_lines.append(f"  • {speaker}: {content}")
        elif phase == 3:
            activity_lines.append("[yellow]Concluding topic...[/yellow]")
        elif phase == 4:
            activity_lines.append("[blue]Evaluating continuation...[/blue]")
        elif phase == 5:
            activity_lines.append("[magenta]Generating final report...[/magenta]")

        return Panel(
            "\n".join(activity_lines), title="Current Activity", border_style="green"
        )

    def _build_progress_display(self, state: Dict[str, Any]) -> Panel:
        """Build progress display."""
        completed_topics = len(state.get("completed_topics", []))
        total_topics = len(state.get("topic_queue", [])) + completed_topics

        progress_lines = []

        if total_topics > 0:
            progress_lines.append(
                f"Topics: {completed_topics}/{total_topics} completed"
            )

            # Progress bar
            progress_pct = (completed_topics / total_topics) * 100
            bar_width = 20
            filled = int(bar_width * completed_topics / total_topics)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_lines.append(f"[{bar}] {progress_pct:.0f}%")

        # Session duration
        if "start_time" in state:
            duration = datetime.now() - state["start_time"]
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            progress_lines.append(
                f"\nSession time: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )

        return Panel(
            (
                "\n".join(progress_lines)
                if progress_lines
                else "[dim]No progress data[/dim]"
            ),
            title="Progress",
            border_style="cyan",
        )

    def _update_footer(self, state: Dict[str, Any]) -> None:
        """Update footer with status information."""
        footer_text = Text()

        # HITL state
        hitl_state = state.get("hitl_state", {})
        if hitl_state.get("awaiting_approval"):
            footer_text.append("⏳ Awaiting user input", style="yellow")
        else:
            footer_text.append("✓ Running", style="green")

        footer_text.append(" | ", style="dim")
        footer_text.append(
            f"Last update: {format_timestamp(self.last_update, relative=True)}",
            style="dim",
        )
        footer_text.append(" | ", style="dim")
        footer_text.append("Press Ctrl+C for options", style="dim")

        self.layout["footer"].update(Align.center(footer_text, vertical="middle"))

    def render(self) -> Layout:
        """Render the current dashboard state."""
        return self.layout


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
