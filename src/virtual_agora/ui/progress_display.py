"""Progress visualization system for Virtual Agora.

This module provides comprehensive progress tracking and visualization
for agenda progression, round completion, and overall session progress.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.columns import Columns
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ProgressType(Enum):
    """Types of progress to track."""

    AGENDA = "agenda"
    ROUNDS = "rounds"
    SESSION = "session"
    TOPIC = "topic"


class TopicStatus(Enum):
    """Status of individual topics."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ProgressVisualizer:
    """Advanced progress visualization system."""

    def __init__(self):
        """Initialize progress visualizer."""
        self.console = get_console()
        self.theme = get_current_theme()

    def show_agenda_progress(
        self,
        agenda: List[str],
        current_topic: Optional[str] = None,
        completed_topics: List[str] = None,
        skipped_topics: List[str] = None,
        show_estimated_time: bool = True,
    ) -> None:
        """Show comprehensive agenda progress with visual indicators.

        Args:
            agenda: Complete list of agenda topics
            current_topic: Currently active topic
            completed_topics: List of completed topics
            skipped_topics: List of skipped topics
            show_estimated_time: Whether to show time estimates
        """
        completed_topics = completed_topics or []
        skipped_topics = skipped_topics or []

        # Calculate progress statistics
        total_topics = len(agenda)
        completed_count = len(completed_topics)
        progress_pct = (completed_count / total_topics * 100) if total_topics > 0 else 0

        # Create progress bar
        progress_bar = self._create_visual_progress_bar(progress_pct, width=40)

        # Create agenda table
        table = Table(
            title="[bold cyan]📋 Discussion Agenda Progress[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_style="bold cyan",
        )

        table.add_column("#", width=3, justify="center")
        table.add_column("Topic", style="white", min_width=40)
        table.add_column("Status", justify="center", width=15)
        table.add_column("Progress", justify="center", width=12)

        for i, topic in enumerate(agenda, 1):
            # Determine status
            if topic in completed_topics:
                status = TopicStatus.COMPLETED
                status_display = "[green]✅ Complete[/green]"
                progress_display = "[green]100%[/green]"
                topic_style = "dim"
            elif topic in skipped_topics:
                status = TopicStatus.SKIPPED
                status_display = "[yellow]⏭️ Skipped[/yellow]"
                progress_display = "[yellow]--[/yellow]"
                topic_style = "dim"
            elif topic == current_topic:
                status = TopicStatus.ACTIVE
                status_display = "[bold yellow]🎯 Active[/bold yellow]"
                progress_display = "[yellow]In Progress[/yellow]"
                topic_style = "bold yellow"
            else:
                status = TopicStatus.PENDING
                status_display = "[dim]⏳ Pending[/dim]"
                progress_display = "[dim]0%[/dim]"
                topic_style = "dim"

            # Format topic with appropriate style
            topic_formatted = f"[{topic_style}]{topic}[/{topic_style}]"

            table.add_row(str(i), topic_formatted, status_display, progress_display)

        # Create summary panel
        summary_lines = [
            f"[bold]Overall Progress:[/bold] {progress_bar} {progress_pct:.1f}%",
            "",
            f"[bold]Completed:[/bold] {completed_count}/{total_topics} topics",
            f"[bold]Remaining:[/bold] {total_topics - completed_count - len(skipped_topics)} topics",
        ]

        if skipped_topics:
            summary_lines.append(f"[bold]Skipped:[/bold] {len(skipped_topics)} topics")

        if show_estimated_time and completed_count > 0:
            # Simple time estimation (could be enhanced with actual timing data)
            avg_time_per_topic = 15  # minutes, could be dynamic
            remaining_topics = total_topics - completed_count - len(skipped_topics)
            estimated_time = remaining_topics * avg_time_per_topic

            if estimated_time > 60:
                time_str = f"{estimated_time // 60}h {estimated_time % 60}m"
            else:
                time_str = f"{estimated_time}m"

            summary_lines.append(f"[bold]Estimated Remaining:[/bold] {time_str}")

        summary_panel = Panel(
            "\n".join(summary_lines),
            title="[bold]📊 Progress Summary[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )

        # Display both panels
        layout = Columns(
            [Panel(table, border_style="cyan", padding=(1, 1)), summary_panel],
            equal=False,
            expand=True,
        )

        self.console.print()
        self.console.print(layout)
        self.console.print()

    def show_round_progress(
        self,
        current_round: int,
        total_expected_rounds: int,
        topic: str,
        participants: List[str],
        messages_count: int = 0,
    ) -> None:
        """Show progress within the current topic's rounds.

        Args:
            current_round: Current round number
            total_expected_rounds: Expected total rounds for this topic
            topic: Current topic being discussed
            participants: List of participants
            messages_count: Number of messages in current round
        """
        # Calculate round progress
        round_progress = (
            (current_round / total_expected_rounds * 100)
            if total_expected_rounds > 0
            else 0
        )
        progress_bar = self._create_visual_progress_bar(round_progress, width=30)

        # Create round visualization
        round_display = []
        for i in range(1, max(total_expected_rounds + 1, current_round + 1)):
            if i < current_round:
                round_display.append("[green]●[/green]")  # Completed
            elif i == current_round:
                round_display.append("[yellow]●[/yellow]")  # Current
            else:
                round_display.append("[dim]○[/dim]")  # Future

        content_lines = [
            f"[bold cyan]Round {current_round}[/bold cyan] of {total_expected_rounds}",
            "",
            f"[bold]Topic:[/bold] {topic}",
            "",
            f"[bold]Progress:[/bold] {progress_bar} {round_progress:.0f}%",
            "",
            f"[bold]Round Timeline:[/bold] {' '.join(round_display)}",
            "",
            f"[bold]Participants:[/bold] {len(participants)} active",
            f"[bold]Messages:[/bold] {messages_count} in current round",
        ]

        panel = Panel(
            "\n".join(content_lines),
            title="[bold]🔄 Round Progress[/bold]",
            title_align="center",
            border_style="yellow",
            padding=(1, 2),
            width=min(self.console.get_width() - 10, 80),
        )

        self.console.print(panel)

    def show_session_overview(
        self,
        session_start: datetime,
        total_topics: int,
        completed_topics: int,
        current_topic: str,
        total_rounds: int,
        total_messages: int,
        participants_count: int,
    ) -> None:
        """Show comprehensive session progress overview.

        Args:
            session_start: When the session started
            total_topics: Total number of agenda topics
            completed_topics: Number of completed topics
            current_topic: Currently active topic
            total_rounds: Total rounds completed
            total_messages: Total messages sent
            participants_count: Number of participants
        """
        # Calculate session duration
        duration = datetime.now() - session_start
        duration_str = self._format_duration(duration)

        # Calculate progress
        topic_progress = (
            (completed_topics / total_topics * 100) if total_topics > 0 else 0
        )
        topic_bar = self._create_visual_progress_bar(topic_progress, width=25)

        # Create overview table
        overview_table = Table(box=None, show_header=False, padding=(0, 2))
        overview_table.add_column("Metric", style="bold cyan", width=20)
        overview_table.add_column("Value", style="white")

        overview_table.add_row("Session Duration", duration_str)
        overview_table.add_row(
            "Topic Progress", f"{topic_bar} {completed_topics}/{total_topics}"
        )
        overview_table.add_row("Current Topic", current_topic)
        overview_table.add_row("Total Rounds", str(total_rounds))
        overview_table.add_row("Total Messages", str(total_messages))
        overview_table.add_row("Participants", str(participants_count))

        # Calculate session statistics
        avg_messages_per_round = (
            total_messages / total_rounds if total_rounds > 0 else 0
        )
        avg_rounds_per_topic = total_rounds / max(completed_topics, 1)

        stats_table = Table(box=None, show_header=False, padding=(0, 2))
        stats_table.add_column("Statistic", style="bold green", width=20)
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Avg Msgs/Round", f"{avg_messages_per_round:.1f}")
        stats_table.add_row("Avg Rounds/Topic", f"{avg_rounds_per_topic:.1f}")

        if completed_topics > 0:
            avg_time_per_topic = (
                duration.total_seconds() / completed_topics / 60
            )  # minutes
            stats_table.add_row("Avg Time/Topic", f"{avg_time_per_topic:.1f}m")

        # Combine into columns
        layout = Columns(
            [
                Panel(
                    overview_table,
                    title="[bold]📈 Session Overview[/bold]",
                    border_style="cyan",
                ),
                Panel(
                    stats_table,
                    title="[bold]📊 Statistics[/bold]",
                    border_style="green",
                ),
            ],
            equal=True,
            expand=True,
        )

        self.console.print()
        self.console.print(layout)
        self.console.print()

    def show_mini_progress_indicator(
        self,
        current_step: int,
        total_steps: int,
        step_name: str,
        show_percentage: bool = True,
    ) -> None:
        """Show a compact progress indicator for current operations.

        Args:
            current_step: Current step number
            total_steps: Total number of steps
            step_name: Name of current step
            show_percentage: Whether to show percentage
        """
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        mini_bar = self._create_visual_progress_bar(progress_pct, width=20)

        content = f"{mini_bar} {step_name}"
        if show_percentage:
            content += f" ({progress_pct:.0f}%)"

        self.console.print(f"[cyan]⚡ {content}[/cyan]")

    def _create_visual_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual progress bar.

        Args:
            percentage: Progress percentage (0-100)
            width: Width in characters

        Returns:
            Formatted progress bar string
        """
        filled = int(percentage / 100 * width)
        empty = width - filled

        # Use different colors based on progress
        if percentage >= 90:
            color = "bright_green"
        elif percentage >= 70:
            color = "green"
        elif percentage >= 50:
            color = "yellow"
        elif percentage >= 25:
            color = "orange1"
        else:
            color = "red"

        bar = f"[{color}]{'█' * filled}[/{color}]{'░' * empty}"
        return bar

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for display.

        Args:
            duration: Time duration

        Returns:
            Formatted duration string
        """
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


# Global progress visualizer instance
_progress_visualizer: Optional[ProgressVisualizer] = None


def get_progress_visualizer() -> ProgressVisualizer:
    """Get the global progress visualizer instance."""
    global _progress_visualizer
    if _progress_visualizer is None:
        _progress_visualizer = ProgressVisualizer()
    return _progress_visualizer


# Convenience functions for easy integration
def show_agenda_progress(
    agenda: List[str],
    current_topic: Optional[str] = None,
    completed_topics: List[str] = None,
    skipped_topics: List[str] = None,
) -> None:
    """Show agenda progress visualization."""
    get_progress_visualizer().show_agenda_progress(
        agenda, current_topic, completed_topics, skipped_topics
    )


def show_round_progress(
    current_round: int,
    total_expected_rounds: int,
    topic: str,
    participants: List[str],
    messages_count: int = 0,
) -> None:
    """Show round progress visualization."""
    get_progress_visualizer().show_round_progress(
        current_round, total_expected_rounds, topic, participants, messages_count
    )


def show_session_overview(
    session_start: datetime,
    total_topics: int,
    completed_topics: int,
    current_topic: str,
    total_rounds: int,
    total_messages: int,
    participants_count: int,
) -> None:
    """Show session overview."""
    get_progress_visualizer().show_session_overview(
        session_start,
        total_topics,
        completed_topics,
        current_topic,
        total_rounds,
        total_messages,
        participants_count,
    )


def show_mini_progress(current_step: int, total_steps: int, step_name: str) -> None:
    """Show mini progress indicator."""
    get_progress_visualizer().show_mini_progress_indicator(
        current_step, total_steps, step_name
    )
