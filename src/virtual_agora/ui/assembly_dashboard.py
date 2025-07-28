"""Enhanced Assembly Dashboard for Virtual Agora.

This module provides an engaging, television-style dashboard for users watching
the democratic assembly, with progress indicators, participant information,
and atmospheric elements that create an immersive experience.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from rich.rule import Rule
from rich.align import Align
from rich.layout import Layout
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme
from virtual_agora.providers.config import ProviderType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class AssemblyPhase(Enum):
    """Current phase of the assembly."""

    INITIALIZATION = "Initialization"
    TOPIC_PROPOSAL = "Topic Proposal"
    AGENDA_VOTING = "Agenda Setting"
    DISCUSSION = "Discussion"
    VOTING = "Voting"
    CONCLUSION = "Conclusion"


@dataclass
class SessionInfo:
    """Information about the current assembly session."""

    session_id: str
    main_topic: str
    start_time: datetime
    current_phase: AssemblyPhase
    total_agenda_items: int
    completed_agenda_items: int
    current_agenda_item: Optional[str] = None
    total_participants: int = 0
    active_participants: int = 0
    current_round: int = 0
    estimated_completion: Optional[datetime] = None


@dataclass
class ParticipantInfo:
    """Information about an assembly participant."""

    agent_id: str
    provider: ProviderType
    is_active: bool
    message_count: int
    last_activity: Optional[datetime] = None
    current_status: str = "Ready"  # "Ready", "Speaking", "Voting", "Muted"


class AssemblyDashboard:
    """Enhanced dashboard for assembly viewing experience."""

    def __init__(self):
        """Initialize the assembly dashboard."""
        self.console = get_console()
        self.theme = get_current_theme()
        self.session_info: Optional[SessionInfo] = None
        self.participants: Dict[str, ParticipantInfo] = {}
        self._last_update = datetime.now()

    def initialize_session(
        self,
        session_id: str,
        main_topic: str,
        participants: List[Dict[str, Any]],
        agenda_items: List[str],
    ) -> None:
        """Initialize a new assembly session.

        Args:
            session_id: Unique session identifier
            main_topic: Main discussion topic
            participants: List of participant information
            agenda_items: List of agenda items to discuss
        """
        self.session_info = SessionInfo(
            session_id=session_id,
            main_topic=main_topic,
            start_time=datetime.now(),
            current_phase=AssemblyPhase.INITIALIZATION,
            total_agenda_items=len(agenda_items),
            completed_agenda_items=0,
            total_participants=len(participants),
            active_participants=len(participants),
        )

        # Initialize participant information
        self.participants = {}
        for participant in participants:
            agent_id = participant.get("agent_id", "Unknown")
            provider = participant.get("provider", ProviderType.OPENAI)

            self.participants[agent_id] = ParticipantInfo(
                agent_id=agent_id, provider=provider, is_active=True, message_count=0
            )

        logger.info(f"Assembly dashboard initialized for session {session_id}")

    def update_phase(self, phase: AssemblyPhase, details: str = "") -> None:
        """Update the current assembly phase.

        Args:
            phase: New assembly phase
            details: Optional phase details
        """
        if self.session_info:
            old_phase = self.session_info.current_phase
            self.session_info.current_phase = phase
            logger.info(f"Assembly phase updated: {old_phase.value} → {phase.value}")

            # Show phase transition
            self.display_phase_transition(old_phase.value, phase.value, details)

    def update_agenda_progress(
        self, current_item: str, completed_count: int = None
    ) -> None:
        """Update agenda progress information.

        Args:
            current_item: Currently active agenda item
            completed_count: Number of completed agenda items
        """
        if self.session_info:
            self.session_info.current_agenda_item = current_item
            if completed_count is not None:
                self.session_info.completed_agenda_items = completed_count

    def update_participant_status(self, agent_id: str, status: str) -> None:
        """Update a participant's status.

        Args:
            agent_id: Participant identifier
            status: New status ("Speaking", "Voting", "Ready", "Muted")
        """
        if agent_id in self.participants:
            self.participants[agent_id].current_status = status
            self.participants[agent_id].last_activity = datetime.now()

    def record_participant_message(self, agent_id: str) -> None:
        """Record that a participant sent a message.

        Args:
            agent_id: Participant identifier
        """
        if agent_id in self.participants:
            self.participants[agent_id].message_count += 1
            self.participants[agent_id].last_activity = datetime.now()

    def display_session_header(self) -> None:
        """Display the main session header dashboard with enhanced progress visualization."""
        if not self.session_info:
            return

        # Show enhanced progress visualization if available
        try:
            from virtual_agora.ui.progress_display import show_session_overview

            # Get current session statistics
            total_messages = sum(p.message_count for p in self.participants.values())

            show_session_overview(
                session_start=self.session_info.start_time,
                total_topics=self.session_info.total_agenda_items,
                completed_topics=self.session_info.completed_agenda_items,
                current_topic=self.session_info.current_agenda_item
                or "Initializing...",
                total_rounds=self.session_info.current_round,
                total_messages=total_messages,
                participants_count=self.session_info.total_participants,
            )
            return
        except ImportError:
            pass

        # Fallback to original header display
        duration = datetime.now() - self.session_info.start_time
        duration_str = self._format_duration(duration)

        # Calculate progress percentage
        if self.session_info.total_agenda_items > 0:
            progress_pct = (
                self.session_info.completed_agenda_items
                / self.session_info.total_agenda_items
                * 100
            )
        else:
            progress_pct = 0

        # Create main header panel
        header_content = self._create_header_content(duration_str, progress_pct)

        header_panel = Panel(
            header_content,
            title="[bold magenta]🏛️ Virtual Agora - Democratic Assembly[/bold magenta]",
            title_align="center",
            border_style="magenta",
            padding=(1, 2),
            width=min(self.console.get_width(), 120),
        )

        self.console.print()
        self.console.print(header_panel)
        self.console.print()

    def _create_header_content(self, duration_str: str, progress_pct: float) -> str:
        """Create the content for the session header.

        Args:
            duration_str: Formatted session duration
            progress_pct: Progress percentage

        Returns:
            Formatted header content
        """
        lines = []

        # Topic and session info
        lines.append(f"[bold cyan]Topic:[/bold cyan] {self.session_info.main_topic}")
        lines.append("")

        # Progress bar
        progress_bar = self._create_text_progress_bar(progress_pct)
        lines.append(
            f"[bold]Progress:[/bold] {progress_bar} {progress_pct:.0f}% ({self.session_info.completed_agenda_items}/{self.session_info.total_agenda_items} topics)"
        )

        # Current status
        current_item = self.session_info.current_agenda_item or "Setting up..."
        lines.append(f"[bold]Current:[/bold] {current_item}")
        lines.append("")

        # Session stats
        active_count = sum(1 for p in self.participants.values() if p.is_active)
        lines.append(
            f"[bold]Phase:[/bold] {self.session_info.current_phase.value} | [bold]Duration:[/bold] {duration_str} | [bold]Participants:[/bold] {active_count}/{self.session_info.total_participants}"
        )

        if self.session_info.current_round > 0:
            lines.append(f"[bold]Round:[/bold] {self.session_info.current_round}")

        return "\n".join(lines)

    def _create_text_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a text-based progress bar.

        Args:
            percentage: Progress percentage (0-100)
            width: Width of progress bar in characters

        Returns:
            Formatted progress bar string
        """
        filled = int(percentage / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[cyan]{bar}[/cyan]"

    def display_participant_status(self) -> None:
        """Display current participant status."""
        if not self.participants:
            return

        table = Table(
            title="[bold]Assembly Participants[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Participant", style="white", width=20)
        table.add_column("Provider", style="dim", width=12)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Messages", justify="center", width=10)
        table.add_column("Activity", style="dim", width=15)

        # Sort participants by activity
        sorted_participants = sorted(
            self.participants.values(),
            key=lambda p: p.last_activity or datetime.min,
            reverse=True,
        )

        for participant in sorted_participants:
            # Get provider colors
            colors = self.theme.assign_agent_color(
                participant.agent_id, participant.provider
            )

            # Format participant name with color
            name = f"[{colors['primary']}]{participant.agent_id}[/{colors['primary']}]"

            # Format provider
            provider_name = participant.provider.value.title()

            # Format status with appropriate color
            status = participant.current_status
            if status == "Speaking":
                status_formatted = f"[green]🎤 {status}[/green]"
            elif status == "Voting":
                status_formatted = f"[yellow]🗳️ {status}[/yellow]"
            elif status == "Muted":
                status_formatted = f"[red]🔇 {status}[/red]"
            else:
                status_formatted = f"[dim]⏳ {status}[/dim]"

            # Format activity time
            if participant.last_activity:
                activity_delta = datetime.now() - participant.last_activity
                if activity_delta.total_seconds() < 60:
                    activity = "Just now"
                elif activity_delta.total_seconds() < 3600:
                    activity = f"{int(activity_delta.total_seconds() / 60)}m ago"
                else:
                    activity = f"{int(activity_delta.total_seconds() / 3600)}h ago"
            else:
                activity = "No activity"

            table.add_row(
                name,
                provider_name,
                status_formatted,
                str(participant.message_count),
                activity,
            )

        panel = Panel(table, border_style="cyan", padding=(1, 1))

        self.console.print(panel)
        self.console.print()

    def display_phase_transition(
        self, from_phase: str, to_phase: str, details: str = ""
    ) -> None:
        """Display a dramatic phase transition.

        Args:
            from_phase: Previous phase
            to_phase: New phase
            details: Optional transition details
        """
        # Create transition content
        transition_text = f"[yellow]Transitioning from[/yellow] [bold]{from_phase}[/bold] [yellow]to[/yellow] [bold green]{to_phase}[/bold green]"

        if details:
            transition_text += f"\n[dim]{details}[/dim]"

        # Create dramatic panel
        panel = Panel(
            Align.center(transition_text),
            title="[bold]🔄 Phase Transition[/bold]",
            title_align="center",
            border_style="yellow",
            padding=(1, 2),
            width=min(self.console.get_width() - 10, 80),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def display_speaking_queue(self, current_speaker: str, queue: List[str]) -> None:
        """Display who is currently speaking and who's next.

        Args:
            current_speaker: Currently speaking participant
            queue: Upcoming speakers in order
        """
        content_lines = []

        # Current speaker
        if current_speaker in self.participants:
            colors = self.theme.assign_agent_color(
                current_speaker, self.participants[current_speaker].provider
            )
            content_lines.append(
                f"[bold green]🎤 Now Speaking:[/bold green] [{colors['primary']}]{current_speaker}[/{colors['primary']}]"
            )

        # Upcoming speakers
        if queue:
            content_lines.append("")
            content_lines.append("[bold]⏳ Speaking Queue:[/bold]")
            for i, speaker in enumerate(queue[:3], 1):  # Show next 3 speakers
                if speaker in self.participants:
                    colors = self.theme.assign_agent_color(
                        speaker, self.participants[speaker].provider
                    )
                    content_lines.append(
                        f"  {i}. [{colors['primary']}]{speaker}[/{colors['primary']}]"
                    )

        if content_lines:
            panel = Panel(
                "\n".join(content_lines),
                title="[bold]🗣️ Assembly Floor[/bold]",
                border_style="green",
                padding=(1, 2),
                width=min(self.console.get_width() - 20, 60),
            )

            self.console.print(panel)

    def _format_duration(self, duration: timedelta) -> str:
        """Format a duration for display.

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
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


# Global assembly dashboard instance
_assembly_dashboard: Optional[AssemblyDashboard] = None


def get_assembly_dashboard() -> AssemblyDashboard:
    """Get the global assembly dashboard instance."""
    global _assembly_dashboard
    if _assembly_dashboard is None:
        _assembly_dashboard = AssemblyDashboard()
    return _assembly_dashboard


# Convenience functions for easy integration
def initialize_assembly_session(
    session_id: str,
    main_topic: str,
    participants: List[Dict[str, Any]],
    agenda_items: List[str],
) -> None:
    """Initialize the assembly session dashboard."""
    get_assembly_dashboard().initialize_session(
        session_id, main_topic, participants, agenda_items
    )


def update_assembly_phase(phase: AssemblyPhase, details: str = "") -> None:
    """Update the current assembly phase."""
    get_assembly_dashboard().update_phase(phase, details)


def show_session_header() -> None:
    """Display the session header dashboard."""
    get_assembly_dashboard().display_session_header()


def show_participant_status() -> None:
    """Display participant status information."""
    get_assembly_dashboard().display_participant_status()


def record_message(agent_id: str) -> None:
    """Record that a participant sent a message."""
    get_assembly_dashboard().record_participant_message(agent_id)


def update_participant_status(agent_id: str, status: str) -> None:
    """Update a participant's status."""
    get_assembly_dashboard().update_participant_status(agent_id, status)
