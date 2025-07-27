"""Live status dashboard for Virtual Agora terminal UI.

This module provides a real-time status dashboard showing session state,
agent activity, and progress information with live updates.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.console import Group
from rich.align import Align
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme, ProviderType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class PhaseType(Enum):
    """Discussion phase types."""

    INITIALIZATION = "initialization"
    AGENDA_SETTING = "agenda_setting"
    DISCUSSION = "discussion"
    VOTING = "voting"
    TOPIC_CONCLUSION = "topic_conclusion"
    AGENDA_MODIFICATION = "agenda_modification"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"


class AgentStatus(Enum):
    """Agent status types."""

    IDLE = "idle"
    RESPONDING = "responding"
    VOTING = "voting"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class AgentInfo:
    """Agent information for dashboard display."""

    agent_id: str
    provider: ProviderType
    status: AgentStatus
    last_activity: datetime
    message_count: int = 0
    error_count: int = 0
    response_time_avg: float = 0.0


@dataclass
class SessionStatus:
    """Session status information."""

    session_id: str
    start_time: datetime
    current_phase: PhaseType
    main_topic: str
    current_round: Optional[int] = None
    current_topic: Optional[str] = None
    total_rounds: int = 0
    completed_topics: List[str] = field(default_factory=list)
    pending_topics: List[str] = field(default_factory=list)
    agents: Dict[str, AgentInfo] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class VirtualAgoraDashboard:
    """Live status dashboard for Virtual Agora sessions."""

    def __init__(self):
        """Initialize dashboard."""
        self.console = get_console()
        self.theme = get_current_theme()
        self._status: Optional[SessionStatus] = None
        self._live: Optional[Live] = None
        self._running = False
        self._update_interval = 1.0  # Update every second

    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        # Split into header, main, and footer
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main"),
            Layout(name="footer", size=4),
        )

        # Split main into left and right
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        # Split left into session and phase info
        layout["left"].split_column(
            Layout(name="session_info", size=8), Layout(name="phase_info")
        )

        # Split right into agents and topics
        layout["right"].split_column(
            Layout(name="agents", size=12), Layout(name="topics")
        )

        return layout

    def generate_header(self) -> Panel:
        """Generate dashboard header."""
        if not self._status:
            return Panel(
                "[red]No session data available[/red]", title="Virtual Agora Dashboard"
            )

        # Calculate session duration
        duration = datetime.now() - self._status.start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Create header content
        header_content = [
            f"[bold cyan]🏛️ Virtual Agora[/bold cyan] • Session: [yellow]{self._status.session_id}[/yellow]",
            f"Duration: [green]{duration_str}[/green] • Phase: [magenta]{self._status.current_phase.value.title()}[/magenta]",
        ]

        if self._status.current_round:
            header_content.append(f"Round: [cyan]{self._status.current_round}[/cyan]")

        content = " • ".join(header_content)

        return Panel(Align.center(content), border_style="cyan", padding=(1, 2))

    def generate_session_info(self) -> Panel:
        """Generate session information panel."""
        if not self._status:
            return Panel("[red]No session data[/red]", title="Session Info")

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Field", style="bold cyan", width=12)
        table.add_column("Value", style="white")

        # Basic session info
        table.add_row(
            "Topic",
            (
                self._status.main_topic[:30] + "..."
                if len(self._status.main_topic) > 30
                else self._status.main_topic
            ),
        )
        table.add_row("Started", self._status.start_time.strftime("%H:%M:%S"))
        table.add_row("Total Rounds", str(self._status.total_rounds))
        table.add_row("Completed", str(len(self._status.completed_topics)))
        table.add_row("Pending", str(len(self._status.pending_topics)))

        # Error count
        error_style = "red" if self._status.errors else "green"
        table.add_row(
            "Errors", f"[{error_style}]{len(self._status.errors)}[/{error_style}]"
        )

        return Panel(table, title="[bold]Session Info[/bold]", border_style="cyan")

    def generate_phase_info(self) -> Panel:
        """Generate current phase information panel."""
        if not self._status:
            return Panel("[red]No phase data[/red]", title="Phase Info")

        content_lines = []

        # Current phase
        phase_name = self._status.current_phase.value.replace("_", " ").title()
        content_lines.append(f"[bold magenta]{phase_name}[/bold magenta]")

        # Current context
        if self._status.current_topic:
            content_lines.append(
                f"Topic: [yellow]{self._status.current_topic}[/yellow]"
            )

        if self._status.current_round:
            content_lines.append(f"Round: [cyan]{self._status.current_round}[/cyan]")

        # Phase-specific information
        if self._status.current_phase == PhaseType.DISCUSSION:
            active_agents = sum(
                1
                for agent in self._status.agents.values()
                if agent.status in [AgentStatus.RESPONDING, AgentStatus.VOTING]
            )
            content_lines.append(f"Active Agents: [green]{active_agents}[/green]")

        elif self._status.current_phase == PhaseType.VOTING:
            voted_agents = sum(
                1
                for agent in self._status.agents.values()
                if agent.status == AgentStatus.COMPLETE
            )
            total_agents = len(self._status.agents)
            content_lines.append(
                f"Votes: [green]{voted_agents}[/green]/[cyan]{total_agents}[/cyan]"
            )

        content = "\n".join(content_lines)

        return Panel(
            content, title="[bold]Current Phase[/bold]", border_style="magenta"
        )

    def generate_agents_panel(self) -> Panel:
        """Generate agents status panel."""
        if not self._status or not self._status.agents:
            return Panel("[yellow]No agents configured[/yellow]", title="Agents")

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            title="[bold]Agent Status[/bold]",
        )

        table.add_column("Agent", width=15)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Messages", width=8, justify="center")
        table.add_column("Last Seen", width=12)

        # Sort agents by provider and then by name
        sorted_agents = sorted(
            self._status.agents.values(), key=lambda a: (a.provider.value, a.agent_id)
        )

        for agent in sorted_agents:
            # Get agent colors
            colors = self.theme.assign_agent_color(agent.agent_id, agent.provider)

            # Format agent name with color and symbol
            agent_display = f"[{colors['primary']}]{colors['symbol']} {agent.agent_id}[/{colors['primary']}]"

            # Status with color coding
            status_colors = {
                AgentStatus.IDLE: "dim",
                AgentStatus.RESPONDING: "yellow",
                AgentStatus.VOTING: "cyan",
                AgentStatus.ERROR: "red",
                AgentStatus.COMPLETE: "green",
            }

            status_color = status_colors.get(agent.status, "white")
            status_display = (
                f"[{status_color}]{agent.status.value.title()}[/{status_color}]"
            )

            # Last activity
            time_diff = datetime.now() - agent.last_activity
            if time_diff.seconds < 60:
                last_seen = "Just now"
            elif time_diff.seconds < 3600:
                last_seen = f"{time_diff.seconds // 60}m ago"
            else:
                last_seen = f"{time_diff.seconds // 3600}h ago"

            table.add_row(
                agent_display,
                status_display,
                str(agent.message_count),
                f"[dim]{last_seen}[/dim]",
            )

        return Panel(table, border_style="green")

    def generate_topics_panel(self) -> Panel:
        """Generate topics status panel."""
        if not self._status:
            return Panel("[red]No topic data[/red]", title="Topics")

        # Create topics table
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")

        table.add_column("Topic", style="white")
        table.add_column("Status", width=10, justify="center")

        # Completed topics
        for topic in self._status.completed_topics:
            display_topic = topic[:25] + "..." if len(topic) > 25 else topic
            table.add_row(display_topic, "[green]✓ Done[/green]")

        # Current topic
        if self._status.current_topic:
            display_topic = (
                self._status.current_topic[:25] + "..."
                if len(self._status.current_topic) > 25
                else self._status.current_topic
            )
            table.add_row(display_topic, "[yellow]🎯 Current[/yellow]")

        # Pending topics
        for topic in self._status.pending_topics:
            display_topic = topic[:25] + "..." if len(topic) > 25 else topic
            table.add_row(display_topic, "[dim]⏳ Pending[/dim]")

        if not (
            self._status.completed_topics
            or self._status.current_topic
            or self._status.pending_topics
        ):
            table.add_row("[dim]No topics configured[/dim]", "")

        return Panel(
            table, title="[bold]Discussion Topics[/bold]", border_style="yellow"
        )

    def generate_footer(self) -> Panel:
        """Generate dashboard footer with controls."""
        content = [
            "[dim]Dashboard Updates: Live[/dim]",
            "[dim]Press Ctrl+C to exit dashboard mode[/dim]",
        ]

        if self._status and self._status.errors:
            content.append(f"[red]⚠️ {len(self._status.errors)} error(s) detected[/red]")

        return Panel(" • ".join(content), border_style="dim", padding=(0, 1))

    def generate_dashboard(self) -> Layout:
        """Generate complete dashboard layout."""
        layout = self.create_layout()

        # Populate all sections
        layout["header"].update(self.generate_header())
        layout["session_info"].update(self.generate_session_info())
        layout["phase_info"].update(self.generate_phase_info())
        layout["agents"].update(self.generate_agents_panel())
        layout["topics"].update(self.generate_topics_panel())
        layout["footer"].update(self.generate_footer())

        return layout

    def update_session_status(self, status: SessionStatus) -> None:
        """Update the session status data."""
        self._status = status

    def update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update individual agent status."""
        if self._status and agent_id in self._status.agents:
            self._status.agents[agent_id].status = status
            self._status.agents[agent_id].last_activity = datetime.now()

    def increment_agent_messages(self, agent_id: str) -> None:
        """Increment message count for an agent."""
        if self._status and agent_id in self._status.agents:
            self._status.agents[agent_id].message_count += 1
            self._status.agents[agent_id].last_activity = datetime.now()

    def add_error(self, error_message: str) -> None:
        """Add an error to the dashboard."""
        if self._status:
            self._status.errors.append(error_message)

    def set_phase(self, phase: PhaseType) -> None:
        """Update the current phase."""
        if self._status:
            self._status.current_phase = phase

    def set_current_topic(self, topic: str, round_number: Optional[int] = None) -> None:
        """Update current topic and round."""
        if self._status:
            self._status.current_topic = topic
            if round_number:
                self._status.current_round = round_number

    def complete_topic(self, topic: str) -> None:
        """Mark a topic as completed."""
        if self._status:
            if topic in self._status.pending_topics:
                self._status.pending_topics.remove(topic)
            if topic not in self._status.completed_topics:
                self._status.completed_topics.append(topic)

    @asynccontextmanager
    async def live_dashboard(self):
        """Context manager for live dashboard display."""
        if not self._status:
            raise ValueError(
                "No session status available. Call update_session_status() first."
            )

        self._running = True

        with Live(
            self.generate_dashboard(),
            refresh_per_second=1,
            console=self.console.rich_console,
            screen=False,
        ) as live:
            self._live = live

            # Start update task
            async def update_loop():
                while self._running:
                    try:
                        live.update(self.generate_dashboard())
                        await asyncio.sleep(self._update_interval)
                    except Exception as e:
                        logger.error(f"Dashboard update error: {e}")
                        break

            update_task = asyncio.create_task(update_loop())

            try:
                yield self
            finally:
                self._running = False
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass

                self._live = None

    def show_snapshot(self) -> None:
        """Show a static snapshot of the dashboard."""
        if not self._status:
            self.console.print_system_message(
                "No session status available", MessageType.WARNING
            )
            return

        dashboard = self.generate_dashboard()
        self.console.print(dashboard)

    def is_running(self) -> bool:
        """Check if dashboard is currently running."""
        return self._running


# Global dashboard instance
_dashboard: Optional[VirtualAgoraDashboard] = None


def get_dashboard() -> VirtualAgoraDashboard:
    """Get the global dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = VirtualAgoraDashboard()
    return _dashboard


# Convenience functions


def create_session_status(
    session_id: str, main_topic: str, agents: Dict[str, ProviderType]
) -> SessionStatus:
    """Create initial session status."""
    agent_info = {}
    for agent_id, provider in agents.items():
        agent_info[agent_id] = AgentInfo(
            agent_id=agent_id,
            provider=provider,
            status=AgentStatus.IDLE,
            last_activity=datetime.now(),
        )

    return SessionStatus(
        session_id=session_id,
        start_time=datetime.now(),
        current_phase=PhaseType.INITIALIZATION,
        main_topic=main_topic,
        agents=agent_info,
    )


def update_dashboard_status(status: SessionStatus) -> None:
    """Update dashboard with new status."""
    get_dashboard().update_session_status(status)


def update_agent_status(agent_id: str, status: AgentStatus) -> None:
    """Update agent status in dashboard."""
    get_dashboard().update_agent_status(agent_id, status)


def set_dashboard_phase(phase: PhaseType) -> None:
    """Set current phase in dashboard."""
    get_dashboard().set_phase(phase)


def show_dashboard_snapshot() -> None:
    """Show a static dashboard snapshot."""
    get_dashboard().show_snapshot()


async def run_live_dashboard() -> None:
    """Run the live dashboard in an async context."""
    dashboard = get_dashboard()
    async with dashboard.live_dashboard():
        # Dashboard will run until cancelled
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass


# Dashboard context managers and utilities


class DashboardManager:
    """Manager for dashboard lifecycle and updates."""

    def __init__(self):
        self.dashboard = get_dashboard()
        self._session_status: Optional[SessionStatus] = None

    def initialize_session(
        self, session_id: str, main_topic: str, agents: Dict[str, ProviderType]
    ) -> None:
        """Initialize dashboard for a new session."""
        self._session_status = create_session_status(session_id, main_topic, agents)
        self.dashboard.update_session_status(self._session_status)

    def agent_started_responding(self, agent_id: str) -> None:
        """Mark agent as responding."""
        self.dashboard.update_agent_status(agent_id, AgentStatus.RESPONDING)

    def agent_completed_response(self, agent_id: str) -> None:
        """Mark agent response as complete."""
        self.dashboard.update_agent_status(agent_id, AgentStatus.COMPLETE)
        self.dashboard.increment_agent_messages(agent_id)

    def agent_error(self, agent_id: str, error: str) -> None:
        """Mark agent as having an error."""
        self.dashboard.update_agent_status(agent_id, AgentStatus.ERROR)
        self.dashboard.add_error(f"{agent_id}: {error}")

    def start_voting(self) -> None:
        """Start voting phase."""
        self.dashboard.set_phase(PhaseType.VOTING)
        for agent_id in (
            self._session_status.agents.keys() if self._session_status else []
        ):
            self.dashboard.update_agent_status(agent_id, AgentStatus.VOTING)

    def complete_voting(self) -> None:
        """Complete voting phase."""
        for agent_id in (
            self._session_status.agents.keys() if self._session_status else []
        ):
            self.dashboard.update_agent_status(agent_id, AgentStatus.IDLE)

    async def run_with_dashboard(self, main_coroutine):
        """Run main application logic with live dashboard."""
        async with self.dashboard.live_dashboard():
            return await main_coroutine
