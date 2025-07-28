"""Atmospheric elements for Virtual Agora assembly experience.

This module provides visual effects and indicators that create an immersive,
engaging atmosphere for users watching the democratic assembly.
"""

import time
import threading
from typing import Optional, List
from datetime import datetime
from contextlib import contextmanager
from enum import Enum

from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.console import Group
from rich.layout import Layout

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme
from virtual_agora.providers.config import ProviderType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class IndicatorType(Enum):
    """Types of atmospheric indicators."""

    THINKING = "thinking"
    SPEAKING = "speaking"
    VOTING = "voting"
    PROCESSING = "processing"
    WAITING = "waiting"


class AtmosphericIndicator:
    """Manages atmospheric indicators and effects."""

    def __init__(self):
        """Initialize atmospheric indicator."""
        self.console = get_console()
        self.theme = get_current_theme()
        self._active_indicators: List[str] = []
        self._live_display: Optional[Live] = None

    @contextmanager
    def thinking_indicator(
        self, agent_id: str, message: str = "Considering the discussion..."
    ):
        """Context manager for showing thinking indicator.

        Args:
            agent_id: Agent currently thinking
            message: Thinking message to display
        """
        # Get agent colors
        try:
            from virtual_agora.flow.nodes_v13 import get_provider_type_from_agent_id

            provider = get_provider_type_from_agent_id(agent_id)
            colors = self.theme.assign_agent_color(agent_id, provider)
        except:
            colors = {"primary": "cyan", "accent": "yellow"}

        # Create thinking panel
        thinking_text = f"[{colors['primary']}]💭 {agent_id}[/{colors['primary']}] [dim]is {message}[/dim]"

        panel = Panel(
            Align.center(thinking_text),
            border_style=colors.get("border", "dim"),
            padding=(0, 2),
            width=min(self.console.get_width() - 20, 60),
        )

        # Show indicator with proper spacing - ensure clean line break from any active spinner
        self.console.print("\n")  # Force a new line to break from any transient spinner
        self.console.print(panel)
        self._active_indicators.append(f"thinking_{agent_id}")

        try:
            yield
        finally:
            # Remove indicator
            if f"thinking_{agent_id}" in self._active_indicators:
                self._active_indicators.remove(f"thinking_{agent_id}")

            # Clear the thinking indicator with a subtle "done" message
            done_text = f"[{colors['primary']}]✓ {agent_id}[/{colors['primary']}] [dim]ready[/dim]"
            done_panel = Panel(
                Align.center(done_text),
                border_style="dim",
                padding=(0, 2),
                width=min(self.console.get_width() - 20, 60),
            )
            self.console.print(
                "\n"
            )  # Force a new line to break from any transient spinner
            self.console.print(done_panel)
            self.console.print()  # Line break after indicator

            # Brief pause for natural timing
            time.sleep(0.5)

    def show_speaking_queue(self, current_speaker: str, queue: List[str]) -> None:
        """Show who's currently speaking and the upcoming queue.

        Args:
            current_speaker: Agent currently speaking
            queue: List of agents in speaking queue
        """
        content_lines = []

        # Current speaker with animation
        if current_speaker:
            try:
                from virtual_agora.flow.nodes_v13 import get_provider_type_from_agent_id

                provider = get_provider_type_from_agent_id(current_speaker)
                colors = self.theme.assign_agent_color(current_speaker, provider)
            except:
                colors = {"primary": "green"}

            content_lines.append(
                f"[bold green]🎤 Speaking:[/bold green] [{colors['primary']}]{current_speaker}[/{colors['primary']}]"
            )

        # Queue with position indicators
        if queue:
            content_lines.append("")
            content_lines.append("[bold]⏳ Speaking Queue:[/bold]")

            for i, agent in enumerate(queue[:3], 1):  # Show next 3 speakers
                try:
                    from virtual_agora.flow.nodes_v13 import (
                        get_provider_type_from_agent_id,
                    )

                    provider = get_provider_type_from_agent_id(agent)
                    colors = self.theme.assign_agent_color(agent, provider)
                except:
                    colors = {"primary": "white"}

                content_lines.append(
                    f"  {i}. [{colors['primary']}]{agent}[/{colors['primary']}]"
                )

            if len(queue) > 3:
                content_lines.append(f"  ... and {len(queue) - 3} more")

        if content_lines:
            panel = Panel(
                "\n".join(content_lines),
                title="[bold]🗣️ Assembly Floor[/bold]",
                border_style="green",
                padding=(1, 2),
                width=min(self.console.get_width() - 20, 50),
            )

            self.console.print()
            self.console.print(panel)
            self.console.print()

    def show_phase_transition(
        self, from_phase: str, to_phase: str, details: str = "", animate: bool = True
    ) -> None:
        """Show an animated phase transition.

        Args:
            from_phase: Phase transitioning from
            to_phase: Phase transitioning to
            details: Optional transition details
            animate: Whether to show animation
        """
        if animate:
            # Show animated transition
            self._animated_phase_transition(from_phase, to_phase, details)
        else:
            # Show simple transition
            self._simple_phase_transition(from_phase, to_phase, details)

    def _animated_phase_transition(
        self, from_phase: str, to_phase: str, details: str
    ) -> None:
        """Show animated phase transition with progress effect."""

        # Create transition steps
        steps = [
            f"Concluding {from_phase}...",
            "Preparing for transition...",
            f"Initiating {to_phase}...",
            f"Welcome to {to_phase}!",
        ]

        for i, step in enumerate(steps):
            # Create progress bar
            progress = int((i + 1) / len(steps) * 100)
            progress_bar = "█" * (progress // 5) + "░" * (20 - progress // 5)

            content = (
                f"[yellow]{step}[/yellow]\n\n[cyan]{progress_bar}[/cyan] {progress}%"
            )

            if details and i == len(steps) - 1:
                content += f"\n[dim]{details}[/dim]"

            panel = Panel(
                Align.center(content),
                title="[bold]🔄 Phase Transition[/bold]",
                title_align="center",
                border_style="yellow",
                padding=(1, 2),
                width=min(self.console.get_width() - 10, 70),
            )

            if i == 0:  # Add line break before first step
                self.console.print()
            self.console.print(panel)
            if i == len(steps) - 1:  # Add line break after final step
                self.console.print()
            time.sleep(0.8 if i < len(steps) - 1 else 1.5)  # Longer pause on final step

    def _simple_phase_transition(
        self, from_phase: str, to_phase: str, details: str
    ) -> None:
        """Show simple phase transition without animation."""
        content = f"[yellow]Transitioning from[/yellow] [bold]{from_phase}[/bold] [yellow]to[/yellow] [bold green]{to_phase}[/bold green]"

        if details:
            content += f"\n[dim]{details}[/dim]"

        panel = Panel(
            Align.center(content),
            title="[bold]🔄 Phase Transition[/bold]",
            title_align="center",
            border_style="yellow",
            padding=(1, 2),
            width=min(self.console.get_width() - 10, 80),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def show_voting_tension(
        self,
        topic: str,
        votes_collected: int,
        total_voters: int,
        suspense_level: str = "medium",
    ) -> None:
        """Show voting progress with tension and suspense.

        Args:
            topic: Topic being voted on
            votes_collected: Number of votes collected so far
            total_voters: Total number of voters
            suspense_level: Level of suspense (low, medium, high)
        """
        # Calculate progress
        progress_pct = (votes_collected / total_voters * 100) if total_voters > 0 else 0
        progress_bar = "█" * int(progress_pct // 5) + "░" * (
            20 - int(progress_pct // 5)
        )

        # Suspense elements based on level
        if suspense_level == "high":
            title = "[bold red]🗳️ Critical Vote in Progress[/bold red]"
            border = "red"
            status_text = "[red]Every vote matters![/red]"
        elif suspense_level == "medium":
            title = "[bold yellow]🗳️ Vote in Progress[/bold yellow]"
            border = "yellow"
            status_text = "[yellow]Collecting decisions...[/yellow]"
        else:
            title = "[bold]🗳️ Voting[/bold]"
            border = "cyan"
            status_text = "[cyan]Gathering votes...[/cyan]"

        content_lines = [
            f"[bold]Topic:[/bold] {topic}",
            "",
            f"[bold]Progress:[/bold] [{border}]{progress_bar}[/{border}] {votes_collected}/{total_voters} votes",
            "",
            status_text,
        ]

        panel = Panel(
            "\n".join(content_lines),
            title=title,
            title_align="center",
            border_style=border,
            padding=(1, 2),
            width=min(self.console.get_width() - 10, 80),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def show_round_announcement(
        self, round_number: int, topic: str, participants: List[str]
    ) -> None:
        """Show dramatic round announcement.

        Args:
            round_number: Round number
            topic: Topic being discussed
            participants: List of participating agents
        """
        # Create participant list with colors
        participant_list = []
        for agent in participants:
            try:
                from virtual_agora.flow.nodes_v13 import get_provider_type_from_agent_id

                provider = get_provider_type_from_agent_id(agent)
                colors = self.theme.assign_agent_color(agent, provider)
                participant_list.append(
                    f"[{colors['primary']}]{agent}[/{colors['primary']}]"
                )
            except:
                participant_list.append(agent)

        content_lines = [
            f"[bold cyan]Round {round_number}[/bold cyan]",
            "",
            f"[bold]Topic:[/bold] {topic}",
            "",
            f"[bold]Participants:[/bold] {', '.join(participant_list)}",
        ]

        panel = Panel(
            Align.center("\n".join(content_lines)),
            title="[bold magenta]🏛️ New Discussion Round[/bold magenta]",
            title_align="center",
            border_style="magenta",
            padding=(2, 3),
            width=min(self.console.get_width() - 5, 90),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

        # Brief dramatic pause
        time.sleep(1.0)

    def show_agent_status_update(
        self, agent_id: str, old_status: str, new_status: str
    ) -> None:
        """Show a subtle agent status update.

        Args:
            agent_id: Agent identifier
            old_status: Previous status
            new_status: New status
        """
        try:
            from virtual_agora.flow.nodes_v13 import get_provider_type_from_agent_id

            provider = get_provider_type_from_agent_id(agent_id)
            colors = self.theme.assign_agent_color(agent_id, provider)
        except:
            colors = {"primary": "white"}

        # Status icons
        status_icons = {
            "Ready": "⏳",
            "Speaking": "🎤",
            "Voting": "🗳️",
            "Thinking": "💭",
            "Muted": "🔇",
        }

        old_icon = status_icons.get(old_status, "•")
        new_icon = status_icons.get(new_status, "•")

        update_text = f"[{colors['primary']}]{agent_id}[/{colors['primary']}]: {old_icon} {old_status} → {new_icon} {new_status}"

        # Only show if it's a meaningful change
        if old_status != new_status and new_status != "Ready":
            self.console.print(f"[dim]{update_text}[/dim]")


# Global atmospheric indicator instance
_atmospheric: Optional[AtmosphericIndicator] = None


def get_atmospheric() -> AtmosphericIndicator:
    """Get the global atmospheric indicator instance."""
    global _atmospheric
    if _atmospheric is None:
        _atmospheric = AtmosphericIndicator()
    return _atmospheric


# Convenience functions for easy integration
def show_thinking(agent_id: str, message: str = "Considering the discussion..."):
    """Context manager for showing thinking indicator."""
    return get_atmospheric().thinking_indicator(agent_id, message)


def show_speaking_queue(current_speaker: str, queue: List[str]) -> None:
    """Show current speaker and speaking queue."""
    get_atmospheric().show_speaking_queue(current_speaker, queue)


def show_phase_transition(
    from_phase: str, to_phase: str, details: str = "", animate: bool = True
) -> None:
    """Show phase transition with optional animation."""
    get_atmospheric().show_phase_transition(from_phase, to_phase, details, animate)


def show_voting_tension(
    topic: str, votes_collected: int, total_voters: int, suspense_level: str = "medium"
) -> None:
    """Show voting progress with tension."""
    get_atmospheric().show_voting_tension(
        topic, votes_collected, total_voters, suspense_level
    )


def show_round_announcement(
    round_number: int, topic: str, participants: List[str]
) -> None:
    """Show dramatic round announcement."""
    get_atmospheric().show_round_announcement(round_number, topic, participants)


def update_agent_status(agent_id: str, old_status: str, new_status: str) -> None:
    """Show agent status update."""
    get_atmospheric().show_agent_status_update(agent_id, old_status, new_status)
