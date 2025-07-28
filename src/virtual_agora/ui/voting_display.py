"""Dramatic voting display system for Virtual Agora.

This module provides engaging, suspenseful voting presentations that create
tension and drama around democratic decision-making moments.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme
from virtual_agora.providers.config import ProviderType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class VoteType(Enum):
    """Types of votes in the assembly."""

    AGENDA_SELECTION = "agenda_selection"
    TOPIC_CONCLUSION = "topic_conclusion"
    SESSION_CONTINUATION = "session_continuation"


class VotingDisplay:
    """Dramatic voting display system."""

    def __init__(self):
        """Initialize voting display."""
        self.console = get_console()
        self.theme = get_current_theme()

    def show_voting_announcement(
        self,
        vote_type: VoteType,
        topic: str,
        participants: List[str],
        context: str = "",
    ) -> None:
        """Show dramatic voting announcement.

        Args:
            vote_type: Type of vote being conducted
            topic: Topic or subject of the vote
            participants: List of voting participants
            context: Additional context about the vote
        """
        # Vote type specific messaging
        if vote_type == VoteType.AGENDA_SELECTION:
            title = "[bold yellow]🗳️ Agenda Selection Vote[/bold yellow]"
            description = f"Determining the order of discussion topics"
            urgency = "medium"
        elif vote_type == VoteType.TOPIC_CONCLUSION:
            title = "[bold orange1]🗳️ Topic Conclusion Vote[/bold orange1]"
            description = f"Deciding whether to conclude discussion on: {topic}"
            urgency = "high"
        elif vote_type == VoteType.SESSION_CONTINUATION:
            title = "[bold red]🗳️ Session Continuation Vote[/bold red]"
            description = f"Determining whether to continue or end the session"
            urgency = "critical"
        else:
            title = "[bold]🗳️ Assembly Vote[/bold]"
            description = f"Voting on: {topic}"
            urgency = "medium"

        # Build content
        content_lines = [
            f"[bold]{description}[/bold]",
            "",
            f"[bold]Participants:[/bold] {len(participants)} members",
            f"[bold]Voting Method:[/bold] Majority decision",
        ]

        if context:
            content_lines.extend(["", f"[dim]{context}[/dim]"])

        # Create dramatic panel
        panel = Panel(
            Align.center("\n".join(content_lines)),
            title=title,
            title_align="center",
            border_style="yellow" if urgency != "critical" else "red",
            padding=(2, 3),
            width=min(self.console.get_width() - 5, 85),
        )

        # Display with dramatic timing
        self.console.print()
        self.console.print(panel)
        self.console.print()

        # Brief suspenseful pause
        time.sleep(1.5)

    def show_vote_collection_progress(
        self,
        vote_type: VoteType,
        topic: str,
        votes_collected: List[Dict[str, Any]],
        total_voters: int,
        show_intermediate: bool = True,
    ) -> None:
        """Show vote collection in progress with building suspense.

        Args:
            vote_type: Type of vote
            topic: Topic being voted on
            votes_collected: List of votes collected so far
            total_voters: Total number of expected voters
            show_intermediate: Whether to show intermediate results
        """
        collected_count = len(votes_collected)
        progress_pct = (collected_count / total_voters * 100) if total_voters > 0 else 0

        # Create progress bar with tension
        progress_filled = int(progress_pct // 5)
        progress_bar = "█" * progress_filled + "░" * (20 - progress_filled)

        # Build vote status
        if show_intermediate and votes_collected:
            # Count yes/no votes for topic conclusion
            if vote_type == VoteType.TOPIC_CONCLUSION:
                yes_count = sum(
                    1 for v in votes_collected if v.get("vote", "").lower() == "yes"
                )
                no_count = collected_count - yes_count
                vote_summary = (
                    f"[green]Yes: {yes_count}[/green] | [red]No: {no_count}[/red]"
                )
            else:
                vote_summary = f"[cyan]{collected_count} votes received[/cyan]"
        else:
            vote_summary = "[yellow]Votes being collected...[/yellow]"

        content_lines = [
            f"[bold]Voting Progress[/bold]",
            "",
            f"[cyan]{progress_bar}[/cyan] {collected_count}/{total_voters} votes",
            "",
            vote_summary,
            "",
            "[dim]⏳ Waiting for remaining votes...[/dim]",
        ]

        # Determine urgency styling
        if progress_pct > 80:
            border_style = "green"
            title_style = "[bold green]"
        elif progress_pct > 50:
            border_style = "yellow"
            title_style = "[bold yellow]"
        else:
            border_style = "cyan"
            title_style = "[bold cyan]"

        panel = Panel(
            "\n".join(content_lines),
            title=f"{title_style}🗳️ Vote in Progress{title_style[:-1]}[/{title_style[7:-1]}]",
            title_align="center",
            border_style=border_style,
            padding=(1, 2),
            width=min(self.console.get_width() - 10, 70),
        )

        self.console.print(panel)

        # Brief update pause
        time.sleep(0.5)

    def show_dramatic_vote_reveal(
        self,
        vote_type: VoteType,
        topic: str,
        final_votes: List[Dict[str, Any]],
        result: str,
        details: Dict[str, Any] = None,
    ) -> None:
        """Show dramatic vote results with suspense and fanfare.

        Args:
            vote_type: Type of vote
            topic: Topic that was voted on
            final_votes: Complete list of votes
            result: Final result summary
            details: Additional result details
        """
        # Create suspenseful countdown
        self._show_vote_countdown()

        # Determine result styling
        if vote_type == VoteType.TOPIC_CONCLUSION:
            yes_votes = sum(
                1 for v in final_votes if v.get("vote", "").lower() == "yes"
            )
            no_votes = len(final_votes) - yes_votes
            majority_threshold = (len(final_votes) // 2) + 1

            passed = yes_votes >= majority_threshold

            if passed:
                result_color = "green"
                result_icon = "✅"
                result_text = f"MOTION CARRIES"
                subtitle = f"Discussion on '{topic}' will conclude"
            else:
                result_color = "red"
                result_icon = "❌"
                result_text = f"MOTION FAILS"
                subtitle = f"Discussion on '{topic}' will continue"

            vote_breakdown = f"[green]Yes: {yes_votes}[/green] | [red]No: {no_votes}[/red] | [cyan]Majority needed: {majority_threshold}[/cyan]"

        elif vote_type == VoteType.SESSION_CONTINUATION:
            continue_votes = sum(
                1 for v in final_votes if "continue" in v.get("vote", "").lower()
            )
            end_votes = len(final_votes) - continue_votes

            session_continues = continue_votes > end_votes

            if session_continues:
                result_color = "green"
                result_icon = "▶️"
                result_text = "SESSION CONTINUES"
                subtitle = "The assembly will proceed with remaining topics"
            else:
                result_color = "red"
                result_icon = "⏹️"
                result_text = "SESSION CONCLUDES"
                subtitle = "The assembly will now prepare final reports"

            vote_breakdown = f"[green]Continue: {continue_votes}[/green] | [red]End: {end_votes}[/red]"

        else:
            # Generic vote result
            result_color = "cyan"
            result_icon = "🗳️"
            result_text = "VOTE COMPLETE"
            subtitle = result
            vote_breakdown = f"[cyan]{len(final_votes)} votes counted[/cyan]"

        # Create dramatic result display
        result_content = [
            "",
            f"[bold {result_color}]{result_icon} {result_text} {result_icon}[/bold {result_color}]",
            "",
            f"[{result_color}]{subtitle}[/{result_color}]",
            "",
            vote_breakdown,
            "",
        ]

        # Add detailed vote table if requested
        if details and details.get("show_details", False):
            result_content.extend(self._format_detailed_votes(final_votes))

        panel = Panel(
            Align.center("\n".join(result_content)),
            title=f"[bold {result_color}]🏛️ ASSEMBLY DECISION 🏛️[/bold {result_color}]",
            title_align="center",
            border_style=result_color,
            padding=(2, 3),
            width=min(self.console.get_width() - 5, 90),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

        # Victory/conclusion pause
        time.sleep(2.0)

    def _show_vote_countdown(self) -> None:
        """Show dramatic countdown before revealing results."""
        countdown_messages = [
            "📊 Tallying votes...",
            "🔍 Verifying results...",
            "⚖️ Final count...",
        ]

        for message in countdown_messages:
            panel = Panel(
                Align.center(f"[yellow]{message}[/yellow]"),
                border_style="yellow",
                padding=(1, 2),
                width=min(self.console.get_width() - 20, 50),
            )
            self.console.print(panel)
            time.sleep(0.8)

        # Final dramatic pause
        panel = Panel(
            Align.center("[bold red]📢 THE RESULTS ARE IN! 📢[/bold red]"),
            border_style="red",
            padding=(1, 2),
            width=min(self.console.get_width() - 20, 60),
        )
        self.console.print(panel)
        time.sleep(1.0)

    def _format_detailed_votes(self, votes: List[Dict[str, Any]]) -> List[str]:
        """Format detailed vote breakdown.

        Args:
            votes: List of vote objects

        Returns:
            List of formatted strings for detailed display
        """
        lines = ["[bold]Detailed Results:[/bold]", ""]

        for vote in votes:
            voter = vote.get("voter_id", vote.get("agent_id", "Unknown"))
            choice = vote.get("vote", vote.get("choice", "Unknown"))

            # Style the vote choice
            if choice.lower() in ["yes", "continue"]:
                choice_styled = f"[green]{choice}[/green]"
            elif choice.lower() in ["no", "end", "conclude"]:
                choice_styled = f"[red]{choice}[/red]"
            else:
                choice_styled = f"[yellow]{choice}[/yellow]"

            lines.append(f"  • {voter}: {choice_styled}")

        return lines


# Global voting display instance
_voting_display: Optional[VotingDisplay] = None


def get_voting_display() -> VotingDisplay:
    """Get the global voting display instance."""
    global _voting_display
    if _voting_display is None:
        _voting_display = VotingDisplay()
    return _voting_display


# Convenience functions for easy integration
def announce_vote(
    vote_type: VoteType, topic: str, participants: List[str], context: str = ""
) -> None:
    """Announce the start of a vote."""
    get_voting_display().show_voting_announcement(
        vote_type, topic, participants, context
    )


def show_vote_progress(
    vote_type: VoteType,
    topic: str,
    votes_collected: List[Dict[str, Any]],
    total_voters: int,
    show_intermediate: bool = True,
) -> None:
    """Show voting progress."""
    get_voting_display().show_vote_collection_progress(
        vote_type, topic, votes_collected, total_voters, show_intermediate
    )


def reveal_vote_results(
    vote_type: VoteType,
    topic: str,
    final_votes: List[Dict[str, Any]],
    result: str,
    details: Dict[str, Any] = None,
) -> None:
    """Reveal vote results dramatically."""
    get_voting_display().show_dramatic_vote_reveal(
        vote_type, topic, final_votes, result, details
    )
