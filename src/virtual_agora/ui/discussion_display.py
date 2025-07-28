"""Advanced discussion display components for Virtual Agora.

This module provides structured conversation layout with round markers,
agent identification, and formatted message display for the terminal UI.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.layout import Layout
from rich.console import Group
from rich.padding import Padding
from rich.align import Align
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme, MessageType
from virtual_agora.providers.config import ProviderType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class MessageRole(Enum):
    """Message role classification."""

    AGENT = "agent"
    MODERATOR = "moderator"
    USER = "user"
    SYSTEM = "system"


@dataclass
class DiscussionMessage:
    """Structured discussion message."""

    content: str
    agent_id: str
    provider: ProviderType
    role: MessageRole
    timestamp: datetime
    round_number: Optional[int] = None
    topic: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DiscussionRound:
    """Represents a complete discussion round."""

    round_number: int
    topic: str
    messages: List[DiscussionMessage]
    summary: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class DiscussionFormatter:
    """Formats discussion content for display."""

    def __init__(self):
        """Initialize discussion formatter."""
        self.console = get_console()
        self.theme = get_current_theme()

    def format_message(
        self,
        message: DiscussionMessage,
        show_timestamp: bool = True,
        show_round: bool = True,
        compact: bool = False,
    ) -> Panel:
        """Format a single discussion message."""

        # Get agent colors
        colors = self.theme.assign_agent_color(message.agent_id, message.provider)

        # Build header components
        header_parts = []

        # Agent identifier with symbol
        agent_part = f"[{colors['primary']}]{colors['symbol']} {message.agent_id}[/{colors['primary']}]"
        header_parts.append(agent_part)

        # Round number
        if show_round and message.round_number:
            round_part = (
                f"[{colors['accent']}]Round {message.round_number}[/{colors['accent']}]"
            )
            header_parts.append(round_part)

        # Timestamp
        if show_timestamp:
            timestamp_str = message.timestamp.strftime("%H:%M:%S")
            timestamp_part = f"[dim]{timestamp_str}[/dim]"
            header_parts.append(timestamp_part)

        # Topic context
        if message.topic:
            topic_part = f"[italic]{message.topic}[/italic]"
            header_parts.append(topic_part)

        header = " • ".join(header_parts)

        # Format content based on role
        if message.role == MessageRole.MODERATOR:
            content_style = "bold white"
            border_style = "magenta"
        else:
            content_style = "white"
            border_style = colors["border"]

        # Content processing
        content = message.content
        if compact and len(content) > 150:
            content = content[:147] + "..."

        # Create panel
        panel_kwargs = {
            "title": header,
            "border_style": border_style,
            "padding": (0, 1) if compact else (1, 2),
        }

        if not compact:
            panel_kwargs["width"] = min(self.console.get_width() - 4, 100)

        panel = Panel(f"[{content_style}]{content}[/{content_style}]", **panel_kwargs)

        return panel

    def format_round_header(
        self, round_number: int, topic: str, agent_count: int
    ) -> Rule:
        """Format a round header separator."""
        header_text = f"Round {round_number} • {topic} • {agent_count} agents"

        return Rule(
            f"[bold cyan]{header_text}[/bold cyan]", style="cyan", characters="═"
        )

    def format_agent_response(
        self,
        agent_id: str,
        provider: ProviderType,
        content: str,
        round_number: int,
        topic: str,
        timestamp: Optional[datetime] = None,
    ) -> Panel:
        """Format an agent response for assembly-style display with enhanced natural presentation.

        This creates a panel similar to the 'Your Topic' panel but for each agent response,
        giving users the feeling of watching an assembly or deliberation in real-time.
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Get agent colors from theme
        colors = self.theme.assign_agent_color(agent_id, provider)

        # Build the title with agent identification and provider
        provider_name = (
            provider.value.title()
            if provider != ProviderType.MODERATOR
            else "Moderator"
        )
        title_parts = [
            f"[{colors['primary']}]{agent_id}[/{colors['primary']}]",
            f"[dim]({provider_name})[/dim]",
            f"[{colors['accent']}]Round {round_number}[/{colors['accent']}]",
        ]

        # Add timestamp in assembly mode for more formal feel
        try:
            from virtual_agora.ui.display_modes import is_assembly_mode

            if is_assembly_mode():
                time_str = timestamp.strftime("%H:%M:%S")
                title_parts.append(f"[dim]{time_str}[/dim]")
        except ImportError:
            pass

        title = " • ".join(title_parts)

        # Clean and format the content with natural presentation
        formatted_content = self._format_content_naturally(content, agent_id)

        # Create the panel with assembly-style formatting
        panel = Panel(
            f"[white]{formatted_content}[/white]",
            title=title,
            title_align="left",
            border_style=colors["border"],
            padding=(1, 2),
            width=min(
                self.console.get_width() - 4, 80
            ),  # Consistent width with topic panel
        )

        return panel

    def _format_content_naturally(self, content: str, agent_id: str) -> str:
        """Format content to feel more natural and conversational.

        Args:
            content: Raw content from agent
            agent_id: Agent identifier for personalization

        Returns:
            Naturally formatted content
        """
        formatted_content = content.strip()

        # Remove any agent self-references that make it robotic
        formatted_content = formatted_content.replace(f"{agent_id}:", "")
        formatted_content = formatted_content.replace(f"[{agent_id}]:", "")

        # Clean up any leading/trailing whitespace again
        formatted_content = formatted_content.strip()

        # Add subtle personalization for different types of responses
        if any(
            word in formatted_content.lower() for word in ["however", "but", "although"]
        ):
            # This appears to be a counter-argument, which is good for deliberation
            pass
        elif any(
            word in formatted_content.lower()
            for word in ["i agree", "building on", "following"]
        ):
            # This appears to be a building/agreement response
            pass
        elif formatted_content.startswith(("Let me", "I'd like to", "Allow me")):
            # Already has natural opening, keep as is
            pass

        return formatted_content

    def format_round_summary(self, round_data: DiscussionRound) -> Panel:
        """Format a round summary."""
        content_lines = []

        # Basic info
        content_lines.append(f"[bold]Topic:[/bold] {round_data.topic}")
        content_lines.append(f"[bold]Messages:[/bold] {len(round_data.messages)}")

        if round_data.start_time and round_data.end_time:
            duration = round_data.end_time - round_data.start_time
            content_lines.append(
                f"[bold]Duration:[/bold] {duration.total_seconds():.1f}s"
            )

        # Agent participation
        agent_counts = {}
        for msg in round_data.messages:
            agent_counts[msg.agent_id] = agent_counts.get(msg.agent_id, 0) + 1

        content_lines.append("")
        content_lines.append("[bold]Agent Participation:[/bold]")
        for agent_id, count in agent_counts.items():
            content_lines.append(f"  • {agent_id}: {count} messages")

        # Summary if available
        if round_data.summary:
            content_lines.append("")
            content_lines.append("[bold]Summary:[/bold]")
            content_lines.append(round_data.summary)

        content = "\n".join(content_lines)

        return Panel(
            content,
            title=f"[bold]Round {round_data.round_number} Summary[/bold]",
            border_style="green",
            padding=(1, 2),
        )

    def format_voting_display(
        self, votes: Dict[str, Dict[str, Any]], topic: str
    ) -> Panel:
        """Format voting results display."""
        table = Table(
            title=f"[bold]Voting Results: {topic}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Agent", style="cyan", width=20)
        table.add_column("Vote", justify="center", width=8)
        table.add_column("Justification", style="dim")

        # Sort by vote (Yes first, then No)
        sorted_votes = sorted(
            votes.items(), key=lambda x: (x[1].get("vote", "").lower() != "yes", x[0])
        )

        yes_count = 0
        no_count = 0

        for agent_id, vote_data in sorted_votes:
            vote = vote_data.get("vote", "No response")
            justification = vote_data.get("justification", "No justification provided")

            # Style vote
            if vote.lower() == "yes":
                vote_styled = "[green]✓ Yes[/green]"
                yes_count += 1
            elif vote.lower() == "no":
                vote_styled = "[red]✗ No[/red]"
                no_count += 1
            else:
                vote_styled = f"[yellow]{vote}[/yellow]"

            # Truncate long justifications
            if len(justification) > 60:
                justification = justification[:57] + "..."

            table.add_row(agent_id, vote_styled, justification)

        # Add voting summary
        total_votes = yes_count + no_count
        if total_votes > 0:
            majority_needed = (total_votes // 2) + 1
            result = "PASSES" if yes_count >= majority_needed else "FAILS"
            result_color = "green" if result == "PASSES" else "red"

            summary = f"[{result_color}]{result}[/{result_color}] • Yes: {yes_count} • No: {no_count} • Majority needed: {majority_needed}"

            return Panel(
                Group(table, Padding(f"\n[bold]{summary}[/bold]", (0, 2))),
                border_style="cyan",
                padding=(1, 1),
            )
        else:
            return Panel(table, border_style="cyan", padding=(1, 1))

    def format_agenda_display(
        self,
        agenda: List[str],
        current_topic: Optional[str] = None,
        completed_topics: Optional[List[str]] = None,
    ) -> Panel:
        """Format agenda display with status indicators."""

        completed_topics = completed_topics or []
        table = Table(
            title="[bold cyan]Discussion Agenda[/bold cyan]",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("#", width=3, justify="center")
        table.add_column("Topic", style="white")
        table.add_column("Status", justify="center", width=12)

        for i, topic in enumerate(agenda, 1):
            status_icon = ""
            topic_style = "white"

            if topic in completed_topics:
                status_icon = "[green]✓ Complete[/green]"
                topic_style = "dim"
            elif topic == current_topic:
                status_icon = "[yellow]🎯 Current[/yellow]"
                topic_style = "bold yellow"
            else:
                status_icon = "[dim]⏳ Pending[/dim]"

            table.add_row(
                str(i), f"[{topic_style}]{topic}[/{topic_style}]", status_icon
            )

        return Panel(table, border_style="cyan", padding=(1, 1))


class DiscussionDisplay:
    """Main discussion display controller."""

    def __init__(self):
        """Initialize discussion display."""
        self.console = get_console()
        self.formatter = DiscussionFormatter()
        self._message_history: List[DiscussionMessage] = []
        self._rounds: List[DiscussionRound] = []
        self._current_round: Optional[int] = None

    def add_message(self, message: DiscussionMessage) -> None:
        """Add a message to the display history."""
        self._message_history.append(message)

        # Display the message immediately
        formatted_message = self.formatter.format_message(message)
        self.console.print(formatted_message)

    def start_new_round(self, round_number: int, topic: str, agent_count: int) -> None:
        """Start a new discussion round."""
        self._current_round = round_number

        # Display round header
        header = self.formatter.format_round_header(round_number, topic, agent_count)
        self.console.print()
        self.console.print(header)
        self.console.print()

        # Initialize round data
        round_data = DiscussionRound(
            round_number=round_number,
            topic=topic,
            messages=[],
            start_time=datetime.now(),
        )
        self._rounds.append(round_data)

    def complete_round(self, summary: Optional[str] = None) -> None:
        """Complete the current round."""
        if not self._rounds:
            return

        current_round = self._rounds[-1]
        current_round.end_time = datetime.now()
        current_round.summary = summary

        # Collect messages for this round
        round_messages = [
            msg
            for msg in self._message_history
            if msg.round_number == current_round.round_number
        ]
        current_round.messages = round_messages

        # Display round summary if available
        if summary:
            summary_panel = self.formatter.format_round_summary(current_round)
            self.console.print()
            self.console.print(summary_panel)
            self.console.print()

    def display_voting_results(
        self, votes: Dict[str, Dict[str, Any]], topic: str
    ) -> None:
        """Display voting results."""
        voting_panel = self.formatter.format_voting_display(votes, topic)
        self.console.print()
        self.console.print(voting_panel)
        self.console.print()

    def display_agenda(
        self,
        agenda: List[str],
        current_topic: Optional[str] = None,
        completed_topics: Optional[List[str]] = None,
    ) -> None:
        """Display the discussion agenda."""
        agenda_panel = self.formatter.format_agenda_display(
            agenda, current_topic, completed_topics
        )
        self.console.print(agenda_panel)
        self.console.print()

    def display_phase_transition(
        self, from_phase: str, to_phase: str, details: str = ""
    ) -> None:
        """Display a phase transition."""
        content = f"[cyan]Transitioning from[/cyan] [yellow]{from_phase}[/yellow] [cyan]to[/cyan] [green]{to_phase}[/green]"

        if details:
            content += f"\n[dim]{details}[/dim]"

        panel = Panel(
            content,
            title="[bold]Phase Transition[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def display_session_header(self, session_info: Dict[str, Any]) -> None:
        """Display session header information."""
        # Create two-column layout
        left_column = Table(box=None, show_header=False, padding=(0, 1))
        left_column.add_column("Field", style="bold cyan")
        left_column.add_column("Value", style="white")

        left_column.add_row("Session ID", session_info.get("session_id", "Unknown"))
        left_column.add_row("Topic", session_info.get("main_topic", "Not set"))
        left_column.add_row("Started", session_info.get("start_time", "Unknown"))

        right_column = Table(box=None, show_header=False, padding=(0, 1))
        right_column.add_column("Field", style="bold cyan")
        right_column.add_column("Value", style="white")

        agent_info = session_info.get("agents", {})
        right_column.add_row("Total Agents", str(agent_info.get("total", 0)))
        right_column.add_row("Moderator", agent_info.get("moderator", "Unknown"))
        right_column.add_row("Providers", str(len(agent_info.get("providers", []))))

        columns = Columns(
            [
                Panel(
                    left_column, title="[bold]Session Info[/bold]", border_style="cyan"
                ),
                Panel(
                    right_column, title="[bold]Agent Info[/bold]", border_style="green"
                ),
            ],
            equal=True,
            expand=True,
        )

        header_panel = Panel(
            columns,
            title="[bold magenta]🏛️ Virtual Agora Session[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )

        self.console.print(header_panel)
        self.console.print()

    def display_error_message(self, error: str, context: str = "") -> None:
        """Display an error message in the discussion."""
        content = f"[red]❌ Error:[/red] {error}"

        if context:
            content += f"\n[dim]Context: {context}[/dim]"

        panel = Panel(
            content,
            title="[bold red]Error Occurred[/bold red]",
            border_style="red",
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def get_message_history(self) -> List[DiscussionMessage]:
        """Get the complete message history."""
        return self._message_history.copy()

    def get_round_history(self) -> List[DiscussionRound]:
        """Get the complete round history."""
        return self._rounds.copy()

    def get_current_round(self) -> Optional[int]:
        """Get the current round number."""
        return self._current_round

    def clear_history(self) -> None:
        """Clear all message and round history."""
        self._message_history.clear()
        self._rounds.clear()
        self._current_round = None


# Global discussion display instance
_discussion_display: Optional[DiscussionDisplay] = None


def get_discussion_display() -> DiscussionDisplay:
    """Get the global discussion display instance."""
    global _discussion_display
    if _discussion_display is None:
        _discussion_display = DiscussionDisplay()
    return _discussion_display


# Convenience functions


def add_agent_message(
    agent_id: str,
    provider: ProviderType,
    content: str,
    round_number: Optional[int] = None,
    topic: Optional[str] = None,
) -> None:
    """Add an agent message to the discussion display."""
    message = DiscussionMessage(
        content=content,
        agent_id=agent_id,
        provider=provider,
        role=MessageRole.AGENT,
        timestamp=datetime.now(),
        round_number=round_number,
        topic=topic,
    )

    get_discussion_display().add_message(message)


def add_moderator_message(
    content: str, round_number: Optional[int] = None, topic: Optional[str] = None
) -> None:
    """Add a moderator message to the discussion display."""
    message = DiscussionMessage(
        content=content,
        agent_id="Moderator",
        provider=ProviderType.MODERATOR,
        role=MessageRole.MODERATOR,
        timestamp=datetime.now(),
        round_number=round_number,
        topic=topic,
    )

    get_discussion_display().add_message(message)


def start_discussion_round(round_number: int, topic: str, agent_count: int) -> None:
    """Start a new discussion round."""
    get_discussion_display().start_new_round(round_number, topic, agent_count)


def complete_discussion_round(summary: Optional[str] = None) -> None:
    """Complete the current discussion round."""
    get_discussion_display().complete_round(summary)


def show_voting_results(votes: Dict[str, Dict[str, Any]], topic: str) -> None:
    """Show voting results in the discussion display."""
    get_discussion_display().display_voting_results(votes, topic)


def show_agenda(
    agenda: List[str],
    current_topic: Optional[str] = None,
    completed_topics: Optional[List[str]] = None,
) -> None:
    """Show the discussion agenda."""
    get_discussion_display().display_agenda(agenda, current_topic, completed_topics)


def display_agent_response(
    agent_id: str,
    provider: ProviderType,
    content: str,
    round_number: int,
    topic: str,
    timestamp: Optional[datetime] = None,
) -> None:
    """Display an agent response in assembly-style panel format with natural timing.

    This creates a visual panel for each agent response, similar to the 'Your Topic' panel,
    giving users the experience of watching an assembly or deliberation with natural pacing.
    """
    import time

    console = get_console().rich_console
    formatter = DiscussionFormatter()

    # Add natural timing based on content length (simulate reading/processing time)
    try:
        from virtual_agora.ui.display_modes import is_assembly_mode

        if is_assembly_mode():
            # Calculate natural delay based on content length
            # Simulate time for the agent to "gather thoughts" before speaking
            word_count = len(content.split())
            base_delay = 0.5  # Base delay in seconds
            reading_delay = min(
                word_count * 0.02, 2.0
            )  # Max 2 seconds for very long responses
            natural_delay = base_delay + reading_delay

            # Show that agent is about to speak
            dashboard = None
            try:
                from virtual_agora.ui.assembly_dashboard import get_assembly_dashboard

                dashboard = get_assembly_dashboard()
                dashboard.update_participant_status(agent_id, "Speaking")
            except ImportError:
                pass

            # Natural pause before speaking
            time.sleep(natural_delay)
    except ImportError:
        pass

    # Format the agent response as an assembly-style panel
    panel = formatter.format_agent_response(
        agent_id=agent_id,
        provider=provider,
        content=content,
        round_number=round_number,
        topic=topic,
        timestamp=timestamp,
    )

    # Display the panel with some spacing
    console.print()
    console.print(panel)
    console.print()

    # Update assembly dashboard if in assembly mode
    try:
        from virtual_agora.ui.display_modes import is_assembly_mode
        from virtual_agora.ui.assembly_dashboard import get_assembly_dashboard

        if is_assembly_mode():
            dashboard = get_assembly_dashboard()
            # Record the message for statistics
            dashboard.record_participant_message(agent_id)
            # Update participant status back to ready
            dashboard.update_participant_status(agent_id, "Ready")

            # Small pause after speaking for natural flow
            time.sleep(0.3)
    except ImportError:
        # If modules not available, continue without assembly dashboard updates
        pass
