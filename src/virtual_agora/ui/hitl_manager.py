"""Enhanced HITL Manager for Virtual Agora v1.3.

Manages all Human-in-the-Loop interactions with specialized handlers for each
interaction type in the v1.3 architecture.
"""

from typing import Dict, Any, Optional, Callable, List
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box
from datetime import datetime

from virtual_agora.ui.hitl_state import (
    HITLApprovalType,
    HITLInteraction,
    HITLContext,
    HITLResponse,
    HITLStateTracker,
)
from virtual_agora.ui.components import (
    create_status_panel,
    create_options_menu,
    create_info_table,
    VirtualAgoraTheme,
)
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancedHITLManager:
    """Manages all HITL interactions for v1.3."""

    def __init__(self, console: Console):
        self.console = console
        self.state_tracker = HITLStateTracker()
        self.handlers = self._register_handlers()
        self.auto_approve_conditions = {}

    def _register_handlers(self) -> Dict[HITLApprovalType, Callable]:
        """Register specific handlers for each HITL type."""
        return {
            HITLApprovalType.THEME_INPUT: self._handle_theme_input,
            HITLApprovalType.AGENDA_APPROVAL: self._handle_agenda_approval,
            HITLApprovalType.PERIODIC_STOP: self._handle_periodic_stop,
            HITLApprovalType.TOPIC_OVERRIDE: self._handle_topic_override,
            HITLApprovalType.TOPIC_CONTINUATION: self._handle_topic_continuation,
            HITLApprovalType.AGENT_POLL_OVERRIDE: self._handle_agent_poll_override,
            HITLApprovalType.SESSION_CONTINUATION: self._handle_session_continuation,
            HITLApprovalType.FINAL_REPORT_APPROVAL: self._handle_final_report_approval,
        }

    def process_interaction(self, interaction: HITLInteraction) -> HITLResponse:
        """Process a HITL interaction and return response."""

        # Start tracking the interaction
        self.state_tracker.start_interaction(interaction)

        # Display context if available
        if interaction.context:
            self._display_context(interaction.context)

        # Get appropriate handler
        handler = self.handlers.get(interaction.approval_type, self._default_handler)

        # Process interaction
        try:
            response = handler(interaction)
        except KeyboardInterrupt:
            # Handle user interrupt
            response = self._handle_interrupt(interaction)
        except Exception as e:
            logger.error(f"Error in HITL handler: {e}")
            response = HITLResponse(approved=False, action="error", reason=str(e))

        # Complete tracking
        self.state_tracker.complete_interaction(response)

        return response

    def _display_context(self, context: Dict[str, Any]) -> None:
        """Display relevant context before prompting user."""
        if not context:
            return

        # Extract common context fields
        if "session_stats" in context:
            stats_table = create_info_table(
                context["session_stats"], title="Session Statistics"
            )
            self.console.print(stats_table)
            self.console.print()

        if "active_topic" in context:
            self.console.print(f"[bold]Current Topic:[/bold] {context['active_topic']}")

        if "current_round" in context:
            self.console.print(
                f"[bold]Current Round:[/bold] {context['current_round']}"
            )

        self.console.print()

    def _handle_periodic_stop(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle 5-round periodic stop check.

        New in v1.3 - gives user periodic control.
        """
        round_num = interaction.context.get("current_round", 0)
        topic = interaction.context.get("active_topic", "Unknown")

        # Record the periodic stop
        self.state_tracker.record_periodic_stop(round_num)

        # Create visual checkpoint panel
        checkpoint_panel = Panel(
            f"[bold yellow]Round {round_num} Checkpoint[/bold yellow]\n\n"
            f"Current topic: [cyan]{topic}[/cyan]\n"
            f"You've reached a 5-round checkpoint.\n\n"
            f"{interaction.prompt_message}",
            title="🛑 Periodic Stop",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(checkpoint_panel)

        # Show options
        options = {
            "c": "[green]Continue[/green] discussion",
            "e": "[orange]End[/orange] current topic",
            "m": "[yellow]Modify[/yellow] discussion parameters",
            "s": "[blue]Skip[/blue] to final report",
        }

        options_menu = create_options_menu(options, "Checkpoint Options")
        self.console.print(options_menu)

        # Get user decision
        action = Prompt.ask(
            "Select action", choices=list(options.keys()), default="c"
        ).lower()

        if action == "c":
            return HITLResponse(
                approved=True,
                action="continue",
                metadata={
                    "checkpoint_round": round_num,
                    "checkpoint_type": "periodic",
                },
            )
        elif action == "e":
            reason = Prompt.ask(
                "Reason for ending (optional)",
                default="User decision at periodic checkpoint",
            )
            return HITLResponse(
                approved=True,
                action="force_topic_end",
                reason=reason,
                metadata={
                    "checkpoint_round": round_num,
                    "forced_by": "user",
                },
            )
        elif action == "m":
            # TODO: Implement parameter modification
            self.console.print(
                "[yellow]Parameter modification not yet implemented.[/yellow]"
            )
            return HITLResponse(
                approved=True,
                action="continue",
                metadata={"modification_requested": True},
            )
        else:  # skip
            return HITLResponse(
                approved=True,
                action="skip_to_report",
                reason="User requested skip at checkpoint",
                metadata={"checkpoint_round": round_num},
            )

    def _handle_agenda_approval(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle agenda approval with editing capability.

        Enhanced in v1.3 to support inline editing.
        """
        agenda = interaction.context.get("proposed_agenda", [])

        # Display proposed agenda
        table = Table(
            title="Proposed Discussion Agenda",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
        )
        table.add_column("Order", style="cyan", width=6)
        table.add_column("Topic", style="white")
        table.add_column("Priority", justify="center")

        for i, topic in enumerate(agenda, 1):
            priority = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            priority_color = "red" if i <= 2 else "yellow" if i <= 4 else "green"
            table.add_row(
                str(i), topic, f"[{priority_color}]{priority}[/{priority_color}]"
            )

        self.console.print(table)
        self.console.print()

        # Ask for approval
        choices = ["Approve", "Edit", "Reorder", "Reject"]
        action = Prompt.ask(
            "What would you like to do?", choices=choices, default="Approve"
        )

        if action == "Approve":
            return HITLResponse(approved=True, action="approved", modified_data=agenda)

        elif action == "Edit":
            edited_agenda = self._edit_agenda(agenda)
            return HITLResponse(
                approved=True,
                action="edited",
                modified_data=edited_agenda,
                metadata={"original_agenda": agenda},
            )

        elif action == "Reorder":
            reordered_agenda = self._reorder_agenda(agenda)
            return HITLResponse(
                approved=True,
                action="reordered",
                modified_data=reordered_agenda,
                metadata={"original_agenda": agenda},
            )

        else:  # Reject
            reason = Prompt.ask(
                "Reason for rejection (optional)",
                default="User requested new proposals",
            )
            return HITLResponse(approved=False, action="rejected", reason=reason)

    def _edit_agenda(self, agenda: List[str]) -> List[str]:
        """Allow user to edit agenda items."""
        edited_agenda = []

        self.console.print("\n[yellow]Edit each topic (Enter to keep):[/yellow]")

        for i, topic in enumerate(agenda, 1):
            new_topic = Prompt.ask(f"{i}. {topic}", default=topic)
            if new_topic.strip():  # Skip empty entries
                edited_agenda.append(new_topic)

        # Option to add new topics
        while True:
            new_topic = Prompt.ask("Add new topic (Enter to finish)", default="")
            if not new_topic:
                break
            edited_agenda.append(new_topic)

        # Show preview
        self.console.print("\n[bold]Edited Agenda:[/bold]")
        for i, topic in enumerate(edited_agenda, 1):
            self.console.print(f"{i}. {topic}")

        if Confirm.ask("\nSave these changes?", default=True):
            return edited_agenda
        else:
            return agenda  # Return original

    def _reorder_agenda(self, agenda: List[str]) -> List[str]:
        """Allow user to reorder agenda items."""
        reordered = agenda.copy()

        self.console.print(
            "\n[yellow]Drag items to reorder (use item numbers):[/yellow]"
        )

        while True:
            # Display current order
            self.console.print("\n[bold]Current Order:[/bold]")
            for i, topic in enumerate(reordered, 1):
                self.console.print(f"{i}. {topic}")

            # Ask for move
            self.console.print("\n[dim]Enter 'done' when finished[/dim]")
            move_from = Prompt.ask("Move item from position", default="done")

            if move_from.lower() == "done":
                break

            try:
                from_idx = int(move_from) - 1
                if 0 <= from_idx < len(reordered):
                    to_pos = Prompt.ask(
                        "Move to position",
                        choices=[str(i) for i in range(1, len(reordered) + 1)],
                    )
                    to_idx = int(to_pos) - 1

                    # Perform move
                    item = reordered.pop(from_idx)
                    reordered.insert(to_idx, item)
                    self.console.print("[green]✓ Item moved[/green]")
                else:
                    self.console.print("[red]Invalid position[/red]")
            except ValueError:
                self.console.print("[red]Please enter a number[/red]")

        return reordered

    def _handle_topic_continuation(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle topic continuation approval."""
        completed_topic = interaction.context.get("completed_topics", ["Unknown"])[-1]
        remaining_topics = interaction.context.get("remaining_topics", [])

        # Show completion summary
        completion_panel = Panel(
            f"[green]✓[/green] Topic '[bold]{completed_topic}[/bold]' has been concluded.\n\n"
            f"A comprehensive summary has been saved.",
            title="[bold green]Topic Completed[/bold green]",
            border_style="green",
        )
        self.console.print(completion_panel)

        if remaining_topics:
            # Show remaining topics
            remaining_table = Table(
                title="Remaining Topics", show_header=True, header_style="bold blue"
            )
            remaining_table.add_column("#", width=3)
            remaining_table.add_column("Topic")
            remaining_table.add_column("Est. Time", justify="right")

            for i, topic in enumerate(remaining_topics, 1):
                est_time = "15-20 min"  # This would be calculated
                remaining_table.add_row(str(i), topic, est_time)

            self.console.print(remaining_table)
            self.console.print()

            # Show options
            options = {
                "c": "[green]Continue[/green] to the next topic",
                "e": "[red]End[/red] the session",
                "m": "[yellow]Modify[/yellow] the remaining agenda first",
            }

            options_menu = create_options_menu(options)
            self.console.print(options_menu)

            action = Prompt.ask(
                "Select action", choices=list(options.keys()), default="c"
            ).lower()

            if action == "c":
                return HITLResponse(approved=True, action="continue")
            elif action == "e":
                return HITLResponse(approved=False, action="end_session")
            else:  # modify
                return HITLResponse(
                    approved=True,
                    action="modify_agenda",
                    metadata={"modification_requested": True},
                )
        else:
            # No remaining topics
            self.console.print("[bold]No remaining topics in the agenda.[/bold]")
            if Confirm.ask("End session?", default=True):
                return HITLResponse(approved=False, action="end_session")
            else:
                return HITLResponse(approved=True, action="continue")

    def _handle_session_continuation(
        self, interaction: HITLInteraction
    ) -> HITLResponse:
        """Handle session continuation after agent vote."""
        agent_vote = interaction.context.get("agent_vote_result", {})

        # Display agent recommendation
        if agent_vote.get("recommendation") == "end":
            status = create_status_panel(
                "Agents recommend ending the session", style="warning"
            )
        else:
            status = create_status_panel(
                "Agents recommend continuing the session", style="info"
            )

        self.console.print(status)

        # Get user decision
        if Confirm.ask("Do you want to continue the session?", default=True):
            return HITLResponse(approved=True, action="continue_session")
        else:
            return HITLResponse(approved=False, action="end_session")

    def _handle_topic_override(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle user override of topic conclusion."""
        current_topic = interaction.context.get("active_topic", "Unknown")

        self.console.print(
            f"\n[bold yellow]Override Topic Conclusion[/bold yellow]\n"
            f"Force conclusion of topic: [cyan]{current_topic}[/cyan]"
        )

        if Confirm.ask("Force topic conclusion?", default=False):
            reason = Prompt.ask("Reason (optional)", default="User override")
            return HITLResponse(approved=True, action="force_conclusion", reason=reason)
        else:
            return HITLResponse(approved=False, action="continue_discussion")

    def _handle_agent_poll_override(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle user override of agent poll results."""
        poll_result = interaction.context.get("agent_vote_result", {})

        # Display poll results
        self.console.print(
            f"\n[bold]Agent Poll Results:[/bold]\n"
            f"Yes votes: {poll_result.get('yes_votes', 0)}\n"
            f"No votes: {poll_result.get('no_votes', 0)}\n"
            f"Result: {poll_result.get('result', 'Unknown')}"
        )

        self.console.print("\n[yellow]You can override the agent decision.[/yellow]")

        if Confirm.ask("Override agent poll?", default=False):
            new_result = Prompt.ask(
                "New result", choices=["conclude", "continue"], default="conclude"
            )
            return HITLResponse(
                approved=True,
                action="override",
                modified_data={"new_result": new_result},
                reason="User override of agent poll",
            )
        else:
            return HITLResponse(approved=False, action="accept_poll")

    def _handle_final_report_approval(
        self, interaction: HITLInteraction
    ) -> HITLResponse:
        """Handle final report generation approval."""
        completed_topics = interaction.context.get("completed_topics", [])

        # Show session summary
        summary_panel = Panel(
            f"[bold]Session Complete[/bold]\n\n"
            f"Topics discussed: {len(completed_topics)}\n"
            f"Total duration: {interaction.context.get('session_duration', 'Unknown')}\n\n"
            f"Ready to generate comprehensive final report.",
            title="Final Report Generation",
            border_style="blue",
        )
        self.console.print(summary_panel)

        if Confirm.ask("Generate final report?", default=True):
            return HITLResponse(approved=True, action="generate_report")
        else:
            return HITLResponse(
                approved=False,
                action="skip_report",
                reason="User declined final report",
            )

    def _handle_theme_input(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle initial theme input."""
        # Get theme from user
        theme = Prompt.ask(interaction.prompt_message or "Enter discussion theme")

        return HITLResponse(approved=True, action="theme_provided", modified_data=theme)

    def _default_handler(self, interaction: HITLInteraction) -> HITLResponse:
        """Default handler for unknown interaction types."""
        self.console.print(
            f"[yellow]Unknown interaction type: {interaction.approval_type}[/yellow]"
        )

        if interaction.prompt_message:
            self.console.print(interaction.prompt_message)

        if Confirm.ask("Approve?", default=True):
            return HITLResponse(approved=True, action="approved")
        else:
            return HITLResponse(approved=False, action="rejected")

    def _handle_interrupt(self, interaction: HITLInteraction) -> HITLResponse:
        """Handle keyboard interrupt during interaction."""
        self.console.print("\n[yellow]Interaction interrupted![/yellow]")

        options = {
            "r": "[green]Resume[/green] interaction",
            "s": "[blue]Skip[/blue] this interaction",
            "p": "[yellow]Pause[/yellow] session",
            "q": "[red]Quit[/red] session",
        }

        options_menu = create_options_menu(options, "Interrupt Options")
        self.console.print(options_menu)

        action = Prompt.ask(
            "Select action", choices=list(options.keys()), default="r"
        ).lower()

        if action == "r":
            # Re-run the handler
            handler = self.handlers.get(
                interaction.approval_type, self._default_handler
            )
            return handler(interaction)
        elif action == "s":
            return HITLResponse(
                approved=False, action="skipped", reason="User interrupt"
            )
        elif action == "p":
            return HITLResponse(
                approved=False, action="pause_session", reason="User requested pause"
            )
        else:  # quit
            return HITLResponse(
                approved=False, action="quit_session", reason="User requested quit"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get HITL interaction statistics."""
        return self.state_tracker.get_interaction_stats()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get interaction history."""
        return self.state_tracker.approval_history
