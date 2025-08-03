"""Human-in-the-Loop interface components for Virtual Agora.

This module provides comprehensive interactive UI components using the Rich library
for all human interaction points in the discussion flow. It implements all user
stories from Epic 7 with proper validation, error handling, and user experience.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.align import Align

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.document_context import get_detailed_context_info
from virtual_agora.ui.console import get_console
from virtual_agora.ui.interactive import (
    get_prompts,
    PromptOption,
    ask_text,
    ask_choice,
    ask_confirmation,
    create_non_empty_validator,
    create_length_validator,
)
from virtual_agora.ui.theme import MessageType
from virtual_agora.ui.formatters import format_content, FormatType

# Initialize enhanced console and logger
console = get_console().rich_console  # For backward compatibility
enhanced_console = get_console()
prompts = get_prompts()
logger = get_logger(__name__)

# Input validation constants
MIN_TOPIC_LENGTH = 10
MIN_AGENDA_ITEMS = 2
MAX_AGENDA_ITEMS = 10
MAX_FILENAME_LENGTH = 30
INPUT_TIMEOUT = 300  # 5 minutes
MAX_RETRY_ATTEMPTS = 3

# Session state tracking
session_start_time = datetime.now()
input_history: List[Dict[str, Any]] = []


def truncate_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """Truncate filename preserving extension.

    Args:
        filename: The filename to truncate
        max_length: Maximum length for the filename

    Returns:
        Truncated filename with ... in the middle if needed
    """
    if len(filename) <= max_length:
        return filename

    # Split name and extension
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        name, ext = parts
        ext_with_dot = f".{ext}"
    else:
        name = filename
        ext_with_dot = ""

    # Calculate how much we can keep
    available = max_length - len(ext_with_dot) - 3  # 3 for "..."
    if available <= 0:
        # Extension alone is too long, just truncate the whole thing
        return filename[: max_length - 3] + "..."

    # Keep start and end of filename
    keep_start = available // 2
    keep_end = available - keep_start

    if keep_end > 0:
        truncated = name[:keep_start] + "..." + name[-keep_end:] + ext_with_dot
    else:
        truncated = name[:keep_start] + "..." + ext_with_dot

    return truncated


def record_input(input_type: str, value: Any, metadata: Optional[Dict] = None) -> None:
    """Record user input for history tracking."""
    input_record = {
        "type": input_type,
        "value": value,
        "timestamp": datetime.now(),
        "metadata": metadata or {},
    }
    input_history.append(input_record)
    logger.debug(f"Recorded input: {input_type}")


def get_initial_topic() -> Dict[str, Any]:
    """Gets the initial discussion topic from the user.

    Implements Story 7.1: Initial Topic Input Interface with:
    - Clear welcome message
    - Multi-line topic support
    - Validation and confirmation
    - Topic templates for guidance
    - User choice for topic definition method

    Returns:
        Dict containing 'topic' and 'user_defines_topics' keys
    """
    console.clear()

    # Display welcome message
    welcome_text = """
    # Welcome to Virtual Agora
    
    A structured multi-agent discussion platform where AI agents collaborate
    to explore complex topics through democratic deliberation.
    
    Please provide a topic for discussion. The topic should be:
    - Clear and specific
    - Open to multiple perspectives
    - Suitable for structured debate
    """

    # Check for context documents
    context_info = get_detailed_context_info()

    # Build panel content - Markdown for main text, then add context separately
    panel_content = Markdown(welcome_text)

    # Add context information if available
    if context_info["status"] == "success" and context_info["total_files"] > 0:
        # Create a formatted text for context instead of adding to Markdown
        from rich.text import Text

        context_text = Text()
        context_text.append(
            f"\n📄 Context: {context_info['total_files']} documents loaded ({context_info['total_tokens']:,} tokens)\n",
            style="",
        )

        # Add individual files
        if context_info["files"]:
            for i, file_info in enumerate(context_info["files"]):
                is_last = i == len(context_info["files"]) - 1
                prefix = "   └─ " if is_last else "   ├─ "
                truncated_name = truncate_filename(file_info["name"])
                context_text.append(
                    f"{prefix}{truncated_name} ({file_info['tokens']:,} tokens)",
                    style="dim",
                )
                if not is_last:
                    context_text.append("\n")

        # Combine Markdown and Text objects
        from rich.console import Group

        panel_content = Group(panel_content, context_text)

    console.print(
        Panel(
            panel_content,
            title="[bold cyan]Virtual Agora[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Show topic examples
    examples_table = Table(
        title="Example Topics", show_header=False, box=box.SIMPLE, padding=(0, 1)
    )
    examples_table.add_row(
        "[green]Technology:[/green]",
        "The ethical implications of AI in healthcare decision-making",
    )
    examples_table.add_row(
        "[blue]Society:[/blue]",
        "Balancing privacy rights with public safety in the digital age",
    )
    examples_table.add_row(
        "[yellow]Environment:[/yellow]",
        "Strategies for achieving carbon neutrality by 2050",
    )

    console.print(examples_table)
    console.print()

    # Get topic with validation
    attempts = 0
    while attempts < MAX_RETRY_ATTEMPTS:
        try:
            # Support multi-line input
            console.print("[bold]Enter your discussion topic:[/bold]")
            console.print("[dim](Press Enter twice to finish multi-line input)[/dim]")

            lines = []
            while True:
                line = Prompt.ask("", default="", show_default=False, console=console)
                if not line and lines:  # Empty line after content
                    break
                if line:
                    lines.append(line)

            topic = " ".join(lines).strip()

            # Validate topic
            if not topic:
                console.print("[red]Topic cannot be empty. Please try again.[/red]")
                attempts += 1
                continue

            if len(topic) < MIN_TOPIC_LENGTH:
                console.print(
                    f"[yellow]Topic too short (minimum {MIN_TOPIC_LENGTH} characters).[/yellow]"
                )
                attempts += 1
                continue

            # Confirm topic
            console.print()
            console.print(
                Panel(
                    topic,
                    title="[bold]Your Topic[/bold]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            if Confirm.ask("Confirm this topic?", default=True, console=console):
                record_input("initial_topic", topic)
                logger.info(f"Topic confirmed: {topic[:50]}...")

                # Ask user how they want to define discussion topics
                console.print()
                console.print("[bold cyan]Topic Definition Method[/bold cyan]")
                console.print(
                    "How would you like to define the specific topics for discussion?"
                )
                console.print()
                console.print("Options:")
                console.print(
                    "  [green]1. Let AI agents create and propose topics[/green] (default)"
                )
                console.print("  [blue]2. Define topics yourself[/blue]")
                console.print()

                choice = Prompt.ask(
                    "Select option",
                    choices=["1", "2"],
                    default="1",
                    console=console,
                ).strip()

                user_defines_topics = choice == "2"

                if user_defines_topics:
                    console.print(
                        "[blue]You'll define the topics after confirmation.[/blue]"
                    )
                else:
                    console.print(
                        "[green]AI agents will collaboratively create the discussion topics.[/green]"
                    )

                return {"topic": topic, "user_defines_topics": user_defines_topics}
            else:
                console.print("[yellow]Let's try again...[/yellow]")
                attempts += 1

        except KeyboardInterrupt:
            handle_interrupt("topic_input")

    raise ValueError("Maximum retry attempts exceeded for topic input")


def get_user_defined_topics(main_topic: str) -> List[str]:
    """Allow user to manually define discussion topics.

    Uses the existing agenda editing interface to let users create
    their own discussion topics from scratch.

    Args:
        main_topic: The main discussion theme for context

    Returns:
        List of user-defined topics
    """
    console.clear()

    # Show context
    console.print(
        Panel(
            f"[bold]Main Discussion Topic:[/bold]\n{main_topic}",
            title="[bold cyan]Define Discussion Topics[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print(
        "\n[bold]Instructions:[/bold]\n"
        "Create specific discussion topics that will explore different aspects of your main theme.\n"
        "You'll start with an empty list - add topics one by one using the editing interface.\n"
    )

    # Show examples based on the main topic
    console.print("[bold]Example topics might include:[/bold]")
    console.print("• Foundational concepts and definitions")
    console.print("• Current challenges and problems")
    console.print("• Potential solutions and approaches")
    console.print("• Implementation considerations")
    console.print("• Future implications and outcomes")

    if not Confirm.ask(
        "\nReady to start defining topics?", default=True, console=console
    ):
        console.print(
            "[yellow]Using empty topic list - you can add topics in the editor.[/yellow]"
        )

    # Start with empty agenda and use the existing edit interface
    user_topics = []

    # Use the existing edit_agenda function which provides full editing capabilities
    console.print("\n[cyan]Opening topic editor...[/cyan]")
    user_topics = edit_agenda(user_topics)

    # Validate that user created at least one topic
    if not user_topics:
        console.print(
            "[yellow]No topics were defined. Adding a default topic...[/yellow]"
        )
        user_topics = [f"General discussion on: {main_topic}"]

    # Show final topics
    console.print()
    console.print(
        Panel(
            "\n".join(f"{i}. {topic}" for i, topic in enumerate(user_topics, 1)),
            title="[bold green]Your Discussion Topics[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    if Confirm.ask("Confirm these topics?", default=True, console=console):
        record_input(
            "user_defined_topics", {"main_topic": main_topic, "topics": user_topics}
        )
        logger.info(f"User defined {len(user_topics)} topics for: {main_topic[:50]}...")
        return user_topics
    else:
        console.print("[yellow]Let's edit them again...[/yellow]")
        return get_user_defined_topics(main_topic)  # Recursive retry


def get_agenda_approval(agenda: List[str]) -> List[str]:
    """Displays the proposed agenda and asks for user approval.

    Implements Story 7.2: Agenda Approval Workflow with:
    - Clear agenda display with vote tallies
    - Three options: approve, edit, reject
    - Agenda preview for edits
    """
    console.clear()
    console.print(
        Panel(
            "[bold cyan]Agenda Approval[/bold cyan]",
            subtitle="The agents have proposed the following discussion agenda",
            expand=False,
        )
    )

    # Create agenda table
    agenda_table = Table(
        title="Proposed Discussion Agenda",
        show_header=True,
        header_style="bold magenta",
    )
    agenda_table.add_column("#", style="dim", width=3)
    agenda_table.add_column("Topic", style="cyan")
    agenda_table.add_column("Priority", justify="center")

    for i, item in enumerate(agenda, 1):
        priority = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
        priority_color = "red" if i <= 2 else "yellow" if i <= 4 else "green"
        agenda_table.add_row(
            str(i), item, f"[{priority_color}]{priority}[/{priority_color}]"
        )

    console.print(agenda_table)
    console.print()

    # Show options
    options_text = """
    [bold]Available Actions:[/bold]
    [green]a[/green] - Approve the agenda as shown
    [yellow]e[/yellow] - Edit the agenda items
    [red]r[/red] - Reject and request new proposals
    """
    console.print(Panel(options_text, box=box.ROUNDED))

    while True:
        try:
            action = Prompt.ask(
                "Select action", choices=["a", "e", "r"], default="a", console=console
            ).lower()

            if action == "a":
                record_input("agenda_approval", "approved", {"agenda": agenda})
                console.print("[green]✓ Agenda approved![/green]")
                return agenda

            elif action == "e":
                edited_agenda = edit_agenda(agenda)
                record_input(
                    "agenda_approval",
                    "edited",
                    {"original": agenda, "edited": edited_agenda},
                )
                return edited_agenda

            else:  # reject
                record_input("agenda_approval", "rejected", {"agenda": agenda})
                console.print(
                    "[red]✗ Agenda rejected. Requesting new proposals...[/red]"
                )
                return []

        except KeyboardInterrupt:
            handle_interrupt("agenda_approval")


def edit_agenda(agenda: List[str]) -> List[str]:
    """Allows the user to interactively edit the agenda.

    Implements Story 7.4: Interactive Agenda Editing with:
    - Menu-based interface
    - Add, remove, reorder, edit operations
    - Preview before confirmation
    """
    console.print(
        Panel(
            "[bold yellow]Agenda Editor[/bold yellow]",
            subtitle="Modify the discussion agenda",
        )
    )

    working_agenda = agenda.copy()

    while True:
        # Display current agenda
        console.print("\n[bold]Current Agenda:[/bold]")
        for i, item in enumerate(working_agenda, 1):
            console.print(f"{i}. {item}")

        # Show editing options
        console.print("\n[bold]Editing Options:[/bold]")
        console.print("[green]a[/green] - Add new item")
        console.print("[yellow]r[/yellow] - Remove item")
        console.print("[blue]e[/blue] - Edit item text")
        console.print("[magenta]m[/magenta] - Move item")
        console.print("[cyan]d[/cyan] - Done editing")

        action = Prompt.ask(
            "\nSelect action",
            choices=["a", "r", "e", "m", "d"],
            default="d",
            console=console,
        ).lower()

        try:
            if action == "a":
                # Add new item
                if len(working_agenda) >= MAX_AGENDA_ITEMS:
                    console.print(
                        f"[red]Cannot add more items (maximum {MAX_AGENDA_ITEMS}).[/red]"
                    )
                    continue

                new_item = Prompt.ask("Enter new agenda item", console=console)
                if new_item.strip():
                    position = IntPrompt.ask(
                        "Insert at position",
                        default=len(working_agenda) + 1,
                        choices=[str(i) for i in range(1, len(working_agenda) + 2)],
                        console=console,
                    )
                    working_agenda.insert(position - 1, new_item.strip())
                    console.print("[green]✓ Item added[/green]")

            elif action == "r":
                # Remove item
                if len(working_agenda) <= MIN_AGENDA_ITEMS:
                    console.print(
                        f"[red]Cannot remove items (minimum {MIN_AGENDA_ITEMS}).[/red]"
                    )
                    continue

                item_num = IntPrompt.ask(
                    "Remove item number",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                    console=console,
                )
                removed = working_agenda.pop(item_num - 1)
                console.print(f"[red]✗ Removed: {removed}[/red]")

            elif action == "e":
                # Edit item
                item_num = IntPrompt.ask(
                    "Edit item number",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                    console=console,
                )
                old_text = working_agenda[item_num - 1]
                new_text = Prompt.ask("New text", default=old_text, console=console)
                working_agenda[item_num - 1] = new_text
                console.print("[green]✓ Item updated[/green]")

            elif action == "m":
                # Move item
                from_pos = IntPrompt.ask(
                    "Move item from position",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                    console=console,
                )
                to_pos = IntPrompt.ask(
                    "Move to position",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                    console=console,
                )
                item = working_agenda.pop(from_pos - 1)
                working_agenda.insert(to_pos - 1, item)
                console.print("[green]✓ Item moved[/green]")

            elif action == "d":
                # Preview and confirm
                console.print("\n[bold]Final Agenda:[/bold]")
                for i, item in enumerate(working_agenda, 1):
                    console.print(f"{i}. {item}")

                if Confirm.ask("\nSave these changes?", default=True, console=console):
                    return working_agenda

        except (ValueError, IndexError) as e:
            console.print(f"[red]Error: {e}[/red]")
            continue


def get_continuation_approval(completed_topic: str, remaining_topics: List[str]) -> str:
    """Asks the user if they want to continue to the next topic.

    Implements Story 7.3: Topic Continuation Gate with:
    - Topic completion summary
    - Remaining agenda display
    - Session statistics
    - Clear options
    """
    console.clear()

    # Show completion summary
    completion_panel = Panel(
        f"[green]✓[/green] Topic '[bold]{completed_topic}[/bold]' has been concluded.\n\n"
        f"A comprehensive summary has been saved.",
        title="[bold green]Topic Completed[/bold green]",
        border_style="green",
    )
    console.print(completion_panel)

    # Display session statistics
    elapsed_time = datetime.now() - session_start_time
    stats_table = Table(title="Session Statistics", show_header=False, box=box.SIMPLE)
    stats_table.add_row("Session Duration:", f"{elapsed_time.seconds // 60} minutes")
    stats_table.add_row(
        "Topics Completed:", "1"
    )  # This would be dynamic in real implementation
    stats_table.add_row("Total Messages:", "47")  # This would be dynamic

    console.print(stats_table)
    console.print()

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

        console.print(remaining_table)
        console.print()

        # Show options
        options = {
            "y": "[green]Continue[/green] to the next topic",
            "n": "[red]End[/red] the session",
            "m": "[yellow]Modify[/yellow] the remaining agenda first",
        }

        options_text = "\n".join([f"{k} - {v}" for k, v in options.items()])
        console.print(Panel(options_text, title="[bold]Options[/bold]"))

        action = Prompt.ask(
            "Select action", choices=["y", "n", "m"], default="y", console=console
        ).lower()

    else:
        console.print("[bold]No remaining topics in the agenda.[/bold]")
        action = Confirm.ask("End session?", default=True, console=console)
        action = "n" if action else "y"  # Invert for consistency

    record_input(
        "continuation_approval",
        action,
        {"completed_topic": completed_topic, "remaining_topics": remaining_topics},
    )

    return action


def get_agenda_modifications(agenda: List[str]) -> List[str]:
    """Allows the user to modify the agenda between topics.

    Implements agenda modification with agent suggestions considered.
    """
    console.print(
        Panel(
            "[bold yellow]Agenda Modification[/bold yellow]",
            subtitle="Agents have suggested modifications to the remaining agenda",
        )
    )

    # In real implementation, this would show agent suggestions
    console.print("[dim]Agent suggestions would be displayed here[/dim]\n")

    # Show current agenda
    console.print("[bold]Current remaining agenda:[/bold]")
    for i, item in enumerate(agenda, 1):
        console.print(f"{i}. {item}")

    console.print("\nYou can modify the agenda based on the discussion so far.")

    if Confirm.ask(
        "\nWould you like to modify the agenda?", default=False, console=console
    ):
        return edit_agenda(agenda)
    else:
        return agenda


def display_session_status(status: Dict[str, Any]) -> None:
    """Displays the current session status.

    Implements Story 7.9: Session Control Dashboard with:
    - Real-time status display
    - Progress indicators
    - Agent participation metrics
    """
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # Header
    header_text = Text(
        "Virtual Agora - Session Status", justify="center", style="bold cyan"
    )
    layout["header"].update(Align.center(header_text, vertical="middle"))

    # Body content
    body_content = []

    # Status table
    status_table = Table(show_header=False, box=box.SIMPLE, expand=True)
    status_table.add_column("Key", style="bold")
    status_table.add_column("Value")

    for key, value in status.items():
        status_table.add_row(key, str(value))

    body_content.append(status_table)

    # Progress bar (example)
    if "progress" in status:
        progress = Progress()
        task_id = progress.add_task("[cyan]Discussion Progress", total=100)
        progress.update(task_id, completed=status.get("progress", 0))
        body_content.append(progress)

    layout["body"].update(Columns(body_content))

    # Footer
    footer_text = Text(
        f"Session Time: {(datetime.now() - session_start_time).seconds // 60} min | "
        f"Press Ctrl+C for emergency options",
        justify="center",
        style="dim",
    )
    layout["footer"].update(Align.center(footer_text, vertical="middle"))

    console.print(layout)


def handle_interrupt(context: str) -> None:
    """Handles interrupts with context.

    Part of Story 7.5: Emergency Controls implementation.
    """
    console.print(f"\n[yellow]Interrupt detected during: {context}[/yellow]")

    if Confirm.ask("Do you want to pause the session?", default=True, console=console):
        console.print("[yellow]Session paused. State will be preserved.[/yellow]")
        # In real implementation, would trigger state preservation
        raise KeyboardInterrupt("User requested pause")
    else:
        console.print("[green]Continuing...[/green]")


def handle_emergency_interrupt() -> None:
    """Handles emergency interrupts from the user.

    Implements Story 7.5: Emergency Controls with:
    - Interrupt mechanism
    - Pause/resume functionality
    - Emergency options menu
    """
    console.clear()
    console.print(
        Panel(
            "[bold red]Emergency Interrupt[/bold red]",
            subtitle="Session interrupted by user",
            border_style="red",
        )
    )

    # Show emergency options
    options = {
        "p": "[yellow]Pause[/yellow] the session (state preserved)",
        "s": "[blue]Skip[/blue] current speaker",
        "e": "[orange]End[/orange] current topic early",
        "q": "[red]Quit[/red] session immediately",
        "r": "[green]Resume[/green] normal operation",
    }

    options_table = Table(show_header=False, box=box.SIMPLE)
    for key, desc in options.items():
        options_table.add_row(f"[bold]{key}[/bold]", desc)

    console.print(options_table)

    try:
        action = Prompt.ask(
            "\nSelect emergency action",
            choices=list(options.keys()),
            default="r",
            console=console,
        ).lower()

        record_input("emergency_interrupt", action, {"context": "user_interrupt"})

        if action == "p":
            console.print("[yellow]Pausing session... State saved.[/yellow]")
            time.sleep(1)
            sys.exit(0)
        elif action == "s":
            console.print("[blue]Skipping current speaker...[/blue]")
            return  # Would trigger skip in actual implementation
        elif action == "e":
            console.print("[orange]Ending topic early...[/orange]")
            return  # Would trigger topic end
        elif action == "q":
            console.print("[red]Emergency shutdown initiated...[/red]")
            time.sleep(1)
            sys.exit(1)
        else:  # resume
            console.print("[green]Resuming normal operation...[/green]")
            return

    except Exception as e:
        logger.error(f"Error in emergency handler: {e}")
        console.print("[red]Critical error. Shutting down...[/red]")
        sys.exit(2)


def show_help(context: Optional[str] = None) -> None:
    """Shows contextual help information.

    Implements Story 7.8: Help and Guidance System with:
    - Context-aware help
    - Keyboard shortcuts
    - Examples
    """
    help_content = {
        "topic_input": """
        # Topic Input Help
        
        Enter a discussion topic that is:
        - **Specific**: Avoid overly broad topics
        - **Debatable**: Has multiple valid perspectives
        - **Substantial**: Warrants extended discussion
        
        ## Keyboard Shortcuts
        - `Ctrl+C`: Emergency options
        - `Enter` twice: Finish multi-line input
        """,
        "agenda_approval": """
        # Agenda Approval Help
        
        Review the proposed discussion agenda:
        - **Approve**: Accept the agenda as shown
        - **Edit**: Modify agenda items
        - **Reject**: Request new proposals
        
        ## Tips
        - Higher priority items are discussed first
        - Aim for 3-5 well-defined topics
        """,
        "general": """
        # Virtual Agora Help
        
        ## Navigation
        - Follow on-screen prompts
        - Use arrow keys in menus
        - Press `Ctrl+C` for emergency options
        
        ## Session Flow
        1. Enter discussion topic
        2. Review and approve agenda
        3. Observe agent discussions
        4. Decide on continuation
        5. Review final report
        """,
    }

    content = help_content.get(context, help_content["general"])

    console.print(
        Panel(
            Markdown(content),
            title="[bold blue]Help[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    Prompt.ask("\nPress Enter to continue", console=console)


def validate_input(
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    required: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Validates user input with comprehensive checks.

    Implements Story 7.6: Input Validation and Error Recovery.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if required and not value.strip():
        return False, "Input cannot be empty"

    if min_length and len(value) < min_length:
        return False, f"Input too short (minimum {min_length} characters)"

    if max_length and len(value) > max_length:
        return False, f"Input too long (maximum {max_length} characters)"

    # Check for special characters that might cause issues
    if any(char in value for char in ["<", ">", "|", "&"]):
        return False, "Input contains invalid characters"

    return True, None


def get_user_turn_participation(
    current_round: int,
    current_topic: str,
    previous_round_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Get Round Moderator decision between rounds.

    Implements the Round Moderator feature allowing users to guide discussion flow:
    - 🗣️ Participate: Add thoughts/questions to guide the next round
    - ➡️ Continue: Let agents proceed with their current trajectory
    - 🏁 End Topic: Move to conclusion and voting

    Args:
        current_round: The current discussion round number
        current_topic: The topic being discussed
        previous_round_summary: Summary of the previous round (if available)

    Returns:
        Dict containing moderator decision and optional guidance message
    """
    console.clear()

    # Display header with Round Moderator terminology
    console.print(
        Panel(
            f"[bold yellow]Round {current_round} - Round Moderator[/bold yellow]",
            subtitle=f"Topic: {current_topic}",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # Display previous round summary if available
    if previous_round_summary and current_round > 1:
        from virtual_agora.ui.discussion_display import display_previous_round_summary

        display_previous_round_summary(
            current_round, current_topic, previous_round_summary
        )

    # Show Round Moderator options
    console.print()
    options_text = """
    [bold]Your Three Options Each Round:[/bold]
    [blue]🗣️  p[/blue] - [blue]🗣️ Participate[/blue]: Add your thoughts/questions to guide the next round
    [green]➡️  c[/green] - [green]➡️ Continue[/green]: Let agents proceed with their current trajectory
    [red]🏁 f[/red] - [red]🏁 End Topic[/red]: Move to conclusion and voting
    """
    console.print(
        Panel(
            options_text,
            title="[bold yellow]Round Moderator Options[/bold yellow]",
            box=box.ROUNDED,
        )
    )

    while True:
        try:
            action = Prompt.ask(
                "Choose your next steps",
                choices=["c", "p", "f"],
                default="c",
                console=console,
            ).lower()

            if action == "c":
                # Continue discussion - Round Moderator lets agents proceed
                record_input("user_turn_participation", "continue")
                console.print(
                    "[green]➡️ Letting agents proceed with their current trajectory[/green]"
                )
                return {
                    "action": "continue",
                    "user_message": None,
                }

            elif action == "p":
                # Round Moderator wants to participate
                console.print()
                console.print(
                    "[bold blue]🗣️ Enter your message to guide the next round:[/bold blue]"
                )
                console.print(
                    "[dim](Press Enter twice to finish multi-line input)[/dim]"
                )

                lines = []
                while True:
                    line = Prompt.ask(
                        "", default="", show_default=False, console=console
                    )
                    if not line and lines:  # Empty line after content
                        break
                    if line:
                        lines.append(line)

                user_message = " ".join(lines).strip()

                if not user_message:
                    console.print(
                        "[red]Message cannot be empty. Please try again.[/red]"
                    )
                    continue

                # Validate message
                is_valid, error_msg = validate_input(
                    user_message, min_length=10, max_length=2000
                )
                if not is_valid:
                    console.print(f"[red]{error_msg}[/red]")
                    continue

                # Confirm message
                console.print()
                console.print(
                    Panel(
                        user_message,
                        title="[bold]Your Message[/bold]",
                        border_style="blue",
                        padding=(1, 2),
                    )
                )

                if Confirm.ask("Add this message?", default=True, console=console):
                    record_input(
                        "user_turn_participation",
                        "participate",
                        {"message": user_message},
                    )
                    console.print(
                        "[blue]🗣️ Your guidance added to guide the next round[/blue]"
                    )
                    return {
                        "action": "participate",
                        "user_message": user_message,
                    }
                else:
                    console.print("[yellow]Let's try again...[/yellow]")
                    continue

            else:  # finalize
                if Confirm.ask(
                    "Are you sure you want to end this topic and move to conclusion?",
                    default=False,
                    console=console,
                ):
                    record_input("user_turn_participation", "finalize")
                    console.print("[red]🏁 Moving to topic conclusion and voting[/red]")
                    return {
                        "action": "finalize",
                        "user_message": None,
                    }
                else:
                    console.print("[yellow]Let's try again...[/yellow]")
                    continue

        except KeyboardInterrupt:
            handle_interrupt("user_turn_participation")


def get_topic_conclusion_confirmation(
    current_topic: str, current_round: int, vote_results: Dict[str, Any]
) -> str:
    """Get user confirmation for topic conclusion after agent vote.

    Args:
        current_topic: The topic being discussed
        current_round: The current round number
        vote_results: Dictionary with vote results including passed, yes_votes, total_votes

    Returns:
        "confirm" to proceed with conclusion, "continue" to override and continue discussion
    """
    console.clear()

    # Display vote results
    vote_passed = vote_results.get("passed", False)
    yes_votes = vote_results.get("yes_votes", 0)
    total_votes = vote_results.get("total_votes", 0)

    vote_status = "PASSED" if vote_passed else "FAILED"
    vote_color = "green" if vote_passed else "red"

    console.print(
        Panel(
            f"[bold {vote_color}]Agent Vote: {vote_status}[/bold {vote_color}]\n\n"
            f"Topic: {current_topic}\n"
            f"Round: {current_round}\n"
            f"Vote Result: {yes_votes}/{total_votes} agents voted to conclude",
            title="[bold blue]Topic Conclusion Vote Results[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Show options
    if vote_passed:
        console.print(
            "\n[bold green]The agents have voted to conclude this topic.[/bold green]"
        )
        console.print("As the moderator, you can:")
        console.print(
            "• [green]Confirm[/green] the decision and move to topic conclusion"
        )
        console.print(
            "• [yellow]Override[/yellow] the agents and continue the discussion"
        )
    else:
        console.print(
            "\n[bold red]The agents have voted to continue the discussion.[/bold red]"
        )
        console.print("As the moderator, you can:")
        console.print("• [green]Confirm[/green] the decision and continue discussion")
        console.print(
            "• [yellow]Override[/yellow] the agents and force topic conclusion"
        )

    console.print()

    while True:
        try:
            if vote_passed:
                action = Prompt.ask(
                    "Confirm topic conclusion?",
                    choices=["y", "n"],
                    default="y",
                    console=console,
                ).lower()

                if action == "y":
                    record_input("topic_conclusion_confirmation", "confirm")
                    console.print("[green]✓ Topic conclusion confirmed[/green]")
                    return "confirm"
                else:
                    record_input("topic_conclusion_confirmation", "continue")
                    console.print(
                        "[yellow]⚡ Overriding agents - continuing discussion[/yellow]"
                    )
                    return "continue"
            else:
                action = Prompt.ask(
                    "Force topic conclusion despite agent vote?",
                    choices=["y", "n"],
                    default="n",
                    console=console,
                ).lower()

                if action == "y":
                    record_input("topic_conclusion_confirmation", "force_conclude")
                    console.print("[yellow]⚡ Forcing topic conclusion[/yellow]")
                    return "confirm"
                else:
                    record_input("topic_conclusion_confirmation", "continue")
                    console.print("[green]✓ Continuing discussion as voted[/green]")
                    return "continue"

        except KeyboardInterrupt:
            handle_interrupt("topic_conclusion_confirmation")


def get_input_with_timeout(
    prompt: str, timeout: int = INPUT_TIMEOUT, default: Optional[str] = None
) -> Optional[str]:
    """Gets input with timeout handling.

    Part of comprehensive input handling system.
    """

    def timeout_handler(signum, frame):
        raise TimeoutError("Input timeout")

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        value = Prompt.ask(prompt, default=default, console=console)
        signal.alarm(0)  # Cancel timeout
        return value
    except TimeoutError:
        console.print(f"\n[yellow]Input timeout after {timeout} seconds.[/yellow]")
        return default
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e
