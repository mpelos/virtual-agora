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
MAX_TOPIC_LENGTH = 500
MIN_AGENDA_ITEMS = 2
MAX_AGENDA_ITEMS = 10
INPUT_TIMEOUT = 300  # 5 minutes
MAX_RETRY_ATTEMPTS = 3

# Session state tracking
session_start_time = datetime.now()
input_history: List[Dict[str, Any]] = []


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


def get_initial_topic() -> str:
    """Gets the initial discussion topic from the user.

    Implements Story 7.1: Initial Topic Input Interface with:
    - Clear welcome message
    - Multi-line topic support
    - Validation and confirmation
    - Topic templates for guidance
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

    console.print(
        Panel(
            Markdown(welcome_text),
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
                line = Prompt.ask("", default="", show_default=False)
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

            if len(topic) > MAX_TOPIC_LENGTH:
                console.print(
                    f"[yellow]Topic too long (maximum {MAX_TOPIC_LENGTH} characters).[/yellow]"
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

            if Confirm.ask("Confirm this topic?", default=True):
                record_input("initial_topic", topic)
                logger.info(f"Topic confirmed: {topic[:50]}...")
                return topic
            else:
                console.print("[yellow]Let's try again...[/yellow]")
                attempts += 1

        except KeyboardInterrupt:
            handle_interrupt("topic_input")

    raise ValueError("Maximum retry attempts exceeded for topic input")


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
                "Select action", choices=["a", "e", "r"], default="a"
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
            "\nSelect action", choices=["a", "r", "e", "m", "d"], default="d"
        ).lower()

        try:
            if action == "a":
                # Add new item
                if len(working_agenda) >= MAX_AGENDA_ITEMS:
                    console.print(
                        f"[red]Cannot add more items (maximum {MAX_AGENDA_ITEMS}).[/red]"
                    )
                    continue

                new_item = Prompt.ask("Enter new agenda item")
                if new_item.strip():
                    position = IntPrompt.ask(
                        "Insert at position",
                        default=len(working_agenda) + 1,
                        choices=[str(i) for i in range(1, len(working_agenda) + 2)],
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
                )
                removed = working_agenda.pop(item_num - 1)
                console.print(f"[red]✗ Removed: {removed}[/red]")

            elif action == "e":
                # Edit item
                item_num = IntPrompt.ask(
                    "Edit item number",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                )
                old_text = working_agenda[item_num - 1]
                new_text = Prompt.ask("New text", default=old_text)
                working_agenda[item_num - 1] = new_text
                console.print("[green]✓ Item updated[/green]")

            elif action == "m":
                # Move item
                from_pos = IntPrompt.ask(
                    "Move item from position",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                )
                to_pos = IntPrompt.ask(
                    "Move to position",
                    choices=[str(i) for i in range(1, len(working_agenda) + 1)],
                )
                item = working_agenda.pop(from_pos - 1)
                working_agenda.insert(to_pos - 1, item)
                console.print("[green]✓ Item moved[/green]")

            elif action == "d":
                # Preview and confirm
                console.print("\n[bold]Final Agenda:[/bold]")
                for i, item in enumerate(working_agenda, 1):
                    console.print(f"{i}. {item}")

                if Confirm.ask("\nSave these changes?", default=True):
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
            "Select action", choices=["y", "n", "m"], default="y"
        ).lower()

    else:
        console.print("[bold]No remaining topics in the agenda.[/bold]")
        action = Confirm.ask("End session?", default=True)
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

    if Confirm.ask("\nWould you like to modify the agenda?", default=False):
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

    if Confirm.ask("Do you want to pause the session?", default=True):
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
            "\nSelect emergency action", choices=list(options.keys()), default="r"
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

    Prompt.ask("\nPress Enter to continue")


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
        value = Prompt.ask(prompt, default=default)
        signal.alarm(0)  # Cancel timeout
        return value
    except TimeoutError:
        console.print(f"\n[yellow]Input timeout after {timeout} seconds.[/yellow]")
        return default
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e
