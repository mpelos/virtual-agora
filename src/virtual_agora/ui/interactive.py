"""Interactive menus and prompts for Virtual Agora terminal UI.

This module provides enhanced user interaction components including
menus, prompts, confirmations, and keyboard navigation support.
"""

import sys
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass

from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.console import Group
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme, MessageType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of interactive prompts."""

    TEXT = "text"
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    CONFIRMATION = "confirmation"
    NUMBER = "number"
    AGENDA_EDIT = "agenda_edit"


class ValidationResult(Enum):
    """Validation result types."""

    VALID = "valid"
    INVALID = "invalid"
    RETRY = "retry"


@dataclass
class PromptOption:
    """Single prompt option definition."""

    key: str
    label: str
    description: Optional[str] = None
    value: Any = None
    style: Optional[str] = None


@dataclass
class ValidationRule:
    """Validation rule for user input."""

    validator: Callable[[str], bool]
    message: str
    allow_retry: bool = True


class VirtualAgoraPrompts:
    """Enhanced prompt system for Virtual Agora."""

    def __init__(self):
        """Initialize prompt system."""
        self.console = get_console()
        self.theme = get_current_theme()

    def text_prompt(
        self,
        prompt: str,
        default: Optional[str] = None,
        password: bool = False,
        validation_rules: Optional[List[ValidationRule]] = None,
        show_default: bool = True,
    ) -> str:
        """Enhanced text input prompt with validation."""

        # Display prompt panel
        style = self.theme.get_message_style(MessageType.USER_PROMPT)

        prompt_content = (
            f"[{style['color']}]{style['icon']} {prompt}[/{style['color']}]"
        )
        if default and show_default:
            prompt_content += f"\n[dim]Default: {default}[/dim]"

        panel = Panel(
            prompt_content,
            title="[bold]Input Required[/bold]",
            border_style=style["border"],
            padding=(1, 2),
        )

        self.console.print(panel)

        # Input loop with validation
        while True:
            try:
                if password:
                    result = Prompt.ask(
                        "[cyan]❯[/cyan]",
                        default=default,
                        password=True,
                        console=self.console.rich_console,
                    )
                else:
                    result = Prompt.ask(
                        "[cyan]❯[/cyan]",
                        default=default,
                        console=self.console.rich_console,
                    )

                # Apply validation rules
                if validation_rules:
                    for rule in validation_rules:
                        if not rule.validator(result):
                            self.console.print_system_message(
                                rule.message, MessageType.ERROR
                            )
                            if rule.allow_retry:
                                continue
                            else:
                                raise ValueError(rule.message)

                return result

            except KeyboardInterrupt:
                self.console.print_system_message(
                    "Input cancelled by user", MessageType.WARNING
                )
                raise
            except Exception as e:
                self.console.print_system_message(
                    f"Input error: {e}", MessageType.ERROR
                )

    def choice_prompt(
        self,
        prompt: str,
        choices: List[PromptOption],
        default: Optional[str] = None,
        allow_custom: bool = False,
    ) -> Union[PromptOption, str]:
        """Single choice prompt with enhanced display."""

        # Create choice table
        table = Table(
            title=f"[bold cyan]{prompt}[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Key", width=6, justify="center")
        table.add_column("Option", style="white")
        table.add_column("Description", style="dim")

        # Build choice mapping
        choice_map = {}
        valid_keys = []

        for choice in choices:
            choice_map[choice.key.lower()] = choice
            valid_keys.append(choice.key.lower())

            # Style the key
            key_style = choice.style or "cyan"
            styled_key = f"[{key_style}]{choice.key}[/{key_style}]"

            # Add default indicator
            label = choice.label
            if default and choice.key.lower() == default.lower():
                label = f"[bold]{label}[/bold] [dim](default)[/dim]"

            table.add_row(styled_key, label, choice.description or "")

        # Add custom option if allowed
        if allow_custom:
            table.add_row(
                "[yellow]other[/yellow]", "Custom input", "Enter your own value"
            )
            valid_keys.append("other")

        # Display table
        panel = Panel(table, border_style="cyan", padding=(1, 1))

        self.console.print(panel)

        # Input loop
        while True:
            try:
                response = Prompt.ask(
                    "[cyan]Your choice[/cyan]",
                    default=default,
                    console=self.console.rich_console,
                ).lower()

                if response in choice_map:
                    return choice_map[response]
                elif allow_custom and response == "other":
                    custom_value = self.text_prompt("Enter custom value:")
                    return custom_value
                else:
                    valid_options = ", ".join(valid_keys)
                    self.console.print_system_message(
                        f"Invalid choice. Valid options: {valid_options}",
                        MessageType.ERROR,
                    )

            except KeyboardInterrupt:
                self.console.print_system_message(
                    "Selection cancelled by user", MessageType.WARNING
                )
                raise

    def multi_choice_prompt(
        self,
        prompt: str,
        choices: List[PromptOption],
        min_selections: int = 1,
        max_selections: Optional[int] = None,
    ) -> List[PromptOption]:
        """Multiple choice prompt with selection validation."""

        # Display choices
        table = Table(
            title=f"[bold cyan]{prompt}[/bold cyan]\n[dim]Select multiple options (comma-separated)[/dim]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Key", width=6, justify="center")
        table.add_column("Option", style="white")
        table.add_column("Description", style="dim")

        choice_map = {}
        for choice in choices:
            choice_map[choice.key.lower()] = choice

            key_style = choice.style or "cyan"
            styled_key = f"[{key_style}]{choice.key}[/{key_style}]"

            table.add_row(styled_key, choice.label, choice.description or "")

        panel = Panel(table, border_style="cyan", padding=(1, 1))
        self.console.print(panel)

        # Input and validation loop
        while True:
            try:
                response = Prompt.ask(
                    "[cyan]Your selections (comma-separated)[/cyan]",
                    console=self.console.rich_console,
                )

                # Parse selections
                selections = [s.strip().lower() for s in response.split(",")]
                selected_choices = []
                invalid_choices = []

                for selection in selections:
                    if selection in choice_map:
                        selected_choices.append(choice_map[selection])
                    elif selection:  # Ignore empty strings
                        invalid_choices.append(selection)

                # Validation
                if invalid_choices:
                    valid_options = ", ".join(choice_map.keys())
                    self.console.print_system_message(
                        f"Invalid choices: {', '.join(invalid_choices)}. Valid options: {valid_options}",
                        MessageType.ERROR,
                    )
                    continue

                if len(selected_choices) < min_selections:
                    self.console.print_system_message(
                        f"Please select at least {min_selections} option(s). You selected {len(selected_choices)}.",
                        MessageType.ERROR,
                    )
                    continue

                if max_selections and len(selected_choices) > max_selections:
                    self.console.print_system_message(
                        f"Please select at most {max_selections} option(s). You selected {len(selected_choices)}.",
                        MessageType.ERROR,
                    )
                    continue

                return selected_choices

            except KeyboardInterrupt:
                self.console.print_system_message(
                    "Selection cancelled by user", MessageType.WARNING
                )
                raise

    def confirmation_prompt(
        self,
        prompt: str,
        default: bool = True,
        danger: bool = False,
        show_details: Optional[str] = None,
    ) -> bool:
        """Enhanced confirmation prompt with danger styling."""

        # Choose styling based on danger level
        if danger:
            style = self.theme.get_message_style(MessageType.ERROR)
            border_style = "red"
        else:
            style = self.theme.get_message_style(MessageType.WARNING)
            border_style = "yellow"

        # Build content
        content_lines = [
            f"[{style['color']}]{style['icon']} {prompt}[/{style['color']}]"
        ]

        if show_details:
            content_lines.append("")
            content_lines.append(f"[dim]{show_details}[/dim]")

        default_text = "Y/n" if default else "y/N"
        content_lines.append("")
        content_lines.append(f"[cyan]Continue? ({default_text})[/cyan]")

        content = "\n".join(content_lines)

        # Display panel
        panel = Panel(
            content,
            title="[bold]Confirmation Required[/bold]",
            border_style=border_style,
            padding=(1, 2),
        )

        self.console.print(panel)

        # Get confirmation
        return Confirm.ask(
            "[cyan]❯[/cyan]", default=default, console=self.console.rich_console
        )

    def number_prompt(
        self,
        prompt: str,
        default: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """Number input prompt with range validation."""

        # Build prompt text
        range_text = ""
        if min_value is not None and max_value is not None:
            range_text = f" (range: {min_value}-{max_value})"
        elif min_value is not None:
            range_text = f" (minimum: {min_value})"
        elif max_value is not None:
            range_text = f" (maximum: {max_value})"

        full_prompt = f"{prompt}{range_text}"

        # Display prompt
        style = self.theme.get_message_style(MessageType.USER_PROMPT)

        prompt_content = (
            f"[{style['color']}]{style['icon']} {full_prompt}[/{style['color']}]"
        )
        if default is not None:
            prompt_content += f"\n[dim]Default: {default}[/dim]"

        panel = Panel(
            prompt_content,
            title="[bold]Number Input Required[/bold]",
            border_style=style["border"],
            padding=(1, 2),
        )

        self.console.print(panel)

        # Input loop with validation
        while True:
            try:
                result = IntPrompt.ask(
                    "[cyan]❯[/cyan]", default=default, console=self.console.rich_console
                )

                # Range validation
                if min_value is not None and result < min_value:
                    self.console.print_system_message(
                        f"Value must be at least {min_value}. You entered {result}.",
                        MessageType.ERROR,
                    )
                    continue

                if max_value is not None and result > max_value:
                    self.console.print_system_message(
                        f"Value must be at most {max_value}. You entered {result}.",
                        MessageType.ERROR,
                    )
                    continue

                return result

            except KeyboardInterrupt:
                self.console.print_system_message(
                    "Input cancelled by user", MessageType.WARNING
                )
                raise

    def agenda_edit_prompt(self, current_agenda: List[str]) -> List[str]:
        """Interactive agenda editing prompt."""

        self.console.print_system_message(
            "Interactive Agenda Editor", MessageType.SYSTEM_INFO, title="Agenda Editor"
        )

        # Display current agenda
        self._display_agenda_for_editing(current_agenda)

        # Edit operations
        operations = [
            PromptOption("a", "Add topic", "Add a new topic to the agenda"),
            PromptOption("r", "Remove topic", "Remove a topic from the agenda"),
            PromptOption("m", "Move topic", "Change the order of topics"),
            PromptOption("e", "Edit topic", "Modify an existing topic"),
            PromptOption("d", "Done", "Finish editing and save changes", style="green"),
        ]

        agenda = current_agenda.copy()

        while True:
            choice = self.choice_prompt(
                "What would you like to do?", operations, default="d"
            )

            if choice.key == "a":
                new_topic = self.text_prompt("Enter new topic:")
                position = self.number_prompt(
                    "Insert at position",
                    default=len(agenda) + 1,
                    min_value=1,
                    max_value=len(agenda) + 1,
                )
                agenda.insert(position - 1, new_topic)

            elif choice.key == "r":
                if not agenda:
                    self.console.print_system_message(
                        "No topics to remove", MessageType.WARNING
                    )
                    continue

                position = self.number_prompt(
                    "Remove topic at position", min_value=1, max_value=len(agenda)
                )
                removed = agenda.pop(position - 1)
                self.console.print_system_message(
                    f"Removed: {removed}", MessageType.SUCCESS
                )

            elif choice.key == "m":
                if len(agenda) < 2:
                    self.console.print_system_message(
                        "Need at least 2 topics to reorder", MessageType.WARNING
                    )
                    continue

                from_pos = self.number_prompt(
                    "Move topic from position", min_value=1, max_value=len(agenda)
                )
                to_pos = self.number_prompt(
                    "Move to position", min_value=1, max_value=len(agenda)
                )

                topic = agenda.pop(from_pos - 1)
                agenda.insert(to_pos - 1, topic)

            elif choice.key == "e":
                if not agenda:
                    self.console.print_system_message(
                        "No topics to edit", MessageType.WARNING
                    )
                    continue

                position = self.number_prompt(
                    "Edit topic at position", min_value=1, max_value=len(agenda)
                )

                current_topic = agenda[position - 1]
                self.console.print_system_message(
                    f"Current: {current_topic}", MessageType.SYSTEM_INFO
                )

                new_topic = self.text_prompt(
                    "Enter new topic text:", default=current_topic
                )
                agenda[position - 1] = new_topic

            elif choice.key == "d":
                break

            # Show updated agenda
            self._display_agenda_for_editing(agenda)

        return agenda

    def _display_agenda_for_editing(self, agenda: List[str]) -> None:
        """Display agenda in editing format."""
        if not agenda:
            self.console.print_system_message("Agenda is empty", MessageType.WARNING)
            return

        table = Table(
            title="[bold]Current Agenda[/bold]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("#", width=3, justify="center")
        table.add_column("Topic", style="white")

        for i, topic in enumerate(agenda, 1):
            table.add_row(str(i), topic)

        self.console.print(table)
        self.console.print()


# Global prompt instance
_prompts: Optional[VirtualAgoraPrompts] = None


def get_prompts() -> VirtualAgoraPrompts:
    """Get the global prompts instance."""
    global _prompts
    if _prompts is None:
        _prompts = VirtualAgoraPrompts()
    return _prompts


# Convenience functions


def ask_text(prompt: str, **kwargs) -> str:
    """Convenience function for text prompt."""
    return get_prompts().text_prompt(prompt, **kwargs)


def ask_choice(
    prompt: str, choices: List[PromptOption], **kwargs
) -> Union[PromptOption, str]:
    """Convenience function for choice prompt."""
    return get_prompts().choice_prompt(prompt, choices, **kwargs)


def ask_multi_choice(
    prompt: str, choices: List[PromptOption], **kwargs
) -> List[PromptOption]:
    """Convenience function for multi-choice prompt."""
    return get_prompts().multi_choice_prompt(prompt, choices, **kwargs)


def ask_confirmation(prompt: str, **kwargs) -> bool:
    """Convenience function for confirmation prompt."""
    return get_prompts().confirmation_prompt(prompt, **kwargs)


def ask_number(prompt: str, **kwargs) -> int:
    """Convenience function for number prompt."""
    return get_prompts().number_prompt(prompt, **kwargs)


def edit_agenda(current_agenda: List[str]) -> List[str]:
    """Convenience function for agenda editing."""
    return get_prompts().agenda_edit_prompt(current_agenda)


# Validation helpers


def create_length_validator(
    min_length: int = 0, max_length: int = 1000
) -> ValidationRule:
    """Create a text length validation rule."""

    def validator(text: str) -> bool:
        return min_length <= len(text.strip()) <= max_length

    return ValidationRule(
        validator=validator,
        message=f"Text must be between {min_length} and {max_length} characters",
    )


def create_non_empty_validator() -> ValidationRule:
    """Create a non-empty text validation rule."""

    def validator(text: str) -> bool:
        return bool(text.strip())

    return ValidationRule(validator=validator, message="Input cannot be empty")


def create_choice_validator(valid_choices: List[str]) -> ValidationRule:
    """Create a choice validation rule."""
    valid_set = set(choice.lower() for choice in valid_choices)

    def validator(text: str) -> bool:
        return text.lower() in valid_set

    return ValidationRule(
        validator=validator, message=f"Must be one of: {', '.join(valid_choices)}"
    )
