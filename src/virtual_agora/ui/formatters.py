"""Advanced output formatting utilities for Virtual Agora terminal UI.

This module provides comprehensive formatting capabilities including
markdown rendering, code highlighting, table formatting, and export utilities.
"""

import re
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.console import Group
from rich.tree import Tree
from rich.pretty import Pretty
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme, ProviderType
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class FormatType(Enum):
    """Output format types."""

    MARKDOWN = "markdown"
    JSON = "json"
    TABLE = "table"
    CODE = "code"
    LIST = "list"
    TREE = "tree"
    PLAIN = "plain"


class ExportFormat(Enum):
    """Export format types."""

    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    JSON = "json"
    CSV = "csv"


@dataclass
class FormattingOptions:
    """Formatting configuration options."""

    width: Optional[int] = None
    word_wrap: bool = True
    show_line_numbers: bool = False
    highlight_syntax: bool = True
    expand_tables: bool = True
    preserve_whitespace: bool = False
    color_output: bool = True


class VirtualAgoraFormatter:
    """Advanced formatting system for Virtual Agora output."""

    def __init__(self, options: Optional[FormattingOptions] = None):
        """Initialize formatter."""
        self.console = get_console()
        self.theme = get_current_theme()
        self.options = options or FormattingOptions()

    def format_markdown(self, content: str, title: Optional[str] = None) -> Panel:
        """Format markdown content with syntax highlighting."""
        try:
            md = Markdown(content, code_theme="monokai", inline_code_theme="monokai")

            return Panel(
                md,
                title=title,
                border_style="cyan",
                padding=(1, 2),
                width=self.options.width,
                expand=self.options.expand_tables,
            )

        except Exception as e:
            logger.warning(f"Markdown formatting error: {e}")
            return self.format_plain_text(content, title)

    def format_code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
        line_numbers: Optional[bool] = None,
    ) -> Panel:
        """Format code with syntax highlighting."""
        show_line_numbers = (
            line_numbers if line_numbers is not None else self.options.show_line_numbers
        )

        try:
            syntax = Syntax(
                code,
                language,
                theme="monokai",
                line_numbers=show_line_numbers,
                word_wrap=self.options.word_wrap,
                background_color="default",
            )

            return Panel(
                syntax,
                title=title or f"[bold]{language.title()} Code[/bold]",
                border_style="green",
                padding=(1, 2),
                width=self.options.width,
            )

        except Exception as e:
            logger.warning(f"Code formatting error: {e}")
            return self.format_plain_text(code, title)

    def format_json(
        self, data: Union[Dict, List, str], title: Optional[str] = None
    ) -> Panel:
        """Format JSON data with syntax highlighting."""
        try:
            if isinstance(data, str):
                # Try to parse string as JSON
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return self.format_plain_text(data, title)

            # Pretty print JSON
            json_str = json.dumps(data, indent=2, ensure_ascii=False)

            syntax = Syntax(
                json_str,
                "json",
                theme="monokai",
                line_numbers=self.options.show_line_numbers,
                word_wrap=self.options.word_wrap,
            )

            return Panel(
                syntax,
                title=title or "[bold]JSON Data[/bold]",
                border_style="yellow",
                padding=(1, 2),
                width=self.options.width,
            )

        except Exception as e:
            logger.warning(f"JSON formatting error: {e}")
            return self.format_plain_text(str(data), title)

    def format_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        headers: Optional[List[str]] = None,
    ) -> Panel:
        """Format data as a table."""
        if not data:
            return Panel("[dim]No data to display[/dim]", title=title)

        try:
            # Auto-detect headers if not provided
            if headers is None:
                headers = list(data[0].keys()) if data else []

            table = Table(
                title=title,
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                expand=self.options.expand_tables,
            )

            # Add columns
            for header in headers:
                table.add_column(str(header), style="white")

            # Add rows
            for row in data:
                row_values = []
                for header in headers:
                    value = row.get(header, "")
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    row_values.append(str(value))

                table.add_row(*row_values)

            return Panel(table, border_style="cyan", padding=(1, 1))

        except Exception as e:
            logger.warning(f"Table formatting error: {e}")
            return self.format_plain_text(str(data), title)

    def format_list(
        self,
        items: List[Any],
        title: Optional[str] = None,
        numbered: bool = False,
        bullet_style: str = "•",
    ) -> Panel:
        """Format data as a list."""
        if not items:
            return Panel("[dim]No items to display[/dim]", title=title)

        content_lines = []

        for i, item in enumerate(items, 1):
            if numbered:
                prefix = f"[cyan]{i}.[/cyan]"
            else:
                prefix = f"[cyan]{bullet_style}[/cyan]"

            item_str = str(item)
            if len(item_str) > 80:
                item_str = item_str[:77] + "..."

            content_lines.append(f"{prefix} {item_str}")

        content = "\n".join(content_lines)

        return Panel(
            content,
            title=title or "[bold]List[/bold]",
            border_style="green",
            padding=(1, 2),
            width=self.options.width,
        )

    def format_tree(self, data: Dict[str, Any], title: Optional[str] = None) -> Panel:
        """Format hierarchical data as a tree."""
        try:
            tree = Tree(title or "[bold]Tree Structure[/bold]")

            def add_to_tree(node, parent_data):
                if isinstance(parent_data, dict):
                    for key, value in parent_data.items():
                        if isinstance(value, (dict, list)):
                            branch = node.add(f"[bold]{key}[/bold]")
                            add_to_tree(branch, value)
                        else:
                            value_str = str(value)
                            if len(value_str) > 40:
                                value_str = value_str[:37] + "..."
                            node.add(f"{key}: [dim]{value_str}[/dim]")
                elif isinstance(parent_data, list):
                    for i, item in enumerate(parent_data):
                        if isinstance(item, (dict, list)):
                            branch = node.add(f"[bold]Item {i+1}[/bold]")
                            add_to_tree(branch, item)
                        else:
                            item_str = str(item)
                            if len(item_str) > 40:
                                item_str = item_str[:37] + "..."
                            node.add(f"[dim]{item_str}[/dim]")

            add_to_tree(tree, data)

            return Panel(tree, border_style="magenta", padding=(1, 2))

        except Exception as e:
            logger.warning(f"Tree formatting error: {e}")
            return self.format_plain_text(str(data), title)

    def format_plain_text(self, content: str, title: Optional[str] = None) -> Panel:
        """Format plain text content."""
        if self.options.word_wrap and self.options.width:
            # Simple word wrapping
            words = content.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if (
                    current_length + len(word) + 1 > self.options.width - 4
                ):  # Account for padding
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        lines.append(word)
                        current_length = 0
                else:
                    current_line.append(word)
                    current_length += len(word) + 1

            if current_line:
                lines.append(" ".join(current_line))

            content = "\n".join(lines)

        return Panel(
            content,
            title=title,
            border_style="white",
            padding=(1, 2),
            width=self.options.width,
        )

    def format_voting_summary(self, votes: Dict[str, Dict[str, Any]]) -> Panel:
        """Format voting results with analysis."""
        if not votes:
            return Panel("[dim]No votes to display[/dim]", title="Voting Summary")

        # Analyze votes
        yes_count = sum(
            1 for vote in votes.values() if vote.get("vote", "").lower() == "yes"
        )
        no_count = sum(
            1 for vote in votes.values() if vote.get("vote", "").lower() == "no"
        )
        abstain_count = len(votes) - yes_count - no_count

        total_votes = len(votes)
        majority_needed = (total_votes // 2) + 1

        # Create summary table
        summary_table = Table(box=box.SIMPLE, show_header=False)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Votes", str(total_votes))
        summary_table.add_row("Yes Votes", f"[green]{yes_count}[/green]")
        summary_table.add_row("No Votes", f"[red]{no_count}[/red]")
        summary_table.add_row("Abstentions", f"[yellow]{abstain_count}[/yellow]")
        summary_table.add_row("Majority Needed", str(majority_needed))

        # Determine result
        if yes_count >= majority_needed:
            result = "[green]PASSES[/green]"
            result_icon = "✅"
        else:
            result = "[red]FAILS[/red]"
            result_icon = "❌"

        summary_table.add_row("Result", f"{result_icon} {result}")

        # Create detailed votes table
        votes_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")

        votes_table.add_column("Agent", width=20)
        votes_table.add_column("Vote", justify="center", width=8)
        votes_table.add_column("Justification", style="dim")

        # Sort votes by result
        sorted_votes = sorted(
            votes.items(), key=lambda x: (x[1].get("vote", "").lower() != "yes", x[0])
        )

        for agent_id, vote_data in sorted_votes:
            vote = vote_data.get("vote", "No response")
            justification = vote_data.get("justification", "No justification provided")

            # Style vote
            if vote.lower() == "yes":
                vote_styled = "[green]✓ Yes[/green]"
            elif vote.lower() == "no":
                vote_styled = "[red]✗ No[/red]"
            else:
                vote_styled = f"[yellow]{vote}[/yellow]"

            # Truncate long justifications
            if len(justification) > 60:
                justification = justification[:57] + "..."

            votes_table.add_row(agent_id, vote_styled, justification)

        # Combine tables
        content = Group(
            votes_table,
            Text(""),
            Panel(summary_table, title="[bold]Summary[/bold]", border_style="cyan"),
        )

        return Panel(
            content,
            title="[bold]Voting Results[/bold]",
            border_style="magenta",
            padding=(1, 2),
        )

    def format_agent_response(
        self,
        agent_id: str,
        provider: ProviderType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Panel:
        """Format agent response with metadata."""
        colors = self.theme.assign_agent_color(agent_id, provider)

        # Process content for special formatting
        formatted_content = self._process_agent_content(content)

        # Build header
        header_parts = [
            f"[{colors['primary']}]{colors['symbol']} {agent_id}[/{colors['primary']}]"
        ]

        if metadata:
            if metadata.get("timestamp"):
                timestamp = metadata["timestamp"]
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%H:%M:%S")
                else:
                    time_str = str(timestamp)
                header_parts.append(f"[dim]{time_str}[/dim]")

            if metadata.get("round"):
                header_parts.append(
                    f"[{colors['accent']}]Round {metadata['round']}[/{colors['accent']}]"
                )

            if metadata.get("response_time"):
                header_parts.append(f"[dim]{metadata['response_time']:.1f}s[/dim]")

        header = " • ".join(header_parts)

        return Panel(
            formatted_content,
            title=header,
            border_style=colors["border"],
            padding=(1, 2),
            width=self.options.width,
        )

    def _process_agent_content(self, content: str) -> Union[str, Group]:
        """Process agent content for special formatting (code blocks, lists, etc.)."""
        # Look for markdown-style code blocks
        code_block_pattern = r"```(\w+)?\n(.*?)\n```"
        code_blocks = re.findall(code_block_pattern, content, re.DOTALL)

        if code_blocks:
            # Split content around code blocks
            parts = re.split(code_block_pattern, content, flags=re.DOTALL)
            formatted_parts = []

            i = 0
            while i < len(parts):
                if i % 3 == 0:  # Text part
                    if parts[i].strip():
                        formatted_parts.append(Text(parts[i].strip()))
                elif i % 3 == 2:  # Code part
                    language = parts[i - 1] or "text"
                    code = parts[i]

                    try:
                        syntax = Syntax(
                            code, language, theme="monokai", background_color="default"
                        )
                        formatted_parts.append(syntax)
                    except:
                        formatted_parts.append(Text(code))

                i += 1

            return (
                Group(*formatted_parts)
                if len(formatted_parts) > 1
                else formatted_parts[0]
            )
        else:
            return content

    def export_to_format(self, content: Any, format_type: ExportFormat) -> str:
        """Export content to specified format."""
        try:
            if format_type == ExportFormat.MARKDOWN:
                return self._export_to_markdown(content)
            elif format_type == ExportFormat.HTML:
                return self._export_to_html(content)
            elif format_type == ExportFormat.PLAIN_TEXT:
                return self._export_to_plain_text(content)
            elif format_type == ExportFormat.JSON:
                return self._export_to_json(content)
            elif format_type == ExportFormat.CSV:
                return self._export_to_csv(content)
            else:
                return str(content)

        except Exception as e:
            logger.error(f"Export error: {e}")
            return str(content)

    def _export_to_markdown(self, content: Any) -> str:
        """Export content to markdown format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            lines = []
            for key, value in content.items():
                lines.append(f"## {key}\n")
                lines.append(f"{value}\n")
            return "\n".join(lines)
        elif isinstance(content, list):
            lines = []
            for i, item in enumerate(content, 1):
                lines.append(f"{i}. {item}")
            return "\n".join(lines)
        else:
            return str(content)

    def _export_to_html(self, content: Any) -> str:
        """Export content to HTML format."""
        # Basic HTML conversion
        if isinstance(content, str):
            return f"<p>{content}</p>"
        else:
            return f"<pre>{str(content)}</pre>"

    def _export_to_plain_text(self, content: Any) -> str:
        """Export content to plain text format."""
        # Remove all formatting
        if isinstance(content, str):
            # Remove rich markup
            import re

            return re.sub(r"\[.*?\]", "", content)
        else:
            return str(content)

    def _export_to_json(self, content: Any) -> str:
        """Export content to JSON format."""
        try:
            return json.dumps(content, indent=2, ensure_ascii=False, default=str)
        except:
            return json.dumps({"content": str(content)}, indent=2)

    def _export_to_csv(self, content: Any) -> str:
        """Export content to CSV format."""
        if isinstance(content, list) and content and isinstance(content[0], dict):
            import csv
            import io

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=content[0].keys())
            writer.writeheader()
            writer.writerows(content)
            return output.getvalue()
        else:
            return str(content)


# Global formatter instance
_formatter: Optional[VirtualAgoraFormatter] = None


def get_formatter() -> VirtualAgoraFormatter:
    """Get the global formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = VirtualAgoraFormatter()
    return _formatter


# Convenience functions


def format_content(content: Any, format_type: FormatType, **kwargs) -> Panel:
    """Format content with specified type."""
    formatter = get_formatter()

    if format_type == FormatType.MARKDOWN:
        return formatter.format_markdown(str(content), **kwargs)
    elif format_type == FormatType.JSON:
        return formatter.format_json(content, **kwargs)
    elif format_type == FormatType.TABLE:
        return formatter.format_table(content, **kwargs)
    elif format_type == FormatType.CODE:
        return formatter.format_code(str(content), **kwargs)
    elif format_type == FormatType.LIST:
        return formatter.format_list(content, **kwargs)
    elif format_type == FormatType.TREE:
        return formatter.format_tree(content, **kwargs)
    else:
        return formatter.format_plain_text(str(content), **kwargs)


def export_content(content: Any, format_type: ExportFormat) -> str:
    """Export content to specified format."""
    return get_formatter().export_to_format(content, format_type)


def format_agent_response(
    agent_id: str, provider: ProviderType, content: str, **kwargs
) -> Panel:
    """Format agent response with metadata."""
    return get_formatter().format_agent_response(agent_id, provider, content, **kwargs)


def format_voting_summary(votes: Dict[str, Dict[str, Any]]) -> Panel:
    """Format voting results summary."""
    return get_formatter().format_voting_summary(votes)
