"""Error reporting utilities for Virtual Agora.

This module provides user-friendly error reporting, error aggregation,
and session error summary generation.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.error_handler import ErrorContext, ErrorSeverity, RecoveryStrategy
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ProviderError,
    ConfigurationError,
    StateError,
    ValidationError,
    RecoverableError,
    CriticalError,
)


logger = get_logger(__name__)


class ErrorReporter:
    """Formats and reports errors to users."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize error reporter.
        
        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.reported_errors: List[ErrorContext] = []
    
    def report_error(
        self,
        context: ErrorContext,
        show_details: bool = False,
        show_suggestions: bool = True,
    ) -> None:
        """Report an error to the user.
        
        Args:
            context: Error context
            show_details: Whether to show technical details
            show_suggestions: Whether to show recovery suggestions
        """
        # Track error pattern
        pattern = self._get_error_pattern(context)
        self.error_patterns[pattern] += 1
        self.reported_errors.append(context)
        
        # Format message
        message = self._format_error_message(context, show_details)
        
        # Determine panel style based on severity
        styles = {
            ErrorSeverity.LOW: ("dim yellow", "⚠️  Warning"),
            ErrorSeverity.MEDIUM: ("yellow", "⚠️  Warning"),
            ErrorSeverity.HIGH: ("red", "❌ Error"),
            ErrorSeverity.CRITICAL: ("bold red", "🚨 Critical Error"),
        }
        
        style, title = styles.get(
            context.severity,
            ("red", "❌ Error")
        )
        
        # Create panel
        panel = Panel(
            message,
            title=title,
            border_style=style,
            expand=False,
        )
        
        self.console.print(panel)
        
        # Show suggestions if requested
        if show_suggestions:
            suggestions = self._get_recovery_suggestions(context)
            if suggestions:
                self.console.print(
                    Text("💡 Suggestions:", style="cyan"),
                    style="dim",
                )
                for suggestion in suggestions:
                    self.console.print(f"  • {suggestion}", style="dim")
    
    def _format_error_message(
        self,
        context: ErrorContext,
        show_details: bool
    ) -> str:
        """Format error message for display.
        
        Args:
            context: Error context
            show_details: Whether to include technical details
            
        Returns:
            Formatted message
        """
        parts = []
        
        # Main message
        error_type = type(context.error).__name__.replace("Error", "")
        
        if isinstance(context.error, ProviderError):
            provider = context.error.provider or "Unknown"
            parts.append(f"[bold]{provider} Provider Issue[/bold]")
        elif isinstance(context.error, ConfigurationError):
            parts.append("[bold]Configuration Problem[/bold]")
        elif isinstance(context.error, StateError):
            parts.append("[bold]Application State Issue[/bold]")
        elif isinstance(context.error, ValidationError):
            parts.append("[bold]Invalid Input[/bold]")
        else:
            parts.append(f"[bold]{error_type} Error[/bold]")
        
        # Error description
        error_message = str(context.error)
        if error_message:
            parts.append(f"\n{error_message}")
        
        # Context information
        if context.operation:
            parts.append(f"\n[dim]During: {context.operation}[/dim]")
        
        if context.phase:
            parts.append(f"[dim]Phase: {context.phase}[/dim]")
        
        # Technical details
        if show_details:
            if context.breadcrumbs:
                parts.append("\n[dim]Recent operations:[/dim]")
                for crumb in context.breadcrumbs[-3:]:
                    parts.append(f"[dim]  • {crumb}[/dim]")
            
            if context.metadata:
                parts.append("\n[dim]Additional details:[/dim]")
                for key, value in context.metadata.items():
                    parts.append(f"[dim]  • {key}: {value}[/dim]")
        
        # Retry information
        if context.retry_count > 0:
            parts.append(
                f"\n[dim]Retry attempt: {context.retry_count}/{context.max_retries}[/dim]"
            )
        
        return "\n".join(parts)
    
    def _get_recovery_suggestions(self, context: ErrorContext) -> List[str]:
        """Get recovery suggestions for an error.
        
        Args:
            context: Error context
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # General suggestions based on error type
        if isinstance(context.error, ConfigurationError):
            suggestions.extend([
                "Check your configuration file (config.yml) for errors",
                "Ensure all required fields are present",
                "Verify that agent models and providers are valid",
            ])
        
        elif isinstance(context.error, ProviderError):
            provider = context.error.provider
            if provider:
                suggestions.append(f"Check your {provider} API key in the .env file")
            
            if "rate" in str(context.error).lower():
                suggestions.extend([
                    "Wait a moment before retrying",
                    "Consider using a different provider temporarily",
                    "Check your API usage limits",
                ])
            elif "auth" in str(context.error).lower():
                suggestions.append("Verify your API credentials are correct")
        
        elif isinstance(context.error, ValidationError):
            if context.error.field:
                suggestions.append(
                    f"Check the value provided for '{context.error.field}'"
                )
            suggestions.append("Review the expected format and try again")
        
        elif isinstance(context.error, StateError):
            suggestions.extend([
                "The application state may be corrupted",
                "Try restarting the application",
                "Check the logs for more details",
            ])
        
        # Recovery strategy suggestions
        strategy_suggestions = {
            RecoveryStrategy.RETRY: "The operation will be retried automatically",
            RecoveryStrategy.SKIP: "This step can be skipped safely",
            RecoveryStrategy.ROLLBACK: "The system will revert to a previous state",
            RecoveryStrategy.USER_INTERVENTION: "Please provide the required information",
            RecoveryStrategy.GRACEFUL_SHUTDOWN: "Save your work and restart the application",
        }
        
        if context.recovery_strategy in strategy_suggestions:
            suggestions.insert(0, strategy_suggestions[context.recovery_strategy])
        
        return suggestions
    
    def _get_error_pattern(self, context: ErrorContext) -> str:
        """Get error pattern for tracking.
        
        Args:
            context: Error context
            
        Returns:
            Error pattern string
        """
        error_type = type(context.error).__name__
        operation = context.operation or "unknown"
        
        # Special handling for provider errors
        if isinstance(context.error, ProviderError):
            provider = context.error.provider or "unknown"
            return f"{error_type}:{provider}:{operation}"
        
        return f"{error_type}:{operation}"
    
    def show_error_summary(self) -> None:
        """Display a summary of all errors in the session."""
        if not self.reported_errors:
            self.console.print(
                "[green]✅ No errors occurred during this session[/green]"
            )
            return
        
        # Create summary table
        table = Table(
            title="Session Error Summary",
            show_header=True,
            header_style="bold cyan",
        )
        
        table.add_column("Error Type", style="red")
        table.add_column("Count", justify="right")
        table.add_column("Severity", justify="center")
        table.add_column("Last Occurrence")
        
        # Group errors by type
        error_groups: Dict[str, List[ErrorContext]] = defaultdict(list)
        for context in self.reported_errors:
            error_type = type(context.error).__name__
            error_groups[error_type].append(context)
        
        # Add rows to table
        for error_type, contexts in sorted(error_groups.items()):
            count = len(contexts)
            
            # Get most severe instance
            most_severe = max(contexts, key=lambda c: c.severity.value)
            severity_text = self._format_severity(most_severe.severity)
            
            # Get last occurrence
            last_occurrence = max(contexts, key=lambda c: c.timestamp)
            time_str = last_occurrence.timestamp.strftime("%H:%M:%S")
            
            table.add_row(
                error_type,
                str(count),
                severity_text,
                time_str,
            )
        
        self.console.print(table)
        
        # Show pattern analysis
        if len(self.error_patterns) > 1:
            self.console.print("\n[bold]Error Patterns:[/bold]")
            
            sorted_patterns = sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            
            for pattern, count in sorted_patterns[:5]:
                parts = pattern.split(":")
                if len(parts) >= 2:
                    error_type, operation = parts[0], parts[1]
                    if count > 1:
                        self.console.print(
                            f"  • {error_type} during {operation}: "
                            f"[yellow]{count} times[/yellow]"
                        )
    
    def _format_severity(self, severity: ErrorSeverity) -> str:
        """Format severity for display.
        
        Args:
            severity: Error severity
            
        Returns:
            Formatted severity text
        """
        formats = {
            ErrorSeverity.LOW: "[dim yellow]Low[/dim yellow]",
            ErrorSeverity.MEDIUM: "[yellow]Medium[/yellow]",
            ErrorSeverity.HIGH: "[red]High[/red]",
            ErrorSeverity.CRITICAL: "[bold red]Critical[/bold red]",
        }
        
        return formats.get(severity, "[red]Unknown[/red]")
    
    def save_error_report(
        self,
        session_id: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save detailed error report to file.
        
        Args:
            session_id: Session identifier
            output_dir: Output directory
            
        Returns:
            Path to saved report
        """
        if not output_dir:
            output_dir = Path("logs")
        
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_report_{session_id}_{timestamp}.json"
        filepath = output_dir / filename
        
        # Prepare report data
        report = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(self.reported_errors),
            "error_patterns": dict(self.error_patterns),
            "errors": [],
        }
        
        # Add error details
        for context in self.reported_errors:
            error_data = {
                "timestamp": context.timestamp.isoformat(),
                "error_type": type(context.error).__name__,
                "message": str(context.error),
                "severity": context.severity.value,
                "operation": context.operation,
                "phase": context.phase,
                "recovery_strategy": context.recovery_strategy.value,
                "retry_count": context.retry_count,
                "breadcrumbs": context.breadcrumbs[-10:],  # Last 10
                "metadata": context.metadata,
            }
            
            # Add error-specific fields
            if isinstance(context.error, ProviderError):
                error_data["provider"] = context.error.provider
                error_data["error_code"] = context.error.error_code
            
            report["errors"].append(error_data)
        
        # Save report
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved error report to {filepath}")
        return filepath
    
    def detect_error_trends(self) -> List[str]:
        """Detect trends in error patterns.
        
        Returns:
            List of detected trends
        """
        trends = []
        
        if not self.reported_errors:
            return trends
        
        # Time-based analysis
        if len(self.reported_errors) >= 5:
            # Check if errors are increasing
            recent_errors = self.reported_errors[-5:]
            time_diffs = []
            
            for i in range(1, len(recent_errors)):
                diff = (
                    recent_errors[i].timestamp - recent_errors[i-1].timestamp
                ).total_seconds()
                time_diffs.append(diff)
            
            if all(diff < prev for diff, prev in zip(time_diffs[1:], time_diffs)):
                trends.append("Error frequency is increasing")
        
        # Pattern analysis
        if self.error_patterns:
            # Find most common pattern
            most_common = max(self.error_patterns.items(), key=lambda x: x[1])
            if most_common[1] >= 3:
                pattern_parts = most_common[0].split(":")
                if len(pattern_parts) >= 2:
                    trends.append(
                        f"Repeated {pattern_parts[0]} errors during {pattern_parts[1]}"
                    )
        
        # Provider-specific trends
        provider_errors = defaultdict(int)
        for context in self.reported_errors:
            if isinstance(context.error, ProviderError) and context.error.provider:
                provider_errors[context.error.provider] += 1
        
        for provider, count in provider_errors.items():
            if count >= 3:
                trends.append(f"Multiple errors from {provider} provider")
        
        # Severity trends
        critical_count = sum(
            1 for c in self.reported_errors
            if c.severity == ErrorSeverity.CRITICAL
        )
        if critical_count >= 2:
            trends.append("Multiple critical errors detected")
        
        return trends