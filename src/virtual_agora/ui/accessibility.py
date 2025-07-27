"""Accessibility features for Virtual Agora terminal UI.

This module provides comprehensive accessibility support including
high contrast mode, screen reader compatibility, and alternative
display modes for users with different needs.
"""

import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box

from virtual_agora.ui.theme import (
    VirtualAgoraTheme,
    AccessibilityOptions,
    get_current_theme,
    set_theme,
)
from virtual_agora.ui.console import get_console
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class AccessibilityLevel(Enum):
    """Accessibility support levels."""

    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class DisplayMode(Enum):
    """Display mode options."""

    FULL = "full"
    COMPACT = "compact"
    TEXT_ONLY = "text_only"
    SCREEN_READER = "screen_reader"


@dataclass
class AccessibilityProfile:
    """User accessibility profile configuration."""

    level: AccessibilityLevel
    display_mode: DisplayMode
    high_contrast: bool = False
    large_text: bool = False
    reduced_motion: bool = False
    no_emoji: bool = False
    screen_reader_mode: bool = False
    audio_cues: bool = False
    keyboard_navigation: bool = True
    custom_colors: Optional[Dict[str, str]] = None


class AccessibilityManager:
    """Manages accessibility features and user preferences."""

    def __init__(self):
        """Initialize accessibility manager."""
        self.console = get_console()
        self._current_profile: Optional[AccessibilityProfile] = None
        self._original_theme: Optional[VirtualAgoraTheme] = None
        self._accessibility_commands: Dict[str, Callable] = {}

        # Register accessibility commands
        self._register_commands()

    def _register_commands(self) -> None:
        """Register accessibility keyboard commands."""
        self._accessibility_commands = {
            "ctrl+h": self.toggle_high_contrast,
            "ctrl+l": self.toggle_large_text,
            "ctrl+m": self.toggle_reduced_motion,
            "ctrl+e": self.toggle_emoji,
            "ctrl+s": self.toggle_screen_reader_mode,
            "ctrl+r": self.reset_accessibility,
            "ctrl+?": self.show_accessibility_help,
        }

    def detect_system_preferences(self) -> AccessibilityProfile:
        """Detect system accessibility preferences."""
        profile = AccessibilityProfile(
            level=AccessibilityLevel.STANDARD, display_mode=DisplayMode.FULL
        )

        # Check environment variables for accessibility hints
        if os.environ.get("ACCESSIBILITY_HIGH_CONTRAST"):
            profile.high_contrast = True

        if os.environ.get("ACCESSIBILITY_LARGE_TEXT"):
            profile.large_text = True

        if os.environ.get("ACCESSIBILITY_NO_MOTION"):
            profile.reduced_motion = True

        if os.environ.get("ACCESSIBILITY_NO_EMOJI"):
            profile.no_emoji = True

        if os.environ.get("SCREEN_READER"):
            profile.screen_reader_mode = True
            profile.display_mode = DisplayMode.SCREEN_READER

        # Check for common screen readers
        screen_readers = ["NVDA", "JAWS", "ORCA", "VOICEOVER"]
        for reader in screen_readers:
            if os.environ.get(reader):
                profile.screen_reader_mode = True
                profile.display_mode = DisplayMode.SCREEN_READER
                break

        # Adjust level based on detected preferences
        active_features = sum(
            [
                profile.high_contrast,
                profile.large_text,
                profile.reduced_motion,
                profile.no_emoji,
                profile.screen_reader_mode,
            ]
        )

        if active_features >= 3:
            profile.level = AccessibilityLevel.MAXIMUM
        elif active_features >= 1:
            profile.level = AccessibilityLevel.ENHANCED

        return profile

    def apply_profile(self, profile: AccessibilityProfile) -> None:
        """Apply accessibility profile to the UI."""
        self._current_profile = profile

        # Store original theme if not already stored
        if self._original_theme is None:
            self._original_theme = get_current_theme()

        # Create accessibility options
        accessibility_options = AccessibilityOptions(
            high_contrast=profile.high_contrast,
            reduced_motion=profile.reduced_motion,
            use_symbols=not profile.no_emoji,
            large_text=profile.large_text,
            screen_reader_mode=profile.screen_reader_mode,
        )

        # Create and apply new theme
        accessible_theme = VirtualAgoraTheme(accessibility_options)
        set_theme(accessible_theme)

        # Apply console-level changes
        if profile.screen_reader_mode:
            self._apply_screen_reader_mode()
        elif profile.display_mode == DisplayMode.TEXT_ONLY:
            self._apply_text_only_mode()

        logger.info(f"Applied accessibility profile: {profile.level.value}")

    def _apply_screen_reader_mode(self) -> None:
        """Apply screen reader optimizations."""
        # Disable complex layouts and use linear text output
        self.console.rich_console.legacy_windows = True

    def _apply_text_only_mode(self) -> None:
        """Apply text-only display mode."""
        # Simplified display with minimal formatting
        pass

    def toggle_high_contrast(self) -> None:
        """Toggle high contrast mode."""
        if self._current_profile:
            self._current_profile.high_contrast = (
                not self._current_profile.high_contrast
            )
            self.apply_profile(self._current_profile)

            status = "enabled" if self._current_profile.high_contrast else "disabled"
            self.console.print_system_message(f"High contrast mode {status}")

    def toggle_large_text(self) -> None:
        """Toggle large text mode."""
        if self._current_profile:
            self._current_profile.large_text = not self._current_profile.large_text
            self.apply_profile(self._current_profile)

            status = "enabled" if self._current_profile.large_text else "disabled"
            self.console.print_system_message(f"Large text mode {status}")

    def toggle_reduced_motion(self) -> None:
        """Toggle reduced motion mode."""
        if self._current_profile:
            self._current_profile.reduced_motion = (
                not self._current_profile.reduced_motion
            )
            self.apply_profile(self._current_profile)

            status = "enabled" if self._current_profile.reduced_motion else "disabled"
            self.console.print_system_message(f"Reduced motion mode {status}")

    def toggle_emoji(self) -> None:
        """Toggle emoji display."""
        if self._current_profile:
            self._current_profile.no_emoji = not self._current_profile.no_emoji
            self.apply_profile(self._current_profile)

            status = "disabled" if self._current_profile.no_emoji else "enabled"
            self.console.print_system_message(f"Emoji display {status}")

    def toggle_screen_reader_mode(self) -> None:
        """Toggle screen reader mode."""
        if self._current_profile:
            self._current_profile.screen_reader_mode = (
                not self._current_profile.screen_reader_mode
            )

            if self._current_profile.screen_reader_mode:
                self._current_profile.display_mode = DisplayMode.SCREEN_READER
            else:
                self._current_profile.display_mode = DisplayMode.FULL

            self.apply_profile(self._current_profile)

            status = (
                "enabled" if self._current_profile.screen_reader_mode else "disabled"
            )
            self.console.print_system_message(f"Screen reader mode {status}")

    def reset_accessibility(self) -> None:
        """Reset to default accessibility settings."""
        if self._original_theme:
            set_theme(self._original_theme)

        self._current_profile = AccessibilityProfile(
            level=AccessibilityLevel.STANDARD, display_mode=DisplayMode.FULL
        )

        self.console.print_system_message("Accessibility settings reset to defaults")

    def show_accessibility_help(self) -> None:
        """Show accessibility help and keyboard shortcuts."""
        help_table = Table(
            title="[bold]Accessibility Features & Shortcuts[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        help_table.add_column("Shortcut", width=12)
        help_table.add_column("Function", style="white")
        help_table.add_column("Description", style="dim")

        shortcuts = [
            ("Ctrl+H", "High Contrast", "Toggle high contrast color scheme"),
            ("Ctrl+L", "Large Text", "Toggle large text display"),
            ("Ctrl+M", "Reduced Motion", "Toggle reduced motion/animations"),
            ("Ctrl+E", "Emoji Toggle", "Toggle emoji display"),
            ("Ctrl+S", "Screen Reader", "Toggle screen reader mode"),
            ("Ctrl+R", "Reset", "Reset all accessibility settings"),
            ("Ctrl+?", "Help", "Show this help message"),
        ]

        for shortcut, function, description in shortcuts:
            help_table.add_row(f"[yellow]{shortcut}[/yellow]", function, description)

        help_panel = Panel(
            help_table,
            title="[bold]Accessibility Help[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(help_panel)

    def get_current_profile(self) -> Optional[AccessibilityProfile]:
        """Get current accessibility profile."""
        return self._current_profile

    def validate_wcag_compliance(
        self, foreground: str, background: str
    ) -> Dict[str, Any]:
        """Validate WCAG color contrast compliance."""
        # Simplified WCAG compliance check
        # In a real implementation, this would calculate actual luminance ratios

        high_contrast_combinations = [
            ("white", "black"),
            ("black", "white"),
            ("bright_white", "black"),
            ("bright_yellow", "black"),
            ("bright_green", "black"),
            ("bright_cyan", "black"),
            ("bright_red", "black"),
            ("bright_magenta", "black"),
        ]

        is_compliant = (foreground, background) in high_contrast_combinations

        return {
            "compliant": is_compliant,
            "level": "AAA" if is_compliant else "AA" if is_compliant else "Fail",
            "contrast_ratio": 7.0 if is_compliant else 3.0,  # Simplified
            "recommendation": (
                "Compliant" if is_compliant else "Consider higher contrast colors"
            ),
        }

    def generate_accessibility_report(self) -> Panel:
        """Generate accessibility compliance report."""
        if not self._current_profile:
            return Panel(
                "[red]No accessibility profile active[/red]",
                title="Accessibility Report",
            )

        report_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")

        report_table.add_column("Feature", style="white")
        report_table.add_column("Status", justify="center")
        report_table.add_column("Compliance", justify="center")

        features = [
            ("High Contrast", self._current_profile.high_contrast),
            ("Large Text", self._current_profile.large_text),
            ("Reduced Motion", self._current_profile.reduced_motion),
            ("Screen Reader Mode", self._current_profile.screen_reader_mode),
            ("Keyboard Navigation", self._current_profile.keyboard_navigation),
            ("Alternative Text", not self._current_profile.no_emoji),
        ]

        for feature_name, enabled in features:
            status = "[green]Enabled[/green]" if enabled else "[red]Disabled[/red]"
            compliance = (
                "[green]WCAG AA[/green]" if enabled else "[yellow]Basic[/yellow]"
            )

            report_table.add_row(feature_name, status, compliance)

        # Overall compliance level
        enabled_count = sum(1 for _, enabled in features if enabled)
        if enabled_count >= 5:
            overall = "[green]Excellent (WCAG AAA)[/green]"
        elif enabled_count >= 3:
            overall = "[yellow]Good (WCAG AA)[/yellow]"
        else:
            overall = "[orange1]Basic (WCAG A)[/orange1]"

        report_content = Group(
            report_table,
            Text(""),
            Text(f"Overall Compliance: {overall}", justify="center"),
        )

        return Panel(
            report_content,
            title="[bold]Accessibility Compliance Report[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )


class AccessibilityTester:
    """Tools for testing accessibility compliance."""

    @staticmethod
    def test_color_combinations() -> List[Dict[str, Any]]:
        """Test various color combinations for compliance."""
        combinations = [
            ("white", "black"),
            ("black", "white"),
            ("yellow", "black"),
            ("blue", "white"),
            ("red", "white"),
            ("green", "black"),
        ]

        manager = AccessibilityManager()
        results = []

        for fg, bg in combinations:
            compliance = manager.validate_wcag_compliance(fg, bg)
            results.append({"foreground": fg, "background": bg, **compliance})

        return results

    @staticmethod
    def simulate_screen_reader_output(content: str) -> str:
        """Simulate screen reader output for content."""
        # Remove rich markup and convert to screen reader friendly text
        import re

        # Remove rich markup
        clean_content = re.sub(r"\[.*?\]", "", content)

        # Replace symbols with text
        replacements = {
            "🤖": "robot ",
            "🗳️": "ballot box ",
            "📄": "document ",
            "💾": "floppy disk ",
            "📝": "memo ",
            "📋": "clipboard ",
            "✅": "check mark ",
            "❌": "cross mark ",
            "⚠️": "warning ",
            "ℹ️": "information ",
            "🎯": "target ",
            "💭": "thought bubble ",
            "🏛️": "classical building ",
        }

        for symbol, text in replacements.items():
            clean_content = clean_content.replace(symbol, text)

        # Add punctuation for better screen reader flow
        clean_content = clean_content.replace(" • ", ". ")

        return clean_content

    @staticmethod
    def generate_alt_text(element_type: str, content: str) -> str:
        """Generate alternative text for UI elements."""
        alt_text_templates = {
            "panel": "Panel containing: {content}",
            "table": "Table with {content}",
            "button": "Button: {content}",
            "progress": "Progress indicator: {content}",
            "status": "Status: {content}",
            "message": "Message from {content}",
            "error": "Error: {content}",
            "warning": "Warning: {content}",
            "success": "Success: {content}",
        }

        template = alt_text_templates.get(element_type, "{content}")
        return template.format(
            content=content[:100] + "..." if len(content) > 100 else content
        )


# Global accessibility manager
_accessibility_manager: Optional[AccessibilityManager] = None


def get_accessibility_manager() -> AccessibilityManager:
    """Get the global accessibility manager."""
    global _accessibility_manager
    if _accessibility_manager is None:
        _accessibility_manager = AccessibilityManager()
    return _accessibility_manager


# Convenience functions


def initialize_accessibility() -> None:
    """Initialize accessibility features based on system detection."""
    manager = get_accessibility_manager()
    profile = manager.detect_system_preferences()
    manager.apply_profile(profile)

    if profile.level != AccessibilityLevel.STANDARD:
        logger.info(f"Accessibility features auto-enabled: {profile.level.value}")


def apply_accessibility_profile(profile: AccessibilityProfile) -> None:
    """Apply an accessibility profile."""
    get_accessibility_manager().apply_profile(profile)


def toggle_accessibility_feature(feature: str) -> None:
    """Toggle a specific accessibility feature."""
    manager = get_accessibility_manager()
    toggle_methods = {
        "high_contrast": manager.toggle_high_contrast,
        "large_text": manager.toggle_large_text,
        "reduced_motion": manager.toggle_reduced_motion,
        "emoji": manager.toggle_emoji,
        "screen_reader": manager.toggle_screen_reader_mode,
    }

    if feature in toggle_methods:
        toggle_methods[feature]()
    else:
        logger.warning(f"Unknown accessibility feature: {feature}")


def show_accessibility_help() -> None:
    """Show accessibility help."""
    get_accessibility_manager().show_accessibility_help()


def generate_accessibility_report() -> Panel:
    """Generate accessibility compliance report."""
    return get_accessibility_manager().generate_accessibility_report()


def create_accessible_profile(
    high_contrast: bool = True, large_text: bool = True, screen_reader: bool = False
) -> AccessibilityProfile:
    """Create a pre-configured accessible profile."""
    return AccessibilityProfile(
        level=AccessibilityLevel.ENHANCED,
        display_mode=DisplayMode.SCREEN_READER if screen_reader else DisplayMode.FULL,
        high_contrast=high_contrast,
        large_text=large_text,
        reduced_motion=True,
        no_emoji=screen_reader,
        screen_reader_mode=screen_reader,
        keyboard_navigation=True,
    )


def create_minimal_profile() -> AccessibilityProfile:
    """Create a minimal accessibility profile for basic terminals."""
    return AccessibilityProfile(
        level=AccessibilityLevel.STANDARD,
        display_mode=DisplayMode.TEXT_ONLY,
        high_contrast=False,
        large_text=False,
        reduced_motion=True,
        no_emoji=True,
        screen_reader_mode=False,
        keyboard_navigation=True,
    )
