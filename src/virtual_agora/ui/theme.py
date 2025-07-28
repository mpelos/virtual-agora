"""Advanced theming system for Virtual Agora terminal UI.

This module provides comprehensive color schemes, accessibility support,
and dynamic theme management for the Rich-based terminal interface.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from rich.style import Style
from rich.color import Color

from virtual_agora.providers.config import ProviderType


class MessageType(Enum):
    """Message type classification for styling."""

    USER_PROMPT = "user_prompt"
    SYSTEM_INFO = "system_info"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    AGENT_RESPONSE = "agent_response"
    MODERATOR_MESSAGE = "moderator_message"


@dataclass
class ColorPalette:
    """Color palette definition."""

    primary: str
    secondary: str
    accent: str
    background: str
    text: str
    contrast: str


@dataclass
class AccessibilityOptions:
    """Accessibility configuration options."""

    high_contrast: bool = False
    reduced_motion: bool = False
    use_symbols: bool = True  # Enable symbols by default for better UX
    large_text: bool = False
    screen_reader_mode: bool = False


class VirtualAgoraTheme:
    """Enhanced theme system for Virtual Agora."""

    # Base color palettes
    NORMAL_PALETTE = ColorPalette(
        primary="cyan",
        secondary="magenta",
        accent="yellow",
        background="black",
        text="white",
        contrast="bright_white",
    )

    HIGH_CONTRAST_PALETTE = ColorPalette(
        primary="bright_cyan",
        secondary="bright_magenta",
        accent="bright_yellow",
        background="black",
        text="bright_white",
        contrast="black on white",
    )

    # Provider color assignments
    PROVIDER_COLORS = {
        ProviderType.OPENAI: {
            "base": "blue",
            "light": "bright_blue",
            "dark": "dark_blue",
            "accent": "cyan",
            "symbol": "🔵",
        },
        ProviderType.GOOGLE: {
            "base": "green",
            "light": "bright_green",
            "dark": "dark_green",
            "accent": "lime",
            "symbol": "🟢",
        },
        ProviderType.ANTHROPIC: {
            "base": "orange1",
            "light": "orange3",
            "dark": "dark_orange",
            "accent": "yellow",
            "symbol": "🟠",
        },
        ProviderType.GROK: {
            "base": "purple",
            "light": "bright_magenta",
            "dark": "dark_magenta",
            "accent": "plum1",
            "symbol": "🟣",
        },
        ProviderType.MODERATOR: {
            "base": "white",
            "light": "bright_white",
            "dark": "grey70",
            "accent": "silver",
            "symbol": "⚪",
        },
    }

    # Message type styling
    MESSAGE_STYLES = {
        MessageType.USER_PROMPT: {
            "color": "bright_yellow",
            "bold": True,
            "icon": "💬",
            "border": "yellow",
        },
        MessageType.SYSTEM_INFO: {
            "color": "cyan",
            "bold": False,
            "icon": "ℹ️",
            "border": "cyan",
        },
        MessageType.ERROR: {
            "color": "bright_red",
            "bold": True,
            "icon": "❌",
            "border": "red",
        },
        MessageType.WARNING: {
            "color": "yellow",
            "bold": True,
            "icon": "⚠️",
            "border": "yellow",
        },
        MessageType.SUCCESS: {
            "color": "bright_green",
            "bold": True,
            "icon": "✅",
            "border": "green",
        },
        MessageType.AGENT_RESPONSE: {
            "color": "white",
            "bold": False,
            "icon": "💭",
            "border": "cyan",
        },
        MessageType.MODERATOR_MESSAGE: {
            "color": "bright_white",
            "bold": True,
            "icon": "🎯",
            "border": "magenta",
        },
    }

    def __init__(self, accessibility: Optional[AccessibilityOptions] = None):
        """Initialize theme with accessibility options."""
        self.accessibility = accessibility or AccessibilityOptions()
        self._agent_color_assignments: Dict[str, Dict[str, str]] = {}
        self._color_index_counters: Dict[ProviderType, int] = {
            provider: 0 for provider in ProviderType
        }

    def get_palette(self) -> ColorPalette:
        """Get current color palette based on accessibility settings."""
        if self.accessibility.high_contrast:
            return self.HIGH_CONTRAST_PALETTE
        return self.NORMAL_PALETTE

    def assign_agent_color(
        self, agent_id: str, provider: ProviderType
    ) -> Dict[str, str]:
        """Assign colors to an agent based on provider and instance."""
        if agent_id in self._agent_color_assignments:
            return self._agent_color_assignments[agent_id]

        provider_colors = self.PROVIDER_COLORS[provider]
        counter = self._color_index_counters[provider]

        # Cycle through variations for multiple agents of same provider
        color_variations = ["base", "light", "dark"]
        variation = color_variations[counter % len(color_variations)]

        colors = {
            "primary": provider_colors[variation],
            "accent": provider_colors["accent"],
            "symbol": provider_colors["symbol"],
            "border": provider_colors["base"],
        }

        # Apply accessibility modifications
        if self.accessibility.high_contrast:
            colors["primary"] = (
                f"bright_{colors['primary']}"
                if "bright_" not in colors["primary"]
                else colors["primary"]
            )

        self._agent_color_assignments[agent_id] = colors
        self._color_index_counters[provider] += 1

        return colors

    def get_message_style(self, message_type: MessageType) -> Dict[str, str]:
        """Get styling for a message type."""
        base_style = self.MESSAGE_STYLES[message_type].copy()

        # Apply accessibility modifications
        if self.accessibility.high_contrast:
            if "bright_" not in base_style["color"]:
                base_style["color"] = f"bright_{base_style['color']}"

        if self.accessibility.use_symbols:
            # Keep symbols enabled
            pass
        else:
            # Remove emoji icons for terminal compatibility
            base_style["icon"] = ""

        return base_style

    def create_rich_style(
        self, color: str, bold: bool = False, italic: bool = False
    ) -> Style:
        """Create a Rich Style object with accessibility considerations."""
        style_kwargs = {"color": color}

        if bold:
            style_kwargs["bold"] = True
        if italic:
            style_kwargs["italic"] = True

        # Apply accessibility modifications
        if self.accessibility.high_contrast:
            # Enhance contrast
            if not bold:
                style_kwargs["bold"] = True

        if self.accessibility.reduced_motion:
            # Disable any animation-related styles
            pass

        return Style(**style_kwargs)

    def get_progress_style(self) -> Dict[str, str]:
        """Get progress indicator styling."""
        palette = self.get_palette()
        return {
            "bar_color": palette.primary,
            "complete_color": palette.accent,
            "text_color": palette.text,
        }

    def get_dashboard_colors(self) -> Dict[str, str]:
        """Get color scheme for status dashboard."""
        palette = self.get_palette()
        return {
            "header": palette.primary,
            "border": palette.secondary,
            "label": palette.text,
            "value": palette.accent,
            "background": palette.background,
        }

    def apply_accessibility_override(
        self, style_dict: Dict[str, str]
    ) -> Dict[str, str]:
        """Apply accessibility overrides to any style dictionary."""
        if self.accessibility.screen_reader_mode:
            # Simplified styles for screen readers
            return {"color": "white", "bold": False, "icon": "", "border": "white"}

        if self.accessibility.large_text:
            # Enhance visibility
            style_dict["bold"] = True

        return style_dict

    def get_agent_identifier(self, agent_id: str, provider: ProviderType) -> str:
        """Get a formatted agent identifier with colors and symbols."""
        colors = self.assign_agent_color(agent_id, provider)

        if self.accessibility.use_symbols:
            symbol = colors["symbol"]
            return f"{symbol} {agent_id}"
        else:
            return agent_id

    def validate_color_contrast(self, foreground: str, background: str) -> bool:
        """Validate color contrast for accessibility compliance."""
        # Simplified contrast check - in real implementation would calculate luminance
        high_contrast_pairs = [
            ("white", "black"),
            ("black", "white"),
            ("bright_white", "black"),
            ("bright_yellow", "black"),
            ("bright_red", "black"),
            ("bright_green", "black"),
            ("bright_cyan", "black"),
            ("bright_magenta", "black"),
        ]

        return (foreground, background) in high_contrast_pairs

    @classmethod
    def create_accessible_theme(cls) -> "VirtualAgoraTheme":
        """Create a theme optimized for accessibility."""
        accessibility = AccessibilityOptions(
            high_contrast=True, use_symbols=True, large_text=True
        )
        return cls(accessibility)

    @classmethod
    def create_minimal_theme(cls) -> "VirtualAgoraTheme":
        """Create a minimal theme for low-capability terminals."""
        accessibility = AccessibilityOptions(reduced_motion=True, use_symbols=False)
        return cls(accessibility)


# Global theme instance
_current_theme: Optional[VirtualAgoraTheme] = None


def get_current_theme() -> VirtualAgoraTheme:
    """Get the current global theme instance."""
    global _current_theme
    if _current_theme is None:
        _current_theme = VirtualAgoraTheme()
    return _current_theme


def set_theme(theme: VirtualAgoraTheme) -> None:
    """Set the global theme instance."""
    global _current_theme
    _current_theme = theme


def reset_theme() -> None:
    """Reset to default theme."""
    global _current_theme
    _current_theme = VirtualAgoraTheme()
