"""Tests for Virtual Agora theme system."""

import pytest
from virtual_agora.ui.theme import (
    VirtualAgoraTheme,
    ProviderType,
    MessageType,
    AccessibilityOptions,
    get_current_theme,
    set_theme,
    reset_theme,
)


class TestAccessibilityOptions:
    """Test accessibility options."""

    def test_default_options(self):
        """Test default accessibility options."""
        options = AccessibilityOptions()

        assert options.high_contrast is False
        assert options.reduced_motion is False
        assert options.use_symbols is True  # Enabled by default for better UX
        assert options.large_text is False
        assert options.screen_reader_mode is False

    def test_custom_options(self):
        """Test custom accessibility options."""
        options = AccessibilityOptions(
            high_contrast=True, reduced_motion=True, use_symbols=True
        )

        assert options.high_contrast is True
        assert options.reduced_motion is True
        assert options.use_symbols is True
        assert options.large_text is False


class TestVirtualAgoraTheme:
    """Test Virtual Agora theme functionality."""

    def test_default_initialization(self):
        """Test default theme initialization."""
        theme = VirtualAgoraTheme()

        assert theme.accessibility is not None
        assert isinstance(theme.accessibility, AccessibilityOptions)
        assert theme._agent_color_assignments == {}

    def test_accessibility_initialization(self):
        """Test theme with accessibility options."""
        accessibility = AccessibilityOptions(high_contrast=True)
        theme = VirtualAgoraTheme(accessibility)

        assert theme.accessibility.high_contrast is True

    def test_get_palette_normal(self):
        """Test getting normal color palette."""
        theme = VirtualAgoraTheme()
        palette = theme.get_palette()

        assert palette.primary == "cyan"
        assert palette.secondary == "magenta"
        assert palette.background == "black"

    def test_get_palette_high_contrast(self):
        """Test getting high contrast palette."""
        accessibility = AccessibilityOptions(high_contrast=True)
        theme = VirtualAgoraTheme(accessibility)
        palette = theme.get_palette()

        assert palette.primary == "bright_cyan"
        assert palette.secondary == "bright_magenta"

    def test_assign_agent_color_new_agent(self):
        """Test assigning colors to new agent."""
        theme = VirtualAgoraTheme()

        colors = theme.assign_agent_color("agent1", ProviderType.OPENAI)

        assert "primary" in colors
        assert "accent" in colors
        assert "symbol" in colors
        assert "border" in colors

        # Should be consistent on repeat calls
        colors2 = theme.assign_agent_color("agent1", ProviderType.OPENAI)
        assert colors == colors2

    def test_assign_agent_color_multiple_agents(self):
        """Test assigning colors to multiple agents of same provider."""
        theme = VirtualAgoraTheme()

        colors1 = theme.assign_agent_color("agent1", ProviderType.OPENAI)
        colors2 = theme.assign_agent_color("agent2", ProviderType.OPENAI)

        # Should have different primary colors (variations)
        assert colors1["primary"] != colors2["primary"]

    def test_assign_agent_color_different_providers(self):
        """Test assigning colors to agents from different providers."""
        theme = VirtualAgoraTheme()

        openai_colors = theme.assign_agent_color("agent1", ProviderType.OPENAI)
        google_colors = theme.assign_agent_color("agent2", ProviderType.GOOGLE)

        # Should have different base colors
        assert openai_colors["symbol"] != google_colors["symbol"]

    def test_get_message_style_normal(self):
        """Test getting message style in normal mode."""
        theme = VirtualAgoraTheme()

        style = theme.get_message_style(MessageType.ERROR)

        assert style["color"] == "bright_red"
        assert style["bold"] is True
        assert style["icon"] == "❌"

    def test_get_message_style_high_contrast(self):
        """Test getting message style in high contrast mode."""
        accessibility = AccessibilityOptions(high_contrast=True)
        theme = VirtualAgoraTheme(accessibility)

        style = theme.get_message_style(MessageType.SUCCESS)

        # Should enhance contrast
        assert "bright_" in style["color"]

    def test_get_message_style_no_symbols(self):
        """Test getting message style without symbols."""
        accessibility = AccessibilityOptions(use_symbols=False)
        theme = VirtualAgoraTheme(accessibility)

        style = theme.get_message_style(MessageType.WARNING)

        assert style["icon"] == ""

    def test_create_rich_style(self):
        """Test creating Rich style objects."""
        theme = VirtualAgoraTheme()

        style = theme.create_rich_style("red", bold=True, italic=True)

        assert style.color.name == "red"
        assert style.bold is True
        assert style.italic is True

    def test_create_rich_style_high_contrast(self):
        """Test creating Rich style with high contrast."""
        accessibility = AccessibilityOptions(high_contrast=True)
        theme = VirtualAgoraTheme(accessibility)

        style = theme.create_rich_style("blue", bold=False)

        # Should enhance for high contrast
        assert style.bold is True

    def test_get_progress_style(self):
        """Test getting progress indicator styling."""
        theme = VirtualAgoraTheme()

        style = theme.get_progress_style()

        assert "bar_color" in style
        assert "complete_color" in style
        assert "text_color" in style

    def test_get_dashboard_colors(self):
        """Test getting dashboard color scheme."""
        theme = VirtualAgoraTheme()

        colors = theme.get_dashboard_colors()

        assert "header" in colors
        assert "border" in colors
        assert "label" in colors
        assert "value" in colors
        assert "background" in colors

    def test_apply_accessibility_override(self):
        """Test applying accessibility overrides."""
        theme = VirtualAgoraTheme()

        style_dict = {"color": "red", "bold": False, "icon": "❌"}

        result = theme.apply_accessibility_override(style_dict)

        # Should return modified dict for normal mode
        assert result == style_dict

    def test_apply_accessibility_override_screen_reader(self):
        """Test accessibility override for screen reader mode."""
        accessibility = AccessibilityOptions(screen_reader_mode=True)
        theme = VirtualAgoraTheme(accessibility)

        style_dict = {"color": "red", "bold": False, "icon": "❌"}

        result = theme.apply_accessibility_override(style_dict)

        # Should simplify for screen readers
        assert result["color"] == "white"
        assert result["icon"] == ""

    def test_get_agent_identifier_with_symbols(self):
        """Test getting agent identifier with symbols."""
        accessibility = AccessibilityOptions(use_symbols=True)
        theme = VirtualAgoraTheme(accessibility)

        identifier = theme.get_agent_identifier("agent1", ProviderType.GOOGLE)

        # Should include symbol
        assert "🟢" in identifier
        assert "agent1" in identifier

    def test_get_agent_identifier_without_symbols(self):
        """Test getting agent identifier without symbols."""
        accessibility = AccessibilityOptions(use_symbols=False)
        theme = VirtualAgoraTheme(accessibility)

        identifier = theme.get_agent_identifier("agent1", ProviderType.GOOGLE)

        # Should not include symbol
        assert identifier == "agent1"

    def test_validate_color_contrast(self):
        """Test color contrast validation."""
        theme = VirtualAgoraTheme()

        # Test high contrast combination
        assert theme.validate_color_contrast("white", "black") is True

        # Test low contrast combination
        assert theme.validate_color_contrast("gray", "light_gray") is False

    def test_create_accessible_theme(self):
        """Test creating accessible theme."""
        theme = VirtualAgoraTheme.create_accessible_theme()

        assert theme.accessibility.high_contrast is True
        assert theme.accessibility.use_symbols is True
        assert theme.accessibility.large_text is True

    def test_create_minimal_theme(self):
        """Test creating minimal theme."""
        theme = VirtualAgoraTheme.create_minimal_theme()

        assert theme.accessibility.reduced_motion is True
        assert theme.accessibility.use_symbols is False


class TestProviderType:
    """Test provider type enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.GOOGLE.value == "google"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.GROK.value == "grok"
        assert ProviderType.MODERATOR.value == "moderator"

    def test_provider_creation(self):
        """Test creating provider from string."""
        provider = ProviderType("openai")
        assert provider == ProviderType.OPENAI


class TestMessageType:
    """Test message type enum."""

    def test_message_values(self):
        """Test message type enum values."""
        assert MessageType.USER_PROMPT.value == "user_prompt"
        assert MessageType.SYSTEM_INFO.value == "system_info"
        assert MessageType.ERROR.value == "error"
        assert MessageType.WARNING.value == "warning"
        assert MessageType.SUCCESS.value == "success"


class TestGlobalTheme:
    """Test global theme management."""

    def test_get_current_theme(self):
        """Test getting current theme."""
        theme = get_current_theme()

        assert isinstance(theme, VirtualAgoraTheme)

    def test_set_theme(self):
        """Test setting global theme."""
        new_theme = VirtualAgoraTheme(AccessibilityOptions(high_contrast=True))
        set_theme(new_theme)

        current = get_current_theme()
        assert current is new_theme
        assert current.accessibility.high_contrast is True

    def test_reset_theme(self):
        """Test resetting theme to default."""
        # Set custom theme first
        custom_theme = VirtualAgoraTheme(AccessibilityOptions(high_contrast=True))
        set_theme(custom_theme)

        # Reset to default
        reset_theme()

        current = get_current_theme()
        assert current.accessibility.high_contrast is False


class TestThemeIntegration:
    """Test theme integration scenarios."""

    def test_multiple_agents_color_assignment(self):
        """Test color assignment for multiple agents."""
        theme = VirtualAgoraTheme()

        # Assign colors to multiple OpenAI agents
        agent1_colors = theme.assign_agent_color("openai-1", ProviderType.OPENAI)
        agent2_colors = theme.assign_agent_color("openai-2", ProviderType.OPENAI)
        agent3_colors = theme.assign_agent_color("openai-3", ProviderType.OPENAI)

        # Should have different variations but same provider base
        assert (
            agent1_colors["symbol"]
            == agent2_colors["symbol"]
            == agent3_colors["symbol"]
        )

        # Primary colors should cycle through variations
        primary_colors = [
            agent1_colors["primary"],
            agent2_colors["primary"],
            agent3_colors["primary"],
        ]

        # Should have at least some variation
        assert len(set(primary_colors)) > 1

    def test_accessibility_impact_on_colors(self):
        """Test how accessibility settings impact color assignment."""
        normal_theme = VirtualAgoraTheme()
        accessible_theme = VirtualAgoraTheme(AccessibilityOptions(high_contrast=True))

        normal_colors = normal_theme.assign_agent_color(
            "agent1", ProviderType.ANTHROPIC
        )
        accessible_colors = accessible_theme.assign_agent_color(
            "agent1", ProviderType.ANTHROPIC
        )

        # Accessible theme should enhance colors
        assert (
            "bright_" in accessible_colors["primary"]
            or accessible_colors["primary"] != normal_colors["primary"]
        )

    def test_theme_consistency(self):
        """Test theme consistency across calls."""
        theme = VirtualAgoraTheme()

        # Multiple calls should return consistent results
        style1 = theme.get_message_style(MessageType.ERROR)
        style2 = theme.get_message_style(MessageType.ERROR)

        assert style1 == style2

        colors1 = theme.assign_agent_color("test-agent", ProviderType.GOOGLE)
        colors2 = theme.assign_agent_color("test-agent", ProviderType.GOOGLE)

        assert colors1 == colors2


if __name__ == "__main__":
    pytest.main([__file__])
