"""Tests for Virtual Agora console module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from io import StringIO

from virtual_agora.ui.console import (
    VirtualAgoraConsole,
    ConsoleCapabilities,
    get_console,
    print_agent_message,
    print_system_message,
    print_user_prompt,
)
from virtual_agora.ui.theme import ProviderType, MessageType


class TestConsoleCapabilities:
    """Test console capability detection."""

    def test_width_detection(self):
        """Test terminal width detection."""
        capabilities = ConsoleCapabilities()
        assert isinstance(capabilities.width, int)
        assert capabilities.width > 0

    def test_height_detection(self):
        """Test terminal height detection."""
        capabilities = ConsoleCapabilities()
        assert isinstance(capabilities.height, int)
        assert capabilities.height > 0

    @patch.dict("os.environ", {"NO_COLOR": "1"})
    def test_no_color_detection(self):
        """Test NO_COLOR environment variable detection."""
        capabilities = ConsoleCapabilities()
        assert capabilities.color_support == "none"

    @patch.dict("os.environ", {"FORCE_COLOR": "1"})
    def test_force_color_detection(self):
        """Test FORCE_COLOR environment variable detection."""
        capabilities = ConsoleCapabilities()
        assert capabilities.color_support == "truecolor"

    @patch.dict("os.environ", {"TERM": "xterm-256color"})
    def test_256_color_detection(self):
        """Test 256 color terminal detection."""
        capabilities = ConsoleCapabilities()
        assert capabilities.color_support == "256"

    def test_unicode_support_detection(self):
        """Test Unicode support detection."""
        capabilities = ConsoleCapabilities()
        # Should work on most systems
        assert isinstance(capabilities.unicode_support, bool)

    def test_emoji_support_detection(self):
        """Test emoji support detection."""
        capabilities = ConsoleCapabilities()
        assert isinstance(capabilities.emoji_support, bool)


class TestVirtualAgoraConsole:
    """Test Virtual Agora console functionality."""

    def test_singleton_pattern(self):
        """Test that console follows singleton pattern."""
        console1 = VirtualAgoraConsole()
        console2 = VirtualAgoraConsole()
        assert console1 is console2

    def test_initialization(self):
        """Test console initialization."""
        console = VirtualAgoraConsole()
        assert hasattr(console, "capabilities")
        assert hasattr(console, "theme")
        assert hasattr(console, "rich_console")

    def test_print_agent_message(self):
        """Test agent message printing."""
        console = VirtualAgoraConsole()

        # Mock rich console to capture output
        with patch.object(console, "rich_console") as mock_console:
            console.print_agent_message(
                "test-agent", ProviderType.OPENAI, "Test message", "12:00:00", 1
            )

            # Should have called print
            mock_console.print.assert_called_once()

    def test_print_system_message(self):
        """Test system message printing."""
        console = VirtualAgoraConsole()

        with patch.object(console, "rich_console") as mock_console:
            console.print_system_message(
                "System message", MessageType.SYSTEM_INFO, "System"
            )

            mock_console.print.assert_called_once()

    def test_print_user_prompt(self):
        """Test user prompt printing."""
        console = VirtualAgoraConsole()

        with patch.object(console, "rich_console") as mock_console:
            console.print_user_prompt("Enter input:", ["Option 1", "Option 2"])

            mock_console.print.assert_called_once()

    def test_print_round_separator(self):
        """Test round separator printing."""
        console = VirtualAgoraConsole()

        with patch.object(console, "rich_console") as mock_console:
            console.print_round_separator(1, "Test Topic")

            # Should print separator
            assert mock_console.print.call_count >= 1

    def test_print_phase_transition(self):
        """Test phase transition printing."""
        console = VirtualAgoraConsole()

        with patch.object(console, "rich_console") as mock_console:
            console.print_phase_transition("Phase 1", "Phase 2")

            mock_console.print.assert_called_once()

    def test_print_voting_results(self):
        """Test voting results printing."""
        console = VirtualAgoraConsole()

        voting_results = {
            "agent1": {"vote": "Yes", "justification": "Good idea"},
            "agent2": {"vote": "No", "justification": "Not ready"},
        }

        with patch.object(console, "rich_console") as mock_console:
            console.print_voting_results(voting_results)

            mock_console.print.assert_called_once()

    def test_print_agenda(self):
        """Test agenda printing."""
        console = VirtualAgoraConsole()

        agenda = ["Topic A", "Topic B", "Topic C"]

        with patch.object(console, "rich_console") as mock_console:
            console.print_agenda(agenda, "Topic A")

            mock_console.print.assert_called_once()

    def test_print_session_summary(self):
        """Test session summary printing."""
        console = VirtualAgoraConsole()

        summary_data = {
            "session_id": "test_session",
            "duration": "1h 30m",
            "topics_count": 3,
            "rounds_count": 15,
            "agent_stats": {
                "agent1": {"message_count": 5},
                "agent2": {"message_count": 3},
            },
        }

        with patch.object(console, "rich_console") as mock_console:
            console.print_session_summary(summary_data)

            mock_console.print.assert_called_once()

    def test_status_spinner_context(self):
        """Test status spinner context manager."""
        console = VirtualAgoraConsole()

        with patch.object(console.rich_console, "status") as mock_status:
            with console.status_spinner("Testing..."):
                pass

            mock_status.assert_called_once_with("Testing...", spinner="dots")

    def test_clear(self):
        """Test console clearing."""
        console = VirtualAgoraConsole()

        with patch.object(console, "rich_console") as mock_console:
            console.clear()

            mock_console.clear.assert_called_once()

    def test_get_width(self):
        """Test width getter."""
        console = VirtualAgoraConsole()
        width = console.get_width()

        assert isinstance(width, int)
        assert width > 0

    def test_get_height(self):
        """Test height getter."""
        console = VirtualAgoraConsole()
        height = console.get_height()

        assert isinstance(height, int)
        assert height > 0

    def test_supports_color(self):
        """Test color support detection."""
        console = VirtualAgoraConsole()
        supports_color = console.supports_color()

        assert isinstance(supports_color, bool)

    def test_supports_unicode(self):
        """Test Unicode support detection."""
        console = VirtualAgoraConsole()
        supports_unicode = console.supports_unicode()

        assert isinstance(supports_unicode, bool)

    def test_supports_emoji(self):
        """Test emoji support detection."""
        console = VirtualAgoraConsole()
        supports_emoji = console.supports_emoji()

        assert isinstance(supports_emoji, bool)


class TestConsoleFunctions:
    """Test convenience functions."""

    def test_get_console(self):
        """Test global console getter."""
        console1 = get_console()
        console2 = get_console()

        assert console1 is console2
        assert isinstance(console1, VirtualAgoraConsole)

    @patch("virtual_agora.ui.console.get_console")
    def test_print_agent_message_function(self, mock_get_console):
        """Test print_agent_message convenience function."""
        mock_console = Mock()
        mock_get_console.return_value = mock_console

        print_agent_message("test-agent", ProviderType.GOOGLE, "Test content")

        mock_console.print_agent_message.assert_called_once_with(
            "test-agent", ProviderType.GOOGLE, "Test content"
        )

    @patch("virtual_agora.ui.console.get_console")
    def test_print_system_message_function(self, mock_get_console):
        """Test print_system_message convenience function."""
        mock_console = Mock()
        mock_get_console.return_value = mock_console

        print_system_message("Test message")

        mock_console.print_system_message.assert_called_once_with(
            "Test message", MessageType.SYSTEM_INFO
        )

    @patch("virtual_agora.ui.console.get_console")
    def test_print_user_prompt_function(self, mock_get_console):
        """Test print_user_prompt convenience function."""
        mock_console = Mock()
        mock_get_console.return_value = mock_console

        print_user_prompt("Enter choice:", ["A", "B"])

        mock_console.print_user_prompt.assert_called_once_with(
            "Enter choice:", ["A", "B"]
        )


class TestConsoleIntegration:
    """Test console integration with other components."""

    def test_theme_integration(self):
        """Test that console integrates with theme system."""
        console = VirtualAgoraConsole()

        # Should have a theme
        assert console.theme is not None

        # Should be able to set theme
        from virtual_agora.ui.theme import VirtualAgoraTheme

        new_theme = VirtualAgoraTheme()
        console.set_theme(new_theme)

        assert console.theme is new_theme

    def test_resize_handling(self):
        """Test terminal resize handling."""
        console = VirtualAgoraConsole()
        original_width = console.capabilities.width

        # Simulate resize
        console.capabilities.update_on_resize()

        # Width should still be valid
        assert isinstance(console.capabilities.width, int)
        assert console.capabilities.width > 0


@pytest.fixture
def mock_console():
    """Provide a mocked console for testing."""
    with patch("virtual_agora.ui.console.get_console") as mock:
        console_instance = Mock(spec=VirtualAgoraConsole)
        mock.return_value = console_instance
        yield console_instance


class TestConsoleWithMock:
    """Test console functionality with mocked dependencies."""

    def test_mocked_console_usage(self, mock_console):
        """Test using mocked console."""
        print_agent_message("test", ProviderType.ANTHROPIC, "message")

        mock_console.print_agent_message.assert_called_once()

    def test_error_handling(self):
        """Test console error handling."""
        console = VirtualAgoraConsole()

        # Test with invalid data - should not crash
        with patch.object(console, "rich_console") as mock_console:
            mock_console.print.side_effect = Exception("Test error")

            # Should handle gracefully
            try:
                console.print("Test")
            except:
                pytest.fail("Console should handle errors gracefully")


if __name__ == "__main__":
    pytest.main([__file__])
