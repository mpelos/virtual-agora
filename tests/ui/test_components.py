"""Tests for UI components module."""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.virtual_agora.ui.components import (
    VirtualAgoraTheme,
    LoadingSpinner,
    ProgressBar,
    create_header_panel,
    create_info_table,
    create_options_menu,
    create_status_panel,
    SessionTimer,
    AgentMessageDisplay,
    InteractiveList,
    ConfirmationDialog,
    StatusDashboard,
    create_markdown_panel,
    console_section,
    format_timestamp,
    InputValidator,
    create_error_panel,
)


class TestVirtualAgoraTheme(unittest.TestCase):
    """Test theme constants."""

    def test_theme_colors(self):
        """Test that theme colors are defined."""
        self.assertEqual(VirtualAgoraTheme.PRIMARY, "cyan")
        self.assertEqual(VirtualAgoraTheme.SECONDARY, "magenta")
        self.assertEqual(VirtualAgoraTheme.SUCCESS, "green")
        self.assertEqual(VirtualAgoraTheme.WARNING, "yellow")
        self.assertEqual(VirtualAgoraTheme.ERROR, "red")
        self.assertEqual(VirtualAgoraTheme.INFO, "blue")


class TestLoadingSpinner(unittest.TestCase):
    """Test loading spinner context manager."""

    @patch("src.virtual_agora.ui.components.Progress")
    def test_loading_spinner_context(self, mock_progress_class):
        """Test spinner enters and exits correctly."""
        mock_progress = MagicMock()
        mock_progress_class.return_value = mock_progress

        with LoadingSpinner("Testing...") as spinner:
            mock_progress.start.assert_called_once()
            mock_progress.add_task.assert_called_once_with("Testing...", total=None)

            # Test update
            spinner.update("New message")
            mock_progress.update.assert_called()

        mock_progress.stop.assert_called_once()


class TestProgressBar(unittest.TestCase):
    """Test progress bar functionality."""

    @patch("src.virtual_agora.ui.components.Progress")
    def test_progress_bar_context(self, mock_progress_class):
        """Test progress bar enters and exits correctly."""
        mock_progress = MagicMock()
        mock_progress_class.return_value = mock_progress
        mock_progress.add_task.return_value = 1  # task_id

        with ProgressBar(100, "Processing") as progress:
            mock_progress.start.assert_called_once()
            mock_progress.add_task.assert_called_once_with("Processing", total=100)

            # Test update
            progress.update(10)
            mock_progress.advance.assert_called_with(1, 10)

            # Test update with description
            progress.update(5, "New description")
            mock_progress.update.assert_called()

        mock_progress.stop.assert_called_once()


class TestUICreationFunctions(unittest.TestCase):
    """Test UI element creation functions."""

    def test_create_header_panel(self):
        """Test header panel creation."""
        panel = create_header_panel("Test Title", "Test Subtitle")
        self.assertIsNotNone(panel)
        # Panel object has limited testable attributes without rendering

    def test_create_info_table(self):
        """Test info table creation."""
        data = {"Key1": "Value1", "Key2": "Value2"}
        table = create_info_table(data, "Test Table")
        self.assertIsNotNone(table)
        self.assertEqual(len(table.columns), 2)

    def test_create_options_menu(self):
        """Test options menu creation."""
        options = {"a": "Option A", "b": "Option B"}
        panel = create_options_menu(options, "Test Menu")
        self.assertIsNotNone(panel)

    def test_create_status_panel(self):
        """Test status panel creation."""
        # Test different styles
        for style in ["info", "success", "warning", "error"]:
            panel = create_status_panel("Test status", style=style)
            self.assertIsNotNone(panel)


class TestSessionTimer(unittest.TestCase):
    """Test session timer functionality."""

    def test_session_timer_display(self):
        """Test timer display generation."""
        start_time = datetime.now()
        timer = SessionTimer(start_time)

        display = timer.generate_display()
        self.assertIsNotNone(display)


class TestAgentMessageDisplay(unittest.TestCase):
    """Test agent message formatting."""

    def test_format_message(self):
        """Test message formatting for different roles."""
        # Test participant message
        panel = AgentMessageDisplay.format_message(
            "agent-1", "Test message", role="participant"
        )
        self.assertIsNotNone(panel)

        # Test moderator message
        panel = AgentMessageDisplay.format_message(
            "moderator", "Test moderator message", role="moderator"
        )
        self.assertIsNotNone(panel)


class TestInteractiveList(unittest.TestCase):
    """Test interactive list selector."""

    def test_interactive_list_display(self):
        """Test list display generation."""
        items = ["Item 1", "Item 2", "Item 3"]
        interactive_list = InteractiveList(items, "Select an item")

        table = interactive_list.display()
        self.assertIsNotNone(table)
        self.assertEqual(len(table.columns), 2)


class TestConfirmationDialog(unittest.TestCase):
    """Test confirmation dialog."""

    @patch("src.virtual_agora.ui.components.console")
    @patch("src.virtual_agora.ui.components.Confirm.ask")
    def test_confirmation_dialog(self, mock_confirm, mock_console):
        """Test confirmation dialog display."""
        mock_confirm.return_value = True

        result = ConfirmationDialog.ask("Are you sure?")

        self.assertTrue(result)
        mock_console.print.assert_called_once()
        mock_confirm.assert_called_once_with("Proceed?", default=True)

        # Test danger mode
        mock_confirm.reset_mock()
        mock_console.reset_mock()

        result = ConfirmationDialog.ask("Delete everything?", danger=True)
        mock_console.print.assert_called_once()


class TestStatusDashboard(unittest.TestCase):
    """Test status dashboard."""

    def test_dashboard_setup(self):
        """Test dashboard layout setup."""
        dashboard = StatusDashboard()
        dashboard.setup_layout()

        # Verify layout structure
        self.assertIsNotNone(dashboard.layout)

        # Test component updates
        dashboard.update_component("header", "Test Header")
        self.assertEqual(dashboard.components["header"], "Test Header")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        now = datetime.now()

        # Test absolute formatting
        formatted = format_timestamp(now, relative=False)
        self.assertIn(str(now.year), formatted)

        # Test relative formatting
        formatted = format_timestamp(now, relative=True)
        self.assertEqual(formatted, "just now")

    def test_input_validator(self):
        """Test input validation utilities."""
        # Test length validation
        valid, error = InputValidator.validate_length(
            "test", min_length=2, max_length=10
        )
        self.assertTrue(valid)
        self.assertIsNone(error)

        valid, error = InputValidator.validate_length("t", min_length=2)
        self.assertFalse(valid)
        self.assertIsNotNone(error)

        # Test choice validation
        valid, error = InputValidator.validate_choice("a", ["a", "b", "c"])
        self.assertTrue(valid)

        valid, error = InputValidator.validate_choice("d", ["a", "b", "c"])
        self.assertFalse(valid)

        # Test not empty validation
        valid, error = InputValidator.validate_not_empty("test")
        self.assertTrue(valid)

        valid, error = InputValidator.validate_not_empty("  ")
        self.assertFalse(valid)

    def test_create_error_panel(self):
        """Test error panel creation."""
        error = Exception("Test error")

        # Without traceback
        panel = create_error_panel(error, show_traceback=False)
        self.assertIsNotNone(panel)

        # With traceback
        panel = create_error_panel(error, show_traceback=True)
        self.assertIsNotNone(panel)

    def test_create_markdown_panel(self):
        """Test markdown panel creation."""
        content = "# Test Header\n\nTest content"
        panel = create_markdown_panel(content, "Test Title")
        self.assertIsNotNone(panel)

    @patch("src.virtual_agora.ui.components.console")
    def test_console_section(self, mock_console):
        """Test console section context manager."""
        with console_section("Test Section"):
            pass

        mock_console.rule.assert_called_once()
        mock_console.print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
