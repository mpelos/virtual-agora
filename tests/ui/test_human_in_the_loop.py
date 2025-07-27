import unittest
from unittest.mock import patch, MagicMock, call
from typing import List, Dict, Any
from datetime import datetime

from src.virtual_agora.ui.human_in_the_loop import (
    get_initial_topic,
    get_agenda_approval,
    edit_agenda,
    get_continuation_approval,
    get_agenda_modifications,
    display_session_status,
    handle_emergency_interrupt,
    validate_input,
    record_input,
    show_help,
)


class TestHumanInTheLoop(unittest.TestCase):

    def setUp(self):
        """Reset module state before each test."""
        # Clear input history
        from src.virtual_agora.ui import human_in_the_loop

        human_in_the_loop.input_history = []

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Confirm.ask")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_initial_topic(self, mock_prompt, mock_confirm, mock_console):
        # Setup mocks
        mock_prompt.side_effect = ["This is a test topic for discussion", ""]
        mock_confirm.return_value = True

        topic = get_initial_topic()

        self.assertEqual(topic, "This is a test topic for discussion")
        mock_console.clear.assert_called_once()
        # Verify input was recorded
        from src.virtual_agora.ui import human_in_the_loop

        self.assertEqual(len(human_in_the_loop.input_history), 1)
        self.assertEqual(human_in_the_loop.input_history[0]["type"], "initial_topic")

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_agenda_approval_approve(self, mock_prompt, mock_console):
        mock_prompt.return_value = "a"
        agenda = ["Topic 1", "Topic 2"]

        approved_agenda = get_agenda_approval(agenda)

        self.assertEqual(approved_agenda, agenda)
        mock_console.clear.assert_called_once()

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.edit_agenda")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_agenda_approval_edit(self, mock_prompt, mock_edit, mock_console):
        mock_prompt.return_value = "e"
        agenda = ["Topic 1", "Topic 2"]
        edited_agenda = ["New Topic 1", "New Topic 2"]
        mock_edit.return_value = edited_agenda

        result = get_agenda_approval(agenda)

        self.assertEqual(result, edited_agenda)
        mock_edit.assert_called_once_with(agenda)

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_agenda_approval_reject(self, mock_prompt, mock_console):
        mock_prompt.return_value = "r"
        agenda = ["Topic 1", "Topic 2"]

        rejected_agenda = get_agenda_approval(agenda)

        self.assertEqual(rejected_agenda, [])

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Confirm.ask")
    @patch("src.virtual_agora.ui.human_in_the_loop.IntPrompt.ask")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_edit_agenda(
        self, mock_prompt, mock_int_prompt, mock_confirm, mock_console
    ):
        agenda = ["Topic 1", "Topic 2"]
        # Simulate done editing
        mock_prompt.side_effect = ["d"]
        mock_confirm.return_value = True

        edited_agenda = edit_agenda(agenda)

        self.assertEqual(edited_agenda, agenda)

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_continuation_approval_yes(self, mock_prompt, mock_console):
        mock_prompt.return_value = "y"

        action = get_continuation_approval("Topic 1", ["Topic 2"])

        self.assertEqual(action, "y")

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_continuation_approval_no(self, mock_prompt, mock_console):
        mock_prompt.return_value = "n"

        action = get_continuation_approval("Topic 1", ["Topic 2"])

        self.assertEqual(action, "n")

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_get_continuation_approval_modify(self, mock_prompt, mock_console):
        mock_prompt.return_value = "m"

        action = get_continuation_approval("Topic 1", ["Topic 2"])

        self.assertEqual(action, "m")

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.edit_agenda")
    @patch("src.virtual_agora.ui.human_in_the_loop.Confirm.ask")
    def test_get_agenda_modifications(self, mock_confirm, mock_edit, mock_console):
        mock_confirm.return_value = True
        mock_edit.return_value = ["New Topic 1", "New Topic 2"]

        modified_agenda = get_agenda_modifications(["Old Topic"])

        self.assertEqual(modified_agenda, ["New Topic 1", "New Topic 2"])

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    def test_display_session_status(self, mock_console):
        status = {"Topic": "Test", "Round": 1}

        display_session_status(status)

        # Verify console.print was called (Rich handles the actual display)
        self.assertTrue(mock_console.print.called)

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_handle_emergency_interrupt(self, mock_prompt, mock_console):
        mock_prompt.return_value = "q"  # Quit option

        with self.assertRaises(SystemExit) as cm:
            handle_emergency_interrupt()

        self.assertEqual(cm.exception.code, 1)

    def test_validate_input(self):
        # Test valid input
        is_valid, error = validate_input("Valid input", min_length=5, max_length=20)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # Test too short
        is_valid, error = validate_input("Hi", min_length=5)
        self.assertFalse(is_valid)
        self.assertIn("too short", error)

        # Test too long
        is_valid, error = validate_input("A" * 25, max_length=20)
        self.assertFalse(is_valid)
        self.assertIn("too long", error)

        # Test empty required
        is_valid, error = validate_input("", required=True)
        self.assertFalse(is_valid)
        self.assertIn("empty", error)

        # Test invalid characters
        is_valid, error = validate_input("test<script>")
        self.assertFalse(is_valid)
        self.assertIn("invalid characters", error)

    @patch("src.virtual_agora.ui.human_in_the_loop.console")
    @patch("src.virtual_agora.ui.human_in_the_loop.Prompt.ask")
    def test_show_help(self, mock_prompt, mock_console):
        mock_prompt.return_value = ""  # Enter to continue

        # Test general help
        show_help()

        # Test context-specific help
        show_help("topic_input")

        # Verify console was used
        self.assertTrue(mock_console.print.called)


if __name__ == "__main__":
    unittest.main()
