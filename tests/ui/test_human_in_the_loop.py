import unittest
from unittest.mock import patch
from typing import List, Dict, Any

from src.virtual_agora.ui.human_in_the_loop import (
    get_initial_topic,
    get_agenda_approval,
    edit_agenda,
    get_continuation_approval,
    get_agenda_modifications,
    display_session_status,
    handle_emergency_interrupt,
)


class TestHumanInTheLoop(unittest.TestCase):

    @patch("builtins.input", side_effect=["Test Topic"])
    def test_get_initial_topic(self, mock_input):
        topic = get_initial_topic()
        self.assertEqual(topic, "Test Topic")

    @patch("builtins.input", side_effect=["a"])
    def test_get_agenda_approval_approve(self, mock_input):
        agenda = ["Topic 1", "Topic 2"]
        approved_agenda = get_agenda_approval(agenda)
        self.assertEqual(approved_agenda, agenda)

    @patch("builtins.input", side_effect=["e"])
    def test_get_agenda_approval_edit(self, mock_input):
        agenda = ["Topic 1", "Topic 2"]
        # For now, edit_agenda returns the original agenda
        edited_agenda = get_agenda_approval(agenda)
        self.assertEqual(edited_agenda, agenda)

    @patch("builtins.input", side_effect=["r"])
    def test_get_agenda_approval_reject(self, mock_input):
        agenda = ["Topic 1", "Topic 2"]
        rejected_agenda = get_agenda_approval(agenda)
        self.assertEqual(rejected_agenda, [])

    def test_edit_agenda(self):
        agenda = ["Topic 1", "Topic 2"]
        # For now, edit_agenda returns the original agenda
        edited_agenda = edit_agenda(agenda)
        self.assertEqual(edited_agenda, agenda)

    @patch("builtins.input", side_effect=["y"])
    def test_get_continuation_approval_yes(self, mock_input):
        action = get_continuation_approval("Topic 1", ["Topic 2"])
        self.assertEqual(action, "y")

    @patch("builtins.input", side_effect=["n"])
    def test_get_continuation_approval_no(self, mock_input):
        action = get_continuation_approval("Topic 1", ["Topic 2"])
        self.assertEqual(action, "n")

    @patch("builtins.input", side_effect=["m"])
    def test_get_continuation_approval_modify(self, mock_input):
        action = get_continuation_approval("Topic 1", ["Topic 2"])
        self.assertEqual(action, "m")

    @patch("builtins.input", side_effect=["New Topic 1, New Topic 2"])
    def test_get_agenda_modifications(self, mock_input):
        modified_agenda = get_agenda_modifications(["Old Topic"])
        self.assertEqual(modified_agenda, ["New Topic 1", "New Topic 2"])

    @patch("builtins.print")
    def test_display_session_status(self, mock_print):
        status = {"Topic": "Test", "Round": 1}
        display_session_status(status)
        mock_print.assert_any_call("\n--- Session Status ---")
        mock_print.assert_any_call("Topic: Test")
        mock_print.assert_any_call("Round: 1")
        mock_print.assert_any_call("----------------------\n")

    def test_handle_emergency_interrupt(self):
        with self.assertRaises(SystemExit):
            handle_emergency_interrupt()


if __name__ == "__main__":
    unittest.main()
