"""Tests for user preferences module."""

import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

from src.virtual_agora.ui.preferences import (
    UserPreferences,
    PreferencesManager,
    get_preferences_manager,
    get_user_preferences,
    update_user_preferences,
    reset_user_preferences,
)


class TestUserPreferences(unittest.TestCase):
    """Test UserPreferences dataclass."""

    def test_default_preferences(self):
        """Test default preference values."""
        prefs = UserPreferences()

        # Test approval preferences
        self.assertFalse(prefs.auto_approve_unanimous_votes)
        self.assertFalse(prefs.auto_approve_agenda_on_consensus)
        self.assertTrue(prefs.require_confirmation_for_changes)

        # Test default choices
        self.assertEqual(prefs.default_continuation_choice, "y")
        self.assertEqual(prefs.default_agenda_action, "a")

        # Test timeouts
        self.assertEqual(prefs.input_timeout, 300)
        self.assertEqual(prefs.decision_timeout, 600)

        # Test display preferences
        self.assertEqual(prefs.display_verbosity, "normal")
        self.assertTrue(prefs.use_color)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        prefs = UserPreferences()
        data = prefs.to_dict()

        self.assertIsInstance(data, dict)
        self.assertIn("auto_approve_unanimous_votes", data)
        self.assertIn("created_at", data)
        self.assertIn("version", data)

        # Check datetime conversion
        self.assertIsInstance(data["created_at"], str)
        self.assertIsInstance(data["last_modified"], str)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "auto_approve_unanimous_votes": True,
            "display_verbosity": "detailed",
            "input_timeout": 600,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        prefs = UserPreferences.from_dict(data)

        self.assertTrue(prefs.auto_approve_unanimous_votes)
        self.assertEqual(prefs.display_verbosity, "detailed")
        self.assertEqual(prefs.input_timeout, 600)
        self.assertIsInstance(prefs.created_at, datetime)

    def test_update_preferences(self):
        """Test updating preferences."""
        prefs = UserPreferences()
        original_modified = prefs.last_modified

        # Update valid fields
        prefs.update(
            auto_approve_unanimous_votes=True,
            display_verbosity="minimal",
            input_timeout=400,
        )

        self.assertTrue(prefs.auto_approve_unanimous_votes)
        self.assertEqual(prefs.display_verbosity, "minimal")
        self.assertEqual(prefs.input_timeout, 400)
        self.assertNotEqual(prefs.last_modified, original_modified)

    @patch("src.virtual_agora.ui.preferences.logger")
    def test_update_invalid_field(self, mock_logger):
        """Test updating with invalid field names."""
        prefs = UserPreferences()

        # Try to update non-existent field
        prefs.update(invalid_field="value")

        # Should log warning
        mock_logger.warning.assert_called()

    @patch("src.virtual_agora.ui.preferences.logger")
    def test_update_invalid_type(self, mock_logger):
        """Test updating with invalid types."""
        prefs = UserPreferences()

        # Try to update with wrong type
        prefs.update(input_timeout="not_an_int")

        # Should log error
        mock_logger.error.assert_called()


class TestPreferencesManager(unittest.TestCase):
    """Test PreferencesManager class."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.prefs_file = Path(self.temp_dir) / "preferences.json"

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_default_preferences(self):
        """Test loading when no file exists."""
        manager = PreferencesManager(self.prefs_file)
        prefs = manager.load()

        self.assertIsInstance(prefs, UserPreferences)
        # Should create file with defaults
        self.assertTrue(self.prefs_file.exists())

    def test_save_preferences(self):
        """Test saving preferences to file."""
        manager = PreferencesManager(self.prefs_file)
        prefs = manager.load()

        # Modify preferences
        prefs.auto_approve_unanimous_votes = True
        prefs.display_verbosity = "detailed"

        # Save
        manager.save()

        # Verify file contents
        with open(self.prefs_file) as f:
            data = json.load(f)

        self.assertTrue(data["auto_approve_unanimous_votes"])
        self.assertEqual(data["display_verbosity"], "detailed")

    def test_load_existing_preferences(self):
        """Test loading from existing file."""
        # Create a preferences file
        data = {
            "auto_approve_unanimous_votes": True,
            "display_verbosity": "minimal",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        with open(self.prefs_file, "w") as f:
            json.dump(data, f)

        # Load preferences
        manager = PreferencesManager(self.prefs_file)
        prefs = manager.load()

        self.assertTrue(prefs.auto_approve_unanimous_votes)
        self.assertEqual(prefs.display_verbosity, "minimal")

    @patch("src.virtual_agora.ui.preferences.logger")
    def test_load_corrupted_file(self, mock_logger):
        """Test loading from corrupted file."""
        # Create corrupted file
        with open(self.prefs_file, "w") as f:
            f.write("not valid json")

        manager = PreferencesManager(self.prefs_file)
        prefs = manager.load()

        # Should use defaults and log error
        self.assertIsInstance(prefs, UserPreferences)
        mock_logger.error.assert_called()

    def test_update_and_save(self):
        """Test update method."""
        manager = PreferencesManager(self.prefs_file)

        # Update preferences
        manager.update(auto_approve_unanimous_votes=True, input_timeout=400)

        # Verify saved
        with open(self.prefs_file) as f:
            data = json.load(f)

        self.assertTrue(data["auto_approve_unanimous_votes"])
        self.assertEqual(data["input_timeout"], 400)

    def test_reset_preferences(self):
        """Test resetting to defaults."""
        manager = PreferencesManager(self.prefs_file)

        # Modify preferences
        manager.update(auto_approve_unanimous_votes=True)

        # Reset
        manager.reset()

        # Verify defaults
        prefs = manager.get()
        self.assertFalse(prefs.auto_approve_unanimous_votes)

    def test_export_preferences(self):
        """Test exporting preferences."""
        manager = PreferencesManager(self.prefs_file)
        export_file = Path(self.temp_dir) / "export.json"

        # Export
        manager.export(export_file)

        # Verify export file
        self.assertTrue(export_file.exists())
        with open(export_file) as f:
            data = json.load(f)
        self.assertIn("version", data)

    def test_import_preferences(self):
        """Test importing preferences."""
        manager = PreferencesManager(self.prefs_file)
        import_file = Path(self.temp_dir) / "import.json"

        # Create import file
        import_data = {
            "auto_approve_unanimous_votes": True,
            "display_verbosity": "detailed",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        with open(import_file, "w") as f:
            json.dump(import_data, f)

        # Import
        manager.import_from(import_file)

        # Verify imported
        prefs = manager.get()
        self.assertTrue(prefs.auto_approve_unanimous_votes)
        self.assertEqual(prefs.display_verbosity, "detailed")

    def test_get_set_preference(self):
        """Test getting and setting individual preferences."""
        manager = PreferencesManager(self.prefs_file)

        # Get preference
        value = manager.get_preference("display_verbosity")
        self.assertEqual(value, "normal")

        # Get with default
        value = manager.get_preference("non_existent", "default")
        self.assertEqual(value, "default")

        # Set preference
        manager.set_preference("display_verbosity", "minimal")
        value = manager.get_preference("display_verbosity")
        self.assertEqual(value, "minimal")

    def test_validate_preferences(self):
        """Test preference validation."""
        manager = PreferencesManager(self.prefs_file)

        # Valid preferences
        errors = manager.validate_preferences()
        self.assertEqual(len(errors), 0)

        # Invalid preferences
        manager.update(
            input_timeout=5,  # Too low
            default_continuation_choice="x",  # Invalid choice
            display_verbosity="invalid",  # Invalid verbosity
        )

        errors = manager.validate_preferences()
        self.assertIn("input_timeout", errors)
        self.assertIn("default_continuation_choice", errors)
        self.assertIn("display_verbosity", errors)


class TestGlobalFunctions(unittest.TestCase):
    """Test global preference functions."""

    @patch("src.virtual_agora.ui.preferences._preferences_manager", None)
    def test_get_preferences_manager(self):
        """Test getting global preferences manager."""
        manager1 = get_preferences_manager()
        manager2 = get_preferences_manager()

        # Should return same instance
        self.assertIs(manager1, manager2)

    @patch("src.virtual_agora.ui.preferences.get_preferences_manager")
    def test_get_user_preferences(self, mock_get_manager):
        """Test getting user preferences."""
        mock_manager = unittest.mock.MagicMock()
        mock_get_manager.return_value = mock_manager

        prefs = UserPreferences()
        mock_manager.get.return_value = prefs

        result = get_user_preferences()

        self.assertEqual(result, prefs)
        mock_manager.get.assert_called_once()

    @patch("src.virtual_agora.ui.preferences.get_preferences_manager")
    def test_update_user_preferences(self, mock_get_manager):
        """Test updating user preferences."""
        mock_manager = unittest.mock.MagicMock()
        mock_get_manager.return_value = mock_manager

        update_user_preferences(display_verbosity="minimal")

        mock_manager.update.assert_called_once_with(display_verbosity="minimal")

    @patch("src.virtual_agora.ui.preferences.get_preferences_manager")
    def test_reset_user_preferences(self, mock_get_manager):
        """Test resetting user preferences."""
        mock_manager = unittest.mock.MagicMock()
        mock_get_manager.return_value = mock_manager

        reset_user_preferences()

        mock_manager.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
