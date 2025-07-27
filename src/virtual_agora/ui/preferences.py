"""User preferences management for Virtual Agora.

This module implements Story 7.7: User Preference Management with
persistence, validation, and runtime modification support.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)

# Default preferences file location
DEFAULT_PREFS_FILE = Path.home() / ".virtual_agora" / "preferences.json"


@dataclass
class UserPreferences:
    """User preferences for Virtual Agora sessions.

    Implements comprehensive preference management with:
    - Auto-approval settings
    - Default choices
    - Display preferences
    - Timeout configurations
    """

    # Approval preferences
    auto_approve_unanimous_votes: bool = False
    auto_approve_agenda_on_consensus: bool = False
    require_confirmation_for_changes: bool = True

    # Default choices
    default_continuation_choice: str = "y"  # y/n/m
    default_agenda_action: str = "a"  # a/e/r
    prefer_detailed_summaries: bool = True

    # Timeout preferences (in seconds)
    input_timeout: int = 300  # 5 minutes
    decision_timeout: int = 600  # 10 minutes
    emergency_timeout: int = 30  # 30 seconds

    # Display preferences
    display_verbosity: str = "normal"  # minimal/normal/detailed
    show_agent_reasoning: bool = True
    show_voting_details: bool = True
    show_timestamps: bool = True
    use_color: bool = True
    clear_screen_between_phases: bool = True

    # Session preferences
    save_session_logs: bool = True
    log_directory: str = str(Path.home() / ".virtual_agora" / "logs")
    auto_save_interval: int = 300  # 5 minutes

    # Interaction preferences
    enable_help_tooltips: bool = True
    show_keyboard_shortcuts: bool = True
    confirm_emergency_actions: bool = True
    allow_agenda_modifications: bool = True

    # Advanced preferences
    enable_debug_mode: bool = False
    show_performance_metrics: bool = False
    record_interaction_history: bool = True

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["last_modified"] = self.last_modified.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create preferences from dictionary."""
        # Convert ISO format strings back to datetime
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_modified" in data:
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])

        return cls(**data)

    def update(self, **kwargs) -> None:
        """Update preferences with validation."""
        valid_fields = set(self.__dataclass_fields__.keys())

        for key, value in kwargs.items():
            if key not in valid_fields:
                logger.warning(f"Ignoring unknown preference: {key}")
                continue

            # Validate value types
            expected_type = self.__dataclass_fields__[key].type
            if not self._validate_type(value, expected_type):
                logger.error(
                    f"Invalid type for {key}: expected {expected_type}, got {type(value)}"
                )
                continue

            setattr(self, key, value)

        self.last_modified = datetime.now()
        logger.info(f"Updated {len(kwargs)} preferences")

    def _validate_type(self, value: Any, expected_type: type) -> bool:
        """Validate that value matches expected type."""
        # Handle Optional types
        if hasattr(expected_type, "__args__"):
            return isinstance(value, expected_type.__args__)
        return isinstance(value, expected_type)


class PreferencesManager:
    """Manages loading, saving, and accessing user preferences."""

    def __init__(self, preferences_file: Optional[Path] = None):
        """Initialize preferences manager.

        Args:
            preferences_file: Path to preferences file (uses default if None)
        """
        self.preferences_file = preferences_file or DEFAULT_PREFS_FILE
        self._preferences: Optional[UserPreferences] = None
        self._ensure_preferences_dir()

    def _ensure_preferences_dir(self) -> None:
        """Ensure preferences directory exists."""
        self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> UserPreferences:
        """Load preferences from file or create defaults.

        Returns:
            Loaded or default preferences
        """
        if self._preferences is not None:
            return self._preferences

        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, "r") as f:
                    data = json.load(f)
                    self._preferences = UserPreferences.from_dict(data)
                    logger.info(f"Loaded preferences from {self.preferences_file}")
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
                logger.info("Using default preferences")
                self._preferences = UserPreferences()
        else:
            logger.info("No preferences file found, using defaults")
            self._preferences = UserPreferences()
            self.save()  # Save defaults

        return self._preferences

    def save(self) -> None:
        """Save current preferences to file."""
        if self._preferences is None:
            logger.warning("No preferences to save")
            return

        try:
            with open(self.preferences_file, "w") as f:
                json.dump(self._preferences.to_dict(), f, indent=2, sort_keys=True)
            logger.info(f"Saved preferences to {self.preferences_file}")
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")

    def get(self) -> UserPreferences:
        """Get current preferences, loading if necessary.

        Returns:
            Current preferences
        """
        if self._preferences is None:
            self.load()
        return self._preferences

    def update(self, **kwargs) -> None:
        """Update preferences and save.

        Args:
            **kwargs: Preference fields to update
        """
        prefs = self.get()
        prefs.update(**kwargs)
        self.save()

    def reset(self) -> None:
        """Reset preferences to defaults."""
        self._preferences = UserPreferences()
        self.save()
        logger.info("Reset preferences to defaults")

    def export(self, export_file: Path) -> None:
        """Export preferences to a file.

        Args:
            export_file: Path to export file
        """
        prefs = self.get()
        try:
            with open(export_file, "w") as f:
                json.dump(prefs.to_dict(), f, indent=2, sort_keys=True)
            logger.info(f"Exported preferences to {export_file}")
        except Exception as e:
            logger.error(f"Error exporting preferences: {e}")

    def import_from(self, import_file: Path) -> None:
        """Import preferences from a file.

        Args:
            import_file: Path to import file
        """
        try:
            with open(import_file, "r") as f:
                data = json.load(f)
                self._preferences = UserPreferences.from_dict(data)
                self.save()
            logger.info(f"Imported preferences from {import_file}")
        except Exception as e:
            logger.error(f"Error importing preferences: {e}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference value.

        Args:
            key: Preference key
            default: Default value if key not found

        Returns:
            Preference value or default
        """
        prefs = self.get()
        return getattr(prefs, key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """Set a specific preference value.

        Args:
            key: Preference key
            value: New value
        """
        self.update(**{key: value})

    def validate_preferences(self) -> Dict[str, str]:
        """Validate current preferences.

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        prefs = self.get()
        errors = {}

        # Validate timeout values
        if prefs.input_timeout < 10:
            errors["input_timeout"] = "Input timeout must be at least 10 seconds"
        if prefs.decision_timeout < prefs.input_timeout:
            errors["decision_timeout"] = "Decision timeout must be >= input timeout"

        # Validate string choices
        if prefs.default_continuation_choice not in ["y", "n", "m"]:
            errors["default_continuation_choice"] = "Must be 'y', 'n', or 'm'"
        if prefs.default_agenda_action not in ["a", "e", "r"]:
            errors["default_agenda_action"] = "Must be 'a', 'e', or 'r'"
        if prefs.display_verbosity not in ["minimal", "normal", "detailed"]:
            errors["display_verbosity"] = "Must be 'minimal', 'normal', or 'detailed'"

        # Validate paths
        log_dir = Path(prefs.log_directory)
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors["log_directory"] = f"Cannot create log directory: {e}"

        return errors


# Global preferences manager instance
_preferences_manager: Optional[PreferencesManager] = None


def get_preferences_manager() -> PreferencesManager:
    """Get the global preferences manager instance.

    Returns:
        PreferencesManager instance
    """
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = PreferencesManager()
    return _preferences_manager


def get_user_preferences() -> UserPreferences:
    """Get current user preferences.

    Returns:
        Current UserPreferences
    """
    return get_preferences_manager().get()


def update_user_preferences(**kwargs) -> None:
    """Update user preferences.

    Args:
        **kwargs: Preference fields to update
    """
    get_preferences_manager().update(**kwargs)


def reset_user_preferences() -> None:
    """Reset user preferences to defaults."""
    get_preferences_manager().reset()
