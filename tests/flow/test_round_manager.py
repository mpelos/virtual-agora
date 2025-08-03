"""Tests for RoundManager class.

This module provides comprehensive tests for the centralized round management
functionality, ensuring 100% coverage and backward compatibility.
"""

import pytest
from unittest.mock import patch
from typing import Dict, Any

from src.virtual_agora.flow.round_manager import RoundManager
from src.virtual_agora.state.schema import VirtualAgoraState


class TestRoundManager:
    """Test suite for RoundManager class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()

    def create_test_state(self, **kwargs) -> Dict[str, Any]:
        """Create a test state with default values.

        Args:
            **kwargs: Override default state values

        Returns:
            Dictionary representing VirtualAgoraState for testing
        """
        default_state = {
            "current_round": 0,
            "active_topic": "Test Topic",
            "current_phase": 2,
            "checkpoint_interval": 3,
        }
        default_state.update(kwargs)
        return default_state

    def test_get_current_round_with_value(self):
        """Test get_current_round when round is set."""
        state = self.create_test_state(current_round=5)

        result = self.round_manager.get_current_round(state)

        assert result == 5

    def test_get_current_round_default_fallback(self):
        """Test get_current_round falls back to 0 when not set."""
        state = self.create_test_state()
        del state["current_round"]  # Remove current_round

        result = self.round_manager.get_current_round(state)

        assert result == 0

    def test_get_current_round_none_value(self):
        """Test get_current_round handles None value correctly."""
        state = self.create_test_state(current_round=None)

        result = self.round_manager.get_current_round(state)

        assert result == 0

    def test_start_new_round_from_zero(self):
        """Test start_new_round increments from 0."""
        state = self.create_test_state(current_round=0)

        result = self.round_manager.start_new_round(state)

        assert result == 1

    def test_start_new_round_from_existing(self):
        """Test start_new_round increments from existing value."""
        state = self.create_test_state(current_round=7)

        result = self.round_manager.start_new_round(state)

        assert result == 8

    def test_start_new_round_with_missing_field(self):
        """Test start_new_round when current_round is missing."""
        state = self.create_test_state()
        del state["current_round"]

        result = self.round_manager.start_new_round(state)

        assert result == 1

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_start_new_round_logs_transition(self, mock_logger):
        """Test start_new_round logs the transition."""
        state = self.create_test_state(current_round=2)

        self.round_manager.start_new_round(state)

        mock_logger.debug.assert_called_once_with("Starting new round: 2 -> 3")

    def test_can_start_round_with_active_topic(self):
        """Test can_start_round returns True when topic is active."""
        state = self.create_test_state(active_topic="Test Topic")

        result = self.round_manager.can_start_round(state)

        assert result is True

    def test_can_start_round_without_active_topic(self):
        """Test can_start_round returns False when no active topic."""
        state = self.create_test_state(active_topic=None)

        result = self.round_manager.can_start_round(state)

        assert result is False

    def test_can_start_round_missing_topic_field(self):
        """Test can_start_round when active_topic field is missing."""
        state = self.create_test_state()
        del state["active_topic"]

        result = self.round_manager.can_start_round(state)

        assert result is False

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_can_start_round_logs_warning(self, mock_logger):
        """Test can_start_round logs warning when no topic."""
        state = self.create_test_state(active_topic=None)

        self.round_manager.can_start_round(state)

        mock_logger.warning.assert_called_once_with(
            "Cannot start round: no active topic"
        )

    def test_get_round_metadata_complete(self):
        """Test get_round_metadata returns complete metadata."""
        state = self.create_test_state(
            active_topic="Democracy", current_phase=2, checkpoint_interval=3
        )

        result = self.round_manager.get_round_metadata(state, 5)

        expected = {
            "round_number": 5,
            "topic": "Democracy",
            "phase": 2,
            "is_threshold_round": True,  # 5 >= 3
            "is_checkpoint_round": False,  # 5 % 3 != 0
        }
        assert result == expected

    def test_get_round_metadata_checkpoint_round(self):
        """Test get_round_metadata identifies checkpoint rounds."""
        state = self.create_test_state(checkpoint_interval=3)

        result = self.round_manager.get_round_metadata(state, 6)

        assert result["is_checkpoint_round"] is True  # 6 % 3 == 0

    def test_get_round_metadata_threshold_round(self):
        """Test get_round_metadata identifies threshold rounds."""
        state = self.create_test_state()

        result = self.round_manager.get_round_metadata(state, 3)

        assert result["is_threshold_round"] is True  # 3 >= 3

    def test_get_round_metadata_below_threshold(self):
        """Test get_round_metadata for rounds below threshold."""
        state = self.create_test_state()

        result = self.round_manager.get_round_metadata(state, 2)

        assert result["is_threshold_round"] is False  # 2 < 3

    def test_get_round_metadata_round_zero(self):
        """Test get_round_metadata for round 0."""
        state = self.create_test_state(checkpoint_interval=3)

        result = self.round_manager.get_round_metadata(state, 0)

        assert result["is_checkpoint_round"] is False  # round 0 never checkpoint

    def test_get_round_metadata_missing_fields(self):
        """Test get_round_metadata with missing state fields."""
        state = {}

        result = self.round_manager.get_round_metadata(state, 4)

        expected = {
            "round_number": 4,
            "topic": "Unknown Topic",
            "phase": 0,
            "is_threshold_round": True,
            "is_checkpoint_round": False,  # checkpoint_interval defaults to 3, 4 % 3 != 0
        }
        assert result == expected

    def test_get_round_metadata_zero_checkpoint_interval(self):
        """Test get_round_metadata with zero checkpoint interval."""
        state = self.create_test_state(checkpoint_interval=0)

        result = self.round_manager.get_round_metadata(state, 3)

        assert result["is_checkpoint_round"] is False  # interval 0 disables checkpoints

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_get_round_metadata_logs_debug(self, mock_logger):
        """Test get_round_metadata logs debug information."""
        state = self.create_test_state()

        result = self.round_manager.get_round_metadata(state, 2)

        mock_logger.debug.assert_called_once()
        args = mock_logger.debug.call_args[0]
        assert "Round 2 metadata:" in args[0]

    def test_should_trigger_polling_above_threshold(self):
        """Test should_trigger_polling for rounds >= 3."""
        state = self.create_test_state(current_round=3)

        result = self.round_manager.should_trigger_polling(state)

        assert result is True

    def test_should_trigger_polling_below_threshold(self):
        """Test should_trigger_polling for rounds < 3."""
        state = self.create_test_state(current_round=2)

        result = self.round_manager.should_trigger_polling(state)

        assert result is False

    def test_should_trigger_polling_high_round(self):
        """Test should_trigger_polling for high round numbers."""
        state = self.create_test_state(current_round=10)

        result = self.round_manager.should_trigger_polling(state)

        assert result is True

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_should_trigger_polling_logs_true(self, mock_logger):
        """Test should_trigger_polling logs when triggering."""
        state = self.create_test_state(current_round=5)

        self.round_manager.should_trigger_polling(state)

        mock_logger.debug.assert_called_once_with("Round 5 >= 3, triggering polling")

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_should_trigger_polling_logs_false(self, mock_logger):
        """Test should_trigger_polling logs when not triggering."""
        state = self.create_test_state(current_round=1)

        self.round_manager.should_trigger_polling(state)

        mock_logger.debug.assert_called_once_with("Round 1 < 3, no polling")

    def test_should_trigger_checkpoint_on_interval(self):
        """Test should_trigger_checkpoint on checkpoint intervals."""
        state = self.create_test_state(current_round=6, checkpoint_interval=3)

        result = self.round_manager.should_trigger_checkpoint(state)

        assert result is True  # 6 % 3 == 0

    def test_should_trigger_checkpoint_off_interval(self):
        """Test should_trigger_checkpoint off checkpoint intervals."""
        state = self.create_test_state(current_round=5, checkpoint_interval=3)

        result = self.round_manager.should_trigger_checkpoint(state)

        assert result is False  # 5 % 3 != 0

    def test_should_trigger_checkpoint_round_zero(self):
        """Test should_trigger_checkpoint never triggers on round 0."""
        state = self.create_test_state(current_round=0, checkpoint_interval=1)

        result = self.round_manager.should_trigger_checkpoint(state)

        assert result is False  # round 0 never triggers

    def test_should_trigger_checkpoint_zero_interval(self):
        """Test should_trigger_checkpoint with zero interval."""
        state = self.create_test_state(current_round=3, checkpoint_interval=0)

        result = self.round_manager.should_trigger_checkpoint(state)

        assert result is False  # interval 0 disables checkpoints

    def test_should_trigger_checkpoint_missing_interval(self):
        """Test should_trigger_checkpoint with missing interval field."""
        state = self.create_test_state(current_round=3)
        del state["checkpoint_interval"]

        result = self.round_manager.should_trigger_checkpoint(state)

        assert result is True  # defaults to 3, and 3 % 3 == 0

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_should_trigger_checkpoint_logs_true(self, mock_logger):
        """Test should_trigger_checkpoint logs when triggering."""
        state = self.create_test_state(current_round=9, checkpoint_interval=3)

        self.round_manager.should_trigger_checkpoint(state)

        mock_logger.debug.assert_called_once_with(
            "Round 9 triggers checkpoint (interval: 3)"
        )

    @patch("src.virtual_agora.flow.round_manager.logger")
    def test_should_trigger_checkpoint_logs_false(self, mock_logger):
        """Test should_trigger_checkpoint logs when not triggering."""
        state = self.create_test_state(current_round=4, checkpoint_interval=3)

        self.round_manager.should_trigger_checkpoint(state)

        mock_logger.debug.assert_called_once_with("Round 4 no checkpoint (interval: 3)")


class TestRoundManagerIntegration:
    """Integration tests for RoundManager with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.round_manager = RoundManager()

    def test_typical_round_progression(self):
        """Test typical round progression from 0 to 5."""
        state = {
            "current_round": 0,
            "active_topic": "Climate Change",
            "checkpoint_interval": 3,
        }

        # Simulate round progression
        rounds = []
        for i in range(5):
            current = self.round_manager.get_current_round(state)
            new_round = self.round_manager.start_new_round(state)

            rounds.append(
                {
                    "current": current,
                    "new": new_round,
                    "can_start": self.round_manager.can_start_round(state),
                    "should_poll": self.round_manager.should_trigger_polling(state),
                    "should_checkpoint": self.round_manager.should_trigger_checkpoint(
                        state
                    ),
                }
            )

            # Update state for next iteration
            state["current_round"] = new_round

        # Verify progression
        assert rounds[0] == {
            "current": 0,
            "new": 1,
            "can_start": True,
            "should_poll": False,
            "should_checkpoint": False,
        }
        assert rounds[1] == {
            "current": 1,
            "new": 2,
            "can_start": True,
            "should_poll": False,
            "should_checkpoint": False,
        }
        assert rounds[2] == {
            "current": 2,
            "new": 3,
            "can_start": True,
            "should_poll": False,
            "should_checkpoint": False,
        }
        assert rounds[3] == {
            "current": 3,
            "new": 4,
            "can_start": True,
            "should_poll": True,
            "should_checkpoint": True,
        }
        assert rounds[4] == {
            "current": 4,
            "new": 5,
            "can_start": True,
            "should_poll": True,
            "should_checkpoint": False,
        }

    def test_backward_compatibility_patterns(self):
        """Test that RoundManager replaces existing patterns correctly."""
        state = {"current_round": 7, "active_topic": "Test"}

        # Test replacement for: state.get("current_round", 0)
        assert self.round_manager.get_current_round(state) == 7

        # Test replacement for: state.get("current_round", 0) + 1
        assert self.round_manager.start_new_round(state) == 8

        # Test replacement for threshold checks
        assert self.round_manager.should_trigger_polling(state) is True

        # Test missing field behavior
        state_missing = {"active_topic": "Test"}
        assert self.round_manager.get_current_round(state_missing) == 0
        assert self.round_manager.start_new_round(state_missing) == 1

    def test_edge_cases_and_error_conditions(self):
        """Test edge cases and error conditions."""
        round_manager = self.round_manager

        # Empty state
        empty_state = {}
        assert round_manager.get_current_round(empty_state) == 0
        assert round_manager.start_new_round(empty_state) == 1
        assert round_manager.can_start_round(empty_state) is False

        # Negative values (shouldn't happen but test resilience)
        negative_state = {"current_round": -1, "checkpoint_interval": -1}
        assert round_manager.get_current_round(negative_state) == -1
        assert round_manager.start_new_round(negative_state) == 0
        assert round_manager.should_trigger_checkpoint(negative_state) is False

        # Very large values
        large_state = {"current_round": 1000, "checkpoint_interval": 100}
        assert round_manager.get_current_round(large_state) == 1000
        assert round_manager.start_new_round(large_state) == 1001
        assert (
            round_manager.should_trigger_checkpoint(large_state) is True
        )  # 1000 % 100 == 0
