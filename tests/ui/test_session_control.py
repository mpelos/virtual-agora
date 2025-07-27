"""Tests for enhanced session control in v1.3."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import signal
import sys
from datetime import datetime
from rich.console import Console

from virtual_agora.ui.session_control import SessionController, CheckpointManager


@pytest.fixture
def console():
    """Mock console for testing."""
    return Mock(spec=Console)


@pytest.fixture
def session_controller(console):
    """Create session controller instance."""
    return SessionController(console)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    return str(tmp_path / "checkpoints")


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create checkpoint manager with temp directory."""
    return CheckpointManager(temp_checkpoint_dir)


class TestSessionController:
    """Test the enhanced session controller."""

    def test_initialization(self, session_controller):
        """Test session controller initialization."""
        assert session_controller.console is not None
        assert session_controller.interrupt_callback is None
        assert session_controller.session_paused is False
        assert session_controller.checkpoint_history == []

    def test_interrupt_callback_setting(self, session_controller):
        """Test setting interrupt callback."""
        callback = Mock()
        session_controller.set_interrupt_callback(callback)
        assert session_controller.interrupt_callback == callback

    def test_periodic_control_check(self, session_controller):
        """Test periodic control point checking."""
        # Should trigger at multiples of 5
        assert session_controller.check_periodic_control(5) is True
        assert session_controller.check_periodic_control(10) is True
        assert session_controller.check_periodic_control(15) is True

        # Should not trigger at non-multiples
        assert session_controller.check_periodic_control(3) is False
        assert session_controller.check_periodic_control(7) is False
        assert session_controller.check_periodic_control(0) is False

    def test_periodic_control_no_duplicates(self, session_controller):
        """Test that periodic control doesn't trigger twice for same round."""
        # First check should trigger
        assert session_controller.check_periodic_control(5) is True

        # Add checkpoint history for round 5
        session_controller.checkpoint_history.append(
            {"round": 5, "type": "periodic_5_round"}
        )

        # Second check for same round should not trigger
        assert session_controller.check_periodic_control(5) is False

    def test_checkpoint_notification_display(self, session_controller):
        """Test checkpoint notification display."""
        state = {
            "messages": ["msg1", "msg2", "msg3"],
            "completed_topics": ["Topic 1"],
            "start_time": datetime.now(),
        }

        session_controller.display_checkpoint_notification(
            round_num=5, topic="Current Topic", state=state
        )

        # Verify console methods were called
        session_controller.console.bell.assert_called_once()
        session_controller.console.clear.assert_called_once()
        assert session_controller.console.print.called

        # Verify checkpoint was recorded
        assert len(session_controller.checkpoint_history) == 1
        assert session_controller.checkpoint_history[0]["round"] == 5
        assert session_controller.checkpoint_history[0]["topic"] == "Current Topic"

    @patch("rich.prompt.Prompt.ask")
    def test_interrupt_menu_resume(self, mock_prompt, session_controller):
        """Test interrupt menu with resume choice."""
        mock_prompt.return_value = "r"

        session_controller._show_interrupt_menu()

        # Should show menu and resume message
        assert session_controller.console.print.called
        assert any(
            "Resuming session" in str(call)
            for call in session_controller.console.print.call_args_list
        )

    @patch("rich.prompt.Prompt.ask")
    def test_interrupt_menu_end_topic(self, mock_prompt, session_controller):
        """Test interrupt menu with end topic choice."""
        mock_prompt.return_value = "e"
        callback = Mock()
        session_controller.set_interrupt_callback(callback)

        session_controller._show_interrupt_menu()

        # Callback should be called with end_topic action
        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert args["action"] == "end_topic"
        assert args["reason"] == "User interrupt"

    @patch("rich.prompt.Prompt.ask")
    def test_interrupt_menu_skip_to_report(self, mock_prompt, session_controller):
        """Test interrupt menu with skip to report choice."""
        mock_prompt.return_value = "s"
        callback = Mock()
        session_controller.set_interrupt_callback(callback)

        session_controller._show_interrupt_menu()

        # Callback should be called with skip_to_report action
        callback.assert_called_once()
        args = callback.call_args[0][0]
        assert args["action"] == "skip_to_report"

    @patch("sys.exit")
    @patch("rich.prompt.Prompt.ask")
    def test_interrupt_menu_pause(self, mock_prompt, mock_exit, session_controller):
        """Test interrupt menu with pause choice."""
        mock_prompt.return_value = "p"
        callback = Mock()
        session_controller.set_interrupt_callback(callback)
        session_controller.current_state = {"test": "state"}

        session_controller._show_interrupt_menu()

        # Should save checkpoint and exit
        assert len(session_controller.checkpoint_history) == 1
        callback.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("sys.exit")
    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_interrupt_menu_quit(
        self, mock_prompt, mock_confirm, mock_exit, session_controller
    ):
        """Test interrupt menu with quit choice."""
        mock_prompt.return_value = "q"
        mock_confirm.return_value = True

        session_controller._show_interrupt_menu()

        # Should confirm and exit
        mock_confirm.assert_called_once()
        mock_exit.assert_called_once_with(0)

    def test_format_duration(self, session_controller):
        """Test duration formatting."""
        # Test hours
        start_time = datetime.now()
        start_time = start_time.replace(
            hour=start_time.hour - 2, minute=start_time.minute - 30
        )
        duration = session_controller._format_duration(start_time)
        assert "h" in duration and "m" in duration

        # Test None
        assert session_controller._format_duration(None) == "Unknown"

    def test_show_topic_transition(self, session_controller):
        """Test topic transition display."""
        session_controller.show_topic_transition(
            from_topic="Topic A", to_topic="Topic B", reason="Natural transition"
        )

        # Verify console print was called with transition info
        assert session_controller.console.print.called

    def test_show_phase_transition(self, session_controller):
        """Test phase transition display."""
        phase_names = {1: "Agenda Creation", 2: "Discussion"}

        session_controller.show_phase_transition(1, 2, phase_names)

        # Verify console print was called
        assert session_controller.console.print.called

    @patch("rich.prompt.Confirm.ask")
    def test_confirm_session_end(self, mock_confirm, session_controller):
        """Test session end confirmation."""
        mock_confirm.return_value = True

        stats = {"duration": "2h 15m", "topics_completed": 5, "total_messages": 150}

        result = session_controller.confirm_session_end(
            reason="All topics completed", stats=stats
        )

        assert result is True
        assert session_controller.console.print.called

    def test_signal_handler_setup(self, session_controller):
        """Test that signal handler is set up."""
        # Get the current SIGINT handler
        current_handler = signal.getsignal(signal.SIGINT)

        # Should not be the default handler
        assert current_handler != signal.SIG_DFL
        assert callable(current_handler)


class TestCheckpointManager:
    """Test the checkpoint manager."""

    def test_initialization(self, checkpoint_manager):
        """Test checkpoint manager initialization."""
        import os

        assert os.path.exists(checkpoint_manager.checkpoint_dir)

    def test_save_checkpoint(self, checkpoint_manager):
        """Test saving a checkpoint."""
        session_id = "test_session"
        state = {
            "messages": ["msg1", "msg2"],
            "current_round": 5,
            "active_topic": "Test Topic",
        }
        metadata = {"operation": "test_save"}

        checkpoint_id = checkpoint_manager.save_checkpoint(session_id, state, metadata)

        assert checkpoint_id.startswith(f"{session_id}_")

        # Verify file was created
        import os

        filepath = f"{checkpoint_manager.checkpoint_dir}/{checkpoint_id}.json"
        assert os.path.exists(filepath)

    def test_load_checkpoint(self, checkpoint_manager):
        """Test loading a checkpoint."""
        # First save a checkpoint
        session_id = "test_session"
        state = {"test": "data", "round": 10}

        checkpoint_id = checkpoint_manager.save_checkpoint(session_id, state)

        # Then load it
        loaded_data = checkpoint_manager.load_checkpoint(checkpoint_id)

        assert loaded_data["checkpoint_id"] == checkpoint_id
        assert loaded_data["session_id"] == session_id
        assert loaded_data["state"] == state

    def test_list_checkpoints(self, checkpoint_manager):
        """Test listing checkpoints."""
        # Save multiple checkpoints
        session_id = "test_session"

        for i in range(3):
            checkpoint_manager.save_checkpoint(session_id, {"round": i}, {"index": i})

        # List all checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3

        # List by session
        session_checkpoints = checkpoint_manager.list_checkpoints(session_id)
        assert len(session_checkpoints) == 3

        # List non-existent session
        empty_list = checkpoint_manager.list_checkpoints("non_existent")
        assert len(empty_list) == 0

    def test_delete_checkpoint(self, checkpoint_manager):
        """Test deleting a checkpoint."""
        # Save a checkpoint
        session_id = "test_session"
        checkpoint_id = checkpoint_manager.save_checkpoint(session_id, {"test": "data"})

        # Delete it
        result = checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert result is True

        # Try to load deleted checkpoint (should fail)
        with pytest.raises(Exception):
            checkpoint_manager.load_checkpoint(checkpoint_id)

        # Try to delete non-existent checkpoint
        result = checkpoint_manager.delete_checkpoint("non_existent")
        assert result is False

    def test_checkpoint_sorting(self, checkpoint_manager):
        """Test that checkpoints are sorted by timestamp."""
        import time

        session_id = "test_session"

        # Save checkpoints with small delays
        for i in range(3):
            checkpoint_manager.save_checkpoint(session_id, {"index": i})
            time.sleep(0.1)

        # List should be sorted newest first
        checkpoints = checkpoint_manager.list_checkpoints()

        # Verify descending timestamp order
        for i in range(len(checkpoints) - 1):
            assert checkpoints[i]["timestamp"] >= checkpoints[i + 1]["timestamp"]

    def test_checkpoint_error_handling(self, checkpoint_manager):
        """Test error handling in checkpoint operations."""
        # Test loading non-existent checkpoint
        with pytest.raises(Exception):
            checkpoint_manager.load_checkpoint("non_existent")

        # Test saving with invalid path
        checkpoint_manager.checkpoint_dir = "/invalid/path"

        with pytest.raises(Exception):
            checkpoint_manager.save_checkpoint("test", {})
