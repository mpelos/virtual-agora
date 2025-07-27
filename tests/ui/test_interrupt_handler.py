"""Tests for interrupt handler module."""

import unittest
import signal
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, call

from src.virtual_agora.ui.interrupt_handler import (
    InterruptContext,
    InterruptAction,
    InterruptHandler,
    get_interrupt_handler,
    setup_interrupt_handlers,
)


class TestInterruptContext(unittest.TestCase):
    """Test InterruptContext class."""

    def test_interrupt_context_creation(self):
        """Test creating interrupt context."""
        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            current_phase=2,
            current_topic="Test Topic",
            current_speaker="agent-1",
            state_snapshot={"key": "value"},
        )

        self.assertEqual(context.interrupt_type, "user_interrupt")
        self.assertEqual(context.current_phase, 2)
        self.assertEqual(context.current_topic, "Test Topic")
        self.assertEqual(context.current_speaker, "agent-1")
        self.assertIsNotNone(context.state_snapshot)

    def test_interrupt_context_to_dict(self):
        """Test converting context to dictionary."""
        timestamp = datetime.now()
        context = InterruptContext(
            interrupt_type="emergency",
            timestamp=timestamp,
            current_phase=1,
            state_snapshot={"test": "data"},
        )

        data = context.to_dict()

        self.assertEqual(data["interrupt_type"], "emergency")
        self.assertEqual(data["timestamp"], timestamp.isoformat())
        self.assertEqual(data["current_phase"], 1)
        self.assertTrue(data["has_state_snapshot"])


class TestInterruptAction(unittest.TestCase):
    """Test InterruptAction constants."""

    def test_action_constants(self):
        """Test that all action constants are defined."""
        self.assertEqual(InterruptAction.PAUSE, "pause")
        self.assertEqual(InterruptAction.SKIP_SPEAKER, "skip_speaker")
        self.assertEqual(InterruptAction.END_TOPIC, "end_topic")
        self.assertEqual(InterruptAction.END_SESSION, "end_session")
        self.assertEqual(InterruptAction.RESUME, "resume")
        self.assertEqual(InterruptAction.SAVE_STATE, "save_state")
        self.assertEqual(InterruptAction.SHOW_STATUS, "show_status")


class TestInterruptHandler(unittest.TestCase):
    """Test InterruptHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = InterruptHandler()
        self.temp_dir = tempfile.mkdtemp()
        self.handler._emergency_save_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("src.virtual_agora.ui.interrupt_handler.signal.signal")
    def test_setup_handlers(self, mock_signal):
        """Test setting up signal handlers."""
        self.handler.setup()

        # Should install SIGINT handler
        mock_signal.assert_any_call(signal.SIGINT, self.handler._handle_sigint)

    @patch("src.virtual_agora.ui.interrupt_handler.signal.signal")
    def test_teardown_handlers(self, mock_signal):
        """Test tearing down signal handlers."""
        # Setup first
        self.handler._original_handlers[signal.SIGINT] = signal.default_int_handler

        self.handler.teardown()

        # Should restore original handler
        mock_signal.assert_called_with(signal.SIGINT, signal.default_int_handler)

    def test_create_interrupt_context(self):
        """Test creating interrupt context from state."""
        # With state manager
        mock_state_manager = MagicMock()
        mock_state_manager.get_snapshot.return_value = {
            "current_phase": 2,
            "active_topic": "Test Topic",
            "current_speaker_id": "agent-1",
        }

        self.handler.state_manager = mock_state_manager

        context = self.handler._create_interrupt_context("test_interrupt")

        self.assertEqual(context.interrupt_type, "test_interrupt")
        self.assertEqual(context.current_phase, 2)
        self.assertEqual(context.current_topic, "Test Topic")
        self.assertEqual(context.current_speaker, "agent-1")
        self.assertIsNotNone(context.state_snapshot)

    @patch("src.virtual_agora.ui.interrupt_handler.console")
    @patch("src.virtual_agora.ui.interrupt_handler.Prompt.ask")
    def test_show_interrupt_menu(self, mock_prompt, mock_console):
        """Test showing interrupt menu."""
        mock_prompt.return_value = "r"  # Resume

        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            current_phase=2,
            current_topic="Test Topic",
        )

        action = self.handler._show_interrupt_menu(context)

        self.assertEqual(action, InterruptAction.RESUME)
        mock_console.clear.assert_called_once()

    @patch("src.virtual_agora.ui.interrupt_handler.console")
    @patch("src.virtual_agora.ui.interrupt_handler.time.sleep")
    def test_pause_session(self, mock_sleep, mock_console):
        """Test pausing session."""
        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            state_snapshot={"test": "data"},
        )

        # Mock state and recovery managers
        self.handler.state_manager = MagicMock()
        self.handler.recovery_manager = MagicMock()
        self.handler.recovery_manager.create_checkpoint.return_value = MagicMock(
            checkpoint_id="test-id"
        )

        with self.assertRaises(SystemExit) as cm:
            self.handler._pause_session(context)

        self.assertEqual(cm.exception.code, 0)
        mock_sleep.assert_called_once()

    def test_skip_current_speaker(self):
        """Test skipping current speaker."""
        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            current_speaker="agent-1",
        )

        mock_state_manager = MagicMock()
        self.handler.state_manager = mock_state_manager

        self.handler._skip_current_speaker(context)

        mock_state_manager.update_state.assert_called_once_with(
            {"skip_current_speaker": True, "interrupt_action": "skip_speaker"}
        )

    def test_end_current_topic(self):
        """Test ending current topic."""
        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            current_topic="Test Topic",
        )

        mock_state_manager = MagicMock()
        self.handler.state_manager = mock_state_manager

        self.handler._end_current_topic(context)

        mock_state_manager.update_state.assert_called_once_with(
            {"force_end_topic": True, "interrupt_action": "end_topic"}
        )

    @patch("src.virtual_agora.ui.interrupt_handler.console")
    @patch("src.virtual_agora.ui.interrupt_handler.time.sleep")
    def test_end_session(self, mock_sleep, mock_console):
        """Test ending session."""
        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            state_snapshot={"test": "data"},
        )

        with self.assertRaises(SystemExit) as cm:
            self.handler._end_session(context)

        self.assertEqual(cm.exception.code, 1)

    @patch("src.virtual_agora.ui.interrupt_handler.console")
    def test_save_state(self, mock_console):
        """Test saving state without exiting."""
        context = InterruptContext(
            interrupt_type="user_interrupt",
            timestamp=datetime.now(),
            state_snapshot={"test": "data"},
        )

        # With managers
        self.handler.state_manager = MagicMock()
        self.handler.recovery_manager = MagicMock()
        self.handler.recovery_manager.create_checkpoint.return_value = MagicMock(
            checkpoint_id="test-id"
        )

        self.handler._save_state(context)

        # Should create checkpoint
        self.handler.recovery_manager.create_checkpoint.assert_called_once()

    @patch("src.virtual_agora.ui.interrupt_handler.console")
    @patch("src.virtual_agora.ui.interrupt_handler.Prompt.ask")
    def test_show_session_status(self, mock_prompt, mock_console):
        """Test showing session status."""
        mock_prompt.return_value = ""  # Enter to continue

        context = InterruptContext(
            interrupt_type="user_interrupt", timestamp=datetime.now()
        )

        mock_state_manager = MagicMock()
        mock_state_manager.get_snapshot.return_value = {
            "session_id": "test-session",
            "current_phase": 2,
            "active_topic": "Test Topic",
            "completed_topics": ["Topic 1"],
            "total_messages": 42,
        }
        self.handler.state_manager = mock_state_manager

        self.handler._show_session_status(context)

        # Should display status
        mock_console.print.assert_called()
        mock_console.clear.assert_called()

    @patch("src.virtual_agora.ui.interrupt_handler.console")
    def test_emergency_shutdown(self, mock_console):
        """Test emergency shutdown."""
        with self.assertRaises(SystemExit) as cm:
            self.handler._emergency_shutdown()

        self.assertEqual(cm.exception.code, 2)

    def test_create_emergency_checkpoint(self):
        """Test creating emergency checkpoint."""
        context = InterruptContext(
            interrupt_type="emergency",
            timestamp=datetime.now(),
            state_snapshot={"test": "data"},
        )

        # With recovery manager
        mock_recovery_manager = MagicMock()
        mock_checkpoint = MagicMock(checkpoint_id="test-checkpoint-id")
        mock_recovery_manager.create_checkpoint.return_value = mock_checkpoint

        self.handler.recovery_manager = mock_recovery_manager

        checkpoint_id = self.handler._create_emergency_checkpoint(
            context, emergency=True
        )

        self.assertEqual(checkpoint_id, "test-checkpoint-id")
        mock_recovery_manager.create_checkpoint.assert_called_once()

        # Without recovery manager (fallback to file)
        self.handler.recovery_manager = None

        checkpoint_id = self.handler._create_emergency_checkpoint(context)

        self.assertTrue(checkpoint_id.endswith(".json"))
        self.assertTrue(Path(checkpoint_id).exists())

    def test_save_interrupt_context(self):
        """Test saving interrupt context."""
        context = InterruptContext(interrupt_type="test", timestamp=datetime.now())

        self.handler._save_interrupt_context(context)

        # Should create context file
        files = list(self.handler._emergency_save_path.glob("interrupt_context_*.json"))
        self.assertEqual(len(files), 1)

    def test_register_callback(self):
        """Test registering callbacks."""
        callback = MagicMock()

        self.handler.register_callback(InterruptAction.PAUSE, callback)

        self.assertIn(InterruptAction.PAUSE, self.handler._callbacks)
        self.assertIn(callback, self.handler._callbacks[InterruptAction.PAUSE])

    def test_execute_callbacks(self):
        """Test executing callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        self.handler.register_callback(InterruptAction.PAUSE, callback1)
        self.handler.register_callback(InterruptAction.PAUSE, callback2)

        context = InterruptContext(interrupt_type="test", timestamp=datetime.now())

        self.handler._execute_callbacks(InterruptAction.PAUSE, context)

        callback1.assert_called_once_with(context)
        callback2.assert_called_once_with(context)

    def test_interrupt_history(self):
        """Test interrupt history tracking."""
        context1 = InterruptContext("test1", datetime.now())
        context2 = InterruptContext("test2", datetime.now())

        self.handler._interrupt_history.append(context1)
        self.handler._interrupt_history.append(context2)

        history = self.handler.get_interrupt_history()

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].interrupt_type, "test1")
        self.assertEqual(history[1].interrupt_type, "test2")

        # Test clear
        self.handler.clear_interrupt_history()
        history = self.handler.get_interrupt_history()
        self.assertEqual(len(history), 0)
        self.assertEqual(self.handler._interrupt_count, 0)

    @patch("src.virtual_agora.ui.interrupt_handler.time.time")
    @patch("src.virtual_agora.ui.interrupt_handler.console")
    def test_rapid_interrupts(self, mock_console, mock_time):
        """Test rapid interrupt detection (panic mode)."""
        # Simulate rapid interrupts
        mock_time.side_effect = [1.0, 1.5, 2.0]  # Within 2 second window

        self.handler._interrupt_count = 2
        self.handler._last_interrupt_time = 0.5

        with self.assertRaises(SystemExit) as cm:
            self.handler._handle_sigint(signal.SIGINT, None)

        self.assertEqual(cm.exception.code, 2)  # Emergency shutdown


class TestGlobalFunctions(unittest.TestCase):
    """Test global functions."""

    @patch("src.virtual_agora.ui.interrupt_handler._interrupt_handler", None)
    def test_get_interrupt_handler(self):
        """Test getting global interrupt handler."""
        handler1 = get_interrupt_handler()
        handler2 = get_interrupt_handler()

        # Should return same instance
        self.assertIs(handler1, handler2)

    def test_setup_interrupt_handlers(self):
        """Test setting up interrupt handlers with managers."""
        mock_state_manager = MagicMock()
        mock_recovery_manager = MagicMock()

        handler = setup_interrupt_handlers(mock_state_manager, mock_recovery_manager)

        self.assertEqual(handler.state_manager, mock_state_manager)
        self.assertEqual(handler.recovery_manager, mock_recovery_manager)


if __name__ == "__main__":
    unittest.main()
