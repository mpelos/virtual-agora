"""Tests for Virtual Agora progress indicators."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from virtual_agora.ui.progress import (
    VirtualAgoraProgress,
    ProgressType,
    OperationType,
    ProgressConfig,
    ProgressSpinner,
    ProgressBar,
    ProgressCounter,
    CompositeProgress,
    track_agent_responses,
    track_voting_process,
    track_file_operations,
    operation_spinner,
    async_spinner,
    ProgressNotification,
)


class TestProgressConfig:
    """Test progress configuration."""

    def test_default_config(self):
        """Test default progress configuration."""
        config = ProgressConfig()

        assert config.show_percentage is True
        assert config.show_eta is True
        assert config.show_elapsed is False
        assert config.show_speed is False
        assert config.bar_width is None
        assert config.spinner_style == "dots"
        assert config.update_frequency == 0.1

    def test_custom_config(self):
        """Test custom progress configuration."""
        config = ProgressConfig(
            show_percentage=False,
            show_eta=False,
            show_elapsed=True,
            bar_width=50,
            spinner_style="arc",
        )

        assert config.show_percentage is False
        assert config.show_eta is False
        assert config.show_elapsed is True
        assert config.bar_width == 50
        assert config.spinner_style == "arc"


class TestVirtualAgoraProgress:
    """Test Virtual Agora progress system."""

    def test_initialization(self):
        """Test progress system initialization."""
        progress = VirtualAgoraProgress()

        assert progress.config is not None
        assert progress.console is not None
        assert progress.theme is not None
        assert progress._active_operations == {}

    def test_initialization_with_config(self):
        """Test progress system with custom config."""
        config = ProgressConfig(show_percentage=False)
        progress = VirtualAgoraProgress(config)

        assert progress.config.show_percentage is False

    def test_create_progress_columns_spinner(self):
        """Test creating spinner progress columns."""
        progress = VirtualAgoraProgress()
        columns = progress._create_progress_columns(ProgressType.SPINNER)

        assert len(columns) >= 2  # At least spinner and text

    def test_create_progress_columns_bar(self):
        """Test creating bar progress columns."""
        progress = VirtualAgoraProgress()
        columns = progress._create_progress_columns(ProgressType.BAR)

        assert len(columns) >= 3  # At least text, bar, percentage

    def test_create_progress_columns_counter(self):
        """Test creating counter progress columns."""
        progress = VirtualAgoraProgress()
        columns = progress._create_progress_columns(ProgressType.COUNTER)

        assert len(columns) >= 2  # At least text and counter

    @patch("virtual_agora.ui.progress.Progress")
    def test_spinner_context(self, mock_progress_class):
        """Test spinner context manager."""
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress_class.return_value = mock_progress

        progress = VirtualAgoraProgress()

        with progress.spinner("Testing...") as spinner:
            assert isinstance(spinner, ProgressSpinner)

        mock_progress.__enter__.assert_called_once()
        mock_progress.__exit__.assert_called_once()

    @patch("virtual_agora.ui.progress.Progress")
    def test_progress_bar_context(self, mock_progress_class):
        """Test progress bar context manager."""
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress_class.return_value = mock_progress

        progress = VirtualAgoraProgress()

        with progress.progress_bar(100, "Testing...") as bar:
            assert isinstance(bar, ProgressBar)

        mock_progress.__enter__.assert_called_once()
        mock_progress.__exit__.assert_called_once()

    @patch("virtual_agora.ui.progress.Progress")
    def test_counter_context(self, mock_progress_class):
        """Test counter context manager."""
        mock_progress = Mock()
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        mock_progress_class.return_value = mock_progress

        progress = VirtualAgoraProgress()

        with progress.counter(10, "Testing...") as counter:
            assert isinstance(counter, ProgressCounter)

        mock_progress.__enter__.assert_called_once()
        mock_progress.__exit__.assert_called_once()

    def test_get_active_operations(self):
        """Test getting active operations."""
        progress = VirtualAgoraProgress()

        # Initially empty
        operations = progress.get_active_operations()
        assert operations == {}

        # Add mock operation
        progress._active_operations["test"] = {"type": "spinner"}
        operations = progress.get_active_operations()

        assert "test" in operations
        assert operations["test"]["type"] == "spinner"

    def test_operation_specific_spinner(self):
        """Test operation-specific spinner."""
        progress = VirtualAgoraProgress()

        context = progress.operation_specific_spinner(OperationType.AGENT_RESPONSE)

        # Should return a context manager
        assert hasattr(context, "__enter__")
        assert hasattr(context, "__exit__")


class TestProgressSpinner:
    """Test progress spinner controller."""

    def test_initialization(self):
        """Test spinner initialization."""
        mock_progress = Mock()
        spinner = ProgressSpinner(mock_progress, "task_id", "op_id", Mock())

        assert spinner.progress is mock_progress
        assert spinner.task_id == "task_id"
        assert spinner.operation_id == "op_id"

    def test_update_description(self):
        """Test updating spinner description."""
        mock_progress = Mock()
        mock_manager = Mock()
        mock_manager._active_operations = {"op_id": {"description": "old"}}

        spinner = ProgressSpinner(mock_progress, "task_id", "op_id", mock_manager)
        spinner.update("New description")

        mock_progress.update.assert_called_once_with(
            "task_id", description="New description"
        )
        assert (
            mock_manager._active_operations["op_id"]["description"] == "New description"
        )


class TestProgressBar:
    """Test progress bar controller."""

    def test_initialization(self):
        """Test bar initialization."""
        mock_progress = Mock()
        bar = ProgressBar(mock_progress, "task_id", "op_id", Mock())

        assert bar.progress is mock_progress
        assert bar.task_id == "task_id"
        assert bar.operation_id == "op_id"

    def test_update_advance(self):
        """Test updating progress bar advance."""
        mock_progress = Mock()
        bar = ProgressBar(mock_progress, "task_id", "op_id", Mock())

        bar.update(5)

        mock_progress.advance.assert_called_once_with("task_id", 5)

    def test_update_description(self):
        """Test updating progress bar description."""
        mock_progress = Mock()
        bar = ProgressBar(mock_progress, "task_id", "op_id", Mock())

        bar.update(1, "New description")

        mock_progress.update.assert_called_once_with(
            "task_id", description="New description"
        )
        mock_progress.advance.assert_called_once_with("task_id", 1)

    def test_set_total(self):
        """Test setting progress bar total."""
        mock_progress = Mock()
        bar = ProgressBar(mock_progress, "task_id", "op_id", Mock())

        bar.set_total(200)

        mock_progress.update.assert_called_once_with("task_id", total=200)

    def test_set_completed(self):
        """Test setting progress bar completed amount."""
        mock_progress = Mock()
        bar = ProgressBar(mock_progress, "task_id", "op_id", Mock())

        bar.set_completed(50)

        mock_progress.update.assert_called_once_with("task_id", completed=50)


class TestProgressCounter:
    """Test progress counter controller."""

    def test_initialization(self):
        """Test counter initialization."""
        mock_progress = Mock()
        counter = ProgressCounter(mock_progress, "task_id", "op_id", Mock())

        assert counter.progress is mock_progress
        assert counter.task_id == "task_id"
        assert counter.operation_id == "op_id"

    def test_increment(self):
        """Test incrementing counter."""
        mock_progress = Mock()
        counter = ProgressCounter(mock_progress, "task_id", "op_id", Mock())

        counter.increment(3)

        mock_progress.advance.assert_called_once_with("task_id", 3)

    def test_increment_with_description(self):
        """Test incrementing counter with description."""
        mock_progress = Mock()
        counter = ProgressCounter(mock_progress, "task_id", "op_id", Mock())

        counter.increment(1, "New count")

        mock_progress.update.assert_called_once_with("task_id", description="New count")
        mock_progress.advance.assert_called_once_with("task_id", 1)

    def test_set_count(self):
        """Test setting counter value."""
        mock_progress = Mock()
        counter = ProgressCounter(mock_progress, "task_id", "op_id", Mock())

        counter.set_count(42)

        mock_progress.update.assert_called_once_with("task_id", completed=42)


class TestCompositeProgress:
    """Test composite progress tracker."""

    def test_initialization(self):
        """Test composite progress initialization."""
        mock_progress1 = Mock()
        mock_progress2 = Mock()
        progress_objects = [mock_progress1, mock_progress2]

        operations = [
            {"description": "Op 1", "total": 100},
            {"description": "Op 2", "total": 50},
        ]

        mock_manager = Mock()
        mock_manager._active_operations = {}

        composite = CompositeProgress(progress_objects, operations, mock_manager)

        assert len(composite.progress_objects) == 2
        assert len(composite.operations) == 2
        assert len(composite.task_ids) == 2
        assert len(composite.operation_ids) == 2

        # Should have started both progress trackers
        mock_progress1.start.assert_called_once()
        mock_progress2.start.assert_called_once()

    def test_update_specific_progress(self):
        """Test updating specific progress in composite."""
        mock_progress1 = Mock()
        mock_progress2 = Mock()
        progress_objects = [mock_progress1, mock_progress2]

        operations = [{"description": "Op 1"}, {"description": "Op 2"}]

        mock_manager = Mock()
        mock_manager._active_operations = {}

        composite = CompositeProgress(progress_objects, operations, mock_manager)

        # Update first progress
        composite.update(0, 5, "Updated")

        mock_progress1.update.assert_called()
        mock_progress1.advance.assert_called_with(composite.task_ids[0], 5)

    def test_complete_specific_progress(self):
        """Test completing specific progress in composite."""
        mock_progress1 = Mock()
        mock_progress1.tasks = {"task1": Mock(total=100)}

        progress_objects = [mock_progress1]
        operations = [{"description": "Op 1"}]

        mock_manager = Mock()
        mock_manager._active_operations = {}

        composite = CompositeProgress(progress_objects, operations, mock_manager)
        composite.task_ids = ["task1"]  # Mock task ID

        composite.complete(0)

        mock_progress1.update.assert_called_with("task1", completed=100)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("virtual_agora.ui.progress.VirtualAgoraProgress")
    def test_track_voting_process(self, mock_progress_class):
        """Test track_voting_process function."""
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress

        voting_items = ["item1", "item2", "item3"]
        context = track_voting_process(voting_items)

        mock_progress.progress_bar.assert_called_once_with(
            total=3,
            description="Collecting votes",
            operation_type=OperationType.VOTING_COLLECTION,
        )

    @patch("virtual_agora.ui.progress.VirtualAgoraProgress")
    def test_track_file_operations(self, mock_progress_class):
        """Test track_file_operations function."""
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress

        context = track_file_operations(5)

        mock_progress.progress_bar.assert_called_once_with(
            total=5,
            description="Processing files",
            operation_type=OperationType.FILE_OPERATION,
        )

    @patch("virtual_agora.ui.progress.VirtualAgoraProgress")
    def test_operation_spinner(self, mock_progress_class):
        """Test operation_spinner function."""
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress

        context = operation_spinner(OperationType.AGENT_RESPONSE)

        mock_progress.operation_specific_spinner.assert_called_once_with(
            OperationType.AGENT_RESPONSE
        )

    @patch("virtual_agora.ui.progress.VirtualAgoraProgress")
    def test_operation_spinner_with_description(self, mock_progress_class):
        """Test operation_spinner with custom description."""
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress

        context = operation_spinner(
            OperationType.REPORT_GENERATION, "Custom description"
        )

        mock_progress.spinner.assert_called_once_with(
            "Custom description", OperationType.REPORT_GENERATION
        )


class TestAsyncProgress:
    """Test async progress functionality."""

    @pytest.mark.asyncio
    async def test_async_spinner(self):
        """Test async spinner context manager."""
        with patch(
            "virtual_agora.ui.progress.VirtualAgoraProgress"
        ) as mock_progress_class:
            mock_progress = Mock()
            mock_spinner = Mock()
            mock_context = Mock()
            mock_context.__enter__ = Mock(return_value=mock_spinner)
            mock_context.__exit__ = Mock(return_value=None)
            mock_progress.spinner.return_value = mock_context
            mock_progress_class.return_value = mock_progress

            async with async_spinner("Testing...") as spinner:
                assert spinner is mock_spinner
                await asyncio.sleep(0.01)  # Brief pause

            mock_progress.spinner.assert_called_once_with("Testing...")


class TestProgressNotification:
    """Test progress notification system."""

    @patch("virtual_agora.ui.progress.get_console")
    def test_show_operation_start(self, mock_get_console):
        """Test showing operation start notification."""
        mock_console = Mock()
        mock_get_console.return_value = mock_console

        ProgressNotification.show_operation_start(
            OperationType.AGENT_RESPONSE, "Test details"
        )

        mock_console.print_system_message.assert_called_once()
        args = mock_console.print_system_message.call_args[0]
        assert "Agent Response" in args[0]
        assert "Test details" in args[0]

    @patch("virtual_agora.ui.progress.get_console")
    def test_show_operation_complete(self, mock_get_console):
        """Test showing operation completion notification."""
        mock_console = Mock()
        mock_get_console.return_value = mock_console

        ProgressNotification.show_operation_complete(
            OperationType.VOTING_COLLECTION, 2.5, "Completed successfully"
        )

        mock_console.print_system_message.assert_called_once()
        args = mock_console.print_system_message.call_args[0]
        assert "Voting Collection" in args[0]
        assert "2.5s" in args[0]
        assert "Completed successfully" in args[0]


class TestProgressTypes:
    """Test progress type enums."""

    def test_progress_type_values(self):
        """Test progress type enum values."""
        assert ProgressType.SPINNER.value == "spinner"
        assert ProgressType.BAR.value == "bar"
        assert ProgressType.COUNTER.value == "counter"
        assert ProgressType.COMPOSITE.value == "composite"

    def test_operation_type_values(self):
        """Test operation type enum values."""
        assert OperationType.AGENT_RESPONSE.value == "agent_response"
        assert OperationType.VOTING_COLLECTION.value == "voting_collection"
        assert OperationType.REPORT_GENERATION.value == "report_generation"
        assert OperationType.FILE_OPERATION.value == "file_operation"
        assert OperationType.CONTEXT_SUMMARIZATION.value == "context_summarization"
        assert OperationType.AGENDA_SYNTHESIS.value == "agenda_synthesis"


class TestProgressIntegration:
    """Test progress system integration."""

    def test_progress_with_different_configs(self):
        """Test progress system with different configurations."""
        configs = [
            ProgressConfig(show_percentage=False),
            ProgressConfig(show_eta=False),
            ProgressConfig(bar_width=40),
        ]

        for config in configs:
            progress = VirtualAgoraProgress(config)
            assert progress.config == config

            # Should be able to create columns
            spinner_columns = progress._create_progress_columns(ProgressType.SPINNER)
            bar_columns = progress._create_progress_columns(ProgressType.BAR)

            assert len(spinner_columns) > 0
            assert len(bar_columns) > 0


if __name__ == "__main__":
    pytest.main([__file__])
