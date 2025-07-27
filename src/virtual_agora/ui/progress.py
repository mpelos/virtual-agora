"""Advanced progress indicators for Virtual Agora terminal UI.

This module provides comprehensive progress tracking including spinners,
progress bars, and live status updates for agent operations.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from enum import Enum

from rich.progress import (
    Progress,
    ProgressColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TaskID,
    track,
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich import box

from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ProgressType(Enum):
    """Types of progress indicators."""

    SPINNER = "spinner"
    BAR = "bar"
    COUNTER = "counter"
    COMPOSITE = "composite"


class OperationType(Enum):
    """Types of operations being tracked."""

    AGENT_RESPONSE = "agent_response"
    VOTING_COLLECTION = "voting_collection"
    REPORT_GENERATION = "report_generation"
    FILE_OPERATION = "file_operation"
    CONTEXT_SUMMARIZATION = "context_summarization"
    AGENDA_SYNTHESIS = "agenda_synthesis"


@dataclass
class ProgressConfig:
    """Configuration for progress indicators."""

    show_percentage: bool = True
    show_eta: bool = True
    show_elapsed: bool = False
    show_speed: bool = False
    bar_width: Optional[int] = None
    spinner_style: str = "dots"
    update_frequency: float = 0.1


class VirtualAgoraProgress:
    """Enhanced progress tracking system."""

    def __init__(self, config: Optional[ProgressConfig] = None):
        """Initialize progress system."""
        self.config = config or ProgressConfig()
        self.console = get_console()
        self.theme = get_current_theme()
        self._active_operations: Dict[str, Any] = {}

    def _create_progress_columns(
        self, progress_type: ProgressType
    ) -> List[ProgressColumn]:
        """Create progress columns based on type and configuration."""
        columns = []

        if progress_type == ProgressType.SPINNER:
            columns.append(SpinnerColumn(spinner_name=self.config.spinner_style))
            columns.append(TextColumn("[progress.description]{task.description}"))
            if self.config.show_elapsed:
                columns.append(TimeElapsedColumn())

        elif progress_type == ProgressType.BAR:
            columns.append(TextColumn("[bold blue]{task.description}", justify="right"))
            columns.append(BarColumn(bar_width=self.config.bar_width))

            if self.config.show_percentage:
                columns.append(
                    TextColumn("[progress.percentage]{task.percentage:>3.1f}%")
                )

            columns.append(TextColumn("•"))

            if self.config.show_eta:
                columns.append(TimeRemainingColumn())

            if self.config.show_elapsed:
                columns.append(TimeElapsedColumn())

        elif progress_type == ProgressType.COUNTER:
            columns.append(TextColumn("[bold blue]{task.description}"))
            columns.append(
                TextColumn("[progress.completed]{task.completed}/{task.total}")
            )

        return columns

    @contextmanager
    def spinner(
        self,
        description: str = "Processing...",
        operation_type: Optional[OperationType] = None,
    ):
        """Context manager for spinner progress indicator."""
        columns = self._create_progress_columns(ProgressType.SPINNER)

        progress = Progress(*columns, console=self.console.rich_console, transient=True)

        with progress:
            task_id = progress.add_task(description, total=None)
            operation_id = f"spinner_{int(time.time() * 1000)}"

            self._active_operations[operation_id] = {
                "type": "spinner",
                "operation_type": operation_type,
                "start_time": time.time(),
                "description": description,
            }

            try:
                yield ProgressSpinner(progress, task_id, operation_id, self)
            finally:
                self._active_operations.pop(operation_id, None)

    @contextmanager
    def progress_bar(
        self,
        total: int,
        description: str = "Progress",
        operation_type: Optional[OperationType] = None,
    ):
        """Context manager for progress bar."""
        columns = self._create_progress_columns(ProgressType.BAR)

        progress = Progress(*columns, console=self.console.rich_console)

        with progress:
            task_id = progress.add_task(description, total=total)
            operation_id = f"bar_{int(time.time() * 1000)}"

            self._active_operations[operation_id] = {
                "type": "bar",
                "operation_type": operation_type,
                "start_time": time.time(),
                "description": description,
                "total": total,
            }

            try:
                yield ProgressBar(progress, task_id, operation_id, self)
            finally:
                self._active_operations.pop(operation_id, None)

    @contextmanager
    def counter(
        self,
        total: int,
        description: str = "Items",
        operation_type: Optional[OperationType] = None,
    ):
        """Context manager for counter progress indicator."""
        columns = self._create_progress_columns(ProgressType.COUNTER)

        progress = Progress(*columns, console=self.console.rich_console)

        with progress:
            task_id = progress.add_task(description, total=total)
            operation_id = f"counter_{int(time.time() * 1000)}"

            self._active_operations[operation_id] = {
                "type": "counter",
                "operation_type": operation_type,
                "start_time": time.time(),
                "description": description,
                "total": total,
            }

            try:
                yield ProgressCounter(progress, task_id, operation_id, self)
            finally:
                self._active_operations.pop(operation_id, None)

    @contextmanager
    def composite_progress(self, operations: List[Dict[str, Any]]):
        """Context manager for multiple progress indicators."""
        progress_objects = []

        # Create progress for each operation
        for op in operations:
            op_type = op.get("type", ProgressType.BAR)
            columns = self._create_progress_columns(op_type)
            prog = Progress(*columns, console=self.console.rich_console)
            progress_objects.append(prog)

        # Group all progress indicators
        group = Group(*progress_objects)

        with Live(group, console=self.console.rich_console, refresh_per_second=10):
            composite_tracker = CompositeProgress(progress_objects, operations, self)
            try:
                yield composite_tracker
            finally:
                # Cleanup
                for op_id in composite_tracker.operation_ids:
                    self._active_operations.pop(op_id, None)

    def get_active_operations(self) -> Dict[str, Any]:
        """Get currently active operations."""
        return self._active_operations.copy()

    def operation_specific_spinner(
        self, operation_type: OperationType
    ) -> contextmanager:
        """Get operation-specific spinner with customized description."""
        descriptions = {
            OperationType.AGENT_RESPONSE: "🤖 Generating agent response...",
            OperationType.VOTING_COLLECTION: "🗳️ Collecting votes...",
            OperationType.REPORT_GENERATION: "📄 Generating report...",
            OperationType.FILE_OPERATION: "💾 Processing files...",
            OperationType.CONTEXT_SUMMARIZATION: "📝 Summarizing context...",
            OperationType.AGENDA_SYNTHESIS: "📋 Synthesizing agenda...",
        }

        description = descriptions.get(operation_type, "Processing...")
        return self.spinner(description, operation_type)


class ProgressSpinner:
    """Spinner progress indicator controller."""

    def __init__(
        self,
        progress: Progress,
        task_id: TaskID,
        operation_id: str,
        manager: VirtualAgoraProgress,
    ):
        self.progress = progress
        self.task_id = task_id
        self.operation_id = operation_id
        self.manager = manager

    def update(self, description: Optional[str] = None):
        """Update spinner description."""
        if description:
            self.progress.update(self.task_id, description=description)
            if self.operation_id in self.manager._active_operations:
                self.manager._active_operations[self.operation_id][
                    "description"
                ] = description


class ProgressBar:
    """Progress bar controller."""

    def __init__(
        self,
        progress: Progress,
        task_id: TaskID,
        operation_id: str,
        manager: VirtualAgoraProgress,
    ):
        self.progress = progress
        self.task_id = task_id
        self.operation_id = operation_id
        self.manager = manager

    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        if description:
            self.progress.update(self.task_id, description=description)
        self.progress.advance(self.task_id, advance)

    def set_total(self, total: int):
        """Update total for progress bar."""
        self.progress.update(self.task_id, total=total)

    def set_completed(self, completed: int):
        """Set completed amount."""
        self.progress.update(self.task_id, completed=completed)


class ProgressCounter:
    """Counter progress indicator controller."""

    def __init__(
        self,
        progress: Progress,
        task_id: TaskID,
        operation_id: str,
        manager: VirtualAgoraProgress,
    ):
        self.progress = progress
        self.task_id = task_id
        self.operation_id = operation_id
        self.manager = manager

    def increment(self, amount: int = 1, description: Optional[str] = None):
        """Increment counter."""
        if description:
            self.progress.update(self.task_id, description=description)
        self.progress.advance(self.task_id, amount)

    def set_count(self, count: int):
        """Set current count."""
        self.progress.update(self.task_id, completed=count)


class CompositeProgress:
    """Composite progress tracker for multiple operations."""

    def __init__(
        self,
        progress_objects: List[Progress],
        operations: List[Dict[str, Any]],
        manager: VirtualAgoraProgress,
    ):
        self.progress_objects = progress_objects
        self.operations = operations
        self.manager = manager
        self.task_ids = []
        self.operation_ids = []

        # Initialize all progress trackers
        for i, (prog, op) in enumerate(zip(progress_objects, operations)):
            prog.start()
            task_id = prog.add_task(
                op.get("description", f"Operation {i+1}"), total=op.get("total", None)
            )
            self.task_ids.append(task_id)

            operation_id = f"composite_{i}_{int(time.time() * 1000)}"
            self.operation_ids.append(operation_id)

            manager._active_operations[operation_id] = {
                "type": "composite",
                "operation_type": op.get("operation_type"),
                "start_time": time.time(),
                "description": op.get("description"),
                "index": i,
            }

    def update(self, index: int, advance: int = 1, description: Optional[str] = None):
        """Update specific progress tracker."""
        if 0 <= index < len(self.progress_objects):
            prog = self.progress_objects[index]
            task_id = self.task_ids[index]

            if description:
                prog.update(task_id, description=description)
            prog.advance(task_id, advance)

    def complete(self, index: int):
        """Mark specific operation as complete."""
        if 0 <= index < len(self.progress_objects):
            prog = self.progress_objects[index]
            task_id = self.task_ids[index]
            prog.update(task_id, completed=prog.tasks[task_id].total or 100)


# Convenience functions for common operations


def track_agent_responses(agents: List[str], operation_callback: Callable) -> None:
    """Track progress of collecting agent responses."""
    progress_manager = VirtualAgoraProgress()

    with progress_manager.progress_bar(
        total=len(agents),
        description="Collecting agent responses",
        operation_type=OperationType.AGENT_RESPONSE,
    ) as progress:
        for i, agent in enumerate(agents):
            progress.update(description=f"Waiting for {agent}...")
            result = operation_callback(agent)
            progress.update(1)


def track_voting_process(voting_items: List[str]) -> contextmanager:
    """Create progress tracker for voting process."""
    progress_manager = VirtualAgoraProgress()
    return progress_manager.progress_bar(
        total=len(voting_items),
        description="Collecting votes",
        operation_type=OperationType.VOTING_COLLECTION,
    )


def track_file_operations(file_count: int) -> contextmanager:
    """Create progress tracker for file operations."""
    progress_manager = VirtualAgoraProgress()
    return progress_manager.progress_bar(
        total=file_count,
        description="Processing files",
        operation_type=OperationType.FILE_OPERATION,
    )


@contextmanager
def operation_spinner(operation_type: OperationType, description: Optional[str] = None):
    """Convenience function for operation-specific spinners."""
    progress_manager = VirtualAgoraProgress()

    if description is None:
        return progress_manager.operation_specific_spinner(operation_type)
    else:
        return progress_manager.spinner(description, operation_type)


# Async progress support


@asynccontextmanager
async def async_spinner(
    description: str = "Processing...", update_interval: float = 0.1
):
    """Async context manager for spinner with periodic updates."""
    progress_manager = VirtualAgoraProgress()

    with progress_manager.spinner(description) as spinner:
        # Create a task to periodically update the spinner
        async def update_loop():
            counter = 0
            while True:
                await asyncio.sleep(update_interval)
                counter += 1
                # Optional: update description with counter or other info

        update_task = asyncio.create_task(update_loop())

        try:
            yield spinner
        finally:
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass


class ProgressNotification:
    """System for showing progress notifications."""

    @staticmethod
    def show_operation_start(operation_type: OperationType, details: str = ""):
        """Show operation start notification."""
        console = get_console()

        icons = {
            OperationType.AGENT_RESPONSE: "🤖",
            OperationType.VOTING_COLLECTION: "🗳️",
            OperationType.REPORT_GENERATION: "📄",
            OperationType.FILE_OPERATION: "💾",
            OperationType.CONTEXT_SUMMARIZATION: "📝",
            OperationType.AGENDA_SYNTHESIS: "📋",
        }

        icon = icons.get(operation_type, "⚙️")
        message = f"{icon} Starting {operation_type.value.replace('_', ' ').title()}"

        if details:
            message += f": {details}"

        console.print_system_message(message, title="Operation Started")

    @staticmethod
    def show_operation_complete(
        operation_type: OperationType, duration: float, details: str = ""
    ):
        """Show operation completion notification."""
        console = get_console()

        message = f"✅ Completed {operation_type.value.replace('_', ' ').title()} in {duration:.1f}s"

        if details:
            message += f": {details}"

        console.print_system_message(message, title="Operation Complete")
