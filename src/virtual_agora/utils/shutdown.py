"""Graceful shutdown handler for Virtual Agora.

This module provides functionality for clean application shutdown,
including signal handling, resource cleanup, and state preservation.
"""

import asyncio
import atexit
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.error_handler import error_handler
from virtual_agora.utils.error_reporter import ErrorReporter
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager


logger = get_logger(__name__)


class ShutdownHandler:
    """Manages graceful shutdown of the application."""
    
    def __init__(
        self,
        console: Optional[Console] = None,
        timeout: float = 30.0,
    ):
        """Initialize shutdown handler.
        
        Args:
            console: Rich console for output
            timeout: Maximum time to wait for shutdown
        """
        self.console = console or Console()
        self.timeout = timeout
        
        self._shutdown_requested = False
        self._shutdown_in_progress = False
        self._cleanup_tasks: List[Callable[[], None]] = []
        self._async_cleanup_tasks: List[Callable[[], Any]] = []
        self._resource_locks: Set[str] = set()
        self._shutdown_event = threading.Event()
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register atexit handler
        atexit.register(self._atexit_handler)
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            # Unix-like systems
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGHUP, self._signal_handler)
        else:
            # Windows
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT",
        }
        
        if sys.platform != "win32":
            signal_names[signal.SIGHUP] = "SIGHUP"
        
        signal_name = signal_names.get(signum, f"Signal {signum}")
        
        logger.info(f"Received {signal_name}, initiating graceful shutdown")
        
        # Request shutdown
        self.request_shutdown(f"Received {signal_name}")
    
    def _atexit_handler(self) -> None:
        """Handle application exit."""
        if not self._shutdown_in_progress:
            logger.debug("Atexit handler triggered")
            self._perform_shutdown("Application exit")
    
    def register_cleanup_task(
        self,
        task: Callable[[], None],
        name: Optional[str] = None,
    ) -> None:
        """Register a cleanup task to run during shutdown.
        
        Args:
            task: Cleanup function
            name: Optional task name for logging
        """
        if name:
            # Wrap task with logging
            def logged_task():
                logger.debug(f"Running cleanup task: {name}")
                task()
            self._cleanup_tasks.append(logged_task)
        else:
            self._cleanup_tasks.append(task)
    
    def register_async_cleanup_task(
        self,
        task: Callable[[], Any],
        name: Optional[str] = None,
    ) -> None:
        """Register an async cleanup task to run during shutdown.
        
        Args:
            task: Async cleanup function
            name: Optional task name for logging
        """
        if name:
            # Wrap task with logging
            async def logged_task():
                logger.debug(f"Running async cleanup task: {name}")
                await task()
            self._async_cleanup_tasks.append(logged_task)
        else:
            self._async_cleanup_tasks.append(task)
    
    def acquire_resource_lock(self, resource_name: str) -> bool:
        """Acquire a lock on a resource to prevent shutdown.
        
        Args:
            resource_name: Name of resource
            
        Returns:
            True if lock acquired, False if shutdown in progress
        """
        if self._shutdown_in_progress:
            return False
        
        self._resource_locks.add(resource_name)
        logger.debug(f"Acquired resource lock: {resource_name}")
        return True
    
    def release_resource_lock(self, resource_name: str) -> None:
        """Release a resource lock.
        
        Args:
            resource_name: Name of resource
        """
        self._resource_locks.discard(resource_name)
        logger.debug(f"Released resource lock: {resource_name}")
    
    @contextmanager
    def resource_lock(self, resource_name: str):
        """Context manager for resource locks.
        
        Args:
            resource_name: Name of resource
            
        Yields:
            None
            
        Raises:
            RuntimeError: If shutdown is in progress
        """
        if not self.acquire_resource_lock(resource_name):
            raise RuntimeError("Cannot acquire lock, shutdown in progress")
        
        try:
            yield
        finally:
            self.release_resource_lock(resource_name)
    
    def request_shutdown(self, reason: str = "User request") -> None:
        """Request graceful shutdown.
        
        Args:
            reason: Reason for shutdown
        """
        if self._shutdown_requested:
            logger.debug("Shutdown already requested")
            return
        
        self._shutdown_requested = True
        self._shutdown_event.set()
        
        logger.info(f"Shutdown requested: {reason}")
        
        # Perform shutdown
        self._perform_shutdown(reason)
    
    def _perform_shutdown(self, reason: str) -> None:
        """Perform the actual shutdown.
        
        Args:
            reason: Reason for shutdown
        """
        if self._shutdown_in_progress:
            logger.debug("Shutdown already in progress")
            return
        
        self._shutdown_in_progress = True
        
        # Show shutdown message
        self.console.print(
            f"\n[yellow]Shutting down: {reason}[/yellow]",
            style="bold",
        )
        
        # Wait for resource locks with progress
        if self._resource_locks:
            self._wait_for_resources()
        
        # Run cleanup tasks
        self._run_cleanup_tasks()
        
        # Show completion
        self.console.print(
            "[green]✅ Shutdown complete[/green]",
            style="bold",
        )
    
    def _wait_for_resources(self) -> None:
        """Wait for resource locks to be released."""
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Waiting for resources to finish...",
                total=None,
            )
            
            while self._resource_locks:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout:
                    logger.warning(
                        f"Timeout waiting for resources: {self._resource_locks}"
                    )
                    break
                
                # Update progress
                remaining = list(self._resource_locks)
                progress.update(
                    task,
                    description=f"Waiting for {len(remaining)} resources..."
                )
                
                time.sleep(0.1)
    
    def _run_cleanup_tasks(self) -> None:
        """Run all registered cleanup tasks."""
        if not self._cleanup_tasks and not self._async_cleanup_tasks:
            return
        
        total_tasks = len(self._cleanup_tasks) + len(self._async_cleanup_tasks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Running {total_tasks} cleanup tasks...",
                total=total_tasks,
            )
            
            # Run sync cleanup tasks
            for cleanup_task in self._cleanup_tasks:
                try:
                    cleanup_task()
                    progress.advance(task)
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
            
            # Run async cleanup tasks
            if self._async_cleanup_tasks:
                # Create new event loop if needed
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run async tasks
                async def run_async_tasks():
                    for cleanup_task in self._async_cleanup_tasks:
                        try:
                            await cleanup_task()
                            progress.advance(task)
                        except Exception as e:
                            logger.error(f"Error during async cleanup: {e}")
                
                loop.run_until_complete(run_async_tasks())
    
    def save_session_state(
        self,
        state_manager: Optional[StateManager] = None,
        recovery_manager: Optional[StateRecoveryManager] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Save current session state before shutdown.
        
        Args:
            state_manager: State manager instance
            recovery_manager: Recovery manager instance
            session_id: Session identifier
            
        Returns:
            Path to saved state file or None
        """
        if not state_manager or not state_manager._state:
            logger.debug("No state to save")
            return None
        
        try:
            # Create checkpoint
            if recovery_manager:
                checkpoint = recovery_manager.create_checkpoint(
                    state_manager.state,
                    operation="shutdown",
                    save_to_disk=True,
                )
                logger.info(f"Created shutdown checkpoint: {checkpoint.checkpoint_id}")
            
            # Export session
            session_data = state_manager.export_session()
            
            # Save to file
            if not session_id:
                session_id = session_data.get("session_id", "unknown")
            
            output_dir = Path("sessions")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_id}_{timestamp}_shutdown.json"
            filepath = output_dir / filename
            
            import json
            with open(filepath, "w") as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Saved session state to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return None
    
    def generate_shutdown_report(
        self,
        error_reporter: Optional[ErrorReporter] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Generate final shutdown report.
        
        Args:
            error_reporter: Error reporter instance
            session_id: Session identifier
            
        Returns:
            Path to report file or None
        """
        try:
            # Get error summary
            error_summary = error_handler.get_error_summary()
            
            # Save error report if reporter available
            error_report_path = None
            if error_reporter and session_id:
                error_report_path = error_reporter.save_error_report(session_id)
            
            # Create shutdown report
            report = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "shutdown_reason": "graceful_shutdown",
                "error_summary": error_summary,
                "error_report": str(error_report_path) if error_report_path else None,
            }
            
            # Save report
            output_dir = Path("logs")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shutdown_report_{timestamp}.json"
            filepath = output_dir / filename
            
            import json
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Generated shutdown report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate shutdown report: {e}")
            return None
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested.
        
        Returns:
            True if shutdown requested
        """
        return self._shutdown_requested
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to be requested.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if shutdown was requested
        """
        return self._shutdown_event.wait(timeout)


# Global shutdown handler instance
shutdown_handler = ShutdownHandler()


@contextmanager
def graceful_shutdown_context(
    state_manager: Optional[StateManager] = None,
    recovery_manager: Optional[StateRecoveryManager] = None,
    error_reporter: Optional[ErrorReporter] = None,
    session_id: Optional[str] = None,
):
    """Context manager for graceful shutdown handling.
    
    Args:
        state_manager: State manager instance
        recovery_manager: Recovery manager instance
        error_reporter: Error reporter instance
        session_id: Session identifier
        
    Yields:
        Shutdown handler instance
    """
    # Register cleanup tasks
    if state_manager:
        shutdown_handler.register_cleanup_task(
            lambda: shutdown_handler.save_session_state(
                state_manager,
                recovery_manager,
                session_id,
            ),
            name="save_session_state",
        )
    
    if error_reporter:
        shutdown_handler.register_cleanup_task(
            lambda: shutdown_handler.generate_shutdown_report(
                error_reporter,
                session_id,
            ),
            name="generate_shutdown_report",
        )
    
    try:
        yield shutdown_handler
    finally:
        # Ensure cleanup runs
        if not shutdown_handler._shutdown_in_progress:
            shutdown_handler.request_shutdown("Context exit")