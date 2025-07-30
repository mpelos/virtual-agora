"""Stream coordination for Virtual Agora.

This module provides the StreamCoordinator class which handles the complex
stream lifecycle management without the premature breaks that cause session
termination in the original main.py architecture.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Iterator
from datetime import datetime

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class ContinuationResult(Enum):
    """Results of interrupt continuation processing."""

    RESUME = "resume"  # Continue with main stream
    RESTART = "restart"  # Restart stream from checkpoint
    TERMINATE = "terminate"  # Gracefully terminate session
    ERROR = "error"  # Error occurred, handle gracefully


@dataclass
class InterruptContext:
    """Context for interrupt processing."""

    interrupt_data: Any
    config_dict: Dict[str, Any]
    session_id: str
    timestamp: datetime


class StreamCoordinator:
    """Handles stream lifecycle management without premature termination.

    This class replaces the complex nested stream logic in main.py that
    caused premature session termination due to incorrect break statements.
    The key insight is that after handling interrupts, we should RESUME
    the main stream rather than breaking out of it.
    """

    def __init__(self, flow_instance, interrupt_processor: Callable):
        """Initialize the stream coordinator.

        Args:
            flow_instance: The VirtualAgoraV13Flow instance
            interrupt_processor: Function to process interrupts (from main.py)
        """
        self.flow = flow_instance
        self.process_interrupt = interrupt_processor
        self._active_streams = 0
        self._stream_stack = []

        logger.info("StreamCoordinator initialized")

    def coordinate_stream_execution(
        self, config_dict: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """Coordinate stream execution with proper interrupt handling.

        This method replaces the problematic nested stream logic in main.py
        and ensures that interrupts are handled without breaking the main
        execution flow.

        Args:
            config_dict: Configuration for LangGraph execution

        Yields:
            Dict[str, Any]: Stream updates from the flow
        """
        logger.info("Starting coordinated stream execution")
        self._active_streams += 1

        try:
            # Start the main stream
            stream_id = f"main_stream_{self._active_streams}"
            self._stream_stack.append(stream_id)
            logger.debug(f"Starting stream: {stream_id}")

            # Track if we need to resume from checkpoint
            resume_from_checkpoint = False

            while True:  # Continue until natural completion or termination
                # Start or resume the stream
                if resume_from_checkpoint:
                    logger.info(
                        "Resuming stream from checkpoint after interrupt processing"
                    )
                    stream_iter = self.flow.stream(
                        config_dict, resume_from_checkpoint=True
                    )
                else:
                    logger.debug("Starting fresh stream execution")
                    stream_iter = self.flow.stream(config_dict)

                # Reset resume flag for next iteration
                stream_completed = False
                resume_from_checkpoint = False

                for update in stream_iter:
                    logger.debug(
                        f"Stream update received: {list(update.keys()) if isinstance(update, dict) else type(update)}"
                    )

                    # Handle interrupts without breaking the main stream
                    if "__interrupt__" in update:
                        result = self._handle_interrupt_coordinated(
                            update["__interrupt__"], config_dict
                        )

                        if result == ContinuationResult.TERMINATE:
                            logger.info("Interrupt handling requested termination")
                            return
                        elif result == ContinuationResult.ERROR:
                            logger.error("Error during interrupt handling")
                            return
                        elif result == ContinuationResult.RESUME:
                            # After interrupt processing, we need to resume from checkpoint
                            logger.info(
                                "Interrupt processed, will resume stream from checkpoint"
                            )
                            resume_from_checkpoint = True
                            break  # Break inner loop to restart stream with resume_from_checkpoint=True

                    else:
                        # Yield non-interrupt updates to the caller
                        yield update

                        # Check for natural completion
                        if "__end__" in update:
                            logger.info(f"Stream {stream_id} completed naturally")
                            stream_completed = True
                            break

                # If stream completed naturally, exit the main loop
                if stream_completed:
                    break

                # If we reach here without setting resume_from_checkpoint,
                # it means the stream ended without __end__ (which shouldn't happen)
                if not resume_from_checkpoint:
                    logger.warning(
                        "Stream ended without __end__ marker and without interrupt"
                    )
                    break

            logger.debug(f"Stream {stream_id} coordination completed")

        except Exception as e:
            logger.error(f"Error in stream coordination: {e}", exc_info=True)
            raise

        finally:
            if self._stream_stack and self._stream_stack[-1] == stream_id:
                self._stream_stack.pop()
            self._active_streams = max(0, self._active_streams - 1)
            logger.debug(f"Stream coordination cleanup completed")

    def _handle_interrupt_coordinated(
        self, interrupt_data: Any, config_dict: Dict[str, Any]
    ) -> ContinuationResult:
        """Handle interrupts in a coordinated manner without breaking main stream.

        This is the key method that fixes the premature termination bug.
        Instead of breaking the main stream after handling interrupts,
        we properly coordinate the interrupt handling and allow the main
        stream to continue.

        Args:
            interrupt_data: Interrupt data from LangGraph
            config_dict: Configuration dictionary

        Returns:
            ContinuationResult: How to proceed after interrupt handling
        """
        logger.info(
            "=== STREAM COORDINATOR: Handling interrupt in coordinated manner ==="
        )

        # Extract interrupt type for debugging
        interrupt_type = "unknown"
        if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
            interrupt_obj = interrupt_data[0]
            if hasattr(interrupt_obj, "value") and isinstance(
                interrupt_obj.value, dict
            ):
                interrupt_type = interrupt_obj.value.get("type", "unknown")

        logger.info(f"Interrupt type: {interrupt_type}")
        logger.debug(f"Interrupt data structure: {type(interrupt_data)}")

        try:
            # Create interrupt context
            context = InterruptContext(
                interrupt_data=interrupt_data,
                config_dict=config_dict,
                session_id=config_dict.get("configurable", {}).get(
                    "thread_id", "unknown"
                ),
                timestamp=datetime.now(),
            )

            logger.info(
                "=== STREAM COORDINATOR: Processing interrupt with interrupt processor ==="
            )
            # Process the interrupt using the existing interrupt processor
            user_response = self.process_interrupt(
                interrupt_data,
                config_dict,
                depth=0,  # Managed by InterruptStackManager
                stream_depth=0,  # Managed by InterruptStackManager
            )

            if not user_response:
                logger.warning("No user response received from interrupt processing")
                logger.info(
                    "=== STREAM COORDINATOR: Will resume stream without state update ==="
                )
                return ContinuationResult.RESUME

            logger.info(
                f"=== STREAM COORDINATOR: User response received for {interrupt_type}: {list(user_response.keys())} ==="
            )
            logger.debug(f"User response content: {user_response}")

            # Update the flow state with the user response
            logger.info(
                "=== STREAM COORDINATOR: Updating flow state with user response ==="
            )
            self._update_flow_state_coordinated(
                interrupt_data, config_dict, user_response
            )

            logger.info(
                f"=== STREAM COORDINATOR: Interrupt {interrupt_type} handled successfully, will resume stream from checkpoint ==="
            )
            return ContinuationResult.RESUME

        except Exception as e:
            logger.error(
                f"=== STREAM COORDINATOR: Error handling interrupt {interrupt_type}: {e} ===",
                exc_info=True,
            )
            return ContinuationResult.ERROR

    def _update_flow_state_coordinated(
        self,
        interrupt_data: Any,
        config_dict: Dict[str, Any],
        user_response: Dict[str, Any],
    ) -> None:
        """Update flow state in a coordinated manner.

        This method handles the state update without the complex nested
        stream logic that caused issues in the original implementation.

        Args:
            interrupt_data: Original interrupt data
            config_dict: Configuration dictionary
            user_response: User's response to the interrupt
        """
        logger.info(
            "=== STREAM COORDINATOR: Updating flow state after interrupt handling ==="
        )
        logger.debug(f"User response keys: {list(user_response.keys())}")
        logger.debug(f"Config dict: {config_dict}")

        try:
            # Extract node information for proper state update
            node_name = self._extract_node_name(interrupt_data)

            if node_name:
                logger.info(
                    f"=== STREAM COORDINATOR: Updating state for specific node: {node_name} ==="
                )
                logger.debug(f"Calling update_state with as_node={node_name}")
                self.flow.compiled_graph.update_state(
                    config_dict, user_response, as_node=node_name
                )
            else:
                logger.info(
                    "=== STREAM COORDINATOR: Updating state without specific node ==="
                )
                logger.debug("Calling update_state without as_node parameter")
                self.flow.compiled_graph.update_state(config_dict, user_response)

            logger.info(
                "=== STREAM COORDINATOR: Flow state updated successfully, stream can now resume ==="
            )

            # Get current state for debugging
            try:
                current_state = self.flow.compiled_graph.get_state(config_dict)
                logger.debug(
                    f"State after update - Current values: agenda_approved={current_state.values.get('agenda_approved')}"
                )
                logger.debug(f"State after update - Next nodes: {current_state.next}")
            except Exception as state_debug_error:
                logger.debug(f"Could not get state for debugging: {state_debug_error}")

        except Exception as e:
            logger.error(
                f"=== STREAM COORDINATOR: Failed to update flow state: {e} ===",
                exc_info=True,
            )
            raise

    def _extract_node_name(self, interrupt_data: Any) -> Optional[str]:
        """Extract node name from interrupt data.

        Args:
            interrupt_data: Interrupt data from LangGraph

        Returns:
            Node name if extractable, None otherwise
        """
        try:
            if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
                interrupt_obj = interrupt_data[0]
                if hasattr(interrupt_obj, "ns") and interrupt_obj.ns:
                    node_name = (
                        interrupt_obj.ns[0].split(":")[0]
                        if ":" in interrupt_obj.ns[0]
                        else interrupt_obj.ns[0]
                    )
                    return node_name
        except Exception as e:
            logger.debug(f"Could not extract node name from interrupt data: {e}")

        return None

    def get_active_stream_count(self) -> int:
        """Get the number of currently active streams.

        Returns:
            Number of active streams
        """
        return self._active_streams

    def get_stream_stack(self) -> list:
        """Get the current stream stack for debugging.

        Returns:
            List of active stream IDs
        """
        return self._stream_stack.copy()

    def is_coordinating(self) -> bool:
        """Check if the coordinator is currently managing streams.

        Returns:
            True if streams are active, False otherwise
        """
        return self._active_streams > 0


class StreamHealth:
    """Monitor stream health and detect issues."""

    def __init__(self):
        self.stream_starts = 0
        self.stream_completions = 0
        self.interrupt_count = 0
        self.error_count = 0
        self.start_time = datetime.now()

    def record_stream_start(self):
        """Record a stream start."""
        self.stream_starts += 1

    def record_stream_completion(self):
        """Record a stream completion."""
        self.stream_completions += 1

    def record_interrupt(self):
        """Record an interrupt."""
        self.interrupt_count += 1

    def record_error(self):
        """Record an error."""
        self.error_count += 1

    def get_health_report(self) -> Dict[str, Any]:
        """Get a health report for debugging.

        Returns:
            Dict containing health metrics
        """
        runtime = datetime.now() - self.start_time

        return {
            "runtime_seconds": runtime.total_seconds(),
            "streams_started": self.stream_starts,
            "streams_completed": self.stream_completions,
            "streams_active": self.stream_starts - self.stream_completions,
            "interrupts_handled": self.interrupt_count,
            "errors_encountered": self.error_count,
            "completion_rate": self.stream_completions / max(1, self.stream_starts),
            "health_status": "healthy" if self.error_count == 0 else "degraded",
        }
