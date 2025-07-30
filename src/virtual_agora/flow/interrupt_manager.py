"""Interrupt Stack Manager for Virtual Agora.

This module provides comprehensive interrupt context management with proper
stack operations, depth tracking, and topic isolation to fix the interrupt
depth accumulation bug.
"""

from typing import Optional, Dict, Any, List, NamedTuple
from datetime import datetime
from dataclasses import dataclass, field
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InterruptContext:
    """Context information for a single interrupt."""

    interrupt_id: str
    interrupt_type: str
    timestamp: datetime
    topic_name: Optional[str] = None
    session_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    parent_context_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "interrupt_id": self.interrupt_id,
            "interrupt_type": self.interrupt_type,
            "timestamp": self.timestamp.isoformat(),
            "topic_name": self.topic_name,
            "session_id": self.session_id,
            "depth": self.depth,
            "parent_context_id": self.parent_context_id,
            "has_payload": bool(self.payload),
        }


class InterruptStackManager:
    """Manages interrupt contexts with proper stack operations and depth tracking.

    This class solves the interrupt depth accumulation bug by:
    1. Properly tracking interrupt contexts with push/pop operations
    2. Maintaining correct depth calculations
    3. Isolating interrupt contexts between topics
    4. Providing cleanup mechanisms for proper state management
    """

    def __init__(self):
        """Initialize the interrupt stack manager."""
        self._interrupt_stack: List[InterruptContext] = []
        self._topic_contexts: Dict[str, List[InterruptContext]] = {}
        self._session_id: Optional[str] = None
        self._current_topic: Optional[str] = None
        self._interrupt_counter = 0

        logger.debug("InterruptStackManager initialized")

    def set_session_context(self, session_id: str, current_topic: Optional[str] = None):
        """Set the current session and topic context.

        Args:
            session_id: Current session identifier
            current_topic: Current topic being discussed (optional)
        """
        self._session_id = session_id
        self._current_topic = current_topic
        logger.debug(
            f"Session context set: session_id={session_id}, topic={current_topic}"
        )

    def push_interrupt(
        self,
        interrupt_type: str,
        payload: Dict[str, Any],
        topic_name: Optional[str] = None,
    ) -> InterruptContext:
        """Push a new interrupt context onto the stack.

        Args:
            interrupt_type: Type of interrupt (e.g., 'topic_continuation', 'agenda_approval')
            payload: Interrupt payload data
            topic_name: Associated topic name (defaults to current topic)

        Returns:
            The created interrupt context
        """
        self._interrupt_counter += 1
        interrupt_id = f"interrupt_{self._interrupt_counter:04d}"

        # Use provided topic or fall back to current topic
        topic = topic_name or self._current_topic

        # Calculate depth - for topic-level interrupts, always use depth 1
        # Only true nested interrupts within same topic should increment depth
        if interrupt_type == "topic_continuation":
            # Topic continuation interrupts should always be at depth 1
            # This fixes the accumulating depth bug
            depth = 1
        else:
            # For other interrupt types, calculate based on current stack
            depth = len(self._interrupt_stack) + 1

        # Create interrupt context
        parent_id = (
            self._interrupt_stack[-1].interrupt_id if self._interrupt_stack else None
        )
        context = InterruptContext(
            interrupt_id=interrupt_id,
            interrupt_type=interrupt_type,
            timestamp=datetime.now(),
            topic_name=topic,
            session_id=self._session_id,
            payload=payload.copy(),
            depth=depth,
            parent_context_id=parent_id,
        )

        # Push to stack and track by topic
        self._interrupt_stack.append(context)
        if topic:
            if topic not in self._topic_contexts:
                self._topic_contexts[topic] = []
            self._topic_contexts[topic].append(context)

        logger.info(f"=== INTERRUPT PUSHED ===")
        logger.info(
            f"Interrupt: {interrupt_id}, Type: {interrupt_type}, Depth: {depth}"
        )
        logger.info(f"Topic: {topic}, Stack size: {len(self._interrupt_stack)}")
        logger.debug(f"Context: {context.to_dict()}")

        return context

    def pop_interrupt(self) -> Optional[InterruptContext]:
        """Pop the most recent interrupt context from the stack.

        Returns:
            The popped interrupt context, or None if stack is empty
        """
        if not self._interrupt_stack:
            logger.warning("Attempted to pop interrupt from empty stack")
            return None

        context = self._interrupt_stack.pop()

        logger.info(f"=== INTERRUPT POPPED ===")
        logger.info(
            f"Interrupt: {context.interrupt_id}, Type: {context.interrupt_type}"
        )
        logger.info(
            f"Topic: {context.topic_name}, Stack size: {len(self._interrupt_stack)}"
        )

        return context

    def get_current_depth(self) -> int:
        """Get the current interrupt depth.

        Returns:
            Current interrupt depth (0 if no interrupts)
        """
        if not self._interrupt_stack:
            return 0
        return self._interrupt_stack[-1].depth

    def get_current_context(self) -> Optional[InterruptContext]:
        """Get the current interrupt context without popping.

        Returns:
            Current interrupt context, or None if stack is empty
        """
        return self._interrupt_stack[-1] if self._interrupt_stack else None

    def clear_topic_context(self, topic_name: str):
        """Clear all interrupt contexts for a specific topic.

        This should be called when a topic is completed to ensure
        proper cleanup and prevent depth accumulation.

        Args:
            topic_name: Name of the topic to clear
        """
        try:
            # Remove topic-specific contexts
            if topic_name in self._topic_contexts:
                topic_contexts = self._topic_contexts[topic_name]
                logger.info(f"=== CLEARING TOPIC CONTEXT: {topic_name} ===")
                logger.info(
                    f"Clearing {len(topic_contexts)} interrupt contexts for topic"
                )

                # Store original stack size for validation
                original_stack_size = len(self._interrupt_stack)

                # Remove topic contexts from main stack
                self._interrupt_stack = [
                    ctx for ctx in self._interrupt_stack if ctx.topic_name != topic_name
                ]

                # Clear topic mapping
                del self._topic_contexts[topic_name]

                # Validate cleanup was successful
                removed_count = original_stack_size - len(self._interrupt_stack)
                expected_count = len(topic_contexts)

                if removed_count != expected_count:
                    logger.warning(
                        f"Stack cleanup mismatch: removed {removed_count}, expected {expected_count}"
                    )
                    logger.warning(
                        "This may indicate stack corruption or context leakage"
                    )

                logger.info(
                    f"Topic context cleared. Stack size: {original_stack_size} -> {len(self._interrupt_stack)}"
                )
            else:
                logger.debug(f"No contexts to clear for topic: {topic_name}")

        except Exception as e:
            logger.error(f"Error clearing topic context for {topic_name}: {e}")
            logger.error(f"Stack state: {self.get_stack_info()}")
            # Don't raise the exception to prevent cascading failures
            # but log it for debugging

    def reset_for_new_topic(self, new_topic: str):
        """Reset interrupt context for a new topic.

        This ensures that each topic starts with a clean interrupt context,
        preventing depth accumulation between topics.

        Args:
            new_topic: Name of the new topic starting
        """
        logger.info(f"=== RESETTING FOR NEW TOPIC: {new_topic} ===")

        # Clear any remaining contexts from previous topic
        if self._current_topic and self._current_topic != new_topic:
            self.clear_topic_context(self._current_topic)

        # Update current topic
        self._current_topic = new_topic

        # If stack is not empty, warn about potential issues
        if self._interrupt_stack:
            logger.warning(
                f"Interrupt stack not empty when starting new topic: {len(self._interrupt_stack)} contexts remaining"
            )
            logger.warning("This may indicate incomplete interrupt cleanup")

            # Log remaining contexts for debugging
            for ctx in self._interrupt_stack:
                logger.warning(
                    f"  Remaining context: {ctx.interrupt_id} ({ctx.interrupt_type}) for topic {ctx.topic_name}"
                )

        logger.info(f"Ready for new topic: {new_topic}")

    def get_stack_info(self) -> Dict[str, Any]:
        """Get detailed information about the current interrupt stack.

        Returns:
            Dictionary with stack information for debugging
        """
        return {
            "stack_size": len(self._interrupt_stack),
            "current_depth": self.get_current_depth(),
            "session_id": self._session_id,
            "current_topic": self._current_topic,
            "topics_tracked": list(self._topic_contexts.keys()),
            "stack_contexts": [ctx.to_dict() for ctx in self._interrupt_stack],
            "topic_context_counts": {
                topic: len(contexts) for topic, contexts in self._topic_contexts.items()
            },
        }

    def validate_stack_integrity(self) -> List[str]:
        """Validate the integrity of the interrupt stack.

        Returns:
            List of warning messages about potential issues
        """
        warnings = []

        # Check for excessive depth
        current_depth = self.get_current_depth()
        if current_depth > 5:
            warnings.append(f"Excessive interrupt depth: {current_depth}")

        # Check for orphaned contexts
        for topic, contexts in self._topic_contexts.items():
            stack_contexts_for_topic = [
                ctx for ctx in self._interrupt_stack if ctx.topic_name == topic
            ]
            if len(contexts) != len(stack_contexts_for_topic):
                warnings.append(
                    f"Context mismatch for topic {topic}: {len(contexts)} tracked, {len(stack_contexts_for_topic)} in stack"
                )

        # Check for very old contexts
        now = datetime.now()
        for ctx in self._interrupt_stack:
            age_minutes = (now - ctx.timestamp).total_seconds() / 60
            if age_minutes > 30:  # 30 minutes
                warnings.append(
                    f"Very old interrupt context: {ctx.interrupt_id} ({age_minutes:.1f} minutes old)"
                )

        return warnings

    def emergency_recovery(self) -> Dict[str, Any]:
        """Perform emergency recovery of the interrupt stack.

        This method should be called when interrupt stack corruption is detected
        or when the system is in an inconsistent state.

        Returns:
            Dictionary with recovery information and actions taken
        """
        logger.error("=== INTERRUPT STACK EMERGENCY RECOVERY ===")

        recovery_info = {
            "timestamp": datetime.now().isoformat(),
            "original_stack_size": len(self._interrupt_stack),
            "original_topic_contexts": len(self._topic_contexts),
            "actions_taken": [],
            "recovered_successfully": False,
        }

        try:
            # Log current state for debugging
            logger.error(f"Stack state before recovery: {self.get_stack_info()}")

            # Validate integrity and collect warnings
            warnings = self.validate_stack_integrity()
            if warnings:
                logger.error(f"Integrity issues found: {warnings}")
                recovery_info["integrity_warnings"] = warnings

            # Action 1: Remove very old contexts (older than 1 hour)
            now = datetime.now()
            old_contexts = []
            cleaned_stack = []

            for ctx in self._interrupt_stack:
                age_minutes = (now - ctx.timestamp).total_seconds() / 60
                if age_minutes > 60:  # 1 hour
                    old_contexts.append(ctx.interrupt_id)
                else:
                    cleaned_stack.append(ctx)

            if old_contexts:
                self._interrupt_stack = cleaned_stack
                recovery_info["actions_taken"].append(
                    f"Removed {len(old_contexts)} old contexts: {old_contexts}"
                )
                logger.info(f"Removed {len(old_contexts)} old interrupt contexts")

            # Action 2: Rebuild topic context mapping from current stack
            new_topic_contexts = {}
            for ctx in self._interrupt_stack:
                if ctx.topic_name:
                    if ctx.topic_name not in new_topic_contexts:
                        new_topic_contexts[ctx.topic_name] = []
                    new_topic_contexts[ctx.topic_name].append(ctx)

            inconsistent_topics = set(self._topic_contexts.keys()) - set(
                new_topic_contexts.keys()
            )
            if inconsistent_topics:
                recovery_info["actions_taken"].append(
                    f"Rebuilt topic contexts, removed inconsistent topics: {list(inconsistent_topics)}"
                )
                logger.info(
                    f"Rebuilt topic context mapping, removed inconsistent topics: {inconsistent_topics}"
                )

            self._topic_contexts = new_topic_contexts

            # Action 3: Reset depths to prevent accumulation
            for i, ctx in enumerate(self._interrupt_stack):
                if ctx.interrupt_type == "topic_continuation":
                    # Topic continuations should always be at depth 1
                    if ctx.depth != 1:
                        old_depth = ctx.depth
                        ctx.depth = 1
                        recovery_info["actions_taken"].append(
                            f"Reset depth for {ctx.interrupt_id}: {old_depth} -> 1"
                        )
                        logger.info(
                            f"Reset depth for topic continuation {ctx.interrupt_id}: {old_depth} -> 1"
                        )

            # Final validation
            final_warnings = self.validate_stack_integrity()
            recovery_info["final_stack_size"] = len(self._interrupt_stack)
            recovery_info["final_topic_contexts"] = len(self._topic_contexts)
            recovery_info["final_warnings"] = final_warnings
            recovery_info["recovered_successfully"] = (
                len(final_warnings) < len(warnings) if warnings else True
            )

            logger.info("=== RECOVERY COMPLETED ===")
            logger.info(
                f"Stack size: {recovery_info['original_stack_size']} -> {recovery_info['final_stack_size']}"
            )
            logger.info(
                f"Topic contexts: {recovery_info['original_topic_contexts']} -> {recovery_info['final_topic_contexts']}"
            )
            logger.info(f"Actions taken: {len(recovery_info['actions_taken'])}")

            if recovery_info["recovered_successfully"]:
                logger.info("✅ Recovery successful")
            else:
                logger.warning("⚠️ Recovery partially successful, some issues remain")

        except Exception as e:
            logger.error(f"Emergency recovery failed: {e}")
            recovery_info["recovery_error"] = str(e)
            recovery_info["recovered_successfully"] = False

        return recovery_info


# Global interrupt manager instance
_interrupt_manager: Optional[InterruptStackManager] = None


def get_interrupt_manager() -> InterruptStackManager:
    """Get the global interrupt manager instance.

    Returns:
        The global InterruptStackManager instance
    """
    global _interrupt_manager
    if _interrupt_manager is None:
        _interrupt_manager = InterruptStackManager()
    return _interrupt_manager


def reset_interrupt_manager():
    """Reset the global interrupt manager instance.

    This is primarily for testing purposes.
    """
    global _interrupt_manager
    _interrupt_manager = None
