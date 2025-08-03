"""Flow routing logic for Virtual Agora discussion flow.

This module centralizes complex routing logic that was previously scattered
throughout the graph definition. It provides:

- User approval routing logic extraction
- Topic queue validation and corruption handling
- Session continuation decision logic
- Error recovery routing strategies

This addresses Step 2.3 of the architecture refactoring by extracting
the complex routing logic from graph_v13.py into a dedicated component.
"""

from typing import Dict, Any, List, Tuple, Literal
from datetime import datetime

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.edges_v13 import V13FlowConditions
from virtual_agora.flow.interrupt_manager import get_interrupt_manager
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class FlowRouter:
    """Centralized routing logic for flow decisions.

    This class extracts complex routing logic from graph_v13.py to provide
    a centralized, testable, and maintainable routing system. It handles:

    - User approval routing with topic queue validation
    - Topic queue corruption detection and cleanup
    - Session continuation logic
    - Complex conditional routing decisions
    - Error recovery routing strategies

    The extracted logic maintains the exact behavior of the original
    implementation while providing better separation of concerns.
    """

    def __init__(self, conditions: V13FlowConditions):
        """Initialize flow router with V13 conditions.

        Args:
            conditions: V13FlowConditions instance for delegating condition evaluation
        """
        self.conditions = conditions
        self.routing_metrics = {
            "total_routing_decisions": 0,
            "user_approval_routes": 0,
            "corruption_cleanups": 0,
            "session_continuations": 0,
            "error_recoveries": 0,
        }

        logger.debug("Initialized FlowRouter")

    def evaluate_user_approval_routing(self, state: VirtualAgoraState) -> str:
        """Evaluate user approval routing with proper state validation.

        This method extracts the complex 150+ line routing logic from
        graph_v13.py:_evaluate_user_approval_routing(). It provides the exact
        same behavior with improved modularity and testability.

        Args:
            state: Current Virtual Agora state

        Returns:
            Routing decision: "end_session", "no_items", "has_items", or "modify_agenda"
        """
        self.routing_metrics["total_routing_decisions"] += 1
        self.routing_metrics["user_approval_routes"] += 1

        logger.info("=== FLOW ROUTER: User Approval Routing Evaluation START ===")

        # LOG: Complete state context for routing decision
        routing_context = {
            "user_approves_continuation": state.get("user_approves_continuation"),
            "agents_vote_end_session": state.get("agents_vote_end_session"),
            "user_forced_conclusion": state.get("user_forced_conclusion"),
            "user_requested_modification": state.get("user_requested_modification"),
            "topic_queue": state.get("topic_queue", []),
            "active_topic": state.get("active_topic"),
            "completed_topics": state.get("completed_topics", []),
            "current_phase": state.get("current_phase"),
            "current_round": state.get("current_round", 0),
        }
        logger.info(f"Routing context: {routing_context}")

        # First check explicit session continuation evaluation
        logger.debug("Checking session continuation evaluation...")
        session_decision = self.conditions.evaluate_session_continuation(state)
        logger.info(f"Session continuation decision: {session_decision}")

        if session_decision == "end_session":
            logger.info(
                "ROUTING DECISION: end_session (from session continuation evaluation)"
            )
            return "end_session"

        # Check if user explicitly requested agenda modification
        if state.get("user_requested_modification", False):
            logger.info(
                "ROUTING DECISION: modify_agenda (user explicitly requested modification)"
            )
            return "modify_agenda"

        # Validate and clean topic_queue
        topic_queue = state.get("topic_queue", [])
        logger.debug(
            f"Raw topic_queue from state: {topic_queue} (Type: {type(topic_queue)})"
        )

        # Handle topic queue corruption and validation
        cleaned_queue, corruption_detected = self.handle_topic_queue_corruption(
            topic_queue
        )

        # Handle interrupt context cleanup for completed topic
        self._handle_topic_completion_cleanup(state)

        # Make routing decision based on cleaned queue
        if not cleaned_queue:
            logger.info(
                "ROUTING DECISION: no_items (no topics remaining in cleaned queue)"
            )
            logger.info(
                f"Decision factors: cleaned_queue={cleaned_queue}, len={len(cleaned_queue)}"
            )
            return "no_items"
        else:
            # Set up interrupt context for next topic if we're continuing
            self._prepare_next_topic_context(cleaned_queue)

            logger.info(
                f"ROUTING DECISION: has_items ({len(cleaned_queue)} topics remaining)"
            )
            logger.info(f"Remaining topics: {self._truncate_topic_list(cleaned_queue)}")
            return "has_items"

    def handle_topic_queue_corruption(
        self, topic_queue: List
    ) -> Tuple[List[str], bool]:
        """Handle topic queue corruption and validation.

        This method extracts the topic queue validation logic that was
        embedded in the user approval routing. It detects nested list
        corruption and other data integrity issues.

        Args:
            topic_queue: Raw topic queue from state (may be corrupted)

        Returns:
            Tuple of (cleaned_queue, corruption_detected)
        """
        self.routing_metrics["corruption_cleanups"] += 1

        logger.debug(
            f"Raw topic_queue: {self._truncate_topic_list(topic_queue)} (Length: {len(topic_queue)})"
        )

        cleaned_queue = []
        corruption_detected = False
        corruption_details = []

        # Handle nested list corruption with detailed logging
        for i, item in enumerate(topic_queue):
            logger.debug(
                f"Processing topic_queue item [{i}]: {item} (Type: {type(item)})"
            )

            if isinstance(item, list):
                corruption_detected = True
                corruption_details.append(f"Nested list at index {i}: {item}")
                logger.warning(f"CORRUPTION: Nested list detected at index {i}: {item}")

                for j, nested_item in enumerate(item):
                    if isinstance(nested_item, str) and nested_item.strip():
                        cleaned_queue.append(nested_item.strip())
                        logger.debug(
                            f"  Extracted from nested[{j}]: '{nested_item.strip()}'"
                        )
                    else:
                        logger.warning(
                            f"  Invalid nested item ignored [{j}]: {nested_item} (Type: {type(nested_item)})"
                        )

            elif isinstance(item, str) and item.strip():
                cleaned_queue.append(item.strip())
                logger.debug(f"Valid topic added: '{item.strip()}'")
            else:
                corruption_detected = True
                corruption_details.append(
                    f"Invalid item at index {i}: {item} (Type: {type(item)})"
                )
                logger.warning(
                    f"CORRUPTION: Invalid topic_queue item ignored [{i}]: {item} (Type: {type(item)})"
                )

        # LOG: Validation results
        logger.info(f"=== TOPIC QUEUE VALIDATION RESULTS ===")
        logger.info(
            f"Original length: {len(topic_queue)} -> Cleaned length: {len(cleaned_queue)}"
        )
        logger.info(f"Corruption detected: {corruption_detected}")

        if corruption_detected:
            logger.error(f"CRITICAL: Topic queue corruptions detected:")
            for detail in corruption_details:
                logger.error(f"  - {detail}")

        logger.debug(f"Cleaned topic_queue: {cleaned_queue}")
        logger.info(f"Cleaned topic_queue: {self._truncate_topic_list(cleaned_queue)}")

        # Update state with cleaned queue if corruption was detected
        if corruption_detected and cleaned_queue != topic_queue:
            logger.error(f"CRITICAL: Topic queue corruption detected!")
            logger.error(f"  Original: {topic_queue}")
            logger.error(f"  Cleaned:  {cleaned_queue}")
            logger.error(
                "  Note: Cannot update state from routing function - this may cause issues!"
            )

        return cleaned_queue, corruption_detected

    def determine_session_continuation(self, state: VirtualAgoraState) -> str:
        """Determine if session should continue based on various factors.

        Args:
            state: Current Virtual Agora state

        Returns:
            Session continuation decision: "continue" or "end_session"
        """
        self.routing_metrics["session_continuations"] += 1

        # Delegate to conditions for session continuation logic
        return self.conditions.evaluate_session_continuation(state)

    def route_error_recovery(
        self, node_name: str, error: Exception, state: VirtualAgoraState
    ) -> str:
        """Determine routing after error recovery attempt.

        Args:
            node_name: Name of the node that failed
            error: The error that occurred
            state: Current state

        Returns:
            Recovery routing decision
        """
        self.routing_metrics["error_recoveries"] += 1

        # Define error recovery routing strategies
        if "user" in node_name.lower() and "interaction" in node_name.lower():
            # User interaction errors: continue with default action
            logger.info(
                f"User interaction error in '{node_name}': continue with defaults"
            )
            return "continue_with_defaults"

        elif "discussion" in node_name.lower():
            # Discussion errors: skip to summarization
            logger.info(f"Discussion error in '{node_name}': skip to summarization")
            return "skip_to_summary"

        elif "report" in node_name.lower() or "output" in node_name.lower():
            # Reporting errors: try alternative output
            logger.info(f"Reporting error in '{node_name}': try alternative output")
            return "alternative_output"

        else:
            # Default: continue with error flag
            logger.info(f"Generic error in '{node_name}': continue with error flag")
            return "continue_with_error"

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing metrics and statistics.

        Returns:
            Dictionary containing routing metrics
        """
        return self.routing_metrics.copy()

    def _handle_topic_completion_cleanup(self, state: VirtualAgoraState) -> None:
        """Handle interrupt context cleanup for completed topics.

        Args:
            state: Current Virtual Agora state
        """
        try:
            interrupt_manager = get_interrupt_manager()
            completed_topics = state.get("completed_topics", [])

            if completed_topics:
                # Get the most recently completed topic for cleanup
                last_completed_topic = (
                    completed_topics[-1]
                    if isinstance(completed_topics[-1], str)
                    else str(completed_topics[-1])
                )
                logger.info(f"=== TOPIC COMPLETION CLEANUP ===")
                logger.info(
                    f"Clearing interrupt context for completed topic: {last_completed_topic}"
                )
                interrupt_manager.clear_topic_context(last_completed_topic)
        except Exception as e:
            logger.warning(f"Failed to cleanup topic completion context: {e}")

    def _prepare_next_topic_context(self, cleaned_queue: List[str]) -> None:
        """Prepare interrupt context for the next topic.

        Args:
            cleaned_queue: Cleaned topic queue
        """
        try:
            if cleaned_queue:
                interrupt_manager = get_interrupt_manager()
                next_topic = cleaned_queue[0]
                logger.info(f"=== PREPARING FOR NEXT TOPIC ===")
                logger.info(f"Resetting interrupt context for next topic: {next_topic}")
                interrupt_manager.reset_for_new_topic(next_topic)
        except Exception as e:
            logger.warning(f"Failed to prepare next topic context: {e}")

    def _truncate_topic(self, topic: Any, max_length: int = 60) -> str:
        """Truncate long topic titles for cleaner logs.

        Args:
            topic: Topic to truncate
            max_length: Maximum length for truncated topic

        Returns:
            Truncated topic string
        """
        if isinstance(topic, str) and len(topic) > max_length:
            return topic[:max_length] + "..."
        return str(topic)

    def _truncate_topic_list(self, topics: List) -> List[str]:
        """Truncate a list of topics for cleaner logs.

        Args:
            topics: List of topics to truncate

        Returns:
            List of truncated topic strings
        """
        return [self._truncate_topic(topic) for topic in topics]


class ConditionalRouter:
    """Router for conditional edge logic with centralized condition evaluation.

    This class provides a centralized way to evaluate complex conditional
    logic that determines flow transitions. It wraps the V13FlowConditions
    with additional routing capabilities and error handling.
    """

    def __init__(self, conditions: V13FlowConditions):
        """Initialize conditional router.

        Args:
            conditions: V13FlowConditions instance for condition evaluation
        """
        self.conditions = conditions
        self.condition_metrics = {
            "total_evaluations": 0,
            "condition_failures": 0,
            "evaluation_times": {},
        }

        logger.debug("Initialized ConditionalRouter")

    def evaluate_condition(self, condition_name: str, state: VirtualAgoraState) -> str:
        """Evaluate a named condition with error handling and metrics.

        Args:
            condition_name: Name of the condition method to evaluate
            state: Current Virtual Agora state

        Returns:
            Condition evaluation result

        Raises:
            AttributeError: If condition method doesn't exist
            Exception: If condition evaluation fails
        """
        self.condition_metrics["total_evaluations"] += 1
        start_time = datetime.now()

        try:
            if not hasattr(self.conditions, condition_name):
                raise AttributeError(f"Condition '{condition_name}' not found")

            condition_method = getattr(self.conditions, condition_name)
            result = condition_method(state)

            # Record timing
            execution_time = (datetime.now() - start_time).total_seconds()
            if condition_name not in self.condition_metrics["evaluation_times"]:
                self.condition_metrics["evaluation_times"][condition_name] = []
            self.condition_metrics["evaluation_times"][condition_name].append(
                execution_time
            )

            logger.debug(
                f"Condition '{condition_name}' evaluated to '{result}' in {execution_time:.3f}s"
            )
            return result

        except Exception as e:
            self.condition_metrics["condition_failures"] += 1
            logger.error(f"Condition evaluation failed for '{condition_name}': {e}")
            raise

    def get_condition_metrics(self) -> Dict[str, Any]:
        """Get condition evaluation metrics.

        Returns:
            Dictionary containing condition metrics
        """
        metrics = self.condition_metrics.copy()

        # Calculate average evaluation times
        avg_times = {}
        for condition, times in self.condition_metrics["evaluation_times"].items():
            if times:
                avg_times[condition] = sum(times) / len(times)
        metrics["average_evaluation_times"] = avg_times

        return metrics


def create_flow_router(conditions: V13FlowConditions) -> FlowRouter:
    """Factory function to create a configured FlowRouter.

    Args:
        conditions: V13FlowConditions instance

    Returns:
        Configured FlowRouter instance
    """
    router = FlowRouter(conditions)
    logger.info("Created FlowRouter with V13FlowConditions")
    return router


def create_conditional_router(conditions: V13FlowConditions) -> ConditionalRouter:
    """Factory function to create a configured ConditionalRouter.

    Args:
        conditions: V13FlowConditions instance

    Returns:
        Configured ConditionalRouter instance
    """
    router = ConditionalRouter(conditions)
    logger.info("Created ConditionalRouter with V13FlowConditions")
    return router
