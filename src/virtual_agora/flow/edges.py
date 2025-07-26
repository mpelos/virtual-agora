"""Conditional edge implementations for the Virtual Agora discussion flow.

This module contains all the conditional logic functions that determine
state transitions and flow control in the discussion workflow.
"""

from typing import Literal
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.flow.cycle_detection import CyclePreventionManager
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class FlowConditions:
    """Container for all conditional edge logic."""

    def __init__(self):
        """Initialize flow conditions with managers."""
        self.context_manager = ContextWindowManager()
        self.cycle_manager = CyclePreventionManager()

    def should_start_discussion(
        self, state: VirtualAgoraState
    ) -> Literal["discussion", "agenda_setting"]:
        """Determine if discussion should start based on agenda approval.

        Args:
            state: Current state

        Returns:
            Next node: "discussion" if approved, "agenda_setting" if rejected
        """
        agenda_approved = state.get("agenda_approved", False)

        if agenda_approved:
            logger.info("Agenda approved, starting discussion")
            return "discussion"
        else:
            logger.info("Agenda rejected, returning to agenda setting")
            return "agenda_setting"

    def should_start_conclusion_poll(
        self, state: VirtualAgoraState
    ) -> Literal["continue_discussion", "conclusion_poll"]:
        """Determine if conclusion poll should start (after round 2).

        Args:
            state: Current state

        Returns:
            Next node: "continue_discussion" or "conclusion_poll"
        """
        current_round = state["current_round"]
        max_rounds = state["flow_control"]["max_rounds_per_topic"]

        # Start polling from round 3 onwards
        if current_round >= 3:
            logger.info(f"Round {current_round} >= 3, starting conclusion poll")
            return "conclusion_poll"
        elif current_round >= max_rounds:
            logger.info(f"Max rounds ({max_rounds}) reached, forcing conclusion poll")
            return "conclusion_poll"
        else:
            logger.info(f"Round {current_round} < 3, continuing discussion")
            return "continue_discussion"

    def evaluate_conclusion_vote(
        self, state: VirtualAgoraState
    ) -> Literal["continue_discussion", "minority_considerations"]:
        """Evaluate the conclusion vote results.

        Args:
            state: Current state

        Returns:
            Next node: "continue_discussion" if vote failed, "minority_considerations" if passed
        """
        conclusion_vote = state.get("conclusion_vote")

        if not conclusion_vote:
            logger.warning("No conclusion vote found, continuing discussion")
            return "continue_discussion"

        vote_passed = conclusion_vote["passed"]

        if vote_passed:
            logger.info("Conclusion vote passed, moving to minority considerations")
            return "minority_considerations"
        else:
            logger.info("Conclusion vote failed, continuing discussion")
            return "continue_discussion"

    def should_continue_session(
        self, state: VirtualAgoraState
    ) -> Literal["continue", "end"]:
        """Determine if session should continue based on user approval.

        Args:
            state: Current state

        Returns:
            Next node: "continue" or "end"
        """
        continue_session = state.get("continue_session", False)

        if continue_session:
            logger.info("User approved continuation")
            return "continue"
        else:
            logger.info("User requested session end")
            return "end"

    def has_topics_remaining(
        self, state: VirtualAgoraState
    ) -> Literal["next_topic", "generate_report", "re_evaluate_agenda"]:
        """Determine next action based on remaining topics.

        Args:
            state: Current state

        Returns:
            Next node based on topic queue status
        """
        topic_queue = state["topic_queue"]
        agenda_modifications = state.get("agenda_modifications")

        if not topic_queue:
            logger.info("No topics remaining, generating final report")
            return "generate_report"

        # If agenda was significantly modified, consider re-evaluation
        if agenda_modifications:
            original_count = len(state.get("proposed_topics", []))
            current_count = len(topic_queue)

            # If more than 50% change, re-evaluate
            if abs(original_count - current_count) / max(original_count, 1) > 0.5:
                logger.info("Significant agenda changes detected, re-evaluating agenda")
                return "re_evaluate_agenda"

        logger.info(f"{len(topic_queue)} topics remaining, moving to next topic")
        return "next_topic"

    def should_terminate_early(self, state: VirtualAgoraState) -> bool:
        """Check if session should terminate early due to various conditions.

        Args:
            state: Current state

        Returns:
            True if session should terminate early
        """
        flow_control = state["flow_control"]

        # Check for comprehensive cycle detection
        if flow_control["cycle_detection_enabled"]:
            cycles = self.cycle_manager.analyze_state(state)
            if self.cycle_manager.should_intervene(cycles):
                intervention = self.cycle_manager.get_intervention_strategy(cycles)
                logger.warning(
                    f"Cycles detected requiring intervention: {intervention}"
                )
                return True

        # Check for maximum phase iterations
        current_phase = state["current_phase"]
        phase_count = sum(
            1 for t in state["phase_history"] if t["to_phase"] == current_phase
        )

        if phase_count >= flow_control["max_iterations_per_phase"]:
            logger.warning(
                f"Maximum iterations ({flow_control['max_iterations_per_phase']}) reached for phase {current_phase}"
            )
            return True

        # Check for error conditions
        error_count = state.get("error_count", 0)
        if error_count >= 5:
            logger.error("Too many errors, terminating session")
            return True

        return False

    def validate_state_transition(
        self, state: VirtualAgoraState, target_phase: int
    ) -> bool:
        """Validate that a state transition is valid.

        Args:
            state: Current state
            target_phase: Target phase number

        Returns:
            True if transition is valid
        """
        current_phase = state["current_phase"]

        # Define valid transitions
        valid_transitions = {
            -1: [0],  # Start -> Initialization
            0: [1],  # Initialization -> Agenda Setting
            1: [2],  # Agenda Setting -> Discussion
            2: [3],  # Discussion -> Topic Conclusion
            3: [4, 2],  # Topic Conclusion -> Agenda Re-evaluation or back to Discussion
            4: [2, 5],  # Agenda Re-evaluation -> Discussion or Final Report
            5: [],  # Final Report -> End
        }

        allowed_targets = valid_transitions.get(current_phase, [])

        if target_phase not in allowed_targets:
            logger.error(
                f"Invalid transition from phase {current_phase} to phase {target_phase}"
            )
            return False

        return True

    def should_compress_context(self, state: VirtualAgoraState) -> bool:
        """Determine if context window needs compression.

        Args:
            state: Current state

        Returns:
            True if context should be compressed
        """
        needs_compression = self.context_manager.needs_compression(state)

        if needs_compression:
            stats = self.context_manager.get_context_stats(state)
            logger.info(
                f"Context approaching limit ({stats['total_tokens']}/{stats['limit']} tokens, "
                f"{stats['usage_percent']:.1f}%), compression needed"
            )

        return needs_compression

    def detect_voting_cycle(self, state: VirtualAgoraState) -> bool:
        """Detect if agents are stuck in a voting cycle.

        Args:
            state: Current state

        Returns:
            True if voting cycle detected
        """
        cycles = self.cycle_manager.analyze_state(state)

        # Check for voting-specific cycles
        for cycle in cycles:
            if cycle.pattern_type == "voting" and cycle.confidence > 0.6:
                logger.warning(
                    f"Voting cycle detected: {cycle.pattern_elements} "
                    f"(occurred {cycle.occurrences} times)"
                )
                return True

        return False

    def should_force_conclusion(self, state: VirtualAgoraState) -> bool:
        """Determine if topic conclusion should be forced.

        Args:
            state: Current state

        Returns:
            True if conclusion should be forced
        """
        current_topic = state.get("active_topic")
        if not current_topic:
            return False

        topic_round_count = state.get("rounds_per_topic", {}).get(current_topic, 0)
        max_rounds = state["flow_control"]["max_rounds_per_topic"]

        # Force conclusion if we've exceeded maximum rounds
        if topic_round_count >= max_rounds:
            logger.info(
                f"Forcing conclusion for topic '{current_topic}' after {topic_round_count} rounds"
            )
            return True

        # Check for consecutive conclusion attempts
        conclusion_attempts = 0
        for vote in reversed(state.get("vote_history", [])):
            if (
                vote.get("vote_type") == "continue_discussion"
                and vote.get("topic") == current_topic
            ):
                conclusion_attempts += 1
            else:
                break

        auto_conclude_threshold = state["flow_control"]["auto_conclude_threshold"]
        if conclusion_attempts >= auto_conclude_threshold:
            logger.info(
                f"Auto-concluding topic '{current_topic}' after {conclusion_attempts} failed conclusion votes"
            )
            return True

        return False
