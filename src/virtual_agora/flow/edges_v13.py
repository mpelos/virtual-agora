"""Conditional edge implementations for Virtual Agora v1.3 discussion flow.

This module contains all the conditional logic functions that determine
state transitions and flow control in the node-centric architecture.
"""

from typing import Literal
from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.context_window import ContextWindowManager
from virtual_agora.flow.cycle_detection import CyclePreventionManager
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class V13FlowConditions:
    """Container for all v1.3 conditional edge logic.

    The v1.3 architecture includes enhanced conditions:
    - Periodic 5-round user stops
    - Dual polling system (agent votes + user override)
    - Agent session end voting
    - More complex phase transitions
    """

    def __init__(self):
        """Initialize flow conditions with managers."""
        self.context_manager = ContextWindowManager()
        self.cycle_manager = CyclePreventionManager()

    # ===== Phase 1 Conditions =====

    def should_start_discussion(
        self, state: VirtualAgoraState
    ) -> Literal["discussion", "agenda_setting"]:
        """Determine if discussion should start based on agenda approval.

        Args:
            state: Current state

        Returns:
            Next node: "discussion_round" if approved, "agenda_setting" if rejected
        """
        agenda_approved = state.get("agenda_approved", False)

        if agenda_approved:
            logger.info("Agenda approved, starting discussion")
            return "discussion"
        else:
            logger.info("Agenda rejected, returning to agenda setting")
            return "agenda_setting"

    # ===== Phase 2 Conditions =====

    def check_round_threshold(
        self, state: VirtualAgoraState
    ) -> Literal["start_polling", "continue_discussion"]:
        """Determine if polling should start (round >= 3).

        Args:
            state: Current state

        Returns:
            Next node based on round number
        """
        current_round = state.get("current_round", 0)

        if current_round >= 3:
            logger.info(
                f"Round {current_round} >= 3, enabling topic conclusion polling"
            )
            return "start_polling"
        else:
            logger.info(
                f"Round {current_round} < 3, continuing discussion without poll"
            )
            return "continue_discussion"

    def check_periodic_stop(
        self, state: VirtualAgoraState
    ) -> Literal["periodic_stop", "check_votes"]:
        """Check if it's time for 5-round user stop.

        New in v1.3 - gives user periodic control.

        Args:
            state: Current state

        Returns:
            Next node based on round number
        """
        current_round = state.get("current_round", 0)

        # Check if this is a 5-round interval
        if current_round % 5 == 0 and current_round > 0:
            logger.info(
                f"Round {current_round} is multiple of 5, triggering periodic user stop"
            )
            return "periodic_stop"
        else:
            return "check_votes"

    def evaluate_conclusion_vote(
        self, state: VirtualAgoraState
    ) -> Literal["continue_discussion", "conclude_topic"]:
        """Evaluate both agent votes and user override.

        Enhanced in v1.3 to handle:
        - Standard agent majority vote
        - User forced conclusion from periodic stops

        Args:
            state: Current state

        Returns:
            Next node based on votes and overrides
        """
        # Check user override first
        if state.get("user_forced_conclusion", False):
            logger.info("User forced topic conclusion")
            return "conclude_topic"

        # Check agent votes
        conclusion_vote = state.get("conclusion_vote", {})

        if not conclusion_vote:
            logger.warning("No conclusion vote found, continuing discussion")
            return "continue_discussion"

        vote_passed = conclusion_vote.get("passed", False)

        if vote_passed:
            yes_votes = conclusion_vote.get("yes_votes", 0)
            total_votes = conclusion_vote.get("total_votes", 0)
            logger.info(
                f"Conclusion vote passed ({yes_votes}/{total_votes}), "
                "moving to topic conclusion"
            )
            return "conclude_topic"
        else:
            logger.info("Conclusion vote failed, continuing discussion")
            return "continue_discussion"

    # ===== Phase 3/4 Transition Conditions =====

    def check_agent_session_vote(
        self, state: VirtualAgoraState
    ) -> Literal["end_session", "check_user"]:
        """Check if agents voted to end the session.

        New in v1.3 - agents can vote to end the ecclesia.

        Args:
            state: Current state

        Returns:
            Next node based on agent vote
        """
        agents_vote_end = state.get("agents_vote_end_session", False)

        if agents_vote_end:
            logger.info("Agents voted to end session")
            return "end_session"
        else:
            logger.info("Agents voted to continue, checking with user")
            return "check_user"

    def evaluate_session_continuation(
        self, state: VirtualAgoraState
    ) -> Literal["end_session", "continue_session"]:
        """Evaluate agent and user decisions on continuation.

        Enhanced in v1.3 to handle:
        - Agent vote to end
        - User override to continue/end
        - Agenda modification requests

        Args:
            state: Current state

        Returns:
            Next node based on combined decisions
        """
        # Agent vote to end session
        if state.get("agents_vote_end_session", False):
            logger.info("Agents voted to end, moving to final report")
            return "end_session"

        # User decision
        if not state.get("user_approves_continuation", True):
            logger.info("User declined continuation, moving to final report")
            return "end_session"

        # Both approve continuation
        return "continue_session"

    def check_agenda_remaining(
        self, state: VirtualAgoraState
    ) -> Literal["no_items_remaining", "items_remaining"]:
        """Check if more agenda items remain.

        Args:
            state: Current state

        Returns:
            Next node based on agenda status
        """
        topic_queue = state.get("topic_queue", [])

        if not topic_queue:
            logger.info("No topics remaining in agenda")
            return "no_items_remaining"
        else:
            logger.info(f"{len(topic_queue)} topics remaining in agenda")
            return "items_remaining"

    def should_modify_agenda(
        self, state: VirtualAgoraState
    ) -> Literal["modify_agenda", "next_topic"]:
        """Check if agenda modification is needed.

        Args:
            state: Current state

        Returns:
            Next node based on modification request
        """
        # Check if user requested modification
        if state.get("user_requested_modification", False):
            logger.info("User requested agenda modification")
            return "modify_agenda"

        # Check if significant changes detected
        agenda_modifications = state.get("agenda_modifications")
        if agenda_modifications:
            original_count = len(agenda_modifications.get("original", []))
            revised_count = len(agenda_modifications.get("revised", []))

            # If more than 50% change, suggest re-evaluation
            if (
                original_count > 0
                and abs(original_count - revised_count) / original_count > 0.5
            ):
                logger.info(
                    "Significant agenda changes detected, suggesting modification"
                )
                return "modify_agenda"

        return "next_topic"

    # ===== Advanced Conditions =====

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

    def detect_discussion_cycle(self, state: VirtualAgoraState) -> bool:
        """Detect if discussion is stuck in a cycle.

        Args:
            state: Current state

        Returns:
            True if cycle detected
        """
        cycles = self.cycle_manager.analyze_state(state)

        for cycle in cycles:
            if cycle.confidence > 0.7:
                logger.warning(
                    f"Discussion cycle detected: {cycle.pattern_type} "
                    f"(confidence: {cycle.confidence:.2f})"
                )
                return True

        return False

    def should_force_conclusion(self, state: VirtualAgoraState) -> bool:
        """Determine if topic conclusion should be forced.

        Enhanced in v1.3 to consider:
        - Maximum rounds per topic
        - Consecutive failed conclusion votes
        - User intervention

        Args:
            state: Current state

        Returns:
            True if conclusion should be forced
        """
        current_topic = state.get("active_topic")
        if not current_topic:
            return False

        # Check max rounds
        topic_round_count = state.get("rounds_per_topic", {}).get(current_topic, 0)
        max_rounds = state.get("flow_control", {}).get("max_rounds_per_topic", 10)

        if topic_round_count >= max_rounds:
            logger.info(
                f"Forcing conclusion for topic '{current_topic}' "
                f"after {topic_round_count} rounds (max: {max_rounds})"
            )
            return True

        # Check consecutive failed votes
        vote_history = state.get("vote_history", [])
        recent_votes = [
            v
            for v in vote_history[-3:]  # Last 3 votes
            if v.get("vote_type") == "topic_conclusion"
            and v.get("topic") == current_topic
            and v.get("result") == "failed"
        ]

        if len(recent_votes) >= 3:
            logger.info(
                f"Forcing conclusion for topic '{current_topic}' "
                f"after {len(recent_votes)} consecutive failed votes"
            )
            return True

        return False

    def validate_state_transition(
        self, state: VirtualAgoraState, target_phase: int
    ) -> bool:
        """Validate that a state transition is valid.

        Enhanced for v1.3 phase structure.

        Args:
            state: Current state
            target_phase: Target phase number

        Returns:
            True if transition is valid
        """
        current_phase = state.get("current_phase", 0)

        # Define valid transitions for v1.3
        valid_transitions = {
            -1: [0],  # Start -> Initialization
            0: [1],  # Initialization -> Agenda Setting
            1: [2, 1],  # Agenda Setting -> Discussion (or loop back if rejected)
            2: [3, 2],  # Discussion -> Topic Conclusion (or continue discussion)
            3: [4],  # Topic Conclusion -> Continuation Logic
            4: [2, 5, 1],  # Continuation -> Discussion, Final Report, or Re-agenda
            5: [],  # Final Report -> End
        }

        allowed_targets = valid_transitions.get(current_phase, [])

        if target_phase not in allowed_targets:
            logger.error(
                f"Invalid transition from phase {current_phase} to phase {target_phase}"
            )
            return False

        return True

    def get_phase_name(self, phase: int) -> str:
        """Get human-readable phase name.

        Args:
            phase: Phase number

        Returns:
            Phase name
        """
        phase_names = {
            -1: "Pre-Initialization",
            0: "Initialization",
            1: "Agenda Setting",
            2: "Discussion",
            3: "Topic Conclusion",
            4: "Continuation & Re-evaluation",
            5: "Final Report Generation",
        }
        return phase_names.get(phase, f"Unknown Phase {phase}")

    def should_terminate_early(self, state: VirtualAgoraState) -> bool:
        """Check if session should terminate early due to various conditions.

        Args:
            state: Current state

        Returns:
            True if session should terminate early
        """
        flow_control = state.get("flow_control", {})

        # Check for comprehensive cycle detection
        if flow_control.get("cycle_detection_enabled", True):
            if self.detect_discussion_cycle(state):
                logger.warning("Terminating due to discussion cycles")
                return True

        # Check for maximum phase iterations
        current_phase = state.get("current_phase", 0)
        phase_history = state.get("phase_history", [])
        phase_count = sum(
            1 for t in phase_history if t.get("to_phase") == current_phase
        )

        max_iterations = flow_control.get("max_iterations_per_phase", 5)
        if phase_count >= max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached for "
                f"phase {self.get_phase_name(current_phase)}"
            )
            return True

        # Check for error conditions
        error_count = state.get("error_count", 0)
        if error_count >= 5:
            logger.error("Too many errors, terminating session")
            return True

        return False
