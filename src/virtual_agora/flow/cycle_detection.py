"""Cycle Detection and Prevention for Virtual Agora.

This module provides comprehensive cycle detection and circuit breaker
functionality to prevent infinite loops and stuck states in discussion flows.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import deque, Counter

from virtual_agora.state.schema import VirtualAgoraState, PhaseTransition, VoteRound
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CyclePattern:
    """Represents a detected cycle pattern."""

    pattern_type: str  # 'phase', 'voting', 'speaker', 'topic'
    pattern_elements: List[str]  # The repeating elements
    occurrences: int  # Number of times pattern repeated
    first_occurrence: datetime
    last_occurrence: datetime
    confidence: float  # 0.0 to 1.0


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    name: str
    state: str  # 'closed', 'open', 'half_open'
    failure_count: int
    last_failure_time: Optional[datetime]
    next_attempt_time: Optional[datetime]
    threshold: int
    timeout_seconds: int


class CycleDetector:
    """Detects various types of cycles in discussion flow."""

    def __init__(self, pattern_window_size: int = 20, min_pattern_length: int = 2):
        """Initialize cycle detector.

        Args:
            pattern_window_size: Size of window to analyze for patterns
            min_pattern_length: Minimum length of repeating pattern to detect
        """
        self.pattern_window_size = pattern_window_size
        self.min_pattern_length = min_pattern_length

    def detect_phase_cycles(self, state: VirtualAgoraState) -> Optional[CyclePattern]:
        """Detect cycles in phase transitions.

        Args:
            state: Virtual Agora state

        Returns:
            Detected cycle pattern or None
        """
        transitions = state["phase_history"]
        if len(transitions) < self.min_pattern_length * 2:
            return None

        # Extract recent phase sequence
        recent_transitions = transitions[-self.pattern_window_size :]
        phase_sequence = []

        for transition in recent_transitions:
            phase_sequence.append(str(transition["to_phase"]))

        return self._detect_sequence_pattern(
            phase_sequence, "phase", [t["timestamp"] for t in recent_transitions]
        )

    def detect_voting_cycles(self, state: VirtualAgoraState) -> Optional[CyclePattern]:
        """Detect cycles in voting patterns.

        Args:
            state: Virtual Agora state

        Returns:
            Detected cycle pattern or None
        """
        vote_history = state["vote_history"]
        if len(vote_history) < self.min_pattern_length * 2:
            return None

        # Extract recent voting sequence
        recent_votes = vote_history[-self.pattern_window_size :]
        vote_sequence = []
        timestamps = []

        for vote in recent_votes:
            vote_key = f"{vote['vote_type']}:{vote.get('result', 'unknown')}"
            vote_sequence.append(vote_key)
            timestamps.append(vote["start_time"])

        return self._detect_sequence_pattern(vote_sequence, "voting", timestamps)

    def detect_speaker_cycles(self, state: VirtualAgoraState) -> Optional[CyclePattern]:
        """Detect unusual patterns in speaker rotation.

        Args:
            state: Virtual Agora state

        Returns:
            Detected cycle pattern or None
        """
        messages = state["messages"]
        if len(messages) < self.min_pattern_length * 2:
            return None

        # Extract recent speaker sequence
        recent_messages = messages[-self.pattern_window_size :]
        speaker_sequence = []
        timestamps = []

        for message in recent_messages:
            speaker_sequence.append(message["speaker_id"])
            timestamps.append(message["timestamp"])

        # First check for speaker monopolization (higher priority)
        pattern = self._detect_speaker_monopolization(recent_messages)

        # If no monopolization, look for other speaker patterns
        if pattern is None:
            pattern = self._detect_sequence_pattern(
                speaker_sequence, "speaker", timestamps
            )

        return pattern

    def detect_topic_cycles(self, state: VirtualAgoraState) -> Optional[CyclePattern]:
        """Detect cycles in topic transitions.

        Args:
            state: Virtual Agora state

        Returns:
            Detected cycle pattern or None
        """
        # Check topic transition history if available
        topic_transitions = state.get("topic_transition_history", [])
        if len(topic_transitions) < self.min_pattern_length:
            return None

        recent_transitions = topic_transitions[-self.pattern_window_size :]
        topic_sequence = []
        timestamps = []

        for transition in recent_transitions:
            topic_sequence.append(transition.get("topic", "unknown"))
            timestamps.append(transition.get("timestamp", datetime.now()))

        return self._detect_sequence_pattern(topic_sequence, "topic", timestamps)

    def _detect_sequence_pattern(
        self, sequence: List[str], pattern_type: str, timestamps: List[datetime]
    ) -> Optional[CyclePattern]:
        """Detect repeating patterns in a sequence.

        Args:
            sequence: Sequence to analyze
            pattern_type: Type of pattern being detected
            timestamps: Corresponding timestamps

        Returns:
            Detected cycle pattern or None
        """
        if len(sequence) < self.min_pattern_length * 2:
            return None

        # Try different pattern lengths, but avoid detecting normal progression as cycles
        for pattern_length in range(
            self.min_pattern_length, min(6, len(sequence) // 2 + 1)
        ):
            pattern = sequence[-pattern_length:]
            occurrences = 1

            # Skip if pattern looks like normal progression (increasing sequence)
            if pattern_length > 2 and self._is_normal_progression(pattern):
                continue

            # Count how many times this pattern repeats exactly
            search_pos = len(sequence) - pattern_length
            consecutive_matches = 0

            while search_pos >= pattern_length:
                segment = sequence[search_pos - pattern_length : search_pos]
                if segment == pattern:
                    consecutive_matches += 1
                    search_pos -= pattern_length
                else:
                    break

            occurrences = consecutive_matches + 1  # Include the original pattern

            # Consider it a cycle if pattern repeats at least 3 times consecutively
            if occurrences >= 3 and consecutive_matches >= 2:
                confidence = min(
                    1.0, occurrences / 5.0
                )  # Max confidence at 5+ occurrences

                first_idx = max(0, len(sequence) - (occurrences * pattern_length))

                return CyclePattern(
                    pattern_type=pattern_type,
                    pattern_elements=pattern,
                    occurrences=occurrences,
                    first_occurrence=(
                        timestamps[first_idx]
                        if first_idx < len(timestamps)
                        else datetime.now()
                    ),
                    last_occurrence=timestamps[-1] if timestamps else datetime.now(),
                    confidence=confidence,
                )

        return None

    def _is_normal_progression(self, pattern: List[str]) -> bool:
        """Check if pattern represents normal progression (not a cycle).

        Args:
            pattern: Pattern to check

        Returns:
            True if pattern is normal progression
        """
        try:
            # Convert to integers and check for sequential progression
            int_pattern = [int(x) for x in pattern]

            # Check if it's an increasing sequence
            if all(
                int_pattern[i] < int_pattern[i + 1] for i in range(len(int_pattern) - 1)
            ):
                return True

            # Check if it's a normal phase cycle (0,1,2,3,4 or similar)
            if len(int_pattern) >= 3 and max(int_pattern) - min(int_pattern) >= 2:
                return True

        except (ValueError, TypeError):
            # Not numeric, could still be a cycle
            pass

        return False

    def _detect_speaker_monopolization(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[CyclePattern]:
        """Detect if a single speaker is dominating the conversation.

        Args:
            messages: Recent messages to analyze

        Returns:
            Cycle pattern if monopolization detected
        """
        if len(messages) < 10:  # Need enough messages to detect monopolization
            return None

        speaker_counts = Counter(msg["speaker_id"] for msg in messages)
        total_messages = len(messages)

        for speaker_id, count in speaker_counts.items():
            # If one speaker has more than 60% of recent messages, it's monopolization
            if count / total_messages > 0.6 and count >= 6:
                return CyclePattern(
                    pattern_type="speaker_monopolization",
                    pattern_elements=[speaker_id],
                    occurrences=count,
                    first_occurrence=messages[0]["timestamp"],
                    last_occurrence=messages[-1]["timestamp"],
                    confidence=min(1.0, count / total_messages),
                )

        return None


class CircuitBreaker:
    """Circuit breaker for preventing stuck states."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 300,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting half-open state
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.breakers: Dict[str, CircuitBreakerState] = {}

    def get_breaker(self, name: str) -> CircuitBreakerState:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name

        Returns:
            Circuit breaker state
        """
        if name not in self.breakers:
            self.breakers[name] = CircuitBreakerState(
                name=name,
                state="closed",
                failure_count=0,
                last_failure_time=None,
                next_attempt_time=None,
                threshold=self.failure_threshold,
                timeout_seconds=self.timeout_seconds,
            )
        return self.breakers[name]

    def call(self, name: str, operation_name: str = "") -> bool:
        """Check if operation is allowed through circuit breaker.

        Args:
            name: Circuit breaker name
            operation_name: Name of operation being protected

        Returns:
            True if operation is allowed, False if circuit is open
        """
        breaker = self.get_breaker(name)
        now = datetime.now()

        if breaker.state == "open":
            # Check if timeout period has passed
            if breaker.next_attempt_time and now >= breaker.next_attempt_time:
                # Move to half-open state
                breaker.state = "half_open"
                breaker.failure_count = 0
                logger.info(f"Circuit breaker '{name}' moving to half-open state")
                return True
            else:
                logger.warning(
                    f"Circuit breaker '{name}' is OPEN - blocking {operation_name}"
                )
                return False

        elif breaker.state == "half_open":
            # Allow limited calls in half-open state
            if breaker.failure_count < self.half_open_max_calls:
                return True
            else:
                logger.warning(f"Circuit breaker '{name}' half-open limit exceeded")
                return False

        # Closed state - allow operation
        return True

    def record_success(self, name: str):
        """Record successful operation.

        Args:
            name: Circuit breaker name
        """
        breaker = self.get_breaker(name)

        if breaker.state == "half_open":
            # Successful call in half-open state - close circuit
            breaker.state = "closed"
            breaker.failure_count = 0
            breaker.last_failure_time = None
            breaker.next_attempt_time = None
            logger.info(f"Circuit breaker '{name}' closed after successful recovery")

        elif breaker.state == "closed":
            # Reset failure count on success
            breaker.failure_count = max(0, breaker.failure_count - 1)

    def record_failure(self, name: str, error_message: str = ""):
        """Record failed operation.

        Args:
            name: Circuit breaker name
            error_message: Optional error description
        """
        breaker = self.get_breaker(name)
        now = datetime.now()

        breaker.failure_count += 1
        breaker.last_failure_time = now

        if breaker.failure_count >= breaker.threshold:
            # Open the circuit
            breaker.state = "open"
            breaker.next_attempt_time = now + timedelta(seconds=breaker.timeout_seconds)
            logger.warning(
                f"Circuit breaker '{name}' OPENED after {breaker.failure_count} failures: {error_message}"
            )
        else:
            logger.info(
                f"Circuit breaker '{name}' recorded failure {breaker.failure_count}/{breaker.threshold}: {error_message}"
            )

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Status dictionary for all breakers
        """
        status = {}
        for name, breaker in self.breakers.items():
            status[name] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "threshold": breaker.threshold,
                "last_failure_time": (
                    breaker.last_failure_time.isoformat()
                    if breaker.last_failure_time
                    else None
                ),
                "next_attempt_time": (
                    breaker.next_attempt_time.isoformat()
                    if breaker.next_attempt_time
                    else None
                ),
            }
        return status


class CyclePreventionManager:
    """Manages cycle detection and prevention strategies."""

    def __init__(self):
        """Initialize cycle prevention manager."""
        self.detector = CycleDetector()
        self.circuit_breaker = CircuitBreaker()
        self.detected_cycles: List[CyclePattern] = []

    def analyze_state(self, state: VirtualAgoraState) -> List[CyclePattern]:
        """Analyze state for all types of cycles.

        Args:
            state: Virtual Agora state

        Returns:
            List of detected cycle patterns
        """
        detected = []

        # Detect different types of cycles
        phase_cycle = self.detector.detect_phase_cycles(state)
        if phase_cycle:
            detected.append(phase_cycle)

        voting_cycle = self.detector.detect_voting_cycles(state)
        if voting_cycle:
            detected.append(voting_cycle)

        speaker_cycle = self.detector.detect_speaker_cycles(state)
        if speaker_cycle:
            detected.append(speaker_cycle)

        topic_cycle = self.detector.detect_topic_cycles(state)
        if topic_cycle:
            detected.append(topic_cycle)

        # Update detected cycles history
        self.detected_cycles.extend(detected)

        # Log detected cycles
        for cycle in detected:
            logger.warning(
                f"Detected {cycle.pattern_type} cycle: {cycle.pattern_elements} "
                f"(occurred {cycle.occurrences} times, confidence: {cycle.confidence:.2f})"
            )

        return detected

    def should_intervene(self, cycles: List[CyclePattern]) -> bool:
        """Determine if intervention is needed to break cycles.

        Args:
            cycles: List of detected cycles

        Returns:
            True if intervention is needed
        """
        for cycle in cycles:
            # High-confidence cycles with many occurrences need intervention
            if cycle.confidence > 0.7 and cycle.occurrences >= 4:
                return True

            # Speaker monopolization always needs intervention
            if (
                cycle.pattern_type == "speaker_monopolization"
                and cycle.confidence > 0.6
            ):
                return True

            # Voting cycles with high frequency need intervention
            if cycle.pattern_type == "voting" and cycle.occurrences >= 3:
                time_span = (
                    cycle.last_occurrence - cycle.first_occurrence
                ).total_seconds()
                if time_span < 600:  # Less than 10 minutes
                    return True

        return False

    def get_intervention_strategy(self, cycles: List[CyclePattern]) -> Dict[str, Any]:
        """Get recommended intervention strategy for detected cycles.

        Args:
            cycles: List of detected cycles

        Returns:
            Intervention strategy dictionary
        """
        strategies = []

        for cycle in cycles:
            if cycle.pattern_type == "phase":
                strategies.append(
                    {
                        "type": "phase_intervention",
                        "action": "force_phase_progression",
                        "target_phase": 4,  # Force to summary phase
                        "reason": f"Breaking phase cycle: {' -> '.join(cycle.pattern_elements)}",
                    }
                )

            elif cycle.pattern_type == "voting":
                strategies.append(
                    {
                        "type": "voting_intervention",
                        "action": "moderator_decision",
                        "reason": f"Breaking voting cycle: {' -> '.join(cycle.pattern_elements)}",
                    }
                )

            elif cycle.pattern_type == "speaker_monopolization":
                strategies.append(
                    {
                        "type": "speaker_intervention",
                        "action": "enforce_turn_rotation",
                        "excluded_speaker": cycle.pattern_elements[0],
                        "reason": f"Speaker {cycle.pattern_elements[0]} monopolizing conversation",
                    }
                )

            elif cycle.pattern_type == "topic":
                strategies.append(
                    {
                        "type": "topic_intervention",
                        "action": "force_topic_conclusion",
                        "reason": f"Breaking topic cycle: {' -> '.join(cycle.pattern_elements)}",
                    }
                )

        return {
            "intervention_needed": len(strategies) > 0,
            "strategies": strategies,
            "cycles_detected": len(cycles),
            "highest_confidence": max([c.confidence for c in cycles], default=0.0),
        }

    def reset_breakers(self):
        """Reset all circuit breakers."""
        self.circuit_breaker = CircuitBreaker()
        logger.info("All circuit breakers reset")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics information.

        Returns:
            Diagnostics dictionary
        """
        return {
            "total_cycles_detected": len(self.detected_cycles),
            "cycles_by_type": Counter(
                cycle.pattern_type for cycle in self.detected_cycles
            ),
            "circuit_breaker_status": self.circuit_breaker.get_status(),
            "recent_cycles": [
                {
                    "type": cycle.pattern_type,
                    "pattern": cycle.pattern_elements,
                    "occurrences": cycle.occurrences,
                    "confidence": cycle.confidence,
                    "last_occurrence": cycle.last_occurrence.isoformat(),
                }
                for cycle in self.detected_cycles[-10:]  # Last 10 cycles
            ],
        }
