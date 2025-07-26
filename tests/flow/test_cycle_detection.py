"""Tests for cycle detection and prevention."""

import pytest
from datetime import datetime, timedelta

from virtual_agora.flow.cycle_detection import (
    CycleDetector,
    CircuitBreaker,
    CyclePreventionManager,
    CyclePattern,
    CircuitBreakerState,
)


class TestCycleDetector:
    """Test cycle detection functionality."""

    def setup_method(self):
        """Set up test method."""
        self.detector = CycleDetector()

        # Base state for testing
        self.base_state = {
            "phase_history": [],
            "vote_history": [],
            "messages": [],
            "topic_transition_history": [],
        }

    def test_detect_phase_cycles_no_cycle(self):
        """Test phase cycle detection with no cycle."""
        # Create normal phase progression
        transitions = []
        for i in range(10):
            transitions.append(
                {
                    "to_phase": i % 5,  # Normal progression through phases
                    "timestamp": datetime.now() - timedelta(minutes=10 - i),
                }
            )

        state = {**self.base_state, "phase_history": transitions}
        cycle = self.detector.detect_phase_cycles(state)

        assert cycle is None

    def test_detect_phase_cycles_with_cycle(self):
        """Test phase cycle detection with actual cycle."""
        # Create cycling pattern between phases 1 and 2
        transitions = []
        for i in range(12):
            phase = 1 if i % 2 == 0 else 2  # Alternating between phases 1 and 2
            transitions.append(
                {
                    "to_phase": phase,
                    "timestamp": datetime.now() - timedelta(minutes=12 - i),
                }
            )

        state = {**self.base_state, "phase_history": transitions}
        cycle = self.detector.detect_phase_cycles(state)

        assert cycle is not None
        assert cycle.pattern_type == "phase"
        assert cycle.occurrences >= 3
        assert cycle.confidence > 0.0

    def test_detect_voting_cycles_no_cycle(self):
        """Test voting cycle detection with no cycle."""
        votes = []
        for i in range(8):
            votes.append(
                {
                    "vote_type": f"vote_type_{i}",
                    "result": "passed",
                    "start_time": datetime.now() - timedelta(minutes=8 - i),
                }
            )

        state = {**self.base_state, "vote_history": votes}
        cycle = self.detector.detect_voting_cycles(state)

        assert cycle is None

    def test_detect_voting_cycles_with_cycle(self):
        """Test voting cycle detection with actual cycle."""
        # Create repeating voting pattern
        votes = []
        for i in range(12):
            vote_type = "continue_discussion" if i % 2 == 0 else "conclude_topic"
            votes.append(
                {
                    "vote_type": vote_type,
                    "result": "failed",
                    "start_time": datetime.now() - timedelta(minutes=12 - i),
                }
            )

        state = {**self.base_state, "vote_history": votes}
        cycle = self.detector.detect_voting_cycles(state)

        assert cycle is not None
        assert cycle.pattern_type == "voting"
        assert cycle.occurrences >= 3

    def test_detect_speaker_monopolization(self):
        """Test speaker monopolization detection."""
        messages = []

        # Create messages where one speaker dominates
        for i in range(20):
            speaker_id = "agent1" if i < 15 else "agent2"  # agent1 monopolizes
            messages.append(
                {
                    "speaker_id": speaker_id,
                    "timestamp": datetime.now() - timedelta(minutes=20 - i),
                }
            )

        state = {**self.base_state, "messages": messages}
        cycle = self.detector.detect_speaker_cycles(state)

        assert cycle is not None
        assert cycle.pattern_type == "speaker_monopolization"
        assert "agent1" in cycle.pattern_elements
        assert cycle.confidence > 0.6

    def test_detect_topic_cycles(self):
        """Test topic cycle detection."""
        transitions = []

        # Create alternating topic pattern
        for i in range(10):
            topic = "Topic A" if i % 2 == 0 else "Topic B"
            transitions.append(
                {
                    "topic": topic,
                    "timestamp": datetime.now() - timedelta(minutes=10 - i),
                }
            )

        state = {**self.base_state, "topic_transition_history": transitions}
        cycle = self.detector.detect_topic_cycles(state)

        assert cycle is not None
        assert cycle.pattern_type == "topic"
        assert cycle.occurrences >= 3


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def setup_method(self):
        """Set up test method."""
        self.breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=60)

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        assert self.breaker.call("test", "operation")

        breaker_state = self.breaker.get_breaker("test")
        assert breaker_state.state == "closed"
        assert breaker_state.failure_count == 0

    def test_failure_threshold_opens_circuit(self):
        """Test that reaching failure threshold opens circuit."""
        breaker_name = "test_failures"

        # Record failures up to threshold
        for i in range(3):
            self.breaker.record_failure(breaker_name, f"Error {i}")

        # Circuit should now be open
        assert not self.breaker.call(breaker_name, "blocked_operation")

        breaker_state = self.breaker.get_breaker(breaker_name)
        assert breaker_state.state == "open"

    def test_success_resets_failure_count(self):
        """Test that success resets failure count."""
        breaker_name = "test_success"

        # Record some failures
        self.breaker.record_failure(breaker_name, "Error 1")
        self.breaker.record_failure(breaker_name, "Error 2")

        # Record success
        self.breaker.record_success(breaker_name)

        breaker_state = self.breaker.get_breaker(breaker_name)
        assert breaker_state.failure_count == 1  # Should be decremented

    def test_half_open_state_transition(self):
        """Test transition to half-open state after timeout."""
        breaker_name = "test_half_open"

        # Force circuit to open
        for i in range(3):
            self.breaker.record_failure(breaker_name, f"Error {i}")

        # Simulate timeout by manually setting next attempt time to past
        breaker_state = self.breaker.get_breaker(breaker_name)
        breaker_state.next_attempt_time = datetime.now() - timedelta(seconds=1)

        # Should now allow call in half-open state
        assert self.breaker.call(breaker_name, "test_operation")
        assert breaker_state.state == "half_open"

    def test_get_status(self):
        """Test getting status of all circuit breakers."""
        # Create some breakers with different states
        self.breaker.call("breaker1", "op1")
        self.breaker.record_failure("breaker2", "error")

        status = self.breaker.get_status()

        assert "breaker1" in status
        assert "breaker2" in status
        assert status["breaker1"]["state"] == "closed"
        assert status["breaker2"]["failure_count"] == 1


class TestCyclePreventionManager:
    """Test cycle prevention manager."""

    def setup_method(self):
        """Set up test method."""
        self.manager = CyclePreventionManager()

    def test_analyze_state_no_cycles(self):
        """Test state analysis with no cycles."""
        state = {
            "phase_history": [
                {"to_phase": i, "timestamp": datetime.now()} for i in range(5)
            ],
            "vote_history": [],
            "messages": [],
            "topic_transition_history": [],
        }

        cycles = self.manager.analyze_state(state)
        assert len(cycles) == 0

    def test_analyze_state_with_cycles(self):
        """Test state analysis with cycles present."""
        # Create state with phase cycling
        transitions = []
        for i in range(12):
            phase = 1 if i % 2 == 0 else 2
            transitions.append(
                {
                    "to_phase": phase,
                    "timestamp": datetime.now() - timedelta(minutes=12 - i),
                }
            )

        state = {
            "phase_history": transitions,
            "vote_history": [],
            "messages": [],
            "topic_transition_history": [],
        }

        cycles = self.manager.analyze_state(state)
        assert len(cycles) > 0
        assert cycles[0].pattern_type == "phase"

    def test_should_intervene_high_confidence_cycle(self):
        """Test intervention decision for high-confidence cycles."""
        high_confidence_cycle = CyclePattern(
            pattern_type="voting",
            pattern_elements=["continue", "conclude"],
            occurrences=5,
            first_occurrence=datetime.now() - timedelta(minutes=10),
            last_occurrence=datetime.now(),
            confidence=0.9,
        )

        assert self.manager.should_intervene([high_confidence_cycle])

    def test_should_intervene_speaker_monopolization(self):
        """Test intervention for speaker monopolization."""
        monopolization_cycle = CyclePattern(
            pattern_type="speaker_monopolization",
            pattern_elements=["agent1"],
            occurrences=8,
            first_occurrence=datetime.now() - timedelta(minutes=5),
            last_occurrence=datetime.now(),
            confidence=0.7,
        )

        assert self.manager.should_intervene([monopolization_cycle])

    def test_should_not_intervene_low_confidence(self):
        """Test no intervention for low-confidence cycles."""
        low_confidence_cycle = CyclePattern(
            pattern_type="phase",
            pattern_elements=["1", "2"],
            occurrences=2,
            first_occurrence=datetime.now() - timedelta(minutes=5),
            last_occurrence=datetime.now(),
            confidence=0.3,
        )

        assert not self.manager.should_intervene([low_confidence_cycle])

    def test_get_intervention_strategy_phase_cycle(self):
        """Test intervention strategy for phase cycles."""
        phase_cycle = CyclePattern(
            pattern_type="phase",
            pattern_elements=["1", "2"],
            occurrences=4,
            first_occurrence=datetime.now() - timedelta(minutes=10),
            last_occurrence=datetime.now(),
            confidence=0.8,
        )

        strategy = self.manager.get_intervention_strategy([phase_cycle])

        assert strategy["intervention_needed"]
        assert len(strategy["strategies"]) == 1
        assert strategy["strategies"][0]["type"] == "phase_intervention"
        assert strategy["strategies"][0]["action"] == "force_phase_progression"

    def test_get_intervention_strategy_voting_cycle(self):
        """Test intervention strategy for voting cycles."""
        voting_cycle = CyclePattern(
            pattern_type="voting",
            pattern_elements=["continue", "conclude"],
            occurrences=4,
            first_occurrence=datetime.now() - timedelta(minutes=8),
            last_occurrence=datetime.now(),
            confidence=0.8,
        )

        strategy = self.manager.get_intervention_strategy([voting_cycle])

        assert strategy["intervention_needed"]
        assert len(strategy["strategies"]) == 1
        assert strategy["strategies"][0]["type"] == "voting_intervention"
        assert strategy["strategies"][0]["action"] == "moderator_decision"

    def test_get_intervention_strategy_speaker_monopolization(self):
        """Test intervention strategy for speaker monopolization."""
        monopolization_cycle = CyclePattern(
            pattern_type="speaker_monopolization",
            pattern_elements=["agent1"],
            occurrences=10,
            first_occurrence=datetime.now() - timedelta(minutes=5),
            last_occurrence=datetime.now(),
            confidence=0.8,
        )

        strategy = self.manager.get_intervention_strategy([monopolization_cycle])

        assert strategy["intervention_needed"]
        assert len(strategy["strategies"]) == 1
        assert strategy["strategies"][0]["type"] == "speaker_intervention"
        assert strategy["strategies"][0]["action"] == "enforce_turn_rotation"
        assert strategy["strategies"][0]["excluded_speaker"] == "agent1"

    def test_get_diagnostics(self):
        """Test diagnostics information generation."""
        # Add some test cycles to history
        test_cycle = CyclePattern(
            pattern_type="phase",
            pattern_elements=["1", "2"],
            occurrences=3,
            first_occurrence=datetime.now() - timedelta(minutes=5),
            last_occurrence=datetime.now(),
            confidence=0.7,
        )
        self.manager.detected_cycles.append(test_cycle)

        diagnostics = self.manager.get_diagnostics()

        assert "total_cycles_detected" in diagnostics
        assert "cycles_by_type" in diagnostics
        assert "circuit_breaker_status" in diagnostics
        assert "recent_cycles" in diagnostics

        assert diagnostics["total_cycles_detected"] == 1
        assert diagnostics["cycles_by_type"]["phase"] == 1

    def test_reset_breakers(self):
        """Test circuit breaker reset."""
        # Trigger a circuit breaker
        self.manager.circuit_breaker.record_failure("test", "error")

        # Reset all breakers
        self.manager.reset_breakers()

        # Should have a fresh circuit breaker
        status = self.manager.circuit_breaker.get_status()
        assert len(status) == 0  # No breakers should exist after reset
