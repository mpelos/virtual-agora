"""Unit tests for v1.3 edge conditions."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.edges_v13 import V13FlowConditions


@pytest.fixture
def flow_conditions():
    """Create V13FlowConditions instance."""
    return V13FlowConditions(checkpoint_interval=5)


class TestPhase1Conditions:
    """Test Phase 1 edge conditions."""

    def test_should_start_discussion_approved(self, flow_conditions):
        """Test starting discussion when agenda is approved."""
        state = {"agenda_approved": True}
        result = flow_conditions.should_start_discussion(state)
        assert result == "discussion"

    def test_should_start_discussion_rejected(self, flow_conditions):
        """Test returning to agenda setting when rejected."""
        state = {"agenda_approved": False}
        result = flow_conditions.should_start_discussion(state)
        assert result == "agenda_setting"


class TestPhase2Conditions:
    """Test Phase 2 discussion loop conditions."""

    def test_check_round_threshold_below(self, flow_conditions):
        """Test continuing discussion when below round 3."""
        state = {"current_round": 2}
        result = flow_conditions.check_round_threshold(state)
        assert result == "continue_discussion"

    def test_check_round_threshold_met(self, flow_conditions):
        """Test starting polls when round 3+ reached."""
        state = {"current_round": 3}
        result = flow_conditions.check_round_threshold(state)
        assert result == "start_polling"

    def test_check_periodic_stop_not_multiple(self, flow_conditions):
        """Test no periodic stop when round not multiple of 5."""
        state = {"current_round": 3}
        result = flow_conditions.check_periodic_stop(state)
        assert result == "check_votes"

    def test_check_periodic_stop_multiple_of_5(self, flow_conditions):
        """Test periodic stop at round 5, 10, 15, etc."""
        for round_num in [5, 10, 15, 20]:
            state = {"current_round": round_num}
            result = flow_conditions.check_periodic_stop(state)
            assert result == "periodic_stop", f"Failed for round {round_num}"

    def test_evaluate_conclusion_vote_user_forced(self, flow_conditions):
        """Test user forcing topic conclusion."""
        state = {
            "user_forced_conclusion": True,
            "conclusion_vote": {"passed": False},  # Even if vote failed
        }
        result = flow_conditions.evaluate_conclusion_vote(state)
        assert result == "conclude_topic"

    def test_evaluate_conclusion_vote_passed(self, flow_conditions):
        """Test conclusion when vote passes."""
        state = {
            "user_forced_conclusion": False,
            "conclusion_vote": {"passed": True, "yes_votes": 3, "total_votes": 4},
        }
        result = flow_conditions.evaluate_conclusion_vote(state)
        assert result == "conclude_topic"

    def test_evaluate_conclusion_vote_failed(self, flow_conditions):
        """Test continuing discussion when vote fails."""
        state = {
            "user_forced_conclusion": False,
            "conclusion_vote": {"passed": False, "yes_votes": 1, "total_votes": 4},
        }
        result = flow_conditions.evaluate_conclusion_vote(state)
        assert result == "continue_discussion"


class TestPhase4Conditions:
    """Test Phase 4 continuation conditions."""

    def test_check_agent_session_vote_end(self, flow_conditions):
        """Test agents voting to end session."""
        state = {"agents_vote_end_session": True}
        result = flow_conditions.check_agent_session_vote(state)
        assert result == "end_session"

    def test_check_agent_session_vote_continue(self, flow_conditions):
        """Test agents voting to continue."""
        state = {"agents_vote_end_session": False}
        result = flow_conditions.check_agent_session_vote(state)
        assert result == "check_user"

    def test_evaluate_session_continuation_agents_end(self, flow_conditions):
        """Test ending when agents vote to end."""
        state = {
            "agents_vote_end_session": True,
            "user_approves_continuation": True,  # Even if user approves
        }
        result = flow_conditions.evaluate_session_continuation(state)
        assert result == "end_session"

    def test_evaluate_session_continuation_user_end(self, flow_conditions):
        """Test ending when user declines."""
        state = {"agents_vote_end_session": False, "user_approves_continuation": False}
        result = flow_conditions.evaluate_session_continuation(state)
        assert result == "end_session"

    def test_evaluate_session_continuation_both_continue(self, flow_conditions):
        """Test continuing when both approve."""
        state = {"agents_vote_end_session": False, "user_approves_continuation": True}
        result = flow_conditions.evaluate_session_continuation(state)
        assert result == "continue_session"

    def test_check_agenda_remaining_empty(self, flow_conditions):
        """Test when no topics remain."""
        state = {"topic_queue": []}
        result = flow_conditions.check_agenda_remaining(state)
        assert result == "no_items_remaining"

    def test_check_agenda_remaining_has_items(self, flow_conditions):
        """Test when topics remain."""
        state = {"topic_queue": ["Topic A", "Topic B"]}
        result = flow_conditions.check_agenda_remaining(state)
        assert result == "items_remaining"


class TestAdvancedConditions:
    """Test advanced conditions like cycles and compression."""

    def test_should_force_conclusion_max_rounds(self, flow_conditions):
        """Test forcing conclusion at max rounds."""
        state = {
            "active_topic": "Topic A",
            "rounds_per_topic": {"Topic A": 10},
            "flow_control": {"max_rounds_per_topic": 10},
        }
        result = flow_conditions.should_force_conclusion(state)
        assert result is True

    def test_should_force_conclusion_consecutive_failures(self, flow_conditions):
        """Test forcing conclusion after 3 failed votes."""
        state = {
            "active_topic": "Topic A",
            "rounds_per_topic": {"Topic A": 5},
            "flow_control": {"max_rounds_per_topic": 10},
            "vote_history": [
                {
                    "vote_type": "topic_conclusion",
                    "topic": "Topic A",
                    "result": "failed",
                },
                {
                    "vote_type": "topic_conclusion",
                    "topic": "Topic A",
                    "result": "failed",
                },
                {
                    "vote_type": "topic_conclusion",
                    "topic": "Topic A",
                    "result": "failed",
                },
            ],
        }
        result = flow_conditions.should_force_conclusion(state)
        assert result is True

    def test_detect_discussion_cycle(self, flow_conditions):
        """Test discussion cycle detection."""
        with patch.object(
            flow_conditions.cycle_manager, "analyze_state"
        ) as mock_analyze:
            # Mock a cycle detection
            mock_cycle = Mock()
            mock_cycle.confidence = 0.8
            mock_cycle.pattern_type = "repetitive"
            mock_analyze.return_value = [mock_cycle]

            state = {}
            result = flow_conditions.detect_discussion_cycle(state)
            assert result is True

    def test_validate_state_transition_valid(self, flow_conditions):
        """Test valid state transitions."""
        # Test some valid transitions
        valid_transitions = [
            (0, 1),  # Init -> Agenda
            (1, 2),  # Agenda -> Discussion
            (2, 3),  # Discussion -> Conclusion
            (3, 4),  # Conclusion -> Continuation
            (4, 5),  # Continuation -> Report
        ]

        for current, target in valid_transitions:
            state = {"current_phase": current}
            result = flow_conditions.validate_state_transition(state, target)
            assert result is True, f"Transition {current} -> {target} should be valid"

    def test_validate_state_transition_invalid(self, flow_conditions):
        """Test invalid state transitions."""
        # Test some invalid transitions
        invalid_transitions = [
            (0, 3),  # Can't skip from Init to Conclusion
            (2, 5),  # Can't skip from Discussion to Report
            (5, 1),  # Can't go back from Report to Agenda
        ]

        for current, target in invalid_transitions:
            state = {"current_phase": current}
            result = flow_conditions.validate_state_transition(state, target)
            assert (
                result is False
            ), f"Transition {current} -> {target} should be invalid"

    def test_get_phase_name(self, flow_conditions):
        """Test phase name lookup."""
        assert flow_conditions.get_phase_name(0) == "Initialization"
        assert flow_conditions.get_phase_name(1) == "Agenda Setting"
        assert flow_conditions.get_phase_name(2) == "Discussion"
        assert flow_conditions.get_phase_name(3) == "Topic Conclusion"
        assert flow_conditions.get_phase_name(4) == "Continuation & Re-evaluation"
        assert flow_conditions.get_phase_name(5) == "Final Report Generation"
        assert flow_conditions.get_phase_name(99) == "Unknown Phase 99"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
