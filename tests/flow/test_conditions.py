"""Tests for the FlowConditions conditional logic."""

import pytest
from datetime import datetime, timedelta

from virtual_agora.flow.edges import FlowConditions
from virtual_agora.state.schema import VirtualAgoraState, HITLState, FlowControl


class TestFlowConditions:
    """Test FlowConditions conditional logic."""

    def setup_method(self):
        """Set up test method."""
        self.conditions = FlowConditions()

        # Create base state for testing
        self.base_state = {
            "session_id": "test-session",
            "start_time": datetime.now(),
            "config_hash": "test-hash",
            "current_phase": 1,
            "phase_history": [],
            "phase_start_time": datetime.now(),
            "current_round": 0,
            "round_history": [],
            "turn_order_history": [],
            "rounds_per_topic": {},
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": None,
                "prompt_message": None,
                "options": None,
                "approval_history": [],
            },
            "flow_control": {
                "max_rounds_per_topic": 10,
                "auto_conclude_threshold": 3,
                "context_window_limit": 8000,
                "cycle_detection_enabled": True,
                "max_iterations_per_phase": 5,
            },
            "active_topic": "Test Topic",
            "topic_queue": ["Topic 1", "Topic 2"],
            "proposed_topics": ["Topic 1", "Topic 2", "Topic 3"],
            "topics_info": {},
            "completed_topics": [],
            "agents": {},
            "moderator_id": "moderator",
            "current_speaker_id": None,
            "speaking_order": ["agent1", "agent2"],
            "next_speaker_index": 0,
            "messages": [],
            "last_message_id": "",
            "active_vote": None,
            "vote_history": [],
            "votes": [],
            "consensus_proposals": {},
            "consensus_reached": {},
            "phase_summaries": {},
            "topic_summaries": {},
            "consensus_summaries": {},
            "final_report": None,
            "total_messages": 0,
            "messages_by_phase": {},
            "messages_by_agent": {},
            "messages_by_topic": {},
            "vote_participation_rate": {},
            "tool_calls": [],
            "active_tool_calls": {},
            "tool_metrics": {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_execution_time_ms": 0.0,
                "calls_by_tool": {},
                "calls_by_agent": {},
                "errors_by_type": {},
            },
            "tools_enabled_agents": [],
            "last_error": None,
            "error_count": 0,
            "warnings": [],
        }

    def test_should_start_discussion_approved(self):
        """Test should_start_discussion when agenda is approved."""
        state = {**self.base_state, "agenda_approved": True}
        result = self.conditions.should_start_discussion(state)
        assert result == "discussion"

    def test_should_start_discussion_rejected(self):
        """Test should_start_discussion when agenda is rejected."""
        state = {**self.base_state, "agenda_approved": False}
        result = self.conditions.should_start_discussion(state)
        assert result == "agenda_setting"

    def test_should_start_discussion_missing(self):
        """Test should_start_discussion when approval status is missing."""
        state = self.base_state.copy()
        result = self.conditions.should_start_discussion(state)
        assert result == "agenda_setting"  # Defaults to not approved

    def test_should_start_conclusion_poll_early_rounds(self):
        """Test conclusion poll logic for early rounds."""
        state = {**self.base_state, "current_round": 2}
        result = self.conditions.should_start_conclusion_poll(state)
        assert result == "continue_discussion"

    def test_should_start_conclusion_poll_round_three(self):
        """Test conclusion poll logic for round 3."""
        state = {**self.base_state, "current_round": 3}
        result = self.conditions.should_start_conclusion_poll(state)
        assert result == "conclusion_poll"

    def test_should_start_conclusion_poll_max_rounds(self):
        """Test conclusion poll logic when max rounds reached."""
        state = {**self.base_state, "current_round": 10}
        result = self.conditions.should_start_conclusion_poll(state)
        assert result == "conclusion_poll"

    def test_evaluate_conclusion_vote_passed(self):
        """Test conclusion vote evaluation when vote passes."""
        state = {
            **self.base_state,
            "conclusion_vote": {"passed": True, "yes_votes": 3, "total_votes": 4},
        }
        result = self.conditions.evaluate_conclusion_vote(state)
        assert result == "minority_considerations"

    def test_evaluate_conclusion_vote_failed(self):
        """Test conclusion vote evaluation when vote fails."""
        state = {
            **self.base_state,
            "conclusion_vote": {"passed": False, "yes_votes": 1, "total_votes": 4},
        }
        result = self.conditions.evaluate_conclusion_vote(state)
        assert result == "continue_discussion"

    def test_evaluate_conclusion_vote_missing(self):
        """Test conclusion vote evaluation when vote data is missing."""
        state = self.base_state.copy()
        result = self.conditions.evaluate_conclusion_vote(state)
        assert result == "continue_discussion"  # Default behavior

    def test_should_continue_session_approved(self):
        """Test session continuation when user approves."""
        state = {**self.base_state, "continue_session": True}
        result = self.conditions.should_continue_session(state)
        assert result == "continue"

    def test_should_continue_session_rejected(self):
        """Test session continuation when user rejects."""
        state = {**self.base_state, "continue_session": False}
        result = self.conditions.should_continue_session(state)
        assert result == "end"

    def test_has_topics_remaining_with_queue(self):
        """Test topic evaluation when topics remain in queue."""
        state = {**self.base_state, "topic_queue": ["Topic 1", "Topic 2"]}
        result = self.conditions.has_topics_remaining(state)
        assert result == "next_topic"

    def test_has_topics_remaining_empty_queue(self):
        """Test topic evaluation when queue is empty."""
        state = {**self.base_state, "topic_queue": []}
        result = self.conditions.has_topics_remaining(state)
        assert result == "generate_report"

    def test_has_topics_remaining_significant_agenda_changes(self):
        """Test topic evaluation with significant agenda modifications."""
        state = {
            **self.base_state,
            "topic_queue": ["Topic 1"],
            "proposed_topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4"],
            "agenda_modifications": {"revised_agenda": ["Topic 1"]},
        }
        result = self.conditions.has_topics_remaining(state)
        assert result == "re_evaluate_agenda"

    def test_should_terminate_early_cycle_detection(self):
        """Test early termination due to cycle detection."""
        # Create phase history with cycling pattern
        phase_history = []
        for i in range(10):
            phase_history.append(
                {"to_phase": 1 if i % 2 == 0 else 2, "timestamp": datetime.now()}
            )

        state = {
            **self.base_state,
            "phase_history": phase_history,
            "flow_control": {
                **self.base_state["flow_control"],
                "cycle_detection_enabled": True,
            },
        }

        result = self.conditions.should_terminate_early(state)
        assert result == True

    def test_should_terminate_early_max_iterations(self):
        """Test early termination due to max iterations."""
        # Create phase history with too many iterations of same phase
        phase_history = []
        for i in range(6):
            phase_history.append({"to_phase": 2, "timestamp": datetime.now()})

        state = {**self.base_state, "current_phase": 2, "phase_history": phase_history}

        result = self.conditions.should_terminate_early(state)
        assert result == True

    def test_should_terminate_early_too_many_errors(self):
        """Test early termination due to too many errors."""
        state = {**self.base_state, "error_count": 6}
        result = self.conditions.should_terminate_early(state)
        assert result == True

    def test_should_terminate_early_normal_conditions(self):
        """Test early termination under normal conditions."""
        result = self.conditions.should_terminate_early(self.base_state)
        assert result == False

    def test_validate_state_transition_valid(self):
        """Test state transition validation for valid transitions."""
        state = {**self.base_state, "current_phase": 1}
        result = self.conditions.validate_state_transition(state, 2)
        assert result == True

    def test_validate_state_transition_invalid(self):
        """Test state transition validation for invalid transitions."""
        state = {**self.base_state, "current_phase": 1}
        result = self.conditions.validate_state_transition(state, 4)
        assert result == False

    def test_should_compress_context_below_threshold(self):
        """Test context compression when below threshold."""
        # Create proper message objects
        messages = [
            {
                "id": f"msg_{i}",
                "content": "short msg",
                "speaker_id": "agent1",
                "timestamp": datetime.now(),
            }
            for i in range(10)
        ]
        state = {**self.base_state, "messages": messages}
        result = self.conditions.should_compress_context(state)
        assert result == False

    def test_should_compress_context_above_threshold(self):
        """Test context compression when above threshold."""
        # Create proper message objects with long content to trigger compression
        messages = [
            {
                "id": f"msg_{i}",
                "content": "This is a very long message " * 50,
                "speaker_id": "agent1",
                "timestamp": datetime.now(),
            }
            for i in range(50)
        ]
        state = {**self.base_state, "messages": messages}
        result = self.conditions.should_compress_context(state)
        assert result == True

    def test_detect_voting_cycle_no_cycle(self):
        """Test voting cycle detection with no cycle."""
        state = {
            **self.base_state,
            "vote_history": [
                {
                    "vote_type": "agenda_vote",
                    "result": "passed",
                    "start_time": datetime.now(),
                },
                {
                    "vote_type": "conclusion_vote",
                    "result": "failed",
                    "start_time": datetime.now(),
                },
                {
                    "vote_type": "continue_discussion",
                    "result": "passed",
                    "start_time": datetime.now(),
                },
            ],
        }
        result = self.conditions.detect_voting_cycle(state)
        assert result == False

    def test_detect_voting_cycle_with_cycle(self):
        """Test voting cycle detection with cycle present."""
        # Create a repeating pattern to trigger cycle detection
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
        result = self.conditions.detect_voting_cycle(state)
        assert result == True

    def test_should_force_conclusion_max_rounds(self):
        """Test forced conclusion due to max rounds."""
        state = {
            **self.base_state,
            "active_topic": "Test Topic",
            "rounds_per_topic": {"Test Topic": 10},
        }
        result = self.conditions.should_force_conclusion(state)
        assert result == True

    def test_should_force_conclusion_consecutive_attempts(self):
        """Test forced conclusion due to consecutive failed attempts."""
        vote_history = []
        for i in range(4):
            vote_history.append(
                {"vote_type": "continue_discussion", "topic": "Test Topic"}
            )

        state = {
            **self.base_state,
            "active_topic": "Test Topic",
            "vote_history": vote_history,
        }
        result = self.conditions.should_force_conclusion(state)
        assert result == True

    def test_should_force_conclusion_normal_conditions(self):
        """Test forced conclusion under normal conditions."""
        state = {
            **self.base_state,
            "active_topic": "Test Topic",
            "rounds_per_topic": {"Test Topic": 2},
        }
        result = self.conditions.should_force_conclusion(state)
        assert result == False
