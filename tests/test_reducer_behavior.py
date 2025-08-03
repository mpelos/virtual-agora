"""Comprehensive tests for safe_list_append reducer behavior.

This test module ensures that the reducer correctly handles all input patterns
and auto-corrects common bugs from LangGraph processing.
"""

import pytest
from datetime import datetime
from typing import List, Any

from virtual_agora.state.reducers import safe_list_append


class TestSafeListAppendReducer:
    """Test the safe_list_append reducer with various input scenarios."""

    def setup_method(self):
        """Reset reducer state before each test."""
        # Reset the empty list counter
        if hasattr(safe_list_append, "_empty_list_count"):
            delattr(safe_list_append, "_empty_list_count")

    def test_normal_individual_item_append(self):
        """Test normal case: appending individual items to lists."""
        # Test with None (new list)
        result = safe_list_append(None, "item1")
        assert result == ["item1"]

        # Test appending to existing list
        result = safe_list_append(["item1"], "item2")
        assert result == ["item1", "item2"]

        # Test with dict item
        dict_item = {"key": "value", "id": 1}
        result = safe_list_append([], dict_item)
        assert result == [dict_item]

    def test_turn_order_history_valid_pattern(self):
        """Test that turn_order_history receives lists of agent IDs correctly."""
        # Test valid turn order (list of agent IDs)
        agent_list = ["agent1", "agent2", "agent3"]
        result = safe_list_append(None, agent_list)
        assert result == [agent_list]

        # Test appending multiple turn orders
        result = safe_list_append([agent_list], ["agent4", "agent5"])
        assert result == [agent_list, ["agent4", "agent5"]]

    def test_phase_history_unwrapping(self):
        """Test auto-unwrapping of single-item lists for phase_history."""
        phase_transition = {
            "from_phase": 0,
            "to_phase": 1,
            "timestamp": datetime.now(),
            "reason": "Test transition",
            "triggered_by": "test",
        }

        # Test unwrapping single-item list
        wrapped_transition = [phase_transition]
        result = safe_list_append(None, wrapped_transition)
        assert result == [phase_transition]

        # Test appending to existing list
        existing_transitions = [
            {
                "from_phase": -1,
                "to_phase": 0,
                "timestamp": datetime.now(),
                "reason": "Initial",
                "triggered_by": "system",
            }
        ]
        result = safe_list_append(existing_transitions, wrapped_transition)
        assert result == existing_transitions + [phase_transition]

    def test_round_history_unwrapping(self):
        """Test auto-unwrapping of single-item lists for round_history."""
        round_info = {
            "round_id": "test-round-123",
            "round_number": 1,
            "topic": "test topic",
            "start_time": datetime.now(),
            "participants": ["agent1", "agent2"],
            "message_count": 2,
        }

        # Test unwrapping single-item list
        wrapped_round = [round_info]
        result = safe_list_append(None, wrapped_round)
        assert result == [round_info]

    def test_round_summaries_unwrapping(self):
        """Test auto-unwrapping of single-item lists for round_summaries."""
        round_summary = {
            "round_number": 1,
            "topic": "test topic",
            "summary_text": "This is a test summary",
            "created_by": "summarizer",
            "timestamp": datetime.now(),
            "token_count": 50,
            "compression_ratio": 0.1,
        }

        # Test unwrapping single-item list
        wrapped_summary = [round_summary]
        result = safe_list_append(None, wrapped_summary)
        assert result == [round_summary]

    def test_turn_order_nested_list_unwrapping(self):
        """Test auto-unwrapping of nested lists for turn_order_history."""
        turn_order = ["agent1", "agent2", "agent3"]

        # Test unwrapping nested list [[...]]
        nested_turn_order = [turn_order]
        result = safe_list_append(None, nested_turn_order)
        assert result == [turn_order]

        # Test appending to existing turn orders
        existing_turn_orders = [["previous1", "previous2"]]
        result = safe_list_append(existing_turn_orders, nested_turn_order)
        assert result == existing_turn_orders + [turn_order]

    def test_vote_batch_handling(self):
        """Test proper handling of vote batches."""
        votes = [
            {"voter_id": "agent1", "choice": "yes", "timestamp": datetime.now()},
            {"voter_id": "agent2", "choice": "no", "timestamp": datetime.now()},
            {"voter_id": "agent3", "choice": "yes", "timestamp": datetime.now()},
        ]

        # Test vote batch extension (not appending as single item)
        result = safe_list_append(None, votes)
        assert result == votes  # Should extend, not wrap in another list

        # Test appending to existing votes
        existing_votes = [
            {"voter_id": "previous", "choice": "yes", "timestamp": datetime.now()}
        ]
        result = safe_list_append(existing_votes, votes)
        assert result == existing_votes + votes

    def test_empty_list_rejection(self):
        """Test that empty lists are rejected with proper error handling."""
        # Test empty list rejection
        result = safe_list_append(None, [])
        assert result == []

        # Test with existing list
        existing_list = ["item1"]
        result = safe_list_append(existing_list, [])
        assert result == existing_list

        # Test that counter increments
        safe_list_append(None, [])  # Second empty list
        assert getattr(safe_list_append, "_empty_list_count", 0) >= 2

    def test_invalid_list_patterns(self):
        """Test handling of unexpected list patterns."""
        # Test list with mixed types
        mixed_list = ["string", 123, {"key": "value"}]
        result = safe_list_append(None, mixed_list)
        assert result == [mixed_list]  # Should append as single item with warning

        # Test list that doesn't match any expected pattern
        weird_list = [1, 2, 3, 4, 5]
        result = safe_list_append(None, weird_list)
        assert result == [weird_list]  # Should append as single item with warning

    def test_error_recovery_stability(self):
        """Test that the reducer maintains stability even with malformed inputs."""
        # Test various malformed inputs
        test_cases = [
            [],  # Empty list
            [[]],  # Nested empty list
            [None],  # List with None
            [[[], []]],  # Deeply nested empty lists
            [{"incomplete": "dict"}],  # List with incomplete dict
        ]

        current_list = ["existing_item"]
        for test_input in test_cases:
            result = safe_list_append(current_list, test_input)
            # Should either append correctly or maintain existing list
            assert isinstance(result, list)
            assert len(result) >= len(current_list)

    def test_large_vote_batch_rejection(self):
        """Test that oversized vote batches are rejected."""
        # Create a large vote batch (> 10 votes)
        large_vote_batch = [
            {"voter_id": f"agent{i}", "choice": "yes", "timestamp": datetime.now()}
            for i in range(15)
        ]

        # Should be appended as single item, not extended
        result = safe_list_append(None, large_vote_batch)
        assert result == [large_vote_batch]

    def test_pattern_specificity(self):
        """Test that pattern detection is specific and doesn't false positive."""
        # Dict that looks like phase transition but missing required fields
        fake_phase = {"from_phase": 1, "missing_to_phase": 2}
        result = safe_list_append(None, [fake_phase])
        assert result == [[fake_phase]]  # Should not unwrap

        # Dict that looks like round info but missing required fields
        fake_round = {"round_number": 1, "missing_round_id": "test"}
        result = safe_list_append(None, [fake_round])
        assert result == [[fake_round]]  # Should not unwrap

        # Dict that looks like round summary but missing required fields
        fake_summary = {"round_number": 1, "missing_summary_text": "test"}
        result = safe_list_append(None, [fake_summary])
        assert result == [[fake_summary]]  # Should not unwrap

    def test_logging_noise_reduction(self):
        """Test that repeated empty list errors don't spam logs."""
        # This test verifies the counter mechanism works
        # Multiple empty list calls should increment counter
        for i in range(5):
            safe_list_append(None, [])

        # Counter should be at least 5
        assert getattr(safe_list_append, "_empty_list_count", 0) >= 5


class TestReducerIntegration:
    """Integration tests for reducer behavior with realistic data."""

    def test_complete_discussion_flow_data(self):
        """Test reducer with realistic data from a complete discussion flow."""
        # Simulate phase transitions
        phase_history = []

        transitions = [
            {
                "from_phase": -1,
                "to_phase": 0,
                "timestamp": datetime.now(),
                "reason": "Init",
                "triggered_by": "system",
            },
            {
                "from_phase": 0,
                "to_phase": 1,
                "timestamp": datetime.now(),
                "reason": "Agenda",
                "triggered_by": "user",
            },
            {
                "from_phase": 1,
                "to_phase": 2,
                "timestamp": datetime.now(),
                "reason": "Discussion",
                "triggered_by": "system",
            },
        ]

        for transition in transitions:
            # Simulate both correct and wrapped patterns
            if len(phase_history) % 2 == 0:
                # Correct pattern
                phase_history = safe_list_append(phase_history, transition)
            else:
                # Wrapped pattern (simulating LangGraph bug)
                phase_history = safe_list_append(phase_history, [transition])

        assert len(phase_history) == 3
        assert all(isinstance(t, dict) for t in phase_history)
        assert all("from_phase" in t and "to_phase" in t for t in phase_history)

    def test_mixed_pattern_stability(self):
        """Test that mixing correct and incorrect patterns doesn't break the system."""
        # Start with empty state
        turn_orders = []
        round_summaries = []

        # Add data with mixed patterns
        turn_orders = safe_list_append(turn_orders, ["agent1", "agent2"])  # Correct
        turn_orders = safe_list_append(
            turn_orders, [["agent3", "agent4"]]
        )  # Wrapped (incorrect)

        summary1 = {
            "round_number": 1,
            "summary_text": "Summary 1",
            "created_by": "test",
            "timestamp": datetime.now(),
            "token_count": 10,
            "compression_ratio": 0.1,
        }
        summary2 = {
            "round_number": 2,
            "summary_text": "Summary 2",
            "created_by": "test",
            "timestamp": datetime.now(),
            "token_count": 15,
            "compression_ratio": 0.1,
        }

        round_summaries = safe_list_append(round_summaries, summary1)  # Correct
        round_summaries = safe_list_append(
            round_summaries, [summary2]
        )  # Wrapped (incorrect)

        # Verify final state is correct regardless of input pattern
        assert len(turn_orders) == 2
        assert turn_orders[0] == ["agent1", "agent2"]
        assert turn_orders[1] == ["agent3", "agent4"]  # Should be unwrapped

        assert len(round_summaries) == 2
        assert round_summaries[0] == summary1
        assert round_summaries[1] == summary2  # Should be unwrapped

    def test_concurrent_updates_simulation(self):
        """Test behavior when multiple 'concurrent' updates happen rapidly."""
        # Simulate rapid state updates that might cause LangGraph issues
        phase_history = []

        # Rapid transitions with mixed patterns
        for i in range(10):
            transition = {
                "from_phase": i,
                "to_phase": i + 1,
                "timestamp": datetime.now(),
                "reason": f"Transition {i}",
                "triggered_by": "test",
            }

            # Mix correct and incorrect patterns randomly
            if i % 3 == 0:
                phase_history = safe_list_append(phase_history, [transition])  # Wrapped
            else:
                phase_history = safe_list_append(phase_history, transition)  # Correct

        # Should have 10 transitions regardless of input pattern
        assert len(phase_history) == 10
        assert all(isinstance(t, dict) for t in phase_history)
        assert phase_history[0]["from_phase"] == 0
        assert phase_history[-1]["to_phase"] == 10
