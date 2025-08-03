"""Tests for user participation timing strategies.

This module provides comprehensive tests for the participation strategy framework,
ensuring that each timing strategy behaves correctly and provides appropriate
context for user participation prompts.
"""

import pytest
from unittest.mock import Mock

from src.virtual_agora.flow.participation_strategies import (
    ParticipationTiming,
    UserParticipationStrategy,
    StartOfRoundParticipation,
    EndOfRoundParticipation,
    DisabledParticipation,
    create_participation_strategy,
    get_available_timings,
)
from src.virtual_agora.state.schema import VirtualAgoraState


class TestParticipationTiming:
    """Test suite for ParticipationTiming enum."""

    def test_timing_enum_values(self):
        """Test that all expected timing values are defined."""
        assert ParticipationTiming.START_OF_ROUND.value == "start_of_round"
        assert ParticipationTiming.END_OF_ROUND.value == "end_of_round"
        assert ParticipationTiming.DISABLED.value == "disabled"

    def test_timing_enum_completeness(self):
        """Test that all timing values are accounted for."""
        expected_values = {"start_of_round", "end_of_round", "disabled"}
        actual_values = {timing.value for timing in ParticipationTiming}
        assert actual_values == expected_values


class TestUserParticipationStrategy:
    """Test suite for UserParticipationStrategy abstract base class."""

    def test_strategy_is_abstract(self):
        """Test that UserParticipationStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            UserParticipationStrategy()

    def test_abstract_methods_required(self):
        """Test that concrete implementations must implement all abstract methods."""

        class IncompleteStrategy(UserParticipationStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()


class TestStartOfRoundParticipation:
    """Test suite for StartOfRoundParticipation strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = StartOfRoundParticipation()
        self.test_state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
        }

    def test_timing_name(self):
        """Test strategy name."""
        assert self.strategy.get_timing_name() == "StartOfRoundParticipation"

    def test_should_request_participation_before_agents(self):
        """Test participation request before agents."""
        # Round 0 (initial) - no participation
        state_round_0 = {"current_round": 0}
        assert not self.strategy.should_request_participation_before_agents(
            state_round_0
        )

        # Round 1+ - request participation
        state_round_1 = {"current_round": 1}
        assert self.strategy.should_request_participation_before_agents(state_round_1)

        state_round_5 = {"current_round": 5}
        assert self.strategy.should_request_participation_before_agents(state_round_5)

    def test_should_request_participation_after_agents(self):
        """Test participation request after agents."""
        # Never request participation after agents in start-of-round mode
        for round_num in [0, 1, 5, 10]:
            state = {"current_round": round_num}
            assert not self.strategy.should_request_participation_after_agents(state)

    def test_get_participation_context(self):
        """Test participation context generation."""
        context = self.strategy.get_participation_context(self.test_state)

        assert context["timing"] == "round_start"
        assert "Round 1 is about to begin" in context["message"]
        assert "Test Topic" in context["message"]
        assert context["show_previous_summary"] is True
        assert context["participation_type"] == "proactive_guidance"
        assert context["round_phase"] == "pre_discussion"

    def test_get_participation_context_with_missing_topic(self):
        """Test context generation when topic is missing."""
        state_no_topic = {"current_round": 2}
        context = self.strategy.get_participation_context(state_no_topic)

        assert "Unknown Topic" in context["message"]
        assert context["timing"] == "round_start"

    def test_get_participation_context_with_missing_round(self):
        """Test context generation when round is missing."""
        state_no_round = {"active_topic": "Test Topic"}
        context = self.strategy.get_participation_context(state_no_round)

        assert "Round 0 is about to begin" in context["message"]
        assert context["timing"] == "round_start"


class TestEndOfRoundParticipation:
    """Test suite for EndOfRoundParticipation strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = EndOfRoundParticipation()
        self.test_state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
        }

    def test_timing_name(self):
        """Test strategy name."""
        assert self.strategy.get_timing_name() == "EndOfRoundParticipation"

    def test_should_request_participation_before_agents(self):
        """Test participation request before agents."""
        # Never request participation before agents in end-of-round mode
        for round_num in [0, 1, 5, 10]:
            state = {"current_round": round_num}
            assert not self.strategy.should_request_participation_before_agents(state)

    def test_should_request_participation_after_agents(self):
        """Test participation request after agents."""
        # Round 0 (initial) - no participation
        state_round_0 = {"current_round": 0}
        assert not self.strategy.should_request_participation_after_agents(
            state_round_0
        )

        # Round 1+ - request participation
        state_round_1 = {"current_round": 1}
        assert self.strategy.should_request_participation_after_agents(state_round_1)

        state_round_5 = {"current_round": 5}
        assert self.strategy.should_request_participation_after_agents(state_round_5)

    def test_get_participation_context(self):
        """Test participation context generation."""
        context = self.strategy.get_participation_context(self.test_state)

        assert context["timing"] == "round_end"
        assert "Round 1 has completed" in context["message"]
        assert "Test Topic" in context["message"]
        assert context["show_round_summary"] is True
        assert context["participation_type"] == "reactive_feedback"
        assert context["round_phase"] == "post_discussion"

    def test_get_participation_context_with_missing_topic(self):
        """Test context generation when topic is missing."""
        state_no_topic = {"current_round": 3}
        context = self.strategy.get_participation_context(state_no_topic)

        assert "Unknown Topic" in context["message"]
        assert context["timing"] == "round_end"

    def test_get_participation_context_with_missing_round(self):
        """Test context generation when round is missing."""
        state_no_round = {"active_topic": "Test Topic"}
        context = self.strategy.get_participation_context(state_no_round)

        assert "Round 0 has completed" in context["message"]
        assert context["timing"] == "round_end"


class TestDisabledParticipation:
    """Test suite for DisabledParticipation strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = DisabledParticipation()
        self.test_state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
        }

    def test_timing_name(self):
        """Test strategy name."""
        assert self.strategy.get_timing_name() == "DisabledParticipation"

    def test_should_request_participation_before_agents(self):
        """Test participation request before agents."""
        # Never request participation in disabled mode
        for round_num in [0, 1, 5, 10]:
            state = {"current_round": round_num}
            assert not self.strategy.should_request_participation_before_agents(state)

    def test_should_request_participation_after_agents(self):
        """Test participation request after agents."""
        # Never request participation in disabled mode
        for round_num in [0, 1, 5, 10]:
            state = {"current_round": round_num}
            assert not self.strategy.should_request_participation_after_agents(state)

    def test_get_participation_context(self):
        """Test participation context generation."""
        # This method should not be called in normal operation, but test for completeness
        context = self.strategy.get_participation_context(self.test_state)

        assert context["timing"] == "disabled"
        assert context["participation_type"] == "disabled"
        assert context["round_phase"] == "none"


class TestStrategyFactory:
    """Test suite for participation strategy factory functions."""

    def test_create_participation_strategy(self):
        """Test strategy creation for all timing modes."""
        # Test StartOfRound creation
        start_strategy = create_participation_strategy(
            ParticipationTiming.START_OF_ROUND
        )
        assert isinstance(start_strategy, StartOfRoundParticipation)

        # Test EndOfRound creation
        end_strategy = create_participation_strategy(ParticipationTiming.END_OF_ROUND)
        assert isinstance(end_strategy, EndOfRoundParticipation)

        # Test Disabled creation
        disabled_strategy = create_participation_strategy(ParticipationTiming.DISABLED)
        assert isinstance(disabled_strategy, DisabledParticipation)

    def test_create_participation_strategy_invalid_timing(self):
        """Test strategy creation with invalid timing."""
        with pytest.raises(ValueError, match="Unknown participation timing"):
            create_participation_strategy("invalid_timing")

    def test_get_available_timings(self):
        """Test available timings function."""
        timings = get_available_timings()

        assert len(timings) == 3
        assert ParticipationTiming.START_OF_ROUND in timings
        assert ParticipationTiming.END_OF_ROUND in timings
        assert ParticipationTiming.DISABLED in timings

        # Check descriptions are meaningful
        for timing, description in timings.items():
            assert isinstance(description, str)
            assert len(description) > 10  # Ensure descriptions are substantial


class TestStrategyComparison:
    """Test suite for comparing different strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.start_strategy = StartOfRoundParticipation()
        self.end_strategy = EndOfRoundParticipation()
        self.disabled_strategy = DisabledParticipation()

        self.test_states = [
            {"current_round": 0, "active_topic": "Topic 1"},
            {"current_round": 1, "active_topic": "Topic 2"},
            {"current_round": 5, "active_topic": "Topic 3"},
        ]

    def test_mutually_exclusive_timing(self):
        """Test that strategies are mutually exclusive in their timing."""
        for state in self.test_states:
            # Start and end strategies should be mutually exclusive
            start_before = (
                self.start_strategy.should_request_participation_before_agents(state)
            )
            start_after = self.start_strategy.should_request_participation_after_agents(
                state
            )
            end_before = self.end_strategy.should_request_participation_before_agents(
                state
            )
            end_after = self.end_strategy.should_request_participation_after_agents(
                state
            )

            # Start strategy: before OR after, never both
            assert not (start_before and start_after)
            # End strategy: before OR after, never both
            assert not (end_before and end_after)

            # Strategies should complement each other for active rounds
            if state["current_round"] >= 1:
                # For active rounds: start_before should be True, end_after should be True
                # But they operate at different times, so both can be True
                # The key is that within each strategy, timing is exclusive
                assert (
                    start_before == True and start_after == False
                )  # Start: before only
                assert end_before == False and end_after == True  # End: after only

    def test_disabled_strategy_isolation(self):
        """Test that disabled strategy never requests participation."""
        for state in self.test_states:
            assert (
                not self.disabled_strategy.should_request_participation_before_agents(
                    state
                )
            )
            assert not self.disabled_strategy.should_request_participation_after_agents(
                state
            )

    def test_context_differences(self):
        """Test that strategies provide different contexts."""
        state = {"current_round": 2, "active_topic": "Test Topic"}

        start_context = self.start_strategy.get_participation_context(state)
        end_context = self.end_strategy.get_participation_context(state)
        disabled_context = self.disabled_strategy.get_participation_context(state)

        # Different timing values
        assert start_context["timing"] != end_context["timing"]
        assert start_context["timing"] != disabled_context["timing"]
        assert end_context["timing"] != disabled_context["timing"]

        # Different participation types
        assert start_context["participation_type"] != end_context["participation_type"]
        assert (
            start_context["participation_type"]
            != disabled_context["participation_type"]
        )
        assert (
            end_context["participation_type"] != disabled_context["participation_type"]
        )

        # Different round phases
        assert start_context["round_phase"] != end_context["round_phase"]
        assert start_context["round_phase"] != disabled_context["round_phase"]
        assert end_context["round_phase"] != disabled_context["round_phase"]


class TestIntegrationScenarios:
    """Test suite for integration scenarios and edge cases."""

    def test_round_progression_start_strategy(self):
        """Test start strategy behavior across multiple rounds."""
        strategy = StartOfRoundParticipation()

        # Test progression from round 0 to 5
        for round_num in range(6):
            state = {"current_round": round_num, "active_topic": f"Topic {round_num}"}

            before = strategy.should_request_participation_before_agents(state)
            after = strategy.should_request_participation_after_agents(state)

            if round_num == 0:
                assert not before and not after  # No participation in round 0
            else:
                assert before and not after  # Participation before agents only

    def test_round_progression_end_strategy(self):
        """Test end strategy behavior across multiple rounds."""
        strategy = EndOfRoundParticipation()

        # Test progression from round 0 to 5
        for round_num in range(6):
            state = {"current_round": round_num, "active_topic": f"Topic {round_num}"}

            before = strategy.should_request_participation_before_agents(state)
            after = strategy.should_request_participation_after_agents(state)

            if round_num == 0:
                assert not before and not after  # No participation in round 0
            else:
                assert not before and after  # Participation after agents only

    def test_missing_state_handling(self):
        """Test strategy behavior with missing state information."""
        strategies = [
            StartOfRoundParticipation(),
            EndOfRoundParticipation(),
            DisabledParticipation(),
        ]

        # Test with empty state
        empty_state = {}
        for strategy in strategies:
            # Should not raise exceptions
            before = strategy.should_request_participation_before_agents(empty_state)
            after = strategy.should_request_participation_after_agents(empty_state)
            context = strategy.get_participation_context(empty_state)

            assert isinstance(before, bool)
            assert isinstance(after, bool)
            assert isinstance(context, dict)

    def test_configuration_switching_scenario(self):
        """Test scenario where configuration is switched between strategies."""
        # Simulate configuration change scenario
        test_state = {"current_round": 3, "active_topic": "Important Topic"}

        # Create different strategies
        strategies = {
            ParticipationTiming.START_OF_ROUND: create_participation_strategy(
                ParticipationTiming.START_OF_ROUND
            ),
            ParticipationTiming.END_OF_ROUND: create_participation_strategy(
                ParticipationTiming.END_OF_ROUND
            ),
            ParticipationTiming.DISABLED: create_participation_strategy(
                ParticipationTiming.DISABLED
            ),
        }

        # Test that each strategy behaves differently with the same state
        results = {}
        for timing, strategy in strategies.items():
            results[timing] = {
                "before": strategy.should_request_participation_before_agents(
                    test_state
                ),
                "after": strategy.should_request_participation_after_agents(test_state),
                "context": strategy.get_participation_context(test_state),
            }

        # Verify different behaviors
        start_result = results[ParticipationTiming.START_OF_ROUND]
        end_result = results[ParticipationTiming.END_OF_ROUND]
        disabled_result = results[ParticipationTiming.DISABLED]

        # Start strategy: participation before, not after
        assert start_result["before"] and not start_result["after"]

        # End strategy: participation after, not before
        assert not end_result["before"] and end_result["after"]

        # Disabled strategy: no participation
        assert not disabled_result["before"] and not disabled_result["after"]

        # Different contexts
        assert start_result["context"]["timing"] == "round_start"
        assert end_result["context"]["timing"] == "round_end"
        assert disabled_result["context"]["timing"] == "disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
