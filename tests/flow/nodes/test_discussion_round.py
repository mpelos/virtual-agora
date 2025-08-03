"""Tests for DiscussionRoundNode with configurable user participation.

This module provides comprehensive tests for the unified DiscussionRoundNode
that handles complete discussion rounds with configurable user participation
timing through the strategy pattern.
"""

import pytest
import uuid
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.virtual_agora.flow.nodes.discussion_round import DiscussionRoundNode
from src.virtual_agora.flow.participation_strategies import (
    ParticipationTiming,
    StartOfRoundParticipation,
    EndOfRoundParticipation,
    DisabledParticipation,
    create_participation_strategy,
)
from src.virtual_agora.flow.state_manager import FlowStateManager, RoundState
from src.virtual_agora.flow.nodes.base import NodeDependencies
from src.virtual_agora.state.schema import VirtualAgoraState
from src.virtual_agora.agents.llm_agent import LLMAgent


class TestDiscussionRoundNode:
    """Test suite for DiscussionRoundNode class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.mock_flow_state_manager = Mock(spec=FlowStateManager)
        self.mock_dependencies = Mock(spec=NodeDependencies)

        # Create mock agents
        self.mock_discussing_agents = []
        for i in range(3):
            agent = Mock(spec=LLMAgent)
            agent.agent_id = f"agent_{i}"
            agent.system_prompt = f"System prompt for agent {i}"
            self.mock_discussing_agents.append(agent)

        self.mock_specialized_agents = {"moderator": Mock(), "summarizer": Mock()}

        # Create test state
        self.test_state = {
            "session_id": "test_session",
            "current_round": 1,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "messages": [],
            "round_summaries": [],
            "speaking_order": ["agent_0", "agent_1", "agent_2"],
        }

        # Create mock round state
        self.mock_round_state = RoundState(
            round_number=1,
            current_topic="Test Topic",
            speaking_order=["agent_0", "agent_1", "agent_2"],
            round_id=str(uuid.uuid4()),
            round_start_time=datetime.now(),
            theme="Test Theme",
        )

    def create_discussion_node(self, participation_strategy):
        """Helper to create DiscussionRoundNode with given strategy."""
        return DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=self.mock_discussing_agents,
            specialized_agents=self.mock_specialized_agents,
            participation_strategy=participation_strategy,
            node_dependencies=self.mock_dependencies,
        )

    def test_initialization(self):
        """Test DiscussionRoundNode initialization."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        assert node.flow_state_manager is self.mock_flow_state_manager
        assert node.discussing_agents == self.mock_discussing_agents
        assert node.specialized_agents == self.mock_specialized_agents
        assert node.participation_strategy is strategy
        assert node.dependencies is self.mock_dependencies

    def test_get_node_name(self):
        """Test node name generation."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        name = node.get_node_name()
        assert "DiscussionRoundNode" in name
        assert "StartOfRoundParticipation" in name

    def test_validate_preconditions_success(self):
        """Test successful precondition validation."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        is_valid = node.validate_preconditions(self.test_state)
        assert is_valid
        assert len(node.get_validation_errors()) == 0

    def test_validate_preconditions_missing_topic(self):
        """Test precondition validation with missing active topic."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        state_no_topic = {"session_id": "test"}
        is_valid = node.validate_preconditions(state_no_topic)

        assert not is_valid
        errors = node.get_validation_errors()
        assert any("Missing required key: active_topic" in error for error in errors)

    def test_validate_preconditions_no_discussing_agents(self):
        """Test precondition validation with no discussing agents."""
        strategy = StartOfRoundParticipation()
        node = DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=[],  # Empty list
            specialized_agents=self.mock_specialized_agents,
            participation_strategy=strategy,
        )

        is_valid = node.validate_preconditions(self.test_state)

        assert not is_valid
        errors = node.get_validation_errors()
        assert any("No discussing agents available" in error for error in errors)

    def test_validate_preconditions_no_moderator(self):
        """Test precondition validation with no moderator agent."""
        strategy = StartOfRoundParticipation()
        specialized_agents_no_moderator = {"summarizer": Mock()}

        node = DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=self.mock_discussing_agents,
            specialized_agents=specialized_agents_no_moderator,
            participation_strategy=strategy,
        )

        is_valid = node.validate_preconditions(self.test_state)

        assert not is_valid
        errors = node.get_validation_errors()
        assert any("Moderator agent not available" in error for error in errors)

    def test_create_interrupt_payload(self):
        """Test interrupt payload creation."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        # Mock strategy context
        mock_context = {
            "timing": "round_start",
            "message": "Test message",
            "participation_type": "proactive_guidance",
            "round_phase": "pre_discussion",
        }
        strategy.get_participation_context = Mock(return_value=mock_context)

        payload = node.create_interrupt_payload(self.test_state)

        assert payload["type"] == "user_turn_participation"
        assert payload["current_round"] == 1
        assert payload["current_topic"] == "Test Topic"
        assert payload["message"] == "Test message"
        assert payload["timing"] == "round_start"
        assert payload["participation_type"] == "proactive_guidance"
        assert payload["round_phase"] == "pre_discussion"
        assert "options" in payload

    def test_create_interrupt_payload_with_previous_summary(self):
        """Test interrupt payload creation with previous round summary."""
        strategy = EndOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        # State with round summaries
        state_with_summaries = {
            **self.test_state,
            "current_round": 2,
            "round_summaries": [
                {"topic": "Test Topic", "summary": "Previous round summary"},
                {"topic": "Other Topic", "summary": "Other summary"},
            ],
        }

        mock_context = {
            "timing": "round_end",
            "message": "Test message",
            "participation_type": "reactive_feedback",
            "round_phase": "post_discussion",
        }
        strategy.get_participation_context = Mock(return_value=mock_context)

        payload = node.create_interrupt_payload(state_with_summaries)

        assert payload["previous_summary"] == "Previous round summary"

    def test_process_user_input_continue(self):
        """Test processing user input with 'continue' action."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        user_input = {"action": "continue"}
        updates = node.process_user_input(user_input, self.test_state)

        # Should return minimal updates for continue action
        assert isinstance(updates, dict)
        assert "user_forced_conclusion" not in updates

    def test_process_user_input_participate(self):
        """Test processing user input with 'participate' action."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        # Mock FlowStateManager.apply_user_participation
        mock_user_updates = {"user_message_added": True, "round_updated": True}
        self.mock_flow_state_manager.apply_user_participation.return_value = (
            mock_user_updates
        )

        user_input = {
            "action": "participate",
            "message": "User guidance message",
            "timing_phase": "before_agents",
        }

        updates = node.process_user_input(user_input, self.test_state)

        # Should call FlowStateManager and return its updates
        self.mock_flow_state_manager.apply_user_participation.assert_called_once()
        call_args = self.mock_flow_state_manager.apply_user_participation.call_args
        assert call_args[1]["user_input"] == "User guidance message"
        assert (
            call_args[1]["use_next_round"] == False
        )  # before_agents uses current round

        assert updates == mock_user_updates

    def test_process_user_input_participate_after_agents(self):
        """Test processing user input with 'participate' action after agents."""
        strategy = EndOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        # Mock FlowStateManager.apply_user_participation
        mock_user_updates = {"user_message_added": True, "round_updated": True}
        self.mock_flow_state_manager.apply_user_participation.return_value = (
            mock_user_updates
        )

        user_input = {
            "action": "participate",
            "message": "User feedback message",
            "timing_phase": "after_agents",
        }

        updates = node.process_user_input(user_input, self.test_state)

        # Should call FlowStateManager with use_next_round=True for after_agents
        call_args = self.mock_flow_state_manager.apply_user_participation.call_args
        assert call_args[1]["use_next_round"] == True  # after_agents uses next round

    def test_process_user_input_finalize(self):
        """Test processing user input with 'finalize' action."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        user_input = {"action": "finalize"}
        updates = node.process_user_input(user_input, self.test_state)

        assert updates["user_forced_conclusion"] is True
        assert "user_finalize_reason" in updates

    def test_process_user_input_none(self):
        """Test processing None user input."""
        strategy = StartOfRoundParticipation()
        node = self.create_discussion_node(strategy)

        updates = node.process_user_input(None, self.test_state)

        # Should handle None gracefully
        assert isinstance(updates, dict)


class TestDiscussionRoundNodeWithStrategies:
    """Test suite for DiscussionRoundNode with different participation strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_flow_state_manager = Mock(spec=FlowStateManager)
        self.mock_discussing_agents = [Mock(agent_id=f"agent_{i}") for i in range(2)]
        self.mock_specialized_agents = {"moderator": Mock()}

        self.test_state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "messages": [],
        }

    def create_node_with_strategy(self, timing: ParticipationTiming):
        """Helper to create node with specific timing strategy."""
        strategy = create_participation_strategy(timing)
        return DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=self.mock_discussing_agents,
            specialized_agents=self.mock_specialized_agents,
            participation_strategy=strategy,
        )

    @patch("src.virtual_agora.flow.nodes.discussion_round.interrupt")
    def test_execute_start_of_round_participation(self, mock_interrupt):
        """Test execution with start-of-round participation strategy."""
        node = self.create_node_with_strategy(ParticipationTiming.START_OF_ROUND)

        # Mock interrupt response
        mock_interrupt.return_value = {"action": "continue"}

        # Mock FlowStateManager methods
        self.mock_flow_state_manager.prepare_round_state.return_value = RoundState(
            round_number=1,
            current_topic="Test Topic",
            speaking_order=["agent_0", "agent_1"],
            round_id="test",
            round_start_time=datetime.now(),
            theme="Test Theme",
        )
        self.mock_flow_state_manager.finalize_round.return_value = {
            "round_completed": True
        }

        # Mock _execute_agent_discussions to avoid complex agent logic
        with patch.object(
            node, "_execute_agent_discussions", return_value={"agents_completed": True}
        ):
            updates = node.execute(self.test_state)

        # Should call interrupt for user participation before agents
        mock_interrupt.assert_called_once()
        interrupt_payload = mock_interrupt.call_args[0][0]
        assert interrupt_payload["timing_phase"] == "before_agents"

        # Should complete successfully
        assert "agents_completed" in updates

    @patch("src.virtual_agora.flow.nodes.discussion_round.interrupt")
    def test_execute_end_of_round_participation(self, mock_interrupt):
        """Test execution with end-of-round participation strategy."""
        node = self.create_node_with_strategy(ParticipationTiming.END_OF_ROUND)

        # Mock interrupt response
        mock_interrupt.return_value = {"action": "continue"}

        # Mock FlowStateManager methods
        self.mock_flow_state_manager.prepare_round_state.return_value = RoundState(
            round_number=1,
            current_topic="Test Topic",
            speaking_order=["agent_0", "agent_1"],
            round_id="test",
            round_start_time=datetime.now(),
            theme="Test Theme",
        )
        self.mock_flow_state_manager.finalize_round.return_value = {
            "round_completed": True
        }

        # Mock _execute_agent_discussions to avoid complex agent logic
        with patch.object(
            node, "_execute_agent_discussions", return_value={"agents_completed": True}
        ):
            updates = node.execute(self.test_state)

        # Should call interrupt for user participation after agents
        mock_interrupt.assert_called_once()
        interrupt_payload = mock_interrupt.call_args[0][0]
        assert interrupt_payload["timing_phase"] == "after_agents"

        # Should complete successfully
        assert "agents_completed" in updates

    def test_execute_disabled_participation(self):
        """Test execution with disabled participation strategy."""
        node = self.create_node_with_strategy(ParticipationTiming.DISABLED)

        # Mock FlowStateManager methods
        self.mock_flow_state_manager.prepare_round_state.return_value = RoundState(
            round_number=1,
            current_topic="Test Topic",
            speaking_order=["agent_0", "agent_1"],
            round_id="test",
            round_start_time=datetime.now(),
            theme="Test Theme",
        )
        self.mock_flow_state_manager.finalize_round.return_value = {
            "round_completed": True
        }

        # Mock _execute_agent_discussions to avoid complex agent logic
        with patch.object(
            node, "_execute_agent_discussions", return_value={"agents_completed": True}
        ):
            with patch(
                "src.virtual_agora.flow.nodes.discussion_round.interrupt"
            ) as mock_interrupt:
                updates = node.execute(self.test_state)

        # Should NOT call interrupt for user participation
        mock_interrupt.assert_not_called()

        # Should complete successfully with only agent discussion
        assert "agents_completed" in updates

    def test_round_zero_no_participation(self):
        """Test that round 0 has no user participation regardless of strategy."""
        # Test all strategies with round 0
        for timing in [
            ParticipationTiming.START_OF_ROUND,
            ParticipationTiming.END_OF_ROUND,
        ]:
            node = self.create_node_with_strategy(timing)

            state_round_0 = {**self.test_state, "current_round": 0}

            # Mock FlowStateManager methods
            self.mock_flow_state_manager.prepare_round_state.return_value = RoundState(
                round_number=0,
                current_topic="Test Topic",
                speaking_order=["agent_0", "agent_1"],
                round_id="test",
                round_start_time=datetime.now(),
                theme="Test Theme",
            )
            self.mock_flow_state_manager.finalize_round.return_value = {
                "round_completed": True
            }

            # Mock _execute_agent_discussions
            with patch.object(
                node,
                "_execute_agent_discussions",
                return_value={"agents_completed": True},
            ):
                with patch(
                    "src.virtual_agora.flow.nodes.discussion_round.interrupt"
                ) as mock_interrupt:
                    updates = node.execute(state_round_0)

            # Should NOT call interrupt for round 0
            mock_interrupt.assert_not_called()


class TestDiscussionRoundNodeAgentExecution:
    """Test suite for agent discussion execution within DiscussionRoundNode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_flow_state_manager = Mock(spec=FlowStateManager)
        self.mock_discussing_agents = []
        for i in range(2):
            agent = Mock(spec=LLMAgent)
            agent.agent_id = f"agent_{i}"
            agent.system_prompt = f"System prompt {i}"
            self.mock_discussing_agents.append(agent)

        self.mock_moderator = Mock()
        self.mock_specialized_agents = {"moderator": self.mock_moderator}

        self.test_state = {
            "current_round": 1,
            "active_topic": "Test Topic",
            "main_topic": "Test Theme",
            "messages": [],
            "round_summaries": [],
        }

        self.mock_round_state = RoundState(
            round_number=1,
            current_topic="Test Topic",
            speaking_order=["agent_0", "agent_1"],
            round_id="test_round",
            round_start_time=datetime.now(),
            theme="Test Theme",
        )

    def test_execute_agent_discussions_success(self):
        """Test successful agent discussion execution."""
        strategy = (
            DisabledParticipation()
        )  # No user participation to isolate agent logic
        node = DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=self.mock_discussing_agents,
            specialized_agents=self.mock_specialized_agents,
            participation_strategy=strategy,
        )

        # Mock FlowStateManager
        self.mock_flow_state_manager.prepare_round_state.return_value = (
            self.mock_round_state
        )
        self.mock_flow_state_manager.finalize_round.return_value = {
            "round_finalized": True
        }

        # Mock message coordinator
        mock_message_coordinator = Mock()
        mock_message_coordinator.assemble_agent_context.return_value = (
            "Context text",
            [],
        )
        self.mock_flow_state_manager.message_coordinator = mock_message_coordinator

        # Mock moderator relevance check
        self.mock_moderator.evaluate_message_relevance.return_value = {
            "is_relevant": True,
            "relevance_score": 0.9,
        }

        # Mock agent responses
        for agent in self.mock_discussing_agents:
            agent.return_value = {"messages": [Mock(content="Agent response")]}

        # Mock external functions
        with (
            patch(
                "src.virtual_agora.flow.nodes.discussion_round.validate_agent_context",
                return_value=(True, [], 1.0),
            ),
            patch(
                "src.virtual_agora.flow.nodes.discussion_round.retry_agent_call"
            ) as mock_retry,
            patch(
                "src.virtual_agora.flow.nodes.discussion_round.get_provider_type_from_agent_id",
                return_value="test",
            ),
            patch(
                "src.virtual_agora.flow.nodes.discussion_round.create_langchain_message"
            ) as mock_create_msg,
            patch("virtual_agora.ui.discussion_display.display_agent_response"),
        ):

            # Set up mock returns
            mock_retry.side_effect = [
                {"messages": [Mock(content="Response from agent_0")]},
                {"messages": [Mock(content="Response from agent_1")]},
            ]
            mock_create_msg.return_value = Mock()

            updates = node._execute_agent_discussions(self.test_state)

        # Verify execution
        assert updates == {"round_finalized": True}
        self.mock_flow_state_manager.prepare_round_state.assert_called_once_with(
            self.test_state
        )
        self.mock_flow_state_manager.finalize_round.assert_called_once()

        # Verify agent calls
        assert mock_retry.call_count == 2  # Called for each agent

        # Verify relevance checking
        assert self.mock_moderator.evaluate_message_relevance.call_count == 2

    def test_execute_agent_discussions_prepare_round_failure(self):
        """Test agent discussion execution with round state preparation failure."""
        strategy = DisabledParticipation()
        node = DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=self.mock_discussing_agents,
            specialized_agents=self.mock_specialized_agents,
            participation_strategy=strategy,
        )

        # Mock FlowStateManager to raise error
        self.mock_flow_state_manager.prepare_round_state.side_effect = ValueError(
            "Failed to prepare round"
        )

        with pytest.raises(ValueError, match="Failed to prepare round"):
            node._execute_agent_discussions(self.test_state)

    def test_execute_agent_discussions_no_moderator(self):
        """Test agent discussion execution with no moderator agent."""
        strategy = DisabledParticipation()
        specialized_agents_no_moderator = {"summarizer": Mock()}

        node = DiscussionRoundNode(
            flow_state_manager=self.mock_flow_state_manager,
            discussing_agents=self.mock_discussing_agents,
            specialized_agents=specialized_agents_no_moderator,
            participation_strategy=strategy,
        )

        # Mock FlowStateManager
        self.mock_flow_state_manager.prepare_round_state.return_value = (
            self.mock_round_state
        )

        with pytest.raises(ValueError, match="No moderator agent available"):
            node._execute_agent_discussions(self.test_state)


class TestDiscussionRoundNodeIntegration:
    """Integration tests for DiscussionRoundNode with real strategy instances."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # Create minimal real objects where possible
        self.mock_flow_state_manager = Mock(spec=FlowStateManager)
        self.mock_discussing_agents = [Mock(agent_id=f"agent_{i}") for i in range(2)]
        self.mock_specialized_agents = {"moderator": Mock()}

        self.test_state = {
            "current_round": 1,
            "active_topic": "Integration Test Topic",
            "main_topic": "Integration Theme",
            "messages": [],
            "round_summaries": [],
        }

    def test_strategy_switching_same_node_class(self):
        """Test that the same node class works with different strategies."""
        # Create nodes with different strategies
        start_strategy = StartOfRoundParticipation()
        end_strategy = EndOfRoundParticipation()
        disabled_strategy = DisabledParticipation()

        nodes = {
            "start": DiscussionRoundNode(
                self.mock_flow_state_manager,
                self.mock_discussing_agents,
                self.mock_specialized_agents,
                start_strategy,
            ),
            "end": DiscussionRoundNode(
                self.mock_flow_state_manager,
                self.mock_discussing_agents,
                self.mock_specialized_agents,
                end_strategy,
            ),
            "disabled": DiscussionRoundNode(
                self.mock_flow_state_manager,
                self.mock_discussing_agents,
                self.mock_specialized_agents,
                disabled_strategy,
            ),
        }

        # All nodes should be valid instances
        for name, node in nodes.items():
            assert isinstance(node, DiscussionRoundNode)
            assert node.validate_preconditions(self.test_state)
            assert node.participation_strategy is not None

        # Strategies should behave differently
        state_round_1 = {**self.test_state, "current_round": 1}

        start_before = nodes[
            "start"
        ].participation_strategy.should_request_participation_before_agents(
            state_round_1
        )
        start_after = nodes[
            "start"
        ].participation_strategy.should_request_participation_after_agents(
            state_round_1
        )

        end_before = nodes[
            "end"
        ].participation_strategy.should_request_participation_before_agents(
            state_round_1
        )
        end_after = nodes[
            "end"
        ].participation_strategy.should_request_participation_after_agents(
            state_round_1
        )

        disabled_before = nodes[
            "disabled"
        ].participation_strategy.should_request_participation_before_agents(
            state_round_1
        )
        disabled_after = nodes[
            "disabled"
        ].participation_strategy.should_request_participation_after_agents(
            state_round_1
        )

        # Verify different behaviors
        assert start_before and not start_after  # Start: before only
        assert not end_before and end_after  # End: after only
        assert not disabled_before and not disabled_after  # Disabled: neither

    def test_node_name_reflects_strategy(self):
        """Test that node names reflect the participation strategy."""
        strategies_and_names = [
            (StartOfRoundParticipation(), "StartOfRoundParticipation"),
            (EndOfRoundParticipation(), "EndOfRoundParticipation"),
            (DisabledParticipation(), "DisabledParticipation"),
        ]

        for strategy, expected_name in strategies_and_names:
            node = DiscussionRoundNode(
                self.mock_flow_state_manager,
                self.mock_discussing_agents,
                self.mock_specialized_agents,
                strategy,
            )

            node_name = node.get_node_name()
            assert "DiscussionRoundNode" in node_name
            assert expected_name in node_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
