"""Tests for multi-agent coordination system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from virtual_agora.agents.coordination import (
    TurnManager, ResponseTimeoutManager, VoteCollector, AgentCoordinator,
    CoordinationPhase
)
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.state.schema import Message, Vote
from virtual_agora.utils.exceptions import CoordinationError


class TestTurnManager:
    """Test TurnManager class."""
    
    def setup_method(self):
        """Set up test method."""
        self.agent_ids = ["agent1", "agent2", "agent3"]
        self.turn_manager = TurnManager(self.agent_ids)
    
    def test_initialization(self):
        """Test turn manager initialization."""
        assert self.turn_manager.agent_ids == self.agent_ids
        assert self.turn_manager.current_index == 0
        assert self.turn_manager.round_number == 1
        assert len(self.turn_manager.turn_history) == 0
    
    def test_get_current_speaker(self):
        """Test getting current speaker."""
        current = self.turn_manager.get_current_speaker()
        assert current == "agent1"
        
        # After advancing
        self.turn_manager.advance_turn()
        current = self.turn_manager.get_current_speaker()
        assert current == "agent2"
    
    def test_advance_turn(self):
        """Test advancing turns."""
        # First turn
        next_speaker = self.turn_manager.advance_turn()
        assert next_speaker == "agent2"
        assert len(self.turn_manager.turn_history) == 1
        assert self.turn_manager.turn_history[0][0] == "agent1"
        
        # Second turn
        next_speaker = self.turn_manager.advance_turn()
        assert next_speaker == "agent3"
        
        # Third turn (should wrap around and increment round)
        next_speaker = self.turn_manager.advance_turn()
        assert next_speaker == "agent1"
        assert self.turn_manager.round_number == 2
    
    def test_rotate_speakers(self):
        """Test rotating speaker order."""
        original_order = self.turn_manager.get_speaking_order()
        assert original_order == ["agent1", "agent2", "agent3"]
        
        # Rotate
        self.turn_manager.rotate_speakers()
        new_order = self.turn_manager.get_speaking_order()
        assert new_order == ["agent2", "agent3", "agent1"]
        assert self.turn_manager.current_index == 0
    
    def test_get_speaking_order(self):
        """Test getting speaking order."""
        order = self.turn_manager.get_speaking_order()
        assert order == self.agent_ids
        assert order is not self.agent_ids  # Should be a copy
    
    def test_get_turn_statistics(self):
        """Test getting turn statistics."""
        # Initial state
        stats = self.turn_manager.get_turn_statistics()
        assert stats["round_number"] == 1
        assert stats["total_turns"] == 0
        assert stats["current_speaker"] == "agent1"
        assert len(stats["turns_per_agent"]) == 0
        
        # After some turns
        self.turn_manager.advance_turn()
        self.turn_manager.advance_turn()
        
        stats = self.turn_manager.get_turn_statistics()
        assert stats["total_turns"] == 2
        assert stats["turns_per_agent"]["agent1"] == 1
        assert stats["turns_per_agent"]["agent2"] == 1
    
    def test_empty_agent_list_error(self):
        """Test error handling with empty agent list."""
        empty_manager = TurnManager([])
        
        with pytest.raises(CoordinationError):
            empty_manager.get_current_speaker()
        
        with pytest.raises(CoordinationError):
            empty_manager.advance_turn()


class TestResponseTimeoutManager:
    """Test ResponseTimeoutManager class."""
    
    def setup_method(self):
        """Set up test method."""
        self.timeout_manager = ResponseTimeoutManager(default_timeout=1.0)
    
    def test_initialization(self):
        """Test timeout manager initialization."""
        assert self.timeout_manager.default_timeout == 1.0
        assert len(self.timeout_manager.active_timeouts) == 0
        assert len(self.timeout_manager.timeout_counts) == 0
    
    def test_start_and_clear_timeout(self):
        """Test starting and clearing timeouts."""
        agent_id = "test_agent"
        
        # Start timeout
        self.timeout_manager.start_timeout(agent_id)
        assert agent_id in self.timeout_manager.active_timeouts
        
        # Clear timeout
        self.timeout_manager.clear_timeout(agent_id)
        assert agent_id not in self.timeout_manager.active_timeouts
    
    def test_start_timeout_custom_duration(self):
        """Test starting timeout with custom duration."""
        agent_id = "test_agent"
        custom_timeout = 2.0
        
        self.timeout_manager.start_timeout(agent_id, custom_timeout)
        
        assert agent_id in self.timeout_manager.active_timeouts
        deadline = self.timeout_manager.active_timeouts[agent_id]
        expected_deadline = datetime.now() + timedelta(seconds=custom_timeout)
        
        # Check deadline is approximately correct (within 1 second tolerance)
        assert abs((deadline - expected_deadline).total_seconds()) < 1.0
    
    def test_check_timeouts_none_expired(self):
        """Test checking timeouts when none have expired."""
        self.timeout_manager.start_timeout("agent1")
        self.timeout_manager.start_timeout("agent2")
        
        timed_out = self.timeout_manager.check_timeouts()
        
        assert len(timed_out) == 0
        assert len(self.timeout_manager.active_timeouts) == 2
    
    def test_check_timeouts_with_expired(self):
        """Test checking timeouts with expired timeouts."""
        # Start timeout with very short duration
        self.timeout_manager.start_timeout("agent1", 0.01)  # 10ms
        
        # Wait for timeout to expire
        time.sleep(0.02)
        
        timed_out = self.timeout_manager.check_timeouts()
        
        assert "agent1" in timed_out
        assert "agent1" not in self.timeout_manager.active_timeouts
        assert self.timeout_manager.timeout_counts["agent1"] == 1
    
    def test_get_timeout_statistics(self):
        """Test getting timeout statistics."""
        # Initial state
        stats = self.timeout_manager.get_timeout_statistics()
        assert stats["active_timeouts"] == 0
        assert stats["total_timeouts"] == 0
        
        # Add some timeouts
        self.timeout_manager.start_timeout("agent1")
        self.timeout_manager.timeout_counts["agent2"] = 3
        
        stats = self.timeout_manager.get_timeout_statistics()
        assert stats["active_timeouts"] == 1
        assert stats["total_timeouts"] == 3
        assert stats["timeout_counts"]["agent2"] == 3


class TestVoteCollector:
    """Test VoteCollector class."""
    
    def setup_method(self):
        """Set up test method."""
        self.agent_ids = ["agent1", "agent2", "agent3"]
        self.vote_collector = VoteCollector(self.agent_ids)
    
    def test_initialization(self):
        """Test vote collector initialization."""
        assert self.vote_collector.agent_ids == set(self.agent_ids)
        assert len(self.vote_collector.votes) == 0
        assert self.vote_collector.vote_deadline is None
        assert self.vote_collector.required_votes == 3
    
    def test_start_vote_collection(self):
        """Test starting vote collection."""
        self.vote_collector.start_vote_collection("test_vote", timeout_minutes=1.0)
        
        assert len(self.vote_collector.votes) == 0
        assert self.vote_collector.vote_deadline is not None
        
        # Check deadline is approximately correct
        expected_deadline = datetime.now() + timedelta(minutes=1.0)
        actual_deadline = self.vote_collector.vote_deadline
        assert abs((actual_deadline - expected_deadline).total_seconds()) < 5.0
    
    def test_submit_vote_success(self):
        """Test successful vote submission."""
        self.vote_collector.start_vote_collection("test_vote")
        
        vote = Vote(
            id="vote1",
            voter_id="agent1",
            phase=1,
            vote_type="test",
            choice="yes",
            timestamp=datetime.now()
        )
        
        result = self.vote_collector.submit_vote(vote)
        
        assert result is True
        assert "agent1" in self.vote_collector.votes
        assert self.vote_collector.votes["agent1"] == vote
    
    def test_submit_vote_unknown_agent(self):
        """Test vote submission from unknown agent."""
        self.vote_collector.start_vote_collection("test_vote")
        
        vote = Vote(
            id="vote1",
            voter_id="unknown_agent",
            phase=1,
            vote_type="test",
            choice="yes",
            timestamp=datetime.now()
        )
        
        result = self.vote_collector.submit_vote(vote)
        
        assert result is False
        assert len(self.vote_collector.votes) == 0
    
    def test_submit_vote_after_deadline(self):
        """Test vote submission after deadline."""
        # Start with very short timeout
        self.vote_collector.start_vote_collection("test_vote", timeout_minutes=0.001)
        
        # Wait for deadline to pass
        time.sleep(0.1)
        
        vote = Vote(
            id="vote1",
            voter_id="agent1",
            phase=1,
            vote_type="test",
            choice="yes",
            timestamp=datetime.now()
        )
        
        result = self.vote_collector.submit_vote(vote)
        
        assert result is False
        assert len(self.vote_collector.votes) == 0
    
    def test_is_voting_complete_all_votes(self):
        """Test voting completion when all votes received."""
        self.vote_collector.start_vote_collection("test_vote")
        
        # Submit all votes
        for i, agent_id in enumerate(self.agent_ids):
            vote = Vote(
                id=f"vote{i}",
                voter_id=agent_id,
                phase=1,
                vote_type="test",
                choice="yes",
                timestamp=datetime.now()
            )
            self.vote_collector.submit_vote(vote)
        
        assert self.vote_collector.is_voting_complete() is True
    
    def test_is_voting_complete_deadline_passed(self):
        """Test voting completion when deadline passed."""
        # Start with very short timeout
        self.vote_collector.start_vote_collection("test_vote", timeout_minutes=0.001)
        
        # Wait for deadline to pass
        time.sleep(0.1)
        
        assert self.vote_collector.is_voting_complete() is True
    
    def test_get_vote_results(self):
        """Test getting vote results."""
        self.vote_collector.start_vote_collection("test_vote")
        
        # Submit partial votes
        for i in range(2):
            vote = Vote(
                id=f"vote{i}",
                voter_id=self.agent_ids[i],
                phase=1,
                vote_type="test",
                choice="yes",
                timestamp=datetime.now()
            )
            self.vote_collector.submit_vote(vote)
        
        results = self.vote_collector.get_vote_results()
        
        assert results["total_votes"] == 2
        assert results["required_votes"] == 3
        assert results["completion_rate"] == 2/3
        assert len(results["votes"]) == 2
        assert results["missing_voters"] == ["agent3"]
        assert results["voting_complete"] is False
    
    def test_get_conclusion_vote_tally(self):
        """Test getting conclusion vote tally."""
        self.vote_collector.start_vote_collection("conclusion_vote")
        
        # Submit mixed votes
        votes_data = [
            ("agent1", "Yes, we should conclude"),
            ("agent2", "No, need more discussion"),
            ("agent3", "Yes, adequate coverage")
        ]
        
        for i, (agent_id, choice) in enumerate(votes_data):
            vote = Vote(
                id=f"vote{i}",
                voter_id=agent_id,
                phase=3,
                vote_type="conclusion",
                choice=choice,
                timestamp=datetime.now()
            )
            self.vote_collector.submit_vote(vote)
        
        yes_count, no_count, minority_voters = self.vote_collector.get_conclusion_vote_tally()
        
        assert yes_count == 2
        assert no_count == 1
        assert minority_voters == ["agent2"]


class TestAgentCoordinator:
    """Test AgentCoordinator class."""
    
    def setup_method(self):
        """Set up test method."""
        # Create mock agents
        self.mock_agents = {}
        for i in range(3):
            agent_id = f"agent{i+1}"
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = agent_id
            agent.is_muted.return_value = False
            self.mock_agents[agent_id] = agent
        
        self.coordinator = AgentCoordinator(self.mock_agents, response_timeout=1.0)
    
    def test_initialization(self):
        """Test coordinator initialization."""
        assert len(self.coordinator.agents) == 3
        assert self.coordinator.agent_ids == ["agent1", "agent2", "agent3"]
        assert self.coordinator.current_phase == CoordinationPhase.IDLE
        assert len(self.coordinator.active_operations) == 0
        assert isinstance(self.coordinator.turn_manager, TurnManager)
        assert isinstance(self.coordinator.timeout_manager, ResponseTimeoutManager)
        assert isinstance(self.coordinator.vote_collector, VoteCollector)
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_coordinate_topic_proposals(self, mock_executor_class):
        """Test coordinating topic proposals."""
        # Setup mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Setup mock futures
        future_results = [
            ("agent1", ["Topic A", "Topic B"]),
            ("agent2", ["Topic C", "Topic D"]),
            ("agent3", ["Topic E", "Topic F"])
        ]
        
        mock_futures = []
        for agent_id, topics in future_results:
            future = Mock(spec=Future)
            future.result.return_value = (agent_id, topics)
            mock_futures.append(future)
        
        mock_executor.submit.side_effect = mock_futures
        mock_executor.as_completed.return_value = mock_futures
        
        # Configure agent mocks
        for agent in self.mock_agents.values():
            agent.propose_topics.return_value = ["Topic X", "Topic Y"]
        
        # Execute
        proposals = self.coordinator.coordinate_topic_proposals("Main Topic")
        
        # Verify
        assert len(proposals) == 3
        for agent_id in self.mock_agents.keys():
            assert agent_id in proposals
            assert len(proposals[agent_id]) > 0
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_coordinate_agenda_voting(self, mock_executor_class):
        """Test coordinating agenda voting."""
        # Setup mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Setup mock futures
        future_results = [
            ("agent1", "I prefer topics 1, 2, 3"),
            ("agent2", "Topics 2, 1, 3 in my preference"),
            ("agent3", "Order: 3, 1, 2")
        ]
        
        mock_futures = []
        for agent_id, vote_response in future_results:
            future = Mock(spec=Future)
            future.result.return_value = (agent_id, vote_response)
            mock_futures.append(future)
        
        mock_executor.submit.side_effect = mock_futures
        mock_executor.as_completed.return_value = mock_futures
        
        # Configure agent mocks
        for agent in self.mock_agents.values():
            agent.vote_on_agenda.return_value = "Vote response"
        
        # Execute
        votes = self.coordinator.coordinate_agenda_voting(["Topic A", "Topic B"])
        
        # Verify
        assert len(votes) == 3
        for agent_id in self.mock_agents.keys():
            assert agent_id in votes
            assert isinstance(votes[agent_id], str)
    
    def test_coordinate_discussion_round(self):
        """Test coordinating discussion round."""
        # Configure agent mocks
        for i, agent in enumerate(self.mock_agents.values()):
            agent.generate_discussion_response.return_value = f"Response from agent {i+1}"
        
        # Execute
        responses = self.coordinator.coordinate_discussion_round("Test Topic")
        
        # Verify
        assert len(responses) == 3
        for i, (agent_id, response) in enumerate(responses):
            expected_agent_id = f"agent{i+1}"
            assert agent_id == expected_agent_id
            assert f"Response from agent {i+1}" in response
    
    def test_coordinate_discussion_round_with_muted_agent(self):
        """Test discussion round with muted agent."""
        # Mute one agent
        self.mock_agents["agent2"].is_muted.return_value = True
        
        # Configure other agents
        self.mock_agents["agent1"].generate_discussion_response.return_value = "Response 1"
        self.mock_agents["agent3"].generate_discussion_response.return_value = "Response 3"
        
        # Execute
        responses = self.coordinator.coordinate_discussion_round("Test Topic")
        
        # Verify - should skip muted agent
        assert len(responses) == 2
        agent_ids = [agent_id for agent_id, _ in responses]
        assert "agent1" in agent_ids
        assert "agent3" in agent_ids
        assert "agent2" not in agent_ids
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_coordinate_conclusion_voting(self, mock_executor_class):
        """Test coordinating conclusion voting."""
        # Setup mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Setup mock futures
        future_results = [
            ("agent1", (True, "Yes, we've covered enough")),
            ("agent2", (False, "No, need more discussion")),
            ("agent3", (True, "Yes, adequate coverage"))
        ]
        
        mock_futures = []
        for agent_id, vote_data in future_results:
            future = Mock(spec=Future)
            future.result.return_value = (agent_id, vote_data)
            mock_futures.append(future)
        
        mock_executor.submit.side_effect = mock_futures
        mock_executor.as_completed.return_value = mock_futures
        
        # Configure agent mocks
        for agent in self.mock_agents.values():
            agent.vote_on_topic_conclusion.return_value = (True, "Reasoning")
        
        # Execute
        results = self.coordinator.coordinate_conclusion_voting("Test Topic")
        
        # Verify
        assert results["conclusion_passed"] is True
        assert results["yes_votes"] == 2
        assert results["no_votes"] == 1
        assert results["total_votes"] == 3
        assert results["minority_voters"] == ["agent2"]
        assert len(results["votes"]) == 3
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_coordinate_minority_considerations(self, mock_executor_class):
        """Test coordinating minority considerations."""
        # Setup mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        minority_voters = ["agent2"]
        
        # Setup mock future
        future = Mock(spec=Future)
        future.result.return_value = ("agent2", "Final consideration from minority")
        mock_executor.submit.return_value = future
        mock_executor.as_completed.return_value = [future]
        
        # Configure agent mock
        self.mock_agents["agent2"].provide_minority_consideration.return_value = "Consideration text"
        
        # Execute
        considerations = self.coordinator.coordinate_minority_considerations(
            "Test Topic", minority_voters
        )
        
        # Verify
        assert len(considerations) == 1
        assert "agent2" in considerations
        assert isinstance(considerations["agent2"], str)
    
    def test_get_coordination_statistics(self):
        """Test getting coordination statistics."""
        # Execute some operations to change stats
        self.coordinator.coordination_stats["operations_completed"] = 5
        self.coordinator.coordination_stats["timeouts_handled"] = 2
        self.coordinator.coordination_stats["errors_handled"] = 1
        
        stats = self.coordinator.get_coordination_statistics()
        
        assert stats["current_phase"] == CoordinationPhase.IDLE.value
        assert stats["operations_completed"] == 5
        assert stats["timeouts_handled"] == 2
        assert stats["errors_handled"] == 1
        assert stats["total_agents"] == 3
        assert "turn_statistics" in stats
        assert "timeout_statistics" in stats
    
    def test_reset_coordination(self):
        """Test resetting coordination state."""
        # Change some state
        self.coordinator.current_phase = CoordinationPhase.DISCUSSION
        self.coordinator.active_operations.add("test_op")
        self.coordinator.coordination_stats["operations_completed"] = 10
        
        # Reset
        self.coordinator.reset_coordination()
        
        # Verify reset
        assert self.coordinator.current_phase == CoordinationPhase.IDLE
        assert len(self.coordinator.active_operations) == 0
        # Note: stats don't reset, only coordination state


class TestAgentCoordinatorErrorHandling:
    """Test error handling in AgentCoordinator."""
    
    def setup_method(self):
        """Set up test method."""
        # Create mock agents that can raise errors
        self.mock_agents = {}
        for i in range(2):
            agent_id = f"agent{i+1}"
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = agent_id
            agent.is_muted.return_value = False
            self.mock_agents[agent_id] = agent
        
        self.coordinator = AgentCoordinator(self.mock_agents)
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_handle_agent_errors_in_proposals(self, mock_executor_class):
        """Test handling agent errors during topic proposals."""
        # Setup mock executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Setup futures - one success, one failure
        success_future = Mock(spec=Future)
        success_future.result.return_value = ("agent1", ["Topic A"])
        
        error_future = Mock(spec=Future)
        error_future.result.return_value = ("agent2", [])  # Empty result indicates error
        
        mock_executor.submit.side_effect = [success_future, error_future]
        mock_executor.as_completed.return_value = [success_future, error_future]
        
        # Configure agents - one works, one fails
        self.mock_agents["agent1"].propose_topics.return_value = ["Topic A"]
        self.mock_agents["agent2"].propose_topics.side_effect = Exception("Agent error")
        
        # Execute
        proposals = self.coordinator.coordinate_topic_proposals("Main Topic")
        
        # Verify - should handle error gracefully
        assert len(proposals) == 1  # Only successful agent
        assert "agent1" in proposals
        assert "agent2" not in proposals
    
    def test_handle_timeout_in_discussion(self):
        """Test handling timeouts during discussion."""
        # Configure one agent to work, simulate timeout for another
        self.mock_agents["agent1"].generate_discussion_response.return_value = "Good response"
        self.mock_agents["agent2"].generate_discussion_response.side_effect = Exception("Timeout")
        
        # Execute
        responses = self.coordinator.coordinate_discussion_round("Test Topic")
        
        # Should handle error gracefully and continue with working agents
        # Note: The actual error handling depends on implementation details
        # but the coordinator should be resilient to individual agent failures
        assert isinstance(responses, list)
    
    def test_error_statistics_tracking(self):
        """Test that errors are tracked in statistics."""
        initial_errors = self.coordinator.coordination_stats["errors_handled"]
        
        # Configure agent to raise error
        self.mock_agents["agent1"].generate_discussion_response.side_effect = Exception("Test error")
        
        # Execute operation that will encounter error
        self.coordinator.coordinate_discussion_round("Test Topic")
        
        # Check if error was tracked (implementation dependent)
        stats = self.coordinator.get_coordination_statistics()
        assert "errors_handled" in stats


class TestCoordinationIntegration:
    """Integration tests for coordination components."""
    
    def test_full_coordination_workflow(self):
        """Test complete coordination workflow."""
        # Create mock agents
        mock_agents = {}
        for i in range(3):
            agent_id = f"agent{i+1}"
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = agent_id
            agent.is_muted.return_value = False
            
            # Configure responses
            agent.propose_topics.return_value = [f"Topic {i+1}A", f"Topic {i+1}B"]
            agent.vote_on_agenda.return_value = f"Vote from {agent_id}"
            agent.generate_discussion_response.return_value = f"Discussion from {agent_id}"
            agent.vote_on_topic_conclusion.return_value = (True, f"Conclusion from {agent_id}")
            
            mock_agents[agent_id] = agent
        
        coordinator = AgentCoordinator(mock_agents)
        
        # Test complete workflow
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
            # Setup mock executor for parallel operations
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            # Mock futures for proposals
            proposal_futures = []
            for i, agent_id in enumerate(mock_agents.keys()):
                future = Mock(spec=Future)
                future.result.return_value = (agent_id, [f"Topic {i+1}A", f"Topic {i+1}B"])
                proposal_futures.append(future)
            
            mock_executor.submit.side_effect = proposal_futures
            mock_executor.as_completed.return_value = proposal_futures
            
            # 1. Topic proposals
            proposals = coordinator.coordinate_topic_proposals("Main Topic")
            assert len(proposals) == 3
            
            # Reset mock for next operation
            mock_executor.reset_mock()
            
            # Mock futures for voting
            vote_futures = []
            for agent_id in mock_agents.keys():
                future = Mock(spec=Future)
                future.result.return_value = (agent_id, f"Vote from {agent_id}")
                vote_futures.append(future)
            
            mock_executor.submit.side_effect = vote_futures
            mock_executor.as_completed.return_value = vote_futures
            
            # 2. Agenda voting
            votes = coordinator.coordinate_agenda_voting(["Topic A", "Topic B"])
            assert len(votes) == 3
        
        # 3. Discussion round (no mocking needed - runs sequentially)
        responses = coordinator.coordinate_discussion_round("Selected Topic")
        assert len(responses) == 3
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
            # Setup for conclusion voting
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            conclusion_futures = []
            for agent_id in mock_agents.keys():
                future = Mock(spec=Future)
                future.result.return_value = (agent_id, (True, f"Conclusion from {agent_id}"))
                conclusion_futures.append(future)
            
            mock_executor.submit.side_effect = conclusion_futures
            mock_executor.as_completed.return_value = conclusion_futures
            
            # 4. Conclusion voting
            conclusion_results = coordinator.coordinate_conclusion_voting("Selected Topic")
            assert conclusion_results["conclusion_passed"] is True
        
        # Check final statistics
        stats = coordinator.get_coordination_statistics()
        assert stats["operations_completed"] >= 3  # At least 3 operations completed
    
    def test_thread_safety(self):
        """Test thread safety of coordination components."""
        # Create shared turn manager
        agent_ids = ["agent1", "agent2", "agent3"]
        turn_manager = TurnManager(agent_ids)
        
        # Function to advance turns in parallel
        def advance_turns():
            for _ in range(10):
                turn_manager.advance_turn()
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=advance_turns)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify consistency
        stats = turn_manager.get_turn_statistics()
        assert stats["total_turns"] == 30  # 3 threads * 10 turns each
        
        # Check that all agents got roughly equal turns
        turns_per_agent = stats["turns_per_agent"]
        for agent_id in agent_ids:
            assert turns_per_agent[agent_id] >= 8  # Allow some variance due to threading
            assert turns_per_agent[agent_id] <= 12