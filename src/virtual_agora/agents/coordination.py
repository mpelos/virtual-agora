"""Multi-agent coordination system for Virtual Agora.

This module provides coordination mechanisms for managing multiple discussion
agents in structured debate scenarios.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import deque, defaultdict
import threading
import time

from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.state.schema import Message, Vote, VirtualAgoraState
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import CoordinationError

logger = get_logger(__name__)


class CoordinationPhase(Enum):
    """Phases of agent coordination."""
    IDLE = "idle"
    PROPOSAL_COLLECTION = "proposal_collection"
    AGENDA_VOTING = "agenda_voting"
    DISCUSSION = "discussion"
    CONCLUSION_VOTING = "conclusion_voting"
    FINAL_CONSIDERATIONS = "final_considerations"


class TurnManager:
    """Manages turn-based speaking order for agents."""
    
    def __init__(self, agent_ids: List[str]):
        """Initialize turn manager.
        
        Args:
            agent_ids: List of agent IDs to manage
        """
        self.agent_ids = agent_ids.copy()
        self.current_index = 0
        self.round_number = 1
        self.turn_history: List[Tuple[str, datetime]] = []
        
        logger.info(f"Initialized TurnManager with {len(agent_ids)} agents")
    
    def get_current_speaker(self) -> str:
        """Get the ID of the current speaker.
        
        Returns:
            Agent ID of current speaker
        """
        if not self.agent_ids:
            raise CoordinationError("No agents available for speaking")
        
        return self.agent_ids[self.current_index]
    
    def advance_turn(self) -> str:
        """Advance to the next speaker.
        
        Returns:
            Agent ID of the next speaker
        """
        if not self.agent_ids:
            raise CoordinationError("No agents available for speaking")
        
        # Record current turn
        current_speaker = self.agent_ids[self.current_index]
        self.turn_history.append((current_speaker, datetime.now()))
        
        # Advance to next agent
        self.current_index = (self.current_index + 1) % len(self.agent_ids)
        
        # If we've completed a full round, increment round number
        if self.current_index == 0:
            self.round_number += 1
            logger.debug(f"Starting round {self.round_number}")
        
        next_speaker = self.agent_ids[self.current_index]
        logger.debug(f"Turn advanced from {current_speaker} to {next_speaker}")
        
        return next_speaker
    
    def rotate_speakers(self) -> None:
        """Rotate the speaking order for the next round."""
        if not self.agent_ids:
            return
        
        # Move first speaker to end
        first_speaker = self.agent_ids.pop(0)
        self.agent_ids.append(first_speaker)
        self.current_index = 0
        
        logger.debug(f"Rotated speaking order: {self.agent_ids}")
    
    def get_speaking_order(self) -> List[str]:
        """Get the current speaking order.
        
        Returns:
            List of agent IDs in speaking order
        """
        return self.agent_ids.copy()
    
    def get_turn_statistics(self) -> Dict[str, Any]:
        """Get statistics about turns taken.
        
        Returns:
            Dictionary with turn statistics
        """
        turn_counts = defaultdict(int)
        for agent_id, _ in self.turn_history:
            turn_counts[agent_id] += 1
        
        return {
            "round_number": self.round_number,
            "total_turns": len(self.turn_history),
            "turns_per_agent": dict(turn_counts),
            "current_speaker": self.get_current_speaker() if self.agent_ids else None
        }


class ResponseTimeoutManager:
    """Manages timeouts for agent responses."""
    
    def __init__(self, default_timeout: float = 30.0):
        """Initialize timeout manager.
        
        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout
        self.active_timeouts: Dict[str, datetime] = {}
        self.timeout_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def start_timeout(self, agent_id: str, timeout: Optional[float] = None) -> None:
        """Start timeout for an agent.
        
        Args:
            agent_id: ID of the agent
            timeout: Timeout duration in seconds (uses default if None)
        """
        timeout_duration = timeout or self.default_timeout
        deadline = datetime.now() + timedelta(seconds=timeout_duration)
        
        with self._lock:
            self.active_timeouts[agent_id] = deadline
        
        logger.debug(f"Started timeout for agent {agent_id}: {timeout_duration}s")
    
    def clear_timeout(self, agent_id: str) -> None:
        """Clear timeout for an agent.
        
        Args:
            agent_id: ID of the agent
        """
        with self._lock:
            self.active_timeouts.pop(agent_id, None)
        
        logger.debug(f"Cleared timeout for agent {agent_id}")
    
    def check_timeouts(self) -> List[str]:
        """Check for timed out agents.
        
        Returns:
            List of agent IDs that have timed out
        """
        now = datetime.now()
        timed_out = []
        
        with self._lock:
            for agent_id, deadline in list(self.active_timeouts.items()):
                if now > deadline:
                    timed_out.append(agent_id)
                    self.timeout_counts[agent_id] += 1
                    del self.active_timeouts[agent_id]
        
        if timed_out:
            logger.warning(f"Agents timed out: {timed_out}")
        
        return timed_out
    
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get timeout statistics.
        
        Returns:
            Dictionary with timeout statistics
        """
        return {
            "active_timeouts": len(self.active_timeouts),
            "timeout_counts": dict(self.timeout_counts),
            "total_timeouts": sum(self.timeout_counts.values())
        }


class VoteCollector:
    """Collects and manages votes from multiple agents."""
    
    def __init__(self, agent_ids: List[str]):
        """Initialize vote collector.
        
        Args:
            agent_ids: List of agent IDs that can vote
        """
        self.agent_ids = set(agent_ids)
        self.votes: Dict[str, Vote] = {}
        self.vote_deadline: Optional[datetime] = None
        self.required_votes = len(agent_ids)
        self._lock = threading.Lock()
    
    def start_vote_collection(
        self,
        vote_type: str,
        timeout_minutes: float = 5.0
    ) -> None:
        """Start collecting votes.
        
        Args:
            vote_type: Type of vote being collected
            timeout_minutes: Timeout for vote collection in minutes
        """
        with self._lock:
            self.votes.clear()
            self.vote_deadline = datetime.now() + timedelta(minutes=timeout_minutes)
        
        logger.info(f"Started {vote_type} vote collection for {self.required_votes} agents")
    
    def submit_vote(self, vote: Vote) -> bool:
        """Submit a vote from an agent.
        
        Args:
            vote: Vote object from the agent
            
        Returns:
            True if vote was accepted, False otherwise
        """
        if vote.voter_id not in self.agent_ids:
            logger.warning(f"Vote from unknown agent {vote.voter_id}")
            return False
        
        if self.vote_deadline and datetime.now() > self.vote_deadline:
            logger.warning(f"Vote from {vote.voter_id} received after deadline")
            return False
        
        with self._lock:
            self.votes[vote.voter_id] = vote
        
        logger.debug(f"Received vote from {vote.voter_id}")
        return True
    
    def is_voting_complete(self) -> bool:
        """Check if voting is complete.
        
        Returns:
            True if all votes collected or deadline passed
        """
        with self._lock:
            all_votes_received = len(self.votes) >= self.required_votes
            deadline_passed = (
                self.vote_deadline and datetime.now() > self.vote_deadline
            )
        
        return all_votes_received or deadline_passed
    
    def get_vote_results(self) -> Dict[str, Any]:
        """Get the current vote results.
        
        Returns:
            Dictionary with vote results and statistics
        """
        with self._lock:
            votes_copy = self.votes.copy()
        
        results = {
            "total_votes": len(votes_copy),
            "required_votes": self.required_votes,
            "completion_rate": len(votes_copy) / self.required_votes,
            "votes": votes_copy,
            "missing_voters": list(self.agent_ids - set(votes_copy.keys())),
            "voting_complete": self.is_voting_complete()
        }
        
        return results
    
    def get_conclusion_vote_tally(self) -> Tuple[int, int, List[str]]:
        """Get tally for topic conclusion votes.
        
        Returns:
            Tuple of (yes_count, no_count, minority_voters)
        """
        yes_votes = 0
        no_votes = 0
        minority_voters = []
        
        with self._lock:
            for vote in self.votes.values():
                if vote.choice.lower().startswith('yes'):
                    yes_votes += 1
                elif vote.choice.lower().startswith('no'):
                    no_votes += 1
                    minority_voters.append(vote.voter_id)
        
        return yes_votes, no_votes, minority_voters


class AgentCoordinator:
    """Coordinates multiple discussion agents in structured interactions."""
    
    def __init__(
        self,
        agents: Dict[str, DiscussionAgent],
        response_timeout: float = 30.0
    ):
        """Initialize agent coordinator.
        
        Args:
            agents: Dictionary of agent ID to DiscussionAgent
            response_timeout: Timeout for agent responses in seconds
        """
        self.agents = agents
        self.agent_ids = list(agents.keys())
        
        # Coordination components
        self.turn_manager = TurnManager(self.agent_ids)
        self.timeout_manager = ResponseTimeoutManager(response_timeout)
        self.vote_collector = VoteCollector(self.agent_ids)
        
        # State tracking
        self.current_phase = CoordinationPhase.IDLE
        self.active_operations: Set[str] = set()
        self.coordination_stats = {
            "operations_completed": 0,
            "timeouts_handled": 0,
            "errors_handled": 0
        }
        
        # Thread safety
        self._coordination_lock = threading.Lock()
        
        logger.info(f"Initialized AgentCoordinator with {len(agents)} agents")
    
    def coordinate_topic_proposals(
        self,
        main_topic: str,
        context_messages: Optional[List[Message]] = None
    ) -> Dict[str, List[str]]:
        """Coordinate topic proposal collection from all agents.
        
        Args:
            main_topic: Main topic for proposals
            context_messages: Optional context messages
            
        Returns:
            Dictionary mapping agent IDs to their topic proposals
        """
        with self._coordination_lock:
            self.current_phase = CoordinationPhase.PROPOSAL_COLLECTION
        
        logger.info(f"Coordinating topic proposals for: {main_topic}")
        
        proposals = {}
        failed_agents = []
        
        # Collect proposals from all agents in parallel
        def collect_proposal(agent_id: str) -> Tuple[str, List[str]]:
            agent = self.agents[agent_id]
            try:
                self.timeout_manager.start_timeout(agent_id)
                topics = agent.propose_topics(main_topic, context_messages)
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, topics
            except Exception as e:
                logger.error(f"Error collecting proposal from {agent_id}: {e}")
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, []
        
        # Use ThreadPoolExecutor for parallel execution
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(collect_proposal, agent_id): agent_id
                for agent_id in self.agent_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_agent, timeout=60):
                agent_id, topics = future.result()
                if topics:
                    proposals[agent_id] = topics
                else:
                    failed_agents.append(agent_id)
        
        # Handle timeouts
        timed_out = self.timeout_manager.check_timeouts()
        for agent_id in timed_out:
            failed_agents.append(agent_id)
            self.coordination_stats["timeouts_handled"] += 1
        
        self.coordination_stats["operations_completed"] += 1
        
        logger.info(
            f"Collected proposals from {len(proposals)}/{len(self.agents)} agents. "
            f"Failed: {len(failed_agents)}"
        )
        
        return proposals
    
    def coordinate_agenda_voting(
        self,
        proposed_topics: List[str],
        context_messages: Optional[List[Message]] = None
    ) -> Dict[str, str]:
        """Coordinate agenda voting from all agents.
        
        Args:
            proposed_topics: List of topics to vote on
            context_messages: Optional context messages
            
        Returns:
            Dictionary mapping agent IDs to their vote responses
        """
        with self._coordination_lock:
            self.current_phase = CoordinationPhase.AGENDA_VOTING
        
        logger.info(f"Coordinating agenda voting for {len(proposed_topics)} topics")
        
        self.vote_collector.start_vote_collection("agenda", timeout_minutes=3.0)
        votes = {}
        
        def collect_vote(agent_id: str) -> Tuple[str, str]:
            agent = self.agents[agent_id]
            try:
                self.timeout_manager.start_timeout(agent_id)
                vote_response = agent.vote_on_agenda(proposed_topics, context_messages)
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, vote_response
            except Exception as e:
                logger.error(f"Error collecting agenda vote from {agent_id}: {e}")
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, "Unable to vote due to technical difficulties."
        
        # Collect votes in parallel
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(collect_vote, agent_id): agent_id
                for agent_id in self.agent_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_agent, timeout=180):
                agent_id, vote_response = future.result()
                votes[agent_id] = vote_response
        
        # Handle timeouts
        timed_out = self.timeout_manager.check_timeouts()
        for agent_id in timed_out:
            self.coordination_stats["timeouts_handled"] += 1
        
        self.coordination_stats["operations_completed"] += 1
        
        logger.info(f"Collected agenda votes from {len(votes)}/{len(self.agents)} agents")
        return votes
    
    def coordinate_discussion_round(
        self,
        topic: str,
        context_messages: Optional[List[Message]] = None
    ) -> List[Tuple[str, str]]:
        """Coordinate a discussion round with turn-based responses.
        
        Args:
            topic: Current discussion topic
            context_messages: Context from previous rounds
            
        Returns:
            List of (agent_id, response) tuples in turn order
        """
        with self._coordination_lock:
            self.current_phase = CoordinationPhase.DISCUSSION
        
        logger.info(f"Coordinating discussion round for topic: {topic}")
        
        responses = []
        speaking_order = self.turn_manager.get_speaking_order()
        
        for agent_id in speaking_order:
            agent = self.agents[agent_id]
            
            # Skip muted agents
            if agent.is_muted():
                logger.info(f"Skipping muted agent {agent_id}")
                continue
            
            try:
                self.timeout_manager.start_timeout(agent_id)
                response = agent.generate_discussion_response(
                    topic, context_messages, phase=2
                )
                self.timeout_manager.clear_timeout(agent_id)
                
                if response:  # Only include non-empty responses
                    responses.append((agent_id, response))
                    
            except Exception as e:
                logger.error(f"Error in discussion response from {agent_id}: {e}")
                self.timeout_manager.clear_timeout(agent_id)
                self.coordination_stats["errors_handled"] += 1
        
        # Advance turn order for next round
        self.turn_manager.rotate_speakers()
        self.coordination_stats["operations_completed"] += 1
        
        logger.info(f"Completed discussion round with {len(responses)} responses")
        return responses
    
    def coordinate_conclusion_voting(
        self,
        topic: str,
        context_messages: Optional[List[Message]] = None
    ) -> Dict[str, Any]:
        """Coordinate voting on topic conclusion.
        
        Args:
            topic: Topic to vote on concluding
            context_messages: Context from discussion
            
        Returns:
            Dictionary with vote results and analysis
        """
        with self._coordination_lock:
            self.current_phase = CoordinationPhase.CONCLUSION_VOTING
        
        logger.info(f"Coordinating conclusion voting for topic: {topic}")
        
        self.vote_collector.start_vote_collection("conclusion", timeout_minutes=2.0)
        votes = {}
        
        def collect_conclusion_vote(agent_id: str) -> Tuple[str, Tuple[bool, str]]:
            agent = self.agents[agent_id]
            try:
                self.timeout_manager.start_timeout(agent_id)
                vote_result, reasoning = agent.vote_on_topic_conclusion(
                    topic, context_messages
                )
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, (vote_result, reasoning)
            except Exception as e:
                logger.error(f"Error collecting conclusion vote from {agent_id}: {e}")
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, (False, "Unable to vote due to technical difficulties.")
        
        # Collect votes in parallel
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(collect_conclusion_vote, agent_id): agent_id
                for agent_id in self.agent_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_agent, timeout=120):
                agent_id, (vote_result, reasoning) = future.result()
                votes[agent_id] = {
                    "vote": "Yes" if vote_result else "No",
                    "reasoning": reasoning
                }
        
        # Analyze results
        yes_votes = sum(1 for v in votes.values() if v["vote"] == "Yes")
        no_votes = sum(1 for v in votes.values() if v["vote"] == "No")
        total_votes = len(votes)
        
        # Majority + 1 rule
        conclusion_passed = yes_votes > (total_votes / 2)
        minority_voters = [
            agent_id for agent_id, vote_data in votes.items()
            if vote_data["vote"] == "No"
        ] if conclusion_passed else []
        
        results = {
            "conclusion_passed": conclusion_passed,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "total_votes": total_votes,
            "minority_voters": minority_voters,
            "votes": votes
        }
        
        # Handle timeouts
        timed_out = self.timeout_manager.check_timeouts()
        for agent_id in timed_out:
            self.coordination_stats["timeouts_handled"] += 1
        
        self.coordination_stats["operations_completed"] += 1
        
        logger.info(
            f"Conclusion voting completed: {yes_votes} Yes, {no_votes} No. "
            f"Passed: {conclusion_passed}"
        )
        
        return results
    
    def coordinate_minority_considerations(
        self,
        topic: str,
        minority_voters: List[str],
        context_messages: Optional[List[Message]] = None
    ) -> Dict[str, str]:
        """Coordinate final considerations from minority voters.
        
        Args:
            topic: Topic that was concluded
            minority_voters: List of agent IDs who voted against conclusion
            context_messages: Full discussion context
            
        Returns:
            Dictionary mapping agent IDs to their final considerations
        """
        with self._coordination_lock:
            self.current_phase = CoordinationPhase.FINAL_CONSIDERATIONS
        
        logger.info(f"Coordinating minority considerations from {len(minority_voters)} agents")
        
        considerations = {}
        
        def collect_consideration(agent_id: str) -> Tuple[str, str]:
            agent = self.agents[agent_id]
            try:
                self.timeout_manager.start_timeout(agent_id)
                consideration = agent.provide_minority_consideration(
                    topic, context_messages
                )
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, consideration
            except Exception as e:
                logger.error(f"Error collecting consideration from {agent_id}: {e}")
                self.timeout_manager.clear_timeout(agent_id)
                return agent_id, "Unable to provide final consideration due to technical difficulties."
        
        # Collect considerations in parallel
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(minority_voters)) as executor:
            future_to_agent = {
                executor.submit(collect_consideration, agent_id): agent_id
                for agent_id in minority_voters
            }
            
            for future in concurrent.futures.as_completed(future_to_agent, timeout=90):
                agent_id, consideration = future.result()
                if consideration:
                    considerations[agent_id] = consideration
        
        self.coordination_stats["operations_completed"] += 1
        
        logger.info(f"Collected minority considerations from {len(considerations)} agents")
        return considerations
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics.
        
        Returns:
            Dictionary with coordination statistics
        """
        return {
            "current_phase": self.current_phase.value,
            "operations_completed": self.coordination_stats["operations_completed"],
            "timeouts_handled": self.coordination_stats["timeouts_handled"],
            "errors_handled": self.coordination_stats["errors_handled"],
            "turn_statistics": self.turn_manager.get_turn_statistics(),
            "timeout_statistics": self.timeout_manager.get_timeout_statistics(),
            "active_operations": len(self.active_operations),
            "total_agents": len(self.agents)
        }
    
    def reset_coordination(self) -> None:
        """Reset coordination state for new session."""
        with self._coordination_lock:
            self.current_phase = CoordinationPhase.IDLE
            self.active_operations.clear()
            self.turn_manager = TurnManager(self.agent_ids)
            self.timeout_manager = ResponseTimeoutManager()
            self.vote_collector = VoteCollector(self.agent_ids)
            
        logger.info("Coordination state reset")