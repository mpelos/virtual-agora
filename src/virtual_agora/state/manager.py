"""State management for Virtual Agora sessions.

This module provides a high-level interface for managing Virtual Agora
state, including initialization, updates, and queries.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy

from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.state.schema import (
    VirtualAgoraState,
    AgentInfo,
    Message,
    Vote,
    PhaseTransition,
    VoteRound,
    TopicInfo,
)
from virtual_agora.state.validators import StateValidator
from virtual_agora.utils.logging import get_logger
from virtual_agora.utils.exceptions import StateError, ValidationError


logger = get_logger(__name__)


class StateManager:
    """Manages the state of a Virtual Agora session."""
    
    def __init__(self, config: VirtualAgoraConfig):
        """Initialize state manager with configuration.
        
        Args:
            config: Virtual Agora configuration
        """
        self.config = config
        self.validator = StateValidator()
        self._state: Optional[VirtualAgoraState] = None
        self._message_counter = 0
        self._vote_counter = 0
        
    def initialize_state(self, session_id: Optional[str] = None) -> VirtualAgoraState:
        """Initialize a new state for a session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Initialized state
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + \
                        str(uuid.uuid4())[:8]
        
        now = datetime.now()
        
        # Create config hash for validation
        config_str = f"{self.config.moderator.model}:{self.config.moderator.provider}"
        for agent in self.config.agents:
            config_str += f":{agent.model}:{agent.provider}:{agent.count}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Initialize agents
        agents: Dict[str, AgentInfo] = {}
        
        # Add moderator
        moderator_id = "moderator"
        agents[moderator_id] = AgentInfo(
            id=moderator_id,
            model=self.config.moderator.model,
            provider=self.config.moderator.provider.value,
            role="moderator",
            message_count=0,
            created_at=now
        )
        
        # Add participants
        agent_counter = 0
        speaking_order = []
        for agent_config in self.config.agents:
            for i in range(agent_config.count):
                agent_counter += 1
                agent_id = f"{agent_config.model}_{agent_counter}"
                agents[agent_id] = AgentInfo(
                    id=agent_id,
                    model=agent_config.model,
                    provider=agent_config.provider.value,
                    role="participant",
                    message_count=0,
                    created_at=now
                )
                speaking_order.append(agent_id)
        
        # Initialize state
        self._state = VirtualAgoraState(
            # Session metadata
            session_id=session_id,
            start_time=now,
            config_hash=config_hash,
            
            # Phase management
            current_phase=0,
            phase_history=[],
            phase_start_time=now,
            
            # Topic management
            active_topic=None,
            topic_queue=[],
            proposed_topics=[],
            topics_info={},
            completed_topics=[],
            
            # Agent management
            agents=agents,
            moderator_id=moderator_id,
            current_speaker_id=moderator_id,  # Moderator starts
            speaking_order=speaking_order,
            next_speaker_index=0,
            
            # Discussion history
            messages=[],
            last_message_id="0",
            
            # Voting system
            active_vote=None,
            vote_history=[],
            votes=[],
            
            # Consensus tracking
            consensus_proposals={},
            consensus_reached={},
            
            # Generated content
            phase_summaries={},
            topic_summaries={},
            consensus_summaries={},
            final_report=None,
            
            # Runtime statistics
            total_messages=0,
            messages_by_phase={i: 0 for i in range(5)},
            messages_by_agent={aid: 0 for aid in agents},
            messages_by_topic={},
            vote_participation_rate={},
            
            # Error tracking
            last_error=None,
            error_count=0,
            warnings=[]
        )
        
        logger.info(f"Initialized state for session {session_id}")
        logger.debug(f"Agents: {len(agents)} total ({len(speaking_order)} participants)")
        
        return self._state
    
    @property
    def state(self) -> VirtualAgoraState:
        """Get current state.
        
        Returns:
            Current state
            
        Raises:
            StateError: If state is not initialized
        """
        if self._state is None:
            raise StateError("State not initialized")
        return self._state
    
    def transition_phase(self, new_phase: int, reason: str = "Normal progression") -> None:
        """Transition to a new phase.
        
        Args:
            new_phase: Target phase (0-4)
            reason: Reason for transition
            
        Raises:
            ValidationError: If transition is invalid
        """
        # Validate transition
        self.validator.validate_phase_transition(self.state, new_phase)
        
        old_phase = self.state["current_phase"]
        now = datetime.now()
        
        # Record transition
        transition = PhaseTransition(
            from_phase=old_phase,
            to_phase=new_phase,
            timestamp=now,
            reason=reason,
            triggered_by="system"
        )
        self.state["phase_history"].append(transition)
        
        # Update phase
        self.state["current_phase"] = new_phase
        self.state["phase_start_time"] = now
        
        # Phase-specific setup
        if new_phase == 1:
            # Agenda Setting: Participants will speak
            self._set_next_speaker()
        elif new_phase == 2:
            # Discussion: Activate first topic
            if self.state["topic_queue"]:
                self.activate_topic(self.state["topic_queue"][0])
        elif new_phase == 4:
            # Summary: Moderator speaks
            self.state["current_speaker_id"] = self.state["moderator_id"]
        
        logger.info(
            f"Phase transition: {StateValidator.PHASE_NAMES[old_phase]} -> "
            f"{StateValidator.PHASE_NAMES[new_phase]} ({reason})"
        )
    
    def add_message(
        self,
        speaker_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a message to the discussion.
        
        Args:
            speaker_id: ID of the speaking agent
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created message
            
        Raises:
            ValidationError: If speaker is not allowed
        """
        # Validate speaker
        self.validator.validate_speaker(self.state, speaker_id)
        
        # Generate message ID
        self._message_counter += 1
        message_id = f"msg_{self._message_counter:06d}"
        
        # Create message
        agent = self.state["agents"][speaker_id]
        message = Message(
            id=message_id,
            speaker_id=speaker_id,
            speaker_role=agent["role"],
            content=content,
            timestamp=datetime.now(),
            phase=self.state["current_phase"],
            topic=self.state["active_topic"]
        )
        if metadata:
            message["metadata"] = metadata
        
        # Validate message format
        self.validator.validate_message_format(message)
        
        # Add to state
        self.state["messages"].append(message)
        self.state["last_message_id"] = message_id
        
        # Update statistics
        self.state["total_messages"] += 1
        self.state["messages_by_phase"][self.state["current_phase"]] += 1
        self.state["messages_by_agent"][speaker_id] += 1
        self.state["agents"][speaker_id]["message_count"] += 1
        
        if self.state["active_topic"]:
            topic = self.state["active_topic"]
            self.state["messages_by_topic"][topic] = \
                self.state["messages_by_topic"].get(topic, 0) + 1
            if topic in self.state["topics_info"]:
                self.state["topics_info"][topic]["message_count"] += 1
        
        logger.debug(f"Message {message_id} added from {speaker_id}")
        
        # Advance to next speaker (except in phases 0 and 4)
        if self.state["current_phase"] not in [0, 4]:
            self._set_next_speaker()
        
        return message
    
    def start_vote(
        self,
        vote_type: str,
        options: List[str],
        required_votes: Optional[int] = None
    ) -> VoteRound:
        """Start a new voting round.
        
        Args:
            vote_type: Type of vote
            options: Available options
            required_votes: Number of required votes (default: all participants)
            
        Returns:
            Created vote round
            
        Raises:
            StateError: If there's already an active vote
        """
        if self.state["active_vote"]:
            raise StateError("There is already an active vote")
        
        # Generate vote ID
        self._vote_counter += 1
        vote_id = f"vote_{self._vote_counter:04d}"
        
        # Calculate required votes if not specified
        if required_votes is None:
            required_votes = len([
                a for a in self.state["agents"].values()
                if a["role"] == "participant"
            ])
        
        # Create vote round
        vote_round = VoteRound(
            id=vote_id,
            phase=self.state["current_phase"],
            vote_type=vote_type,
            options=options,
            start_time=datetime.now(),
            end_time=None,
            required_votes=required_votes,
            received_votes=0,
            result=None,
            status="active"
        )
        
        self.state["active_vote"] = vote_round
        logger.info(f"Started {vote_type} vote with options: {options}")
        
        return vote_round
    
    def cast_vote(self, voter_id: str, choice: str) -> None:
        """Record a vote from an agent.
        
        Args:
            voter_id: ID of the voting agent
            choice: Vote choice
            
        Raises:
            ValidationError: If vote is invalid
        """
        # Validate vote
        self.validator.validate_vote(self.state, voter_id, choice)
        
        vote_round = self.state["active_vote"]
        assert vote_round is not None  # Validated above
        
        # Create vote record
        vote = Vote(
            id=f"{vote_round['id']}_{voter_id}",
            voter_id=voter_id,
            phase=self.state["current_phase"],
            vote_type=vote_round["vote_type"],
            choice=choice,
            timestamp=datetime.now(),
            metadata={"vote_round_id": vote_round["id"]}
        )
        
        self.state["votes"].append(vote)
        vote_round["received_votes"] += 1
        
        logger.debug(f"Vote recorded: {voter_id} -> {choice}")
        
        # Check if voting is complete
        if vote_round["received_votes"] >= vote_round["required_votes"]:
            self._complete_vote()
    
    def _complete_vote(self) -> None:
        """Complete the active voting round."""
        vote_round = self.state["active_vote"]
        if not vote_round:
            return
        
        vote_round["end_time"] = datetime.now()
        vote_round["status"] = "completed"
        
        # Count votes
        vote_counts: Dict[str, int] = {}
        for vote in self.state["votes"]:
            if vote.get("metadata", {}).get("vote_round_id") == vote_round["id"]:
                vote_counts[vote["choice"]] = vote_counts.get(vote["choice"], 0) + 1
        
        # Determine result (simple majority)
        if vote_counts:
            result = max(vote_counts.items(), key=lambda x: x[1])[0]
            vote_round["result"] = result
        
        # Calculate participation rate
        total_agents = len([
            a for a in self.state["agents"].values()
            if a["role"] == "participant"
        ])
        participation = vote_round["received_votes"] / total_agents if total_agents > 0 else 0
        self.state["vote_participation_rate"][vote_round["id"]] = participation
        
        # Add to history and clear active
        self.state["vote_history"].append(vote_round)
        self.state["active_vote"] = None
        
        logger.info(
            f"Vote completed: {vote_round['vote_type']} -> {vote_round['result']} "
            f"({vote_round['received_votes']}/{vote_round['required_votes']} votes)"
        )
    
    def propose_topic(self, topic: str, proposed_by: str) -> None:
        """Propose a new topic for discussion.
        
        Args:
            topic: Topic to propose
            proposed_by: ID of proposing agent
        """
        if self.state["current_phase"] != 1:
            raise StateError("Topics can only be proposed during Agenda Setting phase")
        
        if topic in self.state["proposed_topics"]:
            logger.warning(f"Topic already proposed: {topic}")
            return
        
        self.state["proposed_topics"].append(topic)
        self.state["topics_info"][topic] = TopicInfo(
            topic=topic,
            proposed_by=proposed_by,
            start_time=None,
            end_time=None,
            message_count=0,
            status="proposed"
        )
        
        logger.debug(f"Topic proposed: {topic} (by {proposed_by})")
    
    def set_topic_queue(self, topics: List[str]) -> None:
        """Set the queue of topics to discuss.
        
        Args:
            topics: Ordered list of topics
        """
        # Validate all topics were proposed
        for topic in topics:
            if topic not in self.state["proposed_topics"]:
                raise ValidationError(f"Topic not proposed: {topic}")
        
        self.state["topic_queue"] = topics.copy()
        logger.info(f"Topic queue set: {topics}")
    
    def activate_topic(self, topic: str) -> None:
        """Activate a topic for discussion.
        
        Args:
            topic: Topic to activate
        """
        # Validate transition
        self.validator.validate_topic_transition(self.state, topic)
        
        # Deactivate current topic if any
        if self.state["active_topic"]:
            self.complete_topic()
        
        # Activate new topic
        self.state["active_topic"] = topic
        if topic in self.state["topic_queue"]:
            self.state["topic_queue"].remove(topic)
        
        # Update topic info
        if topic in self.state["topics_info"]:
            self.state["topics_info"][topic]["status"] = "active"
            self.state["topics_info"][topic]["start_time"] = datetime.now()
        
        logger.info(f"Activated topic: {topic}")
    
    def complete_topic(self) -> None:
        """Mark the current topic as completed."""
        topic = self.state["active_topic"]
        if not topic:
            return
        
        self.state["completed_topics"].append(topic)
        self.state["active_topic"] = None
        
        # Update topic info
        if topic in self.state["topics_info"]:
            self.state["topics_info"][topic]["status"] = "completed"
            self.state["topics_info"][topic]["end_time"] = datetime.now()
        
        logger.info(f"Completed topic: {topic}")
    
    def _set_next_speaker(self) -> None:
        """Set the next speaker in rotation."""
        if not self.state["speaking_order"]:
            return
        
        # Get next speaker from rotation
        idx = self.state["next_speaker_index"]
        self.state["current_speaker_id"] = self.state["speaking_order"][idx]
        
        # Update index
        self.state["next_speaker_index"] = (idx + 1) % len(self.state["speaking_order"])
    
    def add_phase_summary(self, phase: int, summary: str) -> None:
        """Add a summary for a phase.
        
        Args:
            phase: Phase number
            summary: Summary text
        """
        self.state["phase_summaries"][phase] = summary
        logger.debug(f"Added summary for phase {phase}")
    
    def add_topic_summary(self, topic: str, summary: str) -> None:
        """Add a summary for a topic.
        
        Args:
            topic: Topic name
            summary: Summary text
        """
        self.state["topic_summaries"][topic] = summary
        logger.debug(f"Added summary for topic: {topic}")
    
    def add_consensus_summary(self, topic: str, summary: str) -> None:
        """Add a consensus summary for a topic.
        
        Args:
            topic: Topic name
            summary: Consensus summary
        """
        self.state["consensus_summaries"][topic] = summary
        self.state["consensus_reached"][topic] = True
        logger.debug(f"Added consensus summary for topic: {topic}")
    
    def set_final_report(self, report: str) -> None:
        """Set the final report.
        
        Args:
            report: Final report text
        """
        self.state["final_report"] = report
        logger.info("Final report generated")
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current state.
        
        Returns:
            Deep copy of current state
        """
        return deepcopy(dict(self.state))
    
    def export_session(self) -> Dict[str, Any]:
        """Export complete session data for logs.
        
        Returns:
            Session data suitable for JSON export
        """
        snapshot = self.get_snapshot()
        
        # Convert datetime objects to strings
        def convert_dates(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_dates(item) for item in obj]
            return obj
        
        return convert_dates(snapshot)
    
    def validate_consistency(self) -> List[str]:
        """Run consistency validation on current state.
        
        Returns:
            List of warning messages
        """
        return self.validator.validate_state_consistency(self.state)