"""Data models for the Agenda Management System.

This module defines all the Pydantic models and data structures used
throughout the agenda management system, including proposals, votes,
agenda items, and analytics data.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from uuid import uuid4


class VoteType(str, Enum):
    """Types of votes that can be cast."""
    INITIAL_AGENDA = "initial_agenda"
    AGENDA_MODIFICATION = "agenda_modification"
    TOPIC_CONCLUSION = "topic_conclusion"


class ProposalStatus(str, Enum):
    """Status of a proposal."""
    PENDING = "pending"
    COLLECTED = "collected"
    TIMEOUT = "timeout"
    ERROR = "error"


class VoteStatus(str, Enum):
    """Status of a vote."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    INVALID = "invalid"
    TIMEOUT = "timeout"


class AgendaStatus(str, Enum):
    """Status of agenda operations."""
    PENDING = "pending"
    COLLECTING_PROPOSALS = "collecting_proposals"
    COLLECTING_VOTES = "collecting_votes"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"


class Proposal(BaseModel):
    """A topic proposal from an agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    topic: str
    description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    status: ProposalStatus = ProposalStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('topic')
    def topic_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()

    class Config:
        use_enum_values = True


class Vote(BaseModel):
    """A vote cast by an agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    vote_type: VoteType
    vote_content: str  # Natural language vote
    parsed_preferences: Optional[List[str]] = None  # Extracted topic preferences
    timestamp: datetime = Field(default_factory=datetime.now)
    status: VoteStatus = VoteStatus.PENDING
    confidence_score: Optional[float] = None  # For analytics
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('vote_content')
    def vote_content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Vote content cannot be empty')
        return v.strip()

    class Config:
        use_enum_values = True


class AgendaItem(BaseModel):
    """An item in the agenda."""
    topic: str
    description: Optional[str] = None
    rank: int
    proposed_by: List[str] = Field(default_factory=list)  # Agent IDs who proposed this
    vote_score: Optional[float] = None  # Calculated voting score
    discussion_started: Optional[datetime] = None
    discussion_ended: Optional[datetime] = None
    status: str = "pending"  # pending, active, completed, skipped
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('topic')
    def topic_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()

    @validator('rank')
    def rank_positive(cls, v):
        if v < 1:
            raise ValueError('Rank must be positive')
        return v


class ProposalCollection(BaseModel):
    """Collection of proposals from agents."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    round_number: int = 1
    proposals: List[Proposal] = Field(default_factory=list)
    requested_agents: List[str] = Field(default_factory=list)
    responding_agents: List[str] = Field(default_factory=list)
    timeout_agents: List[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    timeout_seconds: int = 300  # 5 minutes default
    status: ProposalStatus = ProposalStatus.PENDING
    error_details: Optional[str] = None
    
    @property
    def completion_rate(self) -> float:
        """Calculate the proposal completion rate."""
        if not self.requested_agents:
            return 0.0
        return len(self.responding_agents) / len(self.requested_agents)


class VoteCollection(BaseModel):
    """Collection of votes from agents."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    vote_type: VoteType
    topic_options: List[str] = Field(default_factory=list)
    votes: List[Vote] = Field(default_factory=list)
    requested_agents: List[str] = Field(default_factory=list)
    responding_agents: List[str] = Field(default_factory=list)
    timeout_agents: List[str] = Field(default_factory=list)
    invalid_votes: List[str] = Field(default_factory=list)  # Vote IDs
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    timeout_seconds: int = 180  # 3 minutes default
    status: VoteStatus = VoteStatus.PENDING
    error_details: Optional[str] = None

    @property
    def participation_rate(self) -> float:
        """Calculate the voting participation rate."""
        if not self.requested_agents:
            return 0.0
        return len(self.responding_agents) / len(self.requested_agents)

    class Config:
        use_enum_values = True


class AgendaState(BaseModel):
    """Current state of the agenda system."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    version: int = 1
    status: AgendaStatus = AgendaStatus.PENDING
    current_agenda: List[AgendaItem] = Field(default_factory=list)
    proposal_collections: List[ProposalCollection] = Field(default_factory=list)
    vote_collections: List[VoteCollection] = Field(default_factory=list)
    current_topic_index: int = 0
    completed_topics: List[str] = Field(default_factory=list)
    active_topic: Optional[str] = None
    last_modification: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def remaining_topics(self) -> List[str]:
        """Get list of remaining topics."""
        return [item.topic for item in self.current_agenda 
                if item.topic not in self.completed_topics]
    
    @property
    def current_topic(self) -> Optional[AgendaItem]:
        """Get the current topic being discussed."""
        if self.active_topic:
            for item in self.current_agenda:
                if item.topic == self.active_topic:
                    return item
        return None

    class Config:
        use_enum_values = True


class AgendaModification(BaseModel):
    """A proposed modification to the agenda."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    modification_type: str  # "add", "remove", "reorder"
    target_topic: Optional[str] = None  # For remove/reorder
    new_topic: Optional[str] = None  # For add
    new_description: Optional[str] = None
    justification: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    applied: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('modification_type')
    def valid_modification_type(cls, v):
        if v not in ['add', 'remove', 'reorder']:
            raise ValueError('Modification type must be add, remove, or reorder')
        return v


class TopicTransition(BaseModel):
    """Record of a topic transition."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    from_topic: Optional[str] = None
    to_topic: Optional[str] = None
    transition_type: str  # "start", "complete", "skip", "error"
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_states_reset: List[str] = Field(default_factory=list)  # Agent IDs
    summary_saved: bool = False
    duration_seconds: Optional[float] = None
    message_count: int = 0
    participant_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('transition_type')
    def valid_transition_type(cls, v):
        if v not in ['start', 'complete', 'skip', 'error']:
            raise ValueError('Transition type must be start, complete, skip, or error')
        return v


class AgendaAnalytics(BaseModel):
    """Analytics data for agenda performance."""
    session_id: str
    total_proposals: int = 0
    unique_topics_proposed: int = 0
    proposal_acceptance_rate: float = 0.0
    average_vote_participation: float = 0.0
    agenda_modifications_count: int = 0
    topics_completed: int = 0
    topics_skipped: int = 0
    average_topic_duration_minutes: float = 0.0
    agent_participation_rates: Dict[str, float] = Field(default_factory=dict)
    topic_proposal_distribution: Dict[str, int] = Field(default_factory=dict)  # agent_id -> count
    voting_patterns: Dict[str, Any] = Field(default_factory=dict)
    modification_patterns: Dict[str, int] = Field(default_factory=dict)
    timeline_events: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class AgendaSynthesisResult(BaseModel):
    """Result of agenda synthesis operation."""
    proposed_agenda: List[str]  # Ordered list of topics
    synthesis_explanation: Optional[str] = None
    tie_breaks_applied: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    synthesis_attempts: int = 1
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('proposed_agenda')
    def agenda_not_empty(cls, v):
        if not v:
            raise ValueError('Proposed agenda cannot be empty')
        return v


class EdgeCaseEvent(BaseModel):
    """Record of an edge case encountered."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    event_type: str  # "empty_proposals", "all_agents_abstain", "invalid_votes", etc.
    description: str
    resolution_strategy: str
    timestamp: datetime = Field(default_factory=datetime.now)
    affected_agents: List[str] = Field(default_factory=list)
    system_response: str
    recovered_successfully: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)