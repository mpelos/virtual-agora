"""State schema definitions for Virtual Agora.

This module defines the core state structure used throughout the Virtual Agora
application. It uses LangGraph's TypedDict for state definition with proper
type hints and reducers for append-only fields.
"""

from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from typing_extensions import NotRequired

# LangGraph's message reducer for append-only message lists
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentInfo(TypedDict):
    """Information about an agent in the discussion."""
    id: str
    model: str
    provider: str
    role: str  # 'moderator' or 'participant'
    message_count: int
    created_at: datetime


class Message(TypedDict):
    """A message in the discussion."""
    id: str
    speaker_id: str
    speaker_role: str  # 'moderator' or 'participant'
    content: str
    timestamp: datetime
    phase: int
    topic: Optional[str]
    metadata: NotRequired[Dict[str, Any]]  # For additional message metadata


class Vote(TypedDict):
    """A vote cast by an agent."""
    id: str
    voter_id: str
    phase: int
    vote_type: str  # 'topic_selection', 'continue_discussion', 'consensus'
    choice: str
    timestamp: datetime
    metadata: NotRequired[Dict[str, Any]]  # For vote context


class PhaseTransition(TypedDict):
    """Record of a phase transition."""
    from_phase: int
    to_phase: int
    timestamp: datetime
    reason: str  # Why the transition occurred
    triggered_by: str  # 'system', 'moderator', 'consensus'


class TopicInfo(TypedDict):
    """Information about a discussion topic."""
    topic: str
    proposed_by: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    message_count: int
    status: str  # 'proposed', 'active', 'completed', 'skipped'


class VoteRound(TypedDict):
    """Information about a voting round."""
    id: str
    phase: int
    vote_type: str
    options: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    required_votes: int
    received_votes: int
    result: Optional[str]
    status: str  # 'active', 'completed', 'cancelled'


class VirtualAgoraState(TypedDict):
    """Complete state for a Virtual Agora session.
    
    This is the main state structure that will be managed by LangGraph.
    Fields with Annotated[..., reducer] are append-only and will use
    the specified reducer function to merge updates.
    """
    
    # Session metadata
    session_id: str
    start_time: datetime
    config_hash: str  # Hash of the configuration for validation
    
    # Phase management (0-4)
    # 0: Initialization, 1: Agenda Setting, 2: Discussion, 3: Consensus, 4: Summary
    current_phase: int
    phase_history: Annotated[List[PhaseTransition], list.append]
    phase_start_time: datetime
    
    # Topic management
    active_topic: Optional[str]
    topic_queue: List[str]  # Topics to be discussed
    proposed_topics: List[str]  # All proposed topics (Phase 1)
    topics_info: Dict[str, TopicInfo]  # Detailed info about each topic
    completed_topics: Annotated[List[str], list.append]
    
    # Agent management
    agents: Dict[str, AgentInfo]  # agent_id -> info
    moderator_id: str
    current_speaker_id: Optional[str]
    speaking_order: List[str]  # Rotating queue of agent IDs
    next_speaker_index: int  # Index in speaking_order
    
    # Discussion history
    messages: Annotated[List[Message], add_messages]
    last_message_id: str  # For generating unique message IDs
    
    # Voting system
    active_vote: Optional[VoteRound]
    vote_history: Annotated[List[VoteRound], list.append]
    votes: Annotated[List[Vote], list.append]  # All individual votes
    
    # Consensus tracking
    consensus_proposals: Dict[str, List[str]]  # topic -> list of proposals
    consensus_reached: Dict[str, bool]  # topic -> consensus status
    
    # Generated content
    phase_summaries: Dict[int, str]
    topic_summaries: Dict[str, str]
    consensus_summaries: Dict[str, str]  # topic -> consensus summary
    final_report: Optional[str]
    
    # Runtime statistics
    total_messages: int
    messages_by_phase: Dict[int, int]
    messages_by_agent: Dict[str, int]
    messages_by_topic: Dict[str, int]
    vote_participation_rate: Dict[str, float]  # vote_id -> participation %
    
    # Error tracking (for recovery)
    last_error: Optional[str]
    error_count: int
    warnings: Annotated[List[str], list.append]


class MessagesState(TypedDict):
    """Simple state for message-based interactions in LangGraph.
    
    This is a lightweight state used for agent nodes that primarily
    deal with message exchanges.
    """
    messages: Annotated[List[BaseMessage], add_messages]